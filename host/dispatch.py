#!/usr/bin/env python3
"""Sweep dispatcher. One experiment = one config = one GPU, never co-tenanted.

Rewritten 2026-08-18. Changes from the v3 dispatcher, each tied to an observed failure:

  1. CPU PINNING AND THREAD CAPS.  Byte-identical control runs re-run in different waves
     produced a val_bpb spread far larger than most effects worth chasing, purely because
     step count moved between waves. The data pipeline is single-threaded Python inside
     the frozen prepare.py, this box is shared and runs near CPU saturation, and torch
     defaults to one intra-op thread per core PER PROCESS -- so our own concurrent
     trainers oversubscribed the machine against each other. Each trainer now gets a
     disjoint core block via taskset plus a hard thread cap. This attacks the noise at its
     source instead of regressing it out afterwards.

  2. DECISION CUTOFF INSTEAD OF A LAUNCH BLOCK.  v3's gate stopped ALL launches when a
     critique went stale, idling GPUs 16 and 48 minutes in one morning on a shared box.
     Here a stale artifact only freezes NEW decisions: an entry may launch if it was
     created at or before the gate's `decision_cutoff`. Compute never waits for prose.

  3. CLAIMS ARE RELEASED ON FAILURE.  v3 claimed an entry with os.mkdir before launching
     and never removed the claim if the launch raised, permanently stranding the entry.

  4. NO SILENT cfg BACKFILL GAPS.  Recovered orphans take their cfg from the queue; a
     record whose cfg cannot be resolved is written ok:false rather than counted as a
     valid run against no axis (v3 produced three such records).
"""
import collections
import fcntl
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path.home() / "ophis_v3" / "sweep"
sys.path.insert(0, str(ROOT))
from direction import axis_state, blocked_reason, label   # noqa: E402

VENV = pathlib.Path.home() / "ophis_v3" / "gpu6" / ".venv" / "bin" / "python"
PREP = pathlib.Path.home() / "ophis_v3" / "gpu6" / "prepare.py"
DEADLINE = float(sys.argv[1]) if len(sys.argv) > 1 else time.time() + 24 * 3600

MINFREE_MB = 1024
MAX_GPUS = 4              # operator allocation on a shared box; never exceed
# Entries sharing a `wave_group` launch TOGETHER on different GPUs or not at all. This
# is how a yoked pair is actually obtained: the previous controls were launched one at a
# time as GPUs freed up, so they ran back-to-back over half an hour, host load drifted
# between them, and their spread measured drift rather than resolution. Concurrent
# launches share the contention, which is the whole point of pairing.
WAIT_LOG_EVERY_S = 30 * 60   # how often to report that we are waiting for capacity
CORES_PER_JOB = 12        # disjoint taskset block per trainer
CORE_BASE = 96            # start high: low cores are where foreign tenants cluster
GATE_MAX_AGE = 20 * 60

_lockf = open(ROOT / "dispatcher.lock", "w")
try:
    fcntl.flock(_lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print("another dispatcher holds the lock; exiting", file=sys.stderr)
    sys.exit(0)
_lockf.write(str(os.getpid()))
_lockf.flush()

for d in ("results", "claims", "work"):
    (ROOT / d).mkdir(parents=True, exist_ok=True)


def log(m):
    with open(ROOT / "dispatch.log", "a") as f:
        f.write(f"{time.strftime('%FT%TZ', time.gmtime())} {m}\n")


def load_queue():
    try:
        return json.loads((ROOT / "queue.json").read_text())
    except (OSError, ValueError):
        return []


def results():
    out = []
    for f in (ROOT / "results").glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return out


def gpu_state():
    o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,uuid",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=False).stdout
    st = {}
    for line in o.strip().splitlines():
        parts = [c.strip() for c in line.split(",")]
        if len(parts) == 3:
            st[int(parts[0])] = (float(parts[1]), parts[2])
    return st


def gate():
    """(may_launch, decision_cutoff, reason). Missing or stale verdict fails SAFE."""
    f = ROOT / "GATE_STATUS.json"
    try:
        age = time.time() - f.stat().st_mtime
        if age > GATE_MAX_AGE:
            return False, 0.0, f"gate verdict stale ({age/60:.0f} min)"
        g = json.loads(f.read_text())
    except (OSError, ValueError) as exc:
        return False, 0.0, f"gate verdict unreadable ({exc})"
    if not g.get("gate_open"):
        return False, 0.0, "blocking: " + ", ".join(g.get("blocking_failures") or ["?"])
    return True, float(g.get("decision_cutoff") or time.time()), ""


def _is_control(cfg):
    from direction import is_platform
    return is_platform(cfg or {})


def runnable(cutoff):
    """Every queue entry eligible to launch right now, in queue order."""
    state = axis_state(results())
    out, frozen, policy = [], 0, 0
    for item in load_queue():
        if (ROOT / "results" / f"{item['name']}.json").exists():
            continue
        if (ROOT / "claims" / item["name"]).exists():
            continue
        # A decision made after the cutoff waits for the overdue council artifact.
        # Controls are exempt: a control is the measuring instrument, not a research
        # decision, and freezing it would starve the noise band that every verdict needs.
        if (float(item.get("created_at") or 0) > cutoff
                and not _is_control(item["cfg"])):
            frozen += 1
            continue
        if blocked_reason(item["cfg"], state):
            policy += 1
            continue
        out.append(item)
    return out, frozen, policy


def next_batch(cutoff, n_free):
    """Entries to launch on this pass: the largest wave_group that fits in n_free GPUs.

    A group launches together or waits. Returning [] with groups pending means we are
    holding capacity for a wave rather than dribbling it out one GPU at a time -- the
    difference between a yoked pair and two runs half an hour apart.
    """
    items, frozen, policy = runnable(cutoff)
    if not items:
        if (frozen or policy) and int(time.time()) % 900 < 6:
            log(f"NO_RUNNABLE_WORK: {frozen} frozen by decision cutoff, "
                f"{policy} blocked by explore/exploit policy")
        return [], None

    groups = {}
    for it in items:
        groups.setdefault(it.get("wave_group") or f"__solo__{it['name']}", []).append(it)

    # Largest group that fits, so the box is filled and pairs stay intact.
    fits = sorted([g for g in groups.values() if len(g) <= n_free],
                  key=lambda g: -len(g))
    if fits:
        return fits[0], None
    smallest = min(len(g) for g in groups.values())
    return [], f"waiting for {smallest} free GPUs to launch a wave ({n_free} free now)"


def claim_all(batch):
    """Claim every member or none. A partial claim would split a wave."""
    taken = []
    for it in batch:
        try:
            os.mkdir(ROOT / "claims" / it["name"])
            taken.append(it)
        except FileExistsError:
            for t in taken:
                release(t)
            return []
    return taken


def release(item):
    try:
        os.rmdir(ROOT / "claims" / item["name"])
    except OSError:
        pass


def parse(txt):
    m = {}
    for line in txt.replace("\r", "\n").splitlines():
        mt = re.match(r"^([a-z_0-9]+):\s+([-\d.]+)\s*$", line)
        if mt:
            try:
                m[mt.group(1)] = float(mt.group(2))
            except ValueError:
                pass
    return m


def recover_orphans():
    for d in (ROOT / "work").iterdir():
        if not d.is_dir():
            continue
        r = ROOT / "results" / f"{d.name}.json"
        if r.exists() or not (d / "out.log").exists():
            continue
        txt = (d / "out.log").read_text(errors="replace")
        if "val_bpb:" not in txt:
            continue
        met = parse(txt)
        cfg = next((e["cfg"] for e in load_queue() if e["name"] == d.name), None)
        r.write_text(json.dumps(
            {"name": d.name, "cfg": cfg or {}, "gpu": -1, "started": 0,
             "ended": (d / "out.log").stat().st_mtime, "returncode": 0, "metrics": met,
             # cfg unresolvable => the run cannot be attributed to any axis, so it is not
             # evidence. v3 wrote ok:true here and silently under-counted three runs.
             "ok": bool(cfg) and "val_bpb" in met,
             "invalid_reason": "" if cfg else "cfg unresolvable from queue",
             "error": "", "recovered": True}, indent=1))
        log(f"RECOVERED {d.name} val_bpb={met.get('val_bpb')} cfg={'ok' if cfg else 'LOST'}")


def release_stranded_claims():
    """A claim with no result AND no work directory can never make progress.

    next_unclaimed() claims by os.mkdir before Popen; a dispatcher killed in that window
    leaves a claim behind, and thereafter the entry is skipped forever -- it has no
    result (so it is not done) and no out.log (so recover_orphans ignores it). Sweep
    them once at startup, when we hold the singleton lock and nothing of ours is running.
    """
    freed = 0
    cdir = ROOT / "claims"
    if not cdir.is_dir():
        return
    for c in cdir.iterdir():
        if not c.is_dir():
            continue
        if (ROOT / "results" / f"{c.name}.json").exists():
            continue
        if (ROOT / "work" / c.name).is_dir():
            continue
        try:
            os.rmdir(c)
            freed += 1
        except OSError:
            pass
    if freed:
        log(f"released {freed} stranded claim(s) with no result and no work dir")


def cores_for(slot):
    lo = CORE_BASE + slot * CORES_PER_JOB
    return f"{lo}-{lo + CORES_PER_JOB - 1}"


def main():
    release_stranded_claims()   # safe here: we hold the singleton lock, nothing running
    running = {}
    launched = 0
    last_wait_log = 0.0
    while time.time() < DEADLINE:
        recover_orphans()

        # co-tenancy: a foreign process arriving mid-run invalidates the measurement
        try:
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                   "--format=csv,noheader"],
                                  capture_output=True, text=True, check=False).stdout
            me = subprocess.run(["whoami"], capture_output=True, text=True,
                                check=False).stdout.strip()
            byuuid = collections.defaultdict(list)
            for ln in apps.splitlines():
                if "," not in ln:
                    continue
                pid, uu = [c.strip() for c in ln.split(",")]
                own = subprocess.run(["ps", "-o", "user=", "-p", pid],
                                     capture_output=True, text=True, check=False).stdout.strip()
                byuuid[uu].append(own)
            for g, job in running.items():
                if any(o and o != me for o in byuuid.get(job["uuid"], [])):
                    if not job["cotenant"]:
                        log(f"CO-TENANT on gpu{g} during {job['item']['name']} -> INVALID")
                    job["cotenant"] = True
        except OSError:
            pass

        for g, job in list(running.items()):
            if job["proc"].poll() is None:
                continue
            txt = (job["dir"] / "out.log").read_text(errors="replace")
            met = parse(txt)
            rec = {"name": job["item"]["name"], "cfg": job["item"]["cfg"], "gpu": g,
                   "started": job["started"], "ended": time.time(),
                   "returncode": job["proc"].returncode, "metrics": met,
                   "cotenant_detected": job["cotenant"], "cores": job["cores"],
                   "ok": job["proc"].returncode == 0 and "val_bpb" in met and not job["cotenant"],
                   "invalid_reason": "gpu co-tenancy during the run" if job["cotenant"] else "",
                   "error": "" if job["proc"].returncode == 0 else txt.strip()[-400:]}
            (ROOT / "results" / f"{job['item']['name']}.json").write_text(json.dumps(rec, indent=1))
            log(f"DONE {job['item']['name']} gpu{g} rc={job['proc'].returncode} "
                f"val_bpb={met.get('val_bpb')} steps={met.get('num_steps')}")
            del running[g]

        may, cutoff, why = gate()
        if not may:
            if int(time.time()) % 600 < 6:
                log(f"GATE CLOSED: {why}")
            time.sleep(5)
            continue

        free_slots = [s for s in range(MAX_GPUS)
                      if s not in {j["slot"] for j in running.values()}]
        free_gpus = [(g, uuid) for g, (used, uuid) in sorted(gpu_state().items())
                     if g not in running and used <= MINFREE_MB]
        capacity = min(len(free_gpus), len(free_slots), MAX_GPUS - len(running))

        batch, waiting = ([], None) if capacity <= 0 else next_batch(cutoff, capacity)
        if waiting or (capacity <= 0 and time.time() < DEADLINE - 500):
            msg = waiting or f"0 of {MAX_GPUS} GPUs free"
            if time.time() - last_wait_log > WAIT_LOG_EVERY_S:
                log(f"WAITING: {msg}; {len(running)}/{MAX_GPUS} of ours running")
                last_wait_log = time.time()
        batch = claim_all(batch)

        for item in batch:
            if time.time() >= DEADLINE - 500 or not free_gpus or not free_slots:
                release(item)
                continue
            g, uuid = free_gpus.pop(0)
            slot = free_slots.pop(0)
            try:
                d = ROOT / "work" / item["name"]
                if d.exists():
                    shutil.rmtree(d)
                for sub in ("inductor", "triton", "tmp"):
                    (d / ".cache" / sub).mkdir(parents=True)
                shutil.copy(PREP, d / "prepare.py")
                shutil.copy(ROOT / "variants" / item["variant"], d / "train.py")
                cores = cores_for(slot)
                env = dict(os.environ)
                env.update({
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "TORCHINDUCTOR_CACHE_DIR": str(d / ".cache" / "inductor"),
                    "TRITON_CACHE_DIR": str(d / ".cache" / "triton"),
                    "TMPDIR": str(d / ".cache" / "tmp"),
                    "AUTORESEARCH_SEED": str(item.get("seed", 42)),
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": uuid,
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                    # thread caps: torch otherwise opens ~nproc intra-op threads PER
                    # process, and four of our trainers then fight each other on a box
                    # already at loadavg ~160/192.
                    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                    "OPENBLAS_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8",
                    "TOKENIZERS_PARALLELISM": "false",
                })
                f = open(d / "out.log", "w")
                p = subprocess.Popen(["taskset", "-c", cores, str(VENV), "train.py"],
                                     cwd=d, stdout=f, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
            except Exception as exc:          # noqa: BLE001
                release(item)                 # an earlier version stranded it forever
                free_slots.insert(0, slot)
                free_gpus.insert(0, (g, uuid))
                log(f"LAUNCH FAILED {item['name']}: {exc} (claim released)")
                continue
            running[g] = {"proc": p, "item": item, "dir": d, "started": time.time(),
                          "uuid": uuid, "cotenant": False, "slot": slot, "cores": cores}
            launched += 1
            log(f"LAUNCH {item['name']} gpu{g} cores={cores} [{label(item['cfg'])}]"
                f"{' wave=' + item['wave_group'] if item.get('wave_group') else ''} "
                f"[{launched} launched]")
        time.sleep(5)

    log(f"DEADLINE reached; waiting on {len(running)} in-flight")
    for g, job in running.items():
        job["proc"].wait()
    log("SWEEP COMPLETE")


if __name__ == "__main__":
    main()
