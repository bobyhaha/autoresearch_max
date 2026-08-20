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
# L002_burned_gpu_reused: the free-GPU test is an INSTANTANEOUS memory poll, so a device
# whose foreign tenant is between jobs passes it and reclaims memory seconds later. That
# is not hypothetical: C01_control was voided by a co-tenant on gpu5, gpu5 was selected
# again ten minutes later, and C02_control was co-tenanted 11 seconds after launch. Two
# 300-second budgets bought nothing. A device that has just destroyed one of our runs is
# excluded for this long; the cost of being wrong is a GPU we skip while others are free.
COTENANT_QUARANTINE_S = 45 * 60
# ...but lift it as soon as the device has been demonstrably clean this long.
QUARANTINE_CLEAR_S = 8 * 60
# How many times each device has burned a run of ours this process. The early-release
# window doubles per burn, so a cycling tenant cannot keep winning the same device.
burn_count: dict = {}
# Repeated identical launch failures. A missing variant file will not appear on
# its own, so a wave that fails the same way repeatedly is reported once and set
# aside rather than retried every poll.
launch_fails: dict = {}
blocked_launch: set = set()
LAUNCH_FAIL_MAX = 3
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
    queue = load_queue()
    # WHICH WAVES ARE PURE CONTROL BLOCKS. The exemption below is for the instrument --
    # a block of controls run to measure the noise band -- and a control block is a whole
    # wave of controls. A control that is one half of a yoked PAIR is not an instrument on
    # its own; it is half of a research decision, and its other half is the treatment.
    #
    # Exempting it per-entry deadlocked the campaign completely. Every pending wave here
    # is a width-2 pair, so the cutoff released the control and froze the treatment, and
    # the wave-held guard below then correctly refused to launch a control whose treatment
    # was missing. Result: `runnable` reported 16 entries, and next_batch could assemble
    # ZERO waves -- while gate.py printed "already-queued work keeps launching; GPUs do
    # not idle for prose". A per-entry exemption inside an all-or-nothing wave launcher
    # cannot do anything except strand the wave.
    _pure_ctl = {}
    for it in queue:
        g = it.get("wave_group")
        if g:
            _pure_ctl[g] = _pure_ctl.get(g, True) and _is_control(it["cfg"])
    for item in queue:
        if (ROOT / "results" / f"{item['name']}.json").exists():
            continue
        if (ROOT / "claims" / item["name"]).exists():
            continue
        # A decision made after the cutoff waits for the overdue council artifact.
        # Exempt only a wave that is ENTIRELY controls: that is the measuring instrument,
        # not a research decision, and freezing it would starve the noise band every
        # verdict needs. A mixed wave freezes as a unit, because it IS a unit.
        _grp = item.get("wave_group")
        _exempt = _pure_ctl.get(_grp, _is_control(item["cfg"])) if _grp else _is_control(item["cfg"])
        if float(item.get("created_at") or 0) > cutoff and not _exempt:
            frozen += 1
            continue
        if blocked_reason(item["cfg"], state):
            policy += 1
            continue
        out.append(item)
    return out, frozen, policy


def wave_sizes(cutoff=None):
    """How many entries each wave_group has IN THE QUEUE, regardless of their state.

    next_batch() must size a wave from this, not from the entries that happen to be
    unclaimed. Grouping the runnable remainder means a 4-wide wave that has already had
    two members claimed presents itself as a 2-wide wave and launches -- which is how
    W03a, queued as a single 4-wide group, ran as two pairs. For a control block that is
    merely untidy; for a treatment it silently destroys the pairing the wave existed to
    create, which is the one thing this dispatcher is supposed to guarantee.
    """
    try:
        queue = load_queue()
    except NameError:           # pure-logic import in tests/test_wave_launch.py
        return {}
    # Count only members that COULD run. An entry frozen by the gate's decision cutoff is
    # not permanently missing -- it comes back the moment a council artifact is refreshed --
    # so counting it made every wave with a treatment look "split" the instant the round
    # went stale, and the split guard then refused the wave's CONTROLS too. Two correct
    # rules deadlocked: the freeze exists so prose cannot authorise new science, and the
    # split guard exists so a pairing cannot silently dissolve, but together they idled
    # four GPUs. The campaign's own rule is that already-queued work keeps launching and
    # GPUs do not idle for prose, so the cutoff-frozen members are excluded from the
    # denominator rather than counted as casualties.
    # The exemption must match runnable()'s EXACTLY or the two disagree about what is
    # launchable: this counts a frozen pair's control as a 1-wide wave while runnable()
    # yields nothing from that group, and next_batch is left sizing a wave that has no
    # members. Only a wave that is entirely controls is the instrument; see runnable().
    # `_is_control` imports direction, which the pure-logic test harness does not provide
    # until it stubs it; resolve it dynamically and fall back to "nothing is exempt",
    # which is the conservative direction -- it holds a wave rather than launching half.
    _ic = globals().get("_is_control") or (lambda cfg: False)
    _pure_ctl = {}
    if cutoff is not None:
        for it in queue:
            g = it.get("wave_group")
            if g:
                _pure_ctl[g] = _pure_ctl.get(g, True) and _ic(it.get("cfg") or {})
    sizes = collections.Counter()
    for item in queue:
        g = item.get("wave_group")
        if not g:
            continue
        if (cutoff is not None and float(item.get("created_at") or 0) > cutoff
                and not _pure_ctl.get(g, _ic(item.get("cfg") or {}))):
            continue
        sizes[g] += 1
    return sizes

_WEDGED: set = set()


def _wedged_reason(key) -> str:
    """Why a wave can never assemble, if the cause is a policy block rather than timing."""
    from direction import blocked_reason, axis_state
    import claims as _c
    try:
        state = axis_state(results())
    except Exception:                                          # noqa: BLE001
        return ""
    for it in load_queue():
        if it.get("wave_group") != key:
            continue
        if (ROOT / "results" / f"{it['name']}.json").exists():
            return ""            # something finished; the tombstone path owns this
        if (ROOT / "claims" / it["name"]).exists():
            return ""            # still in flight
        why = blocked_reason(it.get("cfg") or {}, state)
        if why:
            return f"{it['name']}: {why}"
        hits = _c.blocked_values(it.get("cfg") or {})
        if hits:
            k, v, les, _r = hits[0]
            return f"{it['name']}: {k}={v} blocked by {les['id']}"
    return ""


def tombstone_split_wave(key) -> bool:
    """Retire a wave that can never be completed, instead of refusing it forever.

    The split guard is right to refuse a remnant, but refusal alone leaves the stranded
    members in the queue being re-evaluated and re-logged every poll, indefinitely. That
    happened to zloss01_A and zloss01_B: their two controls ran, their two treatments did
    not, and because a wave is sized from the queue the pair could never be reformed. The
    log filled with identical refusals while the entries sat there looking pending, which
    reads as work still to come rather than work that is already lost.

    A wave is PERMANENTLY split when some member has a finished result on disk while
    another has neither result nor claim: the finished member's wave-mates can no longer
    share its host conditions, so any later run of them would be a cross-wave comparison
    wearing a wave_group label -- exactly the confound the grouping exists to prevent.
    Note the deliberate asymmetry: a CLAIMED member is merely in flight and the wave is
    still forming, so this must never fire on one.

    Stranded members are written as invalid results rather than deleted. Deleting them
    would make the loss disappear from the accounting; an invalid record makes
    tools/analyze.py list them and tools/gate.py demand a lesson, which is the honest
    treatment of GPU time that bought nothing. Returns True when it retired something.
    """
    members = [it for it in load_queue() if it.get("wave_group") == key]
    if not members:
        return False
    done, stranded, inflight = [], [], []
    for it in members:
        if (ROOT / "results" / f"{it['name']}.json").exists():
            done.append(it["name"])
        elif (ROOT / "claims" / it["name"]).exists():
            inflight.append(it["name"])
        else:
            stranded.append(it)
    # One member still in flight means the wave is STILL FORMING, so none of its mates is
    # stranded yet -- the claim may be about to produce the result that completes it.
    # Checking only "this member has no claim" was not enough: a wave with one claimed
    # and one unclaimed member would tombstone the unclaimed one out from under a run
    # that was still going. The wave is dead only when nothing is left moving.
    if not done or not stranded or inflight:
        return False
    for it in stranded:
        rec = {
            "name": it["name"], "cfg": it.get("cfg"), "gpu": None,
            "hypothesis_id": it.get("hypothesis_id"),
            "started": None, "ended": time.time(), "returncode": None,
            "cotenant_detected": False, "cores": None, "ok": False,
            "invalid_reason": (
                f"stranded by a permanently split wave: {len(done)} of {len(members)} "
                f"members of wave_group {key} already finished ({', '.join(sorted(done))}) "
                f"while this one was never claimed. It cannot now be run as part of that "
                f"wave, and running it later would be a cross-wave comparison labelled as "
                f"a yoked one. Retired unrun; no val_bpb exists for it and none may be "
                f"inferred."),
            "error": "",
        }
        (ROOT / "results" / f"{it['name']}.json").write_text(json.dumps(rec, indent=1))
        log(f"WAVE RETIRED {key}: {it['name']} stranded unrun and tombstoned "
            f"(its wave-mates already finished; the pairing cannot be reformed)")
    return True


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

    # A wave is launchable only if EVERY member it was queued with is still available.
    # A group whose runnable members are fewer than its queued size has already been
    # split -- launching the remnant would hand back a pair that never shared a wave.
    sizes = wave_sizes(cutoff)
    sizes_all = wave_sizes(None)          # including members the cutoff froze
    intact, broken, held = {}, [], []
    for key, members in groups.items():
        want = sizes.get(key, len(members))
        if len(members) != want:
            broken.append((key, len(members), want))
            continue
        # A wave that is merely FROZEN is not split -- but it is also not ready. Launching
        # its controls now would put them in a different wave from the treatment they are
        # yoked to, which is precisely the cross-wave drift the wave_group mechanism exists
        # to prevent. Hold the whole wave until the freeze lifts.
        if sizes_all.get(key, want) > want:
            held.append((key, want, sizes_all[key]))
            continue
        intact[key] = members
    for key, have, total in held:
        if int(time.time()) % 1800 < 6:
            log(f"WAVE HELD {key}: {total - have} member(s) frozen by the decision cutoff; "
                f"holding the whole wave so its control does not run in a different wave "
                f"from its treatment")
    for key, have, want in broken:
        if tombstone_split_wave(key):
            continue
        # A wave whose missing members are POLICY-BLOCKED rather than claimed or finished
        # can never assemble, and the tombstone cannot retire it either: tombstoning needs
        # a finished member, and none of these ever ran. It therefore logs an identical
        # refusal on every poll, forever, while the entries sit in the queue looking
        # pending. Report it once as WEDGED, with the reason, so it reads as a decision
        # requiring action rather than as work still to come.
        _blocked = _wedged_reason(key)
        if _blocked:
            if key not in _WEDGED:
                _WEDGED.add(key)
                log(f"WAVE WEDGED {key}: {want - have} member(s) refused by policy, not by "
                    f"the cutoff, so this wave can never assemble and cannot be "
                    f"tombstoned (no member has finished). Reason: {_blocked}. Requeue it "
                    f"or lift the block; it will not resolve on its own.")
            continue
        log(f"WAVE SPLIT {key}: {have} of {want} members still runnable; refusing to "
            f"launch the remnant (a partial wave is not a yoked comparison)")
    if not intact:
        return [], (f"every pending wave is split: "
                    f"{', '.join(k for k, _, _ in broken)}") if broken else None

    # Largest group that fits, so the box is filled and pairs stay intact.
    fits = sorted([g for g in intact.values() if len(g) <= n_free],
                  key=lambda g: -len(g))
    if fits:
        return fits[0], None
    smallest = min(len(g) for g in intact.values())
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
        # [A-Za-z] not [a-z]: the pattern silently dropped every metric with a capital in
        # its name. num_params_M, flops_per_token_M and total_tokens_M are printed by
        # EVERY run and appear in ZERO result records because of it -- which is also why
        # the campaign spent a day believing the model had 124M parameters when the run
        # itself had been reporting 50.33M all along (L015_model_is_50M_not_124M).
        # `\s*` not `\s+`: the telemetry block aligns most values with padding but
        # `flops_per_token_M:{v}` is printed with NO space, so a parser demanding one
        # silently dropped it from every run in the campaign -- the same class of failure
        # as the earlier `^[a-z_0-9]+:` pattern that discarded every capitalised metric.
        # A metric that is printed but never parsed is worse than one never printed: the
        # variant looks instrumented and the hypothesis citing it can never activate.
        mt = re.match(r"^([A-Za-z_0-9]+):\s*([-\d.]+)\s*$", line)
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
        # The GPU is RECORDED, not unknown: launch.json sits in this very directory and
        # holds gpu, uuid and started. Writing gpu:-1 here was not a graceful degradation,
        # it was discarding data that was already on disk -- and -1 does not read as
        # "unknown" downstream, it reads as a DEVICE. device_means() pooled it as one, so a
        # recovered run's effect was corrected against a fictional device built from
        # whatever else happened to land there. The two runs recovered from today's
        # dispatcher crash came out right only because the fake slot held exactly one
        # treatment and its own wave-mate control; a recovery landing anywhere else would
        # have imported an unrelated GPU's ~0.0025 bpb bias straight into the effect.
        _lj = {}
        try:
            _lj = json.loads((d / "launch.json").read_text())
        except (OSError, ValueError):
            pass
        r.write_text(json.dumps(
            {"name": d.name, "cfg": cfg or {}, "gpu": _lj.get("gpu", -1),
             "started": _lj.get("started", 0),
             "hypothesis_id": next((e.get("hypothesis_id") for e in load_queue()
                                    if e["name"] == d.name), None),
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

def _crashed(job) -> str:
    """A crash marker in the run's own log, or "" if it finished cleanly.

    An adopted process reports exit code 0 whatever happened to it, so the exit status
    cannot distinguish a crash from a clean finish. A missing val_bpb catches most
    failures, but not one that dies AFTER printing it -- in the telemetry epilogue -- which
    would otherwise be written as valid evidence with a real val_bpb attached.
    """
    # `job` holds proc/item/dir/started/uuid/cotenant/slot/cores -- the run's name lives at
    # job["item"]["name"]. `job["name"]` was a KeyError, and because it sits in the health
    # check that runs on EVERY poll it killed the dispatcher outright at 15:40Z, minutes
    # after launching R6MTP_P1, so the treatment finished on the GPU and was never
    # harvested. This is the SECOND site of the same mistake: the first, in the co-tenancy
    # taint write, was found by an audit and fixed hours earlier, and fixing one instance of
    # a typo class without grepping for the rest left this one to fire.
    try:
        tail = (ROOT / "work" / job["item"]["name"] / "out.log").read_text(
            errors="replace")[-4000:]
    except OSError:
        return ""
    for mark in ("Traceback (most recent call last)", "CUDA out of memory",
                 "torch.cuda.OutOfMemoryError", "Killed", "Segmentation fault"):
        if mark in tail:
            return f"crash marker in out.log after launch: {mark!r}"
    return ""


class AdoptedProc:
    """A live trainer this dispatcher did not start, addressed by pid.

    Adoption is what makes restarting the dispatcher safe. Without it a restart forgets
    every in-flight run, and because a trainer holds no GPU memory during its long CPU
    tokenization phase, the fresh dispatcher re-launches straight onto the same GPUs.
    """

    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            # Exited -- but we did NOT start this process and cannot read its status, so
            # its exit code is unknown, not zero. Reporting 0 made a crash indistinguishable
            # from a clean finish. Usually harmless, because a crashed run prints no
            # val_bpb and fails the `ok` test on that; the gap is a crash AFTER val_bpb is
            # printed, in the telemetry epilogue, which would be recorded as valid
            # evidence. The log is scanned for crash markers at completion instead.
            self.returncode = 0
            return 0
        except PermissionError:
            pass                    # alive, owned by someone else -- treat as running
        return None

    def wait(self):
        while self.poll() is None:
            time.sleep(5)
        return self.returncode


_TAINTED_AT_ADOPT: set = set()


def adopt_running():
    """Re-attach to trainers still alive from a previous dispatcher, keyed by GPU.

    A claim with a work directory and no result is an in-flight run. `launch.json` says
    which GPU it holds and which pid to watch, so the loop can finish, monitor and
    exclude it exactly as if this process had started it.
    """
    out = {}
    cdir = ROOT / "claims"
    if not cdir.is_dir():
        return out
    for c in sorted(cdir.iterdir()):
        if not c.is_dir() or (ROOT / "results" / f"{c.name}.json").exists():
            continue
        d = ROOT / "work" / c.name
        try:
            rec = json.loads((d / "launch.json").read_text())
        except (OSError, ValueError):
            continue
        proc = AdoptedProc(int(rec["pid"]))
        if proc.poll() is not None:
            continue                # already dead; recover_orphans() will write it up
        item = next((e for e in load_queue() if e["name"] == c.name), None)
        if item is None:
            continue
        out[int(rec["gpu"])] = {"proc": proc, "item": item, "dir": d,
                                "started": float(rec.get("started") or time.time()),
                                "uuid": rec["uuid"],
                                # Inherit taint across restarts. Without this an adopted
                                # job forgets it was ever co-tenanted and is written valid.
                                "cotenant": rec["name"] in _TAINTED_AT_ADOPT,
                                "slot": int(rec["slot"]), "cores": rec["cores"]}
        log(f"ADOPTED {c.name} on gpu{rec['gpu']} pid={rec['pid']} "
            f"(in flight from a previous dispatcher)")
    return out


def cores_for(slot):
    lo = CORE_BASE + slot * CORES_PER_JOB
    return f"{lo}-{lo + CORES_PER_JOB - 1}"


def main():
    # Adopt BEFORE releasing stranded claims: an in-flight run has a work dir, so it is
    # not stranded, but it must be in `running` before the first pass computes free GPUs.
    QSTATE = ROOT / "quarantine.json"
    def _load_state():
        try:
            d = json.loads(QSTATE.read_text())
            return (d.get("quarantine", {}), set(d.get("tainted", [])),
                    d.get("burns", {}))
        except (OSError, ValueError):
            return {}, set(), {}
    def _save_state(q, t, b=None):
        # BURN COUNTS PERSIST TOO. They were process-local for exactly one commit, and the
        # consequence appeared within the hour: gpu6 and gpu7 destroyed two waves, the
        # dispatcher was restarted to bind the fix, and the very next release logged "0
        # prior burn(s)" -- full trust restored to the two devices that had just burned us,
        # because the counter died with the process.
        #
        # This is the same defect the comment below describes for quarantine and taint,
        # committed again in the fix for it. Anything that decides how much a device is
        # trusted has to outlive the process that learned it.
        try:
            QSTATE.write_text(json.dumps(
                {"quarantine": q, "tainted": sorted(t),
                 "burns": burn_count if b is None else b}, indent=1))
        except OSError:
            pass

    # Load the persisted taint BEFORE adopting. adopt_running() reads _TAINTED_AT_ADOPT to
    # decide whether a job it is re-adopting was already known to be co-tenanted, and the
    # load used to happen AFTER the call -- so the set was always empty and the taint was
    # never inherited. The fix for "co-tenancy did not survive a restart" therefore never
    # fired: it is the same build-a-guard-and-not-connect-it failure, committed inside the
    # fix for that failure. Ordering is the connection here.
    quarantine, tainted, _burns = _load_state()  # uuid -> release epoch; names seen co-tenanted
    burn_count.update({k: int(v) for k, v in (_burns or {}).items()})
    globals()["_TAINTED_AT_ADOPT"] = set(tainted)
    running = adopt_running()
    release_stranded_claims()   # claims with neither a result nor a work dir
    # Quarantine and co-tenancy live on DISK, not in process memory.
    #
    # Both were locals, so a dispatcher restart forgot every quarantined device and every
    # co-tenancy already observed on a still-running job. adopt_running() then re-adopted
    # that job with cotenant=False, and when it finished it was written `ok: true` -- a run
    # known to be contaminated, recorded as valid evidence. The campaign restarted the
    # dispatcher five times in one session to ship policy fixes, so it sat inside that
    # window repeatedly; it escaped only because every co-tenancy happened to be recorded
    # before the restart that followed it. Timing is not a control.
    clean_since = {}            # gpu uuid -> when it was first seen free again
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
                owners = byuuid.get(job["uuid"], [])
                # Two of OUR OWN trainers on one GPU corrupt the measurement exactly as
                # badly as a stranger's job, and `o != me` is blind to it by construction.
                mine_here = sum(1 for o in owners if o == me)
                if any(o and o != me for o in owners) or mine_here > 1:
                    if not job["cotenant"]:
                        who = "another of OURS" if mine_here > 1 else "a foreign tenant"
                        log(f"CO-TENANT ({who}) on gpu{g} during "
                            f"{job['item']['name']} -> INVALID")
                        quarantine[job["uuid"]] = time.time() + COTENANT_QUARANTINE_S
                        burn_count[job["uuid"]] = burn_count.get(job["uuid"], 0) + 1
                        # job has keys proc/item/dir/started/uuid/cotenant/slot/cores.
                        # `job["name"]` was a KeyError that would have crashed the
                        # dispatcher on the FIRST co-tenancy it detected -- before the
                        # quarantine was persisted -- turning a contamination guard into
                        # an outage. It never fired only because no co-tenancy occurred
                        # after the commit that introduced it.
                        tainted.add(job["item"]["name"])
                        _save_state(quarantine, tainted)
                        log(f"QUARANTINE gpu{g} for {COTENANT_QUARANTINE_S//60} min "
                            f"(L002_burned_gpu_reused)")
                    job["cotenant"] = True
        except OSError:
            pass

        for g, job in list(running.items()):
            if job["proc"].poll() is None:
                continue
            txt = (job["dir"] / "out.log").read_text(errors="replace")
            met = parse(txt)
            # Carry the hypothesis id onto the RESULT. coe.py E3 is opt-in on this field
            # (`hid = r.get("hypothesis_id"); if not hid: continue`), so without it the
            # declared activation rule is never evaluated and a treatment that silently
            # failed to engage is indistinguishable from a clean null -- which is the
            # exact failure the activation predicate exists to prevent.
            rec = {"name": job["item"]["name"], "cfg": job["item"]["cfg"], "gpu": g,
                   # Carried into the RESULT so analysis never depends on the queue still
                   # holding the entry. A cut queue entry used to erase a completed run
                   # from every verdict.
                   "wave_group": job["item"].get("wave_group"),
                   # ROLE, CARRIED FROM THE QUEUE ENTRY THAT ASSIGNED IT. Seven separate
                   # defects in this campaign came from re-deriving a run's role later by
                   # comparing its cfg against direction.PLATFORM, which moves under
                   # adoption: treatments silently became controls and were pooled into
                   # the baseline that judges treatments. The queue knew the role with
                   # certainty; nothing downstream should have to infer it.
                   "role": job["item"].get("role"),
                   # THE VARIANT ID, so the result is self-describing. No result record
                   # on disk carries one, which costs twice. First, provenance: a result
                   # whose queue row is later cut has no way to name the code that
                   # produced it, and rebuilding from its cfg answers with TODAY's
                   # generator rather than the one that ran. Second, diagnostics:
                   # claims.diagnostic_would_discriminate pools control readings of a
                   # named diagnostic across all history, so readings produced by
                   # different generator versions -- including one before and one after a
                   # diagnostic's own implementation was fixed -- are treated as the same
                   # quantity. Neither is repairable for runs already on disk; recording
                   # it here is what makes it repairable from now on.
                   "variant": job["item"].get("variant"),
                   "hypothesis_id": job["item"].get("hypothesis_id"),
                   "started": job["started"], "ended": time.time(),
                   "returncode": job["proc"].returncode, "metrics": met,
                   "cotenant_detected": job["cotenant"], "cores": job["cores"],
                   "ok": (job["proc"].returncode == 0 and "val_bpb" in met
                          and not job["cotenant"] and not _crashed(job)),
                   "invalid_reason": ("gpu co-tenancy during the run" if job["cotenant"]
                                      else _crashed(job) or ""),
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
        now = time.time()
        # Release a quarantined device EARLY once it has been verifiably clean for a
        # while, instead of serving out a fixed timer. The 45 minutes was a guess; the
        # thing it protects against -- a foreign tenant cycling on and off -- is directly
        # observable, so waiting it out while nvidia-smi shows the device empty is
        # substituting a clock for evidence. The ceiling stays as a backstop for the case
        # where we cannot see the tenant at all.
        try:
            _busy_uuids = {ln.split(",")[1].strip()
                           for ln in subprocess.run(
                               ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                                "--format=csv,noheader"], capture_output=True, text=True,
                               check=False).stdout.splitlines() if "," in ln}
        except OSError:
            _busy_uuids = None
        for _u in list(quarantine):
            if _busy_uuids is not None and _u not in _busy_uuids:
                clean_since.setdefault(_u, now)
                # BACK OFF ON REPEAT OFFENDERS. A device that has burned us before is not
                # proved clean by the same quiet window that failed last time. gpu6 and
                # gpu7 were released at 16:47:40Z on "no compute app for 8 min" and the
                # tenant was back 35 seconds after we launched, destroying R6XF_P3 -- the
                # second wave lost to this in ten minutes, on top of R6XF_P2. Eight minutes
                # of quiet is evidence a tenant has PAUSED, not that it has left; an
                # intermittent job looks identical to a departed one on that timescale.
                #
                # So the clean window each device must show doubles with each burn it has
                # caused, capped so a device is never retired permanently. Cheap when a
                # tenant really has gone (one wait), and it stops handing the same device
                # back to the same cycling job every eight minutes.
                _burns = burn_count.get(_u, 0)
                _need = min(QUARANTINE_CLEAR_S * (2 ** _burns), COTENANT_QUARANTINE_S)
                if now - clean_since[_u] >= _need:
                    log(f"QUARANTINE lifted early for {_u[:20]}: no compute app for "
                        f"{_need//60} min quiet required after {_burns} prior "
                        f"burn(s) (evidence, not timer)")
                    del quarantine[_u]
                    _save_state(quarantine, tainted)
                    clean_since.pop(_u, None)
                    continue
            else:
                clean_since.pop(_u, None)      # tenant came back; restart the clock
            if quarantine.get(_u, 0) <= now:
                del quarantine[_u]
                clean_since.pop(_u, None)
        free_gpus = [(g, uuid) for g, (used, uuid) in sorted(gpu_state().items())
                     if g not in running and used <= MINFREE_MB
                     and uuid not in quarantine]
        capacity = min(len(free_gpus), len(free_slots), MAX_GPUS - len(running))

        batch, waiting = ([], None) if capacity <= 0 else next_batch(cutoff, capacity)
        if waiting or (capacity <= 0 and time.time() < DEADLINE - 500):
            msg = waiting or f"0 of {MAX_GPUS} GPUs free"
            if time.time() - last_wait_log > WAIT_LOG_EVERY_S:
                log(f"WAITING: {msg}; {len(running)}/{MAX_GPUS} of ours running")
                last_wait_log = time.time()
        batch = [b for b in batch if b["name"] not in blocked_launch]
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
                # A LAUNCH THAT CANNOT SUCCEED MUST STOP TRYING. A missing variant file
                # does not appear by itself, so retrying it is futile -- and the retry loop
                # runs every poll, which flooded dispatch.log with 30+ identical lines in
                # three minutes and kept re-claiming and releasing two GPUs while a
                # perfectly launchable queue waited behind it. The wave is quarantined from
                # LAUNCHING after a few identical failures and reported once as needing
                # action, the same way WAVE WEDGED reports a wave that can never assemble.
                # The entry stays in the queue: it becomes launchable again the moment the
                # missing file is shipped, which is the actual fix an operator would apply.
                _lk = f"{item['name']}:{type(exc).__name__}"
                launch_fails[_lk] = launch_fails.get(_lk, 0) + 1
                _n = launch_fails[_lk]
                if _n <= LAUNCH_FAIL_MAX:
                    log(f"LAUNCH FAILED {item['name']}: {exc} (claim released)")
                if _n == LAUNCH_FAIL_MAX:
                    log(f"LAUNCH BLOCKED {item['name']}: failed {_n} times with the same "
                        f"error, which will not fix itself. Not retried until the cause is "
                        f"removed -- most often a queue entry whose 'variant' names a file "
                        f"that was never built or never shipped. The entry stays queued.")
                if _n >= LAUNCH_FAIL_MAX:
                    blocked_launch.add(item["name"])
                continue
            running[g] = {"proc": p, "item": item, "dir": d, "started": time.time(),
                          "uuid": uuid, "cotenant": False, "slot": slot, "cores": cores}
            # A trainer spends its first ~200 seconds tokenizing on the CPU and holds NO
            # GPU memory in that window, so nvidia-smi reports its GPU as free. A
            # dispatcher restarted during that window saw gpu6 and gpu7 as free and
            # launched a second pair straight on top of the first -- four of OUR trainers
            # on two GPUs, which the co-tenancy check does not catch because it only
            # looks for FOREIGN owners. This file is what lets a fresh dispatcher adopt a
            # run it did not start instead of double-booking its GPU.
            (d / "launch.json").write_text(json.dumps(
                {"pid": p.pid, "gpu": g, "uuid": uuid, "slot": slot, "cores": cores,
                 "started": running[g]["started"], "name": item["name"]}))
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
