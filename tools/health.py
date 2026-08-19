#!/usr/bin/env python3
"""The 20-minute health check: is Claude actually working, and are OUR GPUs running?

Two independent questions. Answering either one with a proxy is how the v3 campaign
convinced itself it was alive while it was not:

  * "GPUs running" was read off `nvidia-smi` totals on a box shared with other tenants.
    Foreign processes at 87-127 GB were counted as activity. Every claim about GPU
    liveness here is therefore filtered to processes OWNED BY US, by uid, and nothing
    else is ever reported as our work.
  * "Claude working" was read off a heartbeat file that the monitor itself wrote, so it
    stayed fresh while nothing scientific happened. Progress here means a RESEARCH
    ARTIFACT changed on disk -- a result, a round, a critique, a code edit -- not that a
    logger ran.

Exit status: 0 healthy, 1 degraded (something to fix), 2 unreachable.

    python3 tools/health.py            # human-readable
    python3 tools/health.py --json
    python3 tools/health.py --loop     # run forever, every CHECK_EVERY_S
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
SWEEP = REPO / "runs" / "sweep"
CHECK_EVERY_S = 20 * 60

import hostcfg  # noqa: E402  remote target lives in the environment, not in git

# Progress = one of these changed. A log file is deliberately NOT on this list.
# lit/ is on it because building the corpus and extracting claims is the research during
# Phase 0 -- without these entries the checker reports STALLED for the hours that the
# campaign is doing exactly what it was told to do.
PROGRESS_GLOBS = ("runs/sweep/results/*.json", "rounds/*.md", "critiques/*.md",
                  "papers/*.md", "tools/*.py", "runs/sweep/queue.json",
                  "lit/index.json", "lit/claims.jsonl", "lit/mechanisms.jsonl",
                  "lit/sources/*.txt")
STALL_S = 45 * 60          # no artifact touched this long => Claude is not working
NO_RESULT_S = 40 * 60      # no experiment landed this long while GPUs idle => stalled queue


def _sh(argv, timeout=60):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except OSError as exc:
        return 1, "", str(exc)


def newest_artifact():
    """(relative_path, age_seconds) of the most recently modified research artifact."""
    best, best_t = None, 0.0
    for pat in PROGRESS_GLOBS:
        for f in REPO.glob(pat):
            try:
                t = f.stat().st_mtime
            except OSError:
                continue
            if t > best_t:
                best, best_t = f, t
    if best is None:
        return None, None
    return str(best.relative_to(REPO)), time.time() - best_t


def remote_state():
    """GPU and process state, filtered to OUR uid. Never reports foreign work as ours."""
    script = r'''
me=$(id -un)
echo "USER=$me"
echo "LOADAVG=$(cut -d" " -f1-3 /proc/loadavg)"
echo "NPROC=$(nproc)"
# our processes only
ps -u "$me" -o pid=,etimes=,args= 2>/dev/null | grep -E "train\.py|dispatch\.py" | grep -v grep \
  | while read -r pid etime rest; do echo "OURPROC=$pid|$etime|$rest"; done
# every compute app on every gpu, with its owner resolved
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null \
  | while IFS=, read -r pid uuid mem; do
      pid=$(echo "$pid" | tr -d " ")
      own=$(ps -o user= -p "$pid" 2>/dev/null | tr -d " ")
      echo "APP=$pid|$own|$(echo "$uuid" | tr -d ' ')|$(echo "$mem" | tr -d ' ')"
    done
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
  | while IFS=, read -r i uuid mem util; do echo "GPU=$(echo $i|tr -d ' ')|$(echo $uuid|tr -d ' ')|$(echo $mem|tr -d ' ')|$(echo $util|tr -d ' ')"; done
'''
    if not hostcfg.configured():
        return {"reachable": False, "error": hostcfg.why_not()}
    rc, out, err = _sh(hostcfg.ssh_argv([script]), timeout=90)
    if rc != 0 and not out:
        return {"reachable": False, "error": err or f"ssh rc={rc}"}

    st = {"reachable": True, "user": None, "loadavg": None, "nproc": None,
          "our_procs": [], "gpus": [], "apps": []}
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k == "USER":
            st["user"] = v
        elif k == "LOADAVG":
            st["loadavg"] = v
        elif k == "NPROC":
            st["nproc"] = int(v) if v.isdigit() else None
        elif k == "OURPROC":
            pid, etime, args = (v.split("|", 2) + ["", ""])[:3]
            st["our_procs"].append({"pid": pid, "elapsed_s": int(etime or 0), "args": args})
        elif k == "APP":
            pid, own, uuid, mem = (v.split("|") + ["", "", "", ""])[:4]
            st["apps"].append({"pid": pid, "owner": own, "uuid": uuid, "mem": mem})
        elif k == "GPU":
            i, uuid, mem, util = (v.split("|") + ["", "", "", ""])[:4]
            st["gpus"].append({"index": i, "uuid": uuid, "mem_mb": mem, "util": util})

    me = st["user"]
    ours = {a["uuid"] for a in st["apps"] if a["owner"] == me}
    foreign = {a["uuid"] for a in st["apps"] if a["owner"] and a["owner"] != me}
    for g in st["gpus"]:
        g["ours"] = g["uuid"] in ours
        g["foreign"] = g["uuid"] in foreign
    st["gpus_ours"] = sorted(g["index"] for g in st["gpus"] if g["ours"])
    st["gpus_foreign"] = sorted(g["index"] for g in st["gpus"] if g["foreign"])
    st["gpus_free"] = sorted(g["index"] for g in st["gpus"]
                             if not g["ours"] and not g["foreign"])
    st["trainers"] = [p for p in st["our_procs"] if "train.py" in p["args"]]
    # Count the PYTHON dispatcher, not the shell that launched it. `nohup ... & disown`
    # can leave a bash wrapper resident whose command line also contains "dispatch.py",
    # and counting it reported "2 dispatchers - double-booking risk" against a single
    # healthy dispatcher. A false alarm here is expensive: it is the same signal that
    # would report a genuine double-dispatcher, so it must not cry wolf.
    st["dispatchers"] = [p for p in st["our_procs"]
                         if "dispatch.py" in p["args"]
                         and "/bin/bash" not in p["args"] and "bash -c" not in p["args"]]
    return st


def check():
    art, age = newest_artifact()
    n_results = len(list((SWEEP / "results").glob("*.json"))) if (SWEEP / "results").is_dir() else 0
    res_age = None
    if (SWEEP / "results").is_dir():
        ts = [f.stat().st_mtime for f in (SWEEP / "results").glob("*.json")]
        res_age = (time.time() - max(ts)) if ts else None

    r = remote_state()
    alerts = []

    # --- Q1: is Claude working? -------------------------------------------------
    if age is None:
        claude_working = False
        alerts.append("no research artifact exists yet - campaign has not started")
    elif age > STALL_S:
        claude_working = False
        alerts.append(f"STALLED: newest artifact {art} is {age/60:.0f} min old "
                      f"(limit {STALL_S//60}) - Claude is not doing research")
    else:
        claude_working = True

    # --- Q2: are OUR GPUs running? ----------------------------------------------
    if not r.get("reachable"):
        gpus_running = None
        alerts.append(f"host unreachable: {r.get('error')} - liveness UNKNOWN, not 'idle'")
    else:
        n_tr = len(r["trainers"])
        gpus_running = n_tr > 0
        if not r["dispatchers"]:
            alerts.append("no dispatcher process on host - nothing will ever launch")
        if n_tr == 0 and r["gpus_free"]:
            alerts.append(f"0 trainers of ours while GPUs {r['gpus_free']} are free - "
                          f"queue is empty or the gate is shut")
        if len(r["dispatchers"]) > 1:
            alerts.append(f"{len(r['dispatchers'])} dispatchers running - double-booking risk")
        if res_age is not None and res_age > NO_RESULT_S and n_tr == 0:
            alerts.append(f"no result landed in {res_age/60:.0f} min and nothing is training")
        # contention is a measurement-validity alert, not cosmetic: our data loader is
        # single-threaded Python and its speed sets val_bpb through step count.
        try:
            la = float((r["loadavg"] or "0").split()[0])
            if r["nproc"] and la > 0.85 * r["nproc"]:
                alerts.append(f"host loadavg {la:.0f}/{r['nproc']} cores - CPU contention "
                              f"will inflate val_bpb variance; treat small effects as noise")
        except (ValueError, IndexError):
            pass

    status = ("unreachable" if gpus_running is None else
              ("healthy" if (claude_working and not alerts) else "degraded"))
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "claude_working": claude_working,
            "newest_artifact": art, "artifact_age_min": round(age / 60, 1) if age else None,
            "gpus_running_ours": gpus_running,
            "n_results": n_results,
            "last_result_min": round(res_age / 60, 1) if res_age else None,
            "remote": r, "alerts": alerts}


def render(h):
    r = h.get("remote") or {}
    out = [f"[{h['ts']}] {h['status'].upper()}",
           f"  claude_working : {h['claude_working']}  "
           f"(newest artifact: {h['newest_artifact']} @ {h['artifact_age_min']} min)",
           f"  results        : {h['n_results']} (last {h['last_result_min']} min ago)"]
    if r.get("reachable"):
        out += [f"  our trainers   : {len(r['trainers'])}  our dispatchers: {len(r['dispatchers'])}",
                f"  gpus ours      : {r['gpus_ours']}",
                f"  gpus foreign   : {r['gpus_foreign']}",
                f"  gpus free      : {r['gpus_free']}",
                f"  host load      : {r['loadavg']} on {r['nproc']} cores"]
    else:
        out.append(f"  host           : UNREACHABLE ({r.get('error')})")
    for a in h["alerts"]:
        out.append(f"  ALERT: {a}")
    return "\n".join(out)


def main():
    once = "--loop" not in sys.argv
    while True:
        h = check()
        line = json.dumps(h) if "--json" in sys.argv else render(h)
        print(line, flush=True)
        logdir = REPO / "runs"
        logdir.mkdir(exist_ok=True)
        with open(logdir / "health.log", "a") as f:
            f.write(json.dumps({k: v for k, v in h.items() if k != "remote"}) + "\n")
        (REPO / "runs" / "HEALTH.json").write_text(json.dumps(h, indent=1))
        if once:
            return 2 if h["status"] == "unreachable" else (0 if h["status"] == "healthy" else 1)
        time.sleep(CHECK_EVERY_S)


if __name__ == "__main__":
    sys.exit(main())
