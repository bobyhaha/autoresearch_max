#!/usr/bin/env python3
"""Free-GPU watcher. One ssh round-trip per check; prints an owner-attributed table.

Why this exists: the operator allocation on this shared box is at most 4 GPUs, and every
other GPU carries a foreign tenant. Co-tenancy invalidates a run (dispatch.py marks the
record ok:false), so "free" here means BOTH:
  - resident memory below MINFREE_MB, and
  - no compute app from any user other than us.

It reports; it never launches or kills. host/dispatch.py owns launching and already caps
itself at MAX_GPUS=4.
"""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hostcfg  # noqa: E402  remote target lives in the environment, not in git
MINFREE_MB = 1024
MAX_GPUS = 4                       # operator allocation; mirrors host/dispatch.py
LOG = ROOT / "runs" / "gpu_watch.log"

REMOTE = r"""
whoami
echo '@@GPU'
nvidia-smi --query-gpu=index,memory.used,uuid --format=csv,noheader,nounits
echo '@@APPS'
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=, read -r p u; do
  echo "$(ps -o user= -p ${p// /} 2>/dev/null),$u"
done
echo '@@OURS'
pgrep -u "$(whoami)" -af 'bin/python .*train\.py' | grep -v 'bash -c' | wc -l
# `nohup ... & disown` leaves a resident `bash -c` wrapper whose command line also
# contains the python path and script name, so a bare pgrep counts one dispatcher
# as two. health.py already filters it (see its dispatchers predicate); this file
# did not, and the two tools disagreed. A false '2 dispatchers' is the same signal
# as a real double-booking, and acting on it means killing a healthy dispatcher.
echo '@@DISPATCH'
pgrep -u "$(whoami)" -af 'bin/python .*dispatch\.py' | grep -v 'bash -c' | wc -l
echo '@@LOAD'
cat /proc/loadavg
"""


def probe():
    if not hostcfg.configured():
        return {"ok": False, "error": hostcfg.why_not()}
    out = subprocess.run(hostcfg.ssh_argv([REMOTE]), capture_output=True, text=True,
                         check=False)
    if out.returncode != 0:
        return {"ok": False, "error": (out.stderr or "").strip()[-300:]}
    sec, cur = {}, "me"
    sec[cur] = []
    for ln in out.stdout.splitlines():
        if ln.startswith("@@"):
            cur = ln[2:].lower()
            sec[cur] = []
        else:
            sec[cur].append(ln.strip())
    me = (sec.get("me") or [""])[0]
    owners = {}
    for ln in sec.get("apps", []):
        if "," not in ln:
            continue
        who, uu = [c.strip() for c in ln.split(",", 1)]
        owners.setdefault(uu, []).append(who or "?")
    gpus = []
    for ln in sec.get("gpu", []):
        parts = [c.strip() for c in ln.split(",")]
        if len(parts) != 3:
            continue
        idx, used, uu = int(parts[0]), float(parts[1]), parts[2]
        who = owners.get(uu, [])
        foreign = sorted({w for w in who if w and w != me})
        gpus.append({"idx": idx, "used_mb": used, "uuid": uu,
                     "foreign": foreign, "ours": me in who,
                     "free": used <= MINFREE_MB and not foreign})
    return {"ok": True, "me": me, "gpus": gpus,
            "our_procs": int((sec.get("ours") or ["0"])[0] or 0),
            "dispatchers": int((sec.get("dispatch") or ["0"])[0] or 0),
            "load": (sec.get("load") or [""])[0]}


def render(st):
    ts = time.strftime("%FT%TZ", time.gmtime())
    if not st["ok"]:
        return f"[{ts}] UNREACHABLE: {st['error']}"
    free = [g["idx"] for g in st["gpus"] if g["free"]]
    ours = [g["idx"] for g in st["gpus"] if g["ours"]]
    lines = [f"[{ts}] free={free} ours={ours} "
             f"dispatchers={st['dispatchers']} load={st['load'].split(' ')[0]}"]
    for g in st["gpus"]:
        tag = "FREE" if g["free"] else ("OURS" if g["ours"] else "foreign")
        lines.append(f"   gpu{g['idx']} {g['used_mb']:>7.0f} MiB  {tag:<7} "
                     f"{','.join(g['foreign']) or '-'}")
    if not st["dispatchers"]:
        lines.append("   ALERT: no dispatcher on host - free GPUs will NOT be used")
    elif free and len(ours) < MAX_GPUS:
        lines.append(f"   {len(free)} GPU(s) free and dispatcher up; it will claim up to "
                     f"{MAX_GPUS - len(ours)} more (cap {MAX_GPUS})")
    if len(ours) > MAX_GPUS:
        lines.append(f"   ALERT: {len(ours)} GPUs held, over the {MAX_GPUS} cap")
    return "\n".join(lines)


def main():
    loop = "--loop" in sys.argv
    interval = 1800
    for a in sys.argv[1:]:
        if a.startswith("--interval="):
            interval = int(a.split("=", 1)[1])
    LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        st = probe()
        txt = render(st)
        print(txt, flush=True)
        with open(LOG, "a") as f:
            f.write(txt + "\n")
        (ROOT / "runs" / "gpu_watch.json").write_text(json.dumps(
            {"checked_at": time.time(), **st}, indent=1))
        if not loop:
            return 0 if st["ok"] else 1
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
