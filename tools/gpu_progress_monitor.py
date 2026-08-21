#!/usr/bin/env python3
"""Lightweight GPU/experiment progress monitor for the OPHIS walltime campaign.

Reports, using time.time() for elapsed tracking:
  - remote GPU occupancy (index, mem, util) via a single ControlMaster'd SSH
  - which GPUs are truly free (0 MiB) and thus launchable for a *timed* benchmark
  - any live OPHIS run_gated/train.py processes and their elapsed seconds
  - the tail of the newest ophis_run logs (step/val_bpb progress)

One SSH connection per invocation (ControlMaster reused) to respect remote sshd
MaxStartups. Prints a compact, timestamped block meant to be run every ~5 min.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HOST = "user@223.167.85.180"
PORT = "50002"
REMOTE_WD = "/home/user/ophis_run"
CAMPAIGN_START_FILE = Path("/private/tmp/claude-501/-Users-baiyu-Desktop-OPHIS/"
                           "2fd70fe2-3319-45fc-b55a-518464d549e0/scratchpad/campaign_start.txt")

SSH = [
    "ssh", "-p", PORT,
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=12",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=3",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/ophis-ssh-%r@%h:%p",
    "-o", "ControlPersist=120",
    HOST,
]

REMOTE_CMD = r'''
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "@@PROCS@@"
ps -eo pid,etimes,args | grep -iE "run_gated|ophis_run|temp_autoresearch" | grep -v grep
echo "@@LOGS@@"
ls -t /home/user/ophis_run/*.log 2>/dev/null | head -4 | while read f; do
  echo "--- $f ---"
  grep -aE "step |val_bpb|bpb=|steps" "$f" 2>/dev/null | tail -2
done
'''


def campaign_start() -> float:
    if CAMPAIGN_START_FILE.exists():
        return float(CAMPAIGN_START_FILE.read_text().strip())
    now = time.time()
    CAMPAIGN_START_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAMPAIGN_START_FILE.write_text(str(now))
    return now


def main() -> int:
    now = time.time()
    start = campaign_start()
    elapsed = now - start
    print(f"=== OPHIS progress @ {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))} "
          f"(t+{elapsed/60:.1f} min since monitor start; time.time()={now:.1f}) ===")
    try:
        out = subprocess.run(SSH + [REMOTE_CMD], capture_output=True, text=True,
                             timeout=40).stdout
    except subprocess.TimeoutExpired:
        print("SSH timed out (remote busy / sshd throttling); will retry next tick.")
        return 0

    gpu_sec, _, rest = out.partition("@@PROCS@@")
    proc_sec, _, log_sec = rest.partition("@@LOGS@@")

    free = []
    print("GPUs (idx: mem_used / util):")
    for line in gpu_sec.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx, mem, util = parts
        mem_mib = int(mem.split()[0]) if mem.split() else 0
        flag = "  <-- FREE" if mem_mib < 500 else ""
        if mem_mib < 500:
            free.append(idx)
        print(f"  {idx}: {mem:>12} / {util:>5}{flag}")
    print(f"FREE GPUs (<500 MiB): {free if free else 'NONE'} "
          f"-> {'launchable (need >=2 for a pair)' if len(free) >= 2 else 'NOT launchable'}")

    procs = proc_sec.strip()
    print("OPHIS live procs:", procs if procs else "none")
    logs = log_sec.strip()
    if logs:
        print("Newest ophis_run log tails:")
        print(logs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
