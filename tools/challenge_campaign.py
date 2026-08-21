#!/usr/bin/env python3
"""Collect the historical five-minute shadow campaign without minting evidence.

This tool cannot produce an adoption verdict.  It preserves the canonical
reconciled results document and updates only its `shadow_collection` section.
New experiments belong in tools/run_stage.py and tools/run_gated.py.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

import os
import shlex

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/challenges/walltime_5min_h200/results.json"
SETUP = ROOT / "research/setup/reconciliation.json"
FRAME = "walltime_5min_h200"


def _ssh_command() -> list[str]:
    """Resolve the remote shell prefix from OPHIS_SSH, never a hardcode.

    A hardcoded host is how this tool became unrunnable anywhere but one
    operator's box. Example:
      export OPHIS_SSH="ssh -p 50002 -o BatchMode=yes user@host"
    """
    prefix = os.environ.get("OPHIS_SSH", "").strip()
    if not prefix:
        raise SystemExit(
            "OPHIS_SSH is not set. Export the remote shell prefix, e.g.\n"
            '  export OPHIS_SSH="ssh -p 50002 -o BatchMode=yes user@host"'
        )
    return shlex.split(prefix)


def _remote_dir() -> str:
    """Resolve the remote working directory from OPHIS_REMOTE_DIR, never a hardcode."""
    remote = os.environ.get("OPHIS_REMOTE_DIR", "").strip()
    if not remote:
        raise SystemExit(
            "OPHIS_REMOTE_DIR is not set. Export the remote working directory, e.g.\n"
            "  export OPHIS_REMOTE_DIR=/home/user/ai4ai/baiyu_5min_challenge"
        )
    return remote


def _frame() -> dict:
    setup = json.loads(SETUP.read_text(encoding="utf-8"))
    return next(item for item in setup["scopes"] if item["scope_id"] == FRAME)


def _sh(command: str) -> str:
    try:
        result = subprocess.run(
            _ssh_command() + [command],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(f"shadow collection failed: {stderr.strip() or exc}") from exc
    return result.stdout


def collect() -> dict:
    """Scrape finished legacy runs into explicitly non-ledger-grade records."""
    frame = _frame()
    baseline = frame["baseline"]
    sigma = float(baseline["effective_sigma"])
    gate = 2.0 * sigma

    remote = _remote_dir()
    listing = _sh(f"ls -d {remote}/runs/*/")
    experiments = [Path(path).name for path in listing.split() if path.strip()]
    blob = _sh(
        f"cd {remote}/runs && "
        "grep -H 'val_bpb:' */*/seed*/run.log 2>/dev/null || true; "
        "echo '===STEPS==='; "
        "for f in */*/seed*/run.log; do "
        "last=$(grep -oE '^(step [0-9]+|num_steps:[[:space:]]+[0-9]+)' \"$f\" "
        "| tail -1 | grep -oE '[0-9]+'); "
        "echo \"$f $last\"; done"
    )
    values: dict[tuple[str, str, int], float] = {}
    steps: dict[tuple[str, str, int], int] = {}
    value_part, step_part = (
        blob.split("===STEPS===", 1) if "===STEPS===" in blob else (blob, "")
    )
    for line in value_part.splitlines():
        match = re.match(
            r"([^/]+)/([^/]+)/seed(\d+)/run\.log:\s*val_bpb:\s*([0-9.]+)",
            line.strip(),
        )
        if match:
            values[(match[1], match[2], int(match[3]))] = float(match[4])
    for line in step_part.splitlines():
        match = re.match(
            r"([^/]+)/([^/]+)/seed(\d+)/run\.log\s+(\d+)",
            line.strip(),
        )
        if match:
            steps[(match[1], match[2], int(match[3]))] = int(match[4])

    records = []
    for experiment_id in sorted(experiments):
        if experiment_id.startswith("baseline"):
            continue
        seeds = sorted(seed for exp, _arm, seed in values if exp == experiment_id)
        pairs = []
        for seed in sorted(set(seeds)):
            treatment = values.get((experiment_id, "treat", seed))
            control = values.get((experiment_id, "ctrl", seed))
            if treatment is None or control is None:
                continue
            pairs.append(
                {
                    "seed": seed,
                    "treat": treatment,
                    "ctrl": control,
                    "delta": round(treatment - control, 6),
                    "treat_steps": steps.get((experiment_id, "treat", seed)),
                    "ctrl_steps": steps.get((experiment_id, "ctrl", seed)),
                }
            )
        if not pairs:
            continue
        deltas = [float(pair["delta"]) for pair in pairs]
        count = len(deltas)
        mean = st.mean(deltas)
        sd = st.stdev(deltas) if count > 1 else math.nan
        t_stat = (
            mean / (sd / math.sqrt(count))
            if count > 1 and sd > 0
            else math.nan
        )
        consistent_direction = all(value < 0 for value in deltas) or all(
            value > 0 for value in deltas
        )
        positive_screen = (
            count >= int(frame["min_seeds"])
            and all(value < 0 for value in deltas)
            and abs(mean) > gate
        )
        records.append(
            {
                "id": experiment_id,
                "n_pairs": count,
                "treat_mean": round(st.mean(pair["treat"] for pair in pairs), 6),
                "ctrl_mean": round(st.mean(pair["ctrl"] for pair in pairs), 6),
                "mean_delta": round(mean, 6),
                "sd_delta": None if math.isnan(sd) else round(sd, 6),
                "paired_t": None if math.isnan(t_stat) else round(t_stat, 2),
                "all_same_sign": consistent_direction,
                "verdict": (
                    "shadow_positive"
                    if positive_screen
                    else ("shadow_screening" if count < int(frame["min_seeds"]) else "shadow_reject")
                ),
                "pairs": pairs,
            }
        )

    return {
        "frame": FRAME,
        "baseline_mean": float(baseline["observed"]),
        "effective_sigma": sigma,
        "gate_2sigma": gate,
        "experiments": records,
        "provenance": (
            "SHADOW CAMPAIGN ONLY: no RunRecords, authorization, frozen experiment "
            "fingerprints, or adoption authority. Re-run any candidate through run_stage.py."
        ),
    }


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command not in {"collect", "status"}:
        raise SystemExit("usage: challenge_campaign.py [collect|status]")
    report = collect()
    if command == "collect":
        document = json.loads(OUT.read_text(encoding="utf-8"))
        document["shadow_collection"] = report
        OUT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        f"frame {report['frame']} baseline {report['baseline_mean']} "
        f"shadow_experiments {len(report['experiments'])}"
    )
    for experiment in report["experiments"]:
        print(
            f"  [{experiment['verdict']}] {experiment['id']} "
            f"n={experiment['n_pairs']} delta={experiment['mean_delta']:+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
