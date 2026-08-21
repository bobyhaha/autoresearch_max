#!/usr/bin/env bash
# Retired: a historical baseline is not a valid concurrent control.
# Usage: monitor_treat.sh TAG [BASELINE_TAG] [SIGMA_REPRO]
#   BASELINE_TAG default rebase10 (1660). For 2766 experiments pass x_steps2766.
#   SIGMA_REPRO default 0.0014 (STALE 3-shard prior; override with the 10-shard measurement).
set -euo pipefail
echo "ABORT: cross-time treatment verdicts are disabled; use monitor_ab.sh on concurrent arms." >&2
exit 2
