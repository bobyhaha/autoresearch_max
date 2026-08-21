#!/usr/bin/env bash
# Retired: this path bound an environment but did not perform physical-GPU sampling or append
# the verified RunRecord itself. Policy-aware GPU work has one launcher of record.
set -euo pipefail
echo "ABORT: run_h200_training.sh is retired; use tools/run_stage.py." >&2
exit 2
