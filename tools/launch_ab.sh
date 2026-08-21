#!/usr/bin/env bash
# Retired: this launcher authorized labels but still accepted free-form arm env
# and wrote no RunRecords. It cannot satisfy search-policy or GPU-sampling gates.
set -euo pipefail
echo "ABORT: launch_ab.sh is retired; use tools/run_stage.py for paired GPU work." >&2
exit 2
