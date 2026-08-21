#!/usr/bin/env bash
# Retired: this direct diagnostic launcher did not hold the advisory scheduler lock or
# append policy-grade RunRecords. Track-B analysis remains available for old
# artifacts, but new GPU measurements must use the gated staged scheduler.
set -euo pipefail
echo "ABORT: direct Track-B launching is disabled; use tools/run_stage.py." >&2
exit 2
