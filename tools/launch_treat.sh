#!/usr/bin/env bash
# Retired: treatment-only runs cannot satisfy the paired search-policy gate.
set -euo pipefail
echo "ABORT: treatment-only launching is disabled; use tools/run_stage.py." >&2
exit 2
