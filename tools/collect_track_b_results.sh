#!/usr/bin/env bash
# Collect Track B baseline results from remote H200.
#
# Usage:
#   bash tools/collect_track_b_results.sh [NUM_SEEDS] [START_SEED]
#
# Fetches track_b_result_seed{N}.json and the training log for each seed.
set -euo pipefail

NUM_SEEDS="${1:-3}"
START_SEED="${2:-42}"
REMOTE_USER="user"
REMOTE_HOST="223.167.85.180"
REMOTE_PORT="50002"
REMOTE_BASE="/home/user/ph/autoresearch"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_RESULTS="${LOCAL_DIR}/track_b_results"

for value_name in NUM_SEEDS START_SEED; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ABORT: ${value_name} must be a non-negative integer, got ${value}" >&2
        exit 2
    fi
done
if (( NUM_SEEDS < 1 )); then
    echo "ABORT: NUM_SEEDS must be at least 1" >&2
    exit 2
fi

mkdir -p "$LOCAL_RESULTS"

echo "=== Collecting Track B results for ${NUM_SEEDS} seeds ==="

ALL_READY=1
for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=$((START_SEED + i))
    REMOTE_DIR="${REMOTE_BASE}/track_b_seed${SEED}"
    LOCAL_SEED_DIR="${LOCAL_RESULTS}/seed${SEED}"
    mkdir -p "$LOCAL_SEED_DIR"

    # Check if result file exists
    if ssh -p "$REMOTE_PORT" "${REMOTE_USER}@${REMOTE_HOST}" "test -f ${REMOTE_DIR}/track_b_result_seed${SEED}.json" 2>/dev/null; then
        scp -P "$REMOTE_PORT" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/track_b_result_seed${SEED}.json" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/track_b_seed${SEED}.log" \
            "$LOCAL_SEED_DIR/" 2>/dev/null
        echo "  seed ${SEED}: OK"
    else
        echo "  seed ${SEED}: NOT READY (no track_b_result_seed${SEED}.json)"
        ALL_READY=0
    fi
done

if [ "$ALL_READY" -eq 1 ]; then
    echo ""
    echo "=== All seeds collected. Running analysis... ==="
    cd "$LOCAL_DIR"
    python3 -c "
import json, sys, os, math
from tools.track_b_equal_time import validate_result

results = []
for seed in range(${START_SEED}, ${START_SEED} + ${NUM_SEEDS}):
    path = f'track_b_results/seed{seed}/track_b_result_seed{seed}.json'
    if not os.path.exists(path):
        print(f'seed {seed}: missing', file=sys.stderr)
        continue
    with open(path) as f:
        r = json.load(f)
    if int(r.get('seed', -1)) != seed:
        raise SystemExit(
            f'seed {seed}: filename/result seed mismatch ({r.get(\"seed\")!r})'
        )
    validate_result('baseline', r)
    results.append(r)

if not results:
    print('No results found', file=sys.stderr)
    sys.exit(1)
reference_hashes = results[0]['code_hashes']
if any(r['code_hashes'] != reference_hashes for r in results[1:]):
    raise SystemExit('Collected baseline seeds do not use identical code/data hashes')

# Report step-sample distribution for operations diagnostics, but use each
# independent run/seed as the unit for the headline latency estimate.
step_times = []
run_medians = []
for r in results:
    step_times.extend(r['step_times_ms'])
    rs = sorted(r['step_times_ms'])
    run_medians.append((rs[(len(rs)-1)//2] + rs[len(rs)//2]) / 2)

n = len(step_times)
s = sorted(step_times)
median = s[n // 2]
median_run = sorted(run_medians)[len(run_medians) // 2]
mean = sum(s) / n
std = math.sqrt(sum((x - mean) ** 2 for x in s) / n)
p90 = s[int(n * 0.90)]
p95 = s[int(n * 0.95)]

steady_durations = [
    float(r.get('cumulative_steady_time_s', sum(r['step_times_ms']) / 1000.0))
    for r in results
]
t_ref = sorted(steady_durations)[len(steady_durations) // 2]

print(f'Seeds analyzed:      {len(results)}')
print(f'Total step samples:  {n}')
print(f'Median pooled step:  {median:.1f} ms (diagnostic only)')
print(f'Median run median:   {median_run:.1f} ms (run is the unit)')
print(f'Mean step time:      {mean:.1f} ms')
print(f'Std step time:       {std:.1f} ms')
print(f'P90 step time:       {p90:.1f} ms')
print(f'P95 step time:       {p95:.1f} ms')
print(f'CV (std/mean):       {std/mean*100:.1f}%')
print(f'Median steady time:   {t_ref:.1f} s')
print()
print('NOTE: Track B measures synchronized steady-state training-step time,')
print('not end-to-end wall clock. Same-step runs are token/step matched, not')
print('FLOP matched. For an equal-steady-training-time diagnostic, run every')
print('config under the SAME STOP_MODE=time and TIME_BUDGET:')
print('  python3 tools/track_b_equal_time.py')
print()

# Per-seed summary
for r in sorted(results, key=lambda x: x['seed']):
    st = r['step_times_ms']
    ss = sorted(st)
    m = ss[len(ss)//2]
    print(f\"  seed {r['seed']}: median={m:.1f}ms  mean={sum(ss)/len(ss):.1f}ms  final_bpb={r['final_val_bpb']:.6f}\")

# Step-sample CV describes within-run operations stability. It does not make a
# single seed sufficient for a quality or speed verdict; runs/GPU assignments
# are the independent experimental units.
cv = std / mean * 100
print(f'\nPooled step-sample CV={cv:.1f}% (descriptive only).')
print('Effect verdicts still require >=3 paired runs on matched physical GPUs.')
"
else
    echo ""
    echo "=== Some seeds not ready. Check logs for errors. ==="
    echo "To check status: ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} 'ls -la ${REMOTE_BASE}/track_b_seed*/track_b_result_seed*.json 2>/dev/null'"
fi
