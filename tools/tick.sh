#!/bin/bash
# THE 20-MINUTE TICK. Report-and-sync only: it never launches or kills an experiment.
#
# One loop does all four things that must happen on a cadence, because splitting them is
# how v3 ended up with a watchdog running from a DELETED script while its gate check kept
# passing on that dead process's output:
#   1. pull results from the host      (local analysis has something true to read)
#   2. run the health check            (is Claude working, are OUR GPUs running)
#   3. evaluate the gate               (what may launch, and the decision cutoff)
#   4. push the gate verdict to host   (the dispatcher fails safe without a fresh one)
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"

# Remote target comes from the environment or .ophis_host, never from this file.
eval "$(python3 tools/hostcfg.py --shell)" || exit 1
if [ -z "${OPHIS_TARGET:-}" ]; then
  echo "remote host not configured; see .env.example" >&2; exit 1
fi
HOST="$OPHIS_TARGET"
SSHOPT=(-i "$OPHIS_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 \
        -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -p "$OPHIS_PORT")
INTERVAL=${OPHIS_TICK_INTERVAL:-1200}

tick() {
  mkdir -p runs/sweep/results
  scp -q "${SSHOPT[@]/-p/-P}" "$HOST:~/$OPHIS_REMOTE_DIR/sweep/results/*.json" runs/sweep/results/ 2>/dev/null
  python3 tools/health.py || true          # exit 1 = degraded; the loop must not die on it
  python3 tools/gate.py    || true
  scp -q "${SSHOPT[@]/-p/-P}" runs/sweep/GATE_STATUS.json "$HOST:~/$OPHIS_REMOTE_DIR/sweep/GATE_STATUS.json" 2>/dev/null \
    || echo "  WARN: could not publish gate verdict; dispatcher will fail safe and stop launching"
  # Push newly queued work by MERGING on name, never by overwriting. Two sessions (or a
  # session plus this loop) both syncing would otherwise silently delete each other's
  # queued experiments -- the host copy is authoritative for anything already there.
  if [ -f runs/sweep/queue.json ]; then
    scp -q "${SSHOPT[@]/-p/-P}" runs/sweep/variants/*.py "$HOST:~/$OPHIS_REMOTE_DIR/sweep/variants/" 2>/dev/null
    ssh "${SSHOPT[@]}" "$HOST" 'cat > /tmp/ophis_queue_incoming.json' < runs/sweep/queue.json || \
      echo "  WARN: could not ship queue to host"
    ssh -n "${SSHOPT[@]}" "$HOST" "OPHIS_REMOTE_DIR='$OPHIS_REMOTE_DIR' python3 - <<'PYMERGE'
import json, os, pathlib
q = pathlib.Path.home()/os.environ.get('OPHIS_REMOTE_DIR','ophis_v3')/'sweep'/'queue.json'
try: cur = json.loads(q.read_text())
except Exception: cur = []
try: inc = json.loads(open('/tmp/ophis_queue_incoming.json').read())
except Exception: inc = []
have = {e['name'] for e in cur}
new = [e for e in inc if e['name'] not in have]
if new:
    q.write_text(json.dumps(cur + new, indent=1))
print(f'queue: {len(cur)} on host + {len(new)} merged in')
PYMERGE"
  fi
  python3 tools/council.py status
}

if [ "${1:-}" = "--loop" ]; then
  while true; do tick; sleep "$INTERVAL"; done
else
  tick
fi
