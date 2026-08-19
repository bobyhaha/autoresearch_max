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
  # HOST POLICY MODULES: ship and VERIFY every tick. host/dispatch.py imports these from
  # the sweep directory, so a fix made locally does nothing until it lands there -- and a
  # stale copy fails SILENTLY, in the direction of refusing work the policy now allows.
  # This has now cost the campaign three times (L012 first, then a dry-rule fix that left
  # the swdiv repair refused as split, then a metric-parser fix that ran stale for half an
  # hour). Remembering to scp is not a control; comparing digests is.
  for _m in direction.py claims.py lit.py; do
    [ -f "tools/$_m" ] || continue
    _l=$(md5 -q "tools/$_m" 2>/dev/null || md5sum "tools/$_m" | cut -d' ' -f1)
    _r=$(ssh -n "${SSHOPT[@]}" "$HOST" "md5sum ~/$OPHIS_REMOTE_DIR/sweep/$_m 2>/dev/null | cut -d' ' -f1")
    if [ "$_l" != "$_r" ]; then
      scp -q "${SSHOPT[@]/-p/-P}" "tools/$_m" "$HOST:~/$OPHIS_REMOTE_DIR/sweep/$_m" 2>/dev/null
      _v=$(ssh -n "${SSHOPT[@]}" "$HOST" "md5sum ~/$OPHIS_REMOTE_DIR/sweep/$_m 2>/dev/null | cut -d' ' -f1")
      if [ "$_l" = "$_v" ]; then
        echo "  shipped $_m ($_l)"
        # Shipping is NOT taking effect. host/dispatch.py imports these modules once at
        # startup, so a dispatcher already running keeps the OLD code in memory and goes
        # on enforcing the old policy. That is exactly what happened after the dry-rule
        # fix: direction.py landed on the host and the swdiv repair kept being refused as
        # split, because the live dispatcher had imported the previous version minutes
        # earlier. A module is live only after a restart.
        if ssh -n "${SSHOPT[@]}" "$HOST" 'pgrep -f "dispatch.py [0-9]" | grep -qv "bash -c"' 2>/dev/null; then
          echo "    NOTE: a dispatcher is RUNNING and still holds the old $_m in memory."
          echo "    Restart it at the next idle gap or the new policy will not bind."
        fi
      else
        echo "  WARN: $_m FAILED to ship (local $_l host $_v)"
      fi
    fi
  done
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
base = pathlib.Path.home()/os.environ.get('OPHIS_REMOTE_DIR','ophis_v3')/'sweep'
q = base/'queue.json'
try: cur = json.loads(q.read_text())
except Exception: cur = []
try: inc = json.loads(open('/tmp/ophis_queue_incoming.json').read())
except Exception: inc = []

def launched(name):
    # These paths MUST be anchored at the sweep directory. They were previously built
    # with os.path.join of a bare 'results' against a cwd of the home directory, so they
    # never resolved and this predicate was False for every entry -- so the guard that
    # protects a run already on disk from being rewritten had never once fired.
    return ((base/'results'/(name + '.json')).exists()
            or (base/'claims'/name).exists())

# Merge by name, and UPDATE entries that already exist. Appending only new names
# silently pins the host to whatever an entry looked like when it first arrived: a
# locally corrected variant hash or a newly attached hypothesis_id never lands, and
# the run executes stale code while the local queue says otherwise. An entry that has
# already RUN or been CLAIMED is left untouched: rewriting a launched entry would
# misdescribe a run that is already on disk.
have = {e['name']: i for i, e in enumerate(cur)}
added = updated = 0
for e in inc:
    i = have.get(e['name'])
    if i is None:
        cur.append(e); added += 1
        continue
    if launched(e['name']) or cur[i] == e:
        continue
    cur[i] = e; updated += 1

# DELETIONS PROPAGATE, for unlaunched entries only. Merge-without-delete meant a locally
# removed entry lived on here forever: R2V/R2S were rebuilt against a new base and
# re-queued under new names, and the superseded entries stayed on the host sharing the
# SAME wave_group as their replacements. That silently doubled those waves to 8 members
# and left them carrying hypothesis ids belonging to other experiments -- a wave that
# can never launch intact and would misattribute itself if it did. A queue that only
# ever grows is not a queue, it is a log.
removed = []
if inc:                       # never prune against an empty or unreadable local queue
    want = {e['name'] for e in inc}
    keep = []
    for e in cur:
        if e['name'] in want or launched(e['name']):
            keep.append(e)
        else:
            removed.append(e['name'])
    cur = keep

# ORDER PROPAGATES, for entries not yet launched. The merge matched entries by name and
# kept the HOST's ordering, so a locally computed decision -- which wave the selector says
# runs next -- never reached the dispatcher, which reads its queue top-down. The operator
# hand-pushed an ordering to the host twice in one session because of this, and both times
# it was invisible in the repository afterwards. Launched entries keep their position:
# reordering a run already on disk would misdescribe history.
want = [e['name'] for e in inc]
pos = {n: i for i, n in enumerate(want)}
started = [e for e in cur if launched(e['name'])]
rest = [e for e in cur if not launched(e['name'])]
rest.sort(key=lambda e: pos.get(e['name'], 10**6))
cur = started + rest
q.write_text(json.dumps(cur, indent=1))
msg = f'queue: {len(cur)} on host, {added} added, {updated} updated in place'
if removed:
    _names = ', '.join(sorted(removed)[:4])
    _tail = ', ...' if len(removed) > 4 else ''
    msg += ', {} removed ({}{})'.format(len(removed), _names, _tail)
print(msg)
PYMERGE"
  fi
  # SUPERVISION. Nothing restarted the dispatcher, and today that cost an outage: it died
  # at 15:40Z on a KeyError four minutes after launching a wave, the treatment ran to
  # completion on the GPU with nobody left to harvest it, and the box sat idle until a
  # human noticed. There is no cron, no systemd unit, and the one restart script in the
  # tree (restart_sweep.sh) pkill -9's live trainers, so it is not usable as a supervisor.
  #
  # This restarts ONLY when no dispatcher is running. It never kills anything: a dispatcher
  # holding the lock is left alone, and any trainer already on a GPU keeps running -- the
  # new dispatcher adopts in-flight work and harvests it, which is exactly how both halves
  # of today's crashed wave were recovered.
  _alive=$(ssh -n "${SSHOPT[@]}" "$HOST" 'pgrep -f "dispatch.py [0-9]" | grep -cv "bash -c" || true' 2>/dev/null | tr -d "[:space:]")
  if [ "${_alive:-0}" = "0" ]; then
    echo "  DISPATCHER DOWN -- restarting (no trainer is touched; in-flight work is adopted)"
    # Deadline is refreshed each tick rather than fixed at first launch, so the campaign is
    # bounded by the tick loop stopping rather than running unattended forever.
    _deadline=$(python3 -c 'import time; print(time.time() + 6*3600)')
    ssh -n "${SSHOPT[@]}" "$HOST" "cd ~/$OPHIS_REMOTE_DIR/sweep && rm -f dispatcher.lock && nohup ~/$OPHIS_REMOTE_DIR/gpu6/.venv/bin/python -u dispatch.py $_deadline >> dispatch.out 2>&1 < /dev/null & sleep 3" \
      && echo "  dispatcher restarted" \
      || echo "  WARN: dispatcher restart FAILED -- GPUs will not be claimed"
  fi
  python3 tools/council.py status
}

if [ "${1:-}" = "--loop" ]; then
  while true; do tick; sleep "$INTERVAL"; done
else
  tick
fi
