#!/bin/bash
# ssh_retry.sh 'remote command'
# The campaign box drops connections; a background waiter already died at exit
# 255 mid-run and would have left the loop blind if the tranche had not happened
# to finish. Retry with backoff instead of failing the turn.
CMD="$1"; MAX=${2:-8}
for i in $(seq 1 "$MAX"); do
  out=$(ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ConnectTimeout=25 \
            -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
            -p 50002 zhubaiyu@223.167.85.180 "$CMD" 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then printf '%s\n' "$out"; exit 0; fi
  echo "[ssh_retry] attempt $i/$MAX failed rc=$rc; retrying in $((i*10))s" >&2
  sleep $((i*10))
done
echo "[ssh_retry] EXHAUSTED $MAX attempts -- box unreachable" >&2
exit 1
