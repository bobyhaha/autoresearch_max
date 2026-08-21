#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${DASHBOARD_PORT:-8787}"
HOST_URL="http://127.0.0.1:${PORT}/training_dashboard/"

cd "$ROOT_DIR"

list_project_dashboard_pids() {
  ps -ax -o pid= -o command= 2>/dev/null | while IFS= read -r line; do
    case "$line" in
      *"dashboard_server.py"*)
        printf '%s\n' "$line" | awk '{print $1}'
        ;;
    esac
  done
}

stop_existing_dashboards() {
  local pids=()
  local pid
  while IFS= read -r pid; do
    [[ -n "$pid" && "$pid" != "$$" ]] && pids+=("$pid")
  done < <(list_project_dashboard_pids)

  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "Stopping existing training dashboard service(s): ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true

  for _ in {1..30}; do
    local still_running=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        still_running+=("$pid")
      fi
    done
    if [[ "${#still_running[@]}" -eq 0 ]]; then
      return 0
    fi
    sleep 0.1
  done

  echo "Existing dashboard service did not exit after SIGTERM; forcing shutdown."
  kill -9 "${pids[@]}" 2>/dev/null || true
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to start the training dashboard." >&2
  exit 2
fi

if [[ ! -f "$ROOT_DIR/dashboard_server.py" ]]; then
  echo "Missing dashboard_server.py in $ROOT_DIR" >&2
  exit 2
fi

stop_existing_dashboards

if python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.4)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
then
  echo "Port $PORT is already in use by another service." >&2
  echo "Set DASHBOARD_PORT to another port, for example:" >&2
  echo "  DASHBOARD_PORT=8788 ./start_training_dashboard.sh" >&2
  exit 2
fi

echo "Starting training dashboard:"
echo "$HOST_URL"
echo
echo "Press Ctrl-C to exit the dashboard server."
echo

exec python3 "$ROOT_DIR/dashboard_server.py"
