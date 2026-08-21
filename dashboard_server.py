#!/usr/bin/env python3
"""Local dashboard server with training control endpoints."""

from __future__ import annotations

import json
import os
import time
import base64
import hashlib
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".dashboard_state"
ACTIVE_RUN_PATH = STATE_DIR / "active_run.json"
DEFAULT_PORT = 8787


class DashboardHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        rel = parsed.path.lstrip("/")
        resolved = (ROOT / rel).resolve()
        root = ROOT.resolve()
        # Prevent path traversal: never serve anything outside ROOT.
        if resolved != root and root not in resolved.parents:
            return str(root)
        return str(resolved)

    def do_GET(self) -> None:
        if self.path == "/ws/logs":
            self.handle_log_websocket()
            return
        if self.path == "/api/status":
            self.write_json(status_payload())
            return
        super().do_GET()

    def do_POST(self) -> None:
        self.send_error(HTTPStatus.NOT_FOUND)

    def write_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_log_websocket(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if self.headers.get("Upgrade", "").lower() != "websocket" or not key:
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected WebSocket upgrade")
            return

        accept = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        last_payload = None
        while True:
            payload = json.dumps(log_stream_payload(), separators=(",", ":"))
            if payload != last_payload:
                try:
                    self.write_ws_text(payload)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                last_payload = payload
            time.sleep(1.0)

    def write_ws_text(self, text: str) -> None:
        data = text.encode("utf-8")
        header = bytearray([0x81])
        length = len(data)
        if length < 126:
            header.append(length)
        elif length < 65536:
            header.extend([126, (length >> 8) & 0xFF, length & 0xFF])
        else:
            header.extend(
                [
                    127,
                    (length >> 56) & 0xFF,
                    (length >> 48) & 0xFF,
                    (length >> 40) & 0xFF,
                    (length >> 32) & 0xFF,
                    (length >> 24) & 0xFF,
                    (length >> 16) & 0xFF,
                    (length >> 8) & 0xFF,
                    length & 0xFF,
                ]
            )
        self.wfile.write(header + data)


def status_payload() -> dict:
    state = {}
    if ACTIVE_RUN_PATH.exists():
        try:
            state = json.loads(ACTIVE_RUN_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    latest = latest_live_log()
    return {"running": is_log_active(latest), "active": state, "latest": latest}


def latest_live_log() -> dict:
    logs = sorted((ROOT / "remote_training_runs").glob("*/live.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if not logs:
        return {}
    path = logs[-1]
    run_id = path.parent.name
    rel_url = f"/remote_training_runs/{run_id}/live.log"
    return {
        "run_id": run_id,
        "path": str(path),
        "url": rel_url,
        "mtime": path.stat().st_mtime,
        "size": path.stat().st_size,
    }


def is_log_active(log: dict) -> bool:
    if not log:
        return False
    path = Path(log["path"])
    try:
        tail = path.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        return False
    if "training_status=" in tail:
        return False
    return time.time() - float(log.get("mtime", 0)) < 900


def log_stream_payload() -> dict:
    log = latest_live_log()
    text = ""
    if log:
        try:
            text = Path(log["path"]).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            text = f"Could not read log: {exc}"
    payload = status_payload()
    payload.update(
        {
            "type": "log",
            "source_url": log.get("url", ""),
            "run_id": log.get("run_id", ""),
            "log_text": text,
            "running": payload.get("running", False),
        }
    )
    return payload


def main() -> None:
    port = int(os.environ.get("DASHBOARD_PORT", DEFAULT_PORT))
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    print(f"Dashboard server: http://127.0.0.1:{port}/training_dashboard/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
