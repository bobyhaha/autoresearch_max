#!/usr/bin/env python3
"""Where the remote trainer lives. Read from the environment, never committed.

The SSH target is infrastructure, not research: an address, a login and a port together
are enough to start knocking on someone's box, and this repository is shareable. So the
host lives in the environment or in an ignored `.ophis_host` file, and the code carries
only the shape.

Set it once per machine:

    cp .env.example .ophis_host && $EDITOR .ophis_host

or export directly:

    export OPHIS_SSH_USER=... OPHIS_SSH_HOST=... OPHIS_SSH_PORT=...
    export OPHIS_SSH_KEY=~/.ssh/id_ed25519 OPHIS_REMOTE_DIR=ophis_v3
"""
from __future__ import annotations

import os
import pathlib
import shlex

REPO = pathlib.Path(__file__).resolve().parent.parent
CONF = REPO / ".ophis_host"

DEFAULTS = {"OPHIS_SSH_USER": "", "OPHIS_SSH_HOST": "", "OPHIS_SSH_PORT": "22",
            "OPHIS_SSH_KEY": "~/.ssh/id_ed25519", "OPHIS_REMOTE_DIR": "ophis_v3"}


def _load() -> dict:
    cfg = dict(DEFAULTS)
    if CONF.exists():                       # file first, environment wins over it
        for line in CONF.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip().lstrip("export ").strip(), v.strip().strip('"').strip("'")
            if k in DEFAULTS:
                cfg[k] = v
    for k in DEFAULTS:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


CFG = _load()
USER, HOST = CFG["OPHIS_SSH_USER"], CFG["OPHIS_SSH_HOST"]
PORT, KEY = CFG["OPHIS_SSH_PORT"], os.path.expanduser(CFG["OPHIS_SSH_KEY"])
REMOTE_DIR = CFG["OPHIS_REMOTE_DIR"]
TARGET = f"{USER}@{HOST}" if USER and HOST else ""


def configured() -> bool:
    return bool(USER and HOST)


def why_not() -> str:
    return ("remote host not configured: set OPHIS_SSH_USER and OPHIS_SSH_HOST in the "
            "environment or in .ophis_host (see .env.example). The address is kept out "
            "of the repository on purpose.")


def ssh_argv(extra: list[str] | None = None) -> list[str]:
    """Argument vector for ssh. Raises if unconfigured rather than guessing a host."""
    if not configured():
        raise RuntimeError(why_not())
    return ["ssh", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-p", str(PORT), TARGET] + (extra or [])


def scp_argv(src: str, dst: str) -> list[str]:
    if not configured():
        raise RuntimeError(why_not())
    return ["scp", "-q", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-P", str(PORT), src, dst]


def remote(path: str) -> str:
    """A path on the remote box, e.g. remote('sweep/results') -> user@host:~/dir/..."""
    return f"{TARGET}:~/{REMOTE_DIR}/{path}"


def as_shell_env() -> str:
    """Emit the config for tools/tick.sh to eval."""
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in
                     (("OPHIS_TARGET", TARGET), ("OPHIS_PORT", str(PORT)),
                      ("OPHIS_KEY", KEY), ("OPHIS_REMOTE_DIR", REMOTE_DIR)))


if __name__ == "__main__":
    import sys
    if "--shell" in sys.argv:
        print(as_shell_env())
    elif configured():
        print(f"remote configured: {USER}@<host>:{PORT} dir=~/{REMOTE_DIR} key={KEY}")
    else:
        print(why_not())
        sys.exit(1)
