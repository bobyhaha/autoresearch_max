"""Shared SSH hardening for the gated launchers.

Every production hang in the launch path traced to an UNBOUNDED `subprocess.run(ssh
+ [...])` under sshd MaxStartups pressure: when the remote refused/stalled new
connections, the call never returned and the whole scheduler wedged. Two defenses:

1. ControlMaster multiplexing — the scheduler, all its runners, and any operator
   polling collapse onto ONE authenticated TCP connection per host. MaxStartups
   counts PRE-auth connections, so multiplexed sessions are immune to it.
2. ConnectTimeout + ServerAlive — a dead/slow remote fails fast instead of hanging.

`ssh_argv` injects these options at USE TIME only. The raw --ssh string flows
verbatim into the scheduler-capability handshake tuple, so it must NOT be mutated;
callers pass the raw prefix here purely to build subprocess argv.
"""

from __future__ import annotations

import shlex

# ControlPath uses %C (a hash of connection params) to stay under the 104-char
# UNIX-socket path limit on macOS. ControlPersist keeps the master alive between
# calls so short-lived polls reuse it.
SSH_HARDENING = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=~/.ssh/ophis-cm-%C",
    "-o", "ControlPersist=300",
]


def ssh_argv(prefix: str) -> list[str]:
    """Return argv for the ssh prefix with hardening options injected after argv[0].

    e.g. "ssh -p 50002 -o BatchMode=yes user@host" ->
         ["ssh", <hardening>, "-p", "50002", "-o", "BatchMode=yes", "user@host"]
    Duplicate -o options are harmless (last wins in OpenSSH for most, first for
    some; our additions don't conflict with a caller's typical BatchMode/-p).
    """
    parts = shlex.split(prefix)
    if not parts:
        return parts
    return [parts[0], *SSH_HARDENING, *parts[1:]]
