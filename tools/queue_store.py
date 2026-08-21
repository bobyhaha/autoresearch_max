#!/usr/bin/env python3
"""Crash-safe, process-safe JSON queue updates.

Every queue producer used to perform an unlocked read/modify/write.  Two producers could
therefore lose each other's entries, and a dispatcher reading during ``write_text`` could
observe truncated JSON and interpret it as an empty queue.  Keep the lock for the whole
transaction and publish with an atomic rename.
"""
from __future__ import annotations

import fcntl
import json
import os
import pathlib
import tempfile
from collections.abc import Callable


def atomic_write_json(path: pathlib.Path, value) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=1)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def update_queue(path: pathlib.Path, transform: Callable[[list[dict]], list[dict]]) -> list[dict]:
    """Apply ``transform`` under an exclusive lock and atomically publish its result.

    Malformed JSON is deliberately an error.  Treating corruption as ``[]`` converts a
    visible storage fault into silent deletion of every queued experiment.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name("queue.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads(path.read_text()) if path.exists() else []
        if not isinstance(current, list):
            raise ValueError(f"{path} must contain a JSON list")
        updated = transform(list(current))
        if not isinstance(updated, list):
            raise TypeError("queue transform must return a list")
        atomic_write_json(path, updated)
        return updated
