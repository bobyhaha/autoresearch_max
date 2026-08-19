"""Make the existing test scripts collectable by pytest, so CI can run them.

The suite is six executable scripts with assertions at module level. They pass when run
directly and `pytest` collects ZERO of them -- it looks for test_* functions, finds none,
and exits 5. So the project had a real test suite and no way to run it automatically,
which is why the leaked-fixture and stale-contract failures found today survived: nothing
ran the tests between changes except a human remembering to.

This wraps each script as a subprocess and asserts a clean exit. That is deliberately a
THIN wrapper and not a rewrite:

  - Running each script in its own process preserves the isolation they already rely on.
    Several mutate module globals and monkeypatch functions on imported modules; importing
    them all into one pytest process would let those escape into each other.
  - The scripts remain runnable on their own, which is how they are used during a campaign.

WHAT THIS DOES NOT FIX, and it should be fixed. The scripts write fixtures into the LIVE
tree -- `lit/sources/`, `rounds/`, `runs/sweep/queue.json` -- and restore them at the end.
When an assertion fails the script exits immediately and the restore never runs, so a
failure leaves the corpus dirty. That has already bitten twice today: a leaked source
snapshot made a later run fail on an unrelated assertion, and a leaked round fixture would
have been picked up as the latest round and queued onto GPUs. Both now clear their
fixtures up front, which is a guard rather than a cure. The real fix is `tmp_path` and
dependency-injected roots, and it is a genuine refactor rather than a wrapper.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

TESTS = pathlib.Path(__file__).resolve().parent
SCRIPTS = sorted(p.name for p in TESTS.glob("test_*.py") if p.name != "test_suite.py")


@pytest.mark.parametrize("script", SCRIPTS)
def test_script(script):
    """Run one campaign test script in a fresh process; a non-zero exit is a failure."""
    r = subprocess.run([sys.executable, str(TESTS / script)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        # The scripts print a PASS/FAIL line per assertion, so the tail is the useful part.
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
        pytest.fail(f"{script} exited {r.returncode}\n{tail}")


def test_every_script_is_collected():
    """Guard against the failure this file exists to fix: a suite nobody runs.

    If a new test script is added and this parametrisation stops seeing it -- a rename, a
    move, a glob that no longer matches -- the suite silently shrinks and CI keeps passing
    on less than it did. Assert the count against what is on disk.
    """
    on_disk = {p.name for p in TESTS.glob("test_*.py")} - {"test_suite.py"}
    assert set(SCRIPTS) == on_disk, f"collected {set(SCRIPTS)}, on disk {on_disk}"
    assert len(SCRIPTS) >= 6, f"expected at least 6 campaign test scripts, found {SCRIPTS}"
