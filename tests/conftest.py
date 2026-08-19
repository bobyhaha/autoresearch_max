"""Keep pytest from IMPORTING the campaign scripts, which would run them in-process.

The six campaign test scripts put their assertions at module level, so importing one runs
it. pytest imports every `test_*.py` it collects -- so collection alone executed all six
inside the pytest process, and several of them monkeypatch module state to build fixtures:
`test_chain_of_evidence.py` replaces `claims.claims`, `claims.mechanisms` and
`claims.hypotheses` with stubs and never restores them, because as a standalone script it
has no reason to.

The effect was that any test collected afterwards saw a poisoned `claims` module. Three
guard tests failed in the full suite and passed in isolation, which is the signature of
exactly this. It also meant every script ran TWICE -- once implicitly at collection, once
properly through the subprocess wrapper in test_suite.py -- with the implicit run mutating
the live tree.

So the scripts are excluded from collection here. test_suite.py still runs each of them in
its own process, which is what makes their monkeypatching harmless, and they remain
directly runnable during a campaign. Only the accidental in-process execution goes away.
"""
from __future__ import annotations

import pathlib

# The subprocess-wrapped campaign scripts. Anything NOT in this list is a real pytest
# module and is collected normally.
_SCRIPTS = [
    "test_chain_of_evidence.py",
    "test_lessons.py",
    "test_pipeline.py",
    "test_policy_defects.py",
    "test_rotation.py",
    "test_wave_launch.py",
]

collect_ignore = list(_SCRIPTS)


def pytest_collection_modifyitems(config, items):
    """Fail loudly if a script slips back into collection.

    Silently collecting one again would reintroduce the poisoning, and the symptom --
    unrelated tests failing only in the full suite -- is hard to trace back here.
    """
    here = pathlib.Path(__file__).parent
    for item in items:
        name = pathlib.Path(str(item.fspath)).name
        assert name not in _SCRIPTS, (
            f"{name} was collected by pytest; it has module-level assertions and "
            f"monkeypatches shared modules, so importing it poisons the session. It must "
            f"run only via the subprocess wrapper in test_suite.py.")
    # And the wrapper must still be covering every script that exists.
    on_disk = {p.name for p in here.glob("test_*.py")}
    missed = on_disk - set(_SCRIPTS) - {"test_suite.py", "test_guards_fire.py"}
    assert not missed, (
        f"new test file(s) {sorted(missed)} are neither listed as subprocess scripts nor "
        f"known pytest modules; decide which they are rather than letting collection guess")
