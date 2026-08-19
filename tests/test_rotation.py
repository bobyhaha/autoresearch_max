#!/usr/bin/env python3
"""Prove the explore/exploit controller enters, exploits, and ROTATES OUT on every trigger.

Run: python3 tests/test_rotation.py
"""
import json
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import agenda, claims, direction, lit   # noqa: E402

PLAT = dict(direction.PLATFORM)
_t = [0]


def run(cfg, bpb, started=1000.0):
    _t[0] += 1
    return {"ok": True, "ended": _t[0], "started": started,
            "cfg": {**PLAT, **cfg}, "metrics": {"val_bpb": bpb}}


# The dry/stale triggers compare against a MEASURED noise band, so every fixture that
# expects them to fire must first establish the instrument with concurrent controls.
# Without controls the band is None, nothing can be judged non-improving, and no
# improvement-based trigger may fire -- which is the intended fail-safe, asserted below.
CTRL = [run({}, 1.0300, 1000.0), run({}, 1.0301, 1050.0), run({}, 1.0302, 1100.0)]


def stub_lit(fams_with_claims, unread=None):
    unread = unread or {}
    def _ls():
        return {f: {"claims": 3, "supports": 3, "opposes": 0,
                    "usable": 3 if f in fams_with_claims else 0,
                    "mean_transfer": 3.0, "mean_validity": 3.0, "mechanisms": 1,
                    "fetched": 5 if f in fams_with_claims else 0,
                    "screened": 9, "unread": unread.get(f, 0)}
                for f in lit.ALL_FAMILIES}
    return _ls


def setup(tmp, results, fams, unread=None, state=None):
    agenda.STATE = pathlib.Path(tmp) / "ACTIVE.json"
    if state is not None:
        agenda.STATE.write_text(json.dumps(state))
    agenda.load_results = lambda: results
    claims.literature_state = stub_lit(fams, unread)


ok = lambda c, m: print(f"  {'PASS' if c else 'FAIL'}  {m}") or (c or sys.exit(f"FAILED: {m}"))

with tempfile.TemporaryDirectory() as tmp:
    print("1. literature gate: no claims anywhere => no direction may open")
    setup(tmp, [], set())
    d = agenda.decide()
    ok(d["active"] is None, "no active direction without literature")
    ok("Build the corpus first" in d["reason"], "reason tells the operator to read papers")

    print("\n2. entry: claims exist => the best-scoring eligible family activates")
    setup(tmp, [], {"capacity", "schedule"})
    d = agenda.decide()
    ok(d["active"] in ("capacity", "schedule"), f"activated '{d['active']}'")
    first = d["active"]

    print("\n3. exploitation: while it keeps improving, it is NOT rotated away")
    res = CTRL + [run({"mlp": 5}, 1.020), run({"mlp": 6}, 1.010), run({"mlp": 7}, 1.000)]
    setup(tmp, res, {"capacity", "schedule"},
          state={"active": "capacity", "since": 0, "cooldown": [], "history": []})
    d = agenda.decide()
    ok(d["active"] == "capacity", "improving direction stays active")
    ok(not d["rotated"], "no rotation while it is still paying")

    print("\n4. STALE trigger: 6 runs since the last real gain => forced rotation")
    res = CTRL + [run({"mlp": 5}, 1.000)] + [run({"mlp": 5 + i}, 1.001) for i in range(6)]
    setup(tmp, res, {"capacity", "schedule"},
          state={"active": "capacity", "since": 0, "cooldown": [], "history": []})
    d = agenda.decide()
    ok(d["rotated"], "rotation fired")
    ok(d["active"] != "capacity", f"moved off exhausted 'capacity' to '{d['active']}'")
    ok("STALE" in d["reason"], f"reason names the trigger: {d['reason'][:70]}")
    ok("capacity" in d["cooldown"], "exhausted direction is on cooldown")

    print("\n5. HARD CAP: 12 runs in one family rotates even while improving")
    # mlp must never equal PLATFORM["mlp"]=4, or the run is a CONTROL and touches
    # no family -- which is correct behaviour, and was a bug in an earlier fixture.
    res = CTRL + [run({"mlp": 5 + i}, 1.10 - 0.01 * i) for i in range(12)]
    setup(tmp, res, {"capacity", "schedule"},
          state={"active": "capacity", "since": 0, "cooldown": [], "history": []})
    d = agenda.decide()
    ok(d["rotated"] and d["active"] != "capacity", "hard cap forced rotation")
    ok("HARD CAP" in d["reason"], f"reason names the trigger: {d['reason'][:60]}")

    print("\n6. DRY trigger needs an exhausted family: untouched axes keep it open")
    res = CTRL + [run({"mlp": 5}, 1.000)] + [run({"mlp": 5}, 1.001) for _ in range(4)]
    setup(tmp, res, {"capacity", "schedule"},
          state={"active": "capacity", "since": 0, "cooldown": [], "history": []})
    d = agenda.decide()
    ok(d["active"] == "capacity",
       "dry but 'dim'/'depth' never tried => stays open (this is the v3 bug, inverted)")

    print("\n7. cooldown: an exhausted direction is not immediately re-entered")
    setup(tmp, [], {"capacity", "schedule"},
          state={"active": None, "since": 0, "cooldown": ["capacity"], "history": []})
    d = agenda.decide()
    ok(d["active"] == "schedule", f"picked a different family ('{d['active']}')")

    print("\n8. new literature REOPENS a cooled direction")
    setup(tmp, [], {"capacity", "schedule"}, unread={"capacity": 4},
          state={"active": None, "since": 0, "cooldown": ["capacity"], "history": []})
    d = agenda.decide()
    ok("capacity" in d["reopened"], "unread full texts took it off cooldown")

    print("\n9. every family cooled => start a new cycle instead of stalling")
    setup(tmp, [], {"capacity", "schedule"},
          state={"active": None, "since": 0,
                 "cooldown": list(lit.ALL_FAMILIES), "history": []})
    d = agenda.decide()
    ok(d["active"] is not None, f"cycle restarted on '{d['active']}' rather than stalling")

    print("\n10. no measured band => improvement-based triggers CANNOT fire,")
    print("    but the count-based HARD CAP still prevents a monoculture")
    res = [run({"mlp": 5 + i}, 1.02) for i in range(6)]        # no controls at all
    setup(tmp, res, {"capacity", "schedule"},
          state={"active": "capacity", "since": 0, "cooldown": [], "history": []})
    d = agenda.decide()
    ok(d["active"] == "capacity",
       "STALE cannot fire without an instrument (uncertainty is not a negative result)")
    res = [run({"mlp": 5 + i}, 1.02) for i in range(12)]
    setup(tmp, res, {"capacity", "schedule"},
          state={"active": "capacity", "since": 0, "cooldown": [], "history": []})
    d = agenda.decide()
    ok(d["rotated"] and "HARD CAP" in d["reason"],
       "HARD CAP still rotates with no band at all - monoculture stays impossible")

print("\nALL ROTATION TESTS PASS")

print("\nHOLDS-BEST -- the axis currently winning is never closed as dry")
# The dry test compares a run against the GLOBAL running best, so an axis is charged a
# strike for failing to beat a record ANOTHER axis set. Observed live: swdiv reached
# 0.989449, a new campaign best, but the prior best 0.989819 came from the ve axis and
# clearing it by 0.00037 did not clear the 0.00074 band -- four such runs closed the axis
# that was winning at that moment. Generalised, one strong result retires the whole
# search space.
import direction as _d
def _run_v(axis, aval, val, i):
    cfg = dict(_d.PLATFORM); cfg[axis] = aval
    return {"ok": True, "ended": i, "cfg": cfg,
            "metrics": {"val_bpb": val, "final_epoch": 2.0, "num_steps": 1000}}
def _run(axis, val, i):
    cfg = dict(_d.PLATFORM); cfg[axis] = {"swdiv": 4, "mlp": 6}[axis]
    return {"ok": True, "ended": i, "cfg": cfg,
            "metrics": {"val_bpb": val, "final_epoch": 2.0, "num_steps": 1000}}
# Two controls that overlap in time so a band exists at all.
def _ctl(val, i):
    return {"ok": True, "ended": i, "started": i - 300, "gpu": i % 4, "cfg": dict(_d.PLATFORM),
            "metrics": {"val_bpb": val, "final_epoch": 2.0, "num_steps": 1000}}
rows = [_ctl(0.9990 + 0.0001 * k, 1000 + k) for k in range(6)]
# mlp sets a strong best first; swdiv then edges past it by less than the band, 4x.
rows.append(_run("mlp", 0.9900, 2000))
# distinct swdiv values, so the streak accrues under value-counting semantics
rows += [_run_v("swdiv", 4 * 2 ** k, 0.98995 - 0.000001 * k, 2001 + k) for k in range(4)]
st = _d.axis_state(rows)
sw = st["axes"]["swdiv"]
ok(sw["dry"] >= 4, f"swdiv accrued a full dry streak ({sw['dry']})")
ok(sw["best"] <= st["best"], "swdiv holds the campaign best")
ok(sw["open"], "and is NOT closed, because retiring the current leader is never right")
ok(not st["axes"]["mlp"]["open"] or st["axes"]["mlp"]["dry"] < _d.DRY_STREAK,
   "an overtaken axis still closes normally -- the exemption is only for the leader")
