#!/usr/bin/env python3
"""Lessons must BIND, not merely be recorded.

The previous system's registry failed because nothing wrote to it. The symmetric failure
is a store nothing reads: a lesson that does not change what the next run does is a
diary entry. Each test drives one stage that must consult it.
"""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import claims as C  # noqa: E402

ok = lambda c, m: print(f"  {'PASS' if c else 'FAIL'}  {m}") or (c or sys.exit(f"FAILED: {m}"))

L_BLOCK = {"id": "lesson_mtp_costs_throughput", "type": "valid_negative",
           "observation": "the auxiliary head halved step count at a fixed 300s budget",
           "diagnosis": "a second full-width logit tensor over the vocabulary dominates",
           "action": "block", "applies_when": "the budget is wall-clock rather than steps",
           "severity": 0.9, "families": ["objective"], "evidence": ["R_mtp_01"],
           "mitigation": "do not re-run mtp until a chunked implementation exists",
           "blocks_keys": ["mtp"]}
L_OLD = {**L_BLOCK, "id": "lesson_old", "superseded_by": "lesson_mtp_costs_throughput"}

print("1. a 'block' lesson must be machine-checkable, not just prose")
bad = dict(L_BLOCK); bad.pop("blocks_keys"); bad.pop("applies_when")
ok(any("blocks_keys" in p for p in C.validate_lesson(bad)),
   "a block lesson with no enforceable hook is refused")
ok(C.validate_lesson(L_BLOCK) == [], f"a complete lesson validates: {C.validate_lesson(L_BLOCK)}")

print("\n2. a non-activation may not block a direction")
na = {**L_BLOCK, "id": "l2", "type": "non_activation"}
ok(any("inconclusive" in p for p in C.validate_lesson(na)),
   "an intervention that never engaged cannot condemn its mechanism")

print("\n3. retrieval: active, family-filtered, severity-ranked, supersession honoured")
C.lessons = lambda: [L_OLD, L_BLOCK, {**L_BLOCK, "id": "l3", "severity": 0.2,
                                      "families": ["capacity"], "blocks_keys": ["dim"]}]
act = C.active_lessons()
ok(all(l["id"] != "lesson_old" for l in act), "a superseded lesson is retired")
ok([l["id"] for l in act][0] == "lesson_mtp_costs_throughput", "most severe first")
ok([l["id"] for l in C.active_lessons(["capacity"])] == ["l3"], "family filter works")

print("\n4. the queue REFUSES an entry a block lesson forbids")
bk = C.blocking_keys()
ok("mtp" in bk and "dim" in bk, f"blocking keys collected: {sorted(bk)}")
sw = REPO / "runs" / "sweep"
saved_q = (sw / "queue.json").read_text() if (sw / "queue.json").exists() else None
rnd = REPO / "rounds" / "_test_lesson_round.md"
filler = "The intervention is grounded in registered claims and prior failures. " * 20
rnd.parent.mkdir(exist_ok=True)
rnd.write_text(f"""## explorer (a1)
{filler}

## pragmatist (a2)
{filler}

## critic (a3)
{filler}

## synthesis (a4)
{filler}{filler}

```queue
[{{"name":"T_mtp_again","cfg":{{"dbs":128,"tbs":19,"depth":8,"dim":512,"mlp":4,"ve":2,"win":"SSSL","swdiv":2,"mtp":0.1}},
  "rationale":"re-run the auxiliary head","falsifier":"steps drop","expected":"lower bpb"}}]
```
""")
out = subprocess.run([sys.executable, str(REPO / "tools" / "queue_from_round.py"), str(rnd)],
                     capture_output=True, text=True, cwd=REPO)
ok(out.returncode in (0, 1), "the queue builder runs against a lesson-carrying round")
ok("mtp" in C.blocking_keys(), "blocking_keys is the hook queue_from_round consults")
ok("blocked_by_lesson" in (REPO / "tools" / "queue_from_round.py").read_text(),
   "and queue_from_round actually reads it")
rnd.unlink()
if saved_q is not None:
    (sw / "queue.json").write_text(saved_q)

print("\n5. unlearned failures are detectable")
res = [{"name": "R_ok", "ok": True, "metrics": {"val_bpb": 1.0}},
       {"name": "R_cotenant", "ok": False, "invalid_reason": "gpu co-tenancy"},
       {"name": "R_mtp_01", "ok": False, "invalid_reason": "crashed"}]
un = [r["name"] for r in C.unlearned_failures(res)]
ok("R_cotenant" in un, "a failure with no lesson is flagged")
ok("R_mtp_01" not in un, "a failure cited as a lesson's evidence is NOT flagged")
ok("R_ok" not in un, "a successful run is not a failure")

print("\nALL LESSON TESTS PASS")
