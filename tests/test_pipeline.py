#!/usr/bin/env python3
"""End-to-end: a fetched paper becomes a claim, opens a direction, and a council round
becomes executable work. Every seam between the tools is exercised on real files."""
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import claims, council, direction, lit, make_variant  # noqa: E402

ok = lambda c, m: print(f"  {'PASS' if c else 'FAIL'}  {m}") or (c or sys.exit(f"FAILED: {m}"))
TMP = REPO / "lit" / "_test"
for p in (REPO / "lit" / "sources" / "arxiv_2599.99999_fulltext.txt",):
    p.parent.mkdir(parents=True, exist_ok=True)

print("1. a claim without a full-text snapshot is REJECTED")
c = {"belief_key": "k1", "statement": "s", "source_id": "2599.99999",
     "locator": "table 1", "stance": "supports", "families": ["capacity"],
     "internal_validity": 3, "transfer": 3, "effect": "-1%", "comparator": "AdamW"}
ok(any("no full-text snapshot" in p for p in claims.validate_claim(c)),
   "abstract-only source refused")

print("\n2. with the snapshot present it validates")
snap = lit.SOURCES / "arxiv_2599.99999_fulltext.txt"
snap.write_text("x" * 3000)
ok(claims.validate_claim(c) == [], f"valid: {claims.validate_claim(c)}")

print("\n3. validity and transfer are range-checked and never averaged")
ok(any("0-4" in p for p in claims.validate_claim({**c, "transfer": 9})), "transfer>4 refused")
ok(any("unknown family" in p for p in claims.validate_claim({**c, "families": ["nope"]})),
   "unknown family refused")

print("\n4. a mechanism cannot cite an unregistered claim")
m = {"name": "m1", "mediator": "x", "families": ["capacity"], "claim_keys": ["ghost"],
     "falsifier": "f", "activation_observable": "o", "competing_explanation": "e"}
ok(any("unregistered claim" in p for p in claims.validate_mech(m)), "dangling claim_key refused")

print("\n5. council validation rejects a stub and accepts a real round")
bad = REPO / "rounds" / "_test_bad.md"
bad.parent.mkdir(exist_ok=True)
bad.write_text("## explorer (a)\nx\n## pragmatist (a)\nx\n## critic (a)\nx\n## synthesis (a)\nx\n")
probs = council.validate(bad, "round")
ok(any("word" in p for p in probs), "stub sections refused")
ok(any("not distinct" in p for p in probs), "duplicate agent ids refused")
ok(any("synthesizer is also a reviewer" in p for p in probs), "non-independent synth refused")
ok(any("queue block" in p for p in probs), "missing queue block refused")
bad.unlink()

filler = ("The GPU is idle for most of the charged window on this benchmark. " * 20)
good = REPO / "rounds" / "_test_good.md"
good.write_text(f"""## explorer (ag-1)
{filler}

## pragmatist (ag-2)
{filler}

## critic (ag-3)
{filler}

## synthesis (ag-4)
{filler}{filler}

```queue
[{{"name":"T_dim768","cfg":{{"dbs":128,"tbs":19,"depth":8,"dim":768,"mlp":4,"ve":2,"win":"SSSL","swdiv":2}},
  "rationale":"width never varied","falsifier":"steps drop >10%","expected":"lower bpb"}}]
```
""")
ok(council.validate(good, "round") == [], f"real round valid: {council.validate(good,'round')}")

print("\n6. the queue block parses into an executable, deterministic variant")
q = council.queue_entries(good.read_text())
ok(len(q) == 1 and q[0]["cfg"]["dim"] == 768, "queue entry parsed")
src1 = make_variant.build(q[0]["cfg"])
src2 = make_variant.build(q[0]["cfg"])
ok(make_variant.variant_id(src1) == make_variant.variant_id(src2), "variant id is stable")
ok("model_dim = 768" in src1, "the edit actually landed in the generated source")
good.unlink()

print("\n7. a variant whose edit target is missing RAISES rather than shipping a no-op")
try:
    make_variant.sub(src1, "target_that_does_not_exist", "x", "ema activation print")
    ok(False, "absent target must raise")
except make_variant.VariantEditError:
    ok(True, "absent edit target raises VariantEditError")

print("\n8. dispatcher contract: the functions host/dispatch.py imports exist and behave")
st = direction.axis_state([])
ok(direction.blocked_reason({}, st) is None, "controls never blocked")
ok(direction.blocked_reason({"dim": 768}, st) is None, "unexplored axis never blocked")
ok(direction.is_platform({}) and direction.label({"dim": 768}) == "knob:dim", "label/is_platform")

print("\n9. cleanup")
snap.unlink()
ok(not snap.exists(), "test snapshot removed")

print("\nALL PIPELINE TESTS PASS")
