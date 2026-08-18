#!/usr/bin/env python3
"""The chain of evidence must CATCH breaks, not merely report INTACT on an empty repo.

Each check is driven with a deliberately broken record and must object.
ScientistOne's four integrity checks + AutoResearchClaw's numeric registry.
"""
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import claims as C, coe, lit  # noqa: E402

ok = lambda c, m: print(f"  {'PASS' if c else 'FAIL'}  {m}") or (c or sys.exit(f"FAILED: {m}"))
SRC = lit.SOURCES / "arxiv_2599.00001_fulltext.txt"


def setup(claims=(), mechs=(), hyps=(), results=(), index=None):
    C.claims = lambda: list(claims)
    C.mechanisms = lambda: list(mechs)
    C.hypotheses = lambda: list(hyps)
    coe.C = C
    coe._results = lambda: list(results)
    lit.load_index = lambda: (index or {})


print("E1 SOURCE -- an abstract may never back a claim")
setup(claims=[{"belief_key": "k", "source_id": "2599.00001", "locator": "table 1"}])
ok(any("no snapshot on disk" in p for p in coe.e1_source()), "missing snapshot caught")

SRC.parent.mkdir(parents=True, exist_ok=True)
SRC.write_text("x" * 3000)
import hashlib
good = hashlib.sha256(SRC.read_bytes()).hexdigest()
setup(claims=[{"belief_key": "k", "source_id": "2599.00001", "locator": "t1"}],
      index={"2599.00001": {"status": "screened"}})
ok(any("not marked fetched" in p for p in coe.e1_source()), "triage-only source caught")
setup(claims=[{"belief_key": "k", "source_id": "2599.00001", "locator": "t1"}],
      index={"2599.00001": {"status": "fetched", "sha256": "deadbeef"}})
ok(any("digest" in p for p in coe.e1_source()), "source changed under the claim caught")
setup(claims=[{"belief_key": "k", "source_id": "2599.00001", "locator": "t1"}],
      index={"2599.00001": {"status": "fetched", "sha256": good}})
ok(coe.e1_source() == [], "an intact source passes")

print("\nE2 LINK -- no dangling edges")
setup(claims=[{"belief_key": "k", "source_id": "2599.00001", "locator": "t1"}],
      mechs=[{"name": "m", "claim_keys": ["ghost"]}])
ok(any("unregistered claim 'ghost'" in p for p in coe.e2_link()), "dangling claim caught")
setup(mechs=[{"name": "m", "claim_keys": []}])
ok(any("cites no claims" in p for p in coe.e2_link()), "unsupported mechanism caught")
setup(claims=[{"belief_key": "k"}], mechs=[{"name": "m", "claim_keys": ["k"]}],
      hyps=[{"id": "h", "claim_keys": ["k"], "mechanism_names": ["nope"]}])
ok(any("unregistered mechanism 'nope'" in p for p in coe.e2_link()), "dangling mech caught")

print("\nE3 ACTIVATION -- a run that never engaged is INCONCLUSIVE, not a negative")
H = {"id": "h1", "activation": {"diagnostic": "ema_updates", "rule": {"op": "gt", "value": 0}}}
setup(hyps=[H], results=[{"name": "R1", "hypothesis_id": "h1",
                          "metrics": {"val_bpb": 1.1}}])
probs = coe.e3_activation()
ok(any("was NOT" in p and "INCONCLUSIVE" in p for p in probs),
   "missing diagnostic -> inconclusive, not evidence against the mechanism")
setup(hyps=[H], results=[{"name": "R1", "hypothesis_id": "h1",
                          "metrics": {"val_bpb": 1.1, "ema_updates": 0}}])
ok(any("fails the declared activation rule" in p for p in coe.e3_activation()),
   "diagnostic present but rule fails -> inconclusive")
setup(hyps=[H], results=[{"name": "R1", "hypothesis_id": "h1",
                          "metrics": {"val_bpb": 1.1, "ema_updates": 42}}])
ok(coe.e3_activation() == [], "an activated run passes")

print("\nE4 METHOD-CODE -- never spend GPU time on a variant identical to the control")
import direction, make_variant
qdir = REPO / "runs" / "sweep"
ctl_cfg = dict(direction.PLATFORM)
ctl_src = make_variant.build(ctl_cfg)
vid = make_variant.variant_id(ctl_src)
(qdir / "variants").mkdir(parents=True, exist_ok=True)
(qdir / "variants" / vid).write_text(ctl_src)
saved = (qdir / "queue.json").read_text()
(qdir / "queue.json").write_text(json.dumps([
    {"name": "CTL", "cfg": ctl_cfg, "variant": vid},
    {"name": "FAKE_TREATMENT", "cfg": {**ctl_cfg, "mlp": 9}, "variant": vid},
]))
setup()
probs = coe.e4_method_code()
ok(any("BYTE-IDENTICAL" in p for p in probs), "identical-ablation defect caught")
(qdir / "queue.json").write_text(json.dumps([{"name": "X", "cfg": ctl_cfg,
                                              "variant": "nonexistent.py"}]))
ok(any("missing variant" in p for p in coe.e4_method_code()), "missing variant caught")
(qdir / "queue.json").write_text(saved)

print("\nE5 NUMERIC -- a document may not cite a number the registry does not contain")
setup(results=[{"name": "C01", "metrics": {"val_bpb": 1.023456, "num_steps": 640.0}}])
reg = coe.registry()
ok(1.023456 in reg, f"registry built from results only ({len(reg)} values)")
doc = REPO / "papers" / "_test_paper.md"
doc.parent.mkdir(exist_ok=True)
doc.write_text("We measured val_bpb 1.023456, a gain over the 0.987654 baseline.\n")
probs = coe.e5_numeric()
ok(any("0.987654" in p for p in probs), "fabricated number caught")
ok(not any("1.023456" in p for p in probs), "genuine registry number accepted")
doc.write_text("We measured val_bpb 1.023456 at 640 steps.\n")
ok(coe.e5_numeric() == [], "a fully grounded document passes")
doc.unlink()

SRC.unlink()
print("\nALL CHAIN-OF-EVIDENCE TESTS PASS")
