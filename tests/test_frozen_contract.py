#!/usr/bin/env python3
"""The benchmark contract is verified, not remembered.

CLAUDE.md's first non-negotiable is that `prepare.py` is FROZEN and byte-identical to
Karpathy 228791f -- its tokenizer, loader, packing, evaluator and BPB byte accounting ARE
the benchmark. Every val_bpb this campaign has ever reported assumes it.

Until now nothing checked it. `tools/preflight.py` compares the active code against
`baseline/provenance.json`, but that file is editable in the same commit as the thing it
describes, so the check is self-referential: edit `prepare.py` and its recorded digest
together and every automated guard in the tree still passes. `karpathy_pristine/` sits in
the repo for exactly this comparison and no tool had ever read it -- a grep for
`karpathy_pristine` across tools/, tests/ and host/ found one hit, and it was a regex for
formatting citations.

An honour-system freeze on the metric is the one place this campaign cannot afford one.
A silent edit here would not produce a wrong experiment; it would produce a wrong
BENCHMARK, and every number on record would be incomparable to upstream and to itself.
"""
import hashlib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


print("frozen benchmark contract")

pristine = REPO / "karpathy_pristine" / "prepare.py"
active = REPO / "baseline" / "prepare.py"
check("karpathy_pristine/prepare.py exists", pristine.exists())
check("baseline/prepare.py exists", active.exists())

if pristine.exists() and active.exists():
    a, b = sha(active), sha(pristine)
    check("baseline/prepare.py is byte-identical to the pristine upstream copy", a == b,
          f"{a[:16]} vs {b[:16]}")

# The generated trainer must never carry an edit to the evaluator or the budget. Those
# live in prepare.py, but a variant that redefined them locally would shadow the frozen
# ones -- so the generator's output is checked too, not just the file on disk.
sys.path.insert(0, str(REPO / "tools"))
try:
    import direction
    import make_variant
    src = make_variant.build(dict(direction.PLATFORM))
    for forbidden in ("def evaluate_bpb", "def make_dataloader", "EVAL_TOKENS =",
                      "def get_token_bytes"):
        check(f"generated variant does not redefine {forbidden.strip()!r}",
              forbidden not in src)
    check("generated variant still imports the frozen module",
          "from prepare import" in src or "import prepare" in src)
except Exception as exc:                                    # noqa: BLE001
    check("generated control builds for the contract check", False, repr(exc))

print("\nALL FROZEN-CONTRACT TESTS PASS" if not FAILED
      else f"\n{len(FAILED)} FROZEN-CONTRACT TEST(S) FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
