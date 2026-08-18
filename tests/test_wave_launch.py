#!/usr/bin/env python3
"""Wave launching: a yoked pair must go out CONCURRENTLY or not at all.

The dispatcher's grouping logic is exercised directly. Launching one GPU at a time as
capacity trickles in is what produced controls half an hour apart whose spread measured
host drift rather than resolution.
"""
import pathlib
import sys
import types

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
ok = lambda c, m: print(f"  {'PASS' if c else 'FAIL'}  {m}") or (c or sys.exit(f"FAILED: {m}"))

# Load dispatch.py's pure logic without its host-only module scope (flock, ROOT, nvidia-smi).
src = (REPO / "host" / "dispatch.py").read_text()
start = src.index("def runnable(")
end = src.index("def release(")
mod = types.ModuleType("dispatch_logic")
mod.__dict__.update({"os": __import__("os"), "time": __import__("time"),
                     "log": lambda m: None, "release": lambda it: None})
exec(compile(src[start:end], "dispatch_logic", "exec"), mod.__dict__)

ITEMS = [
    {"name": "T_treat", "wave_group": "w1", "cfg": {}},
    {"name": "T_ctrl",  "wave_group": "w1", "cfg": {}},
    {"name": "S_solo",  "cfg": {}},
]
mod.runnable = lambda cutoff: (list(ITEMS), 0, 0)

print("1. a 2-member wave does NOT launch when only 1 GPU is free")
batch, waiting = mod.next_batch(0, 1)
ok([b["name"] for b in batch] == ["S_solo"],
   f"the solo entry launches, the pair waits (got {[b['name'] for b in batch]})")

print("\n2. the pair launches TOGETHER as soon as 2 GPUs are free")
batch, waiting = mod.next_batch(0, 2)
ok(sorted(b["name"] for b in batch) == ["T_ctrl", "T_treat"],
   "both members of the wave go out on the same pass -> overlapping intervals")

print("\n3. with 4 free, the largest fitting wave is chosen first (fill the box)")
big = [{"name": f"Q{i}", "wave_group": "w4", "cfg": {}} for i in range(4)]
mod.runnable = lambda cutoff: (ITEMS + big, 0, 0)
batch, _ = mod.next_batch(0, 4)
ok(len(batch) == 4 and all(b["wave_group"] == "w4" for b in batch),
   "the 4-member wave is preferred over dribbling out singletons")

print("\n4. when nothing fits, it WAITS and says what it is waiting for")
mod.runnable = lambda cutoff: ([{"name": "P1", "wave_group": "w", "cfg": {}},
                                {"name": "P2", "wave_group": "w", "cfg": {}}], 0, 0)
batch, waiting = mod.next_batch(0, 1)
ok(batch == [] and waiting and "waiting for 2 free GPUs" in waiting,
   f"holds capacity for the wave: {waiting}")

print("\n5. a partial claim never splits a wave")
import tempfile
with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    (root / "claims").mkdir()
    (root / "claims" / "P2").mkdir()          # P2 already claimed by another pass
    mod.ROOT = root
    released = []
    mod.release = lambda it: released.append(it["name"])
    got = mod.claim_all([{"name": "P1"}, {"name": "P2"}])
    ok(got == [], "a wave with one member already claimed is refused entirely")
    ok(released == ["P1"], f"and the member it did take is released (got {released})")
    # a fully free wave claims cleanly
    released.clear()
    got = mod.claim_all([{"name": "A"}, {"name": "B"}])
    ok([g["name"] for g in got] == ["A", "B"], "a free wave claims every member")

print("\n6. the cap really is 4")
ok("MAX_GPUS = 4" in src, "MAX_GPUS = 4 in host/dispatch.py")
ok("WAIT_LOG_EVERY_S = 30 * 60" in src, "waiting is reported every 30 minutes")

print("\nALL WAVE-LAUNCH TESTS PASS")
