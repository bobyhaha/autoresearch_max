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
                     "json": __import__("json"),
                     "collections": __import__("collections"),
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

print("\n5b. a wave already split does NOT run as a smaller wave")
# Observed on the host: W03a was queued as ONE 4-wide group; two members were claimed,
# and the next pass grouped only the two that remained -- so a 4-wide wave launched as
# two pairs. Harmless for controls, fatal for a treatment, whose pairing is the entire
# reason the wave exists. next_batch sizes a wave from the QUEUE, not from the
# unclaimed remainder, and refuses the remnant.
quad = [{"name": f"W_{i}", "wave_group": "w4split", "cfg": {}} for i in range(4)]
mod.runnable = lambda cutoff: (quad[2:], 0, 0)          # two members already claimed
mod.load_queue = lambda: quad                            # but four were queued
batch, waiting = mod.next_batch(0, 4)
ok(batch == [], f"the 2 surviving members of a 4-wide wave do not launch (got "
                f"{[b['name'] for b in batch]})")
ok(waiting and "split" in waiting, f"and it says why: {waiting}")

mod.runnable = lambda cutoff: (list(quad), 0, 0)          # nothing claimed yet
batch, _ = mod.next_batch(0, 4)
ok(len(batch) == 4, "an intact 4-wide wave still launches in full")
mod.load_queue = lambda: []                               # restore for later cases

print("\n5c. a wave whose treatment is FROZEN is held, not launched control-only")
# The gate freezes non-control entries when a council artifact goes stale. An earlier fix
# excluded frozen members from the wave-size denominator so the wave would not look split
# -- but that let the wave launch its CONTROL alone, which would put the control in a
# different wave from the treatment it is yoked to. That is the cross-wave drift the
# wave_group mechanism exists to prevent, so the whole wave must wait.
pair = [{"name": "F_treat", "wave_group": "wfrozen", "cfg": {"mlp": 9}, "created_at": 900},
        {"name": "F_ctrl",  "wave_group": "wfrozen", "cfg": {},          "created_at": 900}]
mod.load_queue = lambda: pair
mod._is_control = lambda cfg: not cfg          # the control has an empty cfg here
mod.runnable = lambda cutoff: ([pair[1]], 1, 0)   # treatment frozen, control runnable
batch, waiting = mod.next_batch(500, 4)            # cutoff 500 < created_at 900
ok(batch == [], f"the lone runnable control does not launch (got {[b['name'] for b in batch]})")
mod.runnable = lambda cutoff: (list(pair), 0, 0)   # freeze lifted
batch, _ = mod.next_batch(1000, 4)
ok(len(batch) == 2, "once the freeze lifts the whole wave goes out together")
mod.load_queue = lambda: []

print("\n5d. a PERMANENTLY split wave is tombstoned, not refused forever")
# zloss01_A/B ran their two controls and never their two treatments. Because a wave is
# sized from the queue, the pair could never be reformed -- so every poll re-evaluated
# the same dead entries and logged the same refusal, indefinitely, while the queue went
# on presenting them as pending work. Refusing is right; refusing FOREVER is a leak.
# The stranded members are written as INVALID results so the loss stays in the
# accounting, rather than deleted, which would make it vanish.
import json as _json, tempfile as _tf
with _tf.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    (root / "results").mkdir(); (root / "claims").mkdir()
    mod.ROOT = root
    dead = [{"name": "Z_ctrl0", "wave_group": "zdead", "cfg": {}},
            {"name": "Z_ctrl1", "wave_group": "zdead", "cfg": {}},
            {"name": "Z_treat0", "wave_group": "zdead", "cfg": {"zloss": 0.1}},
            {"name": "Z_treat1", "wave_group": "zdead", "cfg": {"zloss": 0.1}}]
    mod.load_queue = lambda: dead
    (root / "results" / "Z_ctrl0.json").write_text("{}")   # the two controls finished
    (root / "results" / "Z_ctrl1.json").write_text("{}")

    # In flight is NOT stranded: a claimed member means the wave is still forming.
    (root / "claims" / "Z_treat0").mkdir()
    ok(mod.tombstone_split_wave("zdead") is False,
       "a wave with a CLAIMED member is left alone -- it may still complete")
    ok(not (root / "results" / "Z_treat1.json").exists(),
       "and nothing was tombstoned while that claim was live")

    (root / "claims" / "Z_treat0").rmdir()                 # the claim went away unrun
    ok(mod.tombstone_split_wave("zdead") is True, "the dead wave is retired")
    for n in ("Z_treat0", "Z_treat1"):
        rec = _json.loads((root / "results" / f"{n}.json").read_text())
        ok(rec["ok"] is False and "stranded" in rec["invalid_reason"],
           f"{n} recorded as INVALID, so analyze.py counts the loss")
        ok("val_bpb" not in (rec.get("metrics") or {}),
           f"{n} carries no val_bpb -- none exists and none may be inferred")
    ok(mod.tombstone_split_wave("zdead") is False,
       "and it does not fire twice: the entries now have results")

    # An intact wave, none of whose members have finished, is never touched.
    live = [{"name": "L0", "wave_group": "zlive", "cfg": {}},
            {"name": "L1", "wave_group": "zlive", "cfg": {}}]
    mod.load_queue = lambda: live
    ok(mod.tombstone_split_wave("zlive") is False,
       "a wave with no finished member is not a split wave")
mod.load_queue = lambda: []

print("\n6. the cap really is 4")
ok("MAX_GPUS = 4" in src, "MAX_GPUS = 4 in host/dispatch.py")
ok("WAIT_LOG_EVERY_S = 30 * 60" in src, "waiting is reported every 30 minutes")

print("\nALL WAVE-LAUNCH TESTS PASS")
