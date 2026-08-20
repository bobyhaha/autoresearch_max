#!/usr/bin/env python3
"""Regression tests for defects found by an independent critic after the rebuild.

Each test names the failure it prevents. All three were real and all three would have
corrupted results rather than merely inconveniencing the operator.
"""
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import direction as d  # noqa: E402

ok = lambda c, m: print(f"  {'PASS' if c else 'FAIL'}  {m}") or (c or sys.exit(f"FAILED: {m}"))
P = dict(d.PLATFORM)
_t = [0]
def run(cfg, bpb, started=0.0):
    _t[0] += 1
    return {"ok": True, "ended": _t[0], "started": started,
            "cfg": {**P, **cfg}, "metrics": {"val_bpb": bpb}}

print("1. DEFECT: a knob make_variant can express but the policy cannot see was")
print("   labelled 'control' -- it bypassed every budget AND would be pooled into the")
print("   control block that measures the noise band, destroying the instrument.")
import re
src = (REPO / "tools" / "make_variant.py").read_text()
keys = set(re.findall(r'cfg\.get\("([a-z_0-9]+)"', src)) | set(re.findall(r"cfg\['([a-z_0-9]+)'\]", src))
# known_keys() is COMPUTED, not the frozen KNOWN_KEYS constant this line used to read.
# A mechanism registered in make_variant.py teaches the policy its name and its companion
# parameters, so a snapshot taken at import would report an orphan for every mechanism
# added after that moment -- which is the closed-set behaviour this project removed.
orphan = sorted(keys - d.known_keys())
ok(not orphan, f"every expressible knob is known to the policy (orphans: {orphan})")
for k in ("batch_ramp", "compile_mode", "qk_suppress"):
    ok(not d.is_platform({**P, k: 0.5}), f"'{k}' config is not mistaken for a control")
ok(not d.is_platform({**P, "a_key_invented_tomorrow": 1}),
   "ANY unrecognised key fails safe -- the whole class, not just known instances")
ok(d.is_platform(dict(P)) and d.is_platform({}), "a genuine control is still a control")

print("\n2. DEFECT: a hardcoded cross-wave noise band was far wider than the within-wave")
print("   resolution, so nothing counted as an improvement and every axis closed dry --")
print("   reproducing the exact failure the policy was written to prevent.")
band, how = d.noise_band([])
ok(band is None, f"no controls => no instrument ({how})")
st = d.axis_state([run({"mlp": 5 + i}, 1.02) for i in range(8)])
ok(all(s["open"] for s in st["axes"].values()),
   "with no measured band, NO axis may be declared dry (uncertainty != negative result)")

# concurrent controls => within-wave band, which is small
ctl = [run({}, 1.0200, started=1000.0), run({}, 1.0201, started=1100.0),
       run({}, 1.0202, started=1200.0)]
band, how = d.noise_band(ctl)
ok(band is not None and band < 0.001, f"within-wave band is fine-grained: {band:.6f} ({how})")

# an axis that keeps paying at that resolution must stay OPEN
res = ctl + [run({"mlp": 5}, 1.015), run({"mlp": 6}, 1.010),
             run({"mlp": 7}, 1.005), run({"mlp": 8}, 1.000),
             run({"mlp": 9}, 0.995), run({"mlp": 10}, 0.990)]
st = d.axis_state(res)
ok(st["axes"]["mlp"]["open"],
   "an axis improving steadily stays OPEN (an over-wide fixed band closed it)")
ok(d.blocked_reason({"mlp": 11}, st) is None, "and further work on it is not blocked")

# A genuinely flat axis still closes -- but "flat axis" now means DISTINCT VALUES that
# each failed to pay, not one value repeated. The streak counts values, because a
# counterbalanced quad is 4 runs of the SAME value and counting runs retired every axis
# after its first experiment (measured: qk_suppress, tbs and ve each closed having tried
# exactly one value).
res2 = ctl + [run({"swdiv": v}, 1.0200) for v in (4, 8, 16, 32, 64)]
st2 = d.axis_state(res2)
ok(not st2["axes"]["swdiv"]["open"],
   "an axis flat across DISTINCT values still closes as dry")

# ...and repeating ONE value, however many times, does not close an axis: replication
# buys confidence in that value, not information about the axis.
res3 = ctl + [run({"swdiv": 4}, 1.0200) for _ in range(d.DRY_STREAK + 4)]
st3 = d.axis_state(res3)
ok(st3["axes"]["swdiv"]["open"],
   "one value repeated does NOT close the axis -- that is replication, not exploration")

print("\n3. DEFECT: injected activation observables called max() with one non-iterable")
print("   argument, raising TypeError in the trainer's summary block after training.")
for expr, name in ((lambda: max(len([1.0, 2.0][-100:]), 1), "mtp"),
                   (lambda: max([1.0, 2.0, 7][2], 1), "zloss")):
    try:
        expr(); ok(True, f"{name} observable arithmetic no longer raises")
    except TypeError as e:
        ok(False, f"{name} still raises: {e}")
ok("max(len(_h[-100:]), 1)" in src, "mtp observable carries the arity fix in source")
ok("max(_zl_sum[2], 1)" in src, "zloss observable carries the arity fix in source")

print("\n4. the three newly-visible knobs are budgeted under a real family")
for k, fam in (("batch_ramp", "token_exposure"), ("compile_mode", "systems"),
               ("qk_suppress", "attention_detail")):
    ok(k in d.FAMILIES[fam]["axes"], f"'{k}' belongs to family '{fam}'")

print("\nALL DEFECT REGRESSION TESTS PASS")

# --- deep-audit regressions -----------------------------------------------------------
print("\n5. DEEP AUDIT: the noise band must not be too TIGHT at small n")
import statistics as _st
vals = [1.0200, 1.0210]
pop, samp = _st.pstdev(vals), _st.stdev(vals)
ok(samp > pop, f"sample sd {samp:.6f} > population sd {pop:.6f} at n=2")
band, how = d.noise_band([
    {"ok": True, "started": 1000.0, "cfg": {}, "metrics": {"val_bpb": 1.0200}},
    {"ok": True, "started": 1050.0, "cfg": {}, "metrics": {"val_bpb": 1.0210}}])
ok(abs(band - 2 * samp) < 1e-12, f"band uses SAMPLE sd (conservative): {band:.6f} ({how})")
ok("sample sd" in how, "and says so")

print("\n6. DEEP AUDIT: identical controls give NO resolution, not a zero band")
band, how = d.noise_band([
    {"ok": True, "started": 1000.0, "cfg": {}, "metrics": {"val_bpb": 1.02}},
    {"ok": True, "started": 9e9, "cfg": {}, "metrics": {"val_bpb": 1.02}}])
ok(band is None, f"a zero band would make every difference 'significant' ({how})")

print("\n7. DEEP AUDIT: unknown config keys are rejected before they reach the queue")
ok(d.unknown_keys({**P, "invented_knob": 1}) == {"invented_knob"}, "unknown key detected")
ok(d.unknown_keys(dict(P)) == set(), "a clean config has none")


print("\n8. DEEP AUDIT: a 'wave' must mean genuine temporal OVERLAP, not proximity")
# Four controls run back-to-back on one GPU: sequential, NOT a yoked pair. Host load
# drifts between them, so their spread is a cross-wave spread and must be labelled so.
seq = [{"ok": True, "cfg": {}, "started": 1000.0 + i * 500, "ended": 1000.0 + i * 500 + 450,
        "metrics": {"val_bpb": 1.00 + 0.01 * i}} for i in range(4)]
band, how = d.noise_band(seq)
ok("SEQUENTIAL" in how and "NO concurrent pair" in how,
   f"back-to-back runs are not called within-wave: {how[:60]}")

# Two controls that genuinely overlap in wall clock ARE a wave.
conc = [{"ok": True, "cfg": {}, "started": 1000.0, "ended": 1500.0,
         "metrics": {"val_bpb": 1.0200}},
        {"ok": True, "cfg": {}, "started": 1100.0, "ended": 1600.0,
         "metrics": {"val_bpb": 1.0210}}]
band, how = d.noise_band(conc)
ok("within-wave" in how, f"overlapping runs are a wave: {how[:50]}")
ok(band is not None and band < 0.01, "and resolve far finer than the sequential spread")

# One overlapping pair plus a distant run: the pair is the wave, the loner is not.
mixed = conc + [{"ok": True, "cfg": {}, "started": 9e8, "ended": 9e8 + 500,
                 "metrics": {"val_bpb": 1.30}}]
band, how = d.noise_band(mixed)
ok("within-wave" in how, "a far-away run does not contaminate the wave")
ok(band < 0.01, f"and does not inflate the band ({band:.6f})")

print("\nALL DEEP-AUDIT REGRESSIONS PASS")
