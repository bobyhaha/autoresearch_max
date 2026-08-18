#!/usr/bin/env python3
"""Read the campaign honestly.

The v3 analysis pipeline made one deep methodological error that changed verdicts. It
fitted val_bpb ~ a + b*ln(num_steps) across runs and reported the RESIDUAL as the effect,
calling the step term a "control variate". That is valid for removing EXOGENOUS noise --
another tenant stealing CPU makes our loader slower through no fault of the treatment.
It is invalid for ENDOGENOUS cost -- a mechanism that itself costs throughput. The two
were regressed out with the same coefficient, so a change that halved step count was
reported as nearly neutral.

Concretely, a mechanism that cut step count by a large fraction was reported as roughly
break-even once its throughput cost had been divided out, while at a fixed 300-second
budget it was far worse than the control. The residual answered a question nobody asked.

So: RAW val_bpb is the verdict, always. Step count is printed beside it as a diagnostic
that explains WHY, and contention is reported as a confidence qualifier rather than
silently subtracted.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import claims     # noqa: E402
import direction  # noqa: E402

RES = REPO / "runs" / "sweep" / "results"


def load():
    out = []
    for f in sorted(RES.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if r.get("ok") and (r.get("metrics") or {}).get("val_bpb"):
            out.append(r)
    return out


def control_band(rows):
    """The measured noise floor: spread among byte-identical control runs.

    This is the number that decides whether anything else is real. Quote it from the
    controls that actually ran, and prefer concurrent ones: host CPU contention moves step
    count between waves, so a cross-wave spread overstates the resolution a paired
    comparison achieves, while a single wave's spread understates the risk of comparing
    across waves. Report which you used.
    """
    ctl = [r for r in rows if direction.is_platform(r["cfg"] or {})]
    if len(ctl) < 2:
        return None
    v = [r["metrics"]["val_bpb"] for r in ctl]
    s = [r["metrics"].get("num_steps", 0) for r in ctl]
    return {"n": len(v), "mean": st.mean(v), "sd": st.stdev(v),
            "range": max(v) - min(v), "steps_min": min(s), "steps_max": max(s)}


def main():
    rows = load()
    if not rows:
        print("no valid results yet.\n")
        print(direction.report(rows))
        return 0
    rows.sort(key=lambda r: r["metrics"]["val_bpb"])
    band = control_band(rows)

    print(f"=== {len(rows)} valid runs | best val_bpb {rows[0]['metrics']['val_bpb']:.6f} "
          f"({rows[0]['name']}) ===\n")

    if band:
        print(f"CONTROL BAND  n={band['n']}  mean {band['mean']:.6f}  sd {band['sd']:.6f}  "
              f"range {band['range']:.6f}  (steps {band['steps_min']:.0f}-{band['steps_max']:.0f})")
        print(f"  -> an effect below {2*band['sd']:.6f} is not distinguishable at n=1.\n")
    else:
        print("CONTROL BAND  unmeasured: fewer than 2 control runs. Until it exists, no "
              "effect size means anything. Queue controls.\n")

    print(f"{'name':30s} {'val_bpb':>9s} {'steps':>6s} {'mfu':>6s} {'load%':>6s}  label")
    for r in rows[:25]:
        m = r["metrics"]
        print(f"{r['name'][:30]:30s} {m['val_bpb']:9.6f} {m.get('num_steps',0):6.0f} "
              f"{m.get('mfu_percent',0):6.2f} {100*m.get('loader_frac',0):6.1f}  "
              f"{direction.label(r['cfg'] or {})}")
    if len(rows) > 25:
        print(f"... {len(rows)-25} more")

    # Throughput diagnosis: on this benchmark the input pipeline, not the GPU, sets the
    # step count, and step count dominates val_bpb. Say so with numbers every time.
    lf = [r["metrics"]["loader_frac"] for r in rows if r["metrics"].get("loader_frac")]
    if lf:
        print(f"\nloader_frac: median {100*st.median(lf):.1f}% "
              f"(min {100*min(lf):.1f}%, max {100*max(lf):.1f}%)")
        print("  CAUTION: loader_frac and fwdbwd_frac are CPU wall-clock fractions measured")
        print("  around calls that launch ASYNCHRONOUS CUDA work. fwdbwd_frac is kernel")
        print("  LAUNCH time, not GPU busy time, so a high loader_frac does NOT establish")
        print("  that the GPU is idle. Treat these as a CPU-side attribution only; any")
        print("  claim about where the 300s goes needs a controlled FLOPs comparison.")

    xs = [math.log(r["metrics"]["num_steps"]) for r in rows if r["metrics"].get("num_steps")]
    ys = [r["metrics"]["val_bpb"] for r in rows if r["metrics"].get("num_steps")]
    if len(xs) > 3:
        mx, my = st.mean(xs), st.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            resid = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
            expl = 1 - st.pvariance(resid) / st.pvariance(ys) if st.pvariance(ys) else 0
            print(f"\nDIAGNOSTIC ONLY (never subtracted from a verdict):")
            print(f"  val_bpb moves {b:+.5f} per e-fold of steps; step count alone explains "
                  f"{expl:.0%} of all variance.")
            print("  High explained variance means most 'results' are throughput results.")
            print("  A treatment that costs steps is genuinely worse at 300s -- that cost")
            print("  is the finding, not a nuisance term to regress away.")

    allr = []
    for f in sorted(RES.glob("*.json")):
        try:
            allr.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    bad = [r for r in allr if not r.get("ok") or r.get("invalid_reason")]
    if bad:
        un = {r["name"] for r in claims.unlearned_failures(allr)}
        print(f"\nFAILED / INVALID RUNS ({len(bad)}) -- each should produce a lesson:")
        for r in bad:
            mark = "NO LESSON" if r["name"] in un else "lesson registered"
            why = r.get("invalid_reason") or (r.get("error") or "")[:60] or "returncode != 0"
            print(f"  {r['name']:24s} {why[:52]:52s} [{mark}]")
        if un:
            print(f"  -> {len(un)} unlearned. Register with: "
                  f"tools/claims.py lesson file.json (templates/lesson.json)")

    ls = claims.active_lessons()
    if ls:
        print(f"\nACTIVE LESSONS ({len(ls)}), most severe first:")
        for l in ls[:5]:
            blocks = f"  BLOCKS {l['blocks_keys']}" if l.get("blocks_keys") else ""
            print(f"  [{l['severity']:.2f}] {l['id']} ({l['type']} -> {l['action']})"
                  f"{blocks}\n        {l['mitigation'][:90]}")

    print()
    print(direction.report(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
