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

    # THE PLATFORM BASELINE, PRINTED. It was quoted all evening as 0.984205 and printed
    # by no tool at all -- a prose number, exactly the class of defect the step law was
    # (L055), and it had already drifted to 0.984181 as more controls landed while the
    # stale figure kept being repeated. Anything the campaign quotes as a headline has to
    # come out of a tool, or it silently becomes a memory of a number.
    _plat = [r["metrics"]["val_bpb"] for r in rows
             if r.get("ok") and direction.is_platform(r.get("cfg") or {})
             and (r.get("metrics") or {}).get("final_epoch") == 2.0]
    if _plat:
        import collections as _c
        import statistics as _s
        _by = _c.defaultdict(list)
        for r in rows:
            if (r.get("ok") and direction.is_platform(r.get("cfg") or {})
                    and (r.get("metrics") or {}).get("final_epoch") == 2.0):
                _by[r.get("gpu")].append(r["metrics"]["val_bpb"])
        _sds = [_s.stdev(v) for v in _by.values() if len(v) > 1]
        _pooled = (sum(s * s for s in _sds) / len(_sds)) ** 0.5 if _sds else float("nan")
        print(f"PLATFORM BASELINE  {_s.mean(_plat):.6f}  over {len(_plat)} controls on "
              f"{len(_by)} device(s); pooled within-GPU sd {_pooled:.6f} "
              f"({len(_sds)} device(s) with n>1)")
        print(f"  cfg {dict(direction.PLATFORM)}")
        # AND THE POOL WITHOUT ITS OWN JUSTIFYING EVIDENCE (L091). Adopting a value into
        # PLATFORM retroactively reclassifies the TREATMENT arms that justified the
        # adoption as controls, because is_platform() reads the cfg against a platform
        # that has since moved. Four R5T18 tbs=18 treatments entered the pool that way.
        # They are not obviously wrong to include -- they did run the current platform
        # config -- but they were selected for having won, and their mean is WORSE than
        # the rest, which raises the baseline and makes every treatment gap look larger.
        # Repooling silently would move every verdict on record, so both are printed and
        # the reader decides.
        # THE SAME POPULATION AS _plat, final_epoch filter included. Without it these two
        # blocks selected over a different set than the baseline they claim to correct:
        # the printed count read 30 where 33 - 4 = 29, and the correction shrank from
        # 0.000054 to 0.000007 -- erasing about seven eighths of an adjustment that moves
        # the headline in the UNFLATTERING direction. A correction computed over the
        # wrong population is worse than no correction, because it looks like diligence.
        def _pool(pred):
            return [r for r in rows
                    if r.get("ok") and direction.is_platform(r.get("cfg") or {})
                    and (r.get("metrics") or {}).get("final_epoch") == 2.0
                    and pred(r)]
        _entered = _pool(lambda r: r["name"].endswith("_treat"))
        if _entered:
            _clean = [r["metrics"]["val_bpb"]
                      for r in _pool(lambda r: not r["name"].endswith("_treat"))]
            if _clean:
                print(f"  of which {len(_entered)} entered by RECLASSIFICATION after an "
                      f"adoption (queued as treatments): "
                      f"{', '.join(sorted(r['name'] for r in _entered))}")
                print(f"  baseline excluding them  {_s.mean(_clean):.6f}  over "
                      f"{len(_clean)} controls  (L091)")
        print()

    if band:
        print(f"ALL-CONTROL SPREAD (UNPAIRED)  n={band['n']}  mean {band['mean']:.6f}  "
              f"sd {band['sd']:.6f}  range {band['range']:.6f}  "
              f"(steps {band['steps_min']:.0f}-{band['steps_max']:.0f})")
        print(f"  -> an effect below {2*band['sd']:.6f} is not distinguishable by an "
              f"UNPAIRED single run.")
        # Two different bands are printed by this campaign and confusing them inverts
        # verdicts. This one pools controls across waves, so it carries the host drift
        # BETWEEN waves. The band the policy actually judges against is the WITHIN-WAVE
        # one that direction.py measures from concurrent controls, printed in the footer
        # below; it is the resolution a yoked pair buys and it is much smaller. Read a
        # yoked treatment-vs-control delta against the footer band, never against this.
        print(f"  -> this pools across waves and therefore contains host drift; for a "
              f"YOKED pair use the within-wave band in the footer.\n")
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
