#!/usr/bin/env python3
"""EXPLORE / EXPLOIT ledger, with a mechanical saturation test for knob sweeps.

The campaign ran ~96 runs at essentially 100% exploration: every treatment tested a NEW
thing once against control, and nothing was ever swept or stacked. Two causes, both
structural rather than accidental.

  1. Nothing MEASURED the split, so nobody could see it drifting. EXPLORE_FLOOR guards
     one side only -- it stops exploitation crowding out exploration, and is silent when
     exploitation is zero, which is the failure that actually occurred.
  2. The dry rule closed an axis using the very runs that CONFIRMED it, after which the
     proven value could not be reused. direction.blocked_reason now exempts an axis's own
     best-known value, so reuse is allowed while trying new values stays refused.

Classification, per treatment run:
  EXPLOIT  reuses an axis's best-known value, or is the next rung of an active sweep
  EXPLORE  anything else -- a new value, an untouched axis, a mechanism

SATURATION is judged rung-to-rung, not against control. A knob that pays does so sharply
and then flattens; comparing each rung against the CONTROL keeps showing a large win long
after the marginal gain has died, which would spend wave after wave confirming a plateau.
So a sweep continues only while rung N+1 beats rung N by more than the resolution.

    python3 tools/balance.py
"""
from __future__ import annotations

import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import analyze      # noqa: E402
import direction    # noqa: E402

EXPLOIT_FLOOR = 0.30      # exploitation may not sit at zero; the failure observed
EXPLOIT_CEIL = 0.50       # nor crowd out exploration


def _device_means(rows):
    by = {}
    for r in rows:
        if (r.get("ok") and direction.is_platform(r.get("cfg") or {})
                and (r.get("metrics") or {}).get("final_epoch") == 2.0):
            by.setdefault(r.get("gpu"), []).append(r["metrics"]["val_bpb"])
    return {g: st.mean(v) for g, v in by.items() if v}


def classify(rows, state, res=None):
    """Three-way split of treatment runs, in TIME ORDER.

    A first version of this asked only whether a run's values equal the axis's
    best-known value, and so counted the very run that DISCOVERED a value as reusing
    it -- reporting 62% exploitation on a campaign whose true figure was zero, which
    would have hidden exactly the imbalance the ledger exists to surface. Novelty is a
    property of a run relative to what came BEFORE it, so the pass must be chronological.

      EXPLORE    at least one axis value here has never been run before
      EXPLOIT    every value is already proven, but this COMBINATION is new -- i.e.
                 stacking known-good settings, or a sweep rung built on them
      REPLICATE  this exact cfg has run before; verification, neither of the above
    """
    seen_val, seen_cfg, seen_eff = {}, set(), {}
    dm = _device_means(rows)
    exploit, explore, repl = [], [], []
    for r in sorted(rows, key=lambda x: x.get("ended") or 0):
        cfg = r.get("cfg") or {}
        if not r.get("ok") or direction.is_platform(cfg):
            continue
        ax = direction.axes_touched(cfg)
        key = tuple(sorted((a, cfg.get(a)) for a in ax))
        novel = [a for a in ax if cfg.get(a) not in seen_val.get(a, set())]
        # A NEW value on an axis that already has a confirmed win is a SWEEP RUNG:
        # exploring that axis further, but exploiting the finding that the axis pays. That
        # is exactly the "turn the knob until it saturates" move, and filing it under
        # exploration made the ledger disagree with the strategy it exists to steer. A new
        # value on an axis with no win yet is genuine exploration.
        # `pays` must be judged on what was known BEFORE this run, not on the campaign's
        # final state. Using final-state best_eff looked backwards through time and
        # reclassified the very runs that DISCOVERED each lever as exploitation of it,
        # reporting a balanced 50% when the true chronological share was 1/8. That is the
        # third distinct defect in a ledger built to keep the operator honest about this
        # exact number, and the failure mode each time was the same: the measurement
        # flattered the thing it was measuring.
        pays = [a for a in novel if (seen_eff.get(a) or 0) < -(res or 0)]
        if novel and not pays:
            explore.append(r)
        elif novel and pays:
            exploit.append(r)
        elif key in seen_cfg:
            repl.append(r)
        else:
            exploit.append(r)
        eff = r["metrics"]["val_bpb"] - dm.get(r.get("gpu"), r["metrics"]["val_bpb"])
        for a in ax:
            seen_val.setdefault(a, set()).add(cfg.get(a))
            if a not in seen_eff or eff < seen_eff[a]:
                seen_eff[a] = eff          # knowledge as of AFTER this run, not the end
        seen_cfg.add(key)
    return exploit, explore, repl


def sweeps(rows, state):
    """Per-axis rung ladder: each distinct value's device-corrected effect."""
    dm = _device_means(rows)
    out = {}
    for r in rows:
        cfg = r.get("cfg") or {}
        if not r.get("ok") or direction.is_platform(cfg):
            continue
        for a in direction.axes_touched(cfg):
            eff = r["metrics"]["val_bpb"] - dm.get(r.get("gpu"), r["metrics"]["val_bpb"])
            out.setdefault(a, {}).setdefault(cfg.get(a), []).append(eff)
    return {a: {v: st.mean(e) for v, e in rungs.items()} for a, rungs in out.items()}


def main() -> int:
    rows = analyze.load()
    state = direction.axis_state(rows)
    res = (direction.device_resolution(rows) or {}).get("resolution")
    exploit, explore, repl = classify(rows, state, res)
    # The share is measured over EXPERIMENTS, not runs. A counterbalanced experiment is
    # one novel run plus ~7 device replicates, so a share taken over runs cannot exceed
    # about 12.5% even if every experiment were exploitation -- the 30% floor would have
    # been unreachable by construction, which is the same defect as an activation rule the
    # control also passes: a threshold nothing can cross measures nothing. Replicates are
    # the cost of counterbalancing and belong to whichever experiment they serve, not to a
    # third category competing with it.
    n = len(exploit) + len(explore)
    share = len(exploit) / n if n else 0.0

    print(f"=== EXPLORE / EXPLOIT LEDGER ===  resolution {res:.6f}" if res else "=== LEDGER ===")
    print(f"  experiments: {n}   EXPLORE {len(explore)}   EXPLOIT {len(exploit)} "
          f"({share:.0%})    [+{len(repl)} counterbalancing replicates, not counted]")
    if share < EXPLOIT_FLOOR:
        need = max(1, int(round(EXPLOIT_FLOOR * n / (1 - EXPLOIT_FLOOR))) - len(exploit))
        print(f"  BELOW the {EXPLOIT_FLOOR:.0%} exploit floor -- the next launches should "
              f"SWEEP a winning knob or STACK proven values (about {max(need,1)} run(s) "
              f"more experiment(s)). Measuring levers and never spending them is how a "
              f"campaign ends with a notebook and no result.")
    elif share > EXPLOIT_CEIL:
        print(f"  ABOVE the {EXPLOIT_CEIL:.0%} exploit ceiling -- next launches should go "
              f"to an untouched axis or a mechanism, or the search collapses to a "
              f"monoculture around one lever.")
    else:
        print(f"  in band [{EXPLOIT_FLOOR:.0%}, {EXPLOIT_CEIL:.0%}] -- balanced.")

    print("\n  SWEEP LADDERS (device-corrected mean effect per rung; marginal is "
          "rung-to-rung, which is what saturates)")
    lad = sweeps(rows, state)
    for a in sorted(lad):
        rungs = lad[a]
        if len(rungs) < 1:
            continue
        base = direction.PLATFORM.get(a)
        order = sorted(rungs, key=lambda v: rungs[v])
        best, best_eff = order[0], rungs[order[0]]
        line = "  ".join(f"{v}:{rungs[v]:+.6f}" for v in sorted(rungs, key=str))
        print(f"    {a:10s} baseline {base!r}   {line}")
        if len(rungs) >= 2 and res:
            prev = sorted(rungs.items(), key=lambda kv: kv[1])[:2]
            marg = prev[0][1] - prev[1][1]
            verdict = ("CONTINUE the sweep" if marg < -res else
                       f"SATURATED -- marginal {marg:+.6f} is inside +/-{res:.6f}; "
                       f"stop sweeping and return the budget to exploration")
            print(f"    {'':10s} best {best!r} vs runner-up {prev[1][0]!r}: "
                  f"marginal {marg:+.6f} -> {verdict}")
        elif res:
            print(f"    {'':10s} best {best!r} at {best_eff:+.6f}; ONE rung only -- the "
                  f"next rung decides whether this knob is still paying or already flat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
