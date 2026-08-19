#!/usr/bin/env python3
"""Read a counterbalanced treatment verdict, with the slot offset removed by DESIGN.

Why this is not part of analyze.py's table: analyze.py ranks runs by raw val_bpb, which
is correct for "what is the best number we have" and wrong for "did this intervention
work". A treatment is compared to the control IN ITS OWN WAVE, because host contention
moves step count between waves by more than any effect worth chasing; and the two waves
of a counterbalanced pair are AVERAGED, because dispatcher slot 0 carries a fixed
~0.00047 bpb penalty against slot 1 (L006_slot_bias_fakes_effects) that a single pair
cannot distinguish from a result.

The verdict is a mean of within-wave deltas. It never regresses step count out: a
treatment that costs throughput is genuinely worse at a fixed 300 seconds, and that cost
is the finding (CLAUDE.md). Step count is printed beside the verdict to explain it.

    python3 tools/verdict.py            # every counterbalanced treatment
    python3 tools/verdict.py precond    # just cfgs whose label matches
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import claims as C      # noqa: E402
import direction        # noqa: E402

RES = REPO / "runs" / "sweep" / "results"


def _load():
    out = []
    for f in sorted(RES.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if r.get("ok") and (r.get("metrics") or {}).get("val_bpb"):
            out.append(r)
    return out


def _slot(r):
    """0 or 1 from the taskset core block, ordered NUMERICALLY -- '108-119' sorts before
    '96-107' as a string, which silently flips the sign of every slot statistic."""
    c = str(r.get("cores") or "")
    try:
        return 0 if int(c.split("-")[0]) < 108 else 1
    except ValueError:
        return -1


def waves(rows):
    out = {}
    for r in rows:
        g = None
        for e in json.loads((REPO / "runs" / "sweep" / "queue.json").read_text()):
            if e["name"] == r["name"]:
                g = e.get("wave_group")
                break
        if g:
            out.setdefault(g, []).append(r)
    return out


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    rows = _load()
    band, how = direction.noise_band(rows)
    sb = direction.slot_bias(rows)
    print(f"within-wave band {band:.6f} ({how})" if band else "band unmeasured")
    if sb:
        res = 2 * sb["resid_sd"] / math.sqrt(2)
        print(f"slot offset {sb['offset']:+.6f} ({sb['same_sign']}/{sb['n']} same sign); "
              f"COUNTERBALANCED RESOLUTION {res:.6f}")
    else:
        res = band or 0.0

    hyps = {h["id"]: h for h in C.hypotheses()}
    by_cfg = {}
    for g, members in waves(rows).items():
        ctl = [m for m in members if direction.is_platform(m["cfg"] or {})]
        trt = [m for m in members if not direction.is_platform(m["cfg"] or {})]
        if len(ctl) != 1 or len(trt) != 1:
            continue
        c, t = ctl[0], trt[0]
        key = direction.label(t["cfg"])
        by_cfg.setdefault(key, []).append({
            "wave": g, "delta": t["metrics"]["val_bpb"] - c["metrics"]["val_bpb"],
            "treat_slot": _slot(t), "t": t, "c": c})

    print()
    for key, arms in sorted(by_cfg.items()):
        if want and want not in key:
            continue
        slots = {a["treat_slot"] for a in arms}
        deltas = [a["delta"] for a in arms]
        print(f"=== {key} ===")
        for a in arms:
            tm, cm = a["t"]["metrics"], a["c"]["metrics"]
            print(f"  {a['wave']:8s} treat_slot={a['treat_slot']} "
                  f"delta {a['delta']:+.6f}  "
                  f"steps {tm['num_steps']:.0f} vs {cm['num_steps']:.0f}  "
                  f"epoch {tm.get('final_epoch')}/{cm.get('final_epoch')}")
            if tm.get("final_epoch") != cm.get("final_epoch"):
                print("    VOID: treatment and control finished at different final_epoch "
                      "(L005_operating_point_moved_v2) -- a regime comparison, not a result")
        if len(slots) < 2:
            print(f"  NOT COUNTERBALANCED: treatment only ever ran in slot {slots}. "
                  f"The slot offset is inseparable from the effect; queue the swapped wave.")
            print()
            continue
        mean = st.mean(deltas)
        verdict = ("BETTER than control" if mean < -res else
                   "WORSE than control" if mean > res else
                   "INSIDE the resolution -- no effect demonstrated")
        print(f"  counterbalanced mean delta {mean:+.6f}  vs resolution {res:.6f}  -> {verdict}")
        # activation, if a hypothesis was declared
        for a in arms:
            hid = a["t"].get("hypothesis_id")
            if not hid or hid not in hyps:
                continue
            act = hyps[hid]["activation"]
            diag, rule = act["diagnostic"], act["rule"]
            val = (a["t"].get("metrics") or {}).get(diag)
            if val is None:
                print(f"  ACTIVATION {a['wave']}: '{diag}' NOT EMITTED -> INCONCLUSIVE "
                      f"about {hid}, never evidence against it")
            else:
                ok = {"gt": val > rule["value"], "lt": val < rule["value"],
                      "ge": val >= rule["value"], "le": val <= rule["value"],
                      "ne": val != rule["value"],
                      "abs_gt": abs(val) > rule["value"]}.get(rule["op"])
                print(f"  ACTIVATION {a['wave']}: {diag}={val} rule {rule} -> "
                      f"{'ENGAGED' if ok else 'DID NOT ENGAGE -> INCONCLUSIVE'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
