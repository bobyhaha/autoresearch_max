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
    """The GPU INDEX, which is what actually carries the offset.

    This keyed on the taskset core block until a crosstab settled the question. Holding
    the GPU fixed and changing the core block moves nothing; holding the core block fixed
    and changing the GPU moves ~30 steps and ~0.0024 bpb, because the four devices are not
    equally fast (L020_the_offset_is_the_gpu_not_the_core_block). Core block and GPU were
    assigned by the same slot index, so they moved together in every wave until one quad
    wave crossed them.

    Collapsing GPUs to 0/1 was wrong and would have produced a false verdict. The
    measured slot offset (+0.000463, 10/10 same sign) is specifically between cores
    96-107 and 108-119; every one of the campaign's 26 runs used one of those two. When
    four GPUs came free the dispatcher launched two waves at once and the second landed
    on cores 120-131 and 132-143, whose relative bias has never been measured. Averaging
    a delta from one GPU PAIR with a delta from a different pair does not cancel
    anything -- it adds an unknown offset to a known one. So the pair is part of the
    identity of a wave, and counterbalancing means the SAME pair with the roles swapped.
    """
    return f"gpu{r.get('gpu')}"


def _pair(arms):
    return tuple(sorted({a["treat_dev"] for a in arms} | {a["ctl_dev"] for a in arms},
                        key=str))


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
            "treat_dev": _slot(t), "ctl_dev": _slot(c), "t": t, "c": c})

    print()
    for key, arms in sorted(by_cfg.items()):
        if want and want not in key:
            continue
        deltas = [a["delta"] for a in arms]
        print(f"=== {key} ===")
        for a in arms:
            tm, cm = a["t"]["metrics"], a["c"]["metrics"]
            print(f"  {a['wave']:8s} treat_dev={a['treat_dev']:>8s} "
                  f"delta {a['delta']:+.6f}  "
                  f"steps {tm['num_steps']:.0f} vs {cm['num_steps']:.0f}  "
                  f"epoch {tm.get('final_epoch')}/{cm.get('final_epoch')}")
            if tm.get("final_epoch") != cm.get("final_epoch"):
                print("    VOID: treatment and control finished at different final_epoch "
                      "(L005_operating_point_moved_v2) -- a regime comparison, not a result")
        # A QUAD-COUNTERBALANCED design (tools/queue_quad.py) puts the treatment on every
        # slot once and the control on every slot once across two four-wide waves, so the
        # whole slot profile cancels in the mean of the within-pair deltas -- no pairwise
        # swap is needed and no offset is fitted. Detect it and report it as one verdict.
        slots_t = {a["treat_dev"] for a in arms}
        slots_c = {a["ctl_dev"] for a in arms}
        if len(arms) >= 4 and len(slots_t) >= 4 and slots_t == slots_c:
            mean = st.mean(a["delta"] for a in arms)
            verdict = ("BETTER than control" if mean < -res else
                       "WORSE than control" if mean > res else
                       "INSIDE the resolution -- no effect demonstrated")
            print(f"  QUAD-COUNTERBALANCED over {len(arms)} pairings, treatment on every "
                  f"slot: mean delta {mean:+.6f} vs resolution {res:.6f} -> {verdict}")
            for a in arms:
                hid = a["t"].get("hypothesis_id")
                if hid and hid in hyps:
                    act = hyps[hid]["activation"]
                    val = (a["t"].get("metrics") or {}).get(act["diagnostic"])
                    print(f"    ACTIVATION {a['wave']} s{a['treat_dev']}: "
                          f"{act['diagnostic']}={val}")
            print()
            continue

        # Group arms by the GPU PAIR they ran on; only within a pair does swapping
        # the roles cancel that pair's fixed offset.
        groups = {}
        for a in arms:
            groups.setdefault(tuple(sorted((a["treat_dev"], a["ctl_dev"]),
                                           key=str)), []).append(a)
        usable = {}
        for pair, members in groups.items():
            if len({m["treat_dev"] for m in members}) >= 2:
                usable[pair] = members
            else:
                print(f"  pair {pair[0]}/{pair[1]}: treatment only ever on "
                      f"{ {m['treat_dev'] for m in members} } -- NOT counterbalanced on "
                      f"this pair, so its fixed offset is inseparable from the effect")
        if not usable:
            # An effect can be too big for the slot offset to explain. The largest offset
            # ever measured between two slots is ~0.0011 (four-wide, cores 96-107 vs
            # 108-119), so a delta several times that has a safe SIGN even uncounterbalanced
            # -- refusing to say so would be false modesty, and the campaign's own ns=3 and
            # wd_const results would be withheld on a technicality. The MAGNITUDE still
            # needs the swap; only the direction is being claimed here.
            # The largest offset between any two DEVICES we use, from control means:
            # gpu4 0.993558 vs gpu7 0.991111. Plus 2x the within-GPU residual sd.
            MAX_SLOT_OFFSET = 0.00245 + 2 * 0.000105
            if deltas and min(abs(d) for d in deltas) > 3 * MAX_SLOT_OFFSET \
                    and len({d > 0 for d in deltas}) == 1:
                m = st.mean(deltas)
                print(f"  NOT COUNTERBALANCED, but every delta is more than 3x the largest "
                      f"slot offset ever measured ({MAX_SLOT_OFFSET}) and all share a sign: "
                      f"mean {m:+.6f} -> the SIGN is safe "
                      f"({'WORSE' if m > 0 else 'BETTER'} than control). The magnitude still "
                      f"needs a same-pair swap.")
            else:
                print(f"  NO COUNTERBALANCED PAIR. Queue the swapped wave ON THE SAME GPU "
                      f"BLOCKS; a swap on a different pair adds an unmeasured offset instead "
                      f"of cancelling a measured one.")
            print()
            continue
        deltas = [m["delta"] for members in usable.values() for m in members]
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
