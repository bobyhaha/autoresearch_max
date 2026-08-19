#!/usr/bin/env python3
"""Queue a treatment as TWO quad waves that counterbalance across all four slots.

Why not simply pair a treatment with a control: the per-slot penalty is not constant.
Measured under 4-wide load the four core blocks run 980/999/1010/1020 steps and spread
0.002342 bpb -- 13x the counterbalanced resolution and as large as a real effect
(L017_slot_offset_scales_with_our_own_concurrency). Under 2-wide load the same pair
spreads only 0.00047, so a delta measured at one width cannot be averaged with a delta
measured at the other.

This design uses all four GPUs and still cancels the slot profile EXACTLY, with no fitted
correction, by having the treatment occupy every slot once and the control occupy every
slot once across the two waves:

    wave A   [treat, ctrl , ctrl , treat]   treatment on slots 0 and 3
    wave B   [ctrl , treat, treat, ctrl ]   treatment on slots 1 and 2

Within a wave the members share host contention; across the two waves each slot's fixed
offset appears once with each sign, so the mean of the four within-pair deltas is the
treatment effect and nothing else. Eight runs, about 17 minutes, four GPUs busy.

    python3 tools/queue_quad.py --name precond_rms --cfg '{"precond":"pre"}' \
        --hyp hyp_precond_pre_r1_v3 --rationale "..." --falsifier "..." --expected "..."
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import direction        # noqa: E402
import make_variant     # noqa: E402

SWEEP = REPO / "runs" / "sweep"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", required=True, help="JSON of the treatment's non-platform keys")
    ap.add_argument("--hyp", required=True, help="registered hypothesis id, or 'none' for an instrument probe")
    ap.add_argument("--rationale", required=True)
    ap.add_argument("--falsifier", required=True)
    ap.add_argument("--expected", required=True)
    a = ap.parse_args()

    P = dict(direction.PLATFORM)
    T = {**P, **json.loads(a.cfg)}
    if direction.is_platform(T):
        print("refusing: that cfg is the control")
        return 1
    src_t, src_c = make_variant.build(T), make_variant.build(P)
    if src_t == src_c:
        print("refusing: generated variant is byte-identical to the control")
        return 1
    vt, vc = make_variant.variant_id(src_t), make_variant.variant_id(src_c)
    (SWEEP / "variants").mkdir(parents=True, exist_ok=True)
    (SWEEP / "variants" / vt).write_text(src_t)
    (SWEEP / "variants" / vc).write_text(src_c)

    qf = SWEEP / "queue.json"
    q = json.loads(qf.read_text()) if qf.exists() else []
    have = {e["name"] for e in q}
    stamp = time.time()
    layout = {f"{a.name}_A": ["t", "c", "c", "t"],
              f"{a.name}_B": ["c", "t", "t", "c"]}
    new = []
    for grp, roles in layout.items():
        for i, role in enumerate(roles):
            nm = f"{grp}_s{i}_{'treat' if role == 't' else 'ctrl'}"
            if nm in have:
                continue
            e = {"name": nm, "cfg": T if role == "t" else P,
                 "variant": vt if role == "t" else vc,
                 "label": direction.label(T if role == "t" else P),
                 "rationale": a.rationale, "falsifier": a.falsifier, "expected": a.expected,
                 "wave_group": grp, "created_at": stamp,
                 "source_round": "quad-counterbalanced", "vram_est": 50}
            if role == "t" and a.hyp != "none":
                e["hypothesis_id"] = a.hyp
            new.append(e)
    qf.write_text(json.dumps(q + new, indent=1))
    print(f"queued {len(new)} entries in 2 quad waves; treatment variant {vt}")
    for e in new:
        print(f"  {e['name']:34s} {e['label']:14s} wave={e['wave_group']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
