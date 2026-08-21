#!/usr/bin/env python3
"""Queue byte-identical CONTROL waves — the instrument, not a research decision.

Controls are exempt from the gate's decision cutoff (host/dispatch.py `_is_control`)
because they measure the noise band that every verdict is read against, and freezing
them for an overdue council artifact would starve exactly the thing that makes a
verdict possible.

Why this exists as a tool rather than hand-edited JSON: entries must be built through
`make_variant.build(direction.PLATFORM)` and `variant_id()` so the queued variant is the
same content-addressed file the research entries are diffed against. A hand-written
queue entry pointing at a stale variant filename is a control that is not a control.

    python3 tools/queue_controls.py --wave W02 --width 2 --count 3 \
        --measures "within-wave spread of concurrent byte-identical controls"

queues 3 waves of 2 concurrent controls (W02a..W02c), each wave launching together on
separate GPUs or not at all.

STANDALONE CONTROL WAVES ARE NOW REFUSED BY DEFAULT, and that refusal is the point of
this docstring. The campaign spent 43 of its first 54 valid runs on controls -- a 3.91:1
control-to-treatment ratio -- to establish three instrument quantities that are now
MEASURED and will not change by measuring them again: the within-wave band, the slot
offset, and the per-GPU resolution. Every further standalone control wave buys a
marginally tighter estimate of a number that is already small enough to resolve the
effects being chased, at the price of a treatment that was never run.

A control is still required for every treatment -- that is the yoked pair, and it is
queued alongside its treatment by `queue_quad.py` / `queue_from_round.py`, not here.
What this tool queues is a control with NO treatment, which is an instrument calibration
and nothing else. So it now demands `--measures` naming the instrument quantity, and if
that quantity is already measured it refuses unless `--force` is given with a reason.
The failure mode being prevented is not a wrong number; it is a campaign that spends its
budget confirming it can measure, and never measures anything.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import direction        # noqa: E402
import make_variant     # noqa: E402
import queue_store      # noqa: E402

SWEEP = REPO / "runs" / "sweep"


def _already_measured(what: str) -> str:
    """Refuse a calibration wave for a quantity the results already pin down.

    Reads the SAME estimators the verdicts are read against, so this can never refuse on
    the basis of a number no verdict uses. Returns a refusal message, or "" to allow.
    """
    import analyze
    rows = analyze.load()
    band, how = direction.noise_band(rows)
    dev = direction.device_resolution(rows)
    slot = direction.slot_bias(rows)
    ctl = [r for r in rows if direction.is_platform(r.get("cfg") or {})]
    trt = len(rows) - len(ctl)
    have = []
    if band is not None:
        have.append(f"within-wave band {band:.6f} ({how})")
    if dev:
        have.append(f"per-GPU resolution {dev['resolution']:.6f} over {dev['n_devices']} GPUs")
    if slot:
        have.append(f"slot offset {slot['offset']:+.6f}")
    if not have:
        return ""
    return ("REFUSED: the instrument is already measured, so this wave would spend GPU "
            "time re-estimating a known number instead of testing something.\n"
            "  measured: " + "\n            ".join(have) + "\n"
            f"  budget so far: {len(ctl)} control runs vs {trt} treatment runs "
            f"({len(ctl)/max(trt,1):.2f}:1). Controls are already the majority of this "
            f"campaign.\n"
            f"  requested: {what!r}\n"
            "  A yoked control for a specific treatment does NOT need this tool -- it is "
            "queued with its treatment by queue_quad.py / queue_from_round.py, which is "
            "the only control the design actually requires.\n"
            "  If the existing estimate is genuinely insufficient, say why: "
            "--force 'reason'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True, help="wave name prefix, e.g. W02")
    ap.add_argument("--width", type=int, default=2, help="GPUs per wave (concurrent)")
    ap.add_argument("--count", type=int, default=1, help="how many such waves")
    ap.add_argument("--measures", required=True,
                    help="the INSTRUMENT QUANTITY this wave estimates, e.g. "
                         "'within-wave band'. Required: a control wave that cannot name "
                         "what it measures is a run that measures nothing.")
    ap.add_argument("--force", metavar="REASON", default="",
                    help="queue anyway when the quantity is already measured, recording "
                         "why the existing estimate is insufficient")
    ap.add_argument("--rationale", default="")
    a = ap.parse_args()
    a.rationale = a.rationale or f"instrument calibration: {a.measures}"

    refusal = _already_measured(a.measures)
    if refusal and not a.force:
        print(refusal, file=sys.stderr)
        return 2
    if refusal and a.force:
        print(f"OVERRIDE: {a.force}\n(refusal was: {refusal.splitlines()[0]})")

    cfg = dict(direction.PLATFORM)
    assert direction.is_platform(cfg), "PLATFORM must be a pure control"
    src = make_variant.build(cfg)
    vid = make_variant.variant_id(src)
    (SWEEP / "variants").mkdir(parents=True, exist_ok=True)
    (SWEEP / "variants" / vid).write_text(src)

    qf = SWEEP / "queue.json"
    stamp = time.time()

    new = []

    def merge(existing):
        nonlocal new
        have = {q["name"] for q in existing}
        additions = []
        for w in range(a.count):
            group = f"{a.wave}{chr(ord('a') + w)}"
            for i in range(a.width):
                name = f"{group}_{i+1}_control"
                if name in have:
                    continue
                additions.append({
                    "name": name, "cfg": cfg, "role": "ctrl", "variant": vid,
                    "label": direction.label(cfg), "rationale": a.rationale,
                    "falsifier": "if the within-wave spread of these concurrent controls is "
                                 "as large as the 0.019990 sequential range of C03-C06, then "
                                 "pairing buys nothing on this host and the band is "
                                 "irreducible at this budget",
                    "expected": "within-wave spread materially smaller than the sequential "
                                "range, because concurrent runs share the host contention "
                                "that moves step count",
                    "wave_group": group, "created_at": stamp,
                    "source_round": "instrument", "vram_est": 50,
                })
        new = additions
        return existing + additions

    queue_store.update_queue(qf, merge)
    print(f"queued {len(new)} control(s) in {a.count} wave(s) of {a.width} "
          f"[variant {vid}]")
    for e in new:
        print(f"  {e['name']}  wave={e['wave_group']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
