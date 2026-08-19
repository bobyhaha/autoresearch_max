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

    python3 tools/queue_controls.py --wave W02 --width 2 --count 3

queues 3 waves of 2 concurrent controls (W02a..W02c), each wave launching together on
separate GPUs or not at all.
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
    ap.add_argument("--wave", required=True, help="wave name prefix, e.g. W02")
    ap.add_argument("--width", type=int, default=2, help="GPUs per wave (concurrent)")
    ap.add_argument("--count", type=int, default=1, help="how many such waves")
    ap.add_argument("--rationale", default="measure the WITHIN-WAVE noise band from "
                    "concurrent byte-identical controls; the four existing controls ran "
                    "back-to-back on one GPU so their spread is cross-wave drift")
    a = ap.parse_args()

    cfg = dict(direction.PLATFORM)
    assert direction.is_platform(cfg), "PLATFORM must be a pure control"
    src = make_variant.build(cfg)
    vid = make_variant.variant_id(src)
    (SWEEP / "variants").mkdir(parents=True, exist_ok=True)
    (SWEEP / "variants" / vid).write_text(src)

    qf = SWEEP / "queue.json"
    existing = json.loads(qf.read_text()) if qf.exists() else []
    have = {q["name"] for q in existing}
    stamp = time.time()

    new = []
    for w in range(a.count):
        group = f"{a.wave}{chr(ord('a') + w)}"
        for i in range(a.width):
            name = f"{group}_{i+1}_control"
            if name in have:
                continue
            new.append({
                "name": name, "cfg": cfg, "variant": vid, "label": direction.label(cfg),
                "rationale": a.rationale,
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

    qf.write_text(json.dumps(existing + new, indent=1))
    print(f"queued {len(new)} control(s) in {a.count} wave(s) of {a.width} "
          f"[variant {vid}]")
    for e in new:
        print(f"  {e['name']}  wave={e['wave_group']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
