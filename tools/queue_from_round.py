#!/usr/bin/env python3
"""Turn a validated council round into dispatcher queue entries.

This closes the largest workflow hole in v3: rounds and papers proposed experiments in
prose, a human hand-translated some of them into queue JSON, and most proposals were
simply never run. Here the ```queue block IS the queue, so an idea that the council
agreed on cannot be quietly dropped.

Each entry is stamped `created_at`, which the gate's decision-cutoff rule needs: work
decided before an artifact went overdue keeps launching; work decided after it does not.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import council            # noqa: E402
import claims             # noqa: E402
import direction          # noqa: E402
import make_variant       # noqa: E402

SWEEP = REPO / "runs" / "sweep"


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else council.latest("round")
    if path is None:
        print("no round artifact found in rounds/")
        return 1
    problems = council.validate(path, "round")
    if problems:
        print(f"{path.name} is not a valid round; nothing queued:")
        for p in problems:
            print("  -", p)
        return 1

    results = [json.loads(f.read_text()) for f in (SWEEP / "results").glob("*.json")]
    state = direction.axis_state(results)
    # Past failures bind here or they bind nowhere. A `block` lesson naming config keys
    # refuses the entry outright; that is the difference between a lesson store and a
    # diary.
    blocked_by_lesson = claims.blocking_keys()

    (SWEEP / "variants").mkdir(parents=True, exist_ok=True)
    queue, skipped = [], []
    control_src = None
    stamp = path.stat().st_mtime
    for e in council.queue_entries(path.read_text()):
        cfg = e["cfg"]
        # A key no policy rule covers counts toward no axis and no family, so it would be
        # invisible to budgeting and rotation. is_platform() already refuses to call it a
        # control; refusing it at the door as well keeps the accounting total.
        hit = sorted(set(cfg) & set(blocked_by_lesson))
        if hit:
            les = blocked_by_lesson[hit[0]]
            skipped.append((e["name"], f"blocked by lesson {les['id']} (severity "
                                       f"{les['severity']}): {les['mitigation']} "
                                       f"[applies when: {les['applies_when']}]"))
            continue
        unknown = direction.unknown_keys(cfg)
        if unknown:
            skipped.append((e["name"], f"unknown config key(s) {sorted(unknown)}: add them "
                                       f"to direction.KNOB_AXES/MECHANISMS and a family, "
                                       f"or the run is invisible to the policy"))
            continue
        why = direction.blocked_reason(cfg, state)
        if why:
            skipped.append((e["name"], why))
            continue
        try:                            # build now: a variant that cannot be generated
            src = make_variant.build(cfg)   # must fail HERE, not silently at 3am on a GPU
        except Exception as exc:            # noqa: BLE001 - any build failure disqualifies
            skipped.append((e["name"], f"variant build failed: {exc}"))
            continue
        # AutoResearchClaw 3.3: static validation gates catch detectable defects --
        # notably identical ablation implementations -- BEFORE any execution budget is
        # spent. A treatment whose generated source equals the control's is a no-op that
        # would burn ~500s of exclusive GPU to measure nothing.
        if not direction.is_platform(cfg):
            if control_src is None:
                control_src = make_variant.build(dict(direction.PLATFORM))
            if src == control_src:
                skipped.append((e["name"], "generated variant is BYTE-IDENTICAL to the "
                                           "control: the intervention did not apply"))
                continue
        vid = make_variant.variant_id(src)
        (SWEEP / "variants" / vid).write_text(src)
        queue.append({"name": e["name"], "cfg": cfg, "variant": vid,
                      "label": direction.label(cfg), "rationale": e["rationale"],
                      "falsifier": e["falsifier"], "expected": e.get("expected", ""),
                      # Entries sharing a wave_group launch CONCURRENTLY on separate
                      # GPUs or not at all. This is how a yoked pair is obtained: two
                      # runs launched half an hour apart measure host drift, not effect.
                      "wave_group": e.get("wave_group"),
                      "created_at": stamp, "source_round": path.name,
                      "vram_est": e.get("vram_est", 60)})

    existing = []
    qf = SWEEP / "queue.json"
    if qf.exists():
        existing = json.loads(qf.read_text())
    have = {q["name"] for q in existing}
    new = [q for q in queue if q["name"] not in have]
    qf.write_text(json.dumps(existing + new, indent=1))

    print(f"{path.name}: queued {len(new)} new ({len(queue)} valid, "
          f"{len(existing)} already present)")
    for n, w in skipped:
        print(f"  skipped {n}: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
