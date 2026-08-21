#!/usr/bin/env python3
"""Seed the queue with control runs only.

Controls are the instrument. Until the noise band is measured under the CURRENT launcher
(CPU-pinned, thread-capped), no effect size on this benchmark means anything: an earlier
campaign spent its entire run budget chasing effects smaller than a control band it had
never measured.

Six identical runs, no treatment. Launch them concurrently where possible -- the
within-wave spread is the resolution a paired comparison actually achieves.
"""
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import direction        # noqa: E402
import make_variant     # noqa: E402

SWEEP = REPO / "runs" / "sweep"
(SWEEP / "variants").mkdir(parents=True, exist_ok=True)
(SWEEP / "results").mkdir(parents=True, exist_ok=True)

def main():
    cfg = dict(direction.PLATFORM)
    src = make_variant.build(cfg)
    vid = make_variant.variant_id(src)
    (SWEEP / "variants" / vid).write_text(src)

    now = time.time()
    q = [{"name": f"C{i:02d}_control", "cfg": cfg, "variant": vid, "label": "control",
          "rationale": "measure the noise band under CPU-pinned, thread-capped launch",
          "falsifier": "if the within-wave sd is not small relative to the effects you intend to chase, "
                       "every single-run comparison on this box stays uninterpretable",
          "expected": "sd well under the spread measured without pinning",
          "created_at": now, "source_round": "bootstrap", "vram_est": 50}
         for i in range(1, 7)]

    (SWEEP / "queue.json").write_text(json.dumps(q, indent=1))
    print(f"queued {len(q)} controls, variant {vid}")
    print(f"cfg = {json.dumps(cfg)}")


if __name__ == "__main__":
    main()
