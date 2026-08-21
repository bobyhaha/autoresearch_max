#!/usr/bin/env python3
"""Keep the fleet fed. Convert a candidate backlog into n=1 screening runs.

WHY THIS EXISTS
---------------
The v3 campaign ran 226 experiments across 41.5 wall-hours on a 7-GPU fleet and spent
**89.6% of its GPU-hours idle** -- 260 of 290. The idle was not diffuse: six gaps longer
than an hour accounted for 26.8 of the 41.5 hours, and every one of them sits where the
orchestrator was deliberating or repairing its own harness. Nothing in the system
produced work except a human-authored council round, and `tick.sh` is explicit that it
"never launches or kills an experiment". So when the queue drained, the box stopped.

Two peer harnesses on the identical 300s/H200 frame reached a better `val_bpb` on ONE
GPU in less wall-clock: fengheguai 0.9684 in 18h, AI-Scientist-v2 0.9816 in 3 trials.
v3 reached 0.9812 after 226 runs. The gap is not the science and it is not the trainer --
it is that the fleet was switched off while the humans thought.

WHAT A SCREEN IS, AND IS NOT
----------------------------
A screen is n=1, width-1, and decides NOTHING. It is admissible as evidence for exactly
one proposition: whether a candidate is worth spending a real yoked comparison on.

  - It never enters the noise band (it is not a platform control, so `device_means`
    already excludes it).
  - It never forms a verdict pair (it carries no `wave_group`, and `verdict.waves()`
    drops screen rows outright).
  - It can never be cited for adoption. Promotion means writing a round entry with a
    registered hypothesis and a counterbalanced wave, exactly as before.

That is the trade this module makes: screens buy back the idle fleet without buying any
authority. The gate still owns adoption; the screen only owns "look here next".

CANDIDATE EXHAUSTION IS A LOUD STATE
------------------------------------
This module cannot invent science, and it does not pretend to. When the backlog is dry
it says so, on every tick, naming the two things that refill it. An idle fleet with an
empty backlog is a legible request for hypotheses. An idle fleet with a FULL backlog was
the old bug, and it cannot recur silently.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import direction
import make_variant
import queue_store

SWEEP = REPO / "runs" / "sweep"
CANDIDATES = REPO / "baseline" / "candidates"

# Keep this many pending entries per GPU. A tick is 20 minutes and a run is ~8, so a
# just-in-time refill would still drain the box twice between ticks. Depth, not latency,
# is what keeps a slow control loop from idling a fast fleet.
DEPTH_PER_GPU = float(os.environ.get("OPHIS_QUEUE_DEPTH_PER_GPU", "2"))


def fleet_width() -> int:
    raw = os.environ.get("OPHIS_GPUS", "")
    if not raw:
        env = REPO / ".env"
        if env.exists():
            m = re.search(r"^OPHIS_GPUS=(.*)$", env.read_text(), re.MULTILINE)
            raw = m.group(1).strip() if m else ""
    gpus = [g for g in raw.replace(" ", "").split(",") if g]
    return len(gpus) or 1


def _results() -> list[dict]:
    out = []
    for f in (SWEEP / "results").glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return out


def pending(queue: list[dict]) -> list[dict]:
    """Entries that still represent work: no result on disk and no live claim."""
    out = []
    for e in queue:
        if (SWEEP / "results" / f"{e['name']}.json").exists():
            continue
        if (SWEEP / "claims" / e["name"]).exists():
            continue
        out.append(e)
    return out


def _screened_or_run(name: str, results: list[dict], queue: list[dict]) -> bool:
    return any(r.get("name", "").startswith(name) for r in results) or \
           any(e.get("name", "").startswith(name) for e in queue)


def freeform_backlog(results: list[dict], queue: list[dict]) -> list[dict]:
    """Whole-file candidates written to baseline/candidates/ and never screened."""
    if not CANDIDATES.exists():
        return []
    out = []
    for f in sorted(CANDIDATES.glob("*.py")):
        slug = "SCR_" + re.sub(r"[^A-Za-z0-9]+", "", f.stem)[:24]
        if _screened_or_run(slug, results, queue):
            continue
        out.append({"slug": slug, "cfg": {direction.FREEFORM_KEY: f.name},
                    "why": f"freeform candidate {f.name}, never screened"})
    return out


# Legacy inline branches in make_variant.build() predate the registry and so have no
# decorator to carry a screen_value. They are enumerated here because there is nowhere
# else to put it -- not because a hand-maintained table is the right home. A mechanism
# moved into the registry should declare `screen_value=` and drop out of this map.
_LEGACY_SCREEN_VALUES = {"precond": "pre"}


def _screen_value(m: str):
    """The value that screens mechanism `m`, taken from its own registration.

    Defaults to 1, which is right for a mechanism whose key is a plain on/off flag and
    wrong for one that validates its argument. Asking the mechanism is the only approach
    that stays correct as mechanisms are added by someone who is not editing this file.
    """
    entry = make_variant.MECHANISM_REGISTRY.get(m)
    if entry is not None:
        return entry.get("screen_value", 1)
    return _LEGACY_SCREEN_VALUES.get(m, 1)


def mechanism_backlog(results: list[dict], queue: list[dict]) -> list[dict]:
    """Registered mechanisms with zero completed runs.

    The registry spent this campaign advertising mechanisms nobody had ever run -- an
    audit found ten of them at one point. A mechanism that is implemented and never
    executed is the cheapest exploration available: the code exists, the diagnostic
    exists, and the only thing missing is 300 seconds.
    """
    ran = set()
    for r in results:
        ran |= direction.mechanisms_touched(r.get("cfg") or {})
    out = []
    for m in sorted(direction.all_mechanisms()):
        if m in ran or m not in make_variant.implemented():
            continue
        slug = "SCR_" + re.sub(r"[^A-Za-z0-9]+", "", m)[:24]
        if _screened_or_run(slug, results, queue):
            continue
        # THE PLATFORM BASE IS NOT OPTIONAL. make_variant.build() edits a full config;
        # `{mechanism: 1}` alone raises KeyError('dbs') at generation time. Isolating the
        # mechanism means holding every other coordinate at the reference platform, which
        # is what PLATFORM is for -- and it is also what makes the screen attributable to
        # the mechanism rather than to an accidental co-change.
        out.append({"slug": slug, "cfg": {**direction.PLATFORM, m: _screen_value(m)},
                    "why": f"registered mechanism {m!r} has zero completed runs"})
    return out


def build_screen(cand: dict, stamp: float) -> dict | None:
    """One queue entry, or None if the candidate cannot be generated.

    The build happens HERE, not on a GPU at 3am -- the same rule queue_from_round follows,
    for the same reason.
    """
    cfg = cand["cfg"]
    if direction.unknown_keys(cfg):
        return None
    try:
        src = make_variant.build(cfg)
    except Exception as exc:                       # noqa: BLE001 - any failure disqualifies
        print(f"  SKIP {cand['slug']}: variant build failed: {exc}")
        return None
    if src == (REPO / "baseline" / "train.py").read_text():
        print(f"  SKIP {cand['slug']}: generated source is byte-identical to the baseline")
        return None
    vid = make_variant.variant_id(src)
    (SWEEP / "variants").mkdir(parents=True, exist_ok=True)
    (SWEEP / "variants" / vid).write_text(src)
    return {
        "name": f"{cand['slug']}_scr",
        "cfg": cfg,
        "variant": vid,
        "role": "treat",
        # THE FLAG THE DISPATCHER READS. Width-1 and n=1 are enforced by construction:
        # no wave_group is emitted, so this entry can never be half of a yoked pair, and
        # the per-entry cutoff exemption that deadlocked the loop for controls is safe.
        "screen": True,
        "wave_group": None,
        "label": direction.label(cfg),
        "rationale": cand["why"],
        # An instrument probe, declared as such. A screen may not be cited for adoption;
        # promoting it means writing a round entry with a registered hypothesis and a
        # counterbalanced wave, which is the path that was always required.
        "hypothesis_id": "none",
        "falsifier": ("Screening only: this run decides nothing. It is promoted to a "
                      "counterbalanced comparison if it beats the running best by more "
                      "than the measured noise band, and dropped otherwise."),
        "expected": "",
        "created_at": stamp,
        "source_round": "explore_lane",
        "vram_est": 60,
    }


def main() -> int:
    width = fleet_width()
    target = round(width * DEPTH_PER_GPU)
    queue = json.loads((SWEEP / "queue.json").read_text()) if (SWEEP / "queue.json").exists() else []
    results = _results()
    have = len(pending(queue))
    print(f"explore lane: {have} pending / target {target} ({width} GPUs x {DEPTH_PER_GPU})")
    if have >= target:
        return 0

    backlog = freeform_backlog(results, queue) + mechanism_backlog(results, queue)
    if not backlog:
        # THE LOUD STATE. This is the one thing the lane cannot fix by itself.
        print("  CANDIDATE BACKLOG EMPTY -- the fleet will idle and this is the reason.")
        print("  Refill it either way:")
        print(f"    1. write a complete train.py into {CANDIDATES}/ (the open action")
        print("       space: any change, not just registry knobs)")
        print("    2. register a new mechanism in tools/mech_lib.py")
        return 0

    stamp = time.time()
    added = [e for e in (build_screen(c, stamp) for c in backlog[:target - have]) if e]
    if not added:
        print("  no buildable candidate in the backlog")
        return 0

    def _merge(cur: list[dict]) -> list[dict]:
        have_names = {e["name"] for e in cur}
        return cur + [e for e in added if e["name"] not in have_names]

    queue_store.update_queue(SWEEP / "queue.json", _merge)
    for e in added:
        print(f"  QUEUED SCREEN {e['name']}  {e['label']}  ({e['rationale']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
