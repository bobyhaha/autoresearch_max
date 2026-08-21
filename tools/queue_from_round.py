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
import admission          # noqa: E402
import claims             # noqa: E402
import direction          # noqa: E402
import make_variant       # noqa: E402
import queue_store        # noqa: E402

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
    stamp = path.stat().st_mtime
    round_entries = council.queue_entries(path.read_text())
    comparison_ids = {}
    for group, members in admission.grouped(round_entries).items():
        comparison_ids[group] = admission.comparison_group(members)
    expected_per_wave = {group: len(members)
                         for group, members in admission.grouped(round_entries).items()}
    for e in round_entries:
        cfg = e["cfg"]
        # A key no policy rule covers counts toward no axis and no family, so it would be
        # invisible to budgeting and rotation. is_platform() already refuses to call it a
        # control; refusing it at the door as well keeps the accounting total.
        # Value-level blocks first: a lesson may forbid a RANGE of a key without
        # forbidding the key, which is the common case (ns>=5 is a no-op, ns 1..4 is a
        # real experiment). A key-level block would also refuse the legitimate arms.
        # Same pre-check as queue_quad: a hypothesis whose diagnostic the control also
        # satisfies produces a run that cannot tell a null from a no-op, and this door is
        # the last place that costs nothing to catch.
        _hid = e.get("hypothesis_id")
        if _hid and _hid != "none":
            _h = next((h for h in claims.hypotheses() if h["id"] == _hid), None)
            _act = (_h or {}).get("activation") or {}
            if _act.get("diagnostic") and _act.get("rule"):
                _ok, _msg = claims.diagnostic_would_discriminate(
                    _act["diagnostic"], _act["rule"], cfg)
                if not _ok:
                    skipped.append((e["name"], f"activation diagnostic cannot fire: {_msg}"))
                    continue
                # Discriminating and EMITTED are independent properties. A hypothesis can
                # name a perfectly discriminating observable that the generated code never
                # prints, which yields a run that is a non-activation by construction.
                _e_ok, _e_msg = make_variant.emits_diagnostic(cfg, _act["diagnostic"])
                if not _e_ok:
                    skipped.append((e["name"],
                                    f"declared diagnostic is never emitted: {_e_msg}"))
                    continue

        vhits = claims.blocked_values(cfg)
        if vhits:
            k, v, les, rule = vhits[0]
            skipped.append((e["name"], f"blocked by lesson {les['id']} (severity "
                                       f"{les['severity']}): {k}={v} satisfies the "
                                       f"forbidden rule {rule}. {les['mitigation'][:160]}"))
            continue
        # Only a CHANGED key engages a key-level block. Intersecting the whole cfg means
        # the PLATFORM's own keys trip it: once L076 blocked `mlp`, the platform's mlp=4
        # refused every entry, including ones that do not touch mlp. Same defect as the
        # other door; fixed in both, because a second entrance with a different lock is
        # the way in.
        _delta = {k for k, v in cfg.items() if direction.PLATFORM.get(k) != v}
        hit = sorted(_delta & set(blocked_by_lesson))
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
        # A treatment must cite a registered hypothesis. Without one, coe.py E3 skips the
        # result (`hid = r.get("hypothesis_id"); if not hid: continue`) and the activation
        # predicate never runs -- so a null cannot be separated from an intervention that
        # never engaged, which is the single thing the predicate exists to establish. This
        # was prose in the protocol and nothing checked it: wd_const=0.4 reached a GPU
        # with hypothesis_id null. Controls are exempt; they are the instrument, not a
        # claim. A deliberate instrument probe may opt out with "hypothesis_id": "none".
        if direction.recorded_role(e) == "treat" and not e.get("hypothesis_id"):
            skipped.append((e["name"], "no hypothesis_id: a treatment must cite a "
                                       "registered hypothesis, or its activation predicate "
                                       "never runs and a null is indistinguishable from "
                                       "'never engaged'. Register one with claims.py hyp, "
                                       "or set hypothesis_id to 'none' to declare it an "
                                       "instrument probe rather than a research arm."))
            continue
        if (e.get("hypothesis_id") or "none") != "none":
            known = {h["id"] for h in claims.hypotheses()}
            if e["hypothesis_id"] not in known:
                skipped.append((e["name"], f"cites unregistered hypothesis "
                                           f"'{e['hypothesis_id']}'"))
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
        vid = make_variant.variant_id(src)
        (SWEEP / "variants" / vid).write_text(src)
        # ROLE RECORDED AT QUEUE TIME -- see the note in tools/queue_quad.py. A round
        # entry names its members <wave>_s<slot>_<role>, so the role is known here; this
        # writes it down instead of leaving it to be re-derived against a platform that
        # moves. Falls back to the name only if a round ever omits the suffix.
        _role = direction.recorded_role(e)
        queue.append({"name": e["name"], "cfg": cfg, "variant": vid,
                      **({"role": _role} if _role else {}),
                      "label": direction.label(cfg), "rationale": e["rationale"],
                      "falsifier": e["falsifier"], "expected": e.get("expected", ""),
                      # Entries sharing a wave_group launch CONCURRENTLY on separate
                      # GPUs or not at all. This is how a yoked pair is obtained: two
                      # runs launched half an hour apart measure host drift, not effect.
                      "wave_group": e.get("wave_group"),
                      # Swapped waves in the same comparison must reuse the same physical
                      # GPU UUIDs.  The dispatcher persists this key after the first wave.
                      "counterbalance_group": comparison_ids[e["wave_group"]],
                      # Carry the hypothesis forward. The door check above validates
                      # e["hypothesis_id"] on the ROUND entry, but the persisted queue
                      # entry was built without the field -- so 16 validated round-2 runs
                      # were queued attached to nothing, and coe.py E3, E4 and verdict.py
                      # all skip on `if not hid`. The check passed and the linkage was
                      # dropped one line later.
                      **({"hypothesis_id": e["hypothesis_id"]}
                         if e.get("hypothesis_id") and e["hypothesis_id"] != "none" else {}),
                      "created_at": stamp, "source_round": path.name,
                      "vram_est": e.get("vram_est", 60)})

    # L006_slot_bias_fakes_effects, enforced rather than remembered. Slot 0 loses to slot
    # 1 by ~0.00047 bpb in every control wave measured, which is the size of the effects
    # being hunted -- so a treatment that appears in only ONE wave carries that offset as
    # a fake result whose sign depends only on where it landed. The dispatcher assigns
    # slots in queue order within a wave, so a counterbalanced treatment must appear both
    # before and after a control across its two waves.
    # Never leave an orphan control behind when one member of a wave fails a late build
    # check.  A yoked wave is the admission unit, not a bag of independently valid rows.
    actual_per_wave = {}
    for entry in queue:
        actual_per_wave[entry["wave_group"]] = actual_per_wave.get(entry["wave_group"], 0) + 1
    incomplete = {group for group, expected in expected_per_wave.items()
                  if actual_per_wave.get(group, 0) != expected}
    for group in sorted(incomplete):
        skipped.append((group, "entire wave dropped because at least one member was invalid"))
    queue = [entry for entry in queue if entry["wave_group"] not in incomplete]

    slots = {}
    for e in queue:
        if direction.recorded_role(e) == "ctrl":
            continue
        grp = [x for x in queue if x.get("wave_group") == e.get("wave_group")]
        slots.setdefault(json.dumps(e["cfg"], sort_keys=True), []).append(
            grp.index(e) if e in grp else -1)
    for cfgkey, positions in slots.items():
        if len(set(positions)) < 2:
            nm = next(e["name"] for e in queue
                      if json.dumps(e["cfg"], sort_keys=True) == cfgkey)
            skipped.append((nm, f"NOT COUNTERBALANCED: this cfg occupies slot "
                                f"{positions[0]} in every wave it appears in "
                                f"({len(positions)} wave(s)). L006 measured a fixed "
                                f"+0.00047 bpb slot offset, comparable to the effects "
                                f"being tested, so an uncounterbalanced arm reports that "
                                f"offset as its result. Queue it twice with the slot "
                                f"order swapped: [treatment, control] and "
                                f"[control, treatment]."))
    bad_cfgs = {cfgkey for cfgkey, positions in slots.items() if len(set(positions)) < 2}
    if bad_cfgs:
        bad_groups = {
            entry["wave_group"] for entry in queue
            if direction.recorded_role(entry) == "treat"
            and json.dumps(entry["cfg"], sort_keys=True) in bad_cfgs
        }
        queue = [entry for entry in queue if entry["wave_group"] not in bad_groups]

    qf = SWEEP / "queue.json"
    new = []
    existing_count = 0

    def merge(existing):
        nonlocal new, existing_count
        existing_count = len(existing)
        have = {item["name"] for item in existing}
        new = [item for item in queue if item["name"] not in have]
        return existing + new

    queue_store.update_queue(qf, merge)

    print(f"{path.name}: queued {len(new)} new ({len(queue)} valid, "
          f"{existing_count} already present)")
    for n, w in skipped:
        print(f"  skipped {n}: {w}")
    return 0 if queue else 1


if __name__ == "__main__":
    sys.exit(main())
