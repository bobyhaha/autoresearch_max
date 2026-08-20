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
import claims           # noqa: E402
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
    # DEFAULT 2, ON MEASURED CAPACITY. 124 owner-filtered samples of runs/gpu_watch.log
    # over 2026-08-18T12:21Z..2026-08-20T03:29Z: a width-4 wave could have assembled in
    # 12.9% of them, a width-2 wave in 66.1% -- 5.1x as often -- and mean claimable
    # capacity (min(4, free+ours)) is 1.81 GPUs, not 4. A quad is the better DESIGN, since
    # the treatment occupies every slot within two waves and the slot profile cancels
    # exactly; but a held quad measures nothing at all, and this box holds quads roughly
    # seven times out of eight. Four 2-wide waves cancel the same offset across waves
    # rather than within one, which costs the host drift between them and buys a design
    # that actually runs. Pass --width 4 deliberately when the box is quiet.
    ap.add_argument("--width", type=int, default=2, choices=(2, 4),
                    help="GPUs per wave. 4 = two quads, treatment on every slot within a "
                         "wave. 2 = FOUR yoked pairs, treatment on each of the two devices "
                         "twice. Use 2 when the box will not free four GPUs at once: a "
                         "held quad measures nothing at all, and a pair whose swap runs on "
                         "the SAME two devices cancels the same device offset -- just "
                         "across waves rather than within one, so it also carries the host "
                         "drift between those waves.")
    ap.add_argument("--multifactor", default="", help=
                    "Required when the treatment differs from its control in more than "
                    "one key. State why the COMBINATION is the question, since the "
                    "within-pair delta will not be attributable to either factor alone.")
    a = ap.parse_args()

    P = dict(direction.PLATFORM)
    T = {**P, **json.loads(a.cfg)}
    if direction.is_platform(T):
        print("refusing: that cfg is the control")
        return 1

    # The SAME door as queue_from_round.py. A lesson that forbids a value is worth exactly
    # as much as the number of queue paths that honour it, and this one honoured none: the
    # value-level and key-level blocks lived only in the round path, so anything queued
    # directly here -- which is how every counterbalanced quad this campaign has run was
    # queued -- walked straight past the failures the campaign had already paid for. A
    # second entrance with no lock on it is not a second entrance, it is the way in.
    vhits = claims.blocked_values(T)
    if vhits:
        k, v, les, rule = vhits[0]
        print(f"refusing: {k}={v} is blocked by lesson {les['id']} (severity "
              f"{les['severity']}) -- it satisfies the forbidden rule {rule}.\n"
              f"  {les['mitigation'][:300]}")
        return 1
    blocked = claims.blocking_keys()
    # Only a key being CHANGED from the platform counts as engaging the lesson. This
    # intersected the whole cfg, and every cfg carries the PLATFORM's own keys -- so once
    # L076 blocked `mlp`, the platform's own mlp=4 tripped it and the door refused
    # EVERYTHING, including experiments that touch mlp not at all. A block on a key means
    # "do not run experiments on this key again", not "never queue anything".
    _delta = {k for k, v in T.items() if direction.PLATFORM.get(k) != v}
    hit = sorted(_delta & set(blocked))
    if hit:
        les = blocked[hit[0]]
        print(f"refusing: key '{hit[0]}' is blocked by lesson {les['id']} (severity "
              f"{les['severity']}).\n  {les['mitigation'][:300]}\n"
              f"  applies when: {les['applies_when']}")
        return 1
    # The diagnostic must be able to FAIL before the wave is spent. claims.py grew this
    # check after four hypotheses shipped with a diagnostic the control also satisfied --
    # and then nothing called it, so it was itself a check that could not fire, the exact
    # shape it exists to catch. An audit found it unwired. It runs here now, at the door,
    # where refusing costs nothing and passing costs a quad.
    if a.hyp != "none":
        _h = next((h for h in claims.hypotheses() if h["id"] == a.hyp), None)
        _act = (_h or {}).get("activation") or {}
        if _act.get("diagnostic") and _act.get("rule"):
            _ok, _msg = claims.diagnostic_would_discriminate(_act["diagnostic"], _act["rule"], T)
            if not _ok:
                print(f"refusing: hypothesis {a.hyp!r} declares an activation diagnostic "
                      f"that cannot demonstrate engagement.\n  {_msg}")
                return 1
            print(f"  activation pre-check: {_msg}")
            # Discriminating and EMITTED are independent. The z-loss arm passed the first
            # and failed the second, and eight runs were queued that could only ever have
            # come back as non-activations.
            _e_ok, _e_msg = make_variant.emits_diagnostic(T, _act["diagnostic"])
            if not _e_ok:
                print(f"refusing: hypothesis {a.hyp!r} declares a diagnostic the built "
                      f"variant never emits.\n  {_e_msg}")
                return 1
            print(f"  emission pre-check:   {_e_msg}")

    # TWO-FACTOR LINT. A pair whose two arms differ in more than one key cannot attribute
    # its delta, and this campaign has now made that mistake twice: L085 found the tbs=17
    # rung was really {dbs:64, tbs:17}, so its +0.004153 had been read as a pure
    # tokens-per-step effect for a full day and hardened into L083's block on the whole
    # `tbs` key; and R10WINB was queued as {win:SSSS, swdiv:16} against a plain platform
    # control within a day of that lesson being written. Prose did not stop the second one.
    #
    # Multi-factor arms are legitimate -- the three-lever stack is the campaign's strongest
    # result -- so this refuses by default and takes an explicit reason rather than a bare
    # flag. Writing down why you want two factors is the point; it is the sentence that was
    # missing both times.
    _delta_ctl = sorted(k for k, v in T.items() if P.get(k) != v)
    if len(_delta_ctl) > 1 and not a.multifactor:
        print(f"refusing: treatment differs from its control in {len(_delta_ctl)} keys "
              f"{_delta_ctl}, so a within-pair delta cannot be attributed to any one of "
              f"them (L085). Either yoke it to a control that isolates ONE factor -- e.g. "
              f"make the control carry the other key -- or pass "
              f"--multifactor 'why the combination is the question'.")
        return 1
    if len(_delta_ctl) > 1:
        print(f"  multi-factor arm ({len(_delta_ctl)} keys {_delta_ctl}): {a.multifactor}")

    unknown = direction.unknown_keys(T)
    if unknown:
        print(f"refusing: cfg key(s) {sorted(unknown)} are not recognised by the policy, "
              f"so the run would count toward no axis and no family and be invisible to "
              f"rotation and budgeting")
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
    if a.width == 4:
        layout = {f"{a.name}_A": ["t", "c", "c", "t"],
                  f"{a.name}_B": ["c", "t", "t", "c"]}
    else:
        # Four 2-wide waves. The dispatcher zips a wave's members onto the sorted free
        # GPUs, so member ORDER decides which device a role lands on: putting the
        # treatment first in half the waves and second in the other half puts it on each
        # device exactly twice. The outer T,C / C,T / C,T / T,C order also balances the
        # treatment against wave sequence, so a host that drifts monotonically over the
        # hour does not load onto the effect.
        layout = {f"{a.name}_P1": ["t", "c"], f"{a.name}_P2": ["c", "t"],
                  f"{a.name}_P3": ["c", "t"], f"{a.name}_P4": ["t", "c"]}
    # A WIDTH RE-SHAPE IS NOT A NEW DECISION. The decision cutoff freezes entries created
    # after the last council artifact, which is right for a newly proposed experiment and
    # wrong for one that was already proposed, reviewed and queued, and is merely being
    # re-cut from a 4-wide wave into 2-wide pairs because the box does not give four GPUs
    # (L052). Today that distinction cost real time: R5MU2 and R5MC2 -- mechanical
    # re-shapes of the already-approved R5MU and R5MC -- were frozen as new decisions, and
    # they are the very runs L047 gates tbs=18 adoption on. An audit measured roughly 400
    # free-GPU-minutes idled behind council staleness today, 68% of all gate evaluations.
    #
    # So an entry whose EXACT cfg and hypothesis have been queued before inherits the
    # earliest created_at already on record. The experiment keeps the review timestamp it
    # actually earned. A genuinely new cfg or a different hypothesis finds no match and is
    # frozen exactly as before, which is the case the cutoff exists for.
    # The key includes the STATED REASONING, not just the config. Keying on cfg and
    # hypothesis alone let an operator requeue an old reviewed pair under an entirely new
    # rationale, falsifier and expectation and still inherit the ancient timestamp -- an
    # audit demonstrated it live against a real queue entry. That is a council-freeze
    # bypass, and a worse one than the mechanical re-shape this inheritance exists for,
    # because the words a reviewer would actually read are exactly the ones that changed.
    #
    # A genuine width re-shape carries the same cfg, the same hypothesis AND the same
    # rationale, so it still inherits. Rewriting any of the prose makes it a new decision
    # and it is frozen until the next council, which is the correct answer.
    def _ikey(e):
        return (json.dumps(e.get("cfg"), sort_keys=True), e.get("hypothesis_id"),
                (e.get("rationale") or "").strip(), (e.get("falsifier") or "").strip(),
                (e.get("expected") or "").strip())
    _prior = {}
    for e in q:
        k = _ikey(e)
        ts = e.get("created_at")
        if ts and (k not in _prior or ts < _prior[k]):
            _prior[k] = ts

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
                 "wave_group": grp,
                 "created_at": _prior.get(
                     (json.dumps(T if role == "t" else P, sort_keys=True),
                      a.hyp if (role == "t" and a.hyp != "none") else None,
                      a.rationale.strip(), a.falsifier.strip(), a.expected.strip()), stamp),
                 "source_round": "quad-counterbalanced", "vram_est": 50}
            if role == "t" and a.hyp != "none":
                e["hypothesis_id"] = a.hyp
            new.append(e)
    qf.write_text(json.dumps(q + new, indent=1))
    print(f"queued {len(new)} entries in {len(layout)} wave(s) of {a.width}; treatment variant {vt}")
    for e in new:
        print(f"  {e['name']:34s} {e['label']:14s} wave={e['wave_group']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
