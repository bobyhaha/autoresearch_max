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


def _is_ctl(m):
    """Was this member its wave's CONTROL? Answered from the wave, not from the platform.

    Resolving the role through direction.is_platform means a PLATFORM ADOPTION retroactively
    orphans every completed wave whose control was built at the old baseline. Adopting
    tbs=18 collapsed this report from many groups to ONE and dropped R5MU2, R5MC2 and
    R5NOQK2 -- three counterbalanced pairs each, all already reported.

    A first repair replaced the role check at the ctl/trt split and appeared to do nothing.
    It did nothing because there are THREE role sites, and the pairing loop's `ac == bc`
    test re-derived the role from is_platform a few lines later and dropped every wave
    again. An independent method audit traced it and named all three; this is the single
    helper they now share.

    A wave is a self-contained comparison and carries its own baseline. queue_quad names
    members <wave>_s<slot>_<role>, so the role is in the name and survives any later
    platform change. is_platform remains the fallback for records predating the convention.
    """
    mo = _ROLE_RE.search(m.get("name", "") or "")
    if mo:
        return mo.group(1) != "treat"
    return direction.is_platform(m.get("cfg") or {})


_ROLE_RE = __import__("re").compile(r"_s\d+_(treat|ctrl|control)$")


def _variant_of(name, cfg=None):
    """Which generated source a run actually executed.

    Read from the queue when the entry is still there, and otherwise REBUILT from the run's
    own cfg. A run whose queue entry was cut after it completed used to resolve to "?" and
    then group separately from its own wave-mates -- so recovering the run from its name
    was not enough to put it back beside them, and MTP appeared as two verdict blocks, one
    of them a single arm with no verdict. The cfg is in the result record, and the variant
    is a pure function of the cfg, so nothing about the identification depends on the queue.
    """
    try:
        for e in json.loads((REPO / "runs" / "sweep" / "queue.json").read_text()):
            if e["name"] == name:
                return (e.get("variant") or "?")[:12]
    except (OSError, ValueError):
        pass
    if cfg:
        try:
            import hashlib
            import make_variant
            return hashlib.sha256(
                make_variant.build(dict(cfg)).encode()).hexdigest()[:12]
        except Exception:
            pass
    return "?"


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
    # THE RESULT IS THE RECORD, NOT THE QUEUE. This resolved a run's wave ONLY through
    # queue.json, so cutting a queue entry after that run had already completed silently
    # deleted it from every verdict. It happened: R6MTP_P3 ran, its result was written, the
    # queue entry was then cut as redundant, and verdict.py quietly went back to reporting
    # two MTP pairs while the campaign said three. A reproducibility audit found the
    # discrepancy from outside; nothing in the tool announced it.
    #
    # The wave is now taken from the result if it carries one, then from the queue, and
    # finally from the run NAME -- queue_quad builds names as <wave>_s<slot>_<role>, so the
    # wave is recoverable from the name alone and a cut entry can no longer erase evidence.
    import re as _re
    qmap = {}
    try:
        for e in json.loads((REPO / "runs" / "sweep" / "queue.json").read_text()):
            if e.get("wave_group"):
                qmap[e["name"]] = e["wave_group"]
    except (OSError, ValueError):
        pass
    out, orphans = {}, []
    for r in rows:
        g = r.get("wave_group") or qmap.get(r["name"])
        if not g:
            m = _re.match(r"^(.*)_s\d+_(?:treat|ctrl|control)$", r["name"])
            if m:
                g = m.group(1)
                orphans.append(r["name"])
        if g:
            out.setdefault(g, []).append(r)
    if orphans:
        print(f"  NOTE {len(orphans)} result(s) had no queue entry and were recovered from "
              f"their names: {', '.join(sorted(orphans)[:4])}"
              f"{' ...' if len(orphans) > 4 else ''}. A completed run whose queue entry was "
              f"cut is still evidence.")
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
    dr = direction.device_resolution(rows)
    if dr:
        res = dr["resolution"]
        print(f"device model: pooled within-GPU sd {dr['pooled_sd']:.6f} over "
              f"{dr['n_devices']} GPUs -> GPU-COUNTERBALANCED RESOLUTION {res:.6f} (L020)")

    hyps = {h["id"]: h for h in C.hypotheses()}
    by_cfg = {}
    for g, members in waves(rows).items():
        ctl = [m for m in members if _is_ctl(m)]
        trt = [m for m in members if not _is_ctl(m)]
        if not ctl or not trt:
            continue
        # A wave may be 2-wide (one pair) or 4-wide (two pairs). Pair members by ADJACENT
        # GPU index -- gpu4 with gpu5, gpu6 with gpu7 -- which is how the dispatcher fills
        # slots and therefore which runs actually shared their moment of host contention.
        # Requiring exactly one control and one treatment silently DROPPED every quad wave,
        # so the counterbalanced design this campaign switched to produced no verdict at all.
        members_sorted = sorted(members, key=lambda m: m.get("gpu", 0))
        # An ODD member count means the last entry is never paired. The loop steps by two
        # from index 0, so `range(0, n-1, 2)` simply never reaches it -- and it did so
        # SILENTLY, which is the problem: an odd count arises exactly when a wave has lost
        # a member to co-tenancy, a crash, or a VOID exclusion, i.e. when a quietly
        # discarded run is most likely to matter. The ve arm reported n=2 from three
        # usable members for this reason. Pairing adjacent-by-GPU is still correct for the
        # quad layout; what was wrong was doing it without saying what fell off the end.
        if len(members_sorted) % 2:
            _orphan = members_sorted[-1]
            print(f"  NOTE wave {g}: {len(members_sorted)} members is odd, so "
                  f"{_orphan['name']} on {_slot(_orphan)} has no adjacent partner and "
                  f"takes no part in any delta. Its data is not lost, only unpaired -- "
                  f"re-run its wave-mate to recover the cell.")
        for i in range(0, len(members_sorted) - 1, 2):
            a, b = members_sorted[i], members_sorted[i + 1]
            ac, bc = _is_ctl(a), _is_ctl(b)
            if ac == bc:
                continue                      # not a treatment/control pair
            c, t = (a, b) if ac else (b, a)
            # Key on the VARIANT as well as the label. `mech:precond` names the same
            # mechanism before and after its repair, but the pre-repair runs never engaged
            # (secmom_clamp_frac 0.998) and pooling them with the repaired ones reported
            # "no effect demonstrated" for a mechanism whose fixed version had not yet been
            # given a verdict. A repaired implementation is a different experiment.
            key = (f"{direction.label(t['cfg'])}  "
                   f"[variant {t.get('variant') or _variant_of(t['name'], t.get('cfg'))}]")
            by_cfg.setdefault(key, []).append({
                "wave": g, "delta": t["metrics"]["val_bpb"] - c["metrics"]["val_bpb"],
                "treat_dev": _slot(t), "ctl_dev": _slot(c), "t": t, "c": c})

    print()
    for key, arms in sorted(by_cfg.items()):
        if want and want not in key:
            continue
        print(f"=== {key} ===")
        # VOID arms are REMOVED, not merely annotated. Printing "VOID: different
        # final_epoch" beside a delta that then went straight into the counterbalanced mean
        # was a warning with no teeth: a run that crossed the 2-epoch boundary is a
        # different operating point (L005_operating_point_moved_v2), and averaging it in
        # imports that regime difference into the effect. Observed live on R2V_A_s0_treat,
        # which finished at 501.2M tokens and epoch 1.0 while its wave-mates cleared the
        # boundary at 520M+. Excluding it also BREAKS the quad's "treatment on every
        # device" property, so the design degrades to something weaker -- which the reader
        # must be told rather than left to infer from an unchanged verdict line.
        voided = [a for a in arms
                  if a["t"]["metrics"].get("final_epoch") != a["c"]["metrics"].get("final_epoch")]
        for a in arms:
            tm, cm = a["t"]["metrics"], a["c"]["metrics"]
            print(f"  {a['wave']:8s} treat_dev={a['treat_dev']:>8s} "
                  f"delta {a['delta']:+.6f}  "
                  f"steps {tm['num_steps']:.0f} vs {cm['num_steps']:.0f}  "
                  f"epoch {tm.get('final_epoch')}/{cm.get('final_epoch')}")
            if tm.get("final_epoch") != cm.get("final_epoch"):
                print("    VOID -- EXCLUDED: treatment and control finished at different "
                      "final_epoch (L005_operating_point_moved_v2); a regime comparison, "
                      "not a result, so it takes no part in the mean below")
        # SPORADIC vs SYSTEMATIC. The rule above was written for R2V_A_s0_treat, where ONE
        # arm crossed the 2-epoch boundary and its wave-mates did not: a contention-caused
        # shift in one cell, which is a genuine confound and must be excluded.
        #
        # It is the wrong call when EVERY arm shows the same directional gap, because then
        # the regime change is not noise that happened to land on one cell -- it is what the
        # treatment DOES. MTP finished at epoch 1 against control epoch 2 in both waves, on
        # swapped slots, with deltas agreeing to 4e-4 (+0.079681, +0.079288); it is 2.8x
        # slower per step and cannot reach the boundary inside 300 seconds. Voiding that
        # reports NO VERDICT for a treatment whose cost is enormous, reproducible, and
        # precisely the thing being measured.
        #
        # CLAUDE.md settles which reading governs: "Raw val_bpb is the verdict. A treatment
        # that costs throughput is genuinely worse at a fixed 300s budget; that cost is the
        # finding, not a nuisance term." Suppressing the verdict here contradicted the
        # campaign's own measurement principle.
        #
        # So: a mismatch seen in EVERY arm, in the same direction, with at least two arms to
        # tell systematic from sporadic, is REPORTED with the regime difference stated in
        # full. Anything less consistent is still excluded exactly as before.
        # MAGNITUDE GUARD. Consistency of DIRECTION is not evidence of CAUSE. An audit
        # built a fixture where a delta smaller than the noise band, plus a ~1% step-count
        # wobble that happened to straddle the epoch boundary in both waves, was reported
        # with full confidence as "the epoch gap is caused BY the treatment" -- a false
        # causal claim manufactured out of a coin landing the same way twice.
        #
        # A treatment that genuinely cannot reach the boundary has moved throughput a LOT:
        # MTP ran 371 steps against 1011, a 63% departure. A run that merely wobbled across
        # the line has moved it by a percent or two. So require BOTH a large throughput
        # change and an effect the instrument can actually resolve; anything less is
        # excluded as before, which is the fail-safe direction.
        _ratios = []
        for a in arms:
            _ts, _cs = a["t"]["metrics"].get("num_steps"), a["c"]["metrics"].get("num_steps")
            if _ts and _cs:
                _ratios.append(abs(1.0 - (_ts / _cs)))
        _big_throughput = bool(_ratios) and min(_ratios) >= 0.10
        _mean_delta = sum(a["delta"] for a in arms) / len(arms) if arms else 0.0
        _resolvable = abs(_mean_delta) > max(res, band or 0.0)
        systematic = (len(arms) >= 2 and len(voided) == len(arms) and
                      len({(a["t"]["metrics"].get("final_epoch") <
                            a["c"]["metrics"].get("final_epoch")) for a in arms}) == 1
                      and _big_throughput and _resolvable)
        if voided and systematic:
            lo = arms[0]["t"]["metrics"].get("final_epoch")
            hi = arms[0]["c"]["metrics"].get("final_epoch")
            print(f"  SYSTEMATIC REGIME SHIFT, NOT VOID: every arm finished at treatment "
                  f"epoch {lo} against control epoch {hi}, on swapped slots. The epoch gap "
                  f"is caused BY the treatment rather than by contention landing on one "
                  f"cell, so it is the result and not a confound -- the treatment is too "
                  f"slow to reach the boundary in the fixed budget. Reported below with "
                  f"that difference stated; read the step counts, which are the mechanism.")
            voided = []
        if voided:
            arms = [a for a in arms if a not in voided]
            print(f"  {len(voided)} arm(s) excluded as VOID; {len(arms)} usable. The "
                  f"counterbalancing is now INCOMPLETE -- the treatment no longer occupies "
                  f"every device once, so a device offset survives in the mean and the "
                  f"result below is weaker than a full quad. Re-run the excluded cell.")
        if not arms:
            print("  no usable arms after VOID exclusion -- NO VERDICT\n")
            continue
        deltas = [a["delta"] for a in arms]
        # A QUAD-COUNTERBALANCED design (tools/queue_quad.py) puts the treatment on every
        # slot once and the control on every slot once across two four-wide waves, so the
        # whole slot profile cancels in the mean of the within-pair deltas -- no pairwise
        # swap is needed and no offset is fitted. Detect it and report it as one verdict.
        slots_t = {a["treat_dev"] for a in arms}
        slots_c = {a["ctl_dev"] for a in arms}
        # Prefer SAME-GPU pairing whenever at least two devices ran both roles, not only
        # when a full four-device quad survives. The fallback path below pairs within a
        # wave and then discards any GPU-pair that is not internally counterbalanced, so
        # after a VOID exclusion the ve arm reported a mean over TWO deltas while three
        # same-GPU pairs existed -- and the paper, which uses same-GPU pairing, certified a
        # different number than the tool did. Two defensible estimators disagreeing by
        # 0.00006 is survivable; a tool and a paper disagreeing about the same arm is not.
        byg_t_all = {a["treat_dev"]: a["t"] for a in arms}
        byg_c_all = {a["ctl_dev"]: a["c"] for a in arms}
        if len(arms) >= 2 and len(set(byg_t_all) & set(byg_c_all)) >= 2:
            # Re-pair SAME-GPU across the two waves before averaging. Pairing within a
            # wave leaves the whole device profile inside each delta -- gpu4 is the slow
            # device and gpu7 the fast one, a 0.0025 spread as large as any effect -- so
            # the deltas scatter ~6x more than the instrument's real noise and a mean of
            # them fails its own t-test even when the effect is strong. Same-GPU pairing
            # removes the device term inside each difference instead of relying on it to
            # cancel in the mean. Both give the same mean; only this one has a usable sd.
            byg_t = {a["treat_dev"]: a["t"] for a in arms}
            byg_c = {a["ctl_dev"]: a["c"] for a in arms}
            shared = sorted(set(byg_t) & set(byg_c))
            if len(shared) >= 2:
                paired = [byg_t[g]["metrics"]["val_bpb"] - byg_c[g]["metrics"]["val_bpb"]
                          for g in shared]
                mean = st.mean(paired)
                sd = st.stdev(paired) if len(paired) > 1 else 0.0
                sem = sd / (len(paired) ** 0.5) if sd else 0.0
                print(f"  SAME-GPU PAIRED over {len(shared)} devices {shared}: "
                      f"deltas {[round(x, 6) for x in paired]}")
                # BOTH PAIRINGS, ALWAYS, WITH THE DEGREES OF FREEDOM. Re-pairing same-GPU
                # across waves removes the device term from the VARIANCE, which shrinks sd
                # and inflates t -- for R5MU2 from sd 0.000391 to 0.000117 and t from -26.9
                # to -90.2. Printing only the flattering one is how it gets quoted, and it
                # was quoted, twice: z-loss as t=2110 (raw 683.9) and R5MU2 as t=-90.2. The
                # second happened an hour AFTER a lesson recorded the first, which is the
                # evidence that a lesson does not bind behaviour and a print statement does.
                #
                # df is printed because at two pairs it is 1, and a t on one degree of
                # freedom is arithmetic rather than evidence however large it looks.
                _raw = [a["delta"] for a in arms]
                _rm = st.mean(_raw)
                _rsd = st.stdev(_raw) if len(_raw) > 1 else 0.0
                _rsem = _rsd / len(_raw) ** 0.5 if _rsd else 0.0
                print(f"    same-GPU re-paired: mean {mean:+.6f}  sd {sd:.6f}  "
                      f"sem {sem:.6f}"
                      + (f"  t={mean/sem:.1f} (df={len(paired)-1})" if sem else ""))
                print(f"    raw within-wave:    mean {_rm:+.6f}  sd {_rsd:.6f}  "
                      f"sem {_rsem:.6f}"
                      + (f"  t={_rm/_rsem:.1f} (df={len(_raw)-1})" if _rsem else ""))
                if len(paired) <= 2:
                    print("    NOTE df<=1: quote the effect against the measured band, "
                          "not a t computed on one degree of freedom.")
                # STEP-LAW SHARE, with the extrapolation stated. direction.step_law_explains
                # returned an `extrapolated` flag that NO caller consulted -- built and not
                # connected, the failure this campaign has hit seven times. It is consulted
                # here. The law is fitted on controls spanning 621-1020 steps, so an arm
                # that ran 303 is outside it, and a share read off an extrapolation must say
                # so rather than be quoted as though interpolated.
                try:
                    _sl = direction.step_law_explains(rows, arms[0]["t"], arms[0]["c"])
                except Exception:
                    _sl = None
                if _sl:
                    _mark = "  [EXTRAPOLATED beyond the fitted step range]" if _sl.get(
                        "extrapolated") else ""
                    print(f"    step law explains {100*_sl['share']:.0f}% "
                          f"(predicted {_sl['predicted']:+.6f}, residual "
                          f"{_sl['residual']:+.6f}){_mark}")
            else:
                mean = st.mean(a["delta"] for a in arms)
            # Threshold at the n ACTUALLY measured, not the n=4 constant. Excluding a
            # void cell drops n and makes the fixed constant too lenient by ~19%.
            n_eff = len(paired) if len(shared) >= 2 else len(arms)
            res_n = (dr["resolution_at"](n_eff)
                     if dr and dr.get("resolution_at") else res)
            # TWO thresholds, because they answer different questions and can disagree.
            #
            # The CONTROL-DERIVED one assumes treatment and control vary independently,
            # so SD(delta) = sqrt(2)*sigma. Measured against the actual deltas that is
            # wrong in the CONSERVATIVE direction: observed paired sd runs 0.28x to 0.93x
            # that expectation, because arms in one wave share the host contention that
            # dominates this benchmark and it cancels in the difference. The assumption
            # ignores within-wave covariance, which is exactly the objection.
            #
            # The PAIRED one uses the observed spread of the deltas themselves, which
            # captures that covariance -- but on n=4 it carries 3 df and is itself noisy,
            # and a run of luck that shrinks it inflates t.
            #
            # So: the headline verdict uses the conservative threshold, the paired
            # statistic is reported beside it, and a DISAGREEMENT is announced rather
            # than resolved silently in whichever direction suits the arm.
            paired_se = (sd / (n_eff ** 0.5)) if sd else 0.0
            res_paired = 2 * paired_se
            verdict = ("BETTER than control" if mean < -res_n else
                       "WORSE than control" if mean > res_n else
                       "INSIDE the resolution -- no effect demonstrated")
            full = len(slots_t) >= 4 and slots_t == slots_c
            print(f"  {'QUAD-COUNTERBALANCED' if full else 'SAME-GPU PAIRED (INCOMPLETE QUAD)'}"
                  f" over {len(arms)} pairings: mean delta {mean:+.6f} vs resolution {res_n:.6f} "
                  f"(2*SE at n={n_eff}, control-derived) -> {verdict}")
            if res_paired:
                agree = (abs(mean) > res_paired) == (abs(mean) > res_n)
                print(f"    paired-variance check: 2*SE from the observed deltas is "
                      f"{res_paired:.6f} ({paired_se/((res_n/2) or 1):.2f}x the "
                      f"control-derived SE)"
                      + ("" if agree else
                         "  <-- DISAGREES with the headline; the conservative threshold "
                         "governs and this arm needs more pairings"))
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
        # Same correction as the quad branch: threshold at the n actually in hand. This
        # branch is where a VOID-reduced arm lands, so it is exactly the case the fixed
        # n=4 constant judges too leniently -- ve was read here at n=3, where the constant
        # is 1.7*SE rather than 2*SE.
        res_n = (dr["resolution_at"](len(deltas))
                 if dr and dr.get("resolution_at") else res)
        verdict = ("BETTER than control" if mean < -res_n else
                   "WORSE than control" if mean > res_n else
                   "INSIDE the resolution -- no effect demonstrated")
        print(f"  counterbalanced mean delta {mean:+.6f}  vs resolution {res_n:.6f} "
              f"(2*SE at n={len(deltas)})  -> {verdict}")
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
