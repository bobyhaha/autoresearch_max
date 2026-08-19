#!/usr/bin/env python3
"""Which experiment is worth running next -- as a computed score, not a judgement call.

Two independent external reviews reached the same verdict about this campaign: it is far
better at answering "can I trust this result?" than "which experiment should I run next?".
That asymmetry is real and it is structural. Everything upstream of the GPU -- the
counterbalanced design, the activation predicates, the chain-of-evidence audit, the
refusal to give a verdict when a swap breaks -- exists to judge results already in hand.
What chooses the NEXT arm is a set of guardrails: rotate families, keep an explore floor,
close an axis after N distinct values fail, obey lesson blocks. Guardrails forbid bad
choices; they do not rank good ones. Between two admissible hypotheses the tie was broken
by the operator, which is not reproducible and not recorded.

This file is the smallest thing that closes that gap honestly. It is NOT a learned world
model or a Bayesian surrogate -- claiming either would be the same overreach the campaign
keeps catching elsewhere. It is a transparent scoring function over features that already
exist in `runs/sweep/results` and `lit/`, with every term's weight visible and its
provenance stated, plus greedy diversity so a batch is not five variations of one idea.

    python3 tools/selector.py                    # rank everything currently queued
    python3 tools/selector.py --batch 3          # pick a diverse batch of 3
    python3 tools/selector.py --cfg '{"mlp":2}'  # score one candidate

WHAT IT DELIBERATELY DOES NOT DO. It does not estimate a posterior, and it does not
pretend the weights are fitted -- they are not, and a heuristic that has never been fit to
anything cannot carry a calibrated threshold. So the output is a RANKING, never a
pass/fail gate. That distinction is the whole lesson of this campaign's confidence-score
critique: use it to order what you can afford to run, not to admit or refuse.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import analyze      # noqa: E402
import claims       # noqa: E402
import direction    # noqa: E402

# Per-quad cost: two four-wide waves at roughly eight minutes a run.
GPU_MIN_PER_QUAD = 8 * 8

# Weights. Stated, not fitted. Chosen so that a confirmed-family exploit and an untouched
# family come out comparable at equal cost -- if one systematically dominates, that is a
# bug in the weights and the ranking should be read sceptically until they are revised.
W_GAIN, W_INFO, W_NOVEL, W_RESOLVE, W_ACTPEN = 1.0, 0.6, 0.5, 4.0, 8.0


def _changed(cfg):
    """Keys this cfg actually MOVES off the platform.

    Attribution must be by what a config CHANGES, not by what keys it contains. Every cfg
    carries the full platform -- depth, dim, mlp, tbs, dbs, ve, win, swdiv -- so testing
    `set(cfg) & family_axes` matched capacity, token_exposure, attention and ve_placement
    for literally every experiment, including an mtp-only candidate that touches none of
    them. That produced 196 family-run assignments over 125 valid runs, and every term
    built on it -- expected gain, information, diversity -- was computed from other
    families' evidence. The bug was visible as "spread over 188 runs" and was noted as odd
    and not chased; an external review reproduced it precisely.
    """
    return {k: v for k, v in cfg.items() if direction.PLATFORM.get(k) != v}


def _families(cfg):
    ch = _changed(cfg)
    return [f for f, spec in direction.FAMILIES.items()
            if set(ch) & set(spec.get("axes", ())) or ch.get(f)]


def _device_means(rows):
    # Delegates to direction.device_means: one canonical implementation, so a
    # change to what counts as a control cannot silently apply here and not there.
    return direction.device_means(rows)


def family_effects(rows):
    """Device-corrected effect per FAMILY, with its spread. The spread is the point:
    a family measured once and a family measured six times can share a mean while
    differing entirely in what another run would teach."""
    dm = _device_means(rows)
    out = {}
    for r in rows:
        cfg = r.get("cfg") or {}
        if not r.get("ok") or direction.is_platform(cfg):
            continue
        # A run whose config is now BLOCKED contributes no information about what to try
        # next: the campaign will not run that value again. tbs carries the +0.022545
        # tbs=20 catastrophe, which is blocked by L040 and yet dominated the family's
        # spread -- making every tbs arm look like the richest available source of
        # information precisely because of a result already understood and forbidden.
        # Switching to a robust spread reduced the distortion; excluding blocked runs
        # removes its cause.
        if claims.blocked_values(cfg):
            continue
        eff = r["metrics"]["val_bpb"] - dm.get(r.get("gpu"), r["metrics"]["val_bpb"])
        fams = _families(cfg)
        # A run that moves keys in SEVERAL families measures their JOINT effect, and that
        # joint effect is not attributable to any one of them. Appending the full delta to
        # each family -- which this did -- credits every factor with the whole stack.
        #
        # It was not hypothetical. The three-lever arm R4X_A_s3_treat (ve=1, swdiv=4,
        # precond=pre) put its entire -0.004355 into BOTH `attention` and `ve_placement`,
        # where it stood as each family's largest effect. The real single-factor swdiv
        # arms measure -0.002298 (2->4) and -0.003360 (2->8): the inflated figure was
        # roughly double the truth, it set the scoring prior for 12+ queued runs, and it
        # reached round 6's provenance block labelled "best swdiv effect across 16 runs".
        # L042 had already measured this stack at 86% of additive, so the campaign knew
        # the factors were non-additive at the moment it was crediting each with the sum.
        #
        # A multi-family arm now scores under its own combination key. Nothing is
        # discarded -- a stack is real evidence about the stack -- but it can no longer
        # masquerade as evidence about one of its parts.
        key = fams[0] if len(fams) == 1 else "stack:" + "+".join(sorted(fams))
        if fams:
            out.setdefault(key, []).append(eff)
    return out


def score(cfg, rows, state, fx):
    """Return (score, terms). A blocked or unbuildable cfg scores -inf and says why."""
    terms = {}
    blocked = claims.blocked_values(cfg)
    if blocked:
        k, v, les, _ = blocked[0]
        return -math.inf, {"BLOCKED": f"{k}={v} by {les['id']}"}
    why = direction.blocked_reason(cfg, state)
    if why:
        return -math.inf, {"BLOCKED": why}
    unknown = direction.unknown_keys(cfg)
    if unknown:
        return -math.inf, {"BLOCKED": f"unrecognised keys {sorted(unknown)}"}

    fams = _families(cfg)
    obs = [e for f in fams for e in fx.get(f, [])]

    # EXPECTED GAIN. A family that has paid before is likely to pay again, so use its best
    # device-corrected effect. An untouched family gets the campaign's MEDIAN confirmed
    # effect rather than its best -- assuming a new idea will match the best result so far
    # is precisely the optimism that makes an exploit-only search look attractive.
    if obs:
        gain = max(0.0, -min(obs))
        terms["gain"] = f"best observed in family {min(obs):+.6f}"
    else:
        confirmed = [min(v) for v in fx.values() if v and min(v) < 0]
        gain = abs(st.median(confirmed)) if confirmed else 0.002
        terms["gain"] = f"untouched family; median confirmed {-gain:+.6f} as prior"

    # INFORMATION. Highest where the family's effect is uncertain or unmeasured. A family
    # measured many times with a tight spread teaches little more, however large its mean.
    if len(obs) >= 2:
        # ROBUST spread. Raw sd made a family containing one catastrophe look maximally
        # informative: tbs carries the +0.022545 tbs=20 result, which alone drove the
        # spread to 0.0076 and would have made every tbs arm look like the best available
        # source of information. Median absolute deviation ignores the outlier that is
        # already understood and blocked.
        med = st.median(obs)
        info = st.median([abs(e - med) for e in obs]) * 1.4826
        terms["info"] = f"robust spread (MAD) over {len(obs)} runs {info:.6f}"
    else:
        info = 0.002
        terms["info"] = "unmeasured family; spread unknown"

    # NOVELTY. Per-axis, not per-family: a family can look busy while hiding an axis never
    # varied once, which is the failure that closed an entire direction in an earlier
    # campaign.
    ax = direction.axes_touched(cfg)
    seen = [state["axes"][a]["n"] for a in ax] or [0]
    novel = 1.0 / (1.0 + min(seen))
    terms["novelty"] = f"least-explored axis has {min(seen)} runs"

    # DISAMBIGUATION VALUE. The first version of this file ranked three arms at exactly
    # the same score, which made it useless for the only comparison that mattered: two of
    # them decompose a confound sitting on the campaign's largest result, and the third
    # merely extends a ladder. Gain, information and novelty are all family-level and
    # cannot see that difference. An arm whose hypothesis is named in the MITIGATION of an
    # active integrity or non-activation lesson is doing something the others are not --
    # it is buying an interpretation for evidence already paid for, which is worth more
    # per GPU-hour than new evidence nobody can yet read. This is the "value of
    # discriminating between competing explanations" term, and it is weighted heavily on
    # purpose: an unresolved confound makes every downstream arm harder to interpret.
    resolves = []
    for h in claims.hypotheses():
        if {k: v for k, v in (h.get("intervention") or {}).get("cfg", {}).items()
                if direction.PLATFORM.get(k) != v} != {
                    k: v for k, v in cfg.items() if direction.PLATFORM.get(k) != v}:
            continue
        # The link runs BOTH ways and in practice runs the other way. A lesson's
        # mitigation describes what to change ("add a mu_warmup knob"), while the
        # hypothesis written to settle it names the lesson explicitly in its rationale.
        # Checking only lesson->hypothesis found nothing and the term silently never
        # fired, leaving three arms tied -- a scoring term that cannot fire is the same
        # defect as an activation rule the control also passes.
        blob = f"{h.get('rationale','')} {h.get('statement','')}"
        for l in claims.active_lessons():
            if l.get("type") not in ("integrity", "non_activation", "overclaim"):
                continue
            if h["id"] in str(l.get("mitigation", "")) or l["id"].split("_")[0] in blob:
                resolves.append(l["id"])
    resolve = 1.0 if resolves else 0.0
    if resolves:
        terms["resolves"] = f"decomposes {resolves[0]}"

    # COST is the same 1.1 GPU-hours for every quad, so multiplying it by a weight can
    # never reorder anything: it subtracts a constant from every score. The comment here
    # previously also promised "a penalty for arms whose activation cannot be checked",
    # which was never written -- a comment describing code that does not exist, the same
    # shape as a check that cannot fire. Both are now honest: cost is reported because a
    # reader should see it, and NOT scored; the activation penalty is implemented.
    cost_h = GPU_MIN_PER_QUAD / 60.0
    terms["cost"] = f"{cost_h:.1f} GPU-hours (equal for all quads; reported, not scored)"

    # ACTIVATION PENALTY, now real. An arm whose hypothesis declares a diagnostic the
    # control also satisfies cannot tell a null from a no-op, so its evidence is worth
    # much less per GPU-hour whatever else it promises.
    act_pen = 0.0
    for h in claims.hypotheses():
        if {k: v for k, v in (h.get("intervention") or {}).get("cfg", {}).items()
                if direction.PLATFORM.get(k) != v} != _changed(cfg):
            continue
        a = h.get("activation") or {}
        if a.get("diagnostic") and a.get("rule"):
            # Pass the cfg. Both queue doors were updated to do this when the check
            # gained the ability to restrict to THIS arm's own runs; this call site was
            # missed, so the scoring path -- which decides what runs next -- kept using
            # the loose behaviour the argument was added to close. Third instance today of
            # fixing a function and not all of its callers.
            good, _m = claims.diagnostic_would_discriminate(
                a["diagnostic"], a["rule"], cfg)
            if not good:
                act_pen = 1.0
                terms["activation"] = "REFUSED by pre-check: cannot demonstrate engagement"

    s = (W_GAIN * gain / 0.001
         + W_INFO * info / 0.001
         + W_NOVEL * novel
         + W_RESOLVE * resolve
         - W_ACTPEN * act_pen)
    return s, terms


def batch(cands, rows, state, fx, k):
    """Greedy selection with a family-diversity penalty. Top-k by score alone returns
    five variants of whichever family last paid; halving the score of an already-chosen
    family forces the batch to spread without forbidding a genuine follow-up."""
    scored = sorted(((score(c, rows, state, fx)[0], c) for c in cands),
                    key=lambda x: -x[0])
    picked, used = [], {}
    for _ in range(min(k, len(scored))):
        best, bi = None, None
        for i, (s, c) in enumerate(scored):
            if s == -math.inf or any(c is p for p in picked):
                continue
            fams = _families(c)
            pen = 0.5 ** sum(used.get(f, 0) for f in fams)
            if best is None or s * pen > best:
                best, bi = s * pen, i
        if bi is None:
            break
        s, c = scored[bi]
        picked.append(c)
        for f in _families(c):
            used[f] = used.get(f, 0) + 1
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", help="score a single cfg (JSON of non-platform keys)")
    ap.add_argument("--batch", type=int, default=0)
    a = ap.parse_args()

    rows = analyze.load()
    state = direction.axis_state(rows)
    fx = family_effects(rows)

    if a.cfg:
        cfg = {**direction.PLATFORM, **json.loads(a.cfg)}
        s, terms = score(cfg, rows, state, fx)
        print(f"score {s:+.2f}" if s > -math.inf else "REFUSED")
        for k, v in terms.items():
            print(f"   {k:9s} {v}")
        return 0

    # Candidates: whatever is queued but not yet run, deduplicated by cfg.
    q = json.loads((REPO / "runs" / "sweep" / "queue.json").read_text())
    seen, cands = set(), []
    for e in q:
        if (REPO / "runs" / "sweep" / "results" / f"{e['name']}.json").exists():
            continue
        cfg = e.get("cfg") or {}
        if direction.is_platform(cfg):
            continue
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key); cands.append(cfg)

    if not cands:
        print("nothing pending to rank.")
        return 0

    print(f"=== SELECTOR over {len(cands)} pending candidate cfg(s) ===")
    print("ranking only -- the weights are stated, not fitted, so this orders what you")
    print("can afford to run and never admits or refuses on its own.\n")
    for s, cfg in sorted(((score(c, rows, state, fx)[0], c) for c in cands),
                         key=lambda x: -x[0]):
        lbl = direction.label(cfg)
        delta = {k: v for k, v in cfg.items() if direction.PLATFORM.get(k) != v}
        if s == -math.inf:
            _, t = score(cfg, rows, state, fx)
            print(f"  REFUSED  {lbl:22s} {delta}  {t.get('BLOCKED','')[:60]}")
        else:
            print(f"  {s:+7.2f}  {lbl:22s} {delta}")

    if a.batch:
        print(f"\nDIVERSE BATCH OF {a.batch} (family-penalised greedy):")
        for cfg in batch(cands, rows, state, fx, a.batch):
            print(f"  {direction.label(cfg):22s} "
                  f"{ {k: v for k, v in cfg.items() if direction.PLATFORM.get(k) != v} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
