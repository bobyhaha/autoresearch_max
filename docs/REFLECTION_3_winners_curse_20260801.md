# Reflection 3 — the campaign has been measuring its own selections

Written 2026-08-01, triggered by directive (A): **a reported result had to be
withdrawn.** Written from `campaign_log.jsonl` and the batch records.

## The pattern, stated as arithmetic rather than as luck

Twice now a result has looked adoptable and then shrunk when more data arrived:

| result | first reading | final | regression |
|---|---|---|---|
| depth 6 vs 8 | 1.24 gates (n=4) | 0.77 (n=12) | **0.47 gates** |
| depth 6 + LR 0.03 | 1.04 gates (n=10, seeds 42–51) | 0.60 (n=10, disjoint 52–61) | **0.44 gates** |

Both were called "regression toward the true value" at the time. That is a
description, not a cause. The cause is that **both configurations were chosen by
taking the best arm of a multi-arm sweep, and then reported at that arm's value.**

The adoption gate is 2σ of *single-pair* noise. It is calibrated for **one
pre-specified comparison**. It is not calibrated for the maximum of k arms.

## Quantifying it

With the pooled paired sd measured at 0.00133 (n=20, E28+E29), the expected value
of the maximum of k standard normals is 0.564σ (k=2), 0.846σ (k=3), 1.029σ (k=4).
So a 4-arm sweep at n=4 seeds inflates its winner by

    1.029 × 0.00133 / √4 = 0.00068  =  **0.26 gates**

**Case 1 — depth 6.** One selection from a 4-arm sweep at n=4. Predicted inflation
0.26 gates; observed regression 0.47. **Selection explains ~56%. The remainder has
no account.**

**Case 2 — E28.** Two independent selections (depth from a 4-arm sweep, LR from a
4-arm sweep), both inflating the same direction, so the biases add: predicted 0.52
gates. Observed regression to the disjoint-seed estimate 0.44. **Fully explained.**

So winner-selection is a real, quantified contributor — and it is *not* the whole
story for the single-selection case. Half of that regression remains unexplained,
and I am recording that rather than rounding it into the tidy account.

## The consequence, which is larger than either result

A 4-arm sweep reporting **1.20 gates at n=4** is, in expectation, at **0.94** — below
the floor. Two stacked selections need a **0.52-gate** discount. The gate is 1.00.

**Every "promising interim" this campaign has reported is subject to this**, and the
interims were reported in hourly messages before the funnel completed. The funnel
itself was never fooled — n=10 caught both cases — but the narrative around it was.

## What I assumed and never measured

**The control may be mistuned.** MATRIX_LR was swept only at depth 6, where the
basin floor is near 0.03, yet every comparison in this campaign uses depth 8 with
LR 0.04. Reconciliation v21 records that value reverting from the accumulated-SOTA
0.03 in a defaults regression that was never re-derived. If 0.03 also helps at depth
8, part of the measured −0.002139 belongs to the LR and not to the shape, and the
registered baseline itself is slightly wrong.

That is the cheapest unmeasured assumption available and it biases *every* number
the campaign has produced. E30 tests it on seeds 62–71, disjoint from everything.

## Rules that follow

1. **Selection and confirmation never share draws.** Adopted last block; it is what
   caught the E28 retraction.
2. **Discount a swept winner before comparing it to the gate:** ~0.26 gates per
   selection at k=4, n=4; additive across stacked selections.
3. **Do not report interims as findings.** State the arm and the n, and say the
   number is expected to shrink.
4. **Tune the control at its own operating point before trusting a contrast** — the
   same scoped-result error as E14 and E21, in a third costume.

---

# CORRECTION (2026-08-01, added two blocks later after E31)

**This reflection over-explained a single low draw, and the correction matters more
than the original claim.**

The argument above attributed E29's regression (1.04 → 0.60 gates) to
winner-selection bias, with arithmetic predicting a 0.52-gate discount for two
stacked selections. E31 then ran the *same* configuration on seeds 72–81, **equally
disjoint from every selection sweep**, and measured **1.07 gates — above E28.**

If selection bias were driving the E29 reading, both disjoint blocks would read low.
One did, one did not.

| block | seeds | gates |
|---|---|---|
| E28 | 42–51 (selection-contaminated) | 1.04 |
| E29 | 52–61 (disjoint) | **0.60** |
| E31 | 72–81 (disjoint) | **1.07** |

Between-block sd is 0.000690 against 0.000459 expected from n=10 sampling — a
ratio of 1.50×, well inside what three draws produce. **The spread is consistent
with noise.**

## What survives, and what does not

**Survives:** the arithmetic is correct — a 4-arm sweep at n=4 does inflate its
winner by ~0.26 gates — and the rule *selection and confirmation never share draws*
is right, cheap, and worth keeping regardless.

**Does not survive:** the claim that selection bias *explained* the E29 regression.
A single n=10 block is too noisy to attribute to any mechanism, and I attributed it
to one because I had just computed a number that happened to match.

## The lesson that replaces it, which is more useful

**At n=10 this frame cannot distinguish 0.60 gates from 1.07.** Any single 10-pair
tranche is close to a coin-flip about the gate. That is a statement about the
frame's *resolution*, not about bias — and it means the campaign's `min_seeds=10`
is adequate for rejecting large effects and inadequate for resolving ones near the
floor.

That reframes the whole adoption episode. E28 was not a case of me being fooled by
selection; it was a case of a marginal effect measured at a resolution that cannot
settle marginal effects. The retraction was still correct — the pooled estimate is
sub-gate — but the *reason* I gave for it was wrong.
