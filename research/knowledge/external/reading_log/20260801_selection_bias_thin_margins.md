# Reading log — selection bias and effect-size inflation at thin margins

Searched for 2025–2026 work on effect-size inflation under selection, to check
whether the campaign's first adoption (E28, 4% over the floor) has a known failure
mode I had not accounted for.

## Scope verdict, stated first

**None of the retrieved sources are scope-matched to this frame.** They are
clinical trials, neuroimaging, credit-risk modelling and cross-validation theory —
no language-model pretraining, no wall-clock-budgeted training. **No number from
them is transferred here.** What they supply is a named failure mode, and the
question is whether our design has it.

Two general findings worth quoting because they name the mechanism:
- trial effect estimates were **64% higher** where high selection bias was
  identified, versus low;
- small samples "lead to unreliable and overestimated effect sizes", and
  replicability suffers accordingly.

## The flaw this exposes in E28, which I had not named

E28's configuration was not arbitrary. **depth 6 was selected as the best arm of
E10's sweep (seeds 42–45), and MATRIX_LR 0.03 as the best arm of E26/E27 (seeds
42–45).** E28 then measured that selected combination on **seeds 42–51 — which
include all four selection seeds.**

That is selection and evaluation partly on the same draws, and it inflates the
estimate in exactly the direction observed. It plausibly accounts for some of the
**0.000415 super-additive residual** that carried the result over the floor, which
I recorded as "measured, not explained". A winner's-curse contribution is now a
named candidate explanation for that residual.

I did not flag this when recording the adoption. The thin-margin, CI and
registered-baseline caveats were all recorded; **this one was missed**, and it is
the most likely of the four to matter.

## What already addresses it

E29, launched this block before this reading, uses **seeds 52–61 — disjoint from
every arm this campaign has ever run**, including every selection sweep. That is
the correct control for the flaw, and it was chosen for the seed-reuse reason among
others. The reading confirms the design rather than prompting it, but it upgrades
the concern from "good practice" to "a specific, named bias with a plausible
magnitude".

## Consequence for how E29 must be read

The pooled 20-pair figure is the honest headline, **not** the better of the two
tranches. If E29 regresses toward the additive prediction (~0.88 gates, sub-gate),
the adoption is retracted — and under the winner's-curse account, regression is the
*expected* outcome, not a surprise. The decision rule was fixed in the E29 launcher
before this reading and is unchanged by it.
