# Reading log — how to report a CI that straddles a decision threshold

Read because the campaign headline is exactly this shape: a real effect whose 95%
CI crosses the adoption floor, and I wanted to check the reporting convention
before the capstone fixes its wording.

## Scope verdict, first

**No scope match, and none is claimed.** The sources are FDA equivalence-testing
guidance, TOST calculators and the `TOSTER` vignette — biostatistics methodology,
not ML. **No number is transferred.** What transfers is a *convention*, and
conventions are the one thing that legitimately crosses domains.

## The convention

- In TOST each one-sided test uses α=0.05, so the combined decision interval is
  **90%**, not 95%.
- Equivalence is declared **iff the interval lies entirely within the bounds**.
- When the interval **straddles** the bound, the result is explicitly
  **inconclusive** — "the data neither definitively supports equivalence nor rules
  it out" — and the correct report states which of the three cases holds:
  entirely inside, partially crossing, or entirely outside.

## What this changes here, and what it does not

**Does not change:** the campaign's adoption rule is a one-sided point-estimate
rule (`mean ≤ −0.002614` at n≥10 clean), fixed in the registered frame long before
this reading. That rule is satisfied or not on the point estimate alone, and it was
not satisfied at any pooled n. **The verdict stands: no adoption.**

**Does change the wording I should use.** I have been writing "the CI straddles the
floor, so we cannot say whether the true effect clears." The convention gives that
a name and a correct form: the result is **inconclusive with respect to a
threshold test**, distinct from both "clears" and "does not clear". Reporting it as
a settled negative — which is what E33 and E34 were launched to produce — would
require the interval to fall entirely on one side, which at n=78 it does not.

**One thing I got right by accident:** I have consistently reported the 95%
interval alongside the point estimate rather than only the verdict, which is what
the convention asks for. I did not have a principled reason; now I do.

## Consequence for the capstone

Paper 033 should say the pooled effect is **real and its threshold test is
inconclusive at n=78**, not that it "does not clear". Those differ, and the
difference is exactly what E34 is trying to remove. If E34 lands and the interval
falls entirely above the floor, "does not clear" becomes correct and I will say it.
