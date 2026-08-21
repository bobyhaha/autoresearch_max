# Reflection: what the campaign got right, and the one thing it got structurally wrong

Written 2026-07-31 at the 15h mark of run 2, against the standard: *a coherent
idea proposal, followed by a well-supported claim and evidence.*

## The structural failure

**5.7% of the literature corpus has ever done predictive work.** 123 external
claims are recorded; 7 have ever been bound to a mechanism. The other 116 are
inert.

That number explains the campaign better than any individual result. What I have
mostly been running is a **filter**, not a research programme:

> pick a lever with adequate scope match → run it paired → check it against the gate → close it

Sixteen directions closed that way. The process is sound and the bookkeeping is
honest, but a filter cannot compound. Each null teaches almost nothing about the
next candidate, because nothing predicted the null in advance. A claim that only
gets consulted when choosing what to run next, and never makes a quantitative
prediction that the experiment could contradict, is decoration.

The two moments this campaign produced something durable were both the opposite
shape:

1. **fill-flood** — an observed phenomenon (194k tiny fill launches, 19% MFU),
   a proposed cause, and a prediction a rival account *forbade* (the sparse
   penalty must shrink with table size). Confirmed twice, once out-of-sample
   (R²=0.995 on three points).
2. **the 1e-3 vs 1e-2 selection rule** — derived from sixteen failures, and it
   predicted where the first gate-clearing candidate would come from *before*
   depth was run.

Both were built from measurement, not selected from a list.

## What was done right

- **Preregistration held under pressure.** The analysis rule for E6 was fixed
  before launch; applying the control variate afterwards would have converted
  "13% slower" into "essentially null." Depth 6 was never claimed at n=4 (1.24
  gates), n=6 (1.03), or n=8 (0.92, now below gate).
- **Self-refutation was carried out on my own published work.** The token-law
  audit killed my depth-6 decomposition; I withdrew it and downgraded the belief
  to `challenged` rather than defending it.
- **Mechanism and intervention were kept separate.** fill-flood is `active`
  while its hypothesis is `rejected` — the mechanism is true and the change is
  worthless, and collapsing those would have lost the true half.
- **Measurement defects were fixed upstream, not screened around.** Three screen
  versions, then probe-gated launching after a 40-second probe showed a uniform
  1.58× box-wide slowdown that `nvidia-smi` reported as 0% utilization.
- **Eight proposals killed before implementation** by the break-even rule,
  including MIR this hour.

## What was done wrong

- **Idea generation was selection, not synthesis.** E1/E2/E3 were "claims with
  adequate scope match." E5 was "a different axis." Only E10 (shape) was derived,
  and it came from our own failures rather than from any literature.
- **Almost no literature was read during the run.** One corpus re-scan. Zero new
  papers until hour 15, despite repeated instruction. The paper found then
  (arXiv 2606.06888) turned out to be the **best scope match in the entire
  corpus** — 72M–1.4B parameters against our 94M, 100–400M unique tokens against
  our ~250M, explicitly about multi-epoch repeated data, which is exactly the
  regime whose cliff we had independently measured at 17.6 gates. Fifteen hours
  of experiments ran before the most relevant paper was opened.
- **Roughly 40% of the run went into repairing my own apparatus**, much of it
  self-inflicted: three screen revisions, four invented enum values, two
  half-written ledgers. Each recurring error was "fixed" twice — once with a
  written rule that failed, once with a mechanism that worked.
- **No SOTA.** Two runs, ~30 hours, best candidate below the gate and regressing.

## What changes

1. **Read first, then propose.** A block does not start until at least one
   primary source has been read and its scope compared against the frame in
   numbers, not adjectives.
2. **A proposal must carry a prediction a rival forbids.** If no observable
   separates the mechanism from its obvious alternative, it is not ready to run.
3. **Write the mechanism, not the rule.** Every recurring failure here was solved
   only when it became structurally impossible.
4. **The prize sets the priority.** Epoch-3 collapse is the largest measured
   cliff in the campaign (17.6 gates) and gates the recorded −0.022 ceiling on
   all throughput work. It should have been the target long before hour 15.
