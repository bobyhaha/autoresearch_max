# Reading log — arXiv 2605.29152, "Do Deep Networks Forget Initialization?"

## Scope, in numbers

| axis | paper | us | verdict |
|---|---|---|---|
| architecture | BatchNorm ResNets | Transformer LM | **mismatch, stated by the paper** |
| dataset | CIFAR-10, 40k train | 250M-token corpus | mismatch |
| training length | 300 epochs, ablation to 5000 | ~2000–4500 steps | mismatch |
| runs | 6250+ | — | far better powered than us |

The paper's own limitations section says: *"Our experiments are restricted to
BatchNorm ResNets on CIFAR-10"*, with no language models, transformers or
pretraining, and names LLM pretraining as its most pressing extension. **It is not
evidence about this frame.** It is a rival *mechanism* with a testable prediction.

## The claim, and why it matters here

> "The right unit of forgetting is accumulated regularization, not epoch count."

Quantitatively: under low-LR SGD at b=128, going from 300 to 5000 epochs (timescale
7.3e-4 → 1.2e-2) leaves the initialization-scale spread at ~26 percentage points —
*"essentially unchanged"*. Adding L2 (λ=1e-2) with η=1e-2 collapses the same spread
to 0.3 pp. **Duration does not do it; the regularization profile does.**

## Why this is a test of OUR published claim

Paper 027 explained three null refinements (conditioned init, Peri-LN, GPAS) by the
frame: *"such refinements act on asymptotic optimisation quality … whose benefit
accrues over many more updates than 2000."* That is a duration account, and it has
never been tested — every one of those levers ran only at depth 8 (~2016 steps).

Depth 6 runs **3276 steps, 63% more updates at the same 300 s**. So:

- **frame account (ours):** the refinements should do measurably better at depth 6
- **forgetting account (theirs):** unchanged, because duration is not the variable

E23 runs depth 6 × {none, COND_INIT, PERI_LN_ATTN, both} × 4 seeds. It also does
something the campaign has never done: **stack two levers**. Every lever so far was
tested alone against a depth-8 baseline and then closed.

## Verdict

**Not adoptable evidence; a useful rival mechanism.** Recorded as such. No number
from this paper is transferred to our frame, and its architecture mismatch is
stated rather than glossed.
