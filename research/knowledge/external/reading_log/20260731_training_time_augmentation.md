# Reading log — arXiv 2606.16246, Training-Time Augmentation for Data-Constrained LM Pretraining

Frame: 94M dense params (+2.82e9 n-gram table params), ~250M unique tokens,
2–3 passes, 300 s on one H200, gate 0.002614.

## Scope, in numbers

| axis | paper | us | verdict |
|---|---|---|---|
| model size | 150M (20 layers, 512 hidden) | 94M dense | **1.6× — close** |
| unique tokens | 75M | ~250M | same regime, they are *more* constrained |
| repetition | 100 epochs default | 2–3 passes | they are far deeper in |
| cost | **data pipeline only**, no extra pass | — | **free; token law charges nothing** |

Their baseline reaches its optimum at **epoch 16** and augmentation pushes it to
**epoch 60–68**. Our cliff is at pass 3. The kind of failure matches even though
the epoch numbers do not, because our 2.82e9 table parameters over 250M tokens is
a far more over-parameterised ratio than their 150M over 75M.

## Results quoted

Individual: random replacement 15% → 3.841 (baseline 4.015); R2L 50% → 3.910;
offset i≤5 exponential → 3.870; mask 15% → 3.910.
Best combination: random 5% + R2L + offset i≤5 exp → **3.805 at epoch 68**.
Negative results they report: FIM 50% gives "no benefit" and "overfits at the same
rate as the baseline"; every token-noise × offset combination "fails badly",
attributed to noise corrupting the context that the offset target depends on.

Hyperparameter granularity **not stated in the sections retrieved**: whether the
offset `i` is sampled per token or per sequence is absent, and is NOT inferred here.

## The architectural filter — why only ONE of these is usable for us

Our n-gram tables index by **hashes of actual token IDs**, and they are 96.8% of
the model. Therefore:

- **masking / random replacement — REJECTED for this architecture.** They alter
  input token IDs, which changes every n-gram hash, so the tables would be trained
  on corrupted contexts and evaluated on clean ones. This is a far larger
  perturbation for us than for the paper's plain Llama, and the paper's own worst
  result is exactly the combination where noise corrupts context.
- **R2L / FIM — REJECTED.** Reordering the sequence also changes the n-gram
  contexts. FIM additionally gave no benefit in the paper.
- **Target offset prediction — USABLE.** It changes only the **label**, never the
  input. The n-gram hashes see unchanged contexts. It is the only augmentation in
  this paper that survives our architecture, and it is free.

With P(i) ∝ e^{−(i−1)/T}, T=1, over i ∈ {1..5}, the normalised distribution is
P = (0.635, 0.234, 0.086, 0.032, 0.012), so 63.5% of positions remain ordinary
next-token prediction. The augmentation is self-balancing rather than replacing
the objective the metric measures.

## Verdict

**PROPOSE (E19): offset-target augmentation at depth 5, crossed with depth 6.**
Free, label-only, leaves the n-gram inputs intact, and it attacks memorisation at
its source rather than penalising the memoriser's weights — which E15 proved
cannot work, because in a hashed n-gram memory the memorising rows and the
signal-carrying rows are the same rows.
