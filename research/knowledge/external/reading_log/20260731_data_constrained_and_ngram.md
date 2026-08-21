# Reading log — 2026-07-31

Frame for scope arithmetic: 94M dense params (+1.61e9 n-gram table params),
~250M unique tokens, ~300–500M consumed (~2 passes), 300 s on one H200,
gate 0.002614, intrinsic effects ~1e-3, throughput ~1e-2.

## arXiv 2606.06888 — Data-Constrained LM Pretraining: Improved Regularization and Scaling Laws

**Scope: EXCELLENT — the best match in the corpus.** 72M–1.4B params (72M is
*below* our 94M dense count); 100M–400M **unique** tokens (we are at ~250M);
explicitly about multi-epoch repeated data, which is the regime whose cliff we
independently measured at 17.6 gates.

- Proposes **MIR**: `L = L_NTP(x) + λ·L_NTP(x̃)`, where `x̃` replaces each token
  with `[MASK]` at rate `r ~ Unif(r_min, r_max)` sampled per sequence. λ, r_min,
  r_max are not stated in the sections retrieved — **not inferred here**.
- Reports gains "equivalent to roughly 1.3× as much unique training data";
  1.4B: 3.347 → 3.317.
- Proposes **SoftQ**, a scaling law coupling model size and data size under
  repetition, arguing Chinchilla's additive form is misspecified in this regime,
  and that the data-constrained regime favours **smaller models, more epochs**.

**VERDICT on MIR: killed prospectively.** A second forward+backward costs ~2×
step time; the token law charges +0.0416 for halving steps against a reported
gain worth −0.0157. Short by 2.6×. It is affordable in the paper's fixed-step
frame, where the cost is not charged, and not in ours.

**What we took instead:** the paper frames MIR as added *on top of strong weight
decay*, and weight decay is free. E14 tested it where the cliff binds (depth 5,
third pass). **Result: refuted, monotonically in the wrong direction** — WD 0.1
0.9791, WD 0.3 0.9905, WD 0.6 1.0031. The epoch-3 collapse is not rescued by
weight decay at any level tested.

## arXiv 2601.07372 — Engram: Conditional Memory via Scalable Lookup

**Already implemented.** Verified against `train.py`: multigranular n-gram
orders, multi-head hashing, per-layer decorrelated primes, gated fusion into the
value stream, projection back to model dim — all present. The 1.61e9 n-gram table
parameters *are* Engram. E7 and E8 already mapped its capacity optimum
(mult=64 is the knee; 5× and 256× are both worse). **Remove from the priority
list**; it is not an untested lever.

## arXiv 2605.24869 — Lngram: N-gram Conditional Memory in Latent Space

**Scope: POOR.** Smallest model tested is **Qwen3-1.7B (18× our dense count)**;
budgets 35B–140B tokens (**100–500× ours**); no sub-200M and no short-budget
results. Reports +1.41pp average on general tasks at 22B/35B tokens, and
+6.7% decode latency.

Mechanism: replaces tokenizer-ID hashing with learned discrete symbols
binarized from hidden states, removing tokenizer dependence and hash collisions.

**VERDICT: not a candidate for this frame.** Three independent reasons, any one
sufficient: (1) scope mismatch of 18× in params and 100× in tokens, against a
frame where paper 027 registered that refinements validated at longer horizons
do not transfer; (2) it restructures the n-gram memory, and our own E8 tested
exactly that class locally (TN-gram CP sharing) and found it **1 gate worse at
matched parameters** — direct local evidence against; (3) it adds readout
projections and latency in a frame where a 1.5%-of-step cost already exhausts
the intrinsic budget.
