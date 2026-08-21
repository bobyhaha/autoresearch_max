# Mechanistic Analysis 001 — Why the levers worked, why they saturated, and mechanistic proposals

Scope: `walltime_5min_h200` frame (fixed 300s wall-clock, one H200, ~50M-param nanochat GPT,
Muon+AdamW, val_bpb). Every confirmed lever and every open proposal below carries a **causal
chain → prediction → falsifier**, per `docs/AGENT_PROTOCOL.md` idea-generation.

## 0. The objective landscape (the mechanistic frame)

`val_bpb` at 300s is the product of two coupled factors:

- **U = number of optimizer updates completed in 300s** (throughput / (tokens-per-step)), and
- **q = useful loss-reduction per update** (a function of model capacity, context quality, gradient signal).

At this tiny scale the model is **far from convergence in 300s → the regime is update-limited**:
`d(val_bpb) ≈ −(dq · U + q · dU)`. This single equation explains every result: a lever wins iff it
raises the *product* q·U. Throughput levers raise U at fixed q; capacity/context levers raise q, and
are worth it only if the q-gain outweighs the U they cost. This is the lens for all proposals.

## 1. Confirmed levers — mechanisms

**L1. torch.compile warm cache (1.030→0.988).** *Causal chain:* the first-ever run pays cold
Inductor/autotune compilation inside the timed window → ~2.8× fewer steps (671 vs ~1020 warm) at
identical q → higher val_bpb. *Not modeling.* *Falsifier (met):* a warm re-run of the identical
config reproduces ~0.988 (3-seed 0.98768). Pure U effect.

**L2. Batch 2¹⁹→2¹⁸ (0.988→0.980).** *Causal chain:* halving tokens-per-update **doubles U** at
fixed token-throughput; because the model is update-limited, the extra updates lower loss more than
the increased per-update gradient noise costs. Below 2¹⁸ (2¹⁷,2¹⁶) gradient noise per update grows
faster than U helps → q collapses. So 2¹⁸ is the **critical batch size** — the argmax of q·U.
*Prediction (met):* U-shape with min at 2¹⁸; 2¹⁷=0.9863, 2¹⁶=1.0007 worse. *Falsifier:* a monotone
improvement to smaller batch (not seen).

**L3. FFN ratio 4→6× (0.980→0.977).** *Causal chain:* the MLP is per-token feature/associative
capacity; at 50M params the 4× FFN is under-provisioned for this data, so widening raises q (more
hidden units → richer token transformations) more than the ~14% U it costs (bigger matmul, ~1700→~1720? →
actually ~2020→~1720 steps). Beyond 6× the added width is undertrained in fewer steps AND costs more U
→ net worse. *Competing explanation ruled out:* widening embed/attention (aspect_ratio 96) was WORSE,
so it is **FFN capacity specifically**, not "more params." *Prediction (met):* peak at 6×, 5×/7×
bracket it, 8×/10× worse.

**L4. n-gram value embeddings (0.977→0.954).** *Causal chain (the dominant lever):* without it the
model must **reconstruct local n-gram statistics** (which bigram/trigram predicts the next byte) from
scratch through attention+MLP, spending early-layer capacity. A hash-embedding table keyed on the last
2–3 tokens gives **O(1) direct access to a learned per-n-gram value vector** — static associative
memory added to the value stream. This (a) offloads local-statistics modeling from compute to lookup,
freeing the transformer for longer-range structure (Engram's "relieve early layers of static
reconstruction"), and (b) adds ~0 FLOPs (a gather, not a matmul), so **U is preserved** (MFU even rose
28%→43% because the tables give the GPU more parallelizable work). *Table-size sub-mechanism:* bigger
table → fewer hash collisions → less destructive interference between distinct n-grams sharing a slot →
cleaner memory; monotone 2¹⁵→2²⁰. *Saturation mechanism (~2²⁰–2²¹):* added slots are hit too rarely in
~440M tokens/run to train (slot-sparsity undertraining), so collisions stop being the binding
constraint. *Order mechanism:* trigram helps only once the table is big enough that its (sparser) slots
don't collide; **4-gram is data-starved** (too many distinct 4-grams per 440M tokens). *NGRAM_LR-null
mechanism:* tables are undertrained by slot-sparsity, **not** by LR, so raising NGRAM_LR can't fix it
and high LR destabilizes the frequently-hit slots (≥2.0 worse). *All falsifiers met.*

**Saturation summary:** every scalar knob (LR family, schedule, wd, betas, warmup) is null within σ≈0.0015;
n-gram table and LR are saturated. Per protocol we STOP refining these and switch axes.

## 2. Open proposals — mechanisms, predictions, falsifiers

**P1. Document-masked (intra-document) attention — HIGHEST PRIORITY (Round 19).**
*Causal chain:* the loader packs several unrelated documents into each 2048-token block; global causal
attention lets a token attend **across document boundaries** to unrelated prior documents, injecting
off-distribution context the model must learn to suppress — wasted capacity + a noisier attention
distribution → higher q-loss. Masking attention to the current document makes every token's context
**coherent** → cleaner next-byte prediction → lower val_bpb. *Orthogonality:* n-gram = local memory;
doc-mask = context hygiene — different mechanisms, so they should **stack**. *Prediction:* q improves;
small U cost from masking. *Falsifier / competing-explanation control:* if the gain is merely "shorter
effective context is cheaper," a **length-matched sliding window** would reproduce it — but literature
shows sliding-window HURTS, so a doc-mask win that a length-matched window does NOT reproduce isolates
the coherent-context mechanism. *Expected magnitude:* large — the biggest remaining lever (this is a
prime suspect for the gap between the current 0.954 and a fully-levered ~0.93).

**P2. Kernel / throughput (Round 18, running).** *Causal chain (update-limited ⇒ U is first-class):*
(a) `max-autotune-no-cudagraphs` lets Inductor search better GEMM tilings → higher MFU (43% now, ~57%
headroom) → more U at fixed q → pure win. (b) Muon `ns_steps` 5→3: fewer Newton–Schulz matmuls per
matrix param → less compute per step (more U) but **coarser orthogonalization** → possibly worse update
direction (lower q). *Prediction:* max-autotune = free U gain; ns_steps reduction = U↑ but q↓,
net ambiguous (expect a small-ns optimum). *Falsifier:* max-autotune MFU unchanged (autotune found
nothing) or ns3 q-loss outweighs its U gain.

**P3. Per-order table sizing (minor).** *Causal chain:* distinct-trigram count ≫ distinct-bigram count,
so equal table sizes over-collide trigrams and waste bigram slots; sizing each table to its order's
distinct-count equalizes collision rate → marginal q. *Prediction:* small gain, since n-gram is near
saturation. Low priority.

**P4. Data information-density / curriculum (blocked).** *Causal chain:* at fixed tokens, useful signal
per token varies; curriculum/dedup/quality-ordering raises effective information → q. *Blocker:* the
loader lives in read-only `prepare.py`; only order policies expressible outside it are testable. Deferred.

## 3. Ranked next actions (expected Δval_bpb per GPU-hour)
1. **P1 document-mask** — large, orthogonal, literature-backed. Implement + 1-seed screen → 3-seed.
2. **P2 max-autotune** — free U; adopt if MFU/steps rise. ns_steps — screen for the U↔q knee.
3. Stack the winners (doc-mask + max-autotune) and re-confirm the composite SOTA.
4. **P3 per-order tables** only if P1/P2 stall.

_Every future round's launch note will state the mechanism it tests and its falsifier._
