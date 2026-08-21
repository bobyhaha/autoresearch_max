# Paper 001 — Compile-Cache Confounds and an Update-Count-Limited Batch-Size Optimum in Fixed-Wall-Clock LLM Pretraining

**Campaign:** vibeautoresearch_reimplementation, rounds 1–5 (exp000–exp035) · **Date:** 2026-07-27
**Frame:** `walltime_5min_h200` — 5-minute wall-clock budget, single H200 · **Metric:** val_bpb (bits/byte, lower is better)

## Abstract

We report the first five rounds of an autonomous hyperparameter campaign on a fresh Karpathy `autoresearch` baseline (nanochat-style GPT, ~50M parameters, depth 8, SSSL sliding-window attention, Muon+AdamW) under a fixed 5-minute wall-clock training budget on one H200. Two results survive our keep/discard gate. First, the naive first-run baseline (val_bpb 1.0301, 671 steps) is not a fair control: it pays a cold torch.compile/inductor-cache penalty (~650 ms/step vs ~230 ms/step warm), and the identical code with a warm cache reaches 0.9877 (3-seed mean 0.98768, σ ≈ 0.0015, ~1017 steps). The apparent 1.030 → 0.988 "improvement" is throughput, not modeling. Second, against this corrected warm baseline, an extensive sweep of optimizer and schedule knobs and of structural variants (depth, width, attention window) produced no gains, while halving TOTAL_BATCH_SIZE from 2^19 to 2^18 — doubling optimizer updates to ~2020 within the budget — yielded a confirmed 3-seed improvement of −0.008 val_bpb (mean 0.9797, best 0.9784), roughly 4–5σ against the single-seed noise floor. Further halving regresses (2^17: 0.9863; 2^16: 1.0007), locating a batch-size optimum at 2^18. We conclude the frame is update-count-limited near the default configuration and adopt 2^18 as the new reference.

## 1. Central hypothesis

> **H1.** At the 5-min/1-H200 frame, the default configuration (TOTAL_BATCH_SIZE = 2^19) is **update-count-limited**: val_bpb is governed primarily by the number of optimizer steps completed within the wall-clock budget, so reducing tokens per optimizer step improves val_bpb monotonically until per-step gradient noise dominates, with the crossover near 2^18.

This is falsifiable on both flanks: it predicts (a) 2^18 beats 2^19 beyond the noise gate, and (b) batch sizes below the crossover get *worse* despite still more updates. Both predictions were tested. 2^18 improved by −0.008 (3-seed, ~4–5σ); 2^17 recovered only part of the gain (0.9863, 3-seed) and 2^16 regressed past the warm baseline (1.0007). The corollary — that non-throughput knobs at fixed batch size are near-saturated — is supported by ten knob pilots all landing within ±1σ of baseline.

**Confidence: 0.85.** The main effect at 2^18 is replicated across 3 seeds with concurrent controls and exceeds the 2σ gate by a wide margin, and the non-monotonic flank behavior matches the gradient-noise prediction. Held below 0.9 because the sub-2^18 flank confounds batch size with device_batch (Section 6), and because "update-count-limited" is inferred from one lever; a step-count-matched ablation (fixed steps, varied batch) has not been run.

## 2. Methods

- **Frame.** Fixed 5-minute wall-clock training budget per run on a single H200; runs launched 7-wide across GPUs of a shared 8×H200 node. Steps completed are an outcome, not a control — any throughput change moves both steps and val_bpb.
- **Metric.** Validation bits/byte (val_bpb) at end of budget.
- **Noise floor.** 3-seed warm baseline (exp008–010, seeds 42/43/44): mean **0.98768**, σ ≈ **0.0015**, ~1017 steps. Single-seed pilots therefore cannot resolve effects below ~0.003; they are used only for screening.
- **Concurrent-control discipline.** Because compile-cache state and node load shift throughput, comparisons are made only against fresh controls launched in the same wave, never against the cold first run.
- **Keep/discard gate.** A change is kept only if a 3-seed paired comparison against the concurrent baseline clears 2σ; single-seed screening results are logged as `pilot` and never promoted directly. Example rejection: matrix_lr 0.05 looked flat as a pilot but was worse on all 3 seeds under the paired test, and was discarded.

## 3. Evidence

Δ val_bpb is reported against the warm 3-seed baseline mean 0.98768 (negative = better). Data: `campaign_log.jsonl`, exp000–exp035.

| Config | val_bpb | Δ vs warm baseline | Seeds | Verdict |
|---|---|---|---|---|
| Naive baseline (cold compile, 671 steps) | 1.0301 | +0.0424 | 1 | invalid control (F1) |
| Warm baseline, TOTAL_BATCH 2^19 (~1017 steps) | 0.98768 | — | 3 | reference |
| matrix_lr ∈ {0.03, 0.05, 0.06} | 0.9862–0.9881 | within ±1σ | 1 (0.05: 3) | no effect; 0.05 rejected 3-seed paired (worse on 3/3) |
| embedding_lr ∈ {0.45, 0.8} | 0.9875–0.9877 | within ±1σ | 1 | no effect |
| warmdown ∈ {0.4, 0.6} | 0.9874–0.9876 | within ±1σ | 1 | no effect |
| weight_decay ∈ {0.1, 0.35} | 0.9870–0.9878 | within ±1σ | 1 | no effect |
| unembedding_lr 0.006 / scalar_lr 0.3 | 0.9864 / 0.9867 | within ±1σ | 1 | no effect |
| Adam β1 = 0.9 | 0.9904 | +0.0027 | 1 | worse, discard |
| warmup 0.05 | 0.9892 | +0.0015 | 1 | worse, discard |
| device_batch 256 | 0.9868 | ~0 (same steps, 2× VRAM) | 1 | no throughput gain, discard |
| depth 6 / 10 / 12 | 1.0222 / 0.9908 / 1.0052 | +0.035 / +0.003 / +0.018 | 1 | worse, discard |
| aspect_ratio 96 (wider) | 1.0338 | +0.046 | 1 | worse, discard |
| all-full-attention window "L" | 0.9908 | +0.003 | 1 | worse, discard |
| TOTAL_BATCH 2^20 (506 steps) | 1.0131 | +0.025 | 1 | worse, discard |
| **TOTAL_BATCH 2^18 (~2020 steps)** | **0.9797** (best 0.9784) | **−0.0080 (~4–5σ)** | **3** | **keep — new reference (F2)** |
| TOTAL_BATCH 2^17, device_batch 64 (~3760 steps) | 0.9863 | −0.0014 | 3 | worse than 2^18, discard |
| TOTAL_BATCH 2^16, device_batch 32 (~6881 steps) | 1.0007 | +0.0130 | 1 | worse, discard |

## 4. Findings

**F1 — Compile-cache confound: the naive baseline is not a fair control.** The very first run compiles the model from a cold inductor cache, paying ~650 ms/step and completing only 671 optimizer steps; every subsequent run of the *identical* code hits the warm cache, runs ~230 ms/step, and completes ~1017 steps, reaching 0.9877. Under a fixed wall-clock frame, this 2.8× throughput difference masquerades as a −0.042 val_bpb "improvement" that would be misattributed to whatever change happened to ship in run 2. Two consequences: (i) all comparisons must use fresh concurrent controls launched after cache warm-up; (ii) "torch.compile (warm cache)" is nonetheless a *real* throughput win versus the naive 1.030 configuration — it is logged as milestone 1, but it is a systems effect, not a modeling effect, and it moves the reference against which all later changes are judged.

**F2 — Update-count-limited regime with a batch-size optimum at 2^18.** With throughput controlled, no per-parameter learning rate, decay, schedule, or width/depth/attention change moved val_bpb beyond noise — the knob surface around the default is flat. The one confirmed lever changes *how the fixed compute is partitioned*: halving tokens per optimizer step (2^19 → 2^18) doubles updates to ~2020 within the budget and improves val_bpb by −0.008 across 3 seeds (0.9797 mean), while doubling tokens per step (2^20, 506 updates) costs +0.025. This asymmetry indicates the default sits on the update-starved side of the batch-size/update-count trade-off. The improvement does not continue: 2^17 (~3760 updates) gives back most of the gain and 2^16 regresses outright, consistent with per-step gradient noise (and, at 2^17–2^16, reduced per-step efficiency from smaller device batches) overtaking the benefit of more updates. The optimum at this frame is 2^18, adopted as the new reference.

## 5. Threats to validity

1. **Single-seed screening.** Most negative results in Section 3 rest on one seed against σ ≈ 0.0015; effects up to ~±0.003 could be hiding in any of them. Only the batch-size result and the matrix_lr 0.05 rejection meet the 3-seed gate. "No effect" claims should be read as "no effect resolvable at n = 1."
2. **Batch-size × device-batch confound below 2^18.** The 2^17 and 2^16 runs also reduced device_batch (64, 32), changing kernel efficiency and per-step time, so the sub-2^18 regression conflates statistical (gradient noise) and systems (throughput) causes. The *location* of the optimum is therefore less certain than its existence.
3. **Seed variance and shared hardware.** σ ≈ 0.0015 is itself a 3-seed estimate; the node is shared, and co-tenant load can shift step counts between waves. Concurrent controls mitigate but do not eliminate this.
4. **LRs not re-tuned at 2^18.** The −0.008 gain was measured at learning rates tuned (implicitly) for 2^19; the true optimum batch size could shift once LRs are re-optimized.

## 6. Next experiments (ranked)

1. **Re-tune learning rates at TOTAL_BATCH 2^18** (matrix_lr, embedding_lr foremost): smaller batches classically want proportionally adjusted LRs; the flat knob surface at 2^19 may not transfer. This also de-risks threat 4. Run 3-seed paired from the start for any pilot flashing ≥1σ.
2. **Batch-size × schedule interaction:** with 2× the updates, warmup/warmdown fractions and final-LR behavior may have a new optimum; test warmdown ∈ {0.4, 0.6} and warmup at 2^18.
3. **Deconfound the lower flank:** run 2^17 at device_batch 128 (grad-accum-free) to separate gradient noise from kernel-efficiency loss and pin the optimum's location.
4. **Structural exploration** (activation function, optimizer variants, attention window schedule) — per the explore-first discipline, scalar-knob tuning is saturated; new mechanisms are the remaining search space, screened single-seed and promoted only on ≥2σ 3-seed confirmation.

## Overall confidence

**0.8.** The two headline findings are well supported: F1 is mechanically verified (same code, 671 vs ~1017 steps, ~650 vs ~230 ms/step), and F2 clears the pre-registered 3-seed 2σ gate several times over with concurrent controls. Confidence is docked for the single-seed basis of most null results, the device-batch confound below 2^18, and the untested interaction between batch size and learning-rate scale, any of which could shift the reported optimum without overturning the update-count-limited interpretation.
