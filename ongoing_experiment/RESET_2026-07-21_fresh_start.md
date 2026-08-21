# Fresh start (2026-07-21)

Clean-slate reset to begin a new OPHIS run from the `rsi/daniel` baseline
(github.com/dyu056/vibeautoresearch), following the Recursive nanoGPT setup
adapted to a single H200.

## Frozen challenge (new scope)
- Fixed **2000 steps** on one H200 (`STOP_MODE=steps`, `MAX_STEPS=2000`), `ATTN_BACKEND=sdpa`.
- Data: train shards `1-10`, val shard `6542`; `val_bpb` (lower is better).
- **Baseline MEASURED on the target H200** (GPUs 4-7, MAX_STEPS=2000, sdpa, seeds 42/43/44):
  **mean val_bpb = 0.933386**; per-seed 0.933156 / 0.934170 / 0.932831; seed-to-seed
  σ = 0.000698 (n=3); σ_repro = 0.000593 (seed42 ×2, crude). **Effective decision floor
  = 0.0007.** `pending_h200_measurement=false`.
- External benchmark: RSI `0.9109` (different sandbox; a threshold, not an internal control).
  (The OPHIS write-up's 0.9341 placeholder is superseded by the measured 0.933386.)

## Baseline = rsi/daniel hyperparameters (config only)
Reverted 5 contaminated-session knobs in `train.py` to Daniel's values; model/architecture
params were already identical. **No observables ported** — the observable set starts fresh.

| param | was (baiyu) | now (rsi/daniel) |
|---|---|---|
| MATRIX_LR | 0.0405 | 0.04 |
| DEMON_FINAL_BETA1 | 0.60 | 0.55 |
| NGRAM_VE_BETAS | (0.5, 0.99955) | (0.5, 0.999) |
| NGRAM_VE_LR_SCALE | 1.375 | 1.0 |
| FINAL_LR_FRAC | 0.02 | 0.05 |
| MAX_STEPS default | 1660 | 2000 |

## Ledger reset (ledger-only; knowledge kept)
- WIPED (backed up locally; append-only history not on this tree): all beliefs (63),
  mechanisms (70), hypotheses (37), gated/proposed experiments (46), runs (73),
  run-evidence (46), evidence-updates (46), decisions (47), audits (1), observations (1),
  and every prior `ongoing_experiment/*.md` (61 logs).
- KEPT: 349 papers, 1012 literature claims, 37 toolkit interventions, tool definitions.

## Method fixes (from the Fable review of the OPHIS write-up)
`docs/COST_AND_GATES.md` + `docs/AGENT_PROTOCOL.md`:
- **6:4 explore:refine** made firm (≥6/10 trailing explore, max-2 refine streak).
- **Correct σ**: gate on seed-to-seed + temporal heterogeneity (σ≈0.00087), not same-seed
  repro noise. (The write-up's 7.43σ re-scores to ~2.6σ on the right σ.)
- **Selection discipline**: report N; multiplicity-corrected screen threshold; held-out
  confirmation of winners on fresh pre-registered seeds; a never-touched test split;
  variance-aware selection (10 *distinct* seeds, not 10 reps of seed 42); round to noise floor.
- **screen → confirm → hold-out** workflow at the 2000-step decision budget; per-step
  compute parity (fixed steps ≠ fixed FLOPs).
- **Belief freshness**: scope-keyed auto-demotion; build only from recent current-scope
  beliefs; never resurrect a demoted belief without re-confirming it under the current scope.

## Verified
`py_compile` OK · `validate` OK (0 warnings) · `check-setup` passed (max_steps 2000) ·
`audit` 0 findings · **baseline reproduced on the H200** (4 runs, all EXIT_CODE=0 at step 1999).
Observable registry seeded to 11 records (current-tree emissions; no Daniel probes).
Remote work dir: `/home/user/ai4ai/temp_autoresearch_baiyu_freshstart` (logs retained as artifacts).
