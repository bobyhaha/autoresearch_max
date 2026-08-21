<!-- GENERATED FILE: edit registries, then run python -m vibeautoresearch render-state -->
# Current Research State

This view is generated from the structured registries. It contains no raw results.

## Inventory

- papers: 36
- claims: 123
- literature evidence: 126
- run evidence: 38
- observations: 3
- beliefs: 6
- observables: 13
- interventions: 118
- contexts: 1
- outcomes: 1
- tool proposals: 0
- capability gaps: 2
- mechanisms: 17
- hypotheses: 21
- idea archive: 17
- hourly reports: 17
- experiment proposals: 1
- gated experiments: 24
- runs: 101
- evidence updates: 28
- audits: 5
- decisions: 36
- deprecations: 0
- campaign batches: 41
- current evidence: 153 terminal / 164 append-only records
- setup reconciliation: 1

## Setup reconciliation

- Active challenge: **#1 walltime_5min_h200** (selection generation 26, action `activated`)
- Active challenge status: **passed**
- Fingerprint: `7e2b06b51256f737`
- Scope key: `{"data_split_sha256":"ed8ea0554010df9fbfa47746adc6643d6446797753eb377c0b35413bdcf2fca3","max_steps":100000,"outcome_id":"out_val_bpb","stop_mode":"time","time_budget":300}`
- Reference-code comparison: **fork_with_differences** (+3447 /-176 lines across 130 hunks)

### Secondary decision frames

- `walltime_5min_h200` [adopt/passed]: scope `{"data_split_sha256":"ed8ea0554010df9fbfa47746adc6643d6446797753eb377c0b35413bdcf2fca3","max_steps":100000,"outcome_id":"out_val_bpb","stop_mode":"time","time_budget":300}`; run env `{"ATTN_BACKEND":"fa3","COMPILE_MODE":"max-autotune-no-cudagraphs"}`

## Current verified beliefs

1 scope-matched, evidence-backed belief records; 4 superseded records remain in append-only history.

- `blf_docmask_walltime_stage10_confirmed` [strongly_supported]: Ten prospectively governed and physical-role-balanced pairs strongly support an end-to-end FA3 varlen document-isolation benefit in the active wall-time frame. The method is an adoption candidate, not an adopted default, until clean-directory replay and reproducibility review complete.

## Demoted beliefs

- 0 terminal beliefs are stale/unscoped for the current active-challenge scope; they remain in append-only history.
- 1 current-scope beliefs are provisional because they lack structured evidence or at least three completed seeds.

- `blf_token_law_fixed_batch_provisional` [unverified]

## Provenance gaps

- Every current non-speculative belief cites structured evidence.
- Every literature claim has a literature-evidence assessment.

## Hypotheses and gates

Hypotheses without conclusive supporting, opposing, or mixed evidence:

- `hyp_block14_attnres_walltime_paired` [proposed]: Under the active walltime_5min_h200 frame's exact scope, AttnRes's mean paired delta across 3 seeds is directionally consistent with cycle 5's single-seed +0.0219 (confounded by a ~26% step-count loss), i.e. still net-negative in this wall-clock frame even though the underlying per-update quality question (addressed by a true fixed-step design) remains separately open pending explicit authorization to use the fixed_steps_2000 challenge, whose own scope reconciliation is not yet passed.
- `hyp_block14_c1_10sh_bracket` [proposed]: 10-shard delta in [-0.005, +0.003] for lambda in {0.02, 0.05}, an order of magnitude smaller harm than lambda=0.2's measured +0.0272, if harm scales roughly with lambda strength.
- `hyp_block14_c1_c3_redundancy` [proposed]: At 5 shards, combined delta in [-0.095,-0.070], detectably short of the naive additive sum (-0.1168, from run_c1sweep_wd002's -0.0954 + run_c3_k50_data5_treatment's -0.0214) but beating either lever alone. At 10 shards, combined delta within +/-0.004 of baseline (both components individually near-neutral there).
- `hyp_block22_legacy_recipe_control_transport` [proposed]: Across seeds 47-50, the current-source recovered recipe is paired-TOST equivalent to the same-seed historical treatment endpoints within +/-0.002396 BPB, and its concurrent one-sided 95% upper bound versus the explicit current recipe is <= -0.002396 with 4/4 pairs helping. The 90% interval of historical-token log ratios remains inside [log(0.98),log(1.02)].
- `hyp_paper019_fa3_boundary_sidecar` [proposed]: Conditional on exactness, phased state parity, compiler identity, and H200 event-lifetime prerequisites, one placement either kills the sidecar or reaches point ratio >=1.054 and one-sided 95% LCB >=1.041, which requires a separately frozen role-swapped profile. Only a counterbalanced pass may make the ordinary endpoint proposal eligible for the normal typed gate.

## Toolkit

- observables: 4 executable / 13 total
- interventions: 115 executable / 118 total
- contexts: 1 executable / 1 total
- outcomes: 1 executable / 1 total

## Capability gaps

- `gap_fa3_varlen_static_cu_seqlens` [high]: Can the per-step torch.nonzero host synchronization be removed from the adopted fa3 varlen doc-masking path by building cu_seqlens at a static shape?

## Approved or running experiments

Approved gates with no conclusive evidence update yet (invalid attempts remain unresolved):

- `exp_block19_docmask_provenance_s1` [discovery/approved] budget cap=0.0
- `exp_block22_legacy_recipe_transport_q4_v2` [validation/approved] budget cap=0.0
- `exp_paper019_fa3_boundary_sidecar_s1` [discovery/approved] budget cap=0.0

## Next action

- Assess the literature claims needed by the next headroom question; unassessed claims cannot pass `check-gate`.
- Create an active-challenge, scope-keyed proposal with a concurrent control, the required staged seeds, and that challenge's effective noise floor.
- Run `python -m vibeautoresearch audit` and resolve provenance gaps before treating any demoted belief as current.
