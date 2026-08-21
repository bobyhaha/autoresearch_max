# Literature Synthesis

This document converts the seeded literature corpus into a directional prior
map and a ranked planning queue. It is not a result registry, a gate, or an
experiment authorization. Bibliographic records and atomic claims live under
`research/knowledge/external/`; current local evidence and scope freshness live
in `RESEARCH_STATE.md`.

<!-- BEGIN GENERATED LITERATURE SNAPSHOT -->
<!-- GENERATED: run python -m vibeautoresearch render-literature; do not edit this block -->
## Current registry and scope snapshot

- Planning-input fingerprint: `936092037ed613cb`
- Papers: **36**
- Atomic claims: **123**
- Current literature-evidence records: **122** terminal / **126** append-only (covering **123 / 123** claims)
- Current project mechanisms: **17** (these are not the historical literature-theme IDs)
- Active challenge: **#1 walltime_5min_h200**, status **passed**, selection generation **26**
- Setup record: v40, fingerprint `7e2b06b51256f737`
- Setup integrity now: **verified**
- Scope key: `{"data_split_sha256":"ed8ea0554010df9fbfa47746adc6643d6446797753eb377c0b35413bdcf2fca3","max_steps":100000,"outcome_id":"out_val_bpb","stop_mode":"time","time_budget":300}`
- Frozen baseline: `val_bpb=0.931857`; effective σ `0.001307`

The registries, reconciliation record, and generated `RESEARCH_STATE.md` are authoritative. This synthesis is an editorial planning view and cannot authorize an experiment.
<!-- END GENERATED LITERATURE SNAPSHOT -->

## Authority and intended use

Use sources in this order:

1. `research/setup/reconciliation.json` defines the current experimental world.
2. Structured registries and generated `RESEARCH_STATE.md` define current
   evidence, beliefs, open capabilities, and executable work.
3. This synthesis ranks directions worth turning into mechanisms and
   hypotheses.
4. Papers and unassessed claims are priors only. They are not local evidence.

The ranked queue seeds `track:explore` and `track:refine` proposals, but every
proposal still passes the coverage, scope, noise-floor, and execution gates.
When the synthesis conflicts with a current-scope belief, the belief controls
the local decision and the synthesis must be revised.

The historical `mech_lit_*` taxonomy is intentionally not cited here. Those
records exist only in a pre-reset snapshot and are absent from the live
mechanism registry. Themes below are editorial categories, not executable
mechanism IDs.

## How to read a prior

A directional literature prior should make an experiment sharper, not bias its
verdict:

- Pre-register the predicted direction, causal mechanism, minimum useful effect,
  and transfer assumptions before running.
- Do not spend a pilot merely rediscovering a well-supported sign when the
  intervention and scope closely match. Start at the smallest design capable of
  falsifying the predicted effect.
- Do not “confirm or skip” when transfer is uncertain or a counter-prior exists.
  Run a matched control and update symmetrically whether the result supports,
  challenges, contradicts, or makes the claim context-dependent.
- A one-seed sign check may screen an expensive implementation, but it cannot
  mint a belief. Verdicts use the current paired-seed and noise-floor rules.
- Before a claim motivates a gate, create or update its structured
  `literature_evidence` assessment. Corpus membership alone is not assessment.
- Keep literature expectations separate from measured local outcomes. A paper
  predicts; a registered run decides what happened in this scope.

## Stable literature map

These categories organize retrieval from the paper and claim registries. Their
directional statements are broad priors whose transfer must be assessed for the
specific proposal.

1. **Matrix and second-order optimization.** Shampoo, K-FAC, Sophia,
   AdaHessian, Adafactor, SOAP, Muon, and related whitening methods trade
   additional optimizer work for better-conditioned updates.
2. **Adaptive scaling, momentum, and batch size.** Adam-family variants,
   LAMB/LARS, Adan, schedule-free methods, and critical-batch analyses predict
   strong coupling among learning rate, momentum, and batch scale.
3. **Learning-rate schedules.** Warmup, stable phases, cooldown shape, and a
   nonzero terminal floor can matter disproportionately in short training
   regimes.
4. **Scaling laws and hyperparameter transfer.** Scaling-law and µP work
   predicts that proxy tuning can replace some expensive target-scale search,
   subject to parameterization and optimizer compatibility.
5. **Normalization.** LayerNorm/RMSNorm placement and scale control change
   optimization stability and effective learning rates.
6. **Position and length.** Relative/rotary schemes and extrapolation methods
   predict trade-offs between positional inductive bias and long-context
   transfer.
7. **Attention cost and connectivity.** FlashAttention, grouped/latent
   attention, sparse attention, and locality methods can reduce resource cost,
   but connectivity changes can alter quality.
8. **FFN activation, gating, and MoE.** Gated activations and sparse experts
   often improve large-model quality per resource unit, but drop-in transfer to
   a tightly tuned small model is not guaranteed.
9. **Initialization and residual scaling.** Residual scale, depth
   parameterization, and zero-initialized branches control early signal and
   gradient propagation.
10. **Regularization.** Weight decay, dropout, stochastic depth, z-loss, and
    sharpness methods interact with schedule and data repetition.
11. **Data, curriculum, packing, and tokenization.** Data order and information
    density determine useful learning signal per fixed token.
12. **Stability and low precision.** Bounded logits/activations, BF16/FP8
    recipes, and spike controls can enable more aggressive or cheaper training.
13. **Training dynamics.** Edge-of-stability, feature learning, grokking, and
    catapult analyses offer observables for deciding when a mechanism is active.

Recent frontier reports are useful for generating candidates, but are heavily
confounded system recipes. Reproducible base-model ablations closer to this
model and token regime receive more transfer weight than trillion-parameter
adoption reports.

## Current local update to the priors

The live beliefs have already changed the literature-only ordering:

- Fixed steps/tokens are now the frozen quality estimand. Throughput is a
  separate resource axis, not a way to obtain more tokens inside the primary
  score.
- Added n-gram capacity and fourth-order context helped, but capacity saturates
  at the current operating point and fifth-order context is data-starved at the
  current budget.
- Full in-document context matters locally: adding a sliding window hurt.
  Efficient-attention proposals should preserve the winning connectivity rather
  than assume locality is free.
- The current dense segmented-attention implementation improves quality but
  carries a large step-time cost. That is an implementation frontier, not proof
  that document segmentation is intrinsically slow.
- Drop-in SwiGLU, residual scaling, tied embeddings, optimizer perturbations,
  and nearby schedule/regularization refinements were null or harmful in the
  current scope.
- Learned product-key memory failed at this budget, with unresolved ambiguity
  between architectural mismatch and insufficient bootstrap time.

These are summaries for planning. Exact statuses, limitations, and evidence IDs
must be read from `RESEARCH_STATE.md` before creating a hypothesis.

## Ranked bet-list

Ordered by expected information per GPU-hour under the active
`walltime_5min_h200` objective. Every endpoint must use a fresh concurrent
same-seed control; the operational `0.934930` is planning context, not a
substitute control. “Explore” means structurally novel or high uncertainty;
“method” improves the validity or interpretability of future experiments.

1. **[method] Reconcile and re-establish executable scope.** Deliberately
   reconcile the changed training code before any new evidence is registered.
   Then run the provenance-complete, paired, counterbalanced Track B protocol so
   quality and resource claims share exact code and hardware identity.
2. **[explore] Fused or block-sparse document-segmented attention.** Preserve
   full causal attention within each packed document while avoiding the dense
   batch-by-sequence-squared boolean mask. This directly attacks the largest
   measured cost of a validated quality lever.
3. **[explore] Information density with wall-time attribution.** Test deliberate
   curriculum, packing, deduplication, or quality-order policies with fresh
   concurrent controls and unchanged charged time. Plain reshuffling was
   sub-threshold; the next policy must predict a larger mechanism-linked effect
   and record both token exposure and intrinsic quality mediators.
4. **[explore] µP/µTransfer as search infrastructure.** Assess the relevant
   claims, define a proxy-to-target parameterization contract, and test whether
   transferred settings predict the target optimum. The payoff is fewer target
   H200 searches, not a presumed direct loss win.
5. **[explore] Learned memory with an explicit bootstrap.** If sparse learned
   retrieval is revisited in the five-minute frame, initialize or distill it so
   values and routing carry useful signal immediately. Repeating the zero-value,
   slow-routing product-key design does not answer a new question.
6. **[explore] Connectivity-preserving efficient attention.** GQA/MLA/native
   sparse candidates must separate intrinsic quality, charged throughput, and
   endpoint `val_bpb`. More tokens in 300 seconds are a valid mediator in this
   frame, but they do not replace the registered endpoint or its concurrent
   control.
7. **[method] Repair causal observables.** Make diagnostic loss byte-weighted to
   match `val_bpb`, and add context-frequency/collision observables for n-gram
   mechanisms. This raises the information yield of the next architecture run.

### Do not spend current-scope refinement slots on

- More random hash capacity beyond the measured knee.
- Fifth-order hash embeddings without a larger data budget or a generalizing
  mechanism.
- Nearby sparse-embedding LR, schedule-floor, warmup, or weight-decay nudges.
- Sliding-window attention that removes useful in-document context.
- Repeating failed drop-in activation/residual/optimizer changes without a new
  mechanism and their own retuning contract.

### Valuable but outside the active scope

- A longer-token-budget test of learned retrieval and fifth-order context.
- Fixed-step quality-only optimization that removes the active frame's
  throughput-to-token-exposure path.
- Scale-transfer conclusions requiring multiple model sizes.

## Counter-priors and claim revision

Conflicts such as negative Muon comparisons versus frontier adoption reports
are useful only when made testable. For any counter-prior:

1. identify the claims and assess their design, scope, and transfer;
2. state which local intervention distinguishes the competing mechanisms;
3. use a matched, current-scope experiment rather than comparing headline
   numbers across papers;
4. update the local belief with `challenged`, `contradicted`,
   `context_dependent`, or `inconclusive` when warranted.

Adoption by a frontier model is evidence that a method can operate at scale; it
is not a controlled estimate of its marginal benefit here. Conversely, a
negative result at another scale does not prove local failure.

## Maintenance contract

- The generated snapshot is refreshed with:

  ```bash
  python -m vibeautoresearch render-literature
  ```

- `validate` and `audit` warn when its counts, scope, setup integrity, or
  planning-input fingerprint drift.
- A changed fingerprint is a prompt for human review of the ranked queue; merely
  regenerating the block does not certify the editorial conclusions.
- Do not hardcode registry counts or a baseline plateau elsewhere in this file.
- Review the queue after any scope reconciliation, material belief revision,
  capability-gap change, or substantial literature-ingestion pass.
- The synthesis may seed a proposal, but only structured mechanisms,
  hypotheses, evidence assessments, and gates can authorize a run.

_Editorial strategy revised 2026-07-23._
