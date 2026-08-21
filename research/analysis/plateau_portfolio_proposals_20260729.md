# Plateau Portfolio Proposals — Pre-Critic Freeze

- Frozen at: 2026-07-29T19:08:00Z
- Status: proposal material only; not a registry record, paper authority, intervention,
  experiment, comparator change, or launch authority
- Generation: two independent read-only proposer agents
- Scoring order: novelty / primary-source provenance / causal validity /
  expected local impact / transfer reliability / feasibility-low-cost /
  numerical falsifiability / connected-program coherence (N/P/V/I/R/F/X/C),
  each on a 1–5 scale
- Comparator rule: an accepted parent becomes the next concurrent control; a
  rejected parent leaves the comparator unchanged; no first exposures are bundled
- Historical rule: 0.927183 remains the chart origin only

This file freezes the material supplied to an independent adversarial critic. It
does not imply that any proposer claim is correct. Exact source claims, local
scope, arithmetic, compiler behavior, and dependencies remain subject to the
critic and any later paper-bound premise assay.

## Portfolio A — Residual, normalization, initialization, and optimization

| Proposer rank | Candidate | Dependency | Proposer N/P/V/I/R/F/X/C | Total | Proposer disposition |
|---:|---|---|---|---:|---|
| 1 | Split attention/MLP GPAS gates | Full GPAS acceptance | 4/4/5/4/3/5/5/5 | 35 | Send to critic |
| 2 | Gradient-preserving RMS branch cap | GPAS accepted; fixed Peri arm safety-killed or rejected | 5/4/5/4/3/4/5/5 | 35 | Conditional send |
| 3 | Frozen conditioned spectral anchor | Conditioned Q/K/V accepted | 5/5/4/3/2/4/5/5 | 33 | Send to critic |
| 4 | Learnable Peri-RMSNorm gamma | Fixed Peri attention arm accepted | 3/5/4/3/3/5/5/5 | 33 | First reserve |
| 5 | Early-adapt, late-hold GPAS optimizer | GPAS accepted and late gate drift observed | 3/5/4/3/2/5/5/5 | 32 | Reserve |
| 6 | Tangent-projected Q/K Muon | Conditioned initialization accepted and Q/K geometry drifts | 5/4/4/3/2/3/5/5 | 31 | Cost-risk reserve |
| 7 | Full-state headwise attention gate | Current gate has a sink/noise premise | 3/5/4/2/2/5/5/4 | 30 | Likely redundant |
| 8 | Manifold-constrained x0 writeback | Independent | 2/4/3/3/1/4/5/4 | 26 | Duplicate; do not regenerate |

### A1. Split attention/MLP GPAS gates

- Mechanism: the current local GPAS port reuses one layer scalar after two
  structurally different residual additions. Attention carries hash/value memory
  and head gating, while the MLP output is already RMS-normalized. Separate
  zero-initialized attention and MLP scalars preserve the GPAS stop-gradient
  identity-backward construction while avoiding a compromise forward scale.
- Premise: over 32 eager training steps, virtual gradients for the two candidate
  gates must disagree in at least six layers, defined prospectively as opposite
  signs or normalized magnitude difference at least 0.25.
- Mediator: activated gates differ by at least 0.02 in at least four layers and
  reduce the attention-versus-MLP variance-increment imbalance by at least 20%.
- Falsifier: the gates separate and improve the mediator, but fail the active
  governed endpoint floor of -0.002396 raw paired val_bpb.
- Claimed provenance: GPAS, NeurIPS 2025, Section 3.1 Equation 2; Section 4.1
  shares one gate across both sublayers; Section 5.1 reports depth- and
  time-varying gates; Appendix A Tables 4–5 isolate placement and stop-gradient.
  The preserved official code is commit 31980688f4cbb1b0cff59bca9077e6fc52dab3f0.
- Caveat: the split was not tested by the source and doubles the GPAS scalar
  degrees of freedom per layer.

### A2. Gradient-preserving RMS attention-branch cap

- Mechanism: combine GPAS forward scaling with Peri-LN output-scale control
  without amplifying a near-zero `c_proj` branch. For attention branch `a`,
  residual `x`, and a prospectively frozen cap `c`, use
  `s=min(1,c*RMS(x)/(RMS(a)+epsilon))` and
  `a'=a+(stopgrad(s)-1)*stopgrad(a)`. The forward pass caps oversized branches,
  the derivative with respect to `a` is the identity, and small branches are
  never amplified.
- Premise: attention-branch/residual RMS has a max/min layer ratio at least 1.25
  or upper-tail ratio at least 1.5, and branch spikes correlate at least 0.5 with
  residual-variance increments.
- Mediator: at least 50% reduction in upper-tail branch-RMS excess, branch
  gradient scale within 10% of control, and zero amplification events.
- Kill/falsifier: kill for no premise, graph/recompile drift, or token-rate ratio
  below 0.99; falsify efficacy if the cap mediates spikes without endpoint gain.
- Claimed provenance: GPAS Section 3.1 for gradient-preserving forward scaling;
  Peri-LN, ICML 2025, Section 3.3 Equation 3 and Proposition 3.1 for output-scale
  control; Peri-LN Section 5.3 and Figures 8–9 for fixed-gamma activity.
- Caveat: neither source tests this adaptive cap. It is eligible only if a fixed
  Peri attention arm exposes the predicted local amplification problem.

### A3. Frozen conditioned spectral anchor

- Mechanism: after an accepted conditioned Q/K/V initialization, preserve the
  initialized tensor as frozen base `B`, train a zero-initialized residual
  `Delta`, and use `W_eff=B+Delta`. Muon and weight decay act only on `Delta`.
- Premise: during accepted conditioned-initialization training, median per-head
  Q/K condition number exceeds 1.5, orthogonality residual grows by at least 25%,
  or the registered Jacobian surrogate regresses by at least 10% during steps
  64–256.
- Mediator: retain at least 50% more of the initial spectral advantage, preserve
  attention entropy, and retain token-rate ratio at least 0.995.
- Falsifier: geometry is preserved but endpoint quality is null or harmful.
- Claimed provenance: Conditioned Initialization for Attention, ICLR 2026,
  Theorem 3.1, Proposition 3.2, Section 3.3, and Appendix A.2; Spectral
  Conditioning of Attention, NeurIPS 2025, Definition 3.6, Theorem 3.8,
  Section 4.4 Table 4, and Appendix Equations 59–67.
- Caveats: the spectral source uses fixed `lambda I` with `lambda=10`, not a
  frozen semi-orthogonal base; its language evidence is masked-LM/GLUE rather
  than causal-LM BPB; the proposal retains an extra frozen model-sized buffer.

### A4–A8. Reserves and casualties

- Learnable Peri-RMSNorm gamma: after a fixed Peri arm is accepted, add
  per-channel attention/MLP gamma initialized to one. Require stable nonzero
  virtual-gamma gradients and at least 15% better branch-gradient/variance
  balance. Claimed source: Peri-LN Section 5.3 and Figure 9b. Falsify if gamma
  moves and balances branches without endpoint gain.
- Early-adapt, late-hold GPAS optimizer: preserve GPAS learning early, then
  decay gate learning rate to zero only if at least 80% of accepted gate movement
  occurs before 20–25% training progress and later drift is harmful. Claimed
  source: GPAS Section 5.1, Section 5.3, Section 6, and Appendix Table 5.
- Tangent-projected Q/K Muon: after measured conditioned-initialization drift,
  project each per-head update with
  `U_T=U-sym(U W^T)W`, then Frobenius-rescale. Require radial update fraction at
  least 0.15 and kill above 1% slowdown. Do not reuse the locally harmful
  Newton–Schulz iteration reduction.
- Full-state headwise gate: change only the current gate input from the first 32
  hidden channels to the full normalized state while retaining six head outputs,
  zero initialization, `2*sigmoid`, and placement. Claimed source: Gated
  Attention, NeurIPS 2025 Best Paper, Section 2.2 Equation 5, Table 1,
  Section 4.2 Table 4, and Section 4.3. Transfer is weak because OPHIS already
  has head gating, QK/output normalization, and document isolation.
- Manifold-constrained x0 writeback: veto as a duplicate of a local proposal.
  Current primary evidence is only abstract-level arXiv 2512.24880, while local
  AttnRes/Canon results provide a poor residual-addition prior.

## Portfolio B — Memory addressing, document semantics, and conditional lookup

The proposer screened the live memory/data/system space after excluding the
already frozen Paper-020 prime/K=4 candidates and prior killed or played-out
families: canonical keys, unbudgeted random capacity, fifth-order tables,
C1/C3, rowwise optimizer state, eager sparse gradients, and fused hash
construction.

| Proposer rank | Candidate | Dependency | Proposer N/P/V/I/R/F/X/C | Total | Proposer disposition |
|---:|---|---|---|---:|---|
| 1 | Document-bounded suffix keys | Current accepted document-isolated memory comparator | 4/5/5/4/4/5/5/5 | 37 | Send to critic |
| 2 | Constant-parameter Zipf head / hashed tail | Boundary repair terminal; Paper-020 prime branch terminal | 5/4/5/4/3/3/5/5 | 34 | Send to critic |
| 3 | Zero-no-op Engram semantic-agreement gate | Boundary/address parents terminal | 3/5/4/4/3/3/5/5 | 32 | Send to critic |

### B1. Document-bounded suffix keys

- Local observation: `train.py` shifts raw token IDs to construct n-gram keys
  across packed BOS boundaries, while `lib.py` packs multiple BOS-started
  documents and attention uses BOS-derived isolation. Thus some BOS and
  first-content-token keys can include an unrelated preceding document that
  attention cannot see.
- Intervention: reset missing suffix positions inside each BOS-defined document
  to BOS. At BOS, both previous positions are BOS; at the first content token,
  the second previous position is BOS. Data, targets, table sizes, moduli,
  parameters, and RNG remain unchanged.
- Offline premise: on the first eight pinned training shards, eight-fold
  shard-held-out byte-weighted count modeling must show
  `affected_byte_mass * (BPB_raw_crossdoc - BPB_BOSpad) >= 0.0030`, positive on
  at least seven of eight held-out shards.
- Training mediator: exact step-zero non-memory equality; then 256 training-only
  steps must show affected-row updates per key at least 2x control, affected-row
  gradient directional dispersion at least 20% lower, one graph, zero recompiles,
  and token-rate ratio at least 0.995.
- Falsifier: cross-document fragments are too rare or more predictive than BOS
  padding, or the mediator improves without endpoint gain.
- Claimed provenance: DeepSeek Engram, ACL 2026, Section 2.2 Equations 1–2 for
  suffix n-grams; Tensorizing Engram Section 4.1 page 4 for left padding when
  context extends before a sequence boundary. The local document-isolation
  mechanism has prior ten-pair support, but that evidence does not establish
  this key repair.

### B2. Constant-parameter Zipf head / hashed tail addressing

- Mechanism: reserve one eighth of each accepted table for top training-only raw
  n-grams through a frozen minimal-perfect-hash plus a 64-bit membership
  fingerprint; map the tail to the remaining rows. Total rows, widths,
  parameters, optimizer elements, and dtypes are unchanged.
- Offline premise: the selected head covers at least 50% of occurrence mass,
  has zero head aliases, reduces occurrence-times-target-JS collision burden by
  at least 40% overall, increases tail burden by at most 10%, and has an
  eight-shard bootstrap upper bound no greater than 0.75 on total burden ratio.
- Training mediator: one graph, zero recompiles, slowdown at most 0.5%, measured
  head alias rate zero, and at least 25% lower within-row per-key gradient
  conflict on head accesses.
- Falsifier: Zipf mass does not dominate collision burden, or collision removal
  fails the endpoint.
- Claimed provenance: Engram Section 2.2 Equations 1–2 for deterministic
  multi-head hashing; Section 2.3 for collision noise; Section 2.5 page 6 for
  Zipfian access and tiering frequent patterns.
- Caveat: the source tiers storage latency, not trainable address semantics.
  This constant-parameter trainable partition is a disclosed local extrapolation.

### B3. Zero-no-op Engram semantic-agreement gate

- Mechanism: retain the current per-head gate and multiply it by
  `2*sigmoid(<RMSNorm(x_slice),RMSNorm(Wk e)>/sqrt(32))`, where `e` is the
  existing retrieved order vector and the per-site map `Wk:768->32` is
  zero-initialized so the new factor is exactly one at step zero.
- Offline premise: after accepted address semantics, at least 10% byte mass lies
  in multi-key/high-conflict rows and the frozen occurrence-times-target-JS
  recoverable burden is at least 0.0030 BPB.
- Training mediator: exact step-zero logits and old gradients; finite nonzero
  new gradients; over 256 training-only steps, gate IQR at least 0.10,
  high-conflict mean gate at least 0.05 below low-conflict, Spearman correlation
  between gate and row-key target compatibility at least 0.10, one graph, zero
  recompiles, and throughput ratio at least 0.99.
- Falsifier: the learned gate is nonselective, has the wrong mediator sign, costs
  more than 1%, or is selective without endpoint benefit.
- Claimed provenance: Engram Section 2.3 Equations 3–5 for hidden-query versus
  memory-key agreement and value gating; Section 6.2/Figure 5 for the
  context-aware-gating ablation; Section 6.5/Figure 7 for selective activation.

## Proposed connected orders before criticism

- Residual branch: accepted shared GPAS → split GPAS premise/child; accepted
  conditioned Q/K/V initialization → drift premise → frozen spectral anchor;
  GPAS accepted plus fixed Peri rejected for amplification → RMS cap.
- Memory branch: document-bounded suffix repair → constant-parameter Zipf
  addressing → semantic-agreement gate.
- Cross-branch rule: do not combine the two branches as one first exposure.
  A future paper may order their premise assays by information gain and cost,
  but each accepted mechanism must become an explicit comparator before its
  dependent child.

## Independent adversarial criticism

- Completed at: 2026-07-29T19:15:00Z
- Status: read-only critic result; no registry, implementation, paper, or run
  authority
- Scope checked: local `train.py` and `lib.py`, Paper 020, preserved primary
  sources, registered claims/evidence, and prior local negative results

| Critic rank | Candidate | Critic N/P/V/I/R/F/X/C | Total | Critic disposition |
|---:|---|---|---:|---|
| 1 | B1 Document-bounded suffix keys | 4/4/4/3/4/4/5/5 | 33 | Keep after premise-spec amendment |
| 2 | A1 Split attention/MLP GPAS | 4/4/4/3/3/4/5/5 | 32 | Conditional on shared GPAS acceptance |
| 3 | A4 Learnable Peri gamma | 3/5/4/2/3/4/4/5 | 30 | Conditional on fixed Peri acceptance |
| 4 | A2 Gradient-preserving RMS cap | 5/3/4/2/2/3/5/5 | 29 | Hold; only after a specific Peri amplification failure |
| 5 | B2 Zipf head / hashed tail | 5/3/3/3/2/2/5/5 | 28 | Hold; provenance and exact-membership repairs required |
| 6 | B3 Semantic-agreement gate | 3/4/2/3/2/2/5/5 | 26 | Withdraw current specification; correct and re-criticize |
| 7 | A5 Early-adapt, late-hold GPAS | 3/3/3/2/2/5/4/4 | 26 | Veto as currently justified |
| 8 | A7 Full-state headwise gate | 2/4/3/2/2/5/4/3 | 25 | Defer; likely redundant |
| 9 | A6 Tangent-projected Q/K Muon | 5/2/1/2/1/2/4/4 | 21 | Hard veto |
| 10 | A3 Frozen conditioned anchor | 4/2/1/1/1/3/4/4 | 20 | Hard veto |
| 11 | A8 Manifold-constrained x0 writeback | 1/3/1/2/1/3/4/2 | 17 | Hard veto; exact duplicate |

### Critic corrections and mechanism verdicts

- B1 identifies a real code-level semantic defect: packed internal BOS positions
  can receive suffix keys containing tokens from an attention-invisible previous
  document. Its proposed cross-fit score is nevertheless a proxy, not a neural
  BPB lower bound. A future paper must freeze smoothing, fallback, byte weighting,
  held-out grouping, and semantically matched exposure/dispersion comparisons.
- A1 is a local extrapolation because GPAS deliberately shares the scalar across
  attention and FFN. Splitting also changes Adam's nonlinear normalization: two
  component gradients are not equivalent to one summed gradient. Eligibility
  requires virtual insertion gradients measured without changing the accepted
  shared-GPAS forward path, an absolute-signal floor, frozen consistency rules,
  and exact optimizer/compiler parity.
- A4 has the strongest direct reserve provenance, but only after fixed Peri-LN is
  accepted. Joint attention and MLP per-channel gammas add 12,288 local
  parameters. A future definition must freeze whether both sites are one
  intervention, bind the optimizer, require gamma-one step-zero identity, and
  collect disjoint training-only virtual-gradient evidence.
- A2 has correct stop-gradient algebra but no direct paper evidence for the
  adaptive cap. The cap is an open tuning surface until derived deterministically
  from a disjoint training-only calibration trace. Only a fixed-Peri safety
  failure specifically attributed to small-branch amplification can unlock it.
- B2's cited Engram result motivates cache/storage tiering, not collision-free
  trainable addressing. A 64-bit fingerprint cannot prove zero false acceptance.
  Any revision needs exact tuple membership, a serialized complete MPHF map, all
  non-trainable-byte accounting, and the frozen tail-capacity penalty.
- B3 has an inventory error. The depth-eight local model has seven injection
  sites—four bigram and three trigram—with one concatenated 768-dimensional
  vector per site, backed by fourteen half-tables. Seven `768→32` maps therefore
  add 172,032 weights. The proposal must correct this count, RMSNorm epsilon,
  optimizer membership, new-gradient bound, and a high-conflict training-loss
  counterfactual before re-criticism.
- A5 over-interprets observational early gate movement. The source's fixed-gate
  and clipping results do not support late freezing, and choosing a freeze time
  after a favorable trajectory is post-selection leakage.
- A7 changes an already head-gated architecture and adds 35,328 weights while
  preserving locally extrapolated scale and placement. It needs a residual
  attention-sink premise plus held-out evidence that omitted channels predict
  suppression beyond the first 32.
- A6 is not a manifold-preserving optimizer: tangent projection preserves
  row-orthogonality only to first order, finite updates drift at order
  learning-rate squared, and Frobenius rescaling is not a Stiefel retraction.
  A proper QR/Cayley method would be a distinct, costlier intervention that
  conflicts with the fused shape-stacked Muon path.
- A3 is symbolically disproved as an anchor. With `W_eff=B+Delta`, unconstrained
  updates of `Delta` are unconstrained updates of `W_eff`; absent weight decay,
  the parameterization has the same trajectory as ordinary conditioned weights.
  It also adds 14,155,776 frozen elements and per-forward additions.
- A8 exactly duplicates the existing mHC-lite mechanism/hypothesis. Row softmax
  is row-stochastic, not generally doubly stochastic, and the local
  residual-addition prior is poor.

### Critic dependency order

```text
Paper-020 terminal closure
├── shared GPAS accepted
│   └── A1 split-gate premise → child
├── fixed Peri accepted
│   └── A4 learnable gamma
├── fixed Peri safety-killed specifically for amplification
│   └── A2 calibrated non-amplifying cap
└── terminal memory comparator
    └── B1 boundary-consistent suffixes
        └── B2 exact Zipf addressing
            └── B3 corrected residual-ambiguity gate
```

A4 and A2 are mutually exclusive branches. A killed round remains empty; its
diagnostic outcome cannot be used to substitute another candidate into the same
paper. The critic's future-paper shortlist is limited to B1, A1, and A4. B1 is
the only independent near-term candidate, and it is not yet paper-ready until
the premise specification is amended and independently checked. A1 and A4
remain ineligible until their Paper-020 parents are accepted.

## Independent primary-source and local-code verification

- Completed at: 2026-07-29T19:22:00Z
- Status key: C = cited source/locator is correct; P = partial support or local
  transfer; W = source does not support the proposed mechanism
- Rule: venue and organization quality count only toward provenance. They do
  not repair scope mismatch or establish local causal validity.

| Candidate | Verification | Material source correction or transfer boundary |
|---|---|---|
| A1 Split GPAS | C source / P candidate | GPAS Sections 3.1, 4.1, 5.1 and Appendix Tables 4–5 support the shared scalar, placement, dynamics, and ablations. The source deliberately shares one scalar; splitting and separate Adam normalization are untested. |
| A2 RMS cap | P | Peri-LN Proposition 3.1 is in Section 3.4, not Section 3.3, and bounds a last-layer MLP/RMSNorm gradient rather than attention or adaptive clipping. Figure 8 is frozen gamma and Figure 9b is learnable gamma. |
| A3 Frozen anchor | P source / W candidate | Conditioned Initialization construction is Appendix A.1.1; Appendix A.2 covers normalization/partial initialization. Proposition 3.2 explicitly does not prove that the true Jacobian improves. Spectral Conditioning implements fixed `lambda I`, `lambda=10`, not a frozen semi-orthogonal base. |
| A4 Learnable Peri gamma | C source / P transfer | Peri-LN Section 5.3 and Figures 8–9b support frozen-gamma benefit and a small consistent learnable-gamma improvement. The exact local per-channel attention/MLP rollout remains a new configuration. |
| A5 Late-hold GPAS | C observation / W inference | Early movement is descriptive. The source tests neither the proposed freeze policy nor a causal late-drift mechanism; clipping and fixed gates are worse than learnable GPAS in the cited results. |
| A6 Tangent Muon | W | No cited paper supplies this optimizer. First-order tangent projection plus Frobenius rescaling is not a finite-step manifold retraction. |
| A7 Full-state gate | C source / P candidate | Gated Attention uses the full pre-normalized state, a `[0,1]` sigmoid gate, and gates SDPA output before projection. The local width-only change retains a zero-initialized `2*sigmoid` gate after head RMS normalization. |
| A8 x0 writeback | P abstract / W mechanism | Only the mHC abstract is preserved locally. It supports manifold-constrained residual mixing generally, not the proposed x0 construction, and cannot ground exact mechanics or quantitative claims. |
| B1 Document-bounded keys | C source facts / P candidate | Engram defines suffix n-grams and Tensorizing Engram states left padding at a sequence start. Neither tests BOS resetting at packed internal document boundaries or the proposed 0.0030 proxy. |
| B2 Zipf addressing | C motivation / P candidate | Engram's Zipf discussion supports cache/storage tiering, not collision-free trainable address partitioning, MPHF lookup, exact membership, or tail remapping. |
| B3 Agreement gate | C source / P candidate | Engram uses full hidden/key agreement, a `[0,1]` sigmoid, value transform, and convolution. The local zero-no-op `2*sigmoid`, 32-dimensional key, multiplication into an existing gate, and zero initialization are untested. |

### Version and local-scope corrections

- Exact Engram locators above refer to the preserved updated 33-page repository
  PDF. The ACL camera-ready is a 23-page DSE-named version. ACL provenance must
  be version-qualified rather than attached to updated wording by implication.
- Tensorizing Engram is an arXiv-only source. The local mHC record is
  abstract-only.
- The local source has one GPAS scalar per layer reused after both residual
  additions; it is excluded from Muon and receives schedule-exempt AdamW.
- The existing head gate is `32→6`, zero-initialized, and applied as
  `2*sigmoid` after per-head RMS normalization.
- B1's premise is code-real: `prev_idx` and `prev2_idx` are row shifts that can
  cross internal packed BOS boundaries, while attention isolation does not
  repair the memory keys.
- The accepted resolved environment uses FA3 document isolation. Bare source
  defaults are not themselves a valid resolved execution because default SDPA
  conflicts with the varlen implementation requirement.
- Local Q/K/V weights are fused and `c_proj` is zero-initialized. Any
  source-faithful per-head conditioned initialization needs explicit reshaping,
  and first-step Q/K/V gradients are initially blocked by the zero projection.

The provenance audit leaves the critic shortlist unchanged: B1 is the strongest
independent local-code premise; A1 and A4 remain parent-conditional local
extrapolations. It strengthens the veto against transferring prestige from GPAS,
Conditioned Initialization, Spectral Conditioning, Gated Attention, or Engram
to untested local hybrids.
