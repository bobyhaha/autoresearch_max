# B1 Document-Bounded Suffix Keys — Prospective Premise Specification

- Frozen at: 2026-07-29T19:32:00Z
- Status: design only; no corpus assay was run
- Authority: no paper, registry, intervention, experiment, remote mutation, GPU
  run, endpoint, comparator, chart, or SOTA authority
- Parent: `research/analysis/plateau_portfolio_proposals_20260729.md`
- Critic prerequisite: this exact specification requires a fresh independent
  audit before any implementation or data access

## Question and interpretation

The local model isolates packed documents in attention, but its bigram and
trigram suffix construction uses row shifts that can cross an internal BOS
boundary. B1 asks whether resetting unavailable suffix positions to BOS removes
a sufficiently large training-only count-predictor penalty to justify a later
mechanism assay.

Every score below is a held-out count-predictor screening proxy. It is not a
neural BPB estimate, endpoint prediction, lower bound, confidence bound, or
authorization to score validation.

## Frozen input identity

Source and split SHA-256:

| Artifact | SHA-256 |
|---|---|
| `train.py` | `0e053f4a530babbe1bc889a983abd382275022d0d3419f29a987c10589c752fa` |
| `lib.py` | `13de28bcb398866be65cfca489413818d75dc2fcd2461e367be06198f879123d` |
| `prepare.py` | `64fa6178cd67ea7bd8dfd165e80c28bbec4da7991f4243c594c27ce6ad8e3255` |
| `data_split.json` | `ed8ea0554010df9fbfa47746adc6643d6446797753eb377c0b35413bdcf2fca3` |
| `tokenizer.pkl` | `888c9fa865c72c25f9c5e9467136a9cc8655253db459fb97481deb326681c629` |
| semantic token-ID-to-bytes map | `662fa9cfd0dcec68785377b48cbc172642ad88d9e5c5fbd0c0422dd17140f9d8` |
| `token_bytes.pt` | `33f5b2ae0b1765c9e083bfc1c3df87efe9738068e15a80d234623a2c3b261d6d` |

Vocabulary size is 8,192 and BOS token ID is 8,188. The exact training-shard
prefix is `[1,2,3,4,5,6,7,8]`:

| Shard | SHA-256 |
|---:|---|
| 1 | `d4cfe1de19f4fd976022a9047458655968bc5583582ec5fa714801ec2be2c93c` |
| 2 | `5e80d7596d53be648d841ea5811082747281c39f6851e10dd90375a84bc509a7` |
| 3 | `b349c85d4b993d98e2328833ecfd3734692e0395a592d89858bae34a4ba73c59` |
| 4 | `87dae96fe1a6116c54c7873aadb81832178a36755991befd1b3b641c76244474` |
| 5 | `3d9aebf138ef69cff2585a686d7f21585b7ffc0de4655f8e7723590418583bd6` |
| 6 | `6a3f2afde8f1a39f9283ea9d9e9ee6431634d6881b63798176bf64efc3f33576` |
| 7 | `731eea337d941e5b18a9a4e2bbe73a517dac45d91bb93ae2e69346c68c5f37f6` |
| 8 | `e78379132451838afabed9034b5483a2f3bc5a422bed8b55b99f2bbb9a128755` |

A proposed canonical one-line JSON manifest containing these fields plus exact
shard sizes, row counts, and row-group counts has prospective digest
`2b3fc40bd54e77e6d69ce99f9bff5990d636f2f2a124ace75dbc6c19a9e0dda6`
under `json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n'`.
That manifest is not yet a persisted artifact. Its digest is a design value,
not an observed artifact fact, until independently materialized and verified
before data access.

## Packing and folds

Use eight folds. Fold `f` holds out shard `f` and fits counts on the other seven
shards only.

Tokenize and pack each shard independently so the best-fit buffer cannot leak a
held-out document into a training fold:

1. reset buffer, output-row, and document provenance at each shard boundary;
2. traverse row groups and rows in increasing source order;
3. call `encode_ordinary_batch` with batch size 128 and prepend BOS;
4. use buffer capacity 1,000 and packed length `T+1=2049`;
5. select the longest fitting document, ties broken by earliest insertion;
6. if none fits, crop the shortest document, ties broken by earliest insertion,
   and discard its remainder exactly as the current `lib.py` packer;
7. at finite shard EOF, drain the buffer and emit a final variable row only if
   its length is at least two;
8. never shuffle, repeat, or create a cross-shard packed row.

Emit and hash packed rows plus occurrence provenance before scoring.

## Raw and document-bounded keys

For a packed input token `x_t` and target `y_t`, the literal current row-shift
contexts are:

- `r1 = x_0` at `t=0`, otherwise `x_{t-1}`;
- `r2 = x_t` for `t<2`, otherwise `x_{t-2}`;
- raw bigram key `K_R2=(r1,x_t)`;
- raw trigram key `K_R3=(r2,r1,x_t)`.

Let `d_t` be the offset from the latest BOS in the packed row. The proposed
document-bounded contexts are:

- `p1=BOS` if `d_t=0`, otherwise `x_{t-1}`;
- `p2=BOS` if `d_t<=1`, otherwise `x_{t-2}`;
- padded bigram key `K_P2=(p1,x_t)`;
- padded trigram key `K_P3=(p2,p1,x_t)`.

Thus a document BOS gets two BOS prefixes, and its first content token gets BOS
as its second prefix, including when the document starts a packed row.

For order `o`, define affected positions
`A_o={i: K_Ro(i) != K_Po(i)}` and `A=A_2 union A_3`. Define the fixed matching
group

`g=(order, d in {0,1}, document_BOS_is_row_start, K_Po)`.

Scoring additionally requires `token_bytes[y_i] > 0`.

## Fixed count model

For mode `m` in `{R,P}`, order `o` in `{2,3}`, and fold `f`, build exact uint64:

- occurrence-target counts `C_mof(k,y)`;
- occurrence counts `C_mof(k)`;
- global target counts `G_f(y)`;

from the seven fit shards only. Include zero-byte and special targets during
fitting because neural cross-entropy trains them.

Freeze

`q_f(y)=(G_f(y)+0.5)/(sum_y G_f(y)+V/2)`

and

`p_mof(y|k)=(C_mof(k,y)+1.0*q_f(y))/(C_mof(k)+1.0)`.

An unseen key falls back exactly to `q_f`; globally unseen targets remain finite
through the `+0.5` term. No smoothing sweep, backoff, interpolation, key filter,
frequency threshold, or tuning is permitted. Use IEEE float64 natural
logarithms and `math.fsum` in canonical held-out stream order.

The implementation is expected to require streaming external sort/merge;
unbounded in-memory maps are not assumed feasible.

## Atomic proxy and premise gate

There are four bigram and three trigram injection sites, so fix
`w_2=4/7` and `w_3=3/7`.

For fold `f`, define:

- `L_mof = sum_{i in heldout f intersect A_o, bytes(y_i)>0}
  -ln p_mof(y_i | K_mo(i))`;
- `B_f = sum_{all eligible heldout positions i} bytes(y_i)`;
- `delta_f = [(4/7)(L_R2f-L_P2f)+(3/7)(L_R3f-L_P3f)]
  /(ln(2)*B_f)`.

The pooled delta is the same summed numerator over
`ln(2)*sum_f B_f`. This is the atomic form of affected-byte-mass times
conditional BPB gap. Report affected masses separately, but never replace the
atomic estimator with a product of separately rounded values.

The premise passes only if:

1. pooled `delta >= 0.0030`; and
2. the second-smallest of the eight `delta_f` values is strictly positive,
   which is at least seven of eight positive folds and has exact one-sided sign
   probability at most `9/256`.

Report all eight deltas, their second order statistic, median, MAD, minimum, and
maximum. Do not use token-level bootstrap or other pseudoreplication.

## Exact intervention

Change only bigram/trigram context tensors:

```text
prev_pad  = where(idx==BOS, BOS, prev_raw)
prev2_pad = where((idx==BOS) | (prev_raw==BOS), BOS, prev2_raw)
```

Preserve inputs, targets, order, four bigram plus three trigram sites, fourteen
half-tables, table sizes, primes/moduli, parameter shapes/dtypes/initialization,
optimizer groups/state, schedule, and every RNG stream.

## Step-zero boundary

A same-device clone must establish exact pre-forward model, optimizer, CPU RNG,
CUDA RNG, data, cursor, and boundary hashes.

- Raw and padded semantic plus hashed indices must be exact-equal outside
  `A_o`, and differ only as specified inside `A_o`.
- With every bigram/trigram value-embedding contribution temporarily masked to
  zero in eager replay, logits, loss, and every pre-existing non-memory
  gradient must be bitwise equal.
- With normal memory enabled, logits and non-memory gradients may differ at
  step zero because the selected random table rows differ. They must not be
  claimed equal.

## Matched 256-step training-only mediators

Prospective execution uses seed 67, `SHUFFLE_DATA=0`, training shards only, and
no validation import or path.

Occurrence identity is `(step,row,position,order)`. Match both arms on the exact
same occurrences, targets, and fixed semantic group `g`.

For each of fourteen `(layer,half-table)` sites:

- exposure concentration
  `E_a(g)=sum_q n_q^2/sum_q n_q`, where `q=K_R` in control and `q=K_P` in
  treatment;
- update exposure
  `U_a(g)=sum_q n_q*u_q/sum_q n_q`, where `u_q` is the number of distinct steps
  with a nonzero FP32 attributed pre-scatter row-gradient sum.

Aggregate with fixed `n_g` weights across groups and equal `1/14` site weights.
Require treatment/control exposure ratio at least 2.0 and update ratio at least
2.0.

For directional dispersion, use the same newly consolidated occurrence pairs
at each site: pairs within `g` whose control physical rows differ. Admit a pair
only when all four per-occurrence gradient norms are nonzero in both arms.
Compute

`D_a = mean[1-cos(gradient_i^a,gradient_j^a)]`

over the fixed pair set through exact sufficient sums, without sampling.
Require eligible pair count positive, `D_control>0`, and
`D_treatment/D_control<=0.80`.

Report each role, order, and site. Require the expected direction on at least
eleven of fourteen sites. Arbitrary-row controls and post-hoc frequency matching
are forbidden.

## Compiler and throughput gate

Eager gradient instrumentation is not timed. Separately run two clean compiled,
no-probe, no-validation 256-step placements with GPU roles swapped:

- steps 1–32 are warmup;
- steps 33–256 form 28 ordered contiguous eight-step paired blocks;
- require `fullgraph=True`, one Dynamo graph per arm, zero recompiles and graph
  breaks, and identical data digests;
- block ratio is `dt_control/dt_treatment`;
- aggregate with geometric means.

Freeze 10,000 within-placement paired-block bootstrap resamples under
`random.Random(19019067)`, preserving the same resampled block indices across
arms within a placement, independently resampling the two placements, and
geometrically aggregating placements. The one-sided 95% lower bound is
zero-based sorted order statistic 499. Require both point ratio and lower bound
at least 0.995.

Any parity, compiler, data, hash, or finite-gradient failure is
`INVALID_DIAGNOSTIC`, not an efficacy failure.

## Falsifiers and leakage prevention

Kill the premise if any of the following holds:

- pooled proxy below 0.0030;
- at most six of eight fold deltas positive;
- cross-document fragments are more predictive;
- affected exposure is absent;
- exposure or update ratios are below 2.0;
- dispersion ratio exceeds 0.80 or has the wrong sign;
- clean timing, graph, or recompile gate fails.

If all premise gates pass but a later separately authorized concurrent endpoint
misses the then-active raw paired val_bpb floor, efficacy is falsified.

The offline worker may open only the eight listed training shards. A future
runtime worker receives only train-shard capabilities and may not import or
create `evaluate_bpb`, `val_loader`, test/6542 paths, or endpoint code. Freeze
specification, tool, runtime, and output schemas before a scan. Do not reuse the
old validation-derived canonical-key result. Never change smoothing, grouping,
thresholds, filters, or interpretation after observing output. A premise pass
grants no endpoint authority.

## Hard blockers

This specification is premature for implementation or execution:

1. `dec_plateau_portfolio_critic_20260729` registers or executes none and B1
   cannot bypass the active Paper-020 parent chain.
2. Setup reconciliation v31 remains pending for campaign adoption.
3. `aud_paper020_worker_boundary_20260729` records no import-safe training-only
   worker under the current monolith and a cyclic registration/runner digest
   graph; the same structural blockers apply here.
4. Remote SSH authentication is failing.
5. No B1 count tool, per-occurrence gradient collector, detached authority, or
   persisted input manifest exists.
6. The prior canonical-key assay is validation-specific and hash-stale. Only
   its packing, hashing, and atomic-write patterns are reusable.
7. The boundary-sidecar work shows how to transport boundaries but does not
   provide this estimator.

Therefore this exact specification may receive fresh independent criticism, but
corpus scoring, paper inclusion, registration, implementation, and GPU work are
all blocked.

## Fresh independent methods critique

- Completed at: 2026-07-29T19:42:00Z
- Verdict: `REVISE`
- Authority: no implementation, scan, registration, paper inclusion, GPU,
  endpoint, comparator, chart, or SOTA authority

The critic verified:

- the raw shift definitions and document-bounded padding formulas;
- four bigram plus three trigram sites, hence `4/7` and `3/7` weights;
- normalized additive smoothing, exact unseen-key fallback, and BPB dimensions;
- bootstrap index 499 as the lower five-percent order statistic for 10,000
  sorted replicates;
- local hashes for source, split, tokenizer pickle, token bytes, and all eight
  shards.

The semantic tokenizer digest was not freshly recomputed because the critic's
available interpreter lacked `tiktoken`; it remains supported by the prior
matching-pickle artifact rather than this fresh review.

The exact specification requires all corrections below before another critic
pass:

1. Remove the sign-test probability interpretation. The eight fold deltas share
   six or seven fit shards and are not independent randomized units. Retain
   “at least seven positive” only as a deterministic robustness gate; do not
   call `9/256` a valid p-value.
2. Fully specify finite-shard packing. Current `lib.py` refills using whole
   tokenizer batches while buffer length is below 1,000, so the buffer may
   exceed 1,000 and the live generator has no finite-EOF drain. Freeze refill
   until `len(buffer)>=1000` or EOF without splitting a tokenizer batch, exact
   row-group tail batching, EOF selection/cropping, and one-token residual-row
   handling. “Capacity 1,000” is not sufficient.
3. Define mediator keys as actual physical rows:
   `q=h_s(K_a)`, with `E_{a,s}`, `U_{a,s}`, and `D_{a,s}`. Semantic keys must
   differ inside affected positions, but their physical hashes may collide;
   step-zero parity must permit that case.
4. Resolve the eleven-of-fourteen rule. State whether a site passes only when
   all three site gates pass, and whether the site-count gate is conjunctive
   with the separately weighted aggregate gates.
5. Freeze the runtime stream: exact shard IDs, loader/packer semantics, device
   batch, sequence length, gradient accumulation, step numbering, environment,
   resolved configuration, and digest serialization. “Training shards only”
   does not distinguish shards 1–8 from the repository's 1–10 runtime split.
6. Delete or operationalize the phrases “cross-document fragments are more
   predictive,” “affected exposure is absent,” and “wrong sign.” Each must be
   either redundant with an existing gate or receive a distinct statistic and
   threshold.
7. State timing aggregation exactly: geometric mean of 28 block ratios within
   each placement; independently resample 28 paired indices per placement;
   geometrically combine placement statistics; sort; take index 499. Define
   graph-count scope. Requiring one graph across model, backward, and separately
   compiled optimizer kernels is incompatible with the current topology.
8. Materialize and persist the canonical input manifest plus key, count,
   provenance, and output schemas before any data access. The prospective
   manifest digest is not yet independently auditable.

The hard blockers remain correct and binding. The proposal stays outside the
paper and registries until these corrections are prospectively frozen and a
fresh independent critic returns a passing verdict.
