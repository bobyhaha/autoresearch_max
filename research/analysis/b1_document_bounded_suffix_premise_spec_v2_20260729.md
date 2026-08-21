# B1 Document-Bounded Suffix Keys — Prospective Premise Specification v2

- Frozen at: 2026-07-29T20:05:00Z
- Status: preaccess design only; no corpus shard was opened or scanned for v2
- Supersession: additive methods revision; the v1 file remains untouched
- Parent: `research/analysis/plateau_portfolio_proposals_20260729.md`
- Authority: no paper, registry, implementation, experiment, remote mutation,
  GPU, endpoint, comparator, chart, or SOTA authority
- Entry rule: every preaccess blocker below must clear and a fresh independent
  critic must return `PASS` before the first corpus read

## Estimand and interpretation

B1 asks whether the current row-relative bigram and trigram suffixes impose a
training-only count-predictor penalty at packed document boundaries, and,
conditionally, whether BOS padding consolidates the corresponding physical
memory rows without unacceptable compiler or throughput cost.

The offline score is a site-weighted, held-out count-predictor proxy. It is not
a neural BPB estimate, endpoint prediction, lower bound, confidence bound, or
run authority. The eight cross-fit values have overlapping seven-shard fit
sets. They are dependent robustness views, not independent randomized units.
No sign-test probability, p-value, confidence interval, or effective sample
size is assigned to them.

## Preaccess blocker table

| Gate | Required state before first corpus read | Current state |
|---|---|---|
| v2 specification | This file exists with a verified SHA-256 | Satisfied only after final hash verification |
| Embedded input manifest and schemas | Exact canonical payloads below parse and match their stated digests | Pending independent verification |
| Fresh methods criticism | Independent critic returns `PASS` on this exact file hash | Blocked |
| Portfolio authority | `dec_plateau_portfolio_critic_20260729` is prospectively superseded by explicit B1 premise authority | Blocked; current decision says register/execute none |
| Parent chain | Active Paper-020 parent reaches a terminal state that permits B1 | Blocked |
| Setup | Applicable reconciliation status is `passed` on the future worker hashes | Blocked; v31 is pending |
| Training-only boundary | Import-safe worker cannot import or resolve validation/test capabilities | Blocked; `aud_paper020_worker_boundary_20260729` found no safe boundary |
| Authority graph | Detached one-way source manifest → execution authority → worker graph | Blocked; current registration/runner graph is cyclic |
| Input identity | Source, tokenizer, split, and all eight shard bytes match the embedded manifest | Not checked in v2; checking is forbidden before the gates above |
| Offline tooling | Hash-bound finite packer, external counter, scorer, and schema tests exist | Blocked; not implemented |
| Runtime tooling | Hash-bound per-occurrence row-gradient collector and scoped compile telemetry exist | Blocked; not implemented |
| Hardware access | Authenticated, tenant-free H200 pair with immutable UUID census | Blocked; SSH authentication was failing |

A failed prerequisite is `BLOCKED_PREACCESS`, not `PREMISE_FAIL`. The permitted
phase order is: verify this file and embedded payloads; obtain fresh criticism;
resolve governance and worker blockers; materialize the embedded payloads
byte-for-byte; verify their digests; then, and only then, hash the allowed
inputs. A hash mismatch is `INVALID_INPUT` and forbids packing or scoring.

## Canonical serialization

Every embedded JSON payload and future JSON artifact uses:

```text
(json.dumps(value, sort_keys=True, separators=(",",":"),
            ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
```

JSONL is the concatenation of one such canonical line per record in the
declared sort order. Binary integers are unsigned little-endian. Token IDs are
`u16`; counts are `u64`. A floating result is stored as a lowercase
`float.hex()` string, never as a JSON number. SHA-256 is over exact file bytes.
No path, timestamp, hostname, object identity, or nondeterministic map order
enters a scientific digest.

The literal content of each payload block below is exactly its single JSON line
plus one LF; the Markdown fences are not part of the payload.

### Canonical input manifest

Payload ID: `b1_document_bounded_suffix_inputs_v2`.

SHA-256:
`03b38068736c18d7b965e1b9548140acd03231ded27628f4bbdd38bb04e6a7bf`.

```json
{"data":{"data_split_sha256":"ed8ea0554010df9fbfa47746adc6643d6446797753eb377c0b35413bdcf2fca3","ordered_shard_ids":[1,2,3,4,5,6,7,8],"shards":[{"bytes":92124317,"filename":"shard_00001.parquet","id":1,"row_groups":84,"rows":86016,"sha256":"d4cfe1de19f4fd976022a9047458655968bc5583582ec5fa714801ec2be2c93c"},{"bytes":91803619,"filename":"shard_00002.parquet","id":2,"row_groups":83,"rows":84992,"sha256":"5e80d7596d53be648d841ea5811082747281c39f6851e10dd90375a84bc509a7"},{"bytes":91793972,"filename":"shard_00003.parquet","id":3,"row_groups":82,"rows":83968,"sha256":"b349c85d4b993d98e2328833ecfd3734692e0395a592d89858bae34a4ba73c59"},{"bytes":91428652,"filename":"shard_00004.parquet","id":4,"row_groups":82,"rows":83968,"sha256":"87dae96fe1a6116c54c7873aadb81832178a36755991befd1b3b641c76244474"},{"bytes":91845922,"filename":"shard_00005.parquet","id":5,"row_groups":83,"rows":84992,"sha256":"3d9aebf138ef69cff2585a686d7f21585b7ffc0de4655f8e7723590418583bd6"},{"bytes":91455616,"filename":"shard_00006.parquet","id":6,"row_groups":82,"rows":83968,"sha256":"6a3f2afde8f1a39f9283ea9d9e9ee6431634d6881b63798176bf64efc3f33576"},{"bytes":91202032,"filename":"shard_00007.parquet","id":7,"row_groups":82,"rows":83968,"sha256":"731eea337d941e5b18a9a4e2bbe73a517dac45d91bb93ae2e69346c68c5f37f6"},{"bytes":91155507,"filename":"shard_00008.parquet","id":8,"row_groups":82,"rows":83968,"sha256":"e78379132451838afabed9034b5483a2f3bc5a422bed8b55b99f2bbb9a128755"}],"split":"train_only"},"local_source":{"lib.py":"13de28bcb398866be65cfca489413818d75dc2fcd2461e367be06198f879123d","prepare.py":"64fa6178cd67ea7bd8dfd165e80c28bbec4da7991f4243c594c27ce6ad8e3255","train.py":"0e053f4a530babbe1bc889a983abd382275022d0d3419f29a987c10589c752fa"},"manifest_id":"b1_document_bounded_suffix_inputs_v2","runtime_config":{"attention":{"backend":"fa3","doc_mask":true,"doc_mask_impl":"varlen","doc_mask_mode":"both","packer_doc_boundaries":false,"reset_rope":false,"window_pattern":"SSSL"},"batch":{"device_batch_size":128,"gradient_accumulation_steps":1,"sequence_length":2048,"total_batch_size":262144},"compile":{"fullgraph":true,"mode":"max-autotune-no-cudagraphs","model_dynamic":false},"intervention":{"control":"raw_crossdoc","treatment":"bos_pad"},"model":{"dtype":"bfloat16","gpas":false,"n_embd":768,"n_head":6,"n_kv_head":6,"n_layer":8,"ngram_bigram_mult":0,"ngram_fivegram_mult":0,"ngram_fourgram_mult":0,"ngram_fourgram_span":3,"ngram_table_mult":64,"ngram_trigram_mult":0,"pk_memory":false,"shared_trigram_ve":false,"vocab_size":8192},"optimizer":{"adam_beta1":"0x1.999999999999ap-1","adam_beta2":"0x1.e666666666666p-1","embedding_lr":"0x1.3333333333333p-1","matrix_lr":"0x1.eb851eb851eb8p-6","ngram_beta2":"0x1.ff7ced916872bp-1","ngram_lr_scale":"0x1.0000000000000p+0","scalar_lr":"0x1.999999999999ap-1","unembedding_lr":"0x1.0624dd2f1a9fcp-8","weight_decay":"0x1.999999999999ap-4"},"probes":{"gradient_mediator":"eager_separate_untimed","validation_imports":false,"validation_paths":false},"schedule":{"adam_warmdown_ratio":"0x1.4cccccccccccdp-1","final_lr_fraction":"0x1.999999999999ap-5","max_steps":256,"muon_warmdown_ratio":"0x1.e666666666666p-1","ngram_warmdown_ratio":"0x0.0p+0","stop_mode":"steps","warmup_ratio":"0x0.0p+0"},"seed":67,"stream":{"full_rows_required":32768,"repeat":false,"shard_ids":[1,2,3,4,5,6,7,8],"shuffle":false}},"schema_version":2,"tokenizer":{"bos_id":8188,"semantic_id_to_raw_bytes_sha256":"662fa9cfd0dcec68785377b48cbc172642ad88d9e5c5fbd0c0422dd17140f9d8","token_bytes_sha256":"33f5b2ae0b1765c9e083bfc1c3df87efe9738068e15a80d234623a2c3b261d6d","tokenizer_pickle_sha256":"888c9fa865c72c25f9c5e9467136a9cc8655253db459fb97481deb326681c629","vocab_size":8192}}
```

Before extraction, the future worker must also bind these current-control
resolved values: `MLP_TYPE=sqrelu`, `LM_HEAD_INIT_STD=0.001`,
`SOFTCAP_CAP=16.5`, `SOFTCAP_TAU=15.0`, `NGRAM_VE_EPS=1e-10`,
`NGRAM_WD_LAMBDA=0`, `NGRAM_BACKOFF_KAPPA=0`, `CAUTIOUS_UPDATE=0`,
`CAUTIOUS_MUON=1`, `MUON_MOMENTUM_CONTINUOUS=0`, `BF16_CE=0`,
`NGRAM_STATE_ROWWISE=0`, `NGRAM_SPARSE_GRAD=0`, `CANON=0`,
`TIE_EMBED=0`, `RESID_SCALE=0`, `ATTNRES_ENABLE=0`, `TOKEN_SHIFT=0`,
`NGPT_SPHERE=0`, `RHO1_GAMMA=0`, `LR_SCALE=1`, `SHARED_TRIGRAM_VE=0`,
`NGRAM_PK_MEMORY=0`, `GPAS_ENABLE=0`, `TRACK_B_MODE=0`,
`VAL_LOSS_EVERY=0`, `VAL_BPB_PROBE_EVERY=0`, and
`OBSERVE_LAYER_PROBES=0`. The worker accepts no environment override other than
the immutable arm role and assigned GPU UUID. It emits the complete resolved
configuration canonically and its SHA-256 before model construction; any
difference is `INVALID_DIAGNOSTIC`.

### Packed-row/key schema

SHA-256:
`6cfb50893b7f623ad582e0ad676ac8d3ce57b8e17a136cee4f1fca81563e72d0`.

```json
{"file":"packed_rows.bin","magic_ascii":"B1ROWSV2\\n","record":"shard_id:u32le,row_index:u64le,token_count:u16le,token_ids[token_count]:u16le","schema_id":"b1_packed_rows_v2","sort":"shard_id,row_index","token_count_range":[1,2049]}
```

The two key modes are derived from this file and are not stored with ambiguous
Python tuples. For a row token `z_t`, input `x_t=z_t`, and target
`y_t=z_{t+1}`:

```text
r1(t) = z[0] if t=0 else z[t-1]
r2(t) = z[t] if t<2 else z[t-2]
K_R2(t) = (r1(t), z[t])
K_R3(t) = (r2(t), r1(t), z[t])

d(t)  = offset from the latest BOS in this row
p1(t) = BOS if d(t)=0 else z[t-1]
p2(t) = BOS if d(t)<=1 else z[t-2]
K_P2(t) = (p1(t), z[t])
K_P3(t) = (p2(t), p1(t), z[t])
```

`A_o={t:K_Ro(t)!=K_Po(t)}` is semantic affectedness. A semantic difference may
hash to the same physical row; that is a permitted collision, not a parity
failure.

### Count schema

SHA-256:
`ff238fc33ddc239eb1ce435e478db6c4793b6c89cc240d3752aa95af0a11194c`.

```json
{"file":"key_counts.bin","magic_ascii":"B1CNTS2\\n","mode_codes":{"bos_pad":1,"raw_crossdoc":0},"order_codes":{"bigram":2,"trigram":3},"record":"shard_id:u32le,mode:u8,order:u8,k0:u16le,k1:u16le,k2:u16le,target:u16le,count:u64le","schema_id":"b1_key_counts_v2","sort":"shard_id,mode,order,k0,k1,k2,target","unused_bigram_k0":65535}
```

Records are unique after exact `u64` aggregation. The counter must fail on
overflow, duplicate sorted keys, an invalid token, or a noncanonical order.
Fold counts are computed by exact total-minus-held-out subtraction from these
per-shard records; no held-out record enters its seven-shard fit counts.

### Packing-provenance schema

SHA-256:
`c59f0a022e31fb43f59dab828e9389e754a820b6b61cca50f14520555e629a3e`.

```json
{"file":"packing_provenance.jsonl","required_keys":["cropped","end_position_exclusive","original_token_length_with_bos","row_index","row_in_group","row_group","shard_id","source_doc_ordinal","start_position","used_token_length"],"schema_id":"b1_packing_provenance_v2","serialization":"canonical_json_line_per_event","sort":"shard_id,row_index,start_position,source_doc_ordinal"}
```

`packing_summary.json` additionally fixes, per shard: input rows, full rows,
variable-tail rows, dropped one-token tails, source documents, consumed
documents, cropped documents, maximum buffer length, input/target positions,
and SHA-256 values for packed rows and provenance. All are integers or digest
strings.

### Output schema

SHA-256:
`3a150b53086f69e44a338bc5192b5a5df4b8d588a97a82d9de5890eaf6f04541`.

```json
{"files":["input_manifest.json","packed_rows.bin","packing_provenance.jsonl","packing_summary.json","key_counts.bin","offline_result.json","runtime_result.json","output_manifest.json"],"float_encoding":"python_float_hex_string","output_manifest_entry_keys":["bytes","sha256"],"schema_id":"b1_outputs_v2","status_enum":["BLOCKED_PREACCESS","INVALID_INPUT","PREMISE_FAIL","PREMISE_PASS_NO_RUN_AUTHORITY","INVALID_DIAGNOSTIC","MEDIATOR_FAIL","MEDIATOR_PASS_NO_ENDPOINT_AUTHORITY"],"verdict_authority":false}
```

`output_manifest.json` contains `schema_id`, all input/schema/spec digests, and
for every declared file other than itself exactly `{bytes,sha256}`. Undeclared
outputs, temporary files after successful completion, partial results, or a
missing file force invalid status. Writes use same-directory temporary files,
`fsync`, and atomic rename.

## Exact finite-EOF packer

Each shard is an independent fold unit. Reset source ordinal, tokenizer-batch
iterator, document buffer, row index, and current row at each shard boundary.
Never create a cross-shard row.

1. Visit Parquet row groups in increasing index. Read the `text` column in
   source row order. Within each row group, form slices
   `[0:128],[128:256],...`; the last slice may contain 1–127 documents and is
   not combined with the next row group.
2. Tokenize a whole slice with `encode_ordinary_batch(...,num_threads=8)`,
   prepend BOS 8188 to every document, and append the whole slice to the buffer
   in slice order.
3. Before every document selection, while `len(buffer)<1000` and source input
   remains, consume one whole tokenizer batch. Never split a batch to stop at
   1,000. Thus buffer length may overshoot 1,000; record its observed maximum.
4. Let `remaining=2049-len(row)`. Select the longest buffered document with
   length at most `remaining`; break ties by the smallest buffer index. Remove
   it and append it in full.
5. If none fits, select the shortest buffered document; break ties by the
   smallest buffer index. Append exactly its prefix of length `remaining`,
   mark `cropped=true`, discard its entire unconsumed suffix, and complete the
   row.
6. A length-2049 row is emitted immediately. At source EOF, continue the same
   selection/cropping rules until the buffer is empty.
7. If EOF and an empty buffer leave a partial row of length 2–2048, emit that
   one variable row. If its length is exactly one, record one
   `dropped_one_token_tail` with token and provenance in the summary and emit
   no input/target occurrence. If length zero, emit nothing. No padding,
   repetition, borrowing from the next shard, or special EOF token is allowed.

For every emitted row of length `L`, `inputs=z[0:L-1]` and
`targets=z[1:L]`. The offline estimator consumes full and variable rows. The
fixed-shape runtime stream consumes full rows only.

## Eight dependent robustness folds and count model

Fold `f` holds out shard ID `f` and fits on the other seven IDs. Counts include
all targets, including zero-byte specials, because training cross-entropy sees
them. With `V=8192`:

```text
q_f(y) = (G_f(y)+0.5) / (sum_y G_f(y)+V/2)
p_mof(y|k) = (C_mof(k,y)+q_f(y)) / (C_mof(k)+1)
```

An unseen key falls back exactly to `q_f`. A globally unseen target remains
finite through `+0.5`. No smoothing sweep, order backoff, interpolation,
frequency filter, or tuning is permitted. Logs are IEEE float64 natural logs;
sums use `math.fsum` in `(shard,row,position)` order.

With four bigram and three trigram injection sites, fix `w_2=4/7`,
`w_3=3/7`. For eligible held-out targets with `token_bytes[y]>0`:

```text
L_mof = sum over (held-out f intersect A_o) of -ln p_mof(y|K_mo)
B_f = sum over every eligible held-out target of token_bytes[y]
delta_f = ((4/7)(L_R2f-L_P2f)+(3/7)(L_R3f-L_P3f))
          /(ln(2)*B_f)
```

The pooled delta uses the sum of the eight numerators over
`ln(2)*sum_f B_f`. It is the atomic estimator; affected byte masses are
descriptive only. The offline premise is `PREMISE_PASS_NO_RUN_AUTHORITY` iff
both conditions hold:

1. pooled `delta >= 0.0030`; and
2. the second-smallest `delta_f` is strictly positive.

Condition 2 means at least seven dependent folds are directionally positive.
It is only a deterministic robustness rule. Report all fold deltas,
second-smallest, median, MAD, minimum, and maximum without a p-value or
interval.

## Exact physical-row intervention and step-zero boundary

Sites are

```text
S2={(2,layer,half): layer in [1,3,5,7], half in [0,1]}
S3={(3,layer,half): layer in [1,5,7], half in [0,1]}
S=S2 union S3
```

For site `s`, `h_s` is the current literal multiplier/XOR/modulo expression and
table size from the bound `train.py`. For arm `a in {R,P}` and occurrence `i`:

```text
q_a,s(i)=h_s(K_a,o(i))
```

`q_a,s` is the actual physical table row. Hash collisions are allowed:
`K_R!=K_P` with `q_R,s==q_P,s` is recorded as
`semantic_change_physical_collision`, produces no row change at that site, and
must not be rejected or silently removed from exposure/update partitions.

The intervention changes only:

```text
prev_pad  = where(idx==BOS, BOS, prev_raw)
prev2_pad = where((idx==BOS) | (prev_raw==BOS), BOS, prev2_raw)
```

Before an ordinary forward, a same-device clone must match exact model,
optimizer, CPU RNG, CUDA RNG, runtime-config, batch-stream, cursor, and
boundary digests. Semantic keys and physical rows must match outside `A_o`.
Inside `A_o`, semantic keys must differ exactly as specified while physical
rows may collide. With bigram/trigram value contributions masked to zero in a
separate eager replay, logits, loss, and all pre-existing non-memory gradients
must be bitwise equal. Normal-memory logits and gradients are allowed to differ
at step zero.

## Exact 256-step runtime stream and digests

The runtime stream concatenates full packed rows from shards 1 through 8 in
that order. It takes the first 32,768 full rows, never repeats, and groups each
consecutive 128 rows into one batch. Fewer than 32,768 full rows is
`INVALID_INPUT`. There are 256 optimizer steps numbered `0..255`; human-facing
steps `1..256` are `step+1`. Each step consumes one `128x2048` input and target
batch, so gradient accumulation is exactly one and total tokens per step are
262,144. Both arms must have identical stream digests.

The canonical stream digest is SHA-256 over:

```text
ASCII "B1STREAMV2\n";
for step=0..255:
  step:u32le;
  for each of 128 rows:
    shard_id:u32le; row_index:u64le;
    2049 token IDs:u16le;
  boundary_count:u32le;
  compact flattened BOS boundaries[boundary_count]:u32le.
```

Model and optimizer digests sort state keys lexicographically by UTF-8 bytes.
Each tensor contributes: key length `u32le`, key bytes, fixed dtype code,
rank `u8`, each dimension `u64le`, byte length `u64le`, and C-contiguous
logical element bytes. Scalars use type tags plus canonical JSON strings;
floats use `float.hex()`. CPU and CUDA RNG tensors use the same tensor encoding.
The result records pre-forward, post-backward/pre-update, and post-step-256
digests. Any unsupported state type is `INVALID_DIAGNOSTIC`.

## Physical-row matched mediators

Occurrence identity is `(step,batch_row,position,order)`. Its semantic group is
`g=(order,d in {0,1},document_BOS_is_row_start,K_Po)`. The same occurrence IDs
and targets define both arms.

For arm `a`, site `s`, group `g`, and physical row `q`, let
`n_a,s,g,q` be the number of matched affected occurrences assigned to `q`.
Let `u_a,s,g,q` be the number of distinct optimizer steps whose FP32
pre-scatter gradient sum attributable to exactly `(g,q)` has nonzero L2 norm.
Define:

```text
E_a,s(g)=sum_q n_a,s,g,q^2 / sum_q n_a,s,g,q
U_a,s(g)=sum_q n_a,s,g,q*u_a,s,g,q / sum_q n_a,s,g,q
E_a,s=sum_g N_g*E_a,s(g)/sum_g N_g
U_a,s=sum_g N_g*U_a,s(g)/sum_g N_g
```

This notation uses actual physical rows and naturally retains collisions.

For dispersion at site `s`, use the fixed newly consolidated pair set

```text
P_s(g)={(i,j): i<j, i,j in g,
        q_P,s(i)=q_P,s(j), q_R,s(i)!=q_R,s(j)}.
```

Admit a pair only if both per-occurrence pre-scatter row-gradient vectors have
nonzero L2 norm in both arms. The eligibility decision is symmetric and fixed
before arm comparison. Using all eligible pairs without sampling:

```text
D_a,s = mean over union_g P_s(g) of
        (1-cos(grad_a,s(i),grad_a,s(j))).
```

Exact sufficient sums may replace pair enumeration only if tests prove exact
float64 equality on fixed fixtures. Define site ratios
`R_E,s=E_P,s/E_R,s`, `R_U,s=U_P,s/U_R,s`, and
`R_D,s=D_P,s/D_R,s`. A site passes iff all are defined, it has at least one
eligible dispersion pair, `D_R,s>0`, and all three conjunctive gates hold:

```text
R_E,s >= 2.0
R_U,s >= 2.0
R_D,s <= 0.80
```

Aggregate exposure and update first average arm statistics equally over the
fourteen sites and then take the arm ratio. Aggregate dispersion pools the
declared eligible pairs across all sites before taking the arm ratio. The
runtime mediator passes only if all four conditions hold conjunctively:

1. aggregate exposure ratio `>=2.0`;
2. aggregate update ratio `>=2.0`;
3. aggregate dispersion ratio `<=0.80`, with positive control denominator; and
4. at least eleven of fourteen sites pass all three site gates.

There is no “wrong sign,” “affected exposure absent,” or alternative
frequency-matched escape clause. Undefined denominators or no eligible pairs
fail the corresponding mediator gate.

## Scoped compiler and exact throughput gate

The gradient-mediator execution is eager and excluded from timing. Timing uses
two clean, no-probe, no-validation placements on two eligible H200 UUIDs sorted
lexicographically as `u0<u1`:

```text
placement A: control=u0, treatment=u1
placement B: control=u1, treatment=u0
```

Each placement executes both arms concurrently on the same immutable stream.
Steps 1–32 are compilation/warmup. Steps 33–256 are 224 timed steps, partitioned
in order into exactly 28 nonoverlapping eight-step blocks.

One synchronized step duration uses `time.perf_counter_ns`: synchronize the
assigned device; start immediately before fetching the already-packed batch;
include H2D copy, model forward, backward, optimizer update, zero-grad, and the
next device synchronization; stop immediately after that synchronization.
Exclude logging, digest copies, compile callbacks, and mediator probes. Digest
work occurs after timing.

For placement `p` and block `b`, sum the eight arm durations and define
`r_p,b=C_p,b/T_p,b`. The placement statistic is:

```text
theta_p=exp(fsum(log(r_p,b) for b=0..27)/28)
theta=sqrt(theta_A*theta_B)
```

Bootstrap exactly 10,000 times with one `random.Random(19019067)` instance. For
each replicate, draw 28 indices with replacement for placement A, then draw a
separate 28 for placement B, using `randrange(28)` in that order. Within a
placement, apply the same sampled indices to its paired control and treatment
blocks. Compute each resampled placement geometric mean, combine them by square
root, sort the 10,000 float64 values ascending, and take zero-based index 499 as
the one-sided 95% lower percentile bound. Require `theta>=0.995` and
`bootstrap[499]>=0.995`.

Graph accounting is scoped, not global:

- the `torch.compile(model, dynamic=False, fullgraph=True, ...)` model-forward
  scope must have exactly one initial Dynamo graph and zero forward recompiles;
- AOT/lazy backward compile signatures must be identical between arms and have
  zero recompiles per signature;
- separately compiled AdamW, RMSProp, and shape-stacked Muon functions may each
  have their own initial graph; their named function/shape-signature multiset
  must be identical between arms, with zero recompiles per signature;
- any graph break in the fullgraph model-forward scope is invalid.

Thus “one graph” applies only to the model-forward compiled callable, not to the
model, backward, and separately compiled optimizer kernels collectively.

## Operational verdicts and falsifiers

There are no vague or duplicative falsifiers.

- `INVALID_INPUT`: any source/manifest/schema/packed-row/count/provenance digest
  or structural invariant fails.
- `PREMISE_FAIL`: pooled delta is below 0.0030 or the second-smallest dependent
  fold delta is not strictly positive.
- `PREMISE_PASS_NO_RUN_AUTHORITY`: both offline gates pass.
- `INVALID_DIAGNOSTIC`: parity, stream equality, finite gradient, state digest,
  compiler topology, hardware placement, co-tenancy, or timing construction
  fails.
- `MEDIATOR_FAIL`: a valid diagnostic misses any aggregate exposure, aggregate
  update, aggregate dispersion, eleven-site, point-throughput, or
  lower-bound-throughput gate.
- `MEDIATOR_PASS_NO_ENDPOINT_AUTHORITY`: every valid runtime gate passes.

Only after a separate future paper and authority chain may endpoint efficacy be
tested. In that future test, failure to clear its prospectively frozen raw
paired validation-BPB rule would falsify neural efficacy even if this premise
passed. This v2 specification neither defines nor authorizes that endpoint.

## Leakage and mutation prohibition

The offline process receives read capabilities only for the eight manifest
shards, tokenizer files, and bound source files. The runtime process receives
only the materialized training stream. Neither may resolve shard 6542, a test
split, `evaluate_bpb`, `evaluate_bpb_fast`, `val_loader`, endpoint code, or
network access. Capability denial is tested before input hashing.

No output may change a threshold, smoothing constant, fold, row group, shard,
packer choice, collision treatment, mediator group, site rule, runtime seed,
timing block, bootstrap draw, or interpretation. The old validation-derived
canonical-key assay is not an input. A pass cannot register an experiment,
allocate a GPU, modify a comparator, or enter a paper without a new prospective
authority.
