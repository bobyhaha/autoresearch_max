# Paper 019 round 2: canonical-key mediator assay

Verdict: **KILL_BEFORE_GPU_SCORING**.

This is an offline addressing-mediator result, not a BPB result and not run
authority.  The source-faithful portion is the pinned demo's normalization
stage order.  The 8,192-token RustBPE tokenizer, one-to-one protection policy,
packed sample, and local hash-site shifts are OPHIS-specific.

## Frozen inputs

- Engram demo: commit `fb7f84a21f91223715394a33a1dc24bbfb7f788e`, SHA-256
  `9d082070654df217e21bbca9926a4267bdf2cce7777aa6739747c24de30d2044`.
- Tokenizer pickle: `888c9fa865c72c25f9c5e9467136a9cc8655253db459fb97481deb326681c629`;
  semantic ID-to-byte digest
  `662fa9cfd0dcec68785377b48cbc172642ad88d9e5c5fbd0c0422dd17140f9d8`.
- Dataset shard(s): `6542` `769fe59d108dfd2cfa186c63173b83fbfb90a7adb3519519ba6eaa6ca9889f94`.
- Packed input: 131,072 positions,
  `2f12f64e407c1e3308138483ff1146ad335a5c408097012598d3a43a661421d3`.
- Loaded source-document text digest:
  `cae5f32cd47e4095fea880cd838a51fd99c3d65ef4730c038c5e7be8a5959fc4`.

## Vocabulary projection

- Original/canonical cardinality:
  8,192/6,808.
- Collapsed keys: 1,384
  (0.168945).
- Protected IDs: 271; all remain one-to-one.
- Alias sets: 900; explained
  900;
  unexplained 0
  (explained fraction 1.000000).

Full token mappings, alias members and stage traces, unexplained cases, and
packing events are in the sibling JSONL artifacts.

## Observed packed n-gram impact

The strict affected metric counts an occurrence only when its canonical tuple
is shared by at least two *distinct raw tuples observed in this sample*.  It is
therefore stricter than merely containing a vocabulary ID that belongs to an
alias set.

| order | occurrences | raw distinct | canonical distinct | strict affected share | alias-exposed share |
|---:|---:|---:|---:|---:|---:|
| 2 | 131,072 | 70,762 | 67,164 | 0.177887 | 0.789169 |
| 3 | 131,072 | 109,203 | 107,897 | 0.037155 | 0.903343 |

The occurrence gate uses the minimum strict share across orders 2 and 3:
`0.037155`.

## Preregistered gates

| gate | observed | requirement | result |
|---|---:|---:|:---:|
| `vocabulary_key_cardinality_collapse` | 0.168945 | >= 0.100000 | PASS |
| `observed_ngram_occurrence_affected` | 0.037155 | >= 0.100000 | FAIL |
| `alias_set_explained_fraction` | 1.000000 | >= 0.950000 | PASS |

## Interpretation limits

- The upstream demo uses the DeepSeek-V3 tokenizer; this assay uses the current 8,192-token OPHIS RustBPE tokenizer.
- The upstream repository does not pin the tokenizers package version; this assay records the exact local version.
- The local tokenizer cache lacks tokenizer_train_split.json, so the current pickle is identified by bytes and semantic vocabulary digest, not by complete tokenizer-training provenance.
- One-to-one protection for special, BOS, every one-byte token, replacement-character decoding, and empty decoding is a conservative OPHIS safety extension.
- The strict occurrence metric is computed before the existing modulo hash tables; it measures canonical tuple mergers, not random hash collisions or learned value quality.
- The packed sample mirrors current best-fit packing and current row-prefix shifts; it is one frozen validation sample, not a population estimate.
- No model input ID, target, byte count, document boundary, token order, parameter, optimizer state, or RNG state is changed or tested here.
- Passing these mediator gates neither predicts nor establishes BPB improvement and does not authorize a GPU run.

The only supported conclusion is `the frozen local key-collapse mediator fails at least one paper-019 offline threshold and the arm should be killed before GPU scoring`.
