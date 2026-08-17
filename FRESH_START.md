# Fresh start for a new operator

This repository contains only the reusable system structure: implementation, tests,
templates, protocol documentation, and the pinned Karpathy baseline. It deliberately
contains no credentials, local environment, active registry, literature corpus, generated
paper, machine-bound configuration, hypothesis, queue state, or experiment result.

## 1. Create a private working copy

Do not let multiple friends write to the same directory or `.autoresearch` registry. Give
each operator a separate clone or working copy so every experimental `train.py` revision
and every immutable record belongs to one campaign.

## 2. Install locally

From the project root:

```bash
uv sync --group dev
cp .env.example .env
uv run autoresearch --root .autoresearch init
```

Fill `.env` with that operator's SSH target, key, port, and remote work root. Never share
the resulting `.env` or private key.

## 3. Prepare the exact upstream benchmark

On the GPU machine, in each runtime workdir, install the frozen runtime from `runs/code`.
Run the byte-identical fixed `prepare.py` once in the account whose home directory the
workers use:

```bash
cd runs/code
uv sync --frozen
uv run prepare.py --num-shards 10
```

The cache must contain exactly `shard_00000.parquet` through `shard_00009.parquet` plus
`shard_06542.parquet`. Extra parquet files change upstream's data split and are forbidden.
Do not modify `prepare.py` to redirect or filter the cache.

Generate `runs/data_manifest.json` from that exact cache:

```bash
python3 tools/build_karpathy_manifest.py \
  --cache-dir ~/.cache/autoresearch \
  --output runs/data_manifest.json
```

## 4. Bind the operator's scope and resources

Create active, ignored configuration files from the templates:

```bash
cp templates/scope_karpathy_autoresearch.json runs/scope.json
cp templates/execution_remote_h200.json runs/execution.json
```

Replace every `replace_with_...` value with the operator's actual hardware/runtime facts
and the SHA-256 of the generated data manifest. Start with one declared GPU. Keep
`prepare.py`, `pyproject.toml`, `uv.lock`, `launch.sh`, provenance, evaluator, tokenizer,
data manifest, and completed scope immutable once calibration starts.

## 5. Prove readiness

```bash
uv run autoresearch --root .autoresearch validate
uv run autoresearch --root .autoresearch science
uv run python tools/preflight.py
```

Preflight must pass before calibration. Never weaken a check or edit `prepare.py` to turn a
failure green. Resolve cache contents, remote bindings, runtime identity, GPU ownership, or
scope configuration instead.

## 6. Research before execution

Read `CLAUDE.md` completely. Define a versioned research agenda, select an axis, search and
deeply appraise its full-text literature, register atomic claims and claim-backed mechanisms,
seek contradictory evidence, conduct independent Innovator/Pragmatist/Contrarian debate,
and register an original falsifiable hypothesis. Only then calibrate the unedited baseline.
A candidate may edit only `train.py` and only after baseline calibration is valid.

All generated research state remains local and ignored by Git. Share only deliberately
exported, reviewed artifacts in a separate repository or branch.
