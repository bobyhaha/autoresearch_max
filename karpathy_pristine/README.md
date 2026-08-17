# karpathy_pristine — unmodified upstream, for reference runs

Exact bytes of `karpathy/autoresearch` at commit
`228791fb499afffb54b46200aca536f79142f117`. Nothing in this folder is adapted,
instrumented, or corrected. It exists so a reference number can be produced that
owes nothing to this campaign's changes.

Verified against the digests recorded in `../runs/code/provenance.json`:

| file | sha256 | git blob |
|---|---|---|
| train.py | `2954175f4ac42ad65164aef40910ef953789abcd05a5cc886ac9ba5a00814414` | `2e743974c7f06b54311643b314712303fbb26e65` |
| prepare.py | `4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150` | `06bea9165abd3ae94ea82dd733997aec7928f40c` |

## A pristine measurement also needs a clean cache

Upstream `prepare.py` reads `~/.cache/autoresearch`, defines the training split from
the files present there, and may reuse existing tokenizer artifacts. Consequently, a
clean reference requires a freshly prepared, inspected cache root, not just pristine code:

    # 1. Verify the cache is empty or absent FIRST. If it exists, inspect it:
    ls ~/.cache/autoresearch/data | wc -l                  # expect the shards you intend
    find ~/.cache/autoresearch/data -maxdepth 1 -name '*.parquet' -print | sort

    # 2. Prepare into a clean root, then train.
    uv run python prepare.py      # populates ~/.cache/autoresearch
    uv run python train.py        # 300s budget, prints val_bpb at the end

Record the shard list and tokenizer artifact digests with any number reported from here.
Without them the measurement is not reproducible.

## Two things to know before comparing its number to anything

**Its BPB denominator is part of the scope.** Upstream computes token byte lengths as
`decode([id]).encode("utf-8")`. Do not compare that metric with a modified byte-accounting
implementation under the same scope ID. See `../runs/code/UPSTREAM.md`.

**It emits no machine-readable metrics.** There is no `AUTORESEARCH_METRICS` line,
so the evidence harness cannot parse a run from this folder and would judge it
invalid. That is why this folder is for reference runs read by eye, not for banked
measurements.
