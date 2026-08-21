#!/usr/bin/env python3
"""The literature corpus: screen wide, read deep, and know which direction each paper serves.

An earlier campaign ran its entire experiment budget having made ZERO literature
searches. Its mechanisms were drawn from memory, none had published evidence of working
at this operating point, and none of the measured ones worked. Its own critic flagged the
missing literature intake repeatedly and it was never fixed. So literature is on the
critical path here:
`tools/gate.py` refuses to open a direction that has no read papers behind it.

Three stages, each idempotent and resumable:

    lit.py screen  --target 240     # arXiv API across ~47 mechanism-targeted queries,
                                    # 2025-2026 first, scored by operating-point relevance
    lit.py fetch   --limit 40       # LaTeXML full text -> lit/sources/, hashed
    lit.py status                   # corpus coverage per DIRECTION FAMILY

Every paper is assigned to a direction family, so "study this direction intensely" has a
concrete reading list rather than a vibe:

    lit.py read capacity --limit 10   # unread papers for one direction, best first

Screening reads abstracts; that is triage and never evidence. Only a paper with a fetched
full-text snapshot on disk may back a claim (see tools/claims.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import curate_papers as cp      # noqa: E402  query set + relevance scoring
import read_fulltext as rf      # noqa: E402  LaTeXML fetch + text reduction

LIT = REPO / "lit"
SOURCES = LIT / "sources"
INDEX = LIT / "index.json"

# Which direction family a paper serves. A paper may serve several; it is counted in each,
# because a warmup paper genuinely informs both `schedule` and `optimizer_numeric`.
FAMILY_KEYWORDS = {
    "capacity":        ("depth", "width", "aspect ratio", "model shape", "mlp ratio",
                        "parameter allocation", "compute-optimal", "scaling law"),
    "token_exposure":  ("batch size", "critical batch", "gradient accumulation",
                        "tokens per step", "data order", "sample efficiency",
                        "token efficiency"),
    "attention":       ("attention", "sliding window", "sparse attention", "local attention",
                        "receptive", "context window"),
    "ve_placement":    ("value embedding", "embedding", "residual stream", "skip"),
    "signal_scale":    ("initializ", "normaliz", "rmsnorm", "logit", "softcap", "rope",
                        "positional", "signal propagation", "variance"),
    "schedule":        ("learning rate", "warmup", "schedule", "cooldown", "weight decay",
                        "momentum", "averaging", "ema", "horizon"),
    "optimizer_numeric": ("optimizer", "muon", "orthogonal", "newton-schulz", "clipping",
                          "precondition", "second-order", "gradient noise"),
    "objective":       ("multi-token", "auxiliary loss", "z-loss", "objective",
                        "supervision", "next-token", "label smoothing"),
    "signal_path":     ("skip connection", "u-net", "residual", "qk-norm", "query-key",
                        "architecture"),
    "input_pipeline":  ("data loader", "dataloader", "throughput", "wall-clock", "packing",
                        "input pipeline", "prefetch", "kernel", "fp8", "overlap"),
    "systems":         ("compile", "torch.compile", "cuda graph", "fusion", "kernel",
                        "wall-clock", "throughput"),
    "attention_detail": ("qk-norm", "query-key", "attention entropy", "logit growth",
                         "attention sink"),
}
ALL_FAMILIES = tuple(FAMILY_KEYWORDS)


def families_for(paper: dict) -> list[str]:
    hay = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
    hit = [f for f, kws in FAMILY_KEYWORDS.items() if any(k in hay for k in kws)]
    return hit or ["unassigned"]


def load_index(path: pathlib.Path | None = None) -> dict:
    path = path or INDEX
    if path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    return {}


def save_index(ix: dict, path: pathlib.Path | None = None) -> None:
    LIT.mkdir(parents=True, exist_ok=True)
    (path or INDEX).write_text(json.dumps(ix, indent=1, sort_keys=True))


def cmd_merge(args) -> int:
    """Fold sharded index files back into lit/index.json.

    Shards exist because concurrent writers on one index clobber each other. Merging is
    a union keyed on arXiv id; where both carry the same paper the richer record wins on
    `queries` (which query found it) so the provenance of a hit is not lost, and any
    `status`/`sha256` already recorded for a fetched full text is preserved.
    """
    ix = load_index()
    added = 0
    for f in sorted(LIT.glob(args.pattern)):
        if f.resolve() == INDEX.resolve() or f.name.endswith("_progress.json"):
            continue        # the per-shard resume ledger is a LIST of query names
        try:
            shard = json.loads(f.read_text())
        except (OSError, ValueError) as exc:
            print(f"  {f.name}: unreadable ({exc})")
            continue
        for pid, rec in shard.items():
            prev = ix.get(pid)
            if prev is None:
                ix[pid] = rec
                added += 1
            else:
                # Whichever side actually has the full text on disk wins the
                # fetch bookkeeping. Preferring `prev` unconditionally would keep a
                # stale "screened" over a real "fetched" and hide a snapshot the
                # claim validator needs.
                fetched = prev if prev.get("status") == "fetched" else (
                    rec if rec.get("status") == "fetched" else prev)
                merged = {**rec, **{k: v for k, v in fetched.items()
                                    if k in ("status", "sha256", "chars", "error")
                                    and v not in (None, "")}}
                merged["queries"] = sorted(set(prev.get("queries", []))
                                           | set(rec.get("queries", [])))
                ix[pid] = merged
        print(f"  {f.name}: {len(shard)} papers")
    save_index(ix)
    print(f"corpus {len(ix)} papers ({added} new from shards)")
    return 0


def year_of(paper: dict) -> int:
    """Publication year: the arXiv `published` field when present, else the YYMM id."""
    pub = (paper.get("published") or "")[:4]
    if pub.isdigit():
        return int(pub)
    m = re.match(r"(\d{2})(\d{2})\.", paper.get("id", ""))   # arXiv ids are YYMM
    return 2000 + int(m.group(1)) if m else 0


def cmd_screen(args) -> int:
    """Run every query, dedupe, score, and record. Abstracts only -- this is triage.

    SAVE AFTER EVERY QUERY, and remember which queries actually returned something.
    The first run of this took 17+ minutes with nothing on disk, because the index was
    written once at the end: the arXiv export API throttles a sustained sweep, each
    throttled query burns up to three 60-second curl timeouts inside cp.curl(), and a
    kill or a crash anywhere in that window would have thrown away the entire corpus.
    Persisting per query also makes the sweep RESUMABLE -- `--resume` skips the queries
    that already yielded, so a throttled sweep can be finished in a second pass instead
    of re-paying for the queries that worked.
    """
    # SHARDING. A cold relevance-sorted arXiv query costs 50-110s to build server-side
    # (measured: 68.2s, 108.3s, 51.0s, then 0.3s once warm), so a serial 48-query sweep is
    # ~45-90 minutes of wall clock on the critical path. Shards split the query list and
    # run concurrently, each writing its OWN index file -- concurrent writers on one index
    # would clobber each other, since save_index() writes a whole in-memory dict. Merge
    # afterwards with `lit.py merge`.
    ix_path = pathlib.Path(args.index) if getattr(args, "index", None) else INDEX
    ix = load_index(ix_path)
    seen_before = len(ix)
    done_f = ix_path.with_name(ix_path.stem + "_progress.json")
    queries = list(cp.QUERIES)
    if getattr(args, "shard", None):
        i, n = (int(x) for x in args.shard.split("/"))
        queries = [q for k, q in enumerate(queries) if k % n == i]
        print(f"shard {i}/{n}: {len(queries)} of {len(cp.QUERIES)} queries -> {ix_path.name}",
              flush=True)
    try:
        done = set(json.loads(done_f.read_text()))
    except (OSError, ValueError):
        done = set()
    for name, query in queries:
        if args.resume and name in done:
            continue
        try:
            found = cp.search(query, args.per_query)
        except Exception as exc:                    # noqa: BLE001 - one bad query must
            print(f"  {name}: query failed ({exc})")  # not abort a 47-query sweep
            continue
        if not found:
            # A throttled or malformed query is NOT progress: leave it out of `done` so
            # --resume retries it rather than silently baking a hole into the corpus.
            print(f"  {name:26s}   0 found  (throttled or no match; will retry on --resume)",
                  flush=True)
            time.sleep(args.sleep)
            continue
        kept = 0
        for p in found:
            # curate_papers.search() emits "arxiv_id"; the index is keyed on "id".
            pid = p.get("arxiv_id") or p.get("id")
            if not pid:
                continue
            s, why = cp.score(p)
            if s < args.min_score:
                continue
            prev = ix.get(pid, {})
            ix[pid] = {**prev, "id": pid, "title": p["title"].strip(),
                       "abstract": p["abstract"].strip()[:1500],
                       "published": p.get("published", ""), "comment": p.get("comment", ""),
                       "score": s, "why": why[:8], "queries": sorted(
                           set(prev.get("queries", [])) | {name}),
                       "families": families_for(p),
                       "status": prev.get("status", "screened")}
            kept += 1
        print(f"  {name:26s} {len(found):3d} found, {kept:3d} kept  "
              f"(corpus {len(ix)})", flush=True)
        done.add(name)
        save_index(ix, ix_path)         # per query: a kill must not cost the sweep
        done_f.write_text(json.dumps(sorted(done), indent=1))
        time.sleep(args.sleep)          # arXiv asks for >=3s between API calls
        if len(ix) >= args.target:
            print(f"  target {args.target} reached")
            break
    save_index(ix, ix_path)
    print(f"\ncorpus {len(ix)} papers ({len(ix)-seen_before} new). "
          f"2025+: {sum(1 for p in ix.values() if year_of(p) >= 2025)}")
    return 0


def cmd_fetch(args) -> int:
    """Fetch full text for the highest-scoring unfetched papers. Newest first."""
    ix = load_index()
    SOURCES.mkdir(parents=True, exist_ok=True)
    todo = [p for p in ix.values() if p.get("status") != "fetched"]
    if args.family:
        todo = [p for p in todo if args.family in p.get("families", [])]
    # 2025-2026 first, as requested, then by relevance score
    todo.sort(key=lambda p: (-(year_of(p) >= 2025), -p.get("score", 0)))
    todo = todo[: args.limit]
    # Same reason screen shards: this is on the critical path and each LaTeXML render is
    # a separate HTTP fetch of a large page. Shards take disjoint slices of the SAME
    # sorted todo list, so concurrent fetchers never download the same paper twice. Each
    # shard's index bookkeeping is thrown away; the authoritative pass is a final serial
    # `fetch` with no shard, which sees every file already on disk, records its digest,
    # and makes no network call for it.
    if getattr(args, "shard", None):
        i, n = (int(x) for x in args.shard.split("/"))
        todo = [t for k, t in enumerate(todo) if k % n == i]
        print(f"shard {i}/{n}: {len(todo)} papers to fetch", flush=True)
    if not todo:
        print("nothing to fetch")
        return 0
    ok = 0
    for p in todo:
        dest = SOURCES / f"arxiv_{p['id']}_fulltext.txt"
        if dest.exists() and dest.stat().st_size > 2000:
            p["status"] = "fetched"
            p["sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
            p["chars"] = dest.stat().st_size
            ok += 1
            continue
        try:
            text = rf.fulltext(p["id"])
        except Exception as exc:                    # noqa: BLE001
            p["status"] = "fetch_failed"
            p["error"] = str(exc)[:200]
            print(f"  {p['id']}  FAILED  {exc}")
            time.sleep(args.sleep)
            continue
        if not text or len(text) < 2000:
            p["status"] = "fetch_failed"
            p["error"] = f"only {len(text or '')} chars (no LaTeXML render?)"
            print(f"  {p['id']}  too short ({len(text or '')} chars)")
            time.sleep(args.sleep)
            continue
        dest.write_text(text)
        p["status"] = "fetched"
        p["sha256"] = hashlib.sha256(text.encode()).hexdigest()
        p["chars"] = len(text)
        ok += 1
        print(f"  {p['id']}  {len(text):>7,} chars  {p['title'][:60]}")
        time.sleep(args.sleep)
    # RE-READ before writing. A fetch run holds its index copy for many minutes while it
    # downloads, and in that window the corpus can grow (a screen sweep merging in, or a
    # sibling fetch shard). Writing the stale in-memory copy back would delete every
    # paper added meanwhile -- four concurrent shards started at 190 papers would each
    # have clobbered the 233 the corpus had grown to. Only our own per-paper fields are
    # written over the current file.
    cur = load_index()
    for pid, rec in ix.items():
        if pid not in cur:
            cur[pid] = rec
            continue
        for k in ("status", "sha256", "chars", "error"):
            if rec.get(k) not in (None, ""):
                cur[pid][k] = rec[k]
    ix = cur
    save_index(ix)
    print(f"\nfetched {ok}/{len(todo)}; corpus now "
          f"{sum(1 for q in ix.values() if q.get('status') == 'fetched')} full texts")
    return 0


def cmd_reindex(args) -> int:
    """Record status/sha256/chars for every snapshot already on disk. No network.

    The index and lit/sources/ can disagree whenever a fetch is interrupted, sharded, or
    (as happened here) still running while claims are registered against papers whose
    text is already downloaded. coe.py E1 rightly refuses such a claim -- "2512.05620 is
    not marked fetched" -- even though the full text is sitting on disk. This reconciles
    the two from the FILES, which are the authority: a snapshot that exists and hashes is
    fetched, whatever the index last remembered.
    """
    ix = load_index()
    fixed = missing = 0
    for f in sorted(SOURCES.glob("arxiv_*_fulltext.txt")):
        pid = f.name[len("arxiv_"):-len("_fulltext.txt")]
        if f.stat().st_size <= 2000:
            continue
        if pid not in ix:
            missing += 1
            continue
        rec = ix[pid]
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if rec.get("status") != "fetched" or rec.get("sha256") != digest:
            rec["status"] = "fetched"
            rec["sha256"] = digest
            rec["chars"] = f.stat().st_size
            rec.pop("error", None)
            fixed += 1
    save_index(ix)
    print(f"reindexed {fixed} snapshot(s); corpus now "
          f"{sum(1 for q in ix.values() if q.get('status') == 'fetched')} full texts"
          + (f"; {missing} snapshot(s) on disk are not in the index" if missing else ""))
    return 0


def corpus_state() -> dict:
    """Per-family counts: screened, fetched. Claims are added by tools/claims.py."""
    ix = load_index()
    st = {f: {"screened": 0, "fetched": 0} for f in ALL_FAMILIES + ("unassigned",)}
    for p in ix.values():
        for f in p.get("families", ["unassigned"]):
            if f not in st:
                st[f] = {"screened": 0, "fetched": 0}
            st[f]["screened"] += 1
            if p.get("status") == "fetched":
                st[f]["fetched"] += 1
    return st


def cmd_status(args) -> int:
    ix = load_index()
    fetched = [p for p in ix.values() if p.get("status") == "fetched"]
    failed = [p for p in ix.values() if p.get("status") == "fetch_failed"]
    print(f"corpus: {len(ix)} screened, {len(fetched)} full texts on disk, "
          f"{len(failed)} fetch failures")
    print(f"        2025+: {sum(1 for p in ix.values() if year_of(p) >= 2025)} screened, "
          f"{sum(1 for p in fetched if year_of(p) >= 2025)} fetched")
    print(f"\n{'family':20s} {'screened':>9s} {'fetched':>8s}")
    for f, c in sorted(corpus_state().items(), key=lambda kv: kv[1]["fetched"]):
        print(f"{f:20s} {c['screened']:9d} {c['fetched']:8d}")
    return 0


def cmd_read(args) -> int:
    """The reading list for one direction: unread full texts, best first."""
    ix = load_index()
    pool = [p for p in ix.values()
            if args.family in p.get("families", []) and p.get("status") == "fetched"]
    pool.sort(key=lambda p: (-(year_of(p) >= 2025), -p.get("score", 0)))
    if not pool:
        print(f"no fetched full text for '{args.family}'. "
              f"Run: lit.py fetch --family {args.family} --limit 20")
        return 1
    print(f"{len(pool)} full texts for '{args.family}' (2025+ first, then relevance):\n")
    for p in pool[: args.limit]:
        print(f"  [{p['score']:3d}] {p['id']}  {p['title'][:72]}")
        print(f"        lit/sources/arxiv_{p['id']}_fulltext.txt  ({p.get('chars',0):,} chars)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen", help="arXiv triage across the mechanism query set")
    s.add_argument("--target", type=int, default=240)
    s.add_argument("--per-query", type=int, default=14)
    s.add_argument("--min-score", type=int, default=6)
    s.add_argument("--sleep", type=float, default=3.0)
    s.add_argument("--resume", action="store_true",
                   help="skip queries that already returned hits (lit/screen_progress.json)")
    s.add_argument("--shard", default=None, metavar="I/N",
                   help="run only every Nth query starting at I, for concurrent sweeps")
    s.add_argument("--index", default=None,
                   help="write to this index file instead of lit/index.json (use with --shard)")
    s.set_defaults(fn=cmd_screen)

    f = sub.add_parser("fetch", help="download full text for the best unfetched papers")
    f.add_argument("--limit", type=int, default=40)
    f.add_argument("--family", default=None)
    f.add_argument("--sleep", type=float, default=3.0)
    f.add_argument("--shard", default=None, metavar="I/N",
                   help="fetch only every Nth paper of the todo list, for concurrent runs")
    f.set_defaults(fn=cmd_fetch)

    t = sub.add_parser("status", help="corpus coverage per direction family")
    t.set_defaults(fn=cmd_status)

    r = sub.add_parser("read", help="reading list for one direction family")
    r.add_argument("family")
    r.add_argument("--limit", type=int, default=15)
    r.set_defaults(fn=cmd_read)

    ri = sub.add_parser("reindex", help="record digests for snapshots already on disk (no network)")
    ri.set_defaults(fn=cmd_reindex)

    m = sub.add_parser("merge", help="fold sharded index files into lit/index.json")
    m.add_argument("--pattern", default="index_shard*.json")
    m.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
