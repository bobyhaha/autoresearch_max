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


def load_index() -> dict:
    if INDEX.exists():
        try:
            return json.loads(INDEX.read_text())
        except ValueError:
            pass
    return {}


def save_index(ix: dict) -> None:
    LIT.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(ix, indent=1, sort_keys=True))


def year_of(paper: dict) -> int:
    """Publication year: the arXiv `published` field when present, else the YYMM id."""
    pub = (paper.get("published") or "")[:4]
    if pub.isdigit():
        return int(pub)
    m = re.match(r"(\d{2})(\d{2})\.", paper.get("id", ""))   # arXiv ids are YYMM
    return 2000 + int(m.group(1)) if m else 0


def cmd_screen(args) -> int:
    """Run every query, dedupe, score, and record. Abstracts only -- this is triage."""
    ix = load_index()
    seen_before = len(ix)
    for name, query in cp.QUERIES:
        try:
            found = cp.search(query, args.per_query)
        except Exception as exc:                    # noqa: BLE001 - one bad query must
            print(f"  {name}: query failed ({exc})")  # not abort a 47-query sweep
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
              f"(corpus {len(ix)})")
        time.sleep(args.sleep)          # arXiv asks for >=3s between API calls
        if len(ix) >= args.target:
            print(f"  target {args.target} reached")
            break
    save_index(ix)
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
    save_index(ix)
    print(f"\nfetched {ok}/{len(todo)}; corpus now "
          f"{sum(1 for q in ix.values() if q.get('status') == 'fetched')} full texts")
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
    s.set_defaults(fn=cmd_screen)

    f = sub.add_parser("fetch", help="download full text for the best unfetched papers")
    f.add_argument("--limit", type=int, default=40)
    f.add_argument("--family", default=None)
    f.add_argument("--sleep", type=float, default=3.0)
    f.set_defaults(fn=cmd_fetch)

    t = sub.add_parser("status", help="corpus coverage per direction family")
    t.set_defaults(fn=cmd_status)

    r = sub.add_parser("read", help="reading list for one direction family")
    r.add_argument("family")
    r.add_argument("--limit", type=int, default=15)
    r.set_defaults(fn=cmd_read)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
