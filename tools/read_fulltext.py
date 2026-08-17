#!/usr/bin/env python3
"""Fetch arXiv FULL TEXT, preserve a snapshot, and register it as a citable source.

Why this exists. The campaign had read 84 abstracts and exactly one full text, and the
one full text could not even be registered: `fulltext_snapshot` requires a sha256 of a
preserved artefact, and it had been read through a live fetch that saved nothing. An
abstract is an advertisement -- reading arXiv 2605.26895 properly turned one vague
claim into four sharp ones and revealed that a running experiment was misconfigured,
none of which was visible from its abstract.

So: fetch the LaTeXML HTML, reduce it to structured text, write the snapshot to disk,
hash it, and emit a LiteratureSource declaration whose `content` block points at that
file. Provenance is then a file on disk with a digest rather than a memory of a fetch.

Figure captions and tables are kept deliberately. Every number that mattered from the
first paper read this way -- the 0.028/0.015 effect sizes, the 1.04x overhead -- came
from a caption or a table, never from the prose.

    uv run python tools/read_fulltext.py --ids 2605.26895 2607.10959 \
        --topic axis_architecture_signal_path
    uv run python tools/read_fulltext.py --from-file papers.txt --limit 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SNAP = REPO / "runs" / "sources"
DECL = REPO / "runs" / "science" / "sources"
ATOM = "{http://www.w3.org/2005/Atom}"

SKIP_TAGS = {"script", "style", "noscript", "svg"}
HEADING = {"h1", "h2", "h3", "h4", "h5", "h6"}
BREAK = {"p", "div", "li", "tr", "figcaption", "caption", "section", "table", "br"}


class Extract(HTMLParser):
    """LaTeXML HTML -> structured plain text, keeping headings, tables and captions."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.heading = 0

    def handle_starttag(self, tag, attrs):
        if tag == "math":
            # LaTeXML wraps numbers in MathML: the presentation markup would arrive as
            # character soup and the <annotation> would duplicate it. `alttext` carries
            # the original LaTeX, so emit that and skip the subtree. Dropping math
            # outright would silently delete the effect sizes -- the whole point.
            self.out.append(f" {dict(attrs).get('alttext', '')} ")
            self.skip += 1
        elif tag in SKIP_TAGS:
            self.skip += 1
        elif tag in HEADING:
            self.heading += 1
            self.out.append("\n\n## ")
        elif tag in BREAK:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if (tag == "math" or tag in SKIP_TAGS) and self.skip:
            self.skip -= 1
        elif tag in HEADING and self.heading:
            self.heading -= 1
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip:
            return
        text = re.sub(r"[ \t\r\f\v]+", " ", data)
        if text.strip():
            self.out.append(text)

    def text(self) -> str:
        raw = "".join(self.out)
        raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
        return "\n".join(line.rstrip() for line in raw.splitlines()).strip()


def curl(url: str, timeout: int = 60, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        done = subprocess.run(
            ["curl", "-sS", "-L", "--max-time", str(timeout), url],
            capture_output=True, text=True, check=False,
        )
        if done.returncode == 0 and done.stdout.strip():
            return done.stdout
        time.sleep(3 * (attempt + 1))
    return ""


def metadata(arxiv_id: str) -> dict:
    """Title, authors, date, abstract from the Atom API -- the authoritative record."""
    url = ("https://export.arxiv.org/api/query?"
           + urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1}))
    body = curl(url, 45)
    if not body.strip().startswith("<?xml"):
        return {}
    try:
        entry = ET.fromstring(body).find(f"{ATOM}entry")
    except ET.ParseError:
        return {}
    if entry is None:
        return {}
    authors, seen = [], set()
    for a in entry.findall(f"{ATOM}author"):
        n = (a.findtext(f"{ATOM}name") or "").strip()
        if n and n not in seen:
            seen.add(n)
            authors.append(n)
    published = (entry.findtext(f"{ATOM}published") or "")[:10]
    return {
        "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
        "abstract": " ".join((entry.findtext(f"{ATOM}summary") or "").split()),
        "authors": authors,
        "published": published,
        "year": int(published[:4]) if published[:4].isdigit() else 0,
    }


def fulltext(arxiv_id: str) -> str:
    """LaTeXML HTML if arXiv rendered one. Older papers have none; that is not an error."""
    for suffix in ("v1", ""):
        html_doc = curl(f"https://arxiv.org/html/{arxiv_id}{suffix}")
        if "<!DOCTYPE" in html_doc[:200] and len(html_doc) > 20000:
            parser = Extract()
            parser.feed(html_doc)
            text = parser.text()
            if len(text) > 4000:
                return text
    return ""


def agenda_topics(path: Path) -> set[str]:
    """Load the one agenda this import is intended to satisfy."""
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read agenda {path}: {exc}") from exc
    return {
        str(topic["id"])
        for topic in document.get("topics", [])
        if isinstance(topic, dict) and isinstance(topic.get("id"), str)
    }


TAG_TO_AXIS = {
    "speedrun": "axis_optimizer_geometry",
    "optimizer_adam_variants": "axis_optimizer_geometry",
    "optimizer_secondorder": "axis_optimizer_geometry",
    "optimizer_muon": "axis_optimizer_geometry",
    "optimizer_orthogonal": "axis_optimizer_geometry",
    "weight_decay": "axis_optimizer_geometry",
    "mup": "axis_optimizer_geometry",
    "lr_wsd": "axis_schedule_horizon",
    "lr_schedule": "axis_schedule_horizon",
    "lr_warmup": "axis_schedule_horizon",
    "batch_size_law": "axis_schedule_horizon",
    "attention_window": "axis_architecture_signal_path",
    "normalization": "axis_architecture_signal_path",
    "activation": "axis_architecture_signal_path",
    "data_quality": "axis_data_token_efficiency",
    "data_selection": "axis_data_token_efficiency",
    "curriculum": "axis_data_token_efficiency",
    "fp8_training": "axis_systems_throughput",
    "kernel_efficiency": "axis_systems_throughput",
    "memory_efficiency": "axis_systems_throughput",
    "stability": "axis_measurement_integrity",
}


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", s)[:70]


def one(arxiv_id: str, topic: str) -> dict | None:
    meta = metadata(arxiv_id)
    if not meta.get("title"):
        print(f"  !! {arxiv_id}: no metadata")
        return None
    body = fulltext(arxiv_id)
    status = "fulltext_snapshot" if body else "abstract_only"
    base = slug(meta["title"]) or arxiv_id.replace(".", "_")

    content: dict = {"status": status}
    if body:
        snap = SNAP / f"arxiv_{arxiv_id.replace('/', '_')}_fulltext.txt"
        header = (f"# {meta['title']}\n# arXiv {arxiv_id} · {meta['published']}\n"
                  f"# authors: {', '.join(meta['authors'][:12])}\n"
                  f"# retrieved {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                  f"from arxiv.org/html/{arxiv_id}\n\n")
        snap.write_text(header + body, encoding="utf-8")
        content = {
            "status": "fulltext_snapshot",
            "sha256": hashlib.sha256(snap.read_bytes()).hexdigest(),
            "path": f"../../sources/{snap.name}",
            "chars": len(body),
        }
    decl = {
        "id": f"lit_{base}",
        "work_key": f"work_arxiv_{arxiv_id.replace('.', '_').replace('/', '_')}",
        "title": meta["title"],
        "authors": meta["authors"],
        "year": meta["year"],
        "venue": {"name": "arXiv", "tier": "preprint", "peer_reviewed": "unknown"},
        "publication_type": "preprint",
        "identifiers": {"arxiv": arxiv_id},
        "urls": [f"https://arxiv.org/abs/{arxiv_id}"],
        "abstract": meta["abstract"],
        "topics": [topic],
        "retrieval": {"provider": "arxiv_html_fulltext" if body else "arxiv_atom_api",
                      "query": f"id_list={arxiv_id}",
                      "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "content": content,
        "citation_count": 0,
    }
    (DECL / f"{decl['id']}.json").write_text(json.dumps(decl, indent=2, sort_keys=True) + "\n")
    mark = f"{content.get('chars', 0) // 1000}k chars" if body else "ABSTRACT ONLY (no HTML)"
    print(f"  {arxiv_id:14s} {status:18s} {mark:22s} {meta['title'][:60]}")
    return decl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--from-file", help="file with one arXiv id per line, # comments ok")
    ap.add_argument("--topic", help="fallback registered topic; required for bare --ids")
    ap.add_argument(
        "--agenda",
        default=str(REPO / "runs" / "science" / "fresh_start" / "agenda.json"),
        help="the exact agenda whose topics may be stamped onto new sources",
    )
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    ids: list = [(i, None) for i in args.ids]
    if args.from_file:
        for line in Path(args.from_file).read_text().splitlines():
            body, _, comment = line.partition("#")
            parts = body.split()
            if not parts:
                continue
            # Preserve either an explicit agenda id or the curation tag after `[score]`.
            # The previous parser stripped the entire comment before looking for a topic,
            # so every paper silently inherited one unrelated fallback topic.
            explicit = re.search(r"\b(?:axis|topic)_[a-z0-9_]+\b", comment)
            tag_match = re.search(r"\]\s+([a-z0-9_]+)", comment)
            topic = explicit.group(0) if explicit else None
            if topic is None and tag_match:
                topic = TAG_TO_AXIS.get(tag_match.group(1))
            ids.append((parts[0], topic))
    ids = ids[: args.limit]
    if not ids:
        raise SystemExit("give --ids or --from-file")

    SNAP.mkdir(parents=True, exist_ok=True)
    DECL.mkdir(parents=True, exist_ok=True)
    full = 0
    known = agenda_topics(Path(args.agenda))
    for i, (arxiv_id, topic) in enumerate(ids, 1):
        chosen = topic or args.topic
        if not chosen:
            raise SystemExit(
                f"{arxiv_id} has no axis mapping; pass --topic or add an explicit axis_* "
                "identifier to its --from-file comment"
            )
        if chosen not in known:
            raise SystemExit(
                f"topic {chosen!r} is absent from agenda {args.agenda} "
                f"(known: {sorted(known)}).\n"
                f"Register an agenda declaring it before writing sources against it -- an\n"
                f"unregistered topic makes the immutable store invalid and blocks synthesize."
            )
        print(f"[{i}/{len(ids)}]", end=" ")
        decl = one(arxiv_id, chosen)
        if decl and decl["content"]["status"] == "fulltext_snapshot":
            full += 1
        time.sleep(3)  # arXiv asks one request every 3 seconds
    print(f"\n{full}/{len(ids)} with full text preserved under runs/sources/")
    print("register with:")
    print('  for f in runs/science/sources/lit_*.json; do '
          'uv run autoresearch --root .autoresearch literature-source "$f" '
          '| grep -q error && echo "FAILED $f"; done')


if __name__ == "__main__":
    main()
