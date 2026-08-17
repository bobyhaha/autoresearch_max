#!/usr/bin/env python3
"""Print the evidence-bearing parts of a preserved full-text snapshot.

Reading 140 papers end to end verbatim is not the same thing as reading them well. The
parts that decide whether a mechanism can move `val_bpb` in 300 seconds are: the
abstract, the experimental setup (what scale, what horizon, what hardware -- this is
what kills most transfers), every results table, every figure caption (effect sizes hide
there), and the limitations. Motivation and related work almost never change a verdict.

So this prints those parts from the local snapshot, at full length, with the section map
so I can then go back and read any section verbatim with sed. Triage happens on the full
text, never on an abstract.

    uv run python tools/paper_digest.py 2604.01472
    uv run python tools/paper_digest.py 2604.01472 --section "5.1"
    uv run python tools/paper_digest.py --all --brief    # one screen per paper
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SNAP = Path(__file__).resolve().parent.parent / "runs" / "sources"

SETUP = re.compile(
    r"^## .*(setup|experimental|implementation detail|configuration|training detail|"
    r"model and data|hyperparameter|dataset)", re.IGNORECASE)
RESULT = re.compile(
    r"^## .*(result|experiment|ablation|evaluation|comparison|analysis|finding|"
    r"discussion|conclusion|limitation|takeaway)", re.IGNORECASE)
# Tables survive extraction as pipe-joined cells; captions start with Table/Figure.
TABLEISH = re.compile(r"^\s*\|| \| ")
CAPTION = re.compile(r"^(Table|Figure)\s*\d+", re.IGNORECASE)
BOILERPLATE = re.compile(r"Report GitHub Issue|arXiv is now an independent|"
                         r"Content selection saved|License:|^\s*×\s*$", re.IGNORECASE)


def sections(lines: list[str]) -> list[tuple[int, str]]:
    return [(i, ln[3:].strip()) for i, ln in enumerate(lines) if ln.startswith("## ")]


def span(lines: list[str], start: int) -> int:
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            return j
    return len(lines)


def digest(path: Path, brief: bool) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    heads = sections(lines)
    print("=" * 100)
    for ln in lines[:5]:
        if ln.startswith("#"):
            print(ln)
    print("=" * 100)

    print(f"\n---- section map ({len(heads)}) ----")
    for _, title in heads:
        if not BOILERPLATE.search(title):
            print(f"  {title[:96]}")

    for label, want in (("ABSTRACT", re.compile(r"^abstract$", re.IGNORECASE)),
                        ("SETUP", SETUP),
                        ("RESULTS / ABLATIONS / LIMITATIONS", RESULT)):
        hits = [(i, t) for i, t in heads if want.search(t if label == "ABSTRACT" else "## " + t)]
        if not hits:
            continue
        print(f"\n---- {label} ----")
        cap = 2500 if brief else 100000
        for i, title in hits:
            body = [ln for ln in lines[i + 1: span(lines, i)] if not BOILERPLATE.search(ln)]
            text = "\n".join(body).strip()
            if not text:
                continue
            print(f"\n## {title}")
            print(text[:cap] + ("\n  […truncated, read verbatim with sed…]" if len(text) > cap else ""))

    tables = [ln.strip() for ln in lines if TABLEISH.search(ln) and len(ln.strip()) > 12]
    caps = [ln.strip() for ln in lines if CAPTION.match(ln.strip())]
    if tables:
        print(f"\n---- every table row ({len(tables)}) ----")
        for ln in tables[: 60 if brief else 400]:
            print("  " + ln[:180])
    if caps:
        print(f"\n---- every caption ({len(caps)}) ----")
        for ln in caps[: 15 if brief else 200]:
            print("  " + ln[:400])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--section", help="print one section verbatim by title substring")
    args = ap.parse_args()

    paths = (sorted(SNAP.glob("arxiv_*_fulltext.txt")) if args.all
             else [SNAP / f"arxiv_{i}_fulltext.txt" for i in args.ids])
    for path in paths:
        if not path.exists():
            print(f"!! missing snapshot {path.name} -- fetch it with tools/read_fulltext.py")
            continue
        if args.section:
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, title in sections(lines):
                if args.section.lower() in title.lower():
                    print(f"\n## {title}")
                    print("\n".join(lines[i + 1: span(lines, i)]))
        else:
            digest(path, args.brief)


if __name__ == "__main__":
    main()
