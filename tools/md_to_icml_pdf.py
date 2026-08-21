#!/usr/bin/env python3
"""Render an AI_papers/*.md into a PDF using the OFFICIAL ICML 2024 LaTeX template.

Pipeline: split the markdown into title / authors / abstract / body; convert the
body + abstract to LaTeX with pandoc; assemble a document around the official
`icml2024.sty` (extracted under AI_papers/.icml_style/icml2024/) using its own
macros (\\icmltitle, icmlauthorlist, abstract, \\section); compile with tectonic.
Falls back to a CSS+Chrome approximation only if tectonic/template are missing.

Usage: python tools/md_to_icml_pdf.py AI_papers/paper_00X_....md
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STYLE_DIR = REPO / "AI_papers/.icml_style/icml2024"
STYLE_FILES = ["icml2024.sty", "icml2024.bst", "fancyhdr.sty", "algorithm.sty", "algorithmic.sty"]


def _strip_section_numbers(md: str) -> str:
    # ICML auto-numbers sections; drop the leading "1." / "3.10" in md headings.
    return re.sub(r"(?m)^(#{2,4})\s+\d+(?:\.\d+)*\.?\s+", r"\1 ", md)


_UNICODE = {
    "σ": r"$\sigma$", "Σ": r"$\Sigma$", "≈": r"$\approx$", "×": r"$\times$",
    "→": r"$\rightarrow$", "Δ": r"$\Delta$", "·": r"$\cdot$", "≥": r"$\geq$",
    "≤": r"$\leq$", "≪": r"$\ll$", "≫": r"$\gg$", "²": r"$^2$", "³": r"$^3$",
    "½": r"$\tfrac12$", "γ": r"$\gamma$", "λ": r"$\lambda$", "β": r"$\beta$",
    "α": r"$\alpha$", "μ": r"$\mu$", "∞": r"$\infty$", "≠": r"$\neq$",
    "±": r"$\pm$", "₁": r"$_1$", "₂": r"$_2$", "√": r"$\surd$",
    "−": r"-", "–": r"--", "—": r"---", "’": r"'", "‘": r"'",
    "“": r"``", "”": r"''", "…": r"\ldots{}",
}


def _fix_unicode(tex: str) -> str:
    for k, v in _UNICODE.items():
        tex = tex.replace(k, v)
    return tex


def _pandoc(md_text: str) -> str:
    out = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "latex", "--wrap=none"],
        input=md_text, capture_output=True, text=True, check=True,
    ).stdout
    return _fix_unicode(out)


def _spanning_tables(tex: str) -> str:
    # Two-column ICML: make each pandoc longtable a full-width, downscaled table*.
    # pandoc emits \toprule/\midrule/\bottomrule already; convert the longtable
    # wrapper to a resized tabular inside a spanning table*.
    out = []
    i = 0
    for m in re.finditer(r"\\begin\{longtable\}.*?\\end\{longtable\}", tex, flags=re.S):
        out.append(tex[i:m.start()])
        raw = m.group(0)
        ncols = max(raw.count(r"\real"), 1)  # pandoc emits one \real{width} per column
        inner = raw
        # drop the whole \begin{longtable}[]{@{}...@{}} preamble (braced colspec)
        inner = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}.*?@\{\}\}", "", inner, flags=re.S)
        inner = inner.replace(r"\end{longtable}", "")
        for junk in (r"\endhead", r"\endfirsthead", r"\endlastfoot", r"\noalign{}"):
            inner = inner.replace(junk, "")
        # unwrap pandoc p{}-column minipage cells
        inner = re.sub(r"\\begin\{minipage\}[^\n]*?\\raggedright", "", inner)
        inner = inner.replace(r"\end{minipage}", "")
        inner = "\n".join(ln for ln in inner.splitlines() if ln.strip())
        colspec = "l" * ncols
        out.append(
            "\\begin{table*}[t]\\centering\\footnotesize\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\resizebox{\\textwidth}{!}{%\n"
            f"\\begin{{tabular}}{{{colspec}}}\n{inner}\n\\end{{tabular}}}}\n\\end{{table*}}"
        )
        i = m.end()
    out.append(tex[i:])
    return "".join(out)


def _split(md: str):
    lines = md.splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), "Untitled")
    # abstract: text under "## Abstract" up to the next heading/hr
    abs = ""
    m = re.search(r"(?ms)^##\s+Abstract\s*\n(.*?)(?=^\#\#\s|\n---|\Z)", md)
    if m:
        abs = m.group(1).strip()
    # body: from the first numbered section ("## 1.") onward
    bm = re.search(r"(?ms)^(##\s+\d+\.?\s.*)\Z", md)
    body = bm.group(1) if bm else md
    return title, abs, body


def _latex_build(md_path: Path, pdf_path: Path) -> bool:
    if not shutil.which("tectonic") or not (STYLE_DIR / "icml2024.sty").exists():
        return False
    md = md_path.read_text(encoding="utf-8")
    title, abstract_md, body_md = _split(md)
    title_tex = _pandoc(title).strip().replace("\n", " ")
    abstract_tex = _pandoc(abstract_md).strip()
    body_tex = _spanning_tables(_pandoc(_strip_section_numbers(body_md)))
    running = re.sub(r"\\texttt\{[^}]*\}|[^A-Za-z0-9 :-]", "", title_tex)[:70]

    doc = r"""\documentclass{article}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}
\usepackage[T1]{fontenc}
\usepackage{url}
\usepackage{hyperref}
\usepackage[accepted]{icml2024}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\icmltitlerunning{%RUNNING%}
\begin{document}
\twocolumn[
\icmltitle{%TITLE%}
\begin{icmlauthorlist}
\icmlauthor{Fable 5 (hypothesis author)}{ophis}
\end{icmlauthorlist}
\icmlaffiliation{ophis}{OPHIS automated-research ledger, \texttt{vibeautoresearch-baiyu-v3}}
\icmlcorrespondingauthor{Fable 5}{ophis}
\icmlkeywords{automated research, throughput, language model pretraining, ICML}
\vskip 0.3in
]
\printAffiliationsAndNotice{}
\begin{abstract}
%ABSTRACT%
\end{abstract}
%BODY%
\end{document}
"""
    doc = (doc.replace("%RUNNING%", running).replace("%TITLE%", title_tex)
              .replace("%ABSTRACT%", abstract_tex).replace("%BODY%", body_tex))

    build = md_path.parent / (".build_" + md_path.stem)
    build.mkdir(exist_ok=True)
    for f in STYLE_FILES:
        src = STYLE_DIR / f
        if src.exists():
            shutil.copy(src, build / f)
    tex = build / (md_path.stem + ".tex")
    tex.write_text(doc, encoding="utf-8")
    res = subprocess.run(
        ["tectonic", "-X", "compile", "--keep-logs", "-o", str(build), str(tex)],
        capture_output=True, text=True,
    )
    produced = build / (md_path.stem + ".pdf")
    if produced.exists() and produced.stat().st_size > 0:
        shutil.copy(produced, pdf_path)
        return True
    sys.stderr.write(res.stderr[-2000:] + "\n")
    return False


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: md_to_icml_pdf.py <paper.md>", file=sys.stderr)
        return 2
    md = Path(sys.argv[1]).resolve()
    pdf = md.with_suffix(".pdf")
    if _latex_build(md, pdf):
        print(f"wrote {pdf} ({pdf.stat().st_size} bytes) via official ICML LaTeX template")
        return 0
    print("LaTeX path unavailable; run failed (see stderr)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
