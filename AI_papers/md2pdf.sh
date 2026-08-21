#!/usr/bin/env bash
# md2pdf.sh <file.md> — convert a block/report paper markdown to a clean PDF.
set -euo pipefail
SRC="$1"; OUT="${SRC%.md}.pdf"; TMP="$(mktemp).md"
# map glyphs the default LaTeX font lacks to LaTeX-safe equivalents (PDF only; .md keeps unicode)
sed -e 's/≈/~/g' -e 's/≫/>>/g' -e 's/≪/<</g' -e 's/λ/lambda/g' -e 's/κ/kappa/g' \
    -e 's/≥/>=/g' -e 's/≤/<=/g' -e 's/σ/sigma/g' -e 's/×/x/g' -e 's/·/./g' \
    -e 's/→/->/g' -e 's/↓/(down)/g' -e 's/↑/(up)/g' -e 's/Δ/Delta/g' -e 's/ε/eps/g' \
    "$SRC" > "$TMP"
pandoc "$TMP" -o "$OUT" --pdf-engine=tectonic -V geometry:margin=1in -V fontsize=10pt 2>/dev/null
rm -f "$TMP"; echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
