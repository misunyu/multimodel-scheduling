#!/usr/bin/env bash
# paper/build.sh — 배포본 4패스 빌드
# 그림은 ../figures/ 에서 읽는다 (main.tex 의 \graphicspath).
set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > build.log

pages=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
unresolved=$(grep -c 'LaTeX Warning: \(Reference\|Citation\)' build.log || true)

echo "pages:      $pages   (expected 22)"
echo "unresolved: $unresolved   (expected 0)"

[ "$unresolved" -eq 0 ] || { echo "FAIL: 미해결 참조가 있다"; exit 1; }
echo "OK"
