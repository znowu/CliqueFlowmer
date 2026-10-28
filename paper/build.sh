#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
paper_dir="$PWD"

build_dir="${1:-build}"
mkdir -p -- "$build_dir"
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" final.tex
(
  cd -- "$build_dir"
  BIBINPUTS="$paper_dir:${BIBINPUTS:-}" bibtex final
)
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" final.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" final.tex
# The appendix table of contents needs one more pass after bibliography pagination.
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" final.tex
