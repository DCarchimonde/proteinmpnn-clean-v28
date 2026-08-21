#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$paper_dir"

command -v xelatex >/dev/null 2>&1 || { echo "缺少 xelatex，请安装 TeX Live。" >&2; exit 1; }
command -v bibtex >/dev/null 2>&1 || { echo "缺少 bibtex，请安装 TeX Live。" >&2; exit 1; }

xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex

echo "完成：$paper_dir/main.pdf"

