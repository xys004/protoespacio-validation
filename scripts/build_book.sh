#!/usr/bin/env bash
# Build politico: tests antes que PDF.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "==> pytest"
pytest -q

echo "==> latexmk book/main.tex"
cd book
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

echo "==> OK: $ROOT/book/main.pdf"
