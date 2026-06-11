#!/usr/bin/env bash
# Build politico: tests antes que PDF.
# Auto-activa conda env 'protoespacio' y pone TinyTeX en PATH.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# conda env
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate protoespacio
fi

# TinyTeX
if [ -d "$HOME/.TinyTeX/bin/x86_64-linux" ]; then
  export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
fi

echo "==> pytest"
pytest -q

echo "==> latexmk book/main.tex"
cd book
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

echo "==> OK: $ROOT/book/main.pdf"
