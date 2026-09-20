#!/bin/bash
set -e

# Rebuild the Performance Evaluation (Elsevier format) manuscript and figures

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper/performance_evaluation"

echo "=== 1. Regenerating Manuscript Figures from Frozen Artifacts ==="
python3 "$PAPER_DIR/scripts/plot_performance_evaluation_figures.py"

echo "=== 2. Rebuilding LaTeX Manuscript ==="
cd "$PAPER_DIR"

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "=== 3. Updating Canonical Review PDF ==="
cp main.pdf "$ROOT_DIR/paper/llm_scheduler_adaptation_causal_headroom.pdf"

echo "=== Rebuild Complete Successfully! ==="
