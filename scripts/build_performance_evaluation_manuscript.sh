#!/bin/bash
set -e

# Rebuild the Performance Evaluation (Elsevier format) manuscript and figures

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper/performance_evaluation"

echo "=== 1. Regenerating Manuscript Figures from Frozen Artifacts ==="
python3 "$PAPER_DIR/scripts/plot_performance_evaluation_figures.py"
python3 "$PAPER_DIR/scripts/plot_regime_figures.py"
python3 "$PAPER_DIR/scripts/plot_robustness_figures.py"

echo "=== 1b. Verifying the Quantitative Claims Against the Canonical Artifacts ==="
python3 "$PAPER_DIR/scripts/build_claim_manifest.py" --check

echo "=== 2. Rebuilding LaTeX Manuscript ==="
cd "$PAPER_DIR"

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# The canonical PDF is the frozen manuscript. Overwrite it only on request,
# so that verifying the build does not modify a tracked, frozen artifact.
if [ "${UPDATE_CANONICAL_PDF:-0}" = "1" ]; then
    echo "=== 3. Updating Canonical Review PDF ==="
    cp main.pdf "$ROOT_DIR/paper/when_does_llm_serving_scheduler_adaptation_matter.pdf"
else
    echo "=== 3. Built $PAPER_DIR/main.pdf (canonical PDF left unchanged; set UPDATE_CANONICAL_PDF=1 to overwrite it) ==="
fi

echo "=== Rebuild Complete Successfully! ==="
