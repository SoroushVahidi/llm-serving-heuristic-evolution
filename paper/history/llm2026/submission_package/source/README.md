# LLM 2026 Manuscript Package

LNCS manuscript for *The Exploitability Gap in LLM-Serving Scheduler Portfolios*.

## Contents

- `main.tex` / `references.bib` — manuscript source
- `main.pdf` — compiled PDF (15 LNCS pages as of 2026-08-25 freeze)
- `figures/` — vector figures used by the paper
- `scripts/` — figure regeneration from frozen experiment artifacts
- `llncs.cls`, `splncs04.bst` — LNCS template files required to compile

## Build

```bash
cd paper/llm2026
tectonic --keep-logs main.tex
```

Alternatively, with a conventional TeX install:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Figures

Regenerate from frozen local artifacts only (no new experiments):

```bash
python3 paper/llm2026/scripts/plot_joint_complementarity.py
python3 paper/llm2026/scripts/plot_vllm_semantic_validation.py
```

## Provenance

This package is a finalized manuscript snapshot; the working research
history, evidence ledgers, and experiment artifacts live on the project's
development branch and are not part of this `main`-branch publication. See
`docs/current/llm2026_manuscript_publication_20260825.md` for the exact
source commit and build provenance.
