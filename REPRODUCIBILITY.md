# Reproducibility Entry Point

This file points to the compact reproduction surfaces for the FGCS manuscript.
The detailed matrix is [`docs/current/FGCS_REPRODUCIBILITY_MATRIX_V1.md`](docs/current/FGCS_REPRODUCIBILITY_MATRIX_V1.md).

## Core environment

Install the package and development dependencies with:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest --collect-only -q
```

Optional selector and checkpoint extras are documented in `pyproject.toml`.

## Manuscript

From `paper/llm2026/`, regenerate current figures with:

```bash
python scripts/plot_fgcs_figures.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The canonical source, figures, PDF, and build manifest are listed in
`paper/llm2026/FGCS_SUBMISSION_PACKAGE_MANIFEST_V1.md`.

## Data and large campaigns

Compact result tables, manifests, hashes, and scripts are versioned where
appropriate. Azure and BurstGPT raw traces are not redistributed by default;
see `docs/DATA_RELEASE_POLICY.md` and `docs/PUBLIC_RELEASE_MANIFEST.md`.
Large historical campaigns used Wulver where documented, but compact manuscript
analysis does not require Wulver unless its provenance record says otherwise.

Real-vLLM uses a dedicated environment and is not reproduced by the core
simulator environment. See the reproducibility matrix for its environment and
provenance files.
