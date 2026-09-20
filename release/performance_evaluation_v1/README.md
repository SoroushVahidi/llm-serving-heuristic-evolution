# Reproducibility & Data Archive (Performance Evaluation v1)

This archive contains the frozen experimental results, plotting scripts, and LaTeX manuscript sources to accompany the paper:

**"When Does LLM-Serving Scheduler Adaptation Matter? Action Opportunity and Causal Headroom in Production-Derived Replay"**

---

**DOI:** 10.5281/zenodo.22865294

## Archive Structure

- **`main.tex`**: The Performance Evaluation manuscript LaTeX source (Elsevier `elsarticle` format).
- **`references.bib`**: BibTeX bibliography file.
- **`main.pdf`**: Pre-built author-review PDF.
- **`performance_evaluation_highlights.txt`**: Core highlights of the paper (bulleted, <=85 chars/line).
- **`REPRODUCIBILITY.md`**: Guide explaining environment, workload provenance, and reproduction steps.
- **`CITATION.cff`**: Citation metadata.
- **`LICENSE`**: MIT License.
- **`pyproject.toml` / `requirements.txt`**: Python dependencies.
- **`scripts/plot_performance_evaluation_figures.py`**: Python script using matplotlib to regenerate Figures 2-5.
- **`experiments/`**:
  - **`fresh_production_latency_headroom_confirmatory_v1/`**: Directory containing the frozen point estimates and CSV results for Table 3, Figure 4, and Figure 5.
  - **`industry_realism_action_opportunity_phase_b_v2/`**: Contains the transition rate CSV file (`PHASE_B_V2_WORKLOAD_AXIS_SUMMARY.csv`) required to plot Figure 3.

---

## Quickstart Instructions

### 1. Rebuild the Manuscript
Ensure `pdflatex` is installed on your system.
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 2. Regenerate All Figures
Ensure python3 and dependencies (`matplotlib`, `numpy`, `pandas`) are installed.
```bash
# Regenerate Figures 2, 3, 4, and 5
python3 scripts/plot_performance_evaluation_figures.py
```
The figures will be rewritten directly as PDF and PNG files into the `figures/` directory.
