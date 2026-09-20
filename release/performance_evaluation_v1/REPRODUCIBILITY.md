# Performance Evaluation Reproducibility Guide

This document explains the environment requirements, workload provenance, and step-by-step procedures to reproduce the findings, tables, figures, and compiled manuscript for the Performance Evaluation paper.

---

## 1. Fast-Path Quickstart (Manuscript and Figure Regeneration)

For reviewers and readers wanting to verify the manuscript and figures without running the full massive simulation suite, we provide a fast-path that relies on frozen, verified experimental results.

### Rebuild the Manuscript
We use standard `pdflatex` and `bibtex`. The compiled bibliography `.bbl` is committed directly to the repository to allow building the document in environments without active BibTeX configurations.
```bash
cd paper/performance_evaluation/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
The output is written as `main.pdf`.

### Regenerate Manuscript Figures
The figures can be regenerated instantly from the frozen, verified results JSON/CSV files using our visualization scripts:
```bash
# Regenerate Figures 2, 3, 4, 5
python3 paper/performance_evaluation/scripts/plot_performance_evaluation_figures.py
```
This script reads the frozen results in `paper/performance_evaluation/` and writes updated PDF and PNG vectors directly into `paper/performance_evaluation/figures/`.

---

## 2. Python Environment & Installation

The Python environment requires standard scientific and machine learning libraries.

```bash
pip install -e ".[dev]"          # editable install, pulls pyproject.toml deps
python3 -m pytest --collect-only -q   # should report ~2,500 tests, 0 errors
```

Ensure `pandas`, `numpy`, `pyarrow`, `matplotlib`, and `tabulate` are available.

```bash
python3 -c "import pandas, numpy, matplotlib, pyarrow, tabulate"
```

---

## 3. Workload Provenance & Datasets

Our work utilizes production-derived traces from real-world deployments:
1. **Microsoft Azure LLM serving traces (2023):** Contains `azure_2023_code` and `azure_2023_conversation` workloads.
   - Upstream URL: `https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md`
   - Redistribution status: Raw trace is obtained upstream; we include the derived, sampled simulation windows in this repository for reproducibility.
2. **BurstGPT workload trace (2025/2026):** Production-scale dataset.
   - Upstream URL: `https://github.com/SoroushVahidi/BurstGPT`
   - Redistribution status: Raw trace is obtained upstream; the specific derived simulation windows used for replay are redistributed in `data/`.

All pre-processed, derived workload windows used in our simulations are stored under `data/public_trace_corpus_v1/`.

---

## 4. Scientific Freeze & Computation Map

The core scientific computations in this paper are **frozen**. Running the full suite of simulations is computationally expensive and requires a dedicated high-performance computing (HPC) environment.

| Result / Table / Figure | Script Name | Scope / Environment | Execution Type |
|---|---|---|---|
| **Phase A Replay** (0/99,992, etc.) | `scripts/industry_realism_action_opportunity_phase_a_v1.py` | Local or HPC | Frozen (Re-run optional) |
| **Phase B Pressure** | `scripts/industry_realism_action_opportunity_phase_b_v2.py` | Local or HPC | Frozen (Re-run optional) |
| **Fresh Causal Headroom** | `scripts/fresh_latency_causal_confirmatory_v1.py` | HPC (Wulver Cluster) | Frozen |
| **Figure 2** (Disagreement) | `paper/performance_evaluation/scripts/plot_performance_evaluation_figures.py` | Local | Instant from frozen JSON |
| **Figure 5** (Opportunity Map) | `paper/performance_evaluation/scripts/plot_performance_evaluation_figures.py` | Local | Instant from frozen JSON |
| **Table 1** (Closest-work) | (Static synthesis) | N/A | Analytical mapping |
| **Table 2** (Transition) | (Static synthesis) | N/A | Analytical mapping |
| **Table 3** (Causal characterization) | (Static synthesis) | N/A | Scripted from frozen CSVs |

---

## 5. Bootstrap Clustered Correction Provenance

Our fresh causal headroom points are evaluated using a pre-registered bootstrap clustered analysis to estimate the 95% confidence interval of the latency headroom accurately.
- **The bootstrap unit:** `(source_dataset, window_index)` representing 36 distinct source-window clusters.
- **Correction details:** Corrected an earlier implementation that incorrectly merged separate datasets sharing identical window indexes (26 clusters) into the pre-registered 36-cluster key. Point estimates, seeds, and the confirmatory POSITIVE verdict remained unaffected.
- **Detailed Audit:** See `experiments/fresh_production_latency_headroom_confirmatory_v1/BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md` for the step-by-step mathematical and code audit.

---

## 6. Running Tests

To run the lightweight verification suite:
```bash
python3 -m pytest                 # full non-GPU-safe suite
python3 -m pytest -m gpu          # GPU-only tests (requires a CUDA-capable GPU)
```

---

## 7. Real vLLM Bounded Validation

We validate simulator fidelity on a real serving platform (vLLM) under resource pressure.
- Real-system verification requires a dedicated GPU (local RTX 5060 Ti or Wulver HPC A100 node).
- To view real-system logs and outputs, refer to: `experiments/real_llm/vllm_healthcheck_20260703T171021Z/server.log` and `experiments/real_llm/vllm_baseline_comparison_pilot_20260703T165438Z/server.log`.
- No monetary API calls are performed; mock modes are used by default unless `--allow-live-api` is supplied with correct credentials.
