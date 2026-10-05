# Performance Evaluation Reproducibility Guide

> Detailed guide for the current study. The short public entry point is
> [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md); the current-study index is
> [`docs/current/README.md`](README.md).

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
The six figure files (four used in the manuscript) are regenerated from the frozen, verified result files by three scripts that share one style module
(`figstyle.py`); each writes a vector PDF and a 300 dpi PNG into `paper/performance_evaluation/figures/`:
```bash
python3 paper/performance_evaluation/scripts/plot_performance_evaluation_figures.py   # manuscript Figures 1-2 (+ dropped transition figure)
python3 paper/performance_evaluation/scripts/plot_regime_figures.py                   # manuscript Figure 3 (+ dropped fresh-headroom figure)
python3 paper/performance_evaluation/scripts/plot_robustness_figures.py               # manuscript Figure 4
```
`scripts/build_performance_evaluation_manuscript.sh` runs all three, verifies the claim manifest
(`paper/performance_evaluation/FINAL_CLAIM_MANIFEST.json`) and rebuilds `paper/performance_evaluation/main.pdf`; it replaces the canonical PDF only when `UPDATE_CANONICAL_PDF=1` is set. See
`paper/performance_evaluation/README.md` for the inputs each script reads.

---

## 2. Python Environment & Installation

The Python environment requires standard scientific and machine learning libraries.

```bash
pip install -e ".[dev]"          # editable install, pulls pyproject.toml deps
python3 -m pytest --collect-only -q   # collects the whole suite; expect 0 collection errors
```

The suite covers the whole research program, not only this study. On
2026-10-04 it collected 4,711 tests; CI runs a deterministic CPU-only subset
(4,436 tests; see `.github/workflows/ci.yml`), and 21 tests are GPU-marked and
opt-in. The total changes as tests are added, and some excluded modules fail
without uncommitted datasets (see Section 6).

Ensure `pandas`, `numpy`, `pyarrow`, `matplotlib`, and `tabulate` are available.

```bash
python3 -c "import pandas, numpy, matplotlib, pyarrow, tabulate"
```

---

## 3. Workload Provenance & Datasets

Our work utilizes production-derived traces from real-world deployments:
1. **Microsoft Azure LLM serving traces (2023):** Contains `azure_2023_code` and `azure_2023_conversation` workloads.
   - Upstream URL: `https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md`
   - Redistribution status: not redistributed; obtain the trace upstream.
2. **BurstGPT workload trace:** Wang et al., KDD 2025 (10.31 M requests from regional Azure OpenAI GPT services over 213 days).
   - Upstream URL (dataset owner): `https://github.com/HPMLL/BurstGPT` (CC-BY-4.0). This is the dataset owner's repository, not a fork.
   - Redistribution status: not redistributed; obtain the trace upstream.

Raw third-party traces and the derived workload windows built from them are
**not** committed. Only the corpus metadata (`manifest.json`, `schema.json`,
`distribution_stats.json`) is tracked under `data/public_trace_corpus_v1/`; the
parquet window files are built locally from the upstream traces (see
[`docs/DATA_RELEASE_POLICY.md`](../DATA_RELEASE_POLICY.md)). Checking the
manuscript's numbers, figures and tables does not need them: the committed
experiment artifacts are sufficient (Section 1 and
[`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md)). Re-running the simulations
does.

---

## 4. Scientific Freeze & Computation Map

The core scientific computations in this paper are **frozen**. Running the full suite of simulations is computationally expensive and requires a dedicated high-performance computing (HPC) environment.

| Result / Table / Figure | Script Name | Scope / Environment | Execution Type |
|---|---|---|---|
| **Phase A Replay** (0/99,992, etc.) | `scripts/industry_realism_action_opportunity_phase_a_v1.py` | Local or HPC | Frozen (Re-run optional) |
| **Phase B Pressure** | `scripts/industry_realism_action_opportunity_phase_b_v2.py` | Local or HPC | Frozen (Re-run optional) |
| **Fresh Causal Headroom** | `scripts/fresh_latency_causal_confirmatory_v1.py` | HPC (Wulver Cluster) | Frozen |
| **Manuscript Figures 1-2** (evidence chain, native/pressure prevalence; plus the dropped active-cap transition figure) | `paper/performance_evaluation/scripts/plot_performance_evaluation_figures.py` | Local | Instant from frozen CSVs (Phase A / Phase B) |
| **Manuscript Figure 3** (regime map; plus the dropped regime-characterization figure) | `paper/performance_evaluation/scripts/plot_regime_figures.py` | Local | Instant from frozen artifacts |
| **Manuscript Figure 4** (distribution and concentration) | `paper/performance_evaluation/scripts/plot_robustness_figures.py` | Local | Instant from frozen artifacts and robustness outputs |
| **Reference-policy, load-range, overlay and secondary-outcome numbers** (Sec. 3, 5, 6.1, 6.4, 7.5, 9) | `paper/performance_evaluation/scripts/reference_policy_numbers.py` (recomputed from completed artifacts; no simulation) | Local | Instant; checked by `build_claim_manifest.py --check` and `tests/test_manuscript_scientific_corrections.py` |
| **Table 1** (Closest-work) | (Static synthesis) | N/A | Analytical mapping |
| **Table 2** (Transition) | (Static synthesis) | N/A | Analytical mapping |
| **Tables 3-5** (regime characterization, thresholds, sensitivity) | `paper/performance_evaluation/scripts/robustness_numbers.py` | Local | Recomputed from frozen CSVs and asserted against the robustness outputs |

---

## 5. Bootstrap Clustered Correction Provenance

Our fresh causal headroom points are evaluated using a pre-specified bootstrap clustered analysis to estimate the 95% confidence interval of the latency headroom accurately.
- **The bootstrap unit:** `(source_dataset, window_index)` representing 36 distinct source-window clusters.
- **Correction details:** Corrected an earlier implementation that incorrectly merged separate datasets sharing identical window indexes (26 clusters) into the pre-specified 36-cluster key. Point estimates, seeds, and the confirmatory POSITIVE verdict remained unaffected.
- **Detailed Audit:** See `experiments/fresh_production_latency_headroom_confirmatory_v1/BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md` for the step-by-step mathematical and code audit.

---

## 6. Running Tests

```bash
python3 -m pytest tests/test_manuscript_scientific_corrections.py tests/test_peva_prepackage_readiness.py -q   # this study's manuscript checks
python3 -m pytest -m "not gpu"                                     # whole suite without GPU tests
LLMSERVEOPT_RUN_GPU_TESTS=1 python3 -m pytest -m gpu               # GPU-only tests (CUDA GPU required)
```

Without the uncommitted public-trace-corpus parquet files and the staged
BurstGPT data, some historical test modules fail or error. Those are data
availability failures, not code defects; CI excludes them (see
`.github/workflows/ci.yml`).

---

## 7. Bounded Real-vLLM Correspondence Probe

The vLLM probe checks whether resource pressure changes queueing and scheduling
behavior in a real engine. It is a bounded correspondence probe: it does not
validate the simulated latency magnitudes, and its direct full-versus-chunked
comparison did not reproduce the simulator's predicted class reversal.
- Re-running the probe requires a dedicated GPU and a separate vLLM environment; the manuscript's probe used one local RTX 5060 Ti.
- The manuscript's vLLM numbers come from `experiments/real_vllm_mechanism_validation_v1/native_vllm_chunk_budget_semantics_probe_v1/` (`mechanism_summary.json`, `statistical_summary.json`) and `experiments/real_vllm_pressure_action_validation_v1/REAL_VLLM_VALIDATION_RESULT_V1.json`; each is checked by `paper/performance_evaluation/scripts/build_claim_manifest.py`.
- No monetary API calls are performed; mock modes are used by default unless `--allow-live-api` is supplied with correct credentials.

---

## 8. Known documentation corrections

- The BurstGPT upstream URL above was corrected on 2026-09-21 (it previously named an author fork).
- `docs/current/FRESH_PRODUCTION_SUPPORT_MAPPING_REPORT_V1.md` (a stage record, preserved unchanged) wrongly says BurstGPT `kv_8000` was
  invalid; see `docs/current/FRESH_PRODUCTION_SUPPORT_MAPPING_REPORT_V1_ERRATUM_20260921.md`. The CSV governs.
- The frozen state-level file `FRESH_LATENCY_STATE_LEVEL_V1.csv` has mislabeled `mean_ref_latency`/`p95_ref_latency` columns; use
  `experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/` for the SBS-reference latency.
- **Terminology: "pre-specified", not "preregistered".** The confirmatory protocol (`experiments/fresh_production_latency_headroom_confirmatory_v1/PREREGISTRATION_V1.json`,
  `METRIC_PROTOCOL_V1.json`, `CAUSAL_PROTOCOL_V1.json`, `SUPPORT_PROTOCOL_V1.json`) was committed to the project's version-controlled repository
  (commit `b4e6c60`, 2026-09-19 22:12 EDT) and the results were committed at `b196c3e` (2026-09-20 00:26 EDT), 2 h 14 min later. The protocol was not
  deposited in an external or public registry, the author controlled the repository, and the time a push reached the remote could not be verified
  independently. The manuscript therefore says "pre-specified"; the file names keep the historical word "PREREGISTRATION". One post-outcome correction is
  disclosed in the manuscript: the first bootstrap clustered on a bare `window_index` (26 clusters) and was corrected to the source window (36 clusters);
  see `experiments/fresh_production_latency_headroom_confirmatory_v1/BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md`.
