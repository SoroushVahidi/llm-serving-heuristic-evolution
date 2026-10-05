# Reproducibility Entry Point

This page covers the current study, **"When Does LLM-Serving Scheduler
Adaptation Matter? Action Opportunity and Causal Headroom in Production-Derived
Replay"** (manuscript:
[`paper/when_does_llm_serving_scheduler_adaptation_matter.pdf`](paper/when_does_llm_serving_scheduler_adaptation_matter.pdf)).
Instructions for earlier manuscripts are linked [at the end](#historical-manuscripts).

## What can be reproduced, and how

| Level | What it checks | Needs |
|---|---|---|
| 1. Claim check | Recomputes every quantitative claim in the manuscript from committed artifacts and compares it with `main.tex` | Python only, seconds |
| 2. Figures and tables | Regenerates the manuscript figures and robustness tables from frozen artifacts | Python and matplotlib, seconds |
| 3. Archive verification | Checks checksums, frozen-artifact and shard hashes, claims, figures, and correction replay for the v1.1.0 archive | Python, minutes |
| 4. Manuscript rebuild | Rebuilds the PDF from LaTeX source | `pdflatex`, `bibtex`, `elsarticle` |
| 5. Re-simulation | Re-runs native replay, the pressure map, and the fresh causal execution | Raw upstream traces; HPC allocation for the fresh causal run |

Levels 1–4 need no GPU, no raw traces, and no network beyond package
installation.

## Setup

```bash
python3 -m pip install -e ".[dev]"
python3 -c "import pandas, numpy, matplotlib, pyarrow, tabulate"
```

## Level 1: claim check

```bash
python3 paper/performance_evaluation/scripts/build_claim_manifest.py --check
```

[`paper/performance_evaluation/FINAL_CLAIM_MANIFEST.json`](paper/performance_evaluation/FINAL_CLAIM_MANIFEST.json)
maps each claim to its source artifact, source field, and manuscript location.
The headline numbers come from:

| Claim | Artifact |
|---|---|
| 0 disagreement states in 1,002,438 native decision states | [`experiments/industry_realism_action_opportunity_phase_a_v1/PHASE_A_WORKLOAD_SUMMARY_V1.csv`](experiments/industry_realism_action_opportunity_phase_a_v1/PHASE_A_WORKLOAD_SUMMARY_V1.csv) |
| Pressure map, arrival scaling action-null | [`experiments/industry_realism_action_opportunity_phase_b_v2/`](experiments/industry_realism_action_opportunity_phase_b_v2/) |
| 720 states, 590 beneficial (81.9%), mean headroom 1.9958 simulated ms | [`experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_LATENCY_CAUSAL_RESULT_V1.json`](experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_LATENCY_CAUSAL_RESULT_V1.json) |
| 95% CI [0.1909, 3.5462] ms, 36 window clusters | [`experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_LATENCY_BOOTSTRAP_V1.json`](experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_LATENCY_BOOTSTRAP_V1.json) |
| Median 0.197 ms, concentration, sensitivity | [`experiments/fresh_production_latency_headroom_confirmatory_v1_robustness/`](experiments/fresh_production_latency_headroom_confirmatory_v1_robustness/) |
| Bounded vLLM probe | [`experiments/real_vllm_pressure_action_validation_v1/REAL_VLLM_VALIDATION_RESULT_V1.json`](experiments/real_vllm_pressure_action_validation_v1/REAL_VLLM_VALIDATION_RESULT_V1.json) |
| Reference-reserve sensitivity (post hoc, repository only) | [`experiments/reference_reserve_sensitivity_v1/`](experiments/reference_reserve_sensitivity_v1/) |

## Level 2: figures and tables

```bash
python3 paper/performance_evaluation/scripts/plot_performance_evaluation_figures.py   # Figures 1-2
python3 paper/performance_evaluation/scripts/plot_regime_figures.py                   # Figure 3
python3 paper/performance_evaluation/scripts/plot_robustness_figures.py               # Figure 4
python3 paper/performance_evaluation/scripts/robustness_numbers.py                    # Tables 3-5
```

The figure scripts overwrite the files in `paper/performance_evaluation/figures/`;
use `git diff` to compare. [`paper/performance_evaluation/README.md`](paper/performance_evaluation/README.md)
lists which input each script reads.

## Level 3: archive verification

The v1.1.0 archive is published on Zenodo at
[10.5281/zenodo.22866983](https://doi.org/10.5281/zenodo.22866983). The same
ZIP is committed as
[`release/performance_evaluation_v1_1_0.zip`](release/performance_evaluation_v1_1_0.zip).
Verify it from an unpacked copy of the ZIP:

```bash
unzip release/performance_evaluation_v1_1_0.zip -d /tmp/peva
cd /tmp/peva/performance_evaluation_v1_1_0
python3 verify_release.py
```

Do not run the verifier in the unpacked tree under
[`release/performance_evaluation_v1_1_0/`](release/performance_evaluation_v1_1_0/).
The repository's `*.jsonl` ignore rule left one archive file
(`experiments/public_trace_replay_v1/layer3_checkpoint.jsonl`) out of that tree,
so the checksum and claim checks fail there. The ZIP contains the file.

The archive's [`README.md`](release/performance_evaluation_v1_1_0/README.md)
describes each check. The archive's build environment is recorded in `ENVIRONMENT.json`.
A numpy or pandas version different from the one recorded there can change a
recorded version string without changing any number. The reference-reserve
sensitivity analysis came after this archive and is not part of it.

## Level 4: manuscript rebuild

```bash
cd paper/performance_evaluation
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`scripts/build_performance_evaluation_manuscript.sh` runs the figure scripts,
checks the claim manifest, and rebuilds the PDF. It replaces the canonical PDF
only when `UPDATE_CANONICAL_PDF=1` is set.

## Level 5: re-simulation

Runners: [`scripts/industry_realism_action_opportunity_phase_a_v1.py`](scripts/industry_realism_action_opportunity_phase_a_v1.py)
(native replay), [`scripts/industry_realism_action_opportunity_phase_b_v2.py`](scripts/industry_realism_action_opportunity_phase_b_v2.py)
(pressure map), and [`scripts/fresh_latency_causal_confirmatory_v1.py`](scripts/fresh_latency_causal_confirmatory_v1.py)
(fresh causal execution, originally run on an HPC cluster). The simulator
configuration of this study (one simulated GPU, fixed 1 ms step, no
hardware-calibrated service curves) is defined in
[`src/llmserveopt/policy_separation/public_trace_replay_v1.py`](src/llmserveopt/policy_separation/public_trace_replay_v1.py).
Simulated latencies are not hardware measurements.

Raw third-party traces are **not redistributed**. Obtain them upstream:

- Azure LLM inference traces 2023 (code, conversation):
  <https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md>
- BurstGPT (CC-BY-4.0): <https://github.com/HPMLL/BurstGPT>

See [`docs/DATA_RELEASE_POLICY.md`](docs/DATA_RELEASE_POLICY.md). The bounded
vLLM probe ran on one local GPU in a separate environment, and the core
environment does not reproduce it.

## Documented corrections

- **"Pre-specified", not "preregistered."** The fresh causal protocol was
  committed to this author-controlled repository before the results, not
  deposited in an external registry. The frozen file names
  (`PREREGISTRATION_V1.json` and others) keep the historical word.
- **Bootstrap cluster key.** The first bootstrap clustered on a bare window
  index (26 clusters). It was corrected to the pre-specified source window
  (36 clusters). See
  [`BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md`](experiments/fresh_production_latency_headroom_confirmatory_v1/BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md).
- **Mislabeled descriptive columns.** In the frozen
  `FRESH_LATENCY_STATE_LEVEL_V1.csv`, the columns `mean_ref_latency` and
  `p95_ref_latency` are mislabeled. The file is preserved unchanged. Use
  [`experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/`](experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/)
  for the reference latency.

The detailed guide for this study is
[`docs/current/PERFORMANCE_EVALUATION_REPRODUCIBILITY.md`](docs/current/PERFORMANCE_EVALUATION_REPRODUCIBILITY.md).

## Historical manuscripts

These are kept for provenance and are superseded:

- FGCS-oriented package: [`paper/history/fgcs/`](paper/history/fgcs/), with the
  reproducibility matrix in
  [`docs/current/FGCS_REPRODUCIBILITY_MATRIX_V1.md`](docs/current/FGCS_REPRODUCIBILITY_MATRIX_V1.md).
- LLM 2026 conference manuscript (withdrawn before publication):
  [`paper/history/llm2026/`](paper/history/llm2026/).
