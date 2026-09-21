# Reproducibility package, release 1.1.0

When Does LLM-Serving Scheduler Adaptation Matter? Action Opportunity and Causal Headroom in Production-Derived Replay

* v1.0.0 (published record): [10.5281/zenodo.22865294](https://doi.org/10.5281/zenodo.22865294)
* v1.1.0 persistent DOI: assigned by Zenodo on publication of the v1.1.0 version of record 22865294 (concept DOI 10.5281/zenodo.22865293); this bundle is the exact upload payload
* Source repository: https://github.com/SoroushVahidi/llm-serving-heuristic-evolution
* Git tag: `performance-evaluation-v1.1.0`; commit `bba9269d7346d606243210dc574c7d7d59a37341`
* Author: Soroush Vahidi, New Jersey Institute of Technology
* License: MIT (see `LICENSE`). Values derived from third-party traces keep the upstream attribution terms below.

This archive is a snapshot of the repository at the tag above, restricted to what the paper needs. It mirrors the
repository layout, so every script runs from the archive root exactly as it does from a repository checkout.
`MANIFEST.json` lists every file with its size, SHA-256 and role; `SHA256SUMS.txt` is the same in `sha256sum` format.

## What changed since v1.0.0

v1.0.0 held the frozen confirmatory artifacts and an older figure script. This release adds the post hoc robustness
outputs, the corrected derivative, the continuation shards, the six-figure code, the claim manifest, the simulator
code, the manuscript source and the final review PDF. **No frozen artifact and no primary result changed**: the
files under the three frozen `experiments/` directories are byte-identical to v1.0.0.

## Layout

| Path | Content |
|---|---|
| `paper/performance_evaluation/` | Manuscript source (`main.tex`, `references.bib`, `main.bbl`), figures 1-6 (vector PDF + 300 dpi PNG), figure/table scripts, `FINAL_CLAIM_MANIFEST.json`. |
| `paper/when_does_llm_serving_scheduler_adaptation_matter.pdf` | Compiled review copy of the manuscript. |
| `experiments/industry_realism_action_opportunity_phase_a_v1/` | **Frozen.** Native replay (Phase A) with its preregistration. |
| `experiments/industry_realism_action_opportunity_phase_b_v2/` | **Frozen.** Resource-pressure map (Phase B v2). |
| `experiments/fresh_production_latency_headroom_confirmatory_v1/` | **Frozen.** Preregistered fresh causal latency-headroom study: protocols, preregistration, states, actions, result, bootstrap, artifact hashes. |
| `experiments/fresh_production_latency_headroom_confirmatory_v1_robustness/` | Post hoc robustness and concentration outputs (paper Section 7, Tables 3-5, Figure 6). |
| `experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/` | Corrected derivative of two descriptive state-level columns, audit, extracted SBS reference rows, provenance. |
| `experiments/real_vllm_*` | Summaries of the bounded vLLM probe (only the files the paper reads). |
| `provenance_archive/fgcs-finalization-20260920/fresh_latency/fresh-latency-execution-v1/fresh_latency_causal_run_v1/continuation_shards/` | The 289 hash-verified continuation-shard artifacts (96 shards plus their manifests) of the fresh causal run. |
| `scripts/`, `src/llmserveopt/` | Experiment, robustness and correction code and the simulator. |
| `tests/` | The manuscript, robustness and correction tests. |
| `docs/` | Correction and robustness reports, reproducibility guide. |
| `verify_release.py`, `ENVIRONMENT.json`, `PROVENANCE_MAP.md`, `SHARD_PUBLICATION_DECISION.md` | Self-check, build environment, claim-to-artifact map, shard decision. |

## Verify the archive

```bash
pip install numpy pandas matplotlib scipy pyarrow pyyaml pytest   # see pyproject.toml / requirements.txt
python3 verify_release.py            # checksums, frozen-artifact hashes, claim manifest, figures, correction replay, tests
python3 verify_release.py --latex    # additionally rebuild the manuscript PDF (needs pdflatex, bibtex, elsarticle)
```

The checks are: SHA-256 of every file against `MANIFEST.json`; the frozen files and all 289 shard artifacts against
`FRESH_LATENCY_ARTIFACT_HASHES_V1.json`; the 46 claims in `FINAL_CLAIM_MANIFEST.json` recomputed from the artifacts and
checked against `main.tex`; every number of Tables 3-5 recomputed by `robustness_numbers.py`; Figures 1-6 regenerated
into a temporary directory and compared with the shipped PNGs; the correction replayed from the shards and compared
byte-for-byte with the shipped corrected derivative; and the bundled tests.

Re-running the simulator (Phases A, B and the fresh causal execution) is not required to check the paper's numbers and
needs the raw upstream traces (below) and, for the fresh execution, an HPC allocation; the executor code is included.

## Artifact correction (disclosure)

The frozen `FRESH_LATENCY_STATE_LEVEL_V1.csv` is preserved unchanged. Two of its **descriptive** columns,
`mean_ref_latency` and `p95_ref_latency`, were populated from the first counterfactual branch of each state instead of
the SBS reference branch. **No reported scientific result depended on those columns**: the primary endpoint, the
opportunity counts, the bootstrap confidence interval and every table and figure are computed from other fields. A
corrected derivative (`..._corrected/FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv`) supplies the SBS reference values,
extracted from the shards (`SBS_REFERENCE_ROWS_V1.csv`). The primary endpoint and its confidence interval are unchanged.
Details: `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`.

## Third-party data

Raw traces are **not** redistributed. Obtain them from the upstream providers and follow their licence and attribution
terms: the Azure LLM inference trace 2023 (Azure Public Dataset, <https://github.com/Azure/AzurePublicDataset>) and
BurstGPT (Wang et al., KDD 2025, doi:10.1145/3711896.3737413). The artifacts here are derived scheduling outcomes and
window/state identifiers, not trace content. Please credit the trace authors when reusing them.

## How to cite

Cite this archive by its Zenodo persistent DOI (v1.0.0: `10.5281/zenodo.22865294`; the v1.1.0 version DOI is
assigned by Zenodo on publication and is recorded in `CITATION.cff`). The paper's reference list uses the same entry.

## Contents

664 files (excluding this README, `MANIFEST.json` and `SHA256SUMS.txt`).
