# Performance Evaluation manuscript

*When Does LLM-Serving Scheduler Adaptation Matter? Action Opportunity and Causal Headroom in Production-Derived Replay*

Target journal: *Performance Evaluation* (Elsevier). The class is `elsarticle` (`preprint`, 12 pt) with the
numbered `elsarticle-num` bibliography style.

Status: the manuscript text, figures, tables, claim manifest, and canonical PDF
candidate are frozen for integration. The Zenodo v1.1.0 archive, submission
package, and merge to `main` are still open.

## Canonical source

| Path | Content |
|---|---|
| `main.tex` | Manuscript source (canonical). |
| `references.bib` | Bibliography; `main.bbl` is the compiled bibliography that is tracked with the source. |
| `figures/` | Figure files as vector PDF (fonts embedded, drawn at final size) plus 300 dpi PNG preview. The manuscript uses `pe_pipeline`, `pe_disagreement_rates`, `pe_regime_map` and `pe_robustness` (its Figures 1-4); `pe_pressure_transition` and `pe_fresh_headroom` were dropped from the manuscript as redundant with Tables 2-3 and are kept only for the record. |
| `scripts/figstyle.py` | Shared typography, grayscale encoding, size and export rules of all figure files. |
| `scripts/plot_performance_evaluation_figures.py` | `pe_pipeline`, `pe_disagreement_rates` (manuscript Figures 1-2) and the dropped `pe_pressure_transition`; the data-driven figures read the frozen Phase A / Phase B CSVs, no value is typed in. |
| `scripts/plot_regime_figures.py` | `pe_regime_map` (manuscript Figure 3) and the dropped `pe_fresh_headroom`. |
| `scripts/plot_robustness_figures.py` | `pe_robustness` (manuscript Figure 4). |
| `scripts/robustness_numbers.py` | Recomputes every number used in the fresh-regime, threshold and sensitivity tables and in the regime-map and robustness figures from the frozen artifacts and asserts agreement with the robustness outputs. |
| `scripts/build_claim_manifest.py` | Recomputes the paper's quantitative claims from the canonical artifacts and checks them against `main.tex`. |
| `FINAL_CLAIM_MANIFEST.json` | Provenance/verification metadata: for each major claim its value, source artifact, source field or calculation, and manuscript location. It holds no independent scientific data. |
| `../when_does_llm_serving_scheduler_adaptation_matter.pdf` | Compiled review copy (written by the build script). |

## Required local artifacts

Everything the build reads is tracked in this repository, so no download is needed. All are read-only:

* `experiments/industry_realism_action_opportunity_phase_a_v1/` (native replay) and `..._phase_b_v2/` (pressure map);
* `experiments/fresh_production_latency_headroom_confirmatory_v1/` (frozen, pre-specified artifacts),
  `..._robustness/` (post hoc robustness outputs) and `..._corrected/` (corrected derivative of two descriptive
  state-level columns; see `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`);
* `experiments/real_vllm_mechanism_validation_v1/` and `experiments/real_vllm_pressure_action_validation_v1/`
  (vLLM probe summaries, used only by the claim manifest).

The build needs `pdflatex`, `bibtex` (with `elsarticle-num`), Python 3 with `matplotlib` and `numpy`. The tests
additionally use `pytest`, `pymupdf` and `pillow`.

## Build

From the repository root, regenerate the figures from the frozen artifacts, check the claim manifest, rebuild the
PDF and refresh the review copy:

```bash
scripts/build_performance_evaluation_manuscript.sh
```

Figures only:

```bash
python3 paper/performance_evaluation/scripts/plot_performance_evaluation_figures.py   # manuscript Figures 1-2 (+ dropped transition figure)
python3 paper/performance_evaluation/scripts/plot_regime_figures.py                   # manuscript Figure 3 (+ dropped fresh-headroom figure)
python3 paper/performance_evaluation/scripts/plot_robustness_figures.py               # manuscript Figure 4
```

Manuscript only:

```bash
cd paper/performance_evaluation
pdflatex -interaction=nonstopmode main.tex && bibtex main
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

The output is `paper/performance_evaluation/main.pdf` (30 pages), copied to
`paper/when_does_llm_serving_scheduler_adaptation_matter.pdf`. No experiment is run by the build.

## Checks

```bash
python3 -m pytest tests/test_manuscript_front_matter.py tests/test_manuscript_robustness_numbers.py \
  tests/test_manuscript_discussion_conclusion.py tests/test_manuscript_presentation.py \
  tests/test_manuscript_figures.py tests/test_manuscript_claim_manifest.py tests/test_peva_prepackage_readiness.py
python3 paper/performance_evaluation/scripts/build_claim_manifest.py --check
```

These guard journal limits (abstract length, keyword count), the numbers quoted in the abstract, Results and
Discussion against the frozen artifacts, table and figure structure (vector, embedded fonts, size, legibility,
grayscale), terminology, reference metadata, the generative-AI declaration, the Data Availability wording, and the
readiness checklist. After any edit of `main.tex` that changes a number, regenerate the manifest with
`python3 paper/performance_evaluation/scripts/build_claim_manifest.py`.

## Related documentation

* `docs/PEVA_PREPACKAGE_READINESS.md`: pre-package readiness checklist, deferred packaging items and the entries the
  archive step must add.
* `docs/FRESH_CAUSAL_ROBUSTNESS_REPORT.md`: post hoc robustness analysis behind Section 7.
* `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`: artifact defect and corrected derivative.
* `docs/current/PERFORMANCE_EVALUATION_REPRODUCIBILITY.md`: environment and reproduction guide.
* `docs/current/PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md`: journal requirements and submission status.

## Awaiting archive and submission packaging

The existing Zenodo record (DOI 10.5281/zenodo.22865294, tag `performance-evaluation-v1.0.0`) predates
the robustness outputs, the corrected derivative and the final figure code. The Data Availability paragraph is
worded to say so. After the new version is published, update the DOI, add the dataset/software reference and
regenerate the claim manifest; the exact list is in `docs/PEVA_PREPACKAGE_READINESS.md`.
TODO(QUERY_8): ZENODO_NEW_VERSION remains open until the archive is updated.

The stale `submission/` snapshot and v1.1.0 release package were removed from the
branch during manuscript freeze. They must be regenerated from the frozen source
in the packaging step.
`scripts/plot_joint_complementarity.py`, `scripts/plot_vllm_semantic_validation.py` and the corresponding
`figures/joint_complementarity.pdf` and `figures/vllm_semantic_validation.pdf` are legacy files that the current
manuscript does not use.
