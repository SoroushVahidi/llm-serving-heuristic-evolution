# Performance Evaluation manuscript

*When Does LLM-Serving Scheduler Adaptation Matter? Action Opportunity and Causal Headroom in Production-Derived Replay*

Target journal: *Performance Evaluation* (Elsevier). The class is `elsarticle` (`preprint`, 12 pt) with the
numbered `elsarticle-num` bibliography style.

## Canonical source

| Path | Content |
|---|---|
| `main.tex` | Manuscript source (canonical). |
| `references.bib` | Bibliography; `main.bbl` is the compiled bibliography that is tracked with the source. |
| `figures/` | Figure files (vector PDF plus PNG preview). Figures 1-3 come from `scripts/plot_performance_evaluation_figures.py`, Figures 4-5 from `scripts/plot_regime_figures.py`, Figure 6 from `scripts/plot_robustness_figures.py`. |
| `scripts/robustness_numbers.py` | Recomputes every number used in Table 3, Tables 4-5 and Figures 4-6 from the frozen artifacts and asserts agreement with the robustness outputs. |
| `../when_does_llm_serving_scheduler_adaptation_matter.pdf` | Compiled review copy. |

Scientific inputs are read-only: `experiments/fresh_production_latency_headroom_confirmatory_v1/` (frozen
confirmatory artifacts), `..._robustness/` (post hoc robustness outputs) and `..._corrected/` (corrected
derivative of two descriptive state-level columns; see `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`).

## Build

From the repository root, regenerate figures from the frozen artifacts and rebuild the PDF:

```bash
scripts/build_performance_evaluation_manuscript.sh
```

or, for the manuscript only:

```bash
cd paper/performance_evaluation
pdflatex -interaction=nonstopmode main.tex && bibtex main
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

No new experiments are run by the build.

## Checks

```bash
python -m pytest tests/test_manuscript_front_matter.py tests/test_manuscript_robustness_numbers.py \
  tests/test_manuscript_discussion_conclusion.py tests/test_manuscript_presentation.py
```

These guard journal limits (abstract length, keyword count), the numbers quoted in the abstract, Results and
Discussion against the frozen artifacts, table and figure structure, terminology, and reference metadata.

## Related documentation

* `docs/FRESH_CAUSAL_ROBUSTNESS_REPORT.md`: post hoc robustness analysis behind Section 7.
* `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`: artifact defect and corrected derivative.
* `docs/current/PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md`: journal requirements and submission status.

## Submission package

`submission/` holds an earlier packaging snapshot (source zip, highlights, cover letter, checklist). It has **not**
been regenerated from the current `main.tex` and should not be uploaded as is; it is rebuilt in the packaging step.
`scripts/plot_joint_complementarity.py`, `scripts/plot_vllm_semantic_validation.py` and the corresponding
`figures/*.pdf` are legacy files that the current manuscript does not use.
