# Current Study Documentation

This index covers the current study, **"When Does LLM-Serving Scheduler
Adaptation Matter? Action Opportunity and Causal Headroom in Production-Derived
Replay."** Most other files in `docs/current/` are dated records from earlier
research lines. They are kept for provenance; see
[Historical research](#historical-research).

## Read first

1. [`README.md`](../../README.md): public overview, main results, and scope.
2. Manuscript PDF:
   [`paper/when_does_llm_serving_scheduler_adaptation_matter.pdf`](../../paper/when_does_llm_serving_scheduler_adaptation_matter.pdf).
   Source and figure scripts are in
   [`paper/performance_evaluation/`](../../paper/performance_evaluation/README.md).
3. [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md): what can be reproduced and
   how. The detailed guide is
   [`PERFORMANCE_EVALUATION_REPRODUCIBILITY.md`](PERFORMANCE_EVALUATION_REPRODUCIBILITY.md).

## Core evidence

| Evidence | Report | Artifacts |
|---|---|---|
| Native replay: no executable disagreement under abundant resources | [`PHASE_A_NATIVE_REPLAY_REPORT_V1.md`](../../experiments/industry_realism_action_opportunity_phase_a_v1/PHASE_A_NATIVE_REPLAY_REPORT_V1.md) | [`phase_a_v1/`](../../experiments/industry_realism_action_opportunity_phase_a_v1/) |
| Resource pressure creates disagreement; arrival scaling up to 8× does not | [`PHASE_B_V2_PRESSURE_REPORT.md`](../../experiments/industry_realism_action_opportunity_phase_b_v2/PHASE_B_V2_PRESSURE_REPORT.md) | [`phase_b_v2/`](../../experiments/industry_realism_action_opportunity_phase_b_v2/) |
| Fresh-window support mapping | [`FRESH_PRODUCTION_SUPPORT_MAPPING_REPORT_V1.md`](FRESH_PRODUCTION_SUPPORT_MAPPING_REPORT_V1.md) and its [erratum](FRESH_PRODUCTION_SUPPORT_MAPPING_REPORT_V1_ERRATUM_20260921.md) | `FRESH_SUPPORT_*` files in the confirmatory directory below |
| Pre-specified fresh causal latency headroom (720 states) | [`FRESH_LATENCY_CONFIRMATORY_REPORT_V1.md`](../../experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_LATENCY_CONFIRMATORY_REPORT_V1.md); design: [`FRESH_LATENCY_HEADROOM_CONFIRMATORY_DESIGN_REPORT.md`](FRESH_LATENCY_HEADROOM_CONFIRMATORY_DESIGN_REPORT.md) | [`fresh_production_latency_headroom_confirmatory_v1/`](../../experiments/fresh_production_latency_headroom_confirmatory_v1/) (frozen) |
| Post hoc robustness, concentration, relative headroom | [`FRESH_CAUSAL_ROBUSTNESS_REPORT.md`](../FRESH_CAUSAL_ROBUSTNESS_REPORT.md) | [`..._v1_robustness/`](../../experiments/fresh_production_latency_headroom_confirmatory_v1_robustness/) |
| Corrected descriptive columns (frozen file unchanged) | [`FRESH_CAUSAL_ARTIFACT_CORRECTION.md`](../FRESH_CAUSAL_ARTIFACT_CORRECTION.md) | [`..._v1_corrected/`](../../experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/) |
| Reference-reserve sensitivity (post hoc; not in the v1.1.0 archive) | [`DENOMINATOR_COMPLETION_RESULT_V1.md`](../../experiments/reference_reserve_sensitivity_v1/DENOMINATOR_COMPLETION_RESULT_V1.md) | [`reference_reserve_sensitivity_v1/`](../../experiments/reference_reserve_sensitivity_v1/) |
| Bounded real-vLLM correspondence probe (the report's "preregistration" was written after the runs, as it states) | [`REAL_VLLM_PRESSURE_ACTION_VALIDATION_REPORT.md`](REAL_VLLM_PRESSURE_ACTION_VALIDATION_REPORT.md) | [`real_vllm_pressure_action_validation_v1/`](../../experiments/real_vllm_pressure_action_validation_v1/), [`real_vllm_mechanism_validation_v1/`](../../experiments/real_vllm_mechanism_validation_v1/) |

Every quantitative claim in the manuscript is mapped to its source artifact and
field in
[`FINAL_CLAIM_MANIFEST.json`](../../paper/performance_evaluation/FINAL_CLAIM_MANIFEST.json).

## Reproduction

```bash
python3 -m pip install -e ".[dev]"
python3 paper/performance_evaluation/scripts/build_claim_manifest.py --check
```

The v1.1.0 archive is on Zenodo at
[10.5281/zenodo.22866983](https://doi.org/10.5281/zenodo.22866983). It is also
committed as
[`release/performance_evaluation_v1_1_0.zip`](../../release/performance_evaluation_v1_1_0.zip)
and unpacked in
[`release/performance_evaluation_v1_1_0/`](../../release/performance_evaluation_v1_1_0/).
To verify it, run `python3 verify_release.py` from the archive root. Details are
in [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).

## Scope and limitations

- All headroom values are **simulator time**. The study uses one simulated GPU,
  a fixed 1 ms step, and no hardware-calibrated service curves.
- The real-vLLM work is a **bounded correspondence probe**: one GPU, one model,
  and one vLLM version. It supports the link between pressure and scheduling
  behavior. It does not validate the simulated latency magnitudes.
- The study measures oracle opportunity. It does not propose or evaluate a
  deployed adaptive controller.
- The protocol is *pre-specified* (committed before results in this
  repository), not externally *preregistered*. Frozen file names keep the
  historical word.

## Historical research

The following are retained for provenance and do **not** describe the current
study's status:

- Long-term research-program roadmap (as of 2026-08-19):
  [`docs/PROJECT_MAP.md`](../PROJECT_MAP.md).
- SBS-override selector line (ended 2026-09-19): [`RESUME_HERE.md`](RESUME_HERE.md),
  [`WORK_STATUS.md`](WORK_STATUS.md), [`NEXT_ACTIONS.md`](NEXT_ACTIONS.md),
  [`ACTIVE_JOBS.md`](ACTIVE_JOBS.md), and the hash-locked `SBS_OVERRIDE_*.md` files.
- Submission-time roadmap for this study (2026-09-20):
  [`PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md`](PERFORMANCE_EVALUATION_SUBMISSION_ROADMAP_20260920.md);
  manuscript work records `QUERY_*.md` and
  [`FINAL_MANUSCRIPT_FREEZE.md`](FINAL_MANUSCRIPT_FREEZE.md).
- Superseded FGCS manuscript: `FGCS_*.md` and
  [`paper/history/fgcs/`](../../paper/history/fgcs/).
- Withdrawn LLM 2026 manuscript: `llm2026_*.md` and
  [`paper/history/llm2026/`](../../paper/history/llm2026/).
- Earlier selector, composition, synthesis, family, joint-240 and
  external-baseline analyses: the dated `*_YYYYMMDD.md` files in this directory,
  [`docs/audits/`](../audits/), and
  [`docs/BASELINE_STATUS.md`](../BASELINE_STATUS.md).
