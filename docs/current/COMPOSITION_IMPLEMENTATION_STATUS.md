# Composition Implementation Status

## Summary

COMPOSITION_HARNESS_STATUS = READY

The repository now has an experimental, non-production composition harness. Focused tests and the correctness smoke pass through SLURM. The full decisive experiment remains blocked by design until the running Policy Frontier Cartography and Policy Library v2 workflows finish and write final reports.

## Implemented

- `src/llmserveopt/policies/composition.py`
  - typed composition module metadata;
  - normalized rank adapters;
  - static rank ensemble;
  - contextual rank ensemble pipeline;
  - component-wise SCORPIO/WSP prototype;
  - contribution, entropy, switching, fallback, and invalid-composition logging.
- `src/llmserveopt/selector/composition_experiment.py`
  - upstream final-report readiness checks;
  - policy-vector CSV loading;
  - split-group leakage check;
  - development-only best-fixed selection;
  - held-out split guard for treatment selection.
- `tests/test_policy_composition.py`
  - composition-specific unit and integration tests.
- `tools/composition_smoke_experiment.py`
  - lightweight correctness smoke using completed data and tiny observable states.
- `tools/composition_experiment_when_ready.sbatch`
  - unsubmitted SLURM template that refuses to run until upstream reports exist.

## Status Fields

TYPED_MODULE_INTERFACE = IMPLEMENTED

RANK_AGGREGATION = IMPLEMENTED

STATIC_ENSEMBLE = IMPLEMENTED

CONTEXTUAL_ENSEMBLE_PIPELINE = READY

COMPONENT_WISE_PROTOTYPE = IMPLEMENTED

SAFETY_GUARDS = feasibility projection; deterministic fallback policy; sparse top-k expert support; weight entropy logging; switching-frequency logging; optional minimum commitment/hysteresis; invalid composition detection; per-expert contribution logging; held-out split guard; split-group leakage guard

TESTS_RUN = `sbatch tools/composition_harness_tests.sbatch` -> job `1119434`

TESTS_PASSED = 37/37 pytest tests plus composition smoke PASS

UPSTREAM_FRONTIER_STATUS = RUNNING_OR_NOT_FINAL_REPORT_READY

UPSTREAM_POLICY_LIBRARY_STATUS = RUNNING_OR_NOT_FINAL_REPORT_READY

FULL_COMPOSITION_EXPERIMENT_SUBMITTED = NO

EXPECTED_FULL_EXPERIMENT_COMMAND = `sbatch tools/composition_experiment_when_ready.sbatch`

## Known Limitations

- The contextual weighting model is a deterministic placeholder until upstream frontier/library outputs are available for development-only training.
- The component-wise prototype only composes semantics currently supported by `ObservableState` and `Action(admit=...)`.
- Full 27-policy quantitative complementarity is deferred until Policy Library v2 finishes.
- No large composition experiment has been launched.

## Readiness Gate

Run the full experiment only after:

1. `policy_frontier_cartography_20260721T154408Z/reports/FINAL_REPORT.md` exists.
2. `policy_library_v2_expanded_20260721T171933Z/reports/FINAL_POLICY_LIBRARY_REPORT.md` or `reports/FINAL_REPORT.md` exists.
3. Composition tests pass.
4. Expert selection config is frozen from development-only data.
