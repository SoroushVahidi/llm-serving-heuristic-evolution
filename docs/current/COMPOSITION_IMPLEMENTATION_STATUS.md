# Composition Implementation Status

> Historical implementation snapshot. The harness remains implemented, but the
> scientific gate has changed: completed composition/module/simulator evidence
> now says broad composition should wait for simulator calibration and improved
> reward separation. See `PROJECT_STATUS.md` and
> `COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md`.

## Summary

COMPOSITION_HARNESS_STATUS = READY

The repository has an experimental, non-production composition harness. Focused
tests and the correctness smoke pass through SLURM. The full decisive experiment
is blocked by scientific signal quality, not by missing upstream final reports.

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

UPSTREAM_FRONTIER_STATUS = COMPLETE

UPSTREAM_POLICY_LIBRARY_STATUS = COMPLETE

FULL_COMPOSITION_EXPERIMENT_SUBMITTED = NO

EXPECTED_FULL_EXPERIMENT_COMMAND = `sbatch tools/composition_experiment_when_ready.sbatch`

## Known Limitations

- The contextual weighting model is a deterministic placeholder until
  simulator-calibrated development evidence is available for training.
- The component-wise prototype only composes semantics currently supported by `ObservableState` and `Action(admit=...)`.
- Full 27-policy quantitative complementarity should be refreshed after
  simulator calibration if policy separation improves.
- No large composition experiment has been launched.

## Readiness Gate

Run a full experiment only after:

1. simulator calibration and pressure validation pass;
2. bounded re-evaluation shows improved, scientifically defensible policy separation;
3. composition tests pass;
4. expert selection config is frozen from development-only data;
5. held-out/OOD labels remain untouched during treatment selection.
