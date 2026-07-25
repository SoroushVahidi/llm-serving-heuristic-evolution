# Composition Implementation Status

> **Pause addendum 2026-07-25.** The repaired load-discrimination pilot (`PARTIALLY_READY`) does **not** justify reopening composition or synthesis work. Native composition pilot remains `NO_GO` with verified-readable artifacts; structural synthesis remains empirically `NOT_READY`. Prioritize simulator/load discrimination on natural evidence first.


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

## Session Addendum (2026-07-24): score aggregation, reciprocal rank, instrumentation

Added without changing any of the status above (which describes the
`ff29222`/`24559f2`/`4604207` implementation as of 2026-07-21):

- `src/llmserveopt/policies/capabilities.py` — typed `PolicyCapabilities`
  audit (`RANK_CAPABLE_EXPERTS`, `SCORE_CAPABLE_EXPERTS`,
  `ADMISSION_CAPABLE_EXPERTS`, DSL mapping status) so composition code can
  validate "does this expert support scores/ranks/admission?" with a clear
  `CapabilityError` instead of degrading silently.
- `weighted_reciprocal_rank_aggregate()` in `composition.py`, plus a
  `method="borda"|"reciprocal_rank"` parameter on `StaticRankEnsemblePolicy`
  (default unchanged: `"borda"`). `weighted_borda_aggregate()` was also
  factored out of `StaticRankEnsemblePolicy` into a standalone pure function
  with identical behavior (verified against the pre-existing 16/16
  `test_policy_composition.py` tests, which still pass unmodified).
- `src/llmserveopt/policies/score_aggregation.py` — `NormalizationMode`
  (`none`, `min_max`, `zscore`, `robust_mad`), `normalize_scores()`,
  `score_with_named_expert()` for the 8 score-capable policies,
  `weighted_score_aggregate()`, and `StaticScoreEnsemblePolicy` (sparse
  top-k, deterministic fallback, same `CompositionDecisionLog` logging as
  the rank ensemble).
- `src/llmserveopt/policies/instrumentation.py` — `DecisionTraceRecordV1`
  (schema-versioned `DecisionTraceV1`), `DecisionTraceSink` (disabled by
  default; `record()` is a single boolean check when disabled), and
  `InstrumentedPolicy` (wraps any `BasePolicy`, forwards its `Action`
  unmodified, optionally captures a decision trace). Proven not to alter
  outcomes and to add zero records when disabled in
  `tests/test_score_and_reciprocal_rank_composition.py`.
- `tools/composition_score_rank_smoke.py` — small fixed-seed correctness
  smoke (not a performance claim) exercising both new operator families
  against native parents on a tiny contended scenario.
- `docs/current/wolverine_oracle_mixture_spec.json` and
  `WOLVERINE_ORACLE_MIXTURE_HANDOFF.md` — handoff spec for the future
  oracle-mixture sweep (not submitted).

`STATIC_ENSEMBLE_METHODS = borda, reciprocal_rank`

`SCORE_AGGREGATION = IMPLEMENTED (none, min_max, zscore, robust_mad)`

`DECISION_INSTRUMENTATION = IMPLEMENTED (disabled by default, DecisionTraceV1)`

`TESTS_RUN (this addendum) = tests/test_score_and_reciprocal_rank_composition.py`

`TESTS_PASSED (this addendum) = 38/38`

`FULL_TEST_SUITE (this addendum) = 2801 passed, 2 skipped, 3 deselected (gpu marker)`

`PERFORMANCE_VALIDATION (this addendum) = NOT_DONE` — reciprocal-rank and score
aggregation lack large-scale performance validation; the smoke is
correctness-only. No Wolverine oracle-mixture sweep has been launched.

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
