# Contextual Composition Query 3 CC1 Spec Report - 2026-07-31

Branch: `contextual-compositional-heuristics-20260731`

Starting SHA: `09942cf3ba097f9408413b5e46dae679975c6337`

GitHub issue: [#1](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1)

## Repository State

Starting state was the expected branch and SHA. The upstream branch was
`origin/contextual-compositional-heuristics-20260731`, the worktree was clean,
`git fetch --dry-run` reported the branch up to date, and
`python scripts/check_contextual_composition_status.py` passed.

No reset, rebase, or overwrite operation was used.

## Prototypes Found

`src/llmserveopt/policies/composition.py`

- True simulator-executed composition: yes, when used through
  `StaticRankEnsemblePolicy`, `ContextualRankEnsemblePolicy`, or
  `ComponentWiseCompositionPolicy`.
- Stored reward-vector arithmetic: no.
- Causal information: rank experts use `ObservableState` and
  `ObservableRequest` causal fields; tests assert no `actual_output_tokens`.
- Reusable for CC1: yes, `StaticRankEnsemblePolicy(method="borda")` is the
  chosen implementation seed.
- Risks: rank-only normalization loses score magnitude, SCORPIO admission
  omissions are represented as missing ranks, and component-wise composition is
  broader than the minimal CC1 question.

`src/llmserveopt/policies/score_aggregation.py`

- True simulator-executed composition: yes, through
  `StaticScoreEnsemblePolicy`.
- Stored reward-vector arithmetic: no.
- Causal information: yes for registered score-capable experts.
- Reusable for CC1: not chosen for the minimal experiment.
- Risks: only score-capable experts can participate; raw-score semantics remain
  narrower and easier to misuse than rank aggregation.

`src/llmserveopt/policies/capabilities.py`

- True simulator-executed composition: no, metadata only.
- Stored reward-vector arithmetic: no.
- Causal information: no runtime decision path.
- Reusable for CC1: yes for validating the representative expert subset.
- Risks: declarative table can drift from implementation if not asserted.

`src/llmserveopt/policies/instrumentation.py`

- True simulator-executed composition: no, wrapper/instrumentation only.
- Stored reward-vector arithmetic: no.
- Causal information: records current state summaries only.
- Reusable for CC1: yes for optional decision traces.
- Risks: trace volume can grow quickly; keep disabled by default.

`src/llmserveopt/selector/composition_experiment.py`

- True simulator-executed composition: no.
- Stored reward-vector arithmetic: reads policy-vector CSVs and selects
  fixed-policy baselines from stored metrics.
- Causal information: split guards only; no scheduler decision path.
- Reusable for CC1: yes for leakage guards and development-only best fixed
  selection.
- Risks: cannot prove mixture behavior.

`scripts/run_composition_smart_pilot.py`

- True simulator-executed composition: no.
- Stored reward-vector arithmetic: yes for static/contextual mixture proxies.
- Causal information: selectors use `feat_*` columns for prediction, but
  mixture performance is arithmetic over stored policy metrics.
- Reusable for CC1: only as cautionary prior art for reporting fields and
  selector baselines.
- Risks: invalid as composition-opportunity evidence.

`src/llmserveopt/selector/advanced.py`

- True simulator-executed composition: no.
- Stored reward-vector arithmetic: trains/predicts per-policy ANWG labels; hard
  selector outputs one policy.
- Causal information: `validate_feature_columns` rejects leaky feature names.
- Reusable for CC1: yes for learned hard-selector baseline when local data is
  available.
- Risks: per-policy regressors are selector baselines, not mixtures.

`src/llmserveopt/selector/labels.py`

- True simulator-executed composition: no.
- Stored reward-vector arithmetic: stores per-policy reward vectors and best
  fixed/oracle fixed arithmetic.
- Causal information: label construction uses realized metrics and is offline
  only.
- Reusable for CC1: yes for baseline definitions only.
- Risks: legacy `weighted_goodput` label path is not the CC1 primary objective.

`src/llmserveopt/policies/genome.py`

- True simulator-executed composition: no direct rank-mixture execution; it
  compiles typed genomes to verified heuristic policies.
- Stored reward-vector arithmetic: no.
- Causal information: validates DSL variables through allowed causal fields.
- Reusable for CC1: not needed for the minimal experiment.
- Risks: adding DSL/genome composition before CC1 would broaden scope.

`src/llmserveopt/selector/module_credit/*`

- True simulator-executed composition: no.
- Stored reward-vector arithmetic: learns/evaluates module-credit targets from
  offline rows.
- Causal information: depends on encoded rows and targets; not a scheduler
  execution path.
- Reusable for CC1: no, except as evidence that broad structural recombination
  is premature.
- Risks: weak learning signal and no direct action execution.

Historical docs under `docs/current/`

- `POLICY_COMPOSITION_READINESS.md`,
  `COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md`,
  `COMPOSITION_EXPERIMENT_DESIGN.md`,
  `COMPOSITION_IMPLEMENTATION_STATUS.md`, and
  `wolverine_oracle_mixture_spec.json` document earlier harnesses and
  constraints.
- They support the choice of rank aggregation but do not replace a fresh CC1
  simulator-executed measurement.

## Reusable Components

The Query 4 implementation should reuse:

- `StaticRankEnsemblePolicy(method="borda")`;
- `RankExpertSpec`;
- `rank_with_named_expert`;
- `InstrumentedPolicy` and `DecisionTraceSink` for optional traces;
- `select_best_fixed_policy_from_development`;
- `assert_no_split_group_leakage`;
- `validate_treatment_selection_does_not_use_heldout`;
- existing simulator `run_policy` paths and metrics.

## Rejected Or Insufficient Approaches

Reward-vector interpolation is rejected for CC1 because it does not execute
composed actions and cannot model admission, queue, batching, or placement
interactions.

The smart pilot's static/contextual proxy rows are insufficient for CC1 because
they compute weighted sums of stored ANWG/completion/quality columns.

Raw score aggregation is scientifically valid only for the score-capable subset
and excludes rank/admission-only SCORPIO behavior, so it is not the minimal
representative CC1 interface.

Component-wise composition is deferred because it combines several semantics at
once and would make a negative/positive result harder to interpret.

## Final CC1 Design

The approved Query 4 target is a minimal weighted Borda rank-aggregation
experiment over:

- `weighted_shortest_processing`;
- `scorpio_style_slo_guard`;
- `edf`;
- `estimated_service_time_first`;
- `fifo`.

The experiment must execute composed policies through the simulator and use
`arrival_normalized_weighted_goodput` as the primary metric. The primary
composition-opportunity gap is:

```text
mean_ANWG(oracle_best_mixture_per_window)
- mean_ANWG(oracle_best_fixed_policy_per_window)
```

Success requires aggregate non-near-tie gain at least `0.005`, or
regime-specific gain at least `0.01`, without completion-fraction loss greater
than `0.005` or any safety violation.

Stop applies if the non-near-tie gap is less than `0.002`, if oracle per-window
mixture does not beat oracle per-window fixed policy after completion
constraints, or if gains rely on near-tie noise or reward-vector interpolation.

## Files Planned For Query 4

Create:

- `configs/cc1_composition_opportunity_smoke.yaml`
- `configs/cc1_composition_opportunity.yaml`
- `scripts/run_cc1_composition_opportunity.py`
- `src/llmserveopt/experiments/cc1_composition_opportunity.py`
- `tests/test_cc1_composition_opportunity.py`

Modify after results:

- `docs/contextual_composition_roadmap.md`
- `docs/contextual_composition_decisions.md` only if needed
- `docs/START_HERE_CONTEXTUAL_COMPOSITION.md`
- GitHub issue #1
- `scripts/check_contextual_composition_status.py` only if required evidence
  changes

## Validation Results

Commands run:

- `python scripts/check_contextual_composition_status.py`: passed.
- `python -m pytest tests/test_contextual_composition_status_checker.py -q`:
  3 passed.
- Markdown local-link check over `docs/**/*.md`: passed.
- YAML parsing over `configs/**/*.yaml` and `configs/**/*.yml`: 65 files
  parsed.
- `python -m compileall -q src scripts tests`: passed.
- `python -m pytest --collect-only -q`: 2928 tests collected.
- `python -m pytest tests/test_policy_composition.py tests/test_score_and_reciprocal_rank_composition.py tests/test_contextual_composition_status_checker.py -q`:
  57 passed.
- `python -m pytest tests/test_module_credit.py tests/test_policy_genome_coverage.py tests/test_structural_synthesis.py -q`:
  147 passed.

Note: bare `pytest` used a different interpreter and initially failed
collection on missing `pandas`. The validation above used
`/home/soroush/modal-venv/bin/python`, where the repository's declared
dependencies are available.

## Unresolved Risks

- Full local CC1 mode may require reconstructing representative workload
  windows from existing local fixtures if ignored historical policy-vector CSVs
  are absent in a fresh clone.
- The learned hard-selector baseline may be unavailable unless a local artifact
  or trainable dataset is present.
- Rank aggregation may hide useful score magnitude; this is accepted for CC1
  because rank composition is the safest common interface.
- SCORPIO's admission filtering inside rank extraction omits infeasible/unsafe
  candidates for that expert; missing-rank semantics must be clearly logged.
