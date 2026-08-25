# Repository Architecture Map

Current map for `wulver-selector-v2-and-composition-integrated`, combining
`origin/wulver-final-integration-20260721`'s Policy Library v2/composition/
structural-synthesis work with the Phase 2C selector-improvement and
leakage-fix-reconciliation work from `phase2c-final-selector-improvement`.
See [LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md) for how the two
lineages were combined and what was reconciled.

## Core Runtime

- `src/llmserveopt/core/` -- request/state/action/metric types.
- `src/llmserveopt/simulator/` -- simulator, constraints, service models,
  calibrated service-model factory.
- `src/llmserveopt/evaluation/` -- policy execution and metric aggregation.

## Policy Layer

- `src/llmserveopt/policies/base.py` -- policy interface and feasibility helpers.
- `src/llmserveopt/policies/registry.py` -- **single source of truth** for
  the policy inventory: 20 historical `BASELINE_NAMES` plus the 7-policy
  `_POLICY_LIBRARY_V2_REGISTRY` (`sola_style_state_aware`,
  `slai_style_phase_aware`, `flow_control_stability`,
  `kv_constrained_online`, `adaptive_chunked_prefill`, `aging_priority`,
  `weighted_fair_share`), exposed together as `POLICY_LIBRARY_V2_NAMES`
  (27 total).
- `src/llmserveopt/policies/policy_library_v2_helpers.py` -- shared causal
  helpers for the new monolithic policies.
- `src/llmserveopt/policies/*` -- deployable policy implementations.
- `src/llmserveopt/policies/composition.py` -- rank/contextual/component-wise
  composition harness (native Wulver implementation): weighted Borda
  (`weighted_borda_aggregate`) and weighted reciprocal-rank
  (`weighted_reciprocal_rank_aggregate`) aggregation, both selectable via
  `StaticRankEnsemblePolicy(method=...)`.
- `src/llmserveopt/policies/capabilities.py` -- typed
  `PolicyCapabilities`/`capabilities_for` audit of which experts support
  ranks, scores, or admission, used by both `composition.py` and
  `score_aggregation.py` to fail loudly (`CapabilityError`) instead of
  degrading silently.
- `src/llmserveopt/policies/score_aggregation.py` -- weighted normalized
  score aggregation (`none`/`min_max`/`zscore`/`robust_mad`) over the
  `capabilities.SCORE_CAPABLE_EXPERTS` subset that expose a genuine
  comparable scalar; sibling of the rank ensemble in `composition.py`.
- `src/llmserveopt/policies/instrumentation.py` -- optional
  `DecisionTraceSink`/`InstrumentedPolicy` decision-trace recording
  (`DecisionTraceV1` schema), zero-overhead when disabled (the default).
- `src/llmserveopt/policies/genome.py` -- typed `SchedulerGenomeV1`
  representation: canonical JSON serialization, deterministic SHA256 hash,
  semantic validation, causal feature-whitelist enforcement, conversion into
  the verified heuristic DSL. `SUPPORTED_MODULE_TYPES` is the canonical
  module taxonomy (`admission_rule`, `priority_rule`, `prefill_rule`,
  `kv_guard`, `fairness_rule`, `regime_conditions`).
- `src/llmserveopt/policies/structural_synthesis.py` -- structural
  child-generation operators (module swap, conditional regime composition,
  typed subtree crossover, bounded constant mutation, whitelisted
  feature/operator mutation, frontier-value scoring interface).

## Selector Layer

- `src/llmserveopt/selector/` -- selector v1 features/models/datasets.
- `src/llmserveopt/selector/dataset_v2/` -- Selector Dataset v2
  infrastructure. `splits.py` is the **single source of truth** for
  leakage-safe split grouping and verification (see below).
- `src/llmserveopt/selector/advanced.py` -- causal, `feat_*`-only selector
  formulations: per-policy reward regression (RF/Extra Trees/
  HistGradientBoosting), margin/regret-weighted classification
  (`policy_margin_weights`), pairwise policy ranking, prediction-margin
  uncertainty fallback, regime gating (`azure_conv_like_gate`).
  `validate_feature_columns` is the **single source of truth** for
  rejecting leaky (reward/completion/oracle/label) feature columns.
- `src/llmserveopt/selector/composition_experiment.py` -- composition
  experiment helpers (native Wulver implementation).
- `src/llmserveopt/selector/parent_selection.py` -- parent-pair scoring and
  the `SELECT_SINGLE` / `ATTEMPT_STRUCTURAL_COMPOSITION` composition gate
  for structural synthesis.

## Selector Dataset v2 Split-Leakage Architecture (reconciled)

`src/llmserveopt/selector/dataset_v2/splits.py`:

- `leakage_safe_split_group_key(row)` / `attach_leakage_safe_split_group_keys(rows)`
  -- groups real-trace windows by `f"{request_plan_ancestor_id}__pool_{time_slice_pool}"`
  (this exact string format is load-bearing: `assign_group_aware_split`
  hashes it with SHA256 to assign TRAIN/VALIDATION/ID_TEST/OOD_TEST, so the
  format must match what already-generated Wulver pilots used). Raises
  `KeyError` -- never silently falls back to the leaky transform-specific
  `group_key` -- if a real-trace row is missing `request_plan_ancestor_id`
  or `time_slice_pool`.
- `verify_no_cross_split_row_range_overlap(rows)` -- independent second
  guard: directly compares raw `time_slice_row_start`/`time_slice_row_end`
  ranges across splits, regardless of how grouping was done.
- `verify_group_atomicity`, `verify_ood_holdout` -- unchanged core
  invariants.
- The builder script's `_split_group_key(window)` is a thin wrapper over
  `leakage_safe_split_group_key`, kept only for `CandidateWindow`-based
  callers/tests.
- `scripts/build_selector_dataset_v2_calibrated_targeted_pilot.py`,
  `scripts/selector_v2_calibrated_pilot_gates.py`, and
  `scripts/audit_selector_v2_calibrated_pilot_leakage.py` all import these
  functions rather than reimplementing grouping logic. The audit script
  additionally checks leaky `feat_*` columns (via `advanced.py`), duplicate
  windows, and independently re-runs all three split verifiers.

## Heuristic DSL

- `src/llmserveopt/heuristics/` -- verified DSL compiler, schema,
  expressions, and policy wrapper.
- `SchedulerGenomeV1` compiles only into the subset of this DSL that
  remains causally and semantically valid.

## Tools

- `tools/policy_library_v2_experiment.py` -- expanded-library frontier
  workflow driver.
- `tools/composition_smoke_experiment.py` -- correctness-only composition smoke.
- `tools/native_composition_pilot.py` -- small Wulver-native composition
  falsification pilot.
- `tools/composition_score_rank_smoke.py` -- correctness-only smoke for
  `weighted_reciprocal_rank`/`weighted_score` aggregation and
  `instrumentation.py`'s decision-trace recording; explicitly not a
  performance claim (see `smoke_result.json`'s `scientific_claim` field).
- `tools/*.sbatch` -- SLURM launchers for focused tests and deferred
  composition/synthesis work.

## Selector Entry Points (Phase 2C / local lineage)

| Entry point | Scope | Notes |
|---|---|---|
| `scripts/evaluate_selector_v2_clean_pilot.py` | Selector v2 clean-pilot training/evaluation | Gates on both `quality_gates.json` and the independent `leakage_audit.json`. Supersedes `scripts/train_selector_v2_calibrated_prototype.py` (marked historical in-file). |
| `scripts/run_phase2c_final_selector_improvement.py` | Phase 2C selector diagnostics | Result: `SELECTOR_STATUS = IMPROVABLE`, strict best selector remains the prior Phase 2C.3 `native_non_oracle_dt`. See `docs/audits/phase2c_final_selector_improvement_audit.md`. Not yet re-run against the 27-policy library -- see [ROADMAP_GAP_ANALYSIS.md](ROADMAP_GAP_ANALYSIS.md). |
| `scripts/run_local_e2e_smoke.py` | Small end-to-end pipeline smoke test | Real trace -> causal features -> policy reward vector -> selector -> ANWG. Extended for the integrated branch to exercise the full 27-policy registry -- see the integrated smoke test in `tests/`. |
| `scripts/run_composition_smart_pilot.py` | Local composition diagnostic pilot | A precomputed-vector proxy, explicitly distinct from the native `policies/composition.py` harness above; does not duplicate it. |
| `scripts/run_module_credit_report.py` | Focused module-credit report | Falls back to a synthetic fixture (`--use-synthetic-fixture`) if the hard-coded default Wulver artifact path is absent; writes `results/module_credit_report/latest/`. |
| `scripts/run_real_module_credit_evaluation.py` | Real (non-synthetic) module-credit evaluation | Adapts `results/wulver_imports/module_intervention_credit_20260721T224322Z` (read-only) into the canonical row schema and evaluates identity/structural/contextual/suitability credit models. Writes `results/module_credit_report/real_wulver_20260721T224322Z/`. Imported by `run_module_credit_overnight.py` (dynamic module load). |
| `scripts/run_module_credit_overnight.py` | Overnight module-credit model search | Resumable (`--resume-dir`, `checkpoints/`, `heartbeat.json`) local-only driver; never launches Wulver jobs or uses synthetic fixtures for model selection. Result: `MODULE_CREDIT_MODEL_STATUS = WEAK_GENERALIZATION`/`OVERNIGHT_MODULE_CREDIT_STATUS = IMPROVED_BUT_NOT_READY`. Writes `results/module_credit_overnight/`. |

## Tests

- `tests/test_policy_library_v2.py`, `tests/test_policy_composition.py`,
  `tests/test_structural_synthesis.py` -- Policy Library v2/composition/
  structural-synthesis (structural synthesis tests also exercise
  `genome.py` directly; there is no separate dedicated genome test file).
- `tests/test_selector_dataset_v2.py`, `tests/test_selector_v2_clean_pilot_audit.py`
  -- reconciled split-leakage architecture.
- `tests/test_advanced_selector_models.py` -- causal advanced-selector formulations.
- `tests/test_local_e2e_smoke.py` -- local smoke pipeline mechanics.

These are the medium-validation target for this integration; see
[LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md) for exact pass counts.
