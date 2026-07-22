# State-Policy Suitability: Dataset Schema and API

First implementation of the project's next research stage:
policy library -> state-policy suitability -> strong selector -> module
credit -> structural synthesis -> new policies -> expanded library. Code
lives under `src/llmserveopt/selector/suitability/`.

## Long-format dataset schema

One row per `(state, deployable policy)`, produced by
`selector.suitability.dataset.build_long_format_rows`:

| Field | Type | Notes |
|---|---|---|
| `state_id` | str | `f"{trace_family}__w{window_idx}"`. |
| `state_features` | dict[str, float] | `feat_*` columns only, validated via `selector.advanced.validate_feature_columns` (rejects reward/completion/oracle/label columns). |
| `policy_name` | str | One of the 27 `policies.registry.POLICY_LIBRARY_V2_NAMES`. Oracle/hindsight policies are rejected if passed in. |
| `policy_hash` | str | `SchedulerGenomeV1.stable_hash()` -- stable across process runs, distinct per policy name. |
| `policy_representation` | dict[str, float] | Structural features from `selector.suitability.encoders.structural_features` (see below). Never includes the raw hash. |
| `reward_anwg` | float \| None | `metric_arrival_normalized_weighted_goodput` for this (state, policy). |
| `completion_fraction` | float \| None | `metric_completion_fraction`. |
| `completed_request_quality` | float \| None | `metric_weighted_goodput`. |
| `source` | str | Caller-supplied provenance tag (e.g. `"synthetic_controlled_stress_fixture"`). |
| `trace_family` | str | Caller-supplied trace/window-source identifier. |
| `temporal_block` | str | `str(window_idx)`. |
| `split` | str | **Inherited exactly** from the source window row -- never recomputed here. If the source came from `selector.dataset_v2.splits`, the leakage-safe split is preserved verbatim. |
| `seed` | int | Generation seed. |

Guarantees enforced by `build_long_format_rows`:
- rejects any oracle/hindsight policy in `deployable_policies`;
- one row per `(state, policy)` in `deployable_policies`, deterministically sorted by `(state_id, policy_name)`;
- a state's `split` is single-valued (raises `ValueError` on inconsistency) -- no cross-split duplication;
- `state_features` keys are restricted to validated `feat_*` columns.

## Policy encodings (`selector.suitability.encoders`)

- **A. Identity** -- one-hot `policyid_<name>` over a fixed, sorted 27-name vocabulary. Cannot generalize to an unseen policy name (the identity dummy for a truly novel policy is simply never trained).
- **B. Structural** -- `structural_features(genome)`: module presence + mapping-status ordinal (EXACT=2/APPROXIMATE=1/UNSUPPORTED=0) per slot (`admission_rule`, `priority_rule`, `prefill_rule`, `kv_guard`, `fairness_rule`), regime-condition count, one-hot root-op of the admission/priority expressions and of the tie-breaker (bounded vocabularies from `heuristics.dsl_schema`), AST node count, AST max depth, and per-operator AST node counts. **Never includes the raw `policy_hash`** -- doing so would let "structural" smuggle policy identity back in through the hash bytes.
- **C. Hybrid** -- state features + identity + structural, concatenated.

Coverage caveat (load-bearing for interpreting results): only 6 of 27 policies currently have a real (EXACT or APPROXIMATE) genome mapping in `policies.structural_synthesis.map_policy_to_genome` (`weighted_shortest_processing`, `edf` exact; `aging_priority`, `scorpio_style_slo_guard`, `kv_constrained_online`, `adaptive_chunked_prefill` approximate). The other 21 policies get an honest `UNSUPPORTED` placeholder genome (a single generic priority rule, no other modules) -- structurally near-identical to each other. `structural_features` encodes this placeholder faithfully rather than inventing distinguishing signal that isn't there.

`PolicyEncoder(encoding, all_policies)` is a fit-once transformer: column layout is frozen at `fit()` from the union of state-feature keys seen plus the full `all_policies` identity vocabulary, so a held-out policy's row can still be `transform()`-ed consistently.

## Joint model API (`selector.suitability.models`)

`JointRewardModel(name, encoding, all_policies, ...)`:
- `fit(rows)`
- `predict_mean(rows) -> np.ndarray` -- `mu(x, pi)`.
- `predict_uncertainty(rows) -> np.ndarray` -- `u(x, pi)`, nonnegative, policy-specific: std of per-tree predictions within one fitted `RandomForestRegressor` (each tree already fit on a bootstrap resample, so this is a legitimate, single-fit-cost uncertainty estimate).
- `predict_suitability(rows, lam=0.5) -> np.ndarray` -- `S(x, pi) = mu(x, pi) - lambda * u(x, pi)`.

`IndependentPerPolicyRewardModel` implements the same four-method interface over one regressor per policy (state features only) -- a comparison baseline for "does joint modeling beat independent per-policy regression"; the canonical implementation for production selector use remains `selector.advanced.PolicyRewardRegressorSelector`.

## Selector and evaluation API (`selector.suitability.selector`)

- `joint_select(model, rows_by_state, lam=0.5) -> {state_id: policy_name}` -- `pi_select(x) = argmax_i S(x, pi_i)`.
- `evaluate_selection(rows_by_state, selections, best_fixed_policy) -> dict` -- ANWG, regret-to-oracle, gap-closed-fraction, policy-match accuracy, each reported `overall` and stratified by true top-2 margin thresholds `{0, 0.001, 0.005, 0.010}`.
- `margin_weighted_regret(rows_by_state, selections) -> float`.
- `build_delta_rows(rows_by_state, policy_a, policy_b)`, `DeltaModel`, `evaluate_delta_model`, `delta_consistency_with_joint_model` -- the `Delta_SCORPIO_WSP(x) = R_SCORPIO(x) - R_WSP(x)` pairwise-advantage diagnostic (state-only regression, MAE/RMSE/sign-accuracy/near-zero-calibration, plus consistency against the joint model's implied `Rhat_A(x) - Rhat_B(x)`).
- `held_out_policy_split`, `held_out_policy_pilot`, `held_out_family_pilot`, `nearest_structural_policy_baseline`, `load_policy_families` -- leave-one-policy(-family)-out generalization pilots. `load_policy_families(component)` reads the documented, machine-readable `docs/current/policy_component_matrix.json` (Policy Composition Readiness audit) -- never an invented grouping.

## Reusability for the later module-credit model

Every interface above is deliberately factored so a future `C(x, pi, module_m)` module-credit model can reuse it directly: `structural_features` already exposes per-module presence/status, `PolicyEncoder` already supports arbitrary feature-column layouts, and `JointRewardModel`'s fit/predict_mean/predict_uncertainty/predict_suitability shape is designed to be copied for a module-indexed variant without redesign.
