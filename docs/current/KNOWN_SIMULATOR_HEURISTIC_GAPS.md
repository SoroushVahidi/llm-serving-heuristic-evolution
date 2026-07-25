# Known Simulator / Heuristic Integration Gaps

Verified during the 2026-07-24 repository reconciliation against
`origin/wulver-final-integration-20260721` @ `37849b0`. These are
code-path findings, not new scientific experiment results. **No simulator
semantics were changed in that reconciliation pass** — changing them would
risk invalidating historical DSL-heuristic and selector-feature evidence.

Focused regression tests live in
`tests/test_heuristic_simulator_integration_gaps.py`.

## 1. `CalibratedServiceModel.compute_decode_step_time()` does not drive DES timing

**Verified:** `gpu.py` advances work via token-budget / prefill-remaining
accounting (`ServiceModel.step_token_budget`, `compute_prefill_tokens` /
`compute_prefill_steps`). Nothing in the discrete-event step path calls
`compute_decode_step_time()` or `decode_time()`.

**Where it is used:** offline calibration comparison scripts
(`scripts/compare_simulator_to_real_llm_latency.py`,
`scripts/validate_simulator_calibration.py`) and unit tests.

**Implication:** docs or comments that imply calibrated decode wall-clock
predictions reshape simulator step duration overstate current wiring.
Prefill step counts from calibration *can* affect Phase-1.5 prefill
remaining; decode per-token wall-clock predictions currently do not.

**Proposed future change (not implemented here):** optionally convert
predicted decode step time into effective token-budget or wall-clock step
size under an explicit config flag, with a reproducibility note that
pre-flag runs remain on the token-budget model.

## 2. `batch.min_deadline_slack` / `batch.deadline_risk` use absolute deadlines

**Verified** in `src/llmserveopt/heuristics/policy.py` `_build_batch_vars`:

- `min_slack = min(r.slo_deadline for r in admitted)` — absolute deadline,
  not `slo_deadline - now`.
- `deadline_risk` counts `r.slo_deadline < 1.0` — almost never true for
  realistic absolute deadlines.
- Comment in-source still says `"will subtract time later (0 here)"` but
  no `now` argument is passed and subtraction never happens.
- By contrast, `req.deadline_slack` correctly uses `slo_deadline - now`,
  and `sys.slo_pressure` correctly uses queue slack.

**Implication:** DSL heuristics that reference these two `batch.*`
variables do not see true remaining slack. Registered Python policies in
`registry.py` / `external_baselines_registry.py` do not use this path.

**Proposed future change (not implemented here):** pass `now` into
`_build_batch_vars` and mirror the `req.deadline_slack` definition.
Treat as a versioned DSL semantics bump if any published heuristic
artifact depended on the absolute-deadline quirk.

## 3. `batch.*` features are not rescored as the candidate batch grows

**Verified:** `HeuristicPolicy.select_action` scores every candidate once
against an **empty** `batch_vars`, then greedily admits. During admission,
updated `batch_vars` are only passed to `check_admission`, not used to
recompute `request_score`.

**Implication:** The module docstring claim that `batch.*` variables are
"updated incrementally as requests are added to the batch" is only true
for admission-condition checks, not for ranking scores. Heuristics whose
`request_score` depends on `batch.*` effectively see empty-batch values.

**Proposed future change (not implemented here):** optional incremental
rescoring behind an explicit heuristic/compiler flag (default off for
historical reproducibility).

## 4. `HeuristicPolicy.record_completion()` is never called by the simulator

**Verified:** the only call sites under `src/` / `scripts/` / `tests/`
are the method definition itself and
`tests/test_heuristic_policy_wrapper.py`. The discrete-event simulator /
evaluation harness never invokes it.

**Implication:** `sys.recent_slo_violation_rate` stays at `0.0` during
real simulator runs unless a caller manually records completions. This
matches older Phase 2B.12 audit notes that offline windows see a zero
violation-rate feature.

**Proposed future change (not implemented here):** on request completion
(and drop, if treated as violation), call `record_completion` when the
active policy exposes that method. Gate behind a config flag initially.

## 5. Selector ML dependencies were under-declared

**Verified:** `sklearn` / `joblib` are imported throughout
`src/llmserveopt/selector/` and several scripts, and are present in the
Wulver `repo-env`, but were absent from `pyproject.toml` /
`requirements.txt` dependency declarations.

**Fix applied in reconciliation:** optional dependency group
`[project.optional-dependencies] selector` (and matching
`requirements-selector.txt`) listing `scikit-learn` and `joblib`. Core
simulator install remains unchanged.

## Non-issues confirmed

- Faithful external baselines remain `selector_eligible=False` (all 7,
  including `slai_faithful`).
- Live registry counts match the pause handoff: 20 historical + 7 Policy
  Library v2 + 7 faithful external (+ 1 oracle reference-only).
- `origin/main` is an ancestor of the integration branch and is **not**
  the scientific source of truth for this lineage.
