# CC5 Uncertainty / Regime Refinement Report

Date: 2026-08-03
Branch: `contextual-compositional-heuristics-20260731`
Starting SHA: `7718214119e7eff8f242ff974aad00d37063906a`
New SHA: `4d14a0837f3e84688caf1488b9b08054442495ec`
Canonical issue: [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5).
tmux session: `cc5_uncertainty_regime`; log: `logs/cc5_uncertainty_regime_20260803_195020.log`.
Reference run: `results/cc5_uncertainty_regime_refinement/20260803T202108Z/`
(untracked/local; reproducible via `bash results/cc5_uncertainty_regime_refinement/20260803T202108Z/replay_commands.sh`).

## 1. Verdict, Up Front

**CC5 decision-gate verdict: `REGIME_SPECIFIC_ONLY`. Exit gate: NOT PASSED.**

Model-agnostic uncertainty is now implemented and calibrated. The best
completion-safe deployable system (OOD + calibrated uncertainty, with
validation-tuned hybrid fallback to best-global or best-fixed) improves on
the CC4b/CC5 retry point estimate (0.4019 vs 0.4006 ANWG) and still clearly
beats best fixed (0.3895) and the hard selector (0.3938), but remains
**0.0006 ANWG short of best global composition (0.4025)**. Pure
best-global fallback fails the completion constraint (7 violations) and is
rejected. CC6 remains **BLOCKED**. CC5 stays `IN PROGRESS`.

## 2. State Verification

Confirmed at start of this query:

* branch `contextual-compositional-heuristics-20260731`;
* HEAD `7718214119e7eff8f242ff974aad00d37063906a` (expected starting SHA);
* local/remote synchronized at that SHA before work began;
* status checker passed; resume-readiness failed only on a preparatory dirty
  tree (`build_candidate_matrix` extraction already in progress);
* CC5 `IN PROGRESS`; issue #5 OPEN; CC6 BLOCKED;
* CC4b + CC5 retry artifacts present and `validate_cc4_dataset` clean.

## 3. Diagnosis Of Prior Uncertainty Behavior

Audited against `results/cc5_contextual_composition_predictor_retry/20260803T192246Z/`:

| Model class | Native / ensemble uncertainty before this query |
|---|---|
| `random_forest` | Yes -- per-tree prediction std |
| `gradient_boosting` | **No** (selected by LOWO-CV) |
| `knn` / `ridge` / `decision_tree` | No |

Gradient boosting lacked usable uncertainty because
`_predict_with_uncertainty` only inspected `RandomForestRegressor.estimators_`.
Across two independent retrains, LOWO-CV never selected RF, so the deployed
gate was **OOD-only**.

**28/76 abstentions (36.8%)** were all `fallback_reason=ood` (none
uncertainty-triggered). Abstention was concentrated in
`underloaded` (7), `long_output` (6), `long_prompt` (6),
`azure_conversation_like` (3), `burstgpt_derived` (3), plus singletons.

Abstention did **not** correlate with high regret: mean regret on abstained
windows was **0.0194** vs **0.0309** on kept windows. OOD score alone was
structurally sensible (far regimes flagged) but **poorly aligned with
predictor failure** -- it was conservative on easy underloaded windows
(100% abstention, 0 regret) while missing higher-regret ID mistakes.

## 4. Model-Agnostic Uncertainty

Implemented two methods; both calibrated on **VALIDATION only** (TRAIN-fit
residuals/bootstraps; no held-out leakage; seed=0):

1. **normalized split-conformal residual intervals** (selected);
2. **bootstrap refit ensembles** (n=12 window bootstraps).

| Method | Empirical coverage | Calibration error | Fit overhead (s) | Selected |
|---|---|---|---|---|
| `normalized_split_conformal` | 0.8029 | **0.0029** | 0.36 | **yes** |
| `bootstrap_ensemble` | 0.8029 | 0.0029 | 0.89 | no (tie on error; conformal faster) |

Selection thresholds were grid-searched on VALIDATION only (maximize mean
ANWG of predict-or-fallback). Artifact manifests record
`uncertainty_schema_version=2`; stale schemas are rejected.
Mean inference overhead with conformal scoring: **~0.19 ms / window**.

Point-model selection criteria were **unchanged** (LOWO-CV still selects
`gradient_boosting`).

## 5. Per-Regime Analysis

Main deployable variant on 76 held-out windows:

| Regime | n | Pred ANWG | Global ANWG | Hard | Fixed | Oracle | Abstain | Winner vs global |
|---|---|---|---|---|---|---|---|---|
| long_output | 7 | 0.0134 | 0.0077 | 0.0237 | 0.0134 | 0.0237 | 0.86 | predictor |
| burstgpt_derived | 3 | 0.0262 | 0.0131 | 0.0187 | 0.0262 | 0.0337 | 1.00 | predictor |
| azure_conversation_like | 3 | 0.1277 | 0.1099 | 0.1330 | 0.1277 | 0.1383 | 1.00 | predictor |
| kv_pressure | 7 | 0.2241 | 0.2048 | 0.1992 | 0.1992 | 0.2394 | 0.00 | predictor |
| selective_admission_trap | 7 | 0.2429 | 0.2549 | 0.2493 | 0.2324 | 0.2775 | 0.14 | global |
| long_prompt | 7 | 0.2603 | 0.2611 | 0.2549 | 0.2549 | 0.3017 | 0.86 | global |
| saturated | 7 | 0.3575 | 0.3459 | 0.3418 | 0.3418 | 0.3888 | 0.00 | predictor |
| burst_transition | 7 | 0.4835 | 0.4900 | 0.4450 | 0.4343 | 0.5203 | 0.29 | global |
| priority_conflict | 7 | 0.4863 | 0.4974 | 0.4867 | 0.4867 | 0.5183 | 0.00 | global |
| prediction_noise | 7 | 0.5735 | 0.5786 | 0.5766 | 0.5550 | 0.6085 | 0.14 | global |
| mixed_slo | 7 | 0.7165 | 0.7379 | 0.6941 | 0.7059 | 0.7479 | 0.00 | global |
| underloaded | 7 | 0.9391 | 0.9391 | 0.9391 | 0.9391 | 0.9391 | 1.00 | tied |

* **Predictor-win regimes:** `kv_pressure`, `saturated`, `long_output`,
  `azure_conversation_like`, `burstgpt_derived` (last three largely via
  completion-safe fixed fallback, not active model trust).
* **Global-composition-win regimes:** `burst_transition`, `long_prompt`,
  `mixed_slo`, `prediction_noise`, `priority_conflict`,
  `selective_admission_trap`.
* **Tied:** `underloaded`.
* **Worst by ANWG:** `long_output`, `burstgpt_derived`,
  `azure_conversation_like`.
* **Uncertainty catch regimes:** `long_prompt`, `prediction_noise`,
  `selective_admission_trap`.
* **Uncertainty miss regimes:** several ID regimes where the model loses
  to global without abstaining (`mixed_slo`, `priority_conflict`,
  `kv_pressure`, `saturated`, …).

### Restricted operating envelope

Trust the contextual predictor without forcing fallback only on:

* `kv_pressure`
* `saturated`

Elsewhere, use the validation-tuned completion-safe hybrid fallback
(best-global when validation shows it is completion-safe vs best-fixed;
otherwise best-fixed). Do not claim universal superiority over
`best_global_composition`.

## 6. Fallback Variant Comparison

| Variant | Mean ANWG | Abstention | Completion violations |
|---|---|---|---|
| **ood + uncertainty + hybrid fallback (MAIN)** | **0.4019** | 0.382 | **0** |
| regime-aware + hybrid | 0.4019 | 0.382 | 0 |
| never abstain | 0.4014 | 0.000 | 1 |
| uncertainty-only + global | 0.4011 | 0.079 | 2 |
| ood + uncertainty + global | 0.4006 | 0.382 | **7** |
| current OOD-only + fixed | 0.4006 | 0.368 | 0 |
| ood + uncertainty + fixed | 0.4006 | 0.382 | 0 |
| uncertainty-only + fixed | 0.4004 | 0.079 | 1 |

The task's preferred "fallback to best global composition" path was
evaluated and **fails completion constraints**. The retained main system
uses calibrated confidence gating and falls back to a **validation-only
completion-safe choice between best-global and best-fixed**.

## 7. Decision Gate

Applied `determine_cc5_verdict` without threshold changes:

* completion violations = 0
* n_eval = 76 >= 8
* beats fixed: 0.4019 >= 0.3895 → True
* beats global: 0.4019 >= 0.4025 → **False** (gap 0.0006)
* competitive with hard selector: True

→ **`REGIME_SPECIFIC_ONLY`**.

## 8. Exact Next Research Step

Uncertainty gap is closed; regime analysis is done. Remaining PROCEED
blocker is the 0.0006 ANWG deficit vs best global composition, concentrated
in six global-win regimes. Exact next action (still under issue #5; do
**not** begin CC6):

1. Either **freeze the restricted envelope above** as the deployable CC5
   scope and document non-universal status as accepted, or
2. Run a **narrow, validation-only regime-specialist follow-up** targeting
   only the six global-win regimes (no full redesign, no CC6).

## 9. Artifacts

`results/cc5_uncertainty_regime_refinement/20260803T202108Z/` contains:

* `calibration_manifest.json`, `coverage_error_tables.csv`,
  `uncertainty_threshold_grid.csv`, `uncertainty_diagnostics.csv`
* `per_window_predictions.csv`, `per_regime_summaries.csv`,
  `fallback_comparisons.csv`, `confidence_intervals.csv`
* `regime_analysis.json`, `regime_fallback_rules.json`,
  `model_card.md`, `replay_commands.sh`, `manifest.json`, `verdict.json`
* per-variant CSVs

## 10. Tests

Focused uncertainty/fallback tests added in
`tests/test_cc5_uncertainty_regime.py` (calibration, no leakage,
determinism, threshold selection, fallback modes, stale-schema rejection,
artifact compatibility, GB usable uncertainty). Existing CC5 tests updated
for calibration artifacts.
