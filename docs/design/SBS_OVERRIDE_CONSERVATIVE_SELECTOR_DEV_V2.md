# SBS Override Conservative Selector Dev V2

Created: 2026-09-19.

Status: design/source freeze for a later development-only V2 computation.

## Boundary

`CONFIRMATORY_LABEL_ACCESS = NOT_ACCESSED`

`OUTER_OOF_RESULTS_USED_FOR_FINAL_SELECTION = NO`

`FRESH_CONFIRMATORY_OUTCOMES_USED = NO`

V2 uses only the joint240 development corpus. It does not load, join,
summarize, or evaluate fresh confirmatory terminal outcomes.

## 1. Candidate Universe

The development rows are the canonical joint240 SBS-disagreement
state-action rows:

- 8,888 SBS-disagreement states;
- 21,858 unique non-SBS canonical action rows;
- 236 scenarios;
- five frozen scenario folds;
- `STATE_ACTION_V1` feature order: 95 state features plus 70 action-vs-SBS
  features.

V2 candidate model families:

- `EXTRA_TREES`;
- `HIST_GRADIENT_BOOSTING`;
- `ENSEMBLE_50_50`.

V2 weighting strategies:

- `STATE_EQUAL`;
- `SCENARIO_EQUAL`.

V2 model hyperparameters are bounded to the V1 grids:

- ExtraTrees: `n_estimators=160`, `max_depth in {8, None}`,
  `min_samples_leaf in {10, 25}`, `max_features=0.5`;
- HistGradientBoosting: `max_iter=120`, `learning_rate in {0.05, 0.1}`,
  `max_leaf_nodes in {15, 31}`, `l2_regularization=0.1`;
- 50/50 ensembles use one bounded ExtraTrees config and one bounded
  HistGradientBoosting config with no learned blending weight.

V2 gate candidates:

- `MEAN_THRESHOLD` with margin `0`;
- `OOF_RESIDUAL_LOWER_BOUND` with q90 or q95 residual margin.

Tau grid:

`[0, 0.001, 0.0025, 0.005, 0.01, 0.02]`

No new thresholds, model families, features, hyperparameters, residual
quantiles, or weighting schemes are introduced.

## 2. Nested OOF Development Evaluation

The existing V1 nested outer-fold experiment is retained only as an estimate of
the development-side performance of the selection procedure.

Nested outer held-out outcomes must not be used to choose the final V2
confirmatory configuration. In particular, final V2 selection must not maximize
outer-OOF gain, outer bootstrap lower bound, outer beneficial/harmful counts,
outer negative-scenario counts, or any other statistic computed from outer
held-out outcomes.

The V1 design-doc statement suggesting final selection from outer-OOF bootstrap
summaries is superseded for V2 and documented as a V1 ambiguity.

## 3. Full-Development Grouped-CV Final Selection

After nested OOF evaluation exists, V2 performs a separate development-only
final-selection stage over the entire joint240 development corpus.

The stage uses the same frozen five scenario folds as grouped-CV folds. For
each candidate configuration, every development state-action row is predicted
by a model trained without that row's scenario. No in-sample predictions are
permitted for final candidate selection or gate calibration.

For each candidate, V2 evaluates grouped-crossfit predictions with this exact
lexicographic objective:

1. maximize minimum fold mean realized gain;
2. then maximize overall mean realized gain;
3. then minimize number of negative aggregate-gain scenarios;
4. then minimize number of harmful overrides;
5. then minimize override rate;
6. then deterministic stable-key tie-break.

Bootstrap lower bound is not a V2 tuning criterion.

## 4. Gate Calibration

For residual lower-bound gates, residuals are:

`predicted_A_SBS - observed_A_SBS`

computed only from full-development grouped-crossfit predictions.

For the selected candidate, V2 freezes:

- gate family;
- q90/q95 margin name if applicable;
- crossfit-derived residual margin;
- tau.

The final all-development model must not be used to estimate the residual
margin.

## 5. All-Development Final Model Fit

Only after the V2 candidate and gate parameters are frozen:

1. fit the selected estimator or estimators on all development rows;
2. preserve the exact `STATE_ACTION_V1` feature order;
3. serialize the final model bundle;
4. write a machine-readable selector specification;
5. run reload validation.

The later confirmatory evaluator must consume the frozen selector without any
training or calibration.

## 6. Serialization

Expected V2 artifacts after the later full development run:

- `PREREGISTERED_SEARCH_DESIGN_V2.json`;
- `development_full_cv_candidate_results_v2.csv`;
- `FINAL_CONFIRMATORY_SELECTOR_V2.json`;
- `FINAL_CONFIRMATORY_SELECTOR_V2.joblib`;
- `reload_prediction_test_v2.json`;
- `RUN_SUMMARY_V2.json`.

The V1 artifacts remain unchanged and are not overwritten.

## 7. Confirmatory Evaluation

Fresh confirmation remains blocked until V2 design/source/test freeze and
development-only recomputation are complete. The later one-shot confirmatory
task may evaluate only the frozen V2 selector and must not retrain or
recalibrate anything.
