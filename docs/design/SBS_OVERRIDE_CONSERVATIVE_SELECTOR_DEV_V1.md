# SBS_OVERRIDE_CONSERVATIVE_SELECTOR_DEV_V1

## Boundary

This development task freezes a final conservative SBS-relative override selector using only the validated joint240 development corpus. Fresh confirmatory terminal outcomes are not training, validation, calibration, or selection inputs.

Forbidden input substring for training/evaluation code:

`sbs_override_fresh_id_confirmatory_terminal_label_v1`

Known fresh confirmatory labels may be checked only for existence, prior provenance, and structural counts already recorded elsewhere. Outcome columns such as `Q_SBS`, `A_SBS`, utility distributions, positive/negative effect counts, and model performance on fresh labels are out of scope until the final one-shot confirmation task.

## Development Inputs

Primary development table:

`/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1/state_action_rows_full.csv`

Policy/action map:

`/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1/state_policy_action_map_full.csv`

Expected development universe:

- 8,888 SBS-disagreement states
- 21,858 unique non-SBS canonical action rows
- 236 scenarios
- 5 existing scenario folds
- `STATE_ACTION_V1`: 95 `state__*` features and 70 `action__*` features

Target:

`A_SBS(s,a) = Q_SBS(s,a) - Q_SBS(s,SBS)`, stored as `a_sbs_anwg`.

## Limited Search

Feature set:

`STATE_ACTION_V1` only.

Model families:

- `EXTRA_TREES`
- `HIST_GRADIENT_BOOSTING`
- optional preregistered simple ensemble: `0.5 * EXTRA_TREES + 0.5 * HIST_GRADIENT_BOOSTING`

The earlier V1 selector is retained as a fixed reference baseline. Ridge remains a recorded prior baseline and is not part of the main V2 search budget.

Weighting strategies:

- `STATE_EQUAL`: each state contributes total weight 1, split over its unique non-SBS candidate actions.
- `SCENARIO_EQUAL`: each scenario contributes total weight 1, split over states and then over candidate actions.

Safety gates:

- `MEAN_THRESHOLD`: override if predicted advantage is greater than tau.
- `OOF_RESIDUAL_LOWER_BOUND`: estimate one-sided overprediction margins from cross-fitted training residuals `r = A_hat - A_true`, with `q90` or `q95`; override if `A_hat - q > tau`.

Tau grid:

`0, 0.001, 0.0025, 0.005, 0.01, 0.02`

The residual gate is empirical and development-only. It is not described as formal conformal coverage.

## Nested Protocol

Use the existing five scenario folds. For each outer fold, train/tune/calibrate only on the other four folds. Inner validation is grouped by those existing scenario folds. All candidate action rows for a state remain together because folds are scenario-level.

Hyperparameters, weighting, residual margin, and tau are chosen from inner cross-fitted development-training predictions. Outer fold outcomes are used only once, after the selected inner configuration is fixed.

## Selection Objective

The preregistered primary search objective is the computationally bounded fold-robust objective:

maximize minimum inner-fold mean realized gain per state.

Tie-breakers, in order:

1. higher overall inner mean realized gain;
2. fewer inner negative-gain scenarios;
3. fewer harmful overrides;
4. lower override rate;
5. deterministic stable configuration key.

Final V2 choice is made from full outer OOF development summaries using the same safety-first ordering with scenario bootstrap lower bound first:

1. highest scenario-bootstrap 95% lower bound for mean realized gain per state;
2. positive folds out of 5;
3. higher mean realized gain per state;
4. fewer negative scenarios;
5. fewer harmful overrides;
6. lower override rate;
7. deterministic stable configuration key.

Eligibility is compared to V1 but no post-hoc requirement is added after seeing results.

## Confirmatory Protocol Freeze

After the V2 selector is frozen and refit on all 236 development scenarios, the later confirmatory task must:

1. enumerate unique non-SBS candidate actions for each fresh disagreement state;
2. compute `STATE_ACTION_V1` in the frozen feature order;
3. predict with `FINAL_CONFIRMATORY_SELECTOR_V1`;
4. apply the frozen conservative gate;
5. choose one candidate action or abstain to SBS;
6. only then join the fresh terminal labels;
7. calculate realized gain.

Primary confirmatory metric:

mean realized `A_SBS` per fresh disagreement state.

Primary uncertainty:

scenario-level bootstrap over the 78 supported fresh scenarios.

Verdict rules:

- `FRESH_CONFIRMATION_SUCCESS`: mean realized gain > 0, scenario-bootstrap 95% lower bound > 0, benefit is not driven by a tiny number of scenarios, and downside metrics are reported.
- `FRESH_CONFIRMATION_POSITIVE_BUT_UNCERTAIN`: point estimate > 0 but CI includes zero.
- `FRESH_CONFIRMATION_FAIL`: mean realized gain <= 0 or meaningful generalization is absent.

This task stops before opening fresh confirmatory outcomes.
