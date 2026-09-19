# JOINT240 SBS Advantage Learnability V1

## Purpose

This is an offline/counterfactual held-out learnability experiment.  It asks
whether online-safe predecision state/action features predict
`A_SBS(s,a)` well enough on held-out scenarios to select occasional one-step
non-SBS deviations while abstaining to SBS otherwise.

No closed-loop learned scheduler, real-vLLM run, or manuscript edit is part of
this experiment.

## Frozen Inputs

Source basis:
- full analysis code/design commit:
  `0900ee7b565ee1cbfffe86272e30e4d0db13483c`
- full terminal-label campaign source commit:
  `ff34f6fa0303b6f21e13277553f2d5da789b01d4`

Canonical inputs are read-only:
- `/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1/state_action_rows_full.csv`
  - SHA-256: `a96af891b70bcd0895440207df6febb5d1c1545af4eaaf455367e56152950dca`
- `/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1/state_policy_action_map_full.csv`
  - SHA-256: `7a4118b65e4222e189aa1139ba548ade99951ae285a50b7b193fec6ef55d1324`

Expected dataset facts:
- SBS-disagreement states: 8,888
- scenarios: 236
- frozen folds: 5
- unique non-SBS causal action rows: 21,858
- STATE_ACTION_V1 features: 95 `state__` + 70 `action__` numeric columns.

## Learning Unit and Target

Primary rows are unique non-SBS canonical actions only.  SBS action rows are
excluded from model training and represented at decision time by abstention.

Target:

```text
y(s,a) = A_SBS(s,a) = Q_SBS(s,a) - Q_SBS(s,a_SBS)
```

The causal action, not policy identity, is the prediction unit.  Policy aliases
that produced the same canonical action remain descriptive metadata only.

## Feature Sets

Primary:

```text
STATE_ACTION_V1 = 95 state__ features + 70 action__ features
```

Ablation:

```text
STATE_CORE_V1 = 95 state__ features
```

Forbidden learnable columns include `state_id`, `scenario_id`, `fold`, `step`,
policy IDs, target/outcome columns, Q/A columns, terminal metrics, future or
post-decision information, and any `traj_` Alive/control-history field.

No feature subset may be changed after inspecting held-out results.

## Sample Weights

Primary fitting uses state-equal row weights:

```text
w(s,a) = 1 / N_nonSBS_unique_actions(s)
```

Thus every state contributes total training weight 1, regardless of the number
of non-SBS unique actions.  An unweighted sensitivity may be reported only as a
secondary result.

## Splits

Use the frozen 5-fold scenario assignment in the dataset.  For outer fold `k`:
- TEST: all scenarios in fold `k`.
- TRAINING POOL: scenarios in the other four folds.

All actions from a state and all states from a scenario must stay in one split.
Outer-test rows may not influence preprocessing, hyperparameter choice,
threshold choice, calibration, model selection, or feature selection.

Within each outer training pool, rotate the four non-test folds as inner
validation folds.  Cross-fitted predictions on the outer-training pool are
used to select hyperparameters and abstention threshold.

## Models and Frozen Grids

All models are direct regressors for `A_SBS`.

`DUMMY_ZERO`
- predicts zero for every candidate.

`RIDGE`
- `StandardScaler` fit only on training rows inside each fit.
- alpha grid: `[0.01, 0.1, 1.0, 10.0, 100.0]`.

`HIST_GRADIENT_BOOSTING`
- `max_iter`: `[100]`
- `learning_rate`: `[0.05, 0.1]`
- `max_leaf_nodes`: `[15, 31]`
- `l2_regularization`: `[0.0, 0.1]`
- `random_state`: `20260919`

`EXTRA_TREES`
- `n_estimators`: `[100]`
- `max_depth`: `[8, null]`
- `min_samples_leaf`: `[5, 20]`
- `max_features`: `[0.5, 1.0]`
- `random_state`: `20260919`
- `n_jobs`: fixed by command-line worker setting.

Hyperparameter objective on cross-fitted outer-training predictions:
maximize mean realized gain per training state after threshold selection.  Tie
breaks: higher threshold, then smaller model configuration JSON string.

## Abstaining Selector

For held-out state `s`, predict every unique non-SBS candidate action.  Let:

```text
a_hat = argmax_a predicted_A(s,a)
m_hat = max_a predicted_A(s,a)
```

If `m_hat > tau`, select `a_hat`; otherwise abstain to SBS with realized gain
0.

Threshold grid:

```text
tau in {0, 0.001, 0.0025, 0.005, 0.01, 0.02}
```

Threshold selection uses only cross-fitted outer-training predictions.  The
primary threshold objective maximizes mean realized gain per state.  Ties choose
the higher threshold.

All fixed thresholds are also reported.

## Metrics

Primary state-level selector metrics:
- mean, median, and total realized gain per disagreement state;
- override rate;
- beneficial, harmful, and zero-effect override counts/rates;
- precision among overrides;
- mean gain conditional on overriding;
- mean loss conditional on harmful override;
- p5/p10 realized gain;
- maximum harmful override.

Oracle headroom:

```text
oracle_gain(s) = max(0, max_a A_SBS(s,a))
gap_closure = sum(realized_gain) / sum(oracle_gain)
```

Scenario-level metrics aggregate held-out decisions within scenario, including
positive/zero/negative scenario counts, worst scenario gain, p10 scenario gain,
and maximum harmful overrides in one scenario.

Prediction diagnostics on held-out non-SBS rows are secondary: MAE, RMSE,
Spearman, Pearson, nonzero sign accuracy, positive-action precision/recall from
predicted sign, and calibration bins.

## Bootstrap

Primary uncertainty uses scenario-level bootstrap over completed out-of-fold
decision tables, never action-row bootstrap.

Seed: `20260919`.
Replicates: `10000`.

Report 95% bootstrap CIs for mean realized gain per state, mean scenario gain,
gap closure, and override precision where stable.

## Secondary Sign Classifier

A simple secondary beneficial-action classifier may be run if inexpensive:
- logistic regression with `StandardScaler`, `C in [0.1, 1.0, 10.0]`;
- HistGradientBoostingClassifier with the analogous small grid.

It uses the same outer/inner grouped protocol and may not replace the primary
regression verdict.

## Interpretation Gate

Return `HELD_OUT_OVERRIDE_SIGNAL_CONFIRMED` only if primary
`STATE_ACTION_V1` regression:
- has positive pooled OOF mean realized gain over always-SBS;
- scenario-bootstrap 95% CI lower bound for mean gain is greater than 0;
- benefit is not confined to one fold or a tiny number of scenarios;
- harmful overrides are reported and do not erase aggregate benefit.

Other possible verdicts:
- `HELD_OUT_SIGNAL_POSITIVE_BUT_UNCERTAIN`
- `PREDICTION_SIGNAL_WITHOUT_POLICY_GAIN`
- `NO_USEFUL_HELD_OUT_SIGNAL`

The oracle is headroom only.  Always-SBS is the operational baseline.
