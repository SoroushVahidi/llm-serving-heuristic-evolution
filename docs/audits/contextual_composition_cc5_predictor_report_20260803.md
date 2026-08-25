# CC5 Report: Contextual Composition Predictor

Date: 2026-08-03
Branch: `contextual-compositional-heuristics-20260731`
Starting SHA: `db143fc7aef5cb604ed56b778b948b5d4f271891`
Canonical issue: [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5).
tmux session: `cc5_contextual_predictor`; log: `logs/cc5_contextual_predictor_20260803_174457.log`.
Reference run directory: `results/cc5_contextual_composition_predictor/20260803T175456Z/`
(untracked/local, per repository convention -- see §11; reproducible via
`bash results/cc5_contextual_composition_predictor/20260803T175456Z/replay_commands.sh`).

## 1. Verdict, Up Front

**CC5 decision-gate verdict: `INCONCLUSIVE`. Exit gate: NOT PASSED.**

The trained predictor does not clearly beat the best fixed policy on the 6
held-out evaluation windows (predictor mean ANWG 0.2306 vs best-fixed mean
ANWG 0.2310 -- a statistically meaningless difference given bootstrap 95%
CIs of roughly [0.10, 0.37] on both, at n=6). This is not a training
failure -- the pipeline runs correctly end-to-end, is deterministic, and is
fully evaluated -- it is a genuine, honest **data-scarcity finding**: CC4
produced only 6 evaluation windows, which is not enough statistical power
to certify generalization for any model. Per explicit instruction, CC6 is
**not** queued; CC5 remains the roadmap's phase to return to, with one exact
remaining task (§10).

## 2. Dataset Audit

Full audit programmatically enforced by `validate_cc4_dataset()`
(`src/llmserveopt/experiments/cc5_contextual_predictor.py`), not just
narrative -- raises `CC5Error` on any of: non-simulator-executed rows,
`reward_vector_interpolated=True` rows, non-`valid` verification outcomes,
null ANWG/completion-fraction values, missing causal-feature columns, or
development/evaluation split overlap. Result on the CC4 reference dataset
(`results/cc4_oracle_composition_dataset/20260803T170735Z/`): **clean** --
408/408 rows true-simulator-executed and valid, 0 nulls anywhere, 6
development windows and 6 evaluation windows with zero overlap. Full detail
in the session log's dataset-audit section.

**Causal-feature enforcement (the leakage boundary):** `CAUSAL_FEATURE_COLUMNS`
is a fixed 7-column whitelist (`num_requests`, `mean_prompt_tokens`,
`mean_predicted_output_tokens`, `mean_slo_slack`, `arrival_span_s`,
`arrival_rate_est`, `num_slo_classes`) computed purely from a window's
requests before any policy executes. `FeatureEncoder` only ever reads from
this whitelist plus a candidate's own declared recipe (family one-hot,
primitive-weight vector, `k`/`laxity_threshold`/placement-key-count) --
never a `metric_*` outcome column, never `oracle_*`/`regret` (those are only
ever targets). `test_module_never_imports_dsl_synthesis_functions` and
`test_causal_feature_columns_contain_no_outcome_or_oracle_fields` lock this
in structurally, not just by convention.

## 3. Prediction Targets

**Key design simplification**: CC4 already executed all 34 pre-verified
candidates against all 12 windows. CC5's "deployable predictor" therefore
reduces to a **selection function** (causal features -> which pre-verified
candidate to use), not a DSL-synthesis function. Evaluating a selection is
a table lookup into CC4's own results -- no new simulator executions, and
"no model may execute an unverified composition" holds structurally, since
CC5 never proposes new DSL.

1. **Per-candidate regret regression (primary, trained)**: predict
   `regret = window_oracle_anwg - candidate_anwg` for a (window, candidate)
   pair from causal features + the candidate's own recipe. 204 dev rows (6
   windows x 34 candidates). Inference: evaluate all 34 candidates'
   predicted regret for a window, argmin -> recommended (already-verified)
   composition.
2. **Hard composition-family classification (trained, reported as
   underpowered)**: predict `oracle_family` (6-class) from window causal
   features. Only 6 dev examples exist (one per dev window) --
   `composition_class_predictions.csv` shows 100% in-sample match for both
   the decision tree and nearest-regime baseline, which is **not
   generalization evidence**: a depth-3 tree and a 1-NN classifier both
   trivially memorize 6 points. Reported for completeness only, not used in
   the deployable predictor or the decision gate.
3. **Direct parameter regression (explicitly not trained)**: only 3 of 12
   windows have a `weighted_primitive_mixture` oracle winner -- fewer
   positive examples than the 6-dimensional weight-vector target, which is
   definitionally non-identifiable. Per the task's own stated exception
   ("do not force direct parameter regression if oracle weights are
   non-identifiable or highly multimodal"), this was not trained.
   Qualitative observation instead: 2 of those 3 winners share the *exact
   same* weight vector (`laxity_urgency=0.5, predicted_output_length=0.5`),
   a mild positive signal that CC4 windows would need to grow before this
   becomes trainable.

## 4. Models And Comparison

Leave-one-window-out cross-validation (6 dev windows only; evaluation
windows never touched during model selection) ranked regret-regression
models by mean selected-candidate ANWG on the held-out fold:

| Model | LOWO-CV mean ANWG |
|---|---|
| `knn` (k=3) | 0.4778 (selected) |
| `ridge` | 0.4774 |
| `gradient_boosting` | 0.4757 |
| `random_forest` | 0.4707 |
| `decision_tree` | 0.4146 |

The top four models are separated by <0.01 ANWG -- another small-sample
artifact worth flagging explicitly: this ranking should not be read as
"KNN is meaningfully better," only as "KNN was not worse, and happened to
rank first by a margin the data cannot actually distinguish."

Held-out evaluation (6 windows, touched once):

| Method | Mean ANWG | 95% bootstrap CI |
|---|---|---|
| **Best global composition** | **0.2633** | [0.100, 0.442] |
| Predictor (KNN + fallback) | 0.2306 | [0.102, 0.370] |
| Best fixed policy | 0.2310 | [0.086, 0.397] |
| Existing hard selector | 0.2123 | [0.084, 0.357] |

`best_global_composition` (the single candidate with the best mean dev
ANWG, regardless of family) outperforms every other method on this
held-out set, including the trained predictor -- an important, honest
finding: with this little data, a single well-chosen static composition
generalizes at least as well as anything context-adaptive. Completion-
fraction constraint: **0 violations** (predictor never regresses completion
fraction by more than 0.05 vs best fixed on any window). Non-near-tie-only
predictor mean ANWG: 0.2620 (n=3, the 3 evaluation windows with a real
top-2 margin at the primary near-tie threshold).

## 5. Uncertainty, OOD, And Fallback

- **Uncertainty method**: per-tree prediction-std across a `RandomForestRegressor`'s
  `estimators_`, calibrated (75th percentile of dev-window uncertainty) via
  leave-one-window-out. **Not applicable this run**: KNN won model
  selection, and KNN has no ensemble to disagree with itself --
  `uncertainty_method: "unsupported_for_selected_model_type"` is recorded
  explicitly in the manifest rather than silently reporting a meaningless
  0.0 as if it were a real signal. `_predict_with_uncertainty()` is
  implemented and tested (`test_uncertainty_nonzero_for_random_forest`) for
  the case where a future retrain does select an RF.
- **OOD method**: max-abs z-score of a window's causal-feature vector
  against the dev-window feature distribution, threshold 2.0. **A real bug
  was found and fixed here** (§8): all 6 dev windows happen to share
  `num_slo_classes=3` exactly (zero variance), and two eval windows have
  `num_slo_classes=4` -- the double z-scoring this implied produced an
  uninterpretable `999999999.9999999` "score" before the fix clipped
  per-dimension z-scores to a legible ceiling of 50.0 (no `is_ood()`
  decision changed).
- **Abstention/fallback outcome**: **4 of 6 (67%) evaluation windows
  abstained**, all due to OOD (`azure_conversation_like`, `burstgpt_derived`,
  `priority_conflict`, `selective_admission_trap`), falling back to
  `fixed__weighted_shortest_processing` (the dev-fit best fixed policy).
  Only `kv_pressure` and `prediction_noise` used the model's own
  recommendation. This high abstention rate is itself informative: with
  only 6 dev windows, the "in-distribution" region the OOD gate can
  establish is necessarily tiny, so most genuinely-different held-out
  windows are (correctly) flagged as outside it. This is a data-volume
  limitation, not a bug in the gate.
- **Fallback safety**: `fit_best_fixed_policy()` only ever selects among
  `family == "fixed_policy"` candidates -- the simplest, most robust
  baseline family, never an unverified or exotic composition.

## 6. Split And Evaluation Discipline

`development_splits`/`evaluation_splits` reused verbatim from CC4's own
config (not re-derived): 6 development windows (TRAIN+VALIDATION) for
fitting and leave-one-window-out model selection; 6 evaluation windows
(ID_TEST+OOD_TEST, including both real-trace-derived windows --
`azure_conversation_like`, `burstgpt_derived`) held out entirely, touched
exactly once for §4's reported comparison. Near-tie windows (top-2 margin
< 0.005) are reported separately (`non_near_tie_predictor_anwg`, §4), not
excluded, per roadmap invariant 9. `test_split_leakage_detected` and
`test_dev_and_eval_windows_disjoint_and_match_config` lock in the
separation.

## 7. Deployable Artifact

`PredictorArtifact` (in-process; not pickled to disk this run -- see §11)
carries: `model_name`, fitted `model`, `FeatureEncoder`, `UncertaintyOODGate`,
fallback `LookupBaseline`, `dsl_schema_version=2`, `compiler_version="cc3.1"`,
`dataset_config_hash` (pinned to CC4's config hash for staleness detection),
`git_sha`, `feature_schema` (21 named features), `target_definition`,
`split_definition`, `hyperparameters`, `uncertainty_method`, `ood_method`,
`objective_definition`, `training_timestamp`, `dependency_versions`
(sklearn/numpy/pandas/python). `select_composition_with_fallback()` is the
runtime wrapper: extracts causal features -> predicts regret over the
pre-verified pool -> checks OOD/uncertainty -> falls back safely when
triggered -> returns a full decision record (`selected_candidate_id`,
`model_recommended_candidate_id`, `predicted_regret`, `uncertainty`,
`ood_score`, `abstained`, `fallback_reason`) suitable for logging. No
dynamic switching or hysteresis was added (out of CC5 scope, reserved for
CC6).

## 8. Bugs Found And Fixed During Development

1. **OOD z-score double-scoring blow-up** (§5) -- fixed by clipping
   per-dimension z-scores to 50.0 before taking the max; verified the fix
   changes zero decisions (both affected windows remain `is_ood=True`) via
   a byte-identical verdict re-run.
2. A latent factory/instance type mismatch was caught and removed before
   any test ran: an early `build_regret_regressors()` helper returned
   fitted estimator *instances* while `leave_one_window_out_cv()` expected
   zero-argument *factories* (`model_factory()`); replaced with
   `build_regret_regressor_factories()` returning lambdas, consistent with
   its only caller.

## 9. Tests And Exact Commands

```bash
python -m pytest tests/test_cc5_contextual_predictor.py -q
# 22 passed
```

Runs against the real CC4 reference dataset (skips cleanly if that
untracked directory is ever absent) rather than a synthetic fixture, since
exercising the real leakage/split boundaries end-to-end is stronger
evidence than a mock. Covers: causal-feature enforcement (no outcome/oracle
columns in the encoder's input space; the module has no DSL-synthesis
imports at all), dataset validation (clean dataset passes; stale/missing/
malformed datasets raise `CC5Error`; a synthetically-overlapped split
raises), split integrity, target construction (shape + a spot-checked
regret value matching CC4's own table exactly), verified-composition output
(every selection is drawn from the pre-verified pool), uncertainty
calculation (0 for non-ensemble models, non-negative for RF), OOD detection
(a synthetically far point is flagged, the training-distribution mean is
not), fallback (always a valid `fixed_policy` candidate), deterministic
training (two independent runs produce identical model type and verdict),
resume short-circuiting (a second call with `resume_dir` never rewrites
`manifest.json`), the full required-outputs list and manifest-field
completeness, and runtime-wrapper determinism (identical input -> identical
decision record, including on repeat calls).

No live APIs, GPU jobs, or real-vLLM jobs were run.

## 10. CC6 Recommendation (Exact Next Action)

CC5's exit gate did **not** pass -- `INCONCLUSIVE`, not `PROCEED`. Per
explicit instruction, CC6 is **not** queued as `NEXT`; it remains `BLOCKED`.
CC5 itself is not marked `COMPLETE`; it remains the roadmap's phase to
return to.

**Exact remaining task**: the blocker is data volume, not methodology or
code. A future CC5 retry should first request an expanded CC4 dataset
(more windows -- particularly more per regime, so leave-one-window-out CV
folds are not each a single point) before retraining, because:

* n=6 held-out windows cannot statistically distinguish a predictor from
  "ties best fixed policy" at any interesting effect size (§4's CIs all
  span roughly [0.10, 0.40] regardless of method);
* the top 4 LOWO-CV models are separated by less than the CV noise floor,
  so "KNN won" is not a trustworthy model-family conclusion;
* the OOD gate's "in-distribution" region is necessarily minuscule with
  only 6 dev windows, driving a 67% abstention rate that itself cannot be
  distinguished from "correctly conservative" vs "uselessly conservative"
  without more data;
* `best_global_composition` beating the trained predictor on this
  particular 6-window split is exactly the kind of result that either
  flips or firms up with more evaluation windows -- it should not be read
  as "context-adaptive composition doesn't help," only as "this dataset
  can't yet tell."

No code changes are anticipated to be required for a retry beyond
`--dataset-dir` pointing at a larger CC4 run; the pipeline itself
(validation, targets, models, uncertainty/OOD/fallback, evaluation,
verdict) is complete and tested.

## 11. Commit And Reproducibility Policy

Per repository convention (`results/*` is gitignored), the run directory
(`results/cc5_contextual_composition_predictor/20260803T175456Z/`, a few
KB) is **not** committed, and no model artifact is pickled to disk this run
(the fitted `PredictorArtifact` lives only in-process during
`run_training()`; persisting a `.pkl` was judged unnecessary complexity for
an INCONCLUSIVE-verdict run that a future retry will regenerate from
scratch against a larger dataset anyway -- reported as a scope decision,
not an oversight). Reproducible via:

```bash
bash results/cc5_contextual_composition_predictor/20260803T175456Z/replay_commands.sh
```
