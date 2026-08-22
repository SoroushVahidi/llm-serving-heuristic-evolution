# Family-A Stateful Controller V1

Date: 2026-08-20

Status: frozen before controller evaluation.

## Scientific Question

The repaired Family-A observability diagnostic found a continuation-dominated ESTF/WFS signal. The controller question is therefore:

Can a controller decide when the system should enter, remain in, or leave an ESTF-like versus WFS-like scheduling regime, such that persistent policy behavior captures the native-continuation advantage better than either fixed parent?

This design is not a one-step ESTF/WFS action selector. The controller has persistent mode state and delegates scheduling to the selected parent policy.

## Prior Evidence Used

The repaired diagnostic established:

- 91 repaired disagreement events across 32/64 Family-A TRAIN/VAL scenarios.
- Delta_same: 16 positive, 5 negative, 70 zero; mean +0.055.
- Delta_native: 58 positive, 1 negative, 32 zero; mean +1.033.
- Sign disagreement between Delta_same and Delta_native: 42/91.
- Mean absolute continuation dependence about 0.989.
- Continuation magnitude exceeded local-action magnitude in 52/91 events; local-action magnitude never exceeded continuation magnitude.
- Native-sign prediction was stronger than local-sign prediction.
- Service/request-distribution features were strongest; fairness debt was secondary; history was modest; KV/rank geometry was weak.

Because of this, the first constructive scheduler is a persistent regime controller.

## Controller Representation

Primary representation: `STATEFUL_TREE`.

The offline mode scorer is a shallow decision tree:

- estimator: `DecisionTreeClassifier`
- `max_depth = 3`
- `class_weight = "balanced"`
- `random_state = 20260820`

The deployed policy stores the fitted tree as a frozen, deterministic tree table. The runtime policy does not use a random forest, neural network, reinforcement learner, AutoML search, or evolutionary search.

Comparator for evaluation: a stateless version of the same fitted tree, using the same candidate gate and the same features but no dwell memory. It is diagnostic only.

## Supervision Target

Primary target: persistent native-policy advantage.

For each repaired diagnostic event:

- label `ESTF_MODE = 1` if `Delta_native > 0`
- label `WFS_MODE = 0` if `Delta_native <= 0`

Zero-native events are treated conservatively as no evidence to enter ESTF mode. This avoids training a separate WFS-win model from a single negative event and preserves WFS as the fairness-preserving default.

Unit of supervision: repaired disagreement event.

Sample weighting: class-balanced tree fitting only. No scenario-specific tuning and no outcome-magnitude weighting in V1.

Grouped split strategy: grouped cross-validation by canonical Family-A scenario ID. No event from the same scenario can appear in both train and validation folds.

TRAIN/VAL only. TEST scenarios and TEST metrics are excluded from fitting, threshold decisions, and evaluation.

## Online Causal Features

Only features available causally at scheduler decision time may be used.

### A. Request/Service Distribution

- `queue_age_p10`
- `queue_age_p50`
- `queue_age_p90`
- `queue_age_mean`
- `predicted_output_tokens_p10`
- `predicted_output_tokens_p50`
- `predicted_output_tokens_p90`
- `predicted_output_tokens_mean`
- `prompt_tokens_p10`
- `prompt_tokens_p50`
- `prompt_tokens_p90`
- `prompt_tokens_mean`
- `est_service_time_p10`
- `est_service_time_p50`
- `est_service_time_p90`
- `est_service_time_mean`

### B. Fairness/Starvation

- `max_class_deficit_ratio`
- `longest_waiting_age`
- `n_distinct_classes_in_queue`

### C. Workload/Queue Pressure

- `step`
- `queue_length`
- `active_count`
- `completed_count`
- `n_gpus`

### D. Urgency/Slack

- `laxity_p10`
- `laxity_p50`
- `laxity_p90`
- `laxity_mean`
- `fraction_laxity_negative`
- `fraction_laxity_near_deadline`

### E. Resource/KV

- `mean_kv_utilization`
- `max_kv_utilization`
- `free_kv_capacity`
- `prefilling_count`
- `decoding_count`

### F. Short Causal History

Short-history features are not included in the primary V1 model. The repaired analysis found only modest and inconsistent history gain, while a first standalone policy benefits from simpler online semantics. History may be reconsidered only after this V1 evaluation.

### G. Scenario-Constant Metadata

Scenario-generator parameters are not runtime inputs. The controller does not use:

- `favlong` or `favshort`
- synthetic family labels
- hidden generator identity
- random seed
- split labels
- scenario ID
- any TEST indicator

The controller also excludes future-derived or post-outcome fields such as actual output tokens, completed-future outcomes, counterfactual branch outcomes, and simulator future state.

## Candidate Region Guard

The scorer is trained only on repaired ESTF/WFS disagreement events, but the runtime scheduler sees ordinary states too. To prevent event-only training mismatch:

- The controller evaluates switch evidence only when ESTF and WFS produce different canonical actions on the current observable state.
- Parent-action probing must snapshot and restore observable GPU placement counters before the selected parent policy is called.
- Outside this causally defined candidate region, the controller remains in its current mode.

Pair-specific geometry is used only for this causal candidate gate, not as a model feature in V1.

## Temporal Semantics

Controller state:

- `mode in {ESTF_MODE, WFS_MODE}`
- `steps_in_mode`
- switch diagnostics

Initial mode: `WFS_MODE`.

State reset: controller mode and diagnostics reset between scenarios.

Switch evaluation boundary: every scheduler decision step, subject to the candidate-region guard.

Minimum dwell: `20` scheduler steps. This inherits the prior frozen dwell/reaction reference from the hierarchical router evidence.

Hysteresis:

- If mode is `WFS_MODE`, switch to `ESTF_MODE` only if:
  - candidate region is true
  - `steps_in_mode >= 20`
  - estimated `P(ESTF_MODE) >= 0.65`
- If mode is `ESTF_MODE`, switch to `WFS_MODE` only if:
  - candidate region is true
  - `steps_in_mode >= 20`
  - estimated `P(ESTF_MODE) <= 0.35`
- If score is ambiguous, remain in current mode.

Fallback behavior:

- If no fitted model is present, features are malformed, no feasible parent action exists, or the state is outside the candidate region, remain in current mode.
- The selected parent policy makes the actual scheduling decision.

No dwell sensitivity sweep is part of the primary evaluation. The primary value is dwell 20.

## Baselines

The TRAIN/VAL evaluation compares:

1. fixed ESTF
2. fixed WFS
3. best fixed ESTF/WFS parent by scenario
4. stateless version of the same tree predictor
5. stateful controller
6. native-pair ESTF/WFS envelope by scenario
7. six-policy portfolio envelope only if computable apples-to-apples from existing infrastructure

No VTC, vLLM-LTR, FSP, or PARS integration is included in this first internal feasibility experiment.

## Primary and Secondary Metrics

Primary utility: `arrival_normalized_weighted_goodput`.

Paired quantities:

- controller minus ESTF
- controller minus WFS
- controller minus best fixed parent
- controller minus native-pair envelope
- marginal contribution over six-policy portfolio envelope if available

Secondary metrics use existing repository metrics where available:

- completion fraction
- weighted completion fraction
- mean, median, and p95 latency
- mean, median, and p95 queuing time
- SLO violation rate
- GPU utilization

## Safety Gate

A controller does not count as scientifically successful if it improves ANWG by severe degradation of existing fairness/starvation proxies or completion behavior.

Safety checks:

- completion fraction must not collapse relative to both parents
- weighted completion fraction must not collapse relative to both parents
- p95 queuing and p95 latency must not show a large unpaired regression without ANWG justification
- SLO violation rate must not materially worsen relative to both parents
- mode occupancy and switch diagnostics must show genuine stateful behavior rather than fixed-parent collapse or rapid thrashing

These safety checks are descriptive for V1. They are not retuned after outcomes.

## Offline Feasibility Gate

Before full simulation, use only repaired TRAIN/VAL event data.

Required offline reports:

- grouped CV prediction of `sign(Delta_native)`
- majority baseline
- balanced accuracy
- ROC-AUC when both classes exist
- macro F1
- offline event-order replay of dwell/hysteresis
- abstention rate
- predicted mode shares
- switch counts

Offline GO requires all of:

- grouped tree balanced accuracy exceeds the grouped majority baseline
- grouped tree macro F1 exceeds the grouped majority baseline
- grouped ROC-AUC is above 0.50 when valid
- offline replay uses both modes nontrivially, with neither mode below 10% or above 90% of candidate events
- abstention rate is not above 90%
- no mode segment violates the 20-step dwell rule in event step-space
- no TEST rows or TEST-derived fields are present

If any gate fails, full simulation is not launched and the result is `STATEFUL_CONTROLLER_OFFLINE_NO_GO`.

## Full Simulation Gate

If the offline gate passes, run the frozen Family-A TRAIN/VAL controller evaluation only.

Record:

- exact command
- git SHA and dirty status
- hashes of design and repaired inputs
- model parameters
- dwell and hysteresis parameters
- feature list
- split and scenario IDs
- start and end timestamps
- per-scenario results
- failures
- switch counts
- ESTF/WFS occupancy
- dwell segment distribution
- available safety metrics

No TEST scenarios are run.

## Non-Interference Requirements

The implementation must ensure:

- feature collection is observational
- parent-action probing restores observable GPU counters before selected policy execution
- parent policies are unchanged
- switching mode is the only persistent controller state
- selected parent action equivalence holds when the controller is fixed in one mode
- no future-derived feature is used

## Classification Rules

Final classifications:

- `STATEFUL_CONTROLLER_POSITIVE_SIGNAL`: both modes exercised, controller meaningfully beats best fixed parent, safety/fairness proxies do not collapse, held-out TRAIN/VAL behavior is stable, and preferably portfolio marginal contribution is positive.
- `STATEFUL_CONTROLLER_MIXED_SIGNAL`: some gains exist but are weak, concentrated, unstable, or tradeoff-heavy.
- `STATEFUL_CONTROLLER_NO_GO`: no useful gain, fixed-parent collapse, excessive switching, safety/fairness failure, or poor held-out generalization.
- `STATEFUL_CONTROLLER_OFFLINE_NO_GO`: offline feasibility gate fails and full simulation is not launched.

Next-step labels:

- `FREEZE_CONTROLLER_AND_PREPARE_TEST`
- `REFINE_CONTROLLER_ON_TRAINVAL`
- `STOP_FAMILY_A_CONSTRUCTION`
- `NEED_CONTROLLER_INTEGRITY_REPAIR`
