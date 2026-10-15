# Schema and data dictionary

This document defines every column in the v1.0.0 release tables. No machine-learning
train/test split columns are present. All identifiers are synthetic.

**Conventions**

* Nullability: every column in this release is non-null in the released CSVs
  unless noted.
* Finite numerics: all metric and feature values are finite (`NaN`/`Inf` absent).
* Directionality: for ANWG/soft/WCG, higher is better. For WMT/WNT raw values,
  lower time is better; the stored `*_improvement` fields invert the difference
  so that positive improvement means a better (lower) time than SBS.
* Tolerances: use absolute tolerance `1e-10` for stored `a` identities and
  `1e-12` for ANWG sign classification.

## Key roles

| Column | Table | Key role |
|---|---|---|
| `state_id` | state_context, outcomes | Primary key of a labeled state |
| `(state_id, canonical_action_id)` | outcomes | Primary key of an alternative outcome |
| `scenario_id` | all data tables | Foreign key to `scenario_metadata.csv` |
| `(scenario_id, step)` | state_context | Unique within retained states |
| `policy_id` | policy_metadata | Descriptive policy identifier |

---

## `data/state_context.csv` (8,886 rows, 107 columns)

One row per retained labeled SBS disagreement state. Outcome columns are
deliberately absent.

### Identity and disagreement summary (12 columns)

| Column | Type | Units / values | Meaning / origin | Warnings |
|---|---|---|---|---|
| `state_id` | string | `joint240_sbs::<scenario>::<step>` | Synthetic labeled-state ID | Not a production request ID |
| `scenario_id` | string | `joint_mm_*` | Synthetic scenario ID; FK to scenario metadata | Scenario is the independent unit |
| `fold` | int | `{0,1,2,3,4}` | Frozen outer scenario fold label from historical split | Not a release train/test split artifact |
| `step` | int | decision index | SBS trajectory decision step | Correlated within scenario |
| `sim_time` | float | relative seconds | Predecision simulator time | Not an upstream timestamp |
| `sbs_canonical_action_full` | string | serialized action | Full canonical SBS action at the state | Opaque serialized form |
| `sbs_canonical_action_admit` | string | serialized admit component | Admission component of SBS action | May be empty string component |
| `n_distinct_canonical_p6_actions_full` | int | count | Distinct full canonical actions among six policies | ≥2 for disagreement states |
| `n_distinct_canonical_p6_actions_admit` | int | count | Distinct admit components among six policies | Not the selection criterion alone |
| `any_non_sbs_policy_differs` | bool/int | true/1 | At least one non-SBS full action differs | True for all retained rows |
| `n_non_sbs_policies_differ` | int | 1..5 | Count of non-SBS policies differing from SBS | |
| `differing_policy_ids` | string | policy id list | Which candidate policies differed | Serialized list, not JSON-guaranteed beyond release |

### The 95 predecision `state__*` features

All are finite numeric predecision features measured **before** the candidate
action. They are simulator-derived quantities, not terminal outcomes, and must
not be treated as private user attributes. Names are the canonical simulator
feature names; do not invent semantics beyond the encoded quantity/statistic.

Suffix guide:

* `_count` — request/entity count
* `_tokens_sum` / `_token_mass` — token counts / masses
* `_s` — seconds (relative simulator time)
* `_ratio` / `_fraction` / `_entropy` / `_utilization` — dimensionless
* `min` / `p25` / `p50` / `p75` / `p90` / `max` / `mean` / `delta` — summary statistic
* `hist_w{1,5,20}_*` — short history windows over prior decisions

| Column | Type | Units | Scientific meaning |
|---|---|---|---|
| `state__state_time_s` | float | s | Predecision simulator time |
| `state__state_step` | float/int | step | Predecision step index |
| `state__waiting_count` | float | count | Waiting requests |
| `state__migrating_ready_count` | float | count | Migrating-ready requests |
| `state__active_count` | float | count | Active requests |
| `state__admissible_count` | float | count | Admissible requests |
| `state__total_in_system_count` | float | count | Total in-system requests |
| `state__completed_count_pre` | float | count | Completions before the decision |
| `state__waiting_prompt_tokens_sum` | float | tokens | Waiting prompt-token sum |
| `state__waiting_predicted_remaining_tokens_sum` | float | tokens | Waiting predicted remaining tokens |
| `state__waiting_predicted_total_tokens_sum` | float | tokens | Waiting predicted total tokens |
| `state__active_prompt_tokens_sum` | float | tokens | Active prompt-token sum |
| `state__active_predicted_remaining_tokens_sum` | float | tokens | Active predicted remaining tokens |
| `state__active_predicted_total_tokens_sum` | float | tokens | Active predicted total tokens |
| `state__kv_tokens_used_sum` | float | tokens | KV tokens used |
| `state__kv_tokens_capacity_sum` | float | tokens | KV capacity |
| `state__kv_tokens_free_sum` | float | tokens | KV free tokens |
| `state__kv_utilization_mean` | float | fraction | Mean KV utilization |
| `state__kv_utilization_max` | float | fraction | Max KV utilization |
| `state__waiting_estimated_kv_demand_sum` | float | tokens | Estimated waiting KV demand |
| `state__projected_kv_pressure_waiting` | float | dimensionless | Projected waiting KV pressure |
| `state__waiting_large_req_gt_total_free_kv_count` | float | count | Waiting requests larger than free KV |
| `state__active_prefilling_count` | float | count | Active prefilling requests |
| `state__active_decoding_count` | float | count | Active decoding requests |
| `state__queued_prefill_token_mass` | float | tokens | Queued prefill token mass |
| `state__active_decode_remaining_token_mass` | float | tokens | Active decode remaining mass |
| `state__prefill_decode_request_ratio` | float | ratio | Prefill/decode request ratio |
| `state__prefill_decode_token_ratio` | float | ratio | Prefill/decode token ratio |
| `state__urgent_request_count_slack_le_5s` | float | count | Urgent requests with slack ≤ 5 s |
| `state__urgent_request_fraction_slack_le_5s` | float | fraction | Urgent-request fraction (slack ≤ 5 s) |
| `state__urgent_token_mass_slack_le_5s` | float | tokens | Urgent token mass (slack ≤ 5 s) |
| `state__slo_violation_count_slack_le_0` | float | count | Requests with slack ≤ 0 |
| `state__priority_sum_admissible` | float | priority units | Sum of admissible priorities |
| `state__priority_max_admissible` | float | priority units | Max admissible priority |
| `state__priority_entropy_admissible` | float | entropy | Priority entropy among admissible |
| `state__class_entropy_admissible` | float | entropy | Class entropy among admissible |
| `state__class_count_admissible` | float | count | Distinct admissible classes |
| `state__high_priority_waiting_prompt_mass` | float | tokens | High-priority waiting prompt mass |
| `state__waiting_age_s_{min,p25,p50,p75,p90,max,mean}` | float | s | Waiting-age summary statistics |
| `state__admissible_prompt_tokens_{min,p25,p50,p75,p90,max,mean}` | float | tokens | Admissible prompt-token summaries |
| `state__admissible_predicted_output_tokens_{min,p25,p50,p75,p90,max,mean}` | float | tokens | Admissible predicted-output summaries |
| `state__admissible_predicted_total_tokens_{min,p25,p50,p75,p90,max,mean}` | float | tokens | Admissible predicted-total summaries |
| `state__active_predicted_remaining_tokens_{min,p25,p50,p75,p90,max,mean}` | float | tokens | Active remaining-token summaries |
| `state__slack_s_{min,p25,p50,p75,p90,max,mean}` | float | s | Slack summaries |
| `state__hist_w1_waiting_count_delta` | float | count delta | Waiting-count delta over window 1 |
| `state__hist_w1_kv_utilization_mean_delta` | float | util delta | KV util delta over window 1 |
| `state__hist_w1_completed_count_delta` | float | count delta | Completion delta over window 1 |
| `state__hist_w1_queued_token_mass_delta` | float | token delta | Queued-mass delta over window 1 |
| `state__hist_w1_new_request_count` | float | count | New requests over window 1 |
| `state__hist_w5_*` | float | as above | Same five history features for window 5 |
| `state__hist_w20_*` | float | as above | Same five history features for window 20 |

Exact expansion of the braced families matches the CSV header order. The
complete 95-name list is:

```text
state__state_time_s
state__state_step
state__waiting_count
state__migrating_ready_count
state__active_count
state__admissible_count
state__total_in_system_count
state__completed_count_pre
state__waiting_prompt_tokens_sum
state__waiting_predicted_remaining_tokens_sum
state__waiting_predicted_total_tokens_sum
state__active_prompt_tokens_sum
state__active_predicted_remaining_tokens_sum
state__active_predicted_total_tokens_sum
state__kv_tokens_used_sum
state__kv_tokens_capacity_sum
state__kv_tokens_free_sum
state__kv_utilization_mean
state__kv_utilization_max
state__waiting_estimated_kv_demand_sum
state__projected_kv_pressure_waiting
state__waiting_large_req_gt_total_free_kv_count
state__active_prefilling_count
state__active_decoding_count
state__queued_prefill_token_mass
state__active_decode_remaining_token_mass
state__prefill_decode_request_ratio
state__prefill_decode_token_ratio
state__urgent_request_count_slack_le_5s
state__urgent_request_fraction_slack_le_5s
state__urgent_token_mass_slack_le_5s
state__slo_violation_count_slack_le_0
state__priority_sum_admissible
state__priority_max_admissible
state__priority_entropy_admissible
state__class_entropy_admissible
state__class_count_admissible
state__high_priority_waiting_prompt_mass
state__waiting_age_s_min
state__waiting_age_s_p25
state__waiting_age_s_p50
state__waiting_age_s_p75
state__waiting_age_s_p90
state__waiting_age_s_max
state__waiting_age_s_mean
state__admissible_prompt_tokens_min
state__admissible_prompt_tokens_p25
state__admissible_prompt_tokens_p50
state__admissible_prompt_tokens_p75
state__admissible_prompt_tokens_p90
state__admissible_prompt_tokens_max
state__admissible_prompt_tokens_mean
state__admissible_predicted_output_tokens_min
state__admissible_predicted_output_tokens_p25
state__admissible_predicted_output_tokens_p50
state__admissible_predicted_output_tokens_p75
state__admissible_predicted_output_tokens_p90
state__admissible_predicted_output_tokens_max
state__admissible_predicted_output_tokens_mean
state__admissible_predicted_total_tokens_min
state__admissible_predicted_total_tokens_p25
state__admissible_predicted_total_tokens_p50
state__admissible_predicted_total_tokens_p75
state__admissible_predicted_total_tokens_p90
state__admissible_predicted_total_tokens_max
state__admissible_predicted_total_tokens_mean
state__active_predicted_remaining_tokens_min
state__active_predicted_remaining_tokens_p25
state__active_predicted_remaining_tokens_p50
state__active_predicted_remaining_tokens_p75
state__active_predicted_remaining_tokens_p90
state__active_predicted_remaining_tokens_max
state__active_predicted_remaining_tokens_mean
state__slack_s_min
state__slack_s_p25
state__slack_s_p50
state__slack_s_p75
state__slack_s_p90
state__slack_s_max
state__slack_s_mean
state__hist_w1_waiting_count_delta
state__hist_w1_kv_utilization_mean_delta
state__hist_w1_completed_count_delta
state__hist_w1_queued_token_mass_delta
state__hist_w1_new_request_count
state__hist_w5_waiting_count_delta
state__hist_w5_kv_utilization_mean_delta
state__hist_w5_completed_count_delta
state__hist_w5_queued_token_mass_delta
state__hist_w5_new_request_count
state__hist_w20_waiting_count_delta
state__hist_w20_kv_utilization_mean_delta
state__hist_w20_completed_count_delta
state__hist_w20_queued_token_mass_delta
state__hist_w20_new_request_count
```

---

## `data/alternative_action_outcomes.csv` (21,854 rows, 36 columns)

One row per unique retained **alternative** full canonical action. SBS reference
actions are not rows in this table (`is_sbs_action` is always `0`).

| Column | Type | Units / values | Meaning / derivation | Warnings |
|---|---|---|---|---|
| `state_id` | string | see above | FK to `state_context.csv` | |
| `scenario_id` | string | | Scenario FK | |
| `fold` | int | 0..4 | Fold label | |
| `step` | int | | Decision step | |
| `sim_time` | float | s | Predecision time | |
| `seed` | int/string | synthetic seed | Scenario seed copied for convenience | Configuration ID, not a timestamp |
| `canonical_action_id` | string | hash/id | ID of the forced alternative action | Part of outcome PK |
| `canonical_action_full` | string | serialized | Forced full canonical action | |
| `canonical_action_admit` | string | serialized | Admit component of forced action | May be empty component |
| `sbs_canonical_action_id` | string | | SBS action ID at the same state | For contrast only |
| `sbs_canonical_action_full` | string | | SBS full action | |
| `is_sbs_action` | int | always `0` | Marks alternative-only table | Never true here |
| `policies_generating_action` | string | policy list | Which policies proposed this action | |
| `n_policies_generating_action` | int | count | How many policies proposed it | |
| `n_distinct_canonical_actions` | int | count | Distinct actions at the state | |
| `policy_signature` | string | signature | Compact policy-to-action signature | Descriptive |
| `q_sbs_anwg` | float | dimensionless | Terminal ANWG after alternative + SBS continuation | Higher better |
| `q_sbs_soft` | float | dimensionless | Terminal soft goodput | Higher better |
| `q_sbs_wcg` | float | dimensionless | Terminal weighted completion metric | Higher better |
| `q_sbs_wmt` | float | simulator time units | Terminal weighted mean time | Lower better |
| `q_sbs_wnt` | float | normalized time | Terminal weighted normalized time | Lower better |
| `q0_sbs_anwg` | float | dimensionless | Same-state SBS reference ANWG | Included so contrast needs no external table |
| `q0_sbs_soft` | float | | SBS reference soft | |
| `q0_sbs_wcg` | float | | SBS reference WCG | |
| `q0_sbs_wmt` | float | | SBS reference WMT | |
| `q0_sbs_wnt` | float | | SBS reference WNT | |
| `a_sbs_anwg` | float | | `q_sbs_anwg - q0_sbs_anwg` | Primary effect |
| `a_sbs_soft` | float | | `q_sbs_soft - q0_sbs_soft` | |
| `a_sbs_wcg` | float | | `q_sbs_wcg - q0_sbs_wcg` | |
| `a_sbs_wmt_improvement` | float | | `q0_sbs_wmt - q_sbs_wmt` | Positive ⇒ lower (better) time than SBS |
| `a_sbs_wnt_improvement` | float | | `q0_sbs_wnt - q_sbs_wnt` | Positive ⇒ lower (better) normalized time |
| `num_completed` | int | count | Completions in the labeled continuation | |
| `num_dropped` | int | count | Drops in the labeled continuation | |
| `sim_duration` | float | s | Simulator duration associated with the branch | |
| `extra_steps` | int | count | Extra steps relative to reference branch bookkeeping | Interpret with design docs |
| `live_fingerprint_unchanged` | bool | always True here | Continuation integrity flag | All released rows pass |

### Effect identities (mandatory)

```text
a_sbs_anwg = q_sbs_anwg - q0_sbs_anwg
a_sbs_soft = q_sbs_soft - q0_sbs_soft
a_sbs_wcg  = q_sbs_wcg  - q0_sbs_wcg
a_sbs_wmt_improvement = q0_sbs_wmt - q_sbs_wmt
a_sbs_wnt_improvement = q0_sbs_wnt - q_sbs_wnt
```

ANWG sign classes at threshold `1e-12`:

* positive: `a_sbs_anwg > 1e-12`
* negative: `a_sbs_anwg < -1e-12`
* null: `abs(a_sbs_anwg) <= 1e-12`

Expected totals in this release: **2573 / 3624 / 15657**.

---

## `metadata/scenario_metadata.csv` (236 rows, 33 columns)

Synthetic generator parameters for represented scenarios. These are configuration
descriptors, not observations of named users or production traffic.

| Column | Type | Units / meaning |
|---|---|---|
| `scenario_id` | string | Primary key |
| `seed` | int | Synthetic scenario seed (unique in this table) |
| `num_requests` | int | Generated request count |
| `offered_load` | float | Offered load parameter |
| `burstiness` | float | Burstiness parameter |
| `long_fraction` | float | Long-request fraction |
| `prompt_median` | float | Median prompt length parameter |
| `output_median` | float | Median output length parameter |
| `tenant_weight_skew` | float | Tenant weight skew |
| `class_share_skew` | float | Class share skew |
| `slo_tightness` | float | SLO tightness |
| `prediction_noise_sigma` | float | Prediction-noise scale |
| `kv_pressure_target` | float | KV pressure target |
| `late_pressure` | float | Late-phase pressure |
| `late_fraction` | float | Late fraction |
| `late_phase` | categorical/numeric | Late-phase configuration |
| `max_active_sequences` | int | Max active sequences |
| `step_token_budget` | int | Per-step token budget |
| `max_kv_tokens` | int | Max KV tokens |
| `fairness_pressure` | float | Fairness pressure |
| `service_heterogeneity` | float | Service heterogeneity |
| `prefill_decode_pressure` | float | Prefill/decode pressure |
| `kv_pressure` | float | KV pressure |
| `urgency_pressure` | float | Urgency pressure |
| `burst_pressure` | float | Burst pressure |
| `high_fairness_pressure` | 0/1 | Elevated-mechanism flag |
| `high_service_heterogeneity` | 0/1 | Elevated-mechanism flag |
| `high_prefill_decode_pressure` | 0/1 | Elevated-mechanism flag |
| `high_kv_pressure` | 0/1 | Elevated-mechanism flag |
| `high_urgency_pressure` | 0/1 | Elevated-mechanism flag |
| `high_burst_pressure` | 0/1 | Elevated-mechanism flag |
| `n_elevated_mechanisms` | int | Count of elevated mechanisms |
| `fold` | int | Frozen fold assignment `{0..4}` |

Fold counts in this release: fold0=50, fold1=48, fold2=48, fold3=45, fold4=45.

---

## `metadata/policy_metadata.csv` (6 rows, 3 columns)

| Column | Type | Meaning |
|---|---|---|
| `policy_id` | string | Policy name |
| `role` | string | `scan trajectory / SBS reference` or `candidate policy` |
| `continuation` | string | Notes on scan execution vs terminal labeling |

This table is descriptive metadata only and contains no additional outcomes.
