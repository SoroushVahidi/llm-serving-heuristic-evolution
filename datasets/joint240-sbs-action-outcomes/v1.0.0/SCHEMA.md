# Schema and data dictionary

`state_context.csv` has one row per retained labeled SBS disagreement state.
It contains scenario/fold/step identity, the SBS full and admission actions,
the five-policy disagreement summary, and the 95 numeric predecision
`state__*` features. Outcome columns are deliberately absent.

`alternative_action_outcomes.csv` has one row per unique retained alternative
full canonical action. `q_sbs_*` are terminal utilities after the alternative
action and fixed SBS continuation; `q0_sbs_*` are the SBS reference values
retained only to define the within-state contrast; `a_sbs_*` are the signed
alternative-minus-reference effects. Negative and numerically null effects
are retained. `is_sbs_action` is always false in this table by construction.

`scenario_metadata.csv` contains the frozen synthetic generator parameters and
the five-fold assignment. `policy_metadata.csv` identifies the SBS trajectory
policy and the five candidate policies; it is descriptive metadata, not a
new measurement.

All identifiers are synthetic `joint_mm_*` scenarios and
`joint240_sbs::<scenario>::<step>` states. `sim_time` is relative synthetic
simulator time, not an upstream request timestamp. `seed` is a synthetic
scenario seed. ANWG means arrival-normalized weighted goodput.

## Key and common-column contract

| Column family | Type/nullability | Key role and interpretation |
|---|---|---|
| `state_id`, `scenario_id`, `canonical_action_id`, `policy_id` | string, non-null | Synthetic identifiers; state primary key is `state_id`; outcome key is `(state_id, canonical_action_id)`; scenario foreign key is `scenario_id`. |
| `fold`, `step`, counts, token masses | integer, non-null | Synthetic fold/decision position/counts; not independent-replicate identifiers. |
| `sim_time`, metric values, scenario parameters | finite float, non-null | Simulator units; `sim_time` is relative seconds, not a production timestamp. |
| action strings, policy lists, role/continuation | string, non-null | Canonical serialized action or metadata; do not parse as raw requests. |
| boolean flags | `0`/`1` or textual boolean, non-null | Canonical action/disagreement/invariance flags. |

## `state_context.csv`

The first 12 columns are identifiers and disagreement summaries. The 95
`state__*` columns are canonical numeric predecision features. Every state
feature is non-null in this release, is measured before the candidate action,
and is not a terminal outcome. Suffixes define the statistic and units:
`_count` is a request count, `_tokens_sum`/`_token_mass` is a token count,
`_s` is seconds, `_ratio`/`_fraction`/`_entropy`/`_utilization` is
dimensionless, and `min/p25/p50/p75/p90/max/mean/delta` is the indicated
summary statistic. `class` and `priority` quantities are simulator metadata,
not private user attributes.

The complete feature list is:

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

All listed features are `float64`-compatible finite numeric values. The
feature names are the canonical simulator names; no additional semantic
interpretation beyond the encoded quantity/statistic should be inferred.

## `alternative_action_outcomes.csv`

All columns are non-null except where the canonical CSV represents a
serialized empty action component. Identifiers and action strings are strings;
`is_sbs_action` is always `0`; `q_*`, `q0_*`, and `a_*` are finite floats;
`num_completed`, `num_dropped`, and `extra_steps` are integer-valued; and
`sim_duration` is simulator seconds. `live_fingerprint_unchanged` is the
continuation integrity flag.

`q_sbs_anwg`, `q_sbs_soft`, `q_sbs_wcg`, `q_sbs_wmt`, and `q_sbs_wnt` are
terminal metrics after the forced action and SBS continuation. The matching
`q0_sbs_*` columns are SBS-reference metrics for the same state. The matching
`a_sbs_*` columns are signed differences:
`a_sbs_metric = q_sbs_metric - q0_sbs_metric`. For WMT and WNT, the canonical
field names end in `_improvement`; positive values follow the stored
alternative-minus-reference convention. ANWG, soft goodput, and weighted
completion are dimensionless; weighted mean time and weighted normalized time
use simulator seconds/normalized simulator-time units. Use absolute tolerance
`1e-10` for the stored identity and `1e-12` for ANWG sign classification.

## `scenario_metadata.csv`

`scenario_id` and `fold` are identifiers; `seed`, `num_requests`, mechanism
parameters, and pressure summaries are finite numeric synthetic generator
metadata. Count and token-like fields are simulator counts; load, fractions,
skews, pressure, and tightness fields are dimensionless generator parameters.
`late_phase`, `max_active_sequences`, `step_token_budget`, and `max_kv_tokens`
retain the generator's categorical/integer configuration values. These are
synthetic scenario descriptors, not observations of named users or production
traffic.

## `policy_metadata.csv`

`policy_id`, `role`, and `continuation` are non-null strings. The six rows
describe the SBS policy, five candidates, and the fixed-SBS continuation.
This table is metadata only and contains no additional experiment outcomes.
