# Policy Composition Readiness Audit

> **SUPERSEDED FOR CURRENT STATUS.**
> See [`docs/current/RESUME_HERE.md`](RESUME_HERE.md) for authoritative current state.
> Composition was subsequently formally `COMPOSITION_DEMOTED` (`docs/audits/reassessment_composition_hypothesis_20260817.md`, commit `dc5757b`).

> **Pause addendum 2026-07-25.** The repaired load-discrimination pilot (`PARTIALLY_READY`) does **not** justify reopening composition or synthesis work. Native composition pilot remains `NO_GO` with verified-readable artifacts; structural synthesis remains empirically `NOT_READY`. Prioritize simulator/load discrimination on natural evidence first.


> Historical readiness snapshot. This document records the static composition
> audit performed before later Wulver workflows completed. For the current
> scientific decision, read `PROJECT_STATUS.md` and
> `COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md`: naive/native composition did not
> clear the decision bar, module-credit learning remains weak, and broad
> composition/synthesis is gated on simulator calibration.

This audit inspects the current 27-policy scheduling library and the simulator interfaces to determine whether new schedulers can be constructed safely from reusable behavioral primitives. It does not launch a new simulation sweep and does not modify existing policy behavior.

## Scope

Branch: `wulver-policy-composition-readiness`

Inspected policy registry:

- 20 historical deployable policies in `BASELINE_NAMES`
- 7 Policy Library v2 policies in `POLICY_LIBRARY_V2_NEW_NAMES`

Machine-readable artifacts:

- `docs/current/policy_component_matrix.json`
- `docs/current/composable_primitives.json`
- `docs/current/composition_operators.json`
- `docs/current/policy_complementarity.json`

Current running workflows were not awaited or modified:

- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z`

## Current Policy Library

The 27 policies are not just name variants. They cover several genuine behavioral families:

| Family | Policies | Main behavior |
| --- | --- | --- |
| Arrival order / simple placement | `fifo`, `first_fit` | FIFO ordering with simple feasible placement |
| Deadline / laxity / SLO | `edf`, `least_laxity_first`, `slo_slack_score`, `admission_control`, `scorpio_style_slo_guard` | Deadline ranking, laxity ranking, SLO-aware admission |
| Short-work / service-time | `shortest_output_first`, `shortest_prompt_first`, `weighted_shortest_processing`, `estimated_service_time_first` | Short predicted work and priority-weighted work |
| KV and resource packing | `greedy_token_fill`, `best_fit`, `vllm_style_token_budget`, `kv_constrained_online` | KV feasibility, block budgeting, packing, reserve guards |
| Batch shape / continuous batching | `orca_style`, `multi_bin_batching`, `vllm_style_token_budget` | Priority batching and length-bin grouping |
| Prefill / phase interference | `sarathi_style`, `splitfuse_style`, `slai_style_phase_aware`, `adaptive_chunked_prefill` | Prefill budgets, decode pressure, phase-aware penalties |
| Overload stability | `scorpio_style_slo_guard`, `flow_control_stability`, `adaptive_chunked_prefill` | Admission throttling and queue-growth control |
| Fairness / aging | `aging_priority`, `weighted_fair_share`, `slo_slack_score` | Age bonus, class-level active-share deficit, wait-sensitive SLO urgency |
| State-aware composite | `sola_style_state_aware`, `scorpio_style_slo_guard`, `flow_control_stability` | Causal state pressure, KV pressure, SLO pressure |
| Stochastic baseline | `random_feasible` | Seeded random feasible request ordering |

The library still lacks faithful implementations of cache/prefix reuse, cache loading, true disaggregated prefill/decode routing, request splitting, heterogeneous hardware routing, and normal deployable preemption. Some simulator types expose fields such as GPU role and action fields such as preempt/migrate, but the current 27-policy library does not use them as faithful production mechanisms.

## Primitive Decomposition

The current policies decompose into reusable primitives such as:

- `arrival_order`
- `shortest_remaining_work`
- `shortest_decode_work`
- `shortest_prefill_work`
- `earliest_deadline`
- `minimum_laxity`
- `urgency_score`
- `priority_weight`
- `age_bonus`
- `slo_violation_risk`
- `admission_threshold`
- `load_shedding`
- `queue_growth_control`
- `decode_deadline_guard`
- `prefill_budget`
- `phase_balance`
- `kv_pressure_penalty`
- `least_loaded_placement`
- `kv_best_fit_placement`
- `round_robin_feasible_placement`
- `length_bin_batching`
- `fair_share_deficit`

Most of these are strictly causal under the current `ObservableState` and `ObservableRequest` interfaces because they use only arrival time, current time, predicted lengths, priority, class ID, active request metadata, phase counts, and current KV state. They do not require actual future output lengths, oracle rewards, or future arrivals.

## Typed Composition Representation

A practical typed representation should separate modules rather than treat every policy as an opaque score:

```text
Scheduler(
  admission = AdmissionRule,
  priority = PriorityRule,
  batching = BatchingRule,
  phase_control = PhaseControlRule,
  kv_control = KVRule,
  fairness = FairnessRule,
  placement = PlacementRule
)
```

Recommended module types:

| Module | Input | Output | Current support |
| --- | --- | --- | --- |
| `AdmissionRule` | `ObservableState`, `ObservableRequest` | admit / skip / defer | Yes |
| `PriorityRule` | `ObservableState`, request set | ranked list or normalized score | Yes with small policy API extension |
| `BatchingRule` | state, ranked candidates | subset or batch groups | Partial |
| `PhaseControlRule` | state, candidate request | penalty or admission budget | Partial |
| `KVRule` | state, GPU state, request | penalty, constraint, or placement preference | Yes |
| `FairnessRule` | state, request, optional local history | priority bonus or deficit score | Partial |
| `PlacementRule` | state, request | GPU ID or infeasible | Yes |

The simulator can execute admission, ranking, KV guards, and placement today through ordinary `Action(admit=...)` output. It cannot faithfully execute cache-aware routing, request splitting, or true disaggregated scheduling without larger architectural work.

## Closure Under Composition

Weighted raw-score composition is not safe as the default. Existing policies expose incomparable quantities: some produce private scores, some produce sorted request lists, some only output direct actions, some implement binary admission filters, and placement policies do not define request utility scores.

Safer alternatives:

1. Weighted Borda rank aggregation over the current waiting set.
2. Reciprocal-rank fusion.
3. Conditional composition on causal state, such as overload or KV pressure.
4. Component-wise composition: SCORPIO-style admission plus WSP priority plus KV guard plus phase/prefill guard plus deterministic placement.

The safest common representation is normalized deterministic rank, with admission and placement kept as separate typed modules. Raw-score addition should be avoided unless each primitive declares scale, sign, monotonicity, and normalization.

## Similarity And Complementarity

At the time of this snapshot, the full 27-policy Policy Library v2 sweep was
still running, so this audit did not claim final quantitative pairwise reward
correlations. Later completed evidence should be read from
`PROJECT_STATUS.md` and `EXPERIMENT_INDEX.md`. The code-level evidence from
this snapshot still supports several complementarity hypotheses:

- WSP and SCORPIO remain the most important parent pair because previous OOD failures concentrated around WSP-vs-SCORPIO routing.
- KV-aware and prefill/phase-aware v2 policies add behaviors that historical WSP/SCORPIO did not isolate cleanly.
- Aging/fair-share policies add a separate fairness dimension that can reduce starvation risk in composed schedulers.
- Some policies are likely redundant as standalone deployable actions but useful as components, especially FIFO placement variants and short-work variants.

Likely redundant clusters:

- `fifo`, `first_fit`, `best_fit`: same FIFO request order, mostly placement differences.
- `shortest_output_first`, `estimated_service_time_first`, `weighted_shortest_processing`, `vllm_style_token_budget`: overlapping short-work bias.
- `sarathi_style`, `splitfuse_style`, `adaptive_chunked_prefill`: prefill-budget approximations without true chunk-size actions.
- `greedy_token_fill`, `best_fit`, `kv_constrained_online`: KV/resource-pressure family.

Candidate parent sets:

- `weighted_shortest_processing` + `scorpio_style_slo_guard` + `adaptive_chunked_prefill`
- `weighted_shortest_processing` + `kv_constrained_online` + `aging_priority`
- `estimated_service_time_first` + `flow_control_stability` + `weighted_fair_share`
- `scorpio_style_slo_guard` + `slai_style_phase_aware` + `best_fit`
- `sola_style_state_aware` + `weighted_shortest_processing` + `adaptive_chunked_prefill`

## Minimal Simulator Changes

No simulator core change is needed for a first rank-composition experiment if composition is implemented as an ordinary policy that returns `Action(admit=...)`.

Small extensions are recommended:

- Add optional component methods such as `rank_requests`, `admission_filter`, `choose_gpu`, and `component_metadata`.
- Add immutable state snapshot/helper utilities so multiple member policies can be queried without stateful side effects.
- Add deterministic hysteresis/smoothing for conditional state-dependent weights.
- Record composed-policy provenance in manifests.

Major extensions are required for:

- prefix/cache reuse;
- cache loading;
- disaggregated prefill/decode routing;
- request splitting and micro-request scheduling;
- heterogeneous GPU affinity/routing;
- true action-level chunked prefill.

## First Falsifiable Composition Experiment

The smallest decisive experiment should compare:

1. Best fixed policy, expected baseline `weighted_shortest_processing`.
2. Existing discrete dynamic selector, expected baseline RF per-policy reward regressor.
3. Static rank ensemble over `weighted_shortest_processing`, `scorpio_style_slo_guard`, `estimated_service_time_first`, `kv_constrained_online`, and `aging_priority`.
4. Contextual state-dependent rank ensemble using causal load/KV/SLO/phase features.
5. Component-wise composition using SCORPIO-style admission, WSP or ESTF priority, KV reserve guard, adaptive prefill/phase guard, aging tiebreak, and best-fit or least-loaded placement.

Use leakage-safe group-aware train/validation/test splits. Select all thresholds and weights on train/validation or robustness-development folds only. Evaluate frozen candidates on ID, temporal OOD, cross-source OOD, and final OOD. Metrics should include arrival-normalized weighted gain, completion fraction, completed-request quality, mean/p95/worst oracle regret, meaningful-window subset metrics, gap closed versus best fixed, and paired bootstrap confidence intervals by group/window.

The falsifiable question is:

Does composition outperform both selecting one existing policy and the best discrete learned selector, without tuning on final OOD labels?

## Final Answers

COMPOSITION_READINESS = READY_WITH_SMALL_EXTENSIONS

POLICIES_ANALYZED = 27

DISTINCT_BEHAVIORAL_FAMILIES = 10

COMPOSABLE_PRIMITIVES_FOUND = 25

SAFEST_COMPOSITION_OPERATOR = weighted_borda_rank_aggregation_with_separate_admission_and_placement

MOST_EXPRESSIVE_SUPPORTED_OPERATOR = conditional_component_wise_composition

RAW_SCORE_COMPOSITION_VALID = PARTIALLY

RANK_COMPOSITION_SUPPORTED = YES

COMPONENT_WISE_COMPOSITION_SUPPORTED = PARTIALLY

DYNAMIC_STATE_DEPENDENT_WEIGHTS_SUPPORTED = WITH_EXTENSION

MINIMAL_SIMULATOR_EXTENSION = optional_policy_component_api_for_rank_requests_admission_filter_choose_gpu_and_component_metadata

BEST_CANDIDATE_PARENT_POLICIES = weighted_shortest_processing + scorpio_style_slo_guard + adaptive_chunked_prefill; weighted_shortest_processing + kv_constrained_online + aging_priority; scorpio_style_slo_guard + slai_style_phase_aware + best_fit

MOST_REDUNDANT_POLICIES = fifo/first_fit/best_fit placement variants; shortest_output_first/estimated_service_time_first/weighted_shortest_processing/vllm_style_token_budget short-work variants; sarathi_style/splitfuse_style/adaptive_chunked_prefill prefill-budget approximations

FIRST_COMPOSITION_EXPERIMENT = compare best fixed WSP, discrete RF selector, static rank ensemble, contextual rank ensemble, and component-wise SCORPIO-admission + WSP-priority + KV-guard + prefill/phase-guard + aging-tiebreak scheduler on leakage-safe group splits with untouched OOD evaluation

MAJOR_UNSUPPORTED_CAPABILITIES = cache_prefix_reuse, cache_loading, disaggregated_prefill_decode_routing, request_splitting, heterogeneous_gpu_routing, true_action_level_chunked_prefill

RECOMMENDED_NEXT_ACTION = historical snapshot recommendation superseded; current next action is simulator calibration and discriminative-power validation before further composition work
