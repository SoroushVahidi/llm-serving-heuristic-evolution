# Policy Library

Current source-of-truth for deployable scheduler policies as of 2026-07-22.

## Registry Contract

- Historical deployable registry: `BASELINE_NAMES` in `src/llmserveopt/policies/registry.py`
- Expanded deployable registry: `POLICY_LIBRARY_V2_NAMES`
- Policy Library v2 additions: `POLICY_LIBRARY_V2_NEW_NAMES`
- Oracle policies remain excluded from deployable selector/composition action spaces.

Current counts:

- `POLICY_LIBRARY_V1_COUNT = 20`
- `POLICY_LIBRARY_V2_NEW_COUNT = 7`
- `POLICY_LIBRARY_V2_COUNT = 27`

The 7 Policy Library v2 policies are registered through `make_policy_library_v2()` but are intentionally not appended to `BASELINE_NAMES`, because historical Selector v2 tests and documentation pin that library at 20 policies.

## Policy Library v2 Additions

| Policy | Behavioral family | Main causal inputs | Literature status | Composition compatibility | Deployability |
| --- | --- | --- | --- | --- | --- |
| `sola_style_state_aware` | state-aware priority | queue pressure, KV pressure, laxity, priority, predicted service | simulator-level approximation, not a SOLA reproduction | priority, placement penalty | deployable |
| `slai_style_phase_aware` | phase-aware priority | active prefill/decode counts, prompt/output mix, predicted service, laxity | simulator-level approximation, not an SLAI reproduction | priority, phase-control proxy | deployable |
| `flow_control_stability` | overload admission throttling | system pressure, recent arrival-rate slope, waiting queue, predicted service | faithful to representable flow-control behavior, not a paper reproduction | admission, priority | deployable; stateful budget |
| `kv_constrained_online` | KV-pressure guard | predicted KV footprint, current KV utilization, laxity | faithful to current simulator KV-capacity semantics | admission, KV guard, placement | deployable |
| `adaptive_chunked_prefill` | prefill pressure control | prompt length, current pressure, active long-prefill count, laxity | approximation; true chunk-size actions are unsupported | admission, prefill-control proxy, priority | deployable |
| `aging_priority` | fairness/aging | waiting time, priority, predicted service, laxity | library-native fairness primitive | fairness, priority | deployable |
| `weighted_fair_share` | class-level fair share | observable `class_id`, active/waiting class counts, priority, predicted service | approximation of tenant fairness using available class labels | fairness, priority | deployable |

## Historical Deployable Policy Families

| Family | Policies | Notes |
| --- | --- | --- |
| FIFO/random feasibility | `fifo`, `random_feasible` | Simple control baselines. |
| Deadline/SLO priority | `edf`, `least_laxity_first`, `slo_slack_score`, `scorpio_style_slo_guard` | SCORPIO is the strongest SLO-aware historical deployable and a key parent for composition/synthesis. |
| Length/service-time priority | `shortest_output_first`, `shortest_prompt_first`, `estimated_service_time_first`, `weighted_shortest_processing` | WSP remains the strongest fixed policy in several held-out evaluations. |
| Packing/load balancing | `greedy_token_fill`, `least_loaded`, `multi_bin_batching`, `first_fit`, `best_fit` | Resource-placement and batching-style baselines. |
| Serving-system approximations | `orca_style`, `vllm_style_token_budget`, `sarathi_style`, `splitfuse_style` | Faithful only to simulator-supported abstractions; not full serving-system reproductions. |
| Explicit admission | `admission_control` | Deployable admission-control baseline. |

## Unsupported or Deferred Families

These are not faithfully representable without simulator/action-space extensions:

- prefix-cache and cache-reuse-aware scheduling
- cache-loading-aware scheduling
- disaggregated prefill/decode routing
- request splitting or micro-request scheduling
- heterogeneous GPU affinity/routing
- exact action-level chunked prefill
- durable tenant credit accounting beyond observable class-level approximations

## Validation

Focused Policy Library v2 validation exists in `tests/test_policy_library_v2.py` and passed under SLURM job `1118782`. Query 3 final validation also passed the focused policy-library, composition, and structural-synthesis tests plus the full pytest suite under SLURM job `1120358`.

## Current Scientific Status

Policy Library v2 implementation is present and test-covered.

Completed evidence:

- Synthetic/frontier V2 expansion was real but modest.
- The real-trace OOD V2 library audit showed strong oracle-envelope expansion:
  V1 oracle ANWG `0.251666`, V2 oracle ANWG `0.260571`, absolute gain
  `0.008904`, relative gain about `3.54%`, CI `[0.008191, 0.009646]`.
- Several V2 policies contribute genuine competence regions on real-OOD
  windows.
- SwissAI and TraceLab did not show strict V2 marginal oracle gain, despite raw
  KV/cache, long-context, prefix-reuse, and agentic novelty. This is now
  interpreted as objective/simulator saturation evidence, not as proof that the
  policy library is useless.
- SLO/deadline augmentation produced useful incremental support for
  EDF/SCORPIO/admission/laxity-style behavior, but remains synthetic
  regime-probing data.

Current implication: policy-library coverage is no longer the primary
bottleneck. The next issue is whether the simulator/objective creates enough
pressure and reward separation for these policies to be learned, combined, and
evaluated fairly.
