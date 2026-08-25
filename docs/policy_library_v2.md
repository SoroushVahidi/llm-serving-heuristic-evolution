# Policy Library v2 Audit

Policy Library v2 keeps the historical `BASELINE_NAMES` registry frozen for
Selector v2 reproducibility and adds a separate `POLICY_LIBRARY_V2_NAMES`
snapshot for expanded-library experiments.

## Implement Now

| policy | inspiration | faithful simulator behavior | known deviation |
| --- | --- | --- | --- |
| `sola_style_state_aware` | state-aware online scheduling | ranks by causal service proxy, laxity, priority, queue pressure, and KV pressure | not a paper reproduction; no learned state policy |
| `slai_style_phase_aware` | phase-aware prefill/decode control | uses observable active prefill/decode counts to avoid worsening the dominant phase bottleneck | no separate placement between prefill and decode pools |
| `flow_control_stability` | overload-stable flow control | throttles admissions with a deterministic causal budget under load/arrival-growth pressure | no closed-loop controller over real latency measurements |
| `kv_constrained_online` | KV-aware online scheduling | preserves a KV reserve except for urgent requests | no prefix/cache reuse model |
| `adaptive_chunked_prefill` | chunked prefill | limits concurrent long-prompt admissions under pressure and interleaves short work | cannot set per-request chunk sizes through `Action` |
| `aging_priority` | starvation avoidance | monotonic waiting-time priority boost | no tenant-level fairness |
| `weighted_fair_share` | weighted fair sharing | uses observable `class_id` as a group label for class-level deficit scheduling | `class_id` is not a tenant/session identity |

## Implement After Modest Extension

| family | required simulator extension |
| --- | --- |
| `gate_and_route_phase_control` | stable candidate action set for phase routing in mixed colocated/disaggregated mode |
| `decode_deadline_guard` | richer active decode remaining-work/laxity state |
| `prefill_budget_controller` | action-level prefill token-budget or chunk-size control |
| `phase_load_balance` | multiple phase-specific queues or routing actions |

## Defer

| family | reason |
| --- | --- |
| cache/prefix-reuse-aware policies | no prefix/cache identity or reuse semantics in the monolithic simulator |
| cache-loading-aware policies | no cache-loading/memory-transfer action |
| disaggregated prefill/decode routing | faithful baselines exist in external topology, but not comparable as monolithic deployable selector actions |
| request splitting/micro-request scheduling | `Action` admits whole requests only |
| heterogeneous-GPU affinity/routing | current clean selector candidate topology assumes homogeneous monolithic GPUs |
| live multi-tenant fairness | no tenant identity beyond coarse `class_id` |

## Behavioral Coverage

The final SLURM report writes pre/post coverage matrices under the experiment
root. The intended new dimensions are overload stability, phase awareness,
KV pressure awareness, aging/fairness, and chunked-prefill-style admission
control. These are distinct from simply renaming WSP, EDF, or SCORPIO variants.
