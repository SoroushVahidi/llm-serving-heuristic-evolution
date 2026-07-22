# Policy Genome Coverage Audit

**Date:** 2026-07-22
**Branch:** `wulver-selector-v2-and-composition-integrated`
**Scope:** every one of the 27 `policies.registry.POLICY_LIBRARY_V2_NAMES` deployable policies, audited against their actual source implementation, and (where faithful) mapped into `policies.structural_synthesis.map_policy_to_genome`.

## Method

Every policy's `select_action` implementation was read directly (not inferred from names or docs). Two structural facts about `SchedulerGenomeV1`/the heuristic DSL constrain what can be faithfully represented, discovered during this audit:

1. **The DSL's `HeuristicPolicy` runner (`heuristics/policy.py`) always uses the same placement strategy** -- scan `state.gpu_states` in the order given, take the first GPU whose capacity check passes. A genome can shape *admission* (`admission_condition`) and *ranking* (`request_score` + one `tie_breaker`), but **cannot express a custom GPU-placement strategy** (KV-tightest-fit, least-active-load, bin-affinity, etc.). Any policy whose defining behavior *is* its placement strategy cannot be faithfully mapped, no matter how the ranking is encoded.
2. **The DSL is fully deterministic** -- there is no RNG primitive in `ALLOWED_OPS`/`ALLOWED_VARS`. A stochastic policy cannot be expressed at all.
3. **`ALLOWED_VARS` (the causal variable whitelist) is fixed** and does not include per-class aggregate counts or prefill/decode phase-share aggregates. Policies whose defining behavior depends on those cannot be faithfully mapped without a DSL-variable-whitelist change, which is out of this task's scope (SchedulerGenomeV1's typed module taxonomy, not the underlying DSL's variable whitelist).

**Conclusion on taxonomy extension (task step 3):** no new `SchedulerGenomeV1` dataclass field was needed. The existing 6-module taxonomy (`admission_rule`, `priority_rule`, `prefill_rule`, `kv_guard`, `fairness_rule`, `regime_conditions`) was already sufficiently compositional for every policy classified `FULLY_MAPPABLE`/`MAPPABLE_WITH_GENOME_EXTENSION` below; the real gaps are (1) the DSL's fixed placement strategy, (2) its determinism, and (3) its variable whitelist -- none of which "genome taxonomy" extension can fix. This is a deliberate, grounded finding, not an oversight.

## Per-Policy Audit

| Policy | Source | Admission | Priority/ranking | Deadline/laxity | Prefill | KV | Fairness | Regime | Status |
|---|---|---|---|---|---|---|---|---|---|
| `fifo` | `fifo.py` | none | arrival order | none | none | none | none | none | **EXACT** |
| `edf` | `edf.py` | none | earliest deadline | primary | none | none | none | none | **EXACT** (pre-existing) |
| `shortest_output_first` | `shortest_output_first.py` | none | ascending predicted_output_tokens | none | none | none | none | none | **EXACT** |
| `shortest_prompt_first` | `shortest_prompt_first.py` | none | ascending prompt_tokens | none | none | none | none | none | **EXACT** |
| `first_fit` | `first_fit.py` | none | arrival/id order | none | none | none | none | none | **EXACT** (placement matches DSL's default scan iff `gpu_states` is gpu_id-ordered, matching `first_fit.py`'s own explicit sort) |
| `orca_style` | `orca_style.py` | none | priority desc, FCFS within class | none | none | none | none | none | **EXACT** (tie-break uses request_id not arrival_time; equivalent since IDs are arrival-ordered) |
| `slo_slack_score` | `slo_slack_score.py` | none | urgency(slack) + priority | primary | none | none | none | none | **EXACT** -- `weighted_sum` matches `scoring.urgency_score` exactly |
| `weighted_shortest_processing` | `weighted_shortest_processing.py` | none | service_proxy / priority | none | none | none | none | none | **EXACT** (pre-existing) |
| `least_laxity_first` | `least_laxity_first.py` | none | ascending laxity | primary | none | none | none | none | **EXACT** primary score; 3rd/4th tie-break keys not reproducible (DSL: one tie-breaker only) |
| `estimated_service_time_first` | `estimated_service_time_first.py` | none | ascending est. service time | tiebreak only | implicit (prompt term) | none | none | none | **EXACT** primary score; same tie-break limitation |
| `admission_control` | `admission_control.py` | laxity >= -threshold | laxity, priority, ... | primary | none | none | none | none | **APPROXIMATE** -- the registry-default instance (`laxity_threshold=inf`) makes admission a no-op; genome represents the mechanism with the documented `threshold=0.0` case instead |
| `scorpio_style_slo_guard` | `scorpio_style_slo_guard.py` | slack guard | urgency+priority+age(-decode penalty under guard) | primary | none | aggregate KV threshold | none | **kv_pressure_guard regime** | **APPROXIMATE** (pre-existing, upgraded) -- stateful admission-budget refill/consume and per-GPU decode pressure are not representable; `regime_conditions` now captures the guard_active branching |
| `sola_style_state_aware` | `sola_style_state_aware.py` | none | 1.8*priority+1.2*urgency+0.04*wait | secondary | none | load-proxy kv_guard | none | none | **APPROXIMATE** (pre-existing, upgraded to real coefficients) -- `system_pressure`'s queue-length term and the load/KV-scaled service penalty are approximated, not exact |
| `slai_style_phase_aware` | `slai_style_phase_aware.py` | none | priority-2.5*phase_penalty-service-laxity | secondary | phase-dependent | none | none | none | **UNSUPPORTED** -- phase_penalty depends on `prefilling_count`/`decoding_count` phase shares, not in `ALLOWED_VARS` |
| `flow_control_stability` | `flow_control_stability.py` | stateful budget | laxity, WSPT-over-priority | primary | none | none | none | none | **APPROXIMATE** (new) -- the stateful refilling admission budget (this policy's namesake "flow control") and arrival-slope overload trigger are not representable; only the laxity ranking is captured |
| `kv_constrained_online` | `kv_constrained_online.py` | KV reserve unless urgent | urgency per KV cost | secondary | none | aggregate KV threshold | none | none | **APPROXIMATE** (pre-existing, upgraded to real `target_kv_utilization=0.82`) -- per-GPU post-placement KV reserve and urgent-laxity override not exact |
| `adaptive_chunked_prefill` | `adaptive_chunked_prefill.py` | pressure-dependent long-prompt cap | laxity, output tiebreak | secondary | long-prompt penalty proxy | none | none | none | **APPROXIMATE** (pre-existing, upgraded to real thresholds 2048/0.55) -- concurrent-long-prefill running count is not representable |
| `aging_priority` | `aging_priority.py` | none | (priority+0.15*wait)/service + 0.2/laxity | secondary | none | none | age bonus | none | **APPROXIMATE** (pre-existing, upgraded to real coefficients 0.15/0.2 -- previously used unrelated placeholder coefficients) |
| `weighted_fair_share` | `weighted_fair_share.py` | none | deficit*priority/service | none | none | none | class-demand deficit | none | **APPROXIMATE** (new) -- the class-fairness deficit term (this policy's defining mechanism) needs per-class aggregate counts, not in `ALLOWED_VARS`; genome captures only the residual priority/service ranking |
| `multi_bin_batching` | `multi_bin_batching.py` | none | bin(predicted_output_tokens) ascending | none | none | none | none | none | **APPROXIMATE** (new) -- bin-affinity GPU placement (the defining length-mismatch-reduction mechanism) is not representable; only a bucketed ranking approximation via nested `if_then_else` is captured |
| `greedy_token_fill` | `greedy_token_fill.py` | none | FIFO | none | none | **placement-defined** | none | none | **UNSUPPORTED** -- defining behavior is KV-remaining-capacity GPU placement |
| `least_loaded` | `least_loaded.py` | none | FIFO | none | none | **placement-defined** | none | none | **UNSUPPORTED** -- defining behavior is active-sequence-count GPU placement |
| `best_fit` | `best_fit.py` | none | arrival/id order | none | none | **placement-defined** | none | none | **UNSUPPORTED** -- defining behavior is tightest-remaining-KV GPU placement |
| `random_feasible` | `random_feasible.py` | none | **stochastic** | none | none | none | none | none | **UNSUPPORTED** -- the DSL is fully deterministic |
| `vllm_style_token_budget` | `vllm_style_token_budget.py` | per-GPU block-rounded KV loop | ascending predicted_output | none | none | **loop-local KV budget** | none | none | **UNSUPPORTED** -- the per-step, per-GPU running-total admission loop was not verified representable via `admission_condition`/`batch_vars` in this pass; not attempted to avoid an unverified, possibly-misleading mapping |
| `sarathi_style` | `sarathi_style.py` | per-GPU halved-under-load prefill-chunk budget loop | FIFO | none | **loop-local budget** | none | none | none | **UNSUPPORTED** -- same reason as above, plus a starvation safety valve |
| `splitfuse_style` | `splitfuse_style.py` | per-GPU fixed-token-budget-fill loop | FIFO | none | **loop-local budget** | none | none | none | **UNSUPPORTED** -- same reason, plus an oversized-prefill safety valve |

## Coverage Summary

| Status | Count | Policies |
|---|---:|---|
| EXACT | 10 | fifo, edf, shortest_output_first, shortest_prompt_first, first_fit, orca_style, slo_slack_score, weighted_shortest_processing, least_laxity_first, estimated_service_time_first |
| APPROXIMATE | 9 | admission_control, scorpio_style_slo_guard, sola_style_state_aware, flow_control_stability, kv_constrained_online, adaptive_chunked_prefill, aging_priority, weighted_fair_share, multi_bin_batching |
| UNSUPPORTED | 8 | greedy_token_fill, least_loaded, best_fit, random_feasible, vllm_style_token_budget, sarathi_style, splitfuse_style, slai_style_phase_aware |
| **Faithful coverage (EXACT+APPROXIMATE)** | **19/27 (70.4%)** | up from 6/27 (22.2%) before this pass |

Per the task's own stated preference, this is not forced to 27/27: the 8 `UNSUPPORTED` policies are honestly left at the generic placeholder genome rather than given a misleading structural stand-in. Three concrete, named reasons account for all 8: fixed DSL placement strategy (4 policies), DSL determinism (1 policy), unverified budget-loop admission semantics (3 policies), and missing phase-share variables (1 policy, `slai_style_phase_aware`, double-counted conceptually with the variable-whitelist gap that also affects `weighted_fair_share`'s fairness term).

## Verification

Every one of the 27 mappings (including the 8 `UNSUPPORTED` placeholders) passes `SchedulerGenomeV1.validate()` and `compile_genome()`. All 27 `stable_hash()` values are distinct and deterministic across repeated calls (`tests/test_policy_genome_coverage.py`). For the 4 simplest EXACT mappings (`fifo`, `shortest_output_first`, `shortest_prompt_first`, `edf`), the compiled genome's admission *order* was compared against the native policy's on a real deterministic 3-request scenario and found identical -- not just claimed exact, but behaviorally verified.

## Structural Encoder Update

`selector.suitability.encoders.structural_features` now additionally derives, per genome:
- counts of referenced causal variables grouped into SLO-aware / prefill-weighted / decode-weighted / KV-aware / fairness-aware buckets (mechanical `ALLOWED_VARS` subset membership, not an invented label);
- `struct_is_regime_conditional`, `struct_is_admission_heavy`, `struct_is_pure_ranking` summary flags.

This raised the number of *distinct* structural feature vectors across the 27-policy library from effectively 2 (one shape per mapped policy before this pass, one shared placeholder shape) to 19 -- one per EXACT/APPROXIMATE policy, plus the shared placeholder for the 8 honestly-unsupported ones (which are, by design, structurally indistinguishable from each other -- see `test_unsupported_placeholders_are_structurally_identical_by_design`).

## Structural-Distance Diagnostic

Across 171 pairs of the 19 faithfully-mapped policies, the Pearson correlation between z-score-normalized structural distance and mean-absolute-reward disagreement (on the 32-window discriminative fixture) is **r = 0.559** -- a real, moderate positive correlation. `fifo`/`first_fit` (identical structural vectors by construction, since both are `const(0.0)` + `arrival_order`) show exactly zero structural distance and zero reward disagreement. The farthest pairs all involve `scorpio_style_slo_guard` (the most structurally complex genome: admission + priority + kv_guard + a regime condition), paired with simple ranking-only policies, and do show the largest behavioral disagreement. This is correlational evidence only -- it does not establish that structural similarity *causes* behavioral similarity, and is not claimed as such. See `docs/current/STATE_POLICY_SUITABILITY_REPORT.md` §"Re-Evaluation After Genome Expansion" for the full re-run results.
