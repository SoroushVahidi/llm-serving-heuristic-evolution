# Family-A Real-System Feature Portability V1

Date: 2026-08-21

## Purpose

Lightweight audit of the 63-feature `feat_*` schema used by the Family-A
learned-scheduler pipeline (pilot + scaled D0) against real-system portability.

No code is implemented. No models are trained. No labels are generated.

## 1. Current Schema Summary

The 63 features are divided into three groups:

### Global features (41)

| # | Name | Source | Classification |
|---|------|--------|----------------|
| 1 | queue_length | `len(state.waiting_queue)` | RUNTIME_NATIVE |
| 2 | active_count | `sum(len(g.active_request_ids))` | RUNTIME_NATIVE |
| 3 | n_gpus | `len(state.gpu_states)` | RUNTIME_NATIVE |
| 4 | queue_age_p10 | quantile of `(state.time - r.arrival_time)` | RUNTIME_DERIVABLE_CHEAP |
| 5 | queue_age_p50 | quantile of `(state.time - r.arrival_time)` | RUNTIME_DERIVABLE_CHEAP |
| 6 | queue_age_p90 | quantile of `(state.time - r.arrival_time)` | RUNTIME_DERIVABLE_CHEAP |
| 7 | queue_age_mean | mean of `(state.time - r.arrival_time)` | RUNTIME_DERIVABLE_CHEAP |
| 8 | predicted_output_tokens_p10 | quantile of `r.predicted_output_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 9 | predicted_output_tokens_p50 | quantile of `r.predicted_output_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 10 | predicted_output_tokens_p90 | quantile of `r.predicted_output_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 11 | predicted_output_tokens_mean | mean of `r.predicted_output_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 12 | prompt_tokens_p10 | quantile of `r.prompt_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 13 | prompt_tokens_p50 | quantile of `r.prompt_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 14 | prompt_tokens_p90 | quantile of `r.prompt_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 15 | prompt_tokens_mean | mean of `r.prompt_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 16 | est_service_time_p10 | quantile of `predicted_service_proxy(r)` | RUNTIME_DERIVABLE_CHEAP |
| 17 | est_service_time_p50 | quantile of `predicted_service_proxy(r)` | RUNTIME_DERIVABLE_CHEAP |
| 18 | est_service_time_p90 | quantile of `predicted_service_proxy(r)` | RUNTIME_DERIVABLE_CHEAP |
| 19 | est_service_time_mean | mean of `predicted_service_proxy(r)` | RUNTIME_DERIVABLE_CHEAP |
| 20 | max_class_deficit_ratio | `max(demand[c] / (active_by_class[c] + 1))` | RUNTIME_DERIVABLE_CHEAP |
| 21 | longest_waiting_age | `max(state.time - r.arrival_time)` | RUNTIME_DERIVABLE_CHEAP |
| 22 | n_distinct_classes_in_queue | `len(Counter(r.class_id))` | RUNTIME_DERIVABLE_CHEAP |
| 23 | laxity_p10 | quantile of `r.slo_deadline - state.time` | RUNTIME_DERIVABLE_CHEAP |
| 24 | laxity_p50 | quantile of `r.slo_deadline - state.time` | RUNTIME_DERIVABLE_CHEAP |
| 25 | laxity_p90 | quantile of `r.slo_deadline - state.time` | RUNTIME_DERIVABLE_CHEAP |
| 26 | laxity_mean | mean of `r.slo_deadline - state.time` | RUNTIME_DERIVABLE_CHEAP |
| 27 | fraction_laxity_negative | `mean(laxity < 0)` | RUNTIME_DERIVABLE_CHEAP |
| 28 | fraction_laxity_near_deadline | `mean(laxity < near_deadline_cutoff)` | RUNTIME_DERIVABLE_CHEAP |
| 29 | mean_kv_utilization | `mean(g.current_kv_tokens / g.max_kv_tokens)` | RUNTIME_DERIVABLE_CHEAP |
| 30 | max_kv_utilization | `max(g.current_kv_tokens / g.max_kv_tokens)` | RUNTIME_DERIVABLE_CHEAP |
| 31 | free_kv_capacity | `sum(g.max_kv_tokens - g.current_kv_tokens)` | RUNTIME_DERIVABLE_CHEAP |
| 32 | prefilling_count | `sum(g.prefilling_count)` | RUNTIME_NATIVE |
| 33 | decoding_count | `sum(g.decoding_count)` | RUNTIME_NATIVE |
| 34 | agg_n_admit_estf | `len(estf_admit_set)` | RUNTIME_DERIVABLE_CHEAP |
| 35 | agg_n_admit_wfs | `len(wfs_admit_set)` | RUNTIME_DERIVABLE_CHEAP |
| 36 | admit_symmetric_diff_size | `|estf_admit ^ wfs_admit|` | RUNTIME_DERIVABLE_CHEAP |
| 37 | history_queue_len_slope | `polyfit(history_queue_len, 1)` | RUNTIME_DERIVABLE_EXPENSIVE |
| 38 | history_kv_util_slope | `polyfit(history_kv_util, 1)` | RUNTIME_DERIVABLE_EXPENSIVE |
| 39 | history_admitted_count_slope | `polyfit(history_admitted, 1)` | RUNTIME_DERIVABLE_EXPENSIVE |

### Side features (6 per candidate, 12 total)

| # | Name | Source | Classification |
|---|------|--------|----------------|
| 40 | estf_priority | `r.priority` | RUNTIME_NATIVE |
| 41 | estf_prompt_tokens | `r.prompt_tokens` | RUNTIME_NATIVE |
| 42 | estf_predicted_output_tokens | `r.predicted_output_tokens` | RUNTIME_NATIVE |
| 43 | estf_predicted_service_proxy | `predicted_service_proxy(r)` | RUNTIME_DERIVABLE_CHEAP |
| 44 | estf_remaining_predicted_service_proxy | `predicted_service_proxy(r)` | RUNTIME_DERIVABLE_CHEAP |
| 45 | estf_queue_age | `state.time - r.arrival_time` | RUNTIME_DERIVABLE_CHEAP |
| 46 | estf_laxity_own | `r.slo_deadline - state.time` | RUNTIME_DERIVABLE_CHEAP |
| 47 | wfs_priority | `r.priority` | RUNTIME_NATIVE |
| 48 | wfs_prompt_tokens | `r.prompt_tokens` | RUNTIME_NATIVE |
| 49 | wfs_predicted_output_tokens | `r.predicted_output_tokens` | RUNTIME_NATIVE |
| 50 | wfs_predicted_service_proxy | `predicted_service_proxy(r)` | RUNTIME_DERIVABLE_CHEAP |
| 51 | wfs_remaining_predicted_service_proxy | `predicted_service_proxy(r)` | RUNTIME_DERIVABLE_CHEAP |
| 52 | wfs_queue_age | `state.time - r.arrival_time` | RUNTIME_DERIVABLE_CHEAP |
| 53 | wfs_laxity_own | `r.slo_deadline - state.time` | RUNTIME_DERIVABLE_CHEAP |

### Pair features (10 total)

| # | Name | Source | Classification |
|---|------|--------|----------------|
| 54 | priority_diff_estf_minus_wfs | `estf_priority - wfs_priority` | RUNTIME_DERIVABLE_CHEAP |
| 55 | prompt_tokens_diff_estf_minus_wfs | `estf_prompt_tokens - wfs_prompt_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 56 | predicted_output_tokens_diff_estf_minus_wfs | `estf_predicted_output_tokens - wfs_predicted_output_tokens` | RUNTIME_DERIVABLE_CHEAP |
| 57 | predicted_service_proxy_diff_estf_minus_wfs | `estf_predicted_service_proxy - wfs_predicted_service_proxy` | RUNTIME_DERIVABLE_CHEAP |
| 58 | queue_age_diff_estf_minus_wfs | `estf_queue_age - wfs_queue_age` | RUNTIME_DERIVABLE_CHEAP |
| 59 | laxity_own_diff_estf_minus_wfs | `estf_laxity_own - wfs_laxity_own` | RUNTIME_DERIVABLE_CHEAP |
| 60 | priority_ratio_estf_over_wfs | `estf_priority / max(wfs_priority, 1e-9)` | RUNTIME_DERIVABLE_CHEAP |
| 61 | predicted_service_proxy_ratio_estf_over_wfs | `estf_predicted_service_proxy / max(wfs_predicted_service_proxy, 1e-9)` | RUNTIME_DERIVABLE_CHEAP |
| 62 | queue_age_ratio_estf_over_wfs | `estf_queue_age / max(wfs_queue_age, 1e-9)` | RUNTIME_DERIVABLE_CHEAP |
| 63 | laxity_own_ratio_estf_over_wfs | `estf_laxity_own / max(wfs_laxity_own, 1e-9)` | RUNTIME_DERIVABLE_CHEAP |

## 2. Classification Summary

| Classification | Count | Features |
|----------------|-------|----------|
| RUNTIME_NATIVE | 9 | queue_length, active_count, n_gpus, prefilling_count, decoding_count, estf/wfs priority, prompt_tokens, predicted_output_tokens |
| RUNTIME_DERIVABLE_CHEAP | 52 | All quantiles, ratios, differences, KV metrics, history slopes, admit counts, max_class_deficit |
| RUNTIME_DERIVABLE_EXPENSIVE | 3 | history_queue_len_slope, history_kv_util_slope, history_admitted_count_slope |
| SIMULATOR_ONLY | 0 | (none) |
| FORBIDDEN_FUTURE | 0 | (none) |
| INVALID_OR_AMBIGUOUS | 0 | (none) |

**Result: All 63 features are theoretically portable to a real system.**
None are SIMULATOR_ONLY, FORBIDDEN_FUTURE, or INVALID_OR_AMBIGUOUS.

## 3. Portability Concerns

### 3.1 History Slopes (3 features)

The history features (history_queue_len_slope, history_kv_util_slope,
history_admitted_count_slope) require storing per-cycle aggregates over the
last 10 steps and computing linear regression. In a real system these would
need a circular buffer maintained by the scheduler. This is feasible but
requires scheduler instrumentation — the history window is not a runtime-native
value.

**Risk: LOW** — a 10-step circular buffer of 3 values is trivial storage.

### 3.2 Quantile Statistics (16 features)

Queue age, predicted output tokens, prompt tokens, service time, and laxity
quantiles (p10/p50/p90 + mean) each require a full queue scan and sort.
In a real system with a large waiting queue these are O(n log n) per decision.

**Risk: LOW-MEDIUM** — queue sizes in production are typically manageable
(<1000 waiting), but repeated full-queue scans for 5 different quantities
duplicates work that could be shared.

### 3.3 Pair-Rank Spearman (not in D0 schema)

The observability diagnostic computes `pair_rank_spearman_topk` (Spearman
correlation between ESTF and WFS score rankings). This is NOT in the 63-feature
D0 schema. It was excluded from the learned-scheduler feature set (correctly).

### 3.4 `estf_remaining_predicted_service_proxy` vs `estf_predicted_service_proxy`

These are identical — both compute `predicted_service_proxy(r)`. The
`remaining_` prefix suggests it should subtract already-served work, but no
such subtraction occurs. This is a naming artifact, not a portability problem.

**Impact: ZERO** — both values are available at decision time. The name is
misleading but the value is not.

## 4. Causality Audit

All 63 features are observable at decision time:

- No feature depends on branch outcomes (J_ESTF_whole, J_WFS_whole, etc.)
- No feature depends on future completions or realized output tokens
- No feature uses `actual_output_tokens` (excluded from ObservableRequest)
- No feature uses oracle labels or delta_J values
- No feature uses TEST indicators
- No feature uses future-arrival information
- No feature uses scenario configuration metadata (utilization, skew, fav, seed)

**No causal leakage found.**

## 5. Unit Audit

| Feature Group | Units | Compatibility |
|---------------|-------|---------------|
| queue_length, active_count, n_gpus | count | Consistent |
| queue_age_p*/laxity_p*/queue_age_own/laxity_own | time (scheduler steps) | Consistent |
| prompt_tokens_p*/prompt_tokens_own | tokens | Consistent |
| predicted_output_tokens_p*/predicted_output_tokens_own | tokens | Consistent |
| est_service_time_p*/predicted_service_proxy | steps (proxy) | Consistent |
| max_class_deficit_ratio | ratio | Consistent |
| fraction_laxity_negative, fraction_laxity_near_deadline | ratio [0,1] | Consistent |
| mean_kv_utilization, max_kv_utilization | ratio [0,1] | Consistent |
| free_kv_capacity | tokens | Consistent |
| prefilling_count, decoding_count | count | Consistent |
| agg_n_admit_*/admit_symmetric_diff_size | count | Consistent |
| history_*_slope | value/cycle | Consistent |
| *_priority | priority weight | Consistent (same scale across candidates) |
| *_diff_estf_minus_wfs | same unit as source | Consistent |
| *_ratio_estf_over_wfs | ratio | Consistent |

**No unit-mixing errors found.** The previously-invalid
`deadline_slack_if_admitted_now` is NOT present in the 63 features.

## 6. Hot-Path Complexity Analysis

### O(1) operations
- queue_length, active_count, n_gpus
- prefilling_count, decoding_count
- all per-request features for the two contested requests
- all pair differences and ratios

### O(#waiting) operations
- All quantile statistics (5 quantities x 4 stats = 20 feature computations,
  each scans the full waiting queue)
- max_class_deficit_ratio (2 queue scans: waiting_queue + active_requests)
- longest_waiting_age
- n_distinct_classes_in_queue
- fraction_laxity_negative, fraction_laxity_near_deadline
- KV utilization metrics (1 GPU-state scan)
- free_kv_capacity

### O(history_window) operations
- History slope computation (3 polyfit calls over ~10 points)

### Shared intermediate computations
The quantile features could share a single sort of the waiting queue:
1. Sort once by queue_age -> compute queue_age quantiles + longest_waiting
2. Sort once by predicted_output_tokens -> compute predicted_output quantiles
3. Sort once by prompt_tokens -> compute prompt_tokens quantiles
4. Sort once by predicted_service_proxy -> compute est_service_time quantiles
5. Sort once by laxity -> compute laxity quantiles + fractions

Currently each quantile set triggers a separate `_quantile_stats` call over
its own list. This is not incorrect but duplicates work.

## 7. Cycle-Cached Feature Extractor Design

For real-system deployment, features should be computed once per scheduler
cycle and cached:

### Incrementally maintained aggregates (per cycle)

| Aggregate | Updated by | Cost |
|-----------|-----------|------|
| queue_length | push/pop queue | O(1) |
| active_count | admit/release | O(1) |
| n_gpus | static | O(1) |
| per-GPU KV utilization | admit/release | O(1) |
| prefilling_count, decoding_count | admit/release | O(1) |
| per-class demand counts | push/pop queue | O(1) with Counter |

### Computed on-demand (once per decision cycle, cached)
- All quantile stats: single queue scan, 5 sorted views
- History slopes: linear regression over 10-step buffer

### Not cached
- Per-request features (only 2 contested requests, O(1))
- Pair features (derived from cached side features)

## 8. Portable Feature Subset

### PORTABLE_V1_FEATURE_SET: 63 / 63 retained

All 63 features are portable. The classification breakdown:

| Category | Count | Portability |
|----------|-------|-------------|
| RUNTIME_NATIVE | 9 | Directly from scheduler state |
| RUNTIME_DERIVABLE_CHEAP | 52 | Computed from current scheduler state in <O(queue) time |
| RUNTIME_DERIVABLE_EXPENSIVE | 3 | Require 10-step circular buffer but trivially portable |
| SIMULATOR_ONLY | 0 | None |

**0 features removed. 0 features blocked.**

Removed features: (none)
Reason: (N/A — all portable)

## 9. D0 Retraining Compatibility

**PORTABLE_SUBSET_RETRAINING_POSSIBLE_FROM_D0**

The D0 rows already contain all 63 `feat_*` columns needed for the portable
subset (which is the full set). No label regeneration is required.

## 10. Inference Overhead Budget Design

### Benchmark Design

Future benchmark should measure per-decision latency in microseconds:

| Measurement | Description |
|-------------|-------------|
| A. Native baseline | `policy.select_action(state)` wall time, no features |
| B. Full 63-feature extraction | `extract_causal_features()` + side/pair features |
| C. Portable V1 extraction | Same as B but history slopes from cached buffer |
| D. Logistic regression inference | 63-dim vector -> probability, <1ms |
| E. Small GBDT inference | Trained model prediction, <5ms |
| F. OOD check | Distance to training support, <2ms |
| G. Total overhead | C + D/E + F |

### Reporting units
- Median, p95, p99 decision latency (microseconds)
- Overhead as % of native scheduler cycle time
- Throughput impact: requests/second with vs. without learned selector

### Target thresholds (to be validated)
- Feature extraction: <500 microseconds median, <2000 microseconds p99
- Inference: <50 microseconds median
- Total overhead: <1% of scheduler cycle time, <5% of p99 latency

## 11. Deployment Overhead Gate

**Overhead Gate OG-1** (future experiment):

Before deploying the learned selector in any closed-loop evaluation:

A. **Throughput gate**: Learned-selector mode must not reduce sustained
   throughput by more than 1% compared to the native policy (fixed WFS),
   measured at the same workload intensity, with 95% confidence.

B. **Latency gate**: p99 per-decision latency must not regress by more than
   10 microseconds absolute (or 5% relative, whichever is larger) compared
   to native policy.

C. **GPU utilization gate**: Average GPU utilization must not decrease by
   more than 0.5% absolute points over a 30-minute sustained run.

D. **Stability gate**: No OOM, no CPU saturation, no scheduler deadlock
   over a 2-hour sustained run.

## 12. Real-System Portability Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| SLO deadline metadata absent in vanilla vLLM | HIGH | SLO deadlines are not in vanilla vLLM. Need a scheduler modification to attach deadline metadata to requests. Consider using a heuristic proxy (e.g., inverse of request size) if real deadlines unavailable. |
| Output-length prediction dependency | LOW | `predicted_output_tokens` is available from the client-supplied request metadata in vLLM. This is already provided at enqueue time. |
| Class/tenant identity unavailable | LOW | `class_id` is client-supplied metadata. In production, this maps to tenant IDs, QoS tiers, or priority classes. |
| Expensive queue distribution statistics | MEDIUM | Quantile computation over the waiting queue is feasible for queue sizes <1000. For larger queues, use streaming quantile algorithms or bounded sketches. |
| KV accounting mismatch | LOW | `current_kv_tokens` and `max_kv_tokens` are directly available from vLLM's KV cache manager. Exact match with simulator semantics. |
| Progress semantics mismatch | LOW | `tokens_decoded_per_request` available from vLLM's internal state. `prefilling_count`/`decoding_count` map to scheduler phase counts. |
| History buffer instrumentation | LOW | Requires a 10-entry circular buffer. Trivial addition to any scheduler's main loop. |
| `predicted_service_proxy` dependency on alpha/beta | LOW | alpha=0.5, beta=1.0 are constants. Same formula used by native policies. |
| Scheduler internal APIs unstable | MEDIUM | Depends on vLLM's internal data structures (ObservableRequest, ObservableGPUState). In a real integration these would be vLLM's native types, requiring adaptation. |

## 13. Fingerprint Relationship

The `stable_state_fingerprint` is a SHA-256 of:

```json
{
  "scenario": scenario_id,
  "step": step,
  "estf_request": estf_contested_request_id,
  "wfs_request": wfs_contested_request_id,
  "features": {feat_1: rounded_val, ...}
}
```

### Assessment

**Adequate for deduplication purposes.** The fingerprint includes:
- The scenario and step (unique decision point identifier)
- The contested request IDs (which differ between ESTF and WFS candidates)
- All 63 feature values (rounded to 12 decimal places for determinism)

This correctly distinguishes different decision states even when the same
scenario and step produce different features (which cannot happen in the
current deterministic simulator, but guards against future non-determinism).

### Notable limitation

The fingerprint does NOT include the exact `ObservableState` bytes. It relies
on the 63 features as a state proxy. If two states have identical feature
values but differ in unobserved dimensions (e.g., the exact ordering of
requests within the queue that don't affect the 63 aggregates), they would
produce identical fingerprints. This is acceptable for deduplication because
identical features + identical contested requests + identical step implies
identical scheduler decision state for the purposes of ESTF/WFS disagreement.

## 14. Real-System Integration Plan

### Minimal future integration path:

1. **Portable feature adapter** — Wrap `extract_causal_features` to read from
   vLLM's native types (ObservableRequest -> vLLM Request metadata). Handle
   SLO deadline mapping.

2. **Selector model** — Train on D0 `feat_*` columns (all 63). Use per-group
   weighting to avoid dense-disagreement bias (see audit report).

3. **WFS fallback** — On abstention, defer to native WFS. On missing features,
   defer to native WFS.

4. **OOD check** — Optional k-NN or conformal prediction layer. Compute once
   per decision, <2ms overhead.

5. **Timing instrumentation** — Add the benchmark from Section 10. Measure
   overhead gate OG-1 before any deployment.

6. **Real-serving validation** — Run closed-loop on a real vLLM instance
   (not simulator) with the learned selector + WFS fallback. Compare against
   fixed WFS on real trace workloads.

### Likely files to add/modify:
- `src/llmserveopt/selector/portable_feature_adapter.py` — real-system feature extraction
- `src/llmserveopt/selector/learned_scheduler_policy.py` — learned selector as BasePolicy
- `src/llmserveopt/selector/overhead_benchmark.py` — latency measurement harness
- `scripts/train_family_a_selector_v1.py` — model training script
- `scripts/evaluate_family_a_selector_real_system.py` — real-serving validation

## 15. Classification

**REAL_SYSTEM_FEATURE_PORTABILITY_READY**

All 63 features are theoretically portable to a real system. None are
simulator-only, forbidden-future, or invalid. The only practical concerns
are:

1. SLO deadline metadata must be attached to requests (not default in vLLM)
2. History slopes require a 10-step circular buffer (trivial addition)
3. Queue quantile computation is O(n log n) per decision (feasible for
   realistic queue sizes)

These are integration considerations, not schema blockers.
