# Phase 2B.14 Summary: Metric Audit and SCORPIO Ablation

**Date:** 2026-06-26
**Branch:** phase2b14-metric-audit-scorpio-ablation
**Input:** 319 windows from Phase 2B.13
**Output:** `results/phase2b14_metric_audit_scorpio_ablation/`

---

## Q1: What was the old WG denominator?

**Completed requests only.** `weighted_goodput = Σ(priority_i × slo_met_i) / Σ(priority_i)` where the sum is restricted to requests that finished execution. Dropped/rejected requests appear in neither numerator nor denominator.

## Q2: Was the old WG safe to call "goodput"?

**No, for high-rejection policies.** For FIFO/EDF/WSP (completion_fraction ≈ 0.99), the conditional metric and arrival-normalized metric are nearly identical — the name is safe in practice. For SCORPIO (completion_fraction ≈ 0.899), calling it "goodput" inflates apparent system performance by ~10 percentage points. The correct name is `completed_request_quality`.

## Q3: Did old docs or claims overstate denominator correctness?

**Yes.** Phase 2B.10–2B.13 reported `weighted_goodput` without noting the completed-only denominator, allowing readers to interpret it as system-level goodput. The metric is numerically correct; only the semantic framing was imprecise.

## Q4: What new metric names were added?

| Metric | Denominator | SCORPIO | WSP |
|---|---|---|---|
| `completed_request_quality` | Completed | 0.9846 | 0.8571 |
| `arrival_normalized_wg` | All arrivals | 0.8885 | 0.8540 |
| `cp_wg_t095_l05` | Arrival-norm, λ=0.5, tgt=0.95 | 0.8503 | **0.8524** |
| `cp_wg_t095_l10` | Arrival-norm, λ=1.0, tgt=0.95 | 0.8121 | **0.8508** |
| `cp_wg_t099_l05` | Arrival-norm, λ=0.5, tgt=0.99 | 0.8387 | **0.8523** |
| `cp_wg_t099_l10` | Arrival-norm, λ=1.0, tgt=0.99 | 0.7937 | **0.8480** |

**Bold = metric winner.**

## Q5: Under arrival-normalized WG, does SCORPIO still dominate?

**Yes, but with a reduced lead.** SCORPIO arrival-norm WG = 0.8885 vs second-best policy WSP = 0.8540 (gap of 0.0345). SCORPIO retains its rank-1 position under arrival-normalized WG.

## Q6: Under completion-penalized objectives, does SCORPIO still dominate?

**No.** WSP becomes the top policy under all completion-penalized objectives (target 0.95 or 0.99, any lambda). SCORPIO's 10% rejection rate creates a large penalty that eliminates its lead.

Rankings under `cp_wg_t095_l05`:
1. weighted_shortest_processing: 0.8524
2. scorpio_style_slo_guard: 0.8503 ← second
3. shortest_output_first: 0.8284
4. vllm_style_token_budget: 0.8284

Rankings under `cp_wg_t099_l10`:
1. weighted_shortest_processing: 0.8480
2. shortest_output_first: 0.8245
3. vllm_style_token_budget: 0.8245
4. estimated_service_time_first: 0.8220
*(SCORPIO: 0.7937 — rank 6 or lower)*

## Q7: Does SCORPIO appear to game metrics?

**Yes, in the technical sense.** SCORPIO rejects ~10% of arrivals, reducing its denominator and inflating conditional WG. This is not deliberate "gaming" — it is correct behavior for an admission-throttling policy. But it means SCORPIO's strong WG numbers partially reflect metric structure, not pure service quality improvement.

## Q8: Is SCORPIO's lower completion/admission acceptable?

**Context-dependent.** In overloaded regimes where admitting all requests would degrade quality for everyone, SCORPIO's selective rejection is a sound strategy. In regimes where other policies handle all arrivals successfully, SCORPIO's rejection is unnecessary and harmful. The completion-penalized metrics capture this tradeoff explicitly.

## Q9: Which SCORPIO component creates most of the gain?

Ablation run on 7 targeted discriminative workloads (KV-saturated / heavily overloaded).
SCORPIO reference on these workloads: arrival-norm WG = 0.6879, conditional WG = 0.9517.

| Ablation | Arrival-norm WG | Gap vs SCORPIO | Interpretation |
|---|---|---|---|
| scorpio_style_slo_guard (full) | 0.6879 | 0.0 | Reference |
| **scorpio_no_laxity_filter** | **0.000** | **-0.688** | **Catastrophic: no laxity pre-filter → KV deadlock** |
| **scorpio_no_rejection** | **0.000** | **-0.688** | **Catastrophic: accepting all arrivals → system saturation** |
| scorpio_no_decode_penalty | 0.6862 | -0.0017 | Minor loss from skipping long-decode penalty |
| scorpio_deadline_only | 0.6862 | -0.0017 | Laxity filter alone nearly matches full SCORPIO |
| scorpio_no_credit_budget | 0.6879 | 0.0 | Credit budget irrelevant on these workloads |
| scorpio_no_kv_guard | 0.6879 | 0.0 | KV guard irrelevant on these workloads |
| scorpio_no_age_bonus | 0.6882 | +0.0003 | Age bonus slightly hurts on these workloads |
| scorpio_no_priority_weight | 0.6883 | +0.0004 | Priority weight slightly hurts on these workloads |

**Key finding: The laxity pre-filter (rejecting requests with negative laxity) is the single most important SCORPIO component.** Disabling it causes complete system failure (WG=0, CF=0) on overloaded workloads because doomed requests flood the KV cache.

The KV utilization guard, credit budget throttle, and composite scoring contribute negligibly in isolation on these discriminative workloads. SCORPIO's advantage is primarily a **deadline-based pre-filter** that prevents KV cache saturation by requests that cannot meet their deadlines.

Note: The laxity filter also explains SCORPIO's low completion fraction (~0.899 overall). SCORPIO proactively rejects ~10% of arrivals as infeasible, inflating its conditional WG.

See `results/phase2b14_metric_audit_scorpio_ablation/ablation_gap_analysis.json` for raw values.

## Q10: Do selectors beat always-SCORPIO under corrected metrics?

**Yes — multiple selectors beat always-SCORPIO under arrival-normalized WG:**

| Selector | Arrival-norm WG | vs always-SCORPIO (0.8885) |
|---|---|---|
| knn_selector | 0.8970 | **+0.0085** |
| per_policy_regression | 0.8948 | +0.0063 |
| random_forest | 0.8944 | +0.0059 |
| random_forest_regret_weighted | 0.8944 | +0.0059 |
| decision_tree_regret_weighted | 0.8934 | +0.0049 |
| safe_fallback_margin0.005 | 0.8933 | +0.0048 |
| safe_fallback_margin0.001 | 0.8933 | +0.0048 |
| safe_fallback_margin0.010 | 0.8918 | +0.0033 |
| decision_tree | 0.8910 | +0.0025 |
| always_scorpio | 0.8885 | baseline |
| rule_based | 0.8398 | -0.0487 |

**Selector contribution is preserved under corrected metrics.** The KNN, RF, and safe-fallback selectors all select non-SCORPIO policies in some windows, achieving higher arrival-normalized WG than always picking SCORPIO.

## Q11: Are Phase 2B.10–2B.13 results valid, partially valid, or needing reinterpretation?

**Partially valid — numbers correct, semantics need clarification.**

- Relative rankings among non-SCORPIO policies: fully valid (all have CF≈1.0).
- SCORPIO absolute WG: numerically correct but semantically mislabeled.
- Selector vs always-SCORPIO comparison: valid under conditional metric; also valid under arrival-normalized (selectors still win).
- SCORPIO dominance as "best policy": valid under conditional metric; NOT valid under completion-penalized metrics.

## Q12: What exact claims are now safe?

1. "SCORPIO achieves conditional quality 0.9846 on completed requests."
2. "SCORPIO arrival-normalized WG = 0.8885, highest among all 20 policies."
3. "SCORPIO rejects ~10% of arrivals; this is intentional admission throttling."
4. "RF selector arrival-normalized WG = 0.8944, beats always-SCORPIO by +0.0059."
5. "Under completion-penalized objectives (any target/lambda), WSP is the top policy."
6. "Phase 2B.10–2B.13 WG comparisons among FIFO/EDF/WSP/etc. are accurate arrival-normalized comparisons (CF≈1 for those policies)."

## Q13: What exact claims are unsafe?

1. ~~"SCORPIO WG = 0.9846 is a system-level goodput metric."~~
   → Correct: conditional quality on completed requests = 0.9846; arrival-normalized = 0.8885.
2. ~~"SCORPIO dominates all baselines by 10+ percentage points on goodput."~~
   → Under arrival-normalized WG, lead over WSP = 0.0345. Under completion-penalized metrics, SCORPIO is not the leader.
3. ~~"Selector training maximizes system-level goodput."~~
   → Selector training maximized conditional quality labels. Arrival-normalized WG ordering of selectors is consistent but trained objectives were not arrival-normalized.

## Near-Tie Analysis Under Arrival-Normalized WG

| Metric | Old (conditional WG) | New (arrival-norm WG) |
|---|---|---|
| All-complete fraction | ~0.93 | **0.64** |
| Near-tie fraction (ε=0.001) | ~0.74 | 0.70 |
| Near-tie fraction (ε=0.005) | ~0.83 | 0.70 |
| Near-tie fraction (ε=0.010) | ~0.83 | 0.74 |
| Meaningful windows (ε=0.001) | ~84 | **97** |

**Under arrival-normalized WG, fewer windows are all-complete (0.64 vs 0.93) and more windows are meaningfully discriminated.** The corrected metric reveals more genuine differentiation in the data.

## Main Failure Cases

1. **SCORPIO metric inflation**: Old WG denominator (completed-only) overstated SCORPIO's system-level performance by ~10pp.
2. **Completion-penalized reversal**: Under any completion-penalized objective, SCORPIO is not the best policy. WSP wins.
3. **Near-tie labels remain frequent**: Even under arrival-normalized WG, 70% of windows are near-tie at ε=0.001. Only 97 windows have meaningful discrimination.
4. **Selector labels trained on wrong metric**: Selectors were trained on conditional WG labels. This is a defensible approximation (CF≈1 for non-SCORPIO) but was not explicitly justified.
5. **Ablation pending**: SCORPIO component attribution awaits ablation results (tmux session running).
