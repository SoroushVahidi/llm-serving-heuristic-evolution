# Phase 2B.13 Selector Training and SCORPIO Suspicion Audit Summary

**Phase:** 2B.13  
**Date:** 2026-06-26  
**Branch:** `phase2b13-selector-training-and-suspicion-audit`  
**Config:** `configs/phase2b13_selector_training_and_suspicion_audit.yaml`  
**Runner:** `scripts/run_phase2b13_selector_training_and_suspicion_audit.py`  
**Log:** `logs/phase2b13/phase2b13_selector_training.log`  
**Results:** `results/phase2b13_selector_training_and_suspicion_audit/` (gitignored)  
**tmux session:** `phase2b13_selector_training`

---

## Experiment Purpose

Phase 2B.13 extends Phase 2B.12 (172 windows) to **≥200 windows**, then:

1. Audits suspicious SCORPIO dominance and near-tie / all-complete label quality
2. Trains RF/DT and alternative selectors if feasibility criteria pass
3. Compares all selectors against **always-SCORPIO**, best fixed, and per-window oracle

Phase 2B.12 showed label diversity is sufficient (45.9% SCORPIO concentration) but window
count blocked RF/DT training. Phase 2B.13 adds extended seeds and six KV-pressure /
overload workloads targeting regimes where non-SCORPIO policies genuinely win.

---

## Suite Extension Beyond Phase 2B.12

| Change | Detail |
|--------|--------|
| Diversity seeds | `[6,7,8,9]` → `[6,7,8,9,10,11]` (+2 seeds on 18 synthetic diversity workloads) |
| New workloads | 6 targeted regimes: `div_kv_extreme_tight_slo`, `div_kv_extreme_decode_only`, `div_high_overload_tight_priority`, `div_kv_mixed_extreme_noise`, `div_decode_saturation_bursty`, `div_extreme_overload_short_tight` |
| Total workloads | 29 (9 regression + 20 diversity) |
| Training split | train = dev + diversity seeds 6–10; val = diversity seed 11; test = heldout seeds 3–5 |

Regression workloads unchanged from Phase 2B.11/2B.12 for continuity.

---

## Evaluation Windows

| Group | Windows | Notes |
|-------|---------|-------|
| dev (regression) | 27 | seeds 0–2 |
| heldout (regression) | 33 | seeds 3–5 |
| regression total | **60** | unchanged from Phase 2B.9–2B.12 |
| diversity | **259** | 20 workloads × extended seeds (KV-extreme workloads add extra windows) |
| **overall** | **319** | exceeds 200 threshold |

Train / val / test for ML: 245 / 41 / 33 windows.

---

## RF/DT Feasibility

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| Window count | ≥200 | **319** | PASS |
| Policies winning ≥10 windows | ≥3 | **7** | PASS |
| Top policy concentration | <85% | **55.2%** (SCORPIO) | PASS |

**Overall: FEASIBLE** — RF/DT and alternative selectors trained.

---

## Label Distribution

### Overall (n=319, before near-tie filtering)

| Policy | Wins | Fraction |
|--------|------|----------|
| `scorpio_style_slo_guard` | 176 | 55.2% |
| `admission_control` | 37 | 11.6% |
| `best_fit` | 28 | 8.8% |
| `edf` | 20 | 6.3% |
| `multi_bin_batching` | 19 | 6.0% |
| `shortest_output_first` | 16 | 5.0% |
| `estimated_service_time_first` | 12 | 3.8% |
| others | 11 | 3.4% |

**11 distinct policies** appear as per-window best deployable labels.

### Diversity subset (n=259)

SCORPIO label fraction: **~48%** on diversity; KV-extreme workloads increase SCORPIO wins vs Phase 2B.12 diversity subset.

### After near-tie filtering (eps=0.005)

126 meaningful windows remain (193/319 near-tie). See `label_distribution_non_tie.csv`.

---

## Near-Tie / All-Complete Audit

| Metric | Value |
|--------|-------|
| All-complete windows (best WG ≥ 0.99) | 234/319 (73.4%) |
| Meaningful WG-gap windows (eps=0.005) | 126/319 (39.5%) |
| Near-tie at eps=0.001 | 193/319 (60.5%) |

**Finding:** Extended seeds add label margin diversity but most new windows are still
all-complete. The six KV-extreme workloads are the primary source of genuine WG gaps.

---

## Selector Results (Held-Out Test, n=33)

| Selector | Mean WG | Gap vs best fixed | Gap vs always-SCORPIO | Gap vs oracle |
|----------|---------|-------------------|----------------------|---------------|
| **always-SCORPIO** | **0.9975** | 0.0000 | 0.0000 | −0.0020 |
| random_forest | 0.9975 | 0.0000 | 0.0000 | −0.0020 |
| per_policy_regression | **0.9978** | +0.0002 | **+0.0003** | −0.0018 |
| knn_selector | 0.9967 | −0.0008 | −0.0008 | −0.0028 |
| rule_based | 0.9803 | −0.0172 | −0.0172 | −0.0192 |
| decision_tree | 0.9780 | −0.0195 | −0.0195 | −0.0215 |
| safe_fallback (all margins) | 0.9975 | 0.0000 | 0.0000 | −0.0020 |

### Overall (n=319)

| Selector | Mean WG | Gap vs best fixed (0.9846) |
|----------|---------|---------------------------|
| random_forest | 0.9855 | +0.0009 |
| always-SCORPIO | 0.9846 | 0.0000 |
| per_policy_regression | 0.9854 | +0.0008 |
| rule_based | 0.8431 | −0.1415 |
| decision_tree | 0.9805 | −0.0041 |

**Note:** Rule selector diversity WG collapses to 0.8179 on KV-extreme workloads.

---

## Key Findings

### 1. RF ties always-SCORPIO on held-out; per-policy regression marginally ahead

RF held-out WG = 0.9975 = always-SCORPIO. Per-policy regression achieves 0.9978 (+0.0003 vs
always-SCORPIO) — statistically negligible. RF/RF-regret/safe-fallback all collapse to SCORPIO
on all 33 held-out windows.

### 2. Rule selector fails badly on KV-extreme diversity workloads

Diversity WG = 0.8179 vs always-SCORPIO 0.9826 (gap −0.165). KV-pressure regimes expose
rule selector's lack of SCORPIO routing under offline evaluation.

### 5. Near-tie labels dominate training signal

93% all-complete windows create tie-breaking labels, not meaningful regret. Regret-weighted
RF/DT partially mitigates but does not change held-out conclusion.

### 6. SCORPIO admission/completion audit

Per-window `completion_{policy}` fields stored in dataset rows. SCORPIO may show lower
`completion_fraction` than EDF/FIFO on KV-pressure windows while maintaining high WG on
completed requests (WG denominator = completed only). See `completion_admission_summary.csv`.

Under completion-penalized objectives (target=0.95, λ=0.5/1.0), SCORPIO remains rank #1
on regression windows; see `objective_sensitivity.csv`.

### 7. Leakage audit passes

`oracle_srtf` excluded from selector candidates. Features are `online_prefix` only.
No actual output length or future arrivals in feature vector.

---

## Selectors Trained

| Selector | Trained | Held-out beats always-SCORPIO? |
|----------|---------|-------------------------------|
| Standard RF | Yes | No (ties) |
| Standard DT | Yes | No |
| Regret-weighted RF | Yes | No |
| Regret-weighted DT | Yes | No |
| per_policy_regression | Yes | Marginal (+0.0003 WG, not meaningful) |
| KNN (k=5) | Yes | No (ties) |
| Safe fallback (margins 0.001/0.005/0.010) | Yes | No (ties) |

---

## Best Current Selector Contribution Claim

**None on held-out windows.** Always-SCORPIO is the strongest simple baseline.
RF ties always-SCORPIO on WG but does not exceed it; it mostly re-discovers SCORPIO.
The rule selector remains worse than always-SCORPIO.

Publication claim must be reframed: SCORPIO-style guard is a strong fixed baseline;
selector learning over 256 windows does not beat it on held-out workloads.

---

## Before Publication

1. Report always-SCORPIO baseline in all selector comparisons
2. Exclude or down-weight near-tie / all-complete windows in training
3. SCORPIO ablation study (disable admission guard, TTFT guard, TPOT guard separately)
4. Real-trace dataset ingestion for out-of-distribution held-out
5. Do not claim selector beats best fixed unless held-out gap is positive and significant

---

## Recommended Next Step

**Option 2:** always-SCORPIO remains unbeatable on held-out → perform SCORPIO ablation
and dataset ingestion before more selector work. Near-tie label dominance (Option 3) is a
secondary concern; the primary blocker is SCORPIO's strength as a fixed baseline.

---

## tmux / Execution

```bash
tmux new -s phase2b13_selector_training
python scripts/run_phase2b13_selector_training_and_suspicion_audit.py \
  --config configs/phase2b13_selector_training_and_suspicion_audit.yaml \
  --output results/phase2b13_selector_training_and_suspicion_audit \
  2>&1 | tee logs/phase2b13/phase2b13_selector_training.log
```

Expected runtime: ~31 minutes on CPU (row build ~31 min + training ~1 min).
