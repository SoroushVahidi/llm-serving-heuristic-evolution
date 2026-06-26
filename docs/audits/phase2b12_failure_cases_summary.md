# Phase 2B.12 Failure Cases Summary

**Experiment:** `phase2b12_workload_diversity_selector_labels`  
**Branch:** `phase2b12-workload-diversity-selector-labels`  
**Results:** `results/phase2b12_workload_diversity_selector_labels/` (gitignored)

---

## Inherited Failure Cases (All Resolved Before Phase 2B.12)

| ID | Status | Resolution |
|---|---|---|
| fail_001 | Resolved Phase 2B.8 | Rule selector repaired: KV pressure → WSP |
| fail_002 | Resolved Phase 2B.8 | Rule selector repaired: high noise → AC |
| fail_003 | Resolved Phase 2B.8 | Rule selector repaired: decode-heavy → WSP |
| fail_004 | Resolved Phase 2B.11 | `pred_output_cv > 2.0` → `scorpio_style_slo_guard` |
| fail_005 | Resolved Phase 2B.11 | 3 SCORPIO routing rules added to rule selector |
| fail_006 | Resolved Phase 2B.11 | `scorpio_style_slo_guard` added to `_POLICY_CHOICES` |

---

## Phase 2B.12 Failure Cases

### fail_007: Rule selector under-dispatches to SCORPIO across all 172 windows

| Field | Value |
|-------|-------|
| failure_case_id | fail_007 |
| experiment_id | phase2b12_workload_diversity_selector_labels |
| workload group | Overall (regression + diversity) |
| selector method | rule_based |
| selected policy | `slo_slack_score` (54% of windows) |
| best fixed policy | `scorpio_style_slo_guard` |
| oracle label policy | `scorpio_style_slo_guard` (45.9% of windows) |
| selector WG | 0.9721 |
| winning WG | 0.9956 (SCORPIO fixed) |
| metric gap | −0.0235 |
| scorpio dispatched | 2/172 windows (1.2%) |
| oracle label = scorpio | 79/172 windows (45.9%) |
| suspected pattern | Rules 0/3 require `recent_slo_violation_rate > 0.2/0.3`; this feature is 0.0 in offline evaluation (no violation history before first window). Only Rule 2a fires without violation history. |
| status | **Partially deferred** — offline evaluation artifact; Rules 0/3 are designed for online deployment |
| suggested next action | No code change needed; document that offline eval understates SCORPIO dispatch frequency. For selector training, use only features available online (already satisfied). |

---

### fail_008: Rule selector has no coverage for `best_fit`, `multi_bin_batching`, `shortest_output_first`, `estimated_service_time_first`

| Field | Value |
|-------|-------|
| failure_case_id | fail_008 |
| experiment_id | phase2b12_workload_diversity_selector_labels |
| workload group | Diversity group |
| selector method | rule_based |
| missing policies | `best_fit` (oracle 14×), `shortest_output_first` (oracle 13×), `estimated_service_time_first` (oracle 10×), `multi_bin_batching` (oracle 9×) |
| rule selector dispatch | None of these 4 policies ever dispatched across 172 windows |
| selector WG | 0.9831 (diversity) |
| winning WG | 0.9969 (SCORPIO fixed diversity) |
| diversity group gap | −0.0139 |
| suspected pattern | Diversity workloads (long-decode loose-SLO, high-load no-tight-SLO, low-noise uniform-output) require throughput-packing and SJF-type policies not in the current rule set. Rule 1 (→ WSP) and default Rule 8 (→ EDF) catch these but are suboptimal. |
| completion_fraction | All-complete (1.000) in most diversity windows — low penalty for wrong policy choice |
| status | **Open** — rule selector rules do not cover these policy targets |
| suggested next action | Add rule targets for `best_fit` (long-decode loose-SLO), `multi_bin_batching` (high-load no-tight-SLO), and `estimated_service_time_first` (low-noise moderate-rate). OR train RF/DT with 200+ windows to learn dispatch automatically. |

---

### fail_009: `sarathi_style` rule target is wrong for prefill-heavy workloads

| Field | Value |
|-------|-------|
| failure_case_id | fail_009 |
| experiment_id | phase2b12_workload_diversity_selector_labels |
| workload group | div_prefill_heavy_sarathi, div_prefill_moderate_tight |
| selector method | rule_based |
| selected policy | `sarathi_style` (dispatched 1× across 172 windows) |
| oracle label policy | `admission_control` (wins 16/16 windows across these two workloads) |
| expected oracle | `sarathi_style` (design intent) |
| best fixed WG | 1.000 (tied — all-complete regime) |
| gap (WG difference) | ~0 (all-complete; no meaningful WG penalty) |
| suspected pattern | At moderate load with long prompts, AC's urgency sort + request rejection (for borderline-SLO cases) achieves higher priority-weighted WG than sarathi chunking. Sarathi's chunked-prefill benefit is throughput, not WG when completion is already 100%. |
| status | **Open** — design assumption wrong; AC outperforms sarathi on prefill-heavy regimes under this WG objective |
| suggested next action | Remove or demote Rule 5 (sarathi target) from rule selector. Consider AC for tight-SLO prefill-heavy workloads or default EDF. Verify with additional prefill workloads at higher load. |

---

### fail_010: Window count falls short of RF/DT feasibility threshold (172 < 200)

| Field | Value |
|-------|-------|
| failure_case_id | fail_010 |
| experiment_id | phase2b12_workload_diversity_selector_labels |
| workload group | Overall |
| selector method | RF/DT (not trained) |
| n_windows | 172 |
| threshold | 200 |
| shortfall | 28 windows |
| policy spread | 6 policies ≥10 wins (PASS) |
| concentration | top=scorpio 45.9% (PASS, well below 85%) |
| status | **Open** — diversity criteria met but window count fails |
| suggested next action | Add 2–3 more workload variants or additional seeds (e.g., diversity_seeds=[6,7,8,9,10,11]) to reach ≥200 windows, then train RF/DT. |

---

### fail_011: Label diversity in "all-complete" regimes reflects tie-breaking, not meaningful differentiation

| Field | Value |
|-------|-------|
| failure_case_id | fail_011 |
| experiment_id | phase2b12_workload_diversity_selector_labels |
| workload group | Most diversity workloads (best_fixed_wg = 1.000) |
| selector method | oracle label computation |
| observation | Many diversity workloads achieve WG=1.000 for essentially all 20 policies. The oracle label reflects tie-breaking (lower SLO violation rate, then alphabetical) rather than genuine policy advantage. |
| estimated fraction | Approximately 60–70 of 112 diversity windows may be "all-complete" tie-breaking windows. |
| status | **Open** — documentation issue; RF/DT trained on tie-breaking labels would learn spurious patterns |
| suggested next action | Before RF/DT training, filter out windows where max_wg − min_wg < 0.01 (or similar threshold) to retain only genuinely differentiated windows. Add workload regimes with moderate overload (completion < 0.95) to generate differentiated labels. |

---

## Summary

| ID | Description | Status |
|----|-------------|--------|
| fail_007 | Rule selector under-dispatches SCORPIO (offline artifact) | Partially deferred |
| fail_008 | Missing rule targets: best_fit, multi_bin_batching, SOF, estST | Open |
| fail_009 | sarathi_style rule target wrong; AC wins prefill-heavy | Open |
| fail_010 | 172 < 200 window threshold for RF/DT training | Open |
| fail_011 | All-complete diversity windows have tie-breaking labels | Open |

---

## Failure Case Registry Path

If tracking in CSV: `results/failure_cases/failure_case_registry.csv`

| field | fail_007 | fail_008 | fail_009 | fail_010 | fail_011 |
|-------|---------|---------|---------|---------|---------|
| failure_case_id | fail_007 | fail_008 | fail_009 | fail_010 | fail_011 |
| experiment_id | phase2b12 | phase2b12 | phase2b12 | phase2b12 | phase2b12 |
| status | deferred | open | open | open | open |
| selector_method | rule_based | rule_based | rule_based | rf_dt | oracle |
| workload_group | overall | diversity | prefill_heavy | overall | diversity |
| gap_vs_best_fixed | −0.024 | −0.014 | ~0 | n/a | n/a |
