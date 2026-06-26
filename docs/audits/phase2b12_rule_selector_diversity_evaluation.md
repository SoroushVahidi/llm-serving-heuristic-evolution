# Phase 2B.12 Rule Selector Diversity Evaluation

**Experiment:** `phase2b12_workload_diversity_selector_labels`  
**Branch:** `phase2b12-workload-diversity-selector-labels`  
**Config:** `configs/phase2b12_workload_diversity_selector_labels.yaml`  
**Results:** `results/phase2b12_workload_diversity_selector_labels/` (gitignored; summary in this doc)  
**tmux session:** `phase2b12_workload_diversity` (completed, EXIT_CODE=0)

---

## Purpose

Evaluate the Phase 2B.11 rule selector (7 routing rules, 20 candidates) on the Phase 2B.12
diversity workload suite (172 windows: 60 regression + 112 diversity).  Assess whether the
rule selector dispatches to multiple policies across diverse regimes and recovers value relative
to the best fixed baseline.

---

## Rule Selector (Phase 2B.11 version)

Rules in priority order:

| # | Condition | Policy |
|---|-----------|--------|
| 0 | `(tight_slo OR low_slack) AND violation_rate>0.2` | `scorpio_style_slo_guard` |
| 1 | `mean_pred_output>200 OR kv_util>0.7` | `weighted_shortest_processing` |
| 2a | `pred_output_cv>2.0` | `scorpio_style_slo_guard` |
| 2b | `pred_output_cv>1.0` | `admission_control` |
| 3 | `violation_rate>0.3` | `scorpio_style_slo_guard` |
| 4 | `tight_slo OR low_slack` | `slo_slack_score` |
| 5 | `mean_prompt>200` | `sarathi_style` |
| 6 | `pred_output_cv<0.5 AND mean_pred_output<64` | `estimated_service_time_first` |
| 7 | `burstiness_cv>1.5` | `slo_slack_score` |
| 8 | default | `edf` |

---

## Weighted Goodput Results

### Phase 2B.12 vs Phase 2B.11

| Group | n_windows | Rule selector WG | Best fixed WG | Gap vs best fixed | Gap vs oracle |
|-------|-----------|-----------------|---------------|------------------|---------------|
| **dev** | 27 | **0.9168** | **0.9878** | **−0.0710** | −0.0713 |
| **heldout** | 33 | **0.9803** | **0.9975** | **−0.0172** | −0.0192 |
| regression (dev+heldout) | 60 | 0.9518 | 0.9932 | −0.0414 | −0.0426 |
| **diversity** | 112 | **0.9831** | **0.9969** | **−0.0139** | −0.0160 |
| **overall** | 172 | **0.9721** | **0.9956** | **−0.0235** | −0.0253 |

**Phase 2B.11 reference (regression only):**

| Group | Rule selector (P2B.11) | Best fixed (P2B.11) | Gap (P2B.11) |
|-------|----------------------|---------------------|-------------|
| dev | 0.9168 | 0.9878 | −0.071 |
| heldout | 0.9803 | 0.9975 | −0.017 |
| overall (60 windows) | 0.9518 | 0.9932 | −0.041 |

**Regression WG matches Phase 2B.11 exactly** — the rule selector behavior is unchanged on
known workloads.  Diversity group shows smaller gap (−0.014) than regression (−0.041), indicating
the rule selector performs relatively better on diverse regimes even though SCORPIO remains the
best fixed baseline.

### Best fixed policy by group

| Group | Best fixed policy | WG |
|-------|-------------------|-----|
| dev | `scorpio_style_slo_guard` | 0.9878 |
| heldout | `scorpio_style_slo_guard` | 0.9975 |
| regression | `scorpio_style_slo_guard` | 0.9932 |
| diversity | `scorpio_style_slo_guard` | 0.9969 |
| overall | `scorpio_style_slo_guard` | 0.9956 |

SCORPIO is the best fixed policy in all groups, even though it wins the per-window label only
45.9% of the time overall.  This means SCORPIO achieves high WG even on windows where other
policies win per-window: SCORPIO's mean WG of 0.9956 beats any single alternative fixed policy.

### Per-window oracle reference WG

| Group | Oracle WG | Best fixed WG | Oracle − best fixed |
|-------|-----------|---------------|---------------------|
| dev | 0.9881 | 0.9878 | +0.0003 |
| heldout | 0.9995 | 0.9975 | +0.0020 |
| regression | 0.9944 | 0.9932 | +0.0012 |
| diversity | 0.9990 | 0.9969 | +0.0021 |
| overall | 0.9974 | 0.9956 | +0.0018 |

Oracle improvement over best fixed is very small (0.001–0.002 WG), meaning **the selector's
headroom** — what it can gain by choosing optimally per-window — is narrow.  However, the
selector currently loses to best fixed (−0.024 overall), so there is room for improvement.

---

## Rule Selector Chosen-Policy Distribution

### Overall (n=172 windows)

| Policy | Dispatched | Fraction |
|--------|-----------|---------|
| `slo_slack_score` | 93 | 54.1% |
| `weighted_shortest_processing` | 29 | 16.9% |
| `admission_control` | 28 | 16.3% |
| `edf` | 19 | 11.0% |
| `scorpio_style_slo_guard` | 2 | 1.2% |
| `sarathi_style` | 1 | 0.6% |

### By group

| Policy | Regression (n=60) | Diversity (n=112) |
|--------|-------------------|-------------------|
| `slo_slack_score` | 40 | 53 |
| `admission_control` | 11 | 17 |
| `weighted_shortest_processing` | 8 | 21 |
| `scorpio_style_slo_guard` | 1 | 1 |
| `edf` | 0 | 19 |
| `sarathi_style` | 0 | 1 |

---

## Does the Rule Selector Choose SCORPIO Appropriately?

**No — severely under-dispatches to SCORPIO.**

- SCORPIO dispatched **2/172 windows** (1.2%) despite being the oracle label in 79 windows (45.9%).
- SCORPIO dispatch rate on regression: 1/60 (1.7%), same as Phase 2B.11.
- SCORPIO dispatch rate on diversity: 1/112 (0.9%).

**Root causes (unchanged from Phase 2B.11):**
1. Rules 0 and 3 require `recent_slo_violation_rate > 0.2/0.3`, which is 0.0 in offline windows
   (no prior violation history before the window's first evaluation).
2. Only Rule 2a (`pred_output_cv > 2.0`) fires for SCORPIO without violation history; this fires
   once or twice across all 172 windows.

This is an **offline evaluation artifact** — online deployment accumulates violation history
naturally, enabling Rules 0 and 3 to fire.

---

## Does the Rule Selector Over-Choose SCORPIO?

**No — the opposite problem: it under-chooses SCORPIO.**

On the 79 windows where SCORPIO is the oracle label, the rule selector picks slo_slack_score
or admission_control most of the time.  slo_slack_score (54% of dispatches) is the workhorse
fallback but is not the best policy.

---

## Does the Rule Selector Recover Value in Non-SCORPIO Regimes?

**Partially.**

- Diversity group gap: −0.014 (better than regression −0.041).
- The rule selector dispatches WSP (16.9%), AC (16.3%), and EDF (11.0%) — policies that are
  relevant for diversity regimes.
- However, the rule selector never dispatches to:
  - `best_fit` (oracle label for 14 windows)
  - `estimated_service_time_first` (oracle label for 10 windows)
  - `multi_bin_batching` (oracle label for 9 windows)
  - `shortest_output_first` (oracle label for 13 windows)

These gaps represent **rule coverage failures** for specific diversity regimes.

---

## Which Rules Fail?

### Rule 5: `mean_prompt > 200` → `sarathi_style`

Expected to fire on prefill-heavy workloads.  Only dispatched **once** across 172 windows.
The prefill-heavy diversity workloads (`div_prefill_*`) have `mean_prompt ≈ 600-768` tokens,
but the oracle label is `admission_control` (8/8 wins for `div_prefill_heavy_sarathi` and
`div_prefill_moderate_tight`), not `sarathi_style`.  Rule 5 may be firing correctly but
`sarathi_style` is suboptimal in these regimes — the rule's policy target is wrong.

### Missing Rule: No dispatch to `best_fit`

`best_fit` wins 14 windows (primarily `div_decode_all_loose_slo`) but is not in the rule
selector's policy choices at all.  The decode-heavy loose-SLO regime is served by Rule 1
(→ WSP), which is suboptimal here.

### Missing Rule: No dispatch to `multi_bin_batching`

`multi_bin_batching` wins 9 windows (primarily `div_high_load_no_tight_slo`) but is not in
the rule selector.  The high-load no-tight-SLO regime triggers Rule 4 (→ slo_slack_score)
or falls to the EDF default, both of which miss the throughput-packing opportunity.

### Missing Rule: No dispatch to `estimated_service_time_first`

Rule 6 (`pred_output_cv < 0.5 AND mean_pred_output < 64` → estST) fires rarely.  The regime
where estST wins (`div_loose_weight_dominant`, `div_medium_rate_low_noise`) has low noise and
medium output length, which does not trigger Rule 6's strict output-length threshold.

### Missing Rule: No dispatch to `shortest_output_first`

SOF wins 13 windows but there is no rule targeting it.  Low-noise uniform-output regimes tend
to route to estST (Rule 6) or slo_slack_score (Rule 4/7), missing the exact-SJF efficiency
of SOF.

---

## Does the Rule Selector Remain Worse Than Best Fixed?

**Yes, by −0.024 WG overall** (−0.041 on regression, −0.014 on diversity).

The selector does better on diversity than regression because diversity workloads are often
less differentiated (many all-complete windows), so the penalty for wrong policy choices is
smaller.

The regression deficit (−0.041) is structurally due to SCORPIO dispatching only 1/60 times
when SCORPIO is the oracle label for 46/60 windows.

---

## Selector Accuracy (Per-Window Label Match)

| Group | Accuracy |
|-------|---------|
| heldout | 0.030 (1/33 windows where selector choice matches oracle label) |

Full accuracy computation requires per-window label vs selector choice matching; the runner
reports it for heldout only.  The low accuracy (3%) reflects that slo_slack_score dominates
selector dispatches while only 4/33 heldout windows have slo_slack_score as oracle label.

---

## Summary: Selector Performance on Phase 2B.12 Suite

| Question | Answer |
|----------|--------|
| Rule selector WG overall | **0.9721** |
| Rule selector WG regression | **0.9518** (same as Phase 2B.11) |
| Rule selector WG diversity | **0.9831** |
| Best fixed WG overall | **0.9956** (SCORPIO) |
| Gap vs best fixed overall | **−0.0235** |
| Gap vs per-window oracle overall | **−0.0253** |
| SCORPIO dispatched | **2/172 windows (1.2%)** |
| SCORPIO oracle label fraction | **45.9%** |
| Policies not covered by rules | `best_fit`, `multi_bin_batching`, `shortest_output_first`, `estimated_service_time_first` |
| Main failure mode | Structural under-dispatch to SCORPIO (offline artifact) + missing rules for throughput-packing policies |
| RF/DT training recommended | **No — 172 < 200 window threshold; expand suite first** |

---

## Recommended Next Steps

**Priority 1:** Expand window count to ≥200 by adding seeds or new workload variants, then
train RF/DT selector on the diversified suite.  The diversity criterion (policy spread, label
concentration) is already met.

**Priority 2:** Update rule selector to cover `best_fit`, `multi_bin_batching`, and
`estimated_service_time_first` as dispatch targets for high-load loose-SLO regimes.

**Priority 3:** Consider PARS-style LTR as the next external baseline (per
`docs/external_baseline_decision.md`).  PARS would compete in exact-SJF / low-noise regimes
where estST and SOF currently win.

---

## Related Documents

| Document | Purpose |
|----------|---------|
| `docs/audits/phase2b12_selector_label_diversity_summary.md` | Label diversity analysis |
| `docs/audits/phase2b12_failure_cases_summary.md` | Failure cases |
| `docs/audits/phase2b11_scorpio_selector_integration_summary.md` | Phase 2B.11 comparison base |
| `docs/selector.md` | Selector architecture |
