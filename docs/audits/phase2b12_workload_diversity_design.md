# Phase 2B.12 Workload Diversity Design

**Experiment:** `phase2b12_workload_diversity_selector_labels`  
**Branch:** `phase2b12-workload-diversity-selector-labels`  
**Created from:** Phase 2B.11 (commit `6de9e2b`)  

---

## Motivation

Phase 2B.11 found that `scorpio_style_slo_guard` dominates all 60 evaluation windows from the
Phase 2B.9/2B.10 suite.  The rule selector dispatches to SCORPIO on only 1/60 windows (one
`heldout_very_high_noise` window), making RF/DT training infeasible ("always choose SCORPIO"
is the only learnable function).

The root cause: the 60-window suite covers only high-overload + tight-SLO regimes where
SCORPIO's credit throttling is highly effective.  To train a meaningful selector model, the
label distribution must have ≥3 policies each winning ≥10 windows, with no single policy
winning >80-85% of all windows.

This Phase adds 15 new workloads spanning regimes where simpler policies are expected to be
competitive or dominant, broadening the suite from 60 → ~200 windows.

---

## Diversity Targets

| Target policy | Regime conditions | Mechanism |
|---|---|---|
| `sarathi_style` | Long prompts (768+ tokens), short outputs, moderate load | Chunked prefill eliminates decode stalls; other policies stall on long prefills |
| `weighted_shortest_processing` | Long outputs with loose-SLO or moderate KV pressure | WSP frees KV slots faster; SCORPIO throttles requests that WSP would profitably serve |
| `edf` / `slo_slack_score` | Priority-asymmetric: high-weight loose-SLO class, SCORPIO throttles unnecessarily | SCORPIO's guard penalizes completion without SLO benefit when most WG comes from loose requests |
| `admission_control` | High prediction noise (60-80%), tight SLO | AC's rejection of high-uncertainty requests is more targeted than SCORPIO's credit throttle |
| `estimated_service_time_first` | Low prediction noise, uniform short outputs, tight SLO | Accurate service-time estimation enables near-optimal SJF ordering |
| `scorpio_style_slo_guard` | High overload + tight SLO (carried from Phase 2B.9/2B.11 suite) | Credit throttling prevents SLO cascade failures |

---

## Workload Matrix

### Regression Group (9 workloads — same as Phase 2B.9/2B.11)

Seeds: dev=[0,1,2], heldout=[3,4,5]

| Tag | Group | Rate | Output | Noise | SLO | Expected best |
|---|---|---|---|---|---|---|
| dev_overloaded_mixed_slo | dev | 60 | 96 | 15% | tight=0.4s | scorpio |
| dev_high_prediction_noise | dev | 50 | 96 | 70% | tight=0.5s | scorpio/AC |
| dev_kv_pressure_decode_heavy | dev | 45 | 384 | 25% | tight=0.8s | scorpio/WSP |
| dev_overloaded_prefill_heavy | dev | 40 | 16 | 20% | tight=1.0s | scorpio/sarathi |
| heldout_moderate_kv_pressure | heldout | 50 | 150 | 20% | tight=0.8s | scorpio |
| heldout_very_high_noise | heldout | 50 | 96 | 90% | tight=0.5s | scorpio |
| heldout_prefill_overloaded | heldout | 80 | 16 | 20% | tight=1.0s | scorpio |
| heldout_bursty_mixed_slo | heldout | 60 bursty | 80 | 20% | tight=0.5s | scorpio |
| heldout_burstgpt_smoke | heldout | BurstGPT | — | — | tight=0.5s | scorpio |

### Diversity Group (15 workloads — new, seeds 6-9)

#### Group A: Prefill-heavy → sarathi_style expected winner

| Tag | Rate | Prompt | Output | Noise | SLO | Duration |
|---|---|---|---|---|---|---|
| div_prefill_heavy_sarathi | 22 | 768 | 16 | 10% | all tight=1.5s | 12s |
| div_prefill_moderate_tight | 25 | 512 | 16 | 10% | 60% tight=1.0s | 12s |
| div_prefill_bursty | 30 | 600 | 16 | 10% | 60% tight=1.0s | 10s bursty |

**Why sarathi wins:** Long prompts (768 tokens) require 2+ chunked prefill steps.
Sarathi interleaves prefill chunks with decode; other policies stall on full-prompt prefill,
increasing queueing delay and SLO violations.  At moderate rate (22-30 req/s), SCORPIO's
guard stays passive (few violations), but non-sarathi policies still stall on prefills.

#### Group B: Long decode, loose SLO → WSP expected winner

| Tag | Rate | Prompt | Output | Noise | SLO | Duration |
|---|---|---|---|---|---|---|
| div_decode_all_loose_slo | 35 | 64 | 256 | 15% | all loose=8.0s | 10s |
| div_decode_medium_slo | 40 | 64 | 192 | 15% | tight=3.0s, medium=8.0s | 10s |
| div_decode_kv_saturated | 35 | 64 | 384 | 20% | tight=5.0s (loose for SCORPIO) | 10s |

**Why WSP wins:** With long outputs and all-loose SLO, SCORPIO's guard stays passive (no
violations).  WSP's shortest-first ordering frees KV slots faster than EDF/slo_slack,
enabling higher throughput without any throttling overhead.

#### Group C: Priority-asymmetric / moderate competition → EDF/slo_slack expected winner

| Tag | Rate | Prompt | Output | Noise | SLO | Duration |
|---|---|---|---|---|---|---|
| div_loose_weight_dominant | 45 | 128 | 64 | 15% | tight=0.5s(10%), loose=10s(90%) priority | 10s |
| div_medium_rate_low_noise | 35 | 128 | 64 | 5% | tight=0.3s, medium=1.5s | 10s |
| div_bursty_moderate | 45 bursty | 64 | 80 | 15% | tight=0.5s, medium=3.0s | 10s |
| div_high_load_no_tight_slo | 55 | 128 | 64 | 15% | medium=2.0s, loose=10s | 10s |

**Why EDF/slo_slack wins:**
- `div_loose_weight_dominant`: WG denominator = sum(priorities). Tight has priority=3 but only
  10% of requests; loose has priority=1 but 90% of requests.  Max WG contribution: tight=0.30,
  loose=0.90.  SCORPIO protects tight requests (priority 3) but throttles loose (priority 1),
  sacrificing 0.90 WG for 0.30 WG.  EDF admits all and achieves WG≈1.0 if load allows.
- `div_medium_rate_low_noise`: With 5% noise, slo_slack_score/ESTF accurately estimate
  service times; SCORPIO's blunt throttle is unnecessary at moderate load.
- `div_high_load_no_tight_slo`: No tight-SLO class means SCORPIO's guard rarely activates;
  EDF with 2.0s medium slack is sufficient.

#### Group D: High noise → admission_control expected winner

| Tag | Rate | Prompt | Output | Noise | SLO | Duration |
|---|---|---|---|---|---|---|
| div_high_noise_moderate_slo | 45 | 128 | 96 | 60% | tight=0.4s, medium=4.0s | 10s |
| div_very_high_noise_tight_slo | 40 | 128 | 96 | 80% | tight=0.4s, medium=3.0s | 10s |

**Why AC wins:** At 60-80% prediction noise, service time estimates are unreliable.
Admission control's strategy of rejecting high-uncertainty requests avoids SLO cascade failures
while SCORPIO's credit mechanism may respond too slowly.  These are border cases; AC or SCORPIO
may both compete strongly.

#### Group E: BurstGPT traces → policy winner unknown

| Tag | Source | File | Max requests | Notes |
|---|---|---|---|---|
| div_burstgpt_high_load | extended_jsonl | burstgpt_scaled_high_10k.jsonl | 400 | High-load real traffic |
| div_burstgpt_natural | extended_jsonl | burstgpt_natural_10k.jsonl | 400 | Natural (unscaled) traffic |

**Why included:** Real traffic distributions may have very different temporal patterns than
synthetic workloads.  The winner is unknown; including both provides label diversity potential
and tests generalization to real traces.

---

## Window Count Estimate

| Group | Workloads | Seeds | Combos | Windows/combo | Estimated windows |
|---|---|---|---|---|---|
| Regression dev | 4 | 3 | 12 | ~2.25 | ~27 |
| Regression heldout | 5 | 3 (+ 1 BurstGPT) | 16 | ~2.0 | ~33 |
| Diversity synthetic | 13 | 4 | 52 | ~2.5 | ~130 |
| Diversity BurstGPT | 2 | 1 | 2 | ~2.0 | ~4 |
| **Total** | **24** | | **82** | | **~194** |

Target: ≥200 windows.  Actual count depends on Poisson arrival variance per seed.

---

## Label Diversity Criterion for RF/DT Training

RF/DT training proceeds only if **all three** criteria are met:

1. **Window count:** ≥200 total windows across all groups
2. **Policy spread:** ≥3 distinct policies each winning ≥10 windows as per-window oracle
3. **Concentration:** No single policy wins >80-85% of all windows

If SCORPIO still dominates (>85%), the diversity is insufficient and RF/DT is deferred.
Further workload engineering or additional baselines (e.g., PARS-style LTR) are required.

---

## Experiment Configuration Summary

- `window_size: 200`, `min_partial_window: 50`
- `feature_mode: online_prefix`
- Simulator: `step_size=0.001`, `drain_steps=20000`
- GPU: `max_active=4`, `max_kv_tokens=32768`, `max_batch_tokens=4096`
- Deployable policies: 20 (same as Phase 2B.11)
- Rule selector: Phase 2B.11 version (3 SCORPIO rules)
- Seeds: dev=[0,1,2], heldout=[3,4,5], diversity=[6,7,8,9]

---

## Phase 2B.11 Reference (60-window suite)

| Group | Rule selector WG | SCORPIO fixed WG | Gap |
|---|---|---|---|
| Dev | 0.9168 | 0.9878 | −0.071 |
| Held-out | 0.9803 | 0.9975 | −0.017 |
| Overall | 0.9518 | 0.9932 | −0.041 |

SCORPIO won all 60 windows as per-window oracle.  Rule selector dispatched to SCORPIO 1/60 times.

---

## Actual Results (Phase 2B.12 Completed)

**tmux session:** `phase2b12_workload_diversity` — completed, EXIT_CODE=0, ~638s

### Window count

| Group | Actual windows | Estimate |
|-------|---------------|---------|
| dev | 27 | ~27 |
| heldout | 33 | ~33 |
| diversity | 112 | ~134 |
| **overall** | **172** | ~194 |

Target (≥200) not reached: **28 windows short.**

### Label diversity

| Policy | Oracle wins | Fraction |
|--------|------------|---------|
| `scorpio_style_slo_guard` | 79 | 45.9% |
| `admission_control` | 29 | 16.9% |
| `best_fit` | 14 | 8.1% |
| `edf` | 14 | 8.1% |
| `shortest_output_first` | 13 | 7.6% |
| `estimated_service_time_first` | 10 | 5.8% |
| `multi_bin_batching` | 9 | 5.2% |
| `random_feasible` | 3 | 1.7% |
| `shortest_prompt_first` | 1 | 0.6% |

### RF/DT feasibility outcome

| Criterion | Required | Actual | Status |
|-----------|----------|--------|--------|
| Window count | ≥200 | 172 | FAIL |
| Policies ≥10 wins | ≥3 | 6 | PASS |
| Top policy fraction | <85% | 45.9% | PASS |

**RF/DT NOT TRAINED** (fails window count only).

### Design assumption corrections

1. **Prefill-heavy regime:** `admission_control` wins (not `sarathi_style`).
   AC's urgency sort outperforms chunked prefill under priority-weighted WG objective.
2. **Long-decode loose-SLO:** `best_fit` wins (not `weighted_shortest_processing`).
   Best-fit bin packing maximizes batch utilization better than WSP in loose-SLO context.
3. **High-load no-tight-SLO:** `multi_bin_batching` and `estimated_service_time_first` win
   (not `edf`). Throughput-packing beats pure deadline ordering when all SLOs are loose.

### Rule selector results

| Group | Rule WG | Best fixed | Gap |
|-------|---------|------------|-----|
| regression | 0.9518 | 0.9932 | −0.041 |
| diversity | 0.9831 | 0.9969 | −0.014 |
| **overall** | **0.9721** | **0.9956** | **−0.024** |

Full analysis: `docs/audits/phase2b12_rule_selector_diversity_evaluation.md`  
Label diversity full analysis: `docs/audits/phase2b12_selector_label_diversity_summary.md`  
Failure cases: `docs/audits/phase2b12_failure_cases_summary.md`
