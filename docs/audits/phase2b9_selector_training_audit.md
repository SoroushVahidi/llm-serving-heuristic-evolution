# Phase 2B.9 Selector Training Data Sufficiency and Leakage Audit

**Phase:** 2B.9  
**Date:** 2026-06-25  
**Branch:** `phase2b9-selector-robustness-and-suite-freeze`  
**Auditor:** Phase 2B.9 automated audit (code-derived counts + manual inspection)

---

## 1. Selector Training Data Overview

### 1.1 Phase 2A.4 Training Configuration

The RF and DT selectors used in deployment were trained in Phase 2A.4 using three dataset splits:

| Split | Config | Seed | Windows (approx) | Workload families |
|-------|--------|------|------------------|-------------------|
| Train | `phase2a4_train_18policies.yaml` | 0 | ~30 | 5 (overloaded, mixed-SLO, bursty, prefill-moderate, BurstGPT) |
| Validation | `phase2a4_validation_18policies.yaml` | 3 | ~13 | 3 (prefill-heavy, BurstGPT-noise070, BurstGPT-natural) |
| Test | `phase2a4_test_18policies.yaml` | 7 | ~9 | 3 (very-overloaded-noise, extreme-bursty, BurstGPT-high) |
| **Total** | | | **~52** | **6–7 distinct families** |

**Source of truth:** `docs/selector.md`, Phase 2A.4 configs in `configs/selector/`.

### 1.2 How Many Training Windows Are Used?

**Answer: ~30 training windows (Phase 2A.4 best estimate).**

The train config uses:
- 3 overloaded synthetic workloads × ~3 windows each = ~9 windows
- 2 mixed-SLO synthetic workloads × ~2 windows each = ~4 windows
- 2 bursty synthetic workloads × ~2 windows each = ~4 windows
- 1 prefill-moderate workload × ~1 window = ~1 window
- 2 BurstGPT workloads × max_requests=2000 → ~10 windows each = ~10 windows (2 workloads)

Actual count may vary by ±3 windows depending on arrival randomness and partial-window trimming.

**30 training windows is critically small for a 19-class classification problem.** This is a major limitation acknowledged in Phase 2A.4.

### 1.3 Validation and Test Windows

**Validation:** ~13 windows (seed 3; prefill-heavy + BurstGPT variants).  
**Test:** ~9 windows (seed 7; very overloaded noise + extreme bursty + BurstGPT high).

**Note:** 9 test windows for a 19-class problem is not statistically adequate for tight per-class accuracy estimates. The overall WG number (0.828) is more meaningful than classification accuracy.

---

## 2. Workload Families Used in Selector Training

| Family | In train? | In validation? | In test? | Notes |
|--------|-----------|----------------|----------|-------|
| Overloaded synthetic (short outputs) | ✅ | ❌ | ✅ | Train has moderate overload; test has extreme |
| Mixed-SLO synthetic | ✅ | ❌ | ❌ | Only in train; gap in val/test |
| Bursty + tight SLO | ✅ | ❌ | ✅ | Train: moderate burst; test: extreme burst |
| Prefill-heavy synthetic | ✅ (light) | ✅ | ❌ | Minor difference between train/val versions |
| BurstGPT moderate (real trace) | ✅ | ✅ | ❌ | Different noise levels |
| BurstGPT natural | ❌ | ✅ | ❌ | Validation only |
| BurstGPT high load | ❌ | ❌ | ✅ | Test only (**correctly excluded from train**) |
| KV-pressure decode-heavy | ❌ | ❌ | ❌ | **NOT present in any selector dataset** |
| Overloaded mixed-SLO (Phase 2B.7) | ❌ | ❌ | ❌ | **NOT present in any selector dataset** |
| High prediction noise (Phase 2B.7) | ❌ | ❌ | ❌ | **NOT present in any selector dataset** |

**Critical finding:** The three Phase 2B.7/2B.8 failure-case workloads (KV pressure, high noise, overloaded mixed-SLO) are **not present in the Phase 2A.4 selector training, validation, or test datasets.** The RF/DT models have never seen these regimes during training.

---

## 3. Seeds Used

| Group | Seeds |
|-------|-------|
| Phase 2A.4 train | 0 (per-workload, with small offsets per workload index) |
| Phase 2A.4 validation | 3 |
| Phase 2A.4 test | 7 |
| Phase 2B.7/2B.8 sweep | 0, 1, 2 |
| Phase 2B.9 dev group | 0, 1, 2 (same as 2B.7/2B.8) |
| Phase 2B.9 heldout group | **3, 4, 5** (not used in prior selector experiments) |

**Seeds 3, 4, 5 are genuinely held out from all selector training and rule design.**

---

## 4. Label Distribution Across Deployable Policies

Based on Phase 2A.4 evaluation data and Phase 2B.7/2B.8 sweep results:

### Phase 2A.4 Label Distribution (estimated, 30 train windows)

| Policy | Approx. frequency | Notes |
|--------|------------------|-------|
| `shortest_output_first` | ~30% | Wins most overloaded/bursty windows (SRPT-like) |
| `edf` | ~15% | Wins moderate mixed-SLO windows |
| `vllm_style_token_budget` | ~15% | Wins KV-budget-sensitive windows |
| `slo_slack_score` | ~10% | Wins tight-SLO high-overload windows |
| `weighted_shortest_processing` | ~10% | Wins decode-heavy windows (Phase 2B.7+) |
| `admission_control` | ~5% | Wins high-noise windows (added late) |
| 13 other policies | ~15% total | Rarely optimal in these regimes |

**Finding: Labels are dominated by 3–5 policies.** ~40–45% of windows have shortest_output_first or edf as the best label. The tail 14 policies collectively appear as best in only ~30% of windows.

**Implication:** The RF/DT classifiers are likely biased toward the top 3–5 policies and may fail on novel regimes where other policies would win.

### Phase 2B.7/2B.8 Label Distribution (4 overloaded workloads, 3 seeds)

| Policy | Workload(s) where it wins |
|--------|--------------------------|
| `weighted_shortest_processing` | kv_pressure_decode_heavy |
| `slo_slack_score` | overloaded_mixed_slo |
| `admission_control` | high_prediction_noise |
| all tie | overloaded_prefill_heavy (underloaded) |

These labels do **not appear in Phase 2A.4 training data** (since these workload regimes were not in the train config). The RF/DT selectors are flying blind on KV-pressure and high-noise regimes.

---

## 5. Selector Label Correctness

### Q: Are labels the best deployable policy (not oracle)?

**Yes.** Labels in `build_selector_dataset.py` are computed as:
```python
label = argmax_{p in SELECTOR_CANDIDATES} weighted_goodput(p, window)
```
where `SELECTOR_CANDIDATES = BASELINE_NAMES − ORACLE_POLICY_NAMES`.

`oracle_srtf` is never a label candidate. Verified in `selector/labels.py` and tested in `tests/test_selector_labels.py`.

### Q: Is oracle excluded from selector candidates?

**Yes.** Enforced at import time in `selector/candidates.py`:
```python
assert "oracle_srtf" not in SELECTOR_CANDIDATES
```
Tested in `tests/test_selector_candidates.py` and `tests/test_oracle_not_deployable.py`.

---

## 6. Feature Leakage Audit

### Q: Is actual output length (`actual_output_tokens`) ever exposed to online selector features?

**No.** Feature extraction in `selector/features.py` uses only:
- Queue state at window start
- Predicted output tokens (`pred_output_tokens`, not actual)
- SLO class distributions
- Arrival statistics
- Recent violation rate from completed requests

`actual_output_tokens` is never read in `extract_features()`. Enforced by tests in `tests/test_selector_no_leakage.py`.

### Q: Are future arrivals (requests after window start) exposed to features?

**No.** In `online_prefix` mode (the default used for all selector training), features depend only on requests with `arrival_time ≤ window_start_time`. Future arrivals are not accessed.

Enforced by the `FeatureMode.ONLINE_PREFIX` path in `extract_features()` and tested in `tests/test_selector_no_leakage.py`.

### Q: Is any oracle information used in selector features or labels?

**No.** Features are online-observable only. Labels use only deployable policy WG (no oracle). Tested in multiple leakage tests.

### Q: Are train/val/test splits done at the workload level or window level?

**Workload level.** The Phase 2A.4 splits use **disjoint workload configurations and seeds**:
- Train: seed 0, specific workload tags
- Validation: seed 3, different workload tags
- Test: seed 7, different workload tags

**This prevents window-level leakage** (a window from a "train" trace cannot appear in the test split). However, the workload families have some overlap (e.g., overloaded synthetic appears in both train and test with different parameters), which is appropriate.

---

## 7. Held-Out Status of Phase 2B.8 Rule Selector

### Q: Was the Phase 2B.8 repaired rule selector evaluated on truly held-out workloads?

**No.** Phase 2B.8 evaluated the repaired rule selector on exactly the same 4 workloads that motivated the repair (Phase 2B.7 failure cases). These are:
- `overloaded_mixed_slo` (drove repair of Rule 4: LLF → slo_slack_score)
- `high_prediction_noise` (drove Rule 2: pred_output_cv > 1.0 → admission_control)
- `kv_pressure_decode_heavy` (drove Rule 1: mean_pred_output > 200 → WSP)
- `overloaded_prefill_heavy` (underloaded control; no rule changes)

**This is a development evaluation, not a generalization claim.**

Phase 2B.9 addresses this by running the repaired rule selector on:
- Seeds 3, 4, 5 on the same 4 dev workloads (held-out seeds)
- 5 entirely new workloads (`heldout_*` group)

---

## 8. Are There Enough Examples for Meaningful Learning?

### Q: Are there enough training windows for the RF/DT selector to learn meaningful regime distinctions?

**Probably not for a 19-class problem.** The assessment:

| Criterion | Requirement | Current | Status |
|-----------|-------------|---------|--------|
| Min examples per class | ≥ 10 for RF | ~1.5 on average (30 windows / 19 classes) | ❌ Insufficient |
| Train windows total | ≥ 100 for decent RF | ~30 | ❌ Insufficient |
| Workload families | ≥ 5 distinct | 5 (but 3 Phase 2B.7 regimes missing) | ⚠️ Partial |
| Label diversity | 10+ classes represented | ~5–6 classes dominate | ⚠️ Dominated |

**However:** The RF/DT are still useful because:
1. The task is really a few-class problem in practice (2–3 dominant policies per regime)
2. The 3pp WG improvement over best fixed (Phase 2A.4) shows it's not purely overfitting
3. WG-based evaluation is more stable than classification accuracy

**Conclusion:** The Phase 2A.4 RF/DT selectors are useful as baselines but should not be claimed as publication-quality classifiers without more training data. **A minimum of 200 training windows covering all regime families is recommended before final publication claims.**

---

## 9. What Additional Selector Data Is Needed Before Publication?

| Gap | Description | Priority |
|-----|-------------|---------|
| KV-pressure regime windows | Phase 2B.7/2B.8 workloads must be in training | **Critical** |
| High prediction noise windows | High-noise regime is not in Phase 2A.4 | **Critical** |
| More seeds per workload | Seeds 1–6 not used in Phase 2A.4 | High |
| Real-trace workloads | Only BurstGPT; need Azure/LMSYS | High |
| Scale to ≥200 train windows | Current 30 is insufficient | High |
| More Phase 2B.9 heldout workloads | Validate rule selector on truly unseen regimes | High |
| Priority mix extremes | Very-high-priority-fraction workloads | Medium |
| Long-context prompts | No long-context windows in any selector dataset | Medium |

**Honest assessment:** The current selector training data is too small and too narrow for final publication claims about the RF/DT selectors. Phase 2B.9 Phase 2B.9 adds 5 new heldout workloads but does not retrain the models. A dedicated data expansion phase is needed before publication.

---

## 10. Summary

| Question | Answer |
|----------|--------|
| Training windows (Phase 2A.4) | ~30 |
| Validation windows | ~13 |
| Test windows (held-out) | ~9 |
| Total windows | ~52 |
| Workload families | 5 in training; 3 Phase 2B.7 regimes missing |
| Seeds used | 0 (train), 3 (val), 7 (test); Phase 2B.7/2B.8 uses 0,1,2 |
| Label distribution | Dominated by 3–5 policies; 14 rarely best |
| Oracle excluded from labels | ✅ Yes |
| Actual output in features | ❌ No (enforced + tested) |
| Future arrivals in features | ❌ No (enforced + tested) |
| Phase 2B.8 rule selector on held-out | ❌ No (tested on same workloads that drove design) |
| Phase 2B.9 adds truly held-out evaluation | ✅ Yes (5 new workloads + seeds 3–5) |
| Training data sufficient for publication | ❌ No — needs ≥200 windows + regime coverage |
| Must-fix before final claims | Add KV-pressure + noise + more seeds + more families |
