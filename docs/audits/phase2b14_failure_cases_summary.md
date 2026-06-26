# Phase 2B.14 Failure Cases

**Date:** 2026-06-26

---

## fail_014_metric_inflation

**Pattern:** Old WG metric was `completed_request_quality`, not arrival-normalized goodput.

**Status:** Confirmed and documented.

**Detail:** SCORPIO conditional WG = 0.9846; arrival-normalized WG = 0.8885. Gap = 0.0961. The metric inflated SCORPIO's apparent performance because its completion fraction (~0.899) reduces the denominator, boosting the ratio. FIFO/WSP have CF≈0.99 so their conditional vs arrival-norm values nearly coincide.

**Impact:** Phase 2B.10–2B.13 ranking tables showed SCORPIO with misleadingly large leads. The lead is real but smaller (0.0345 over WSP under arrival-norm WG vs 0.1275 under conditional WG).

**Resolution:** Arrival-normalized WG added. Old metric renamed to `completed_request_quality`. All rankings recomputed. Phase 2B.10–2B.13 claims flagged for reinterpretation.

---

## fail_015_completion_penalized_reversal

**Pattern:** SCORPIO dominance disappears under completion-penalized objectives.

**Status:** Confirmed.

**Detail:** Under `cp_wg_t095_l05` (arrival-norm − 0.5×max(0, 0.95−CF)):
- SCORPIO: 0.8503 (rank 2)
- WSP: 0.8524 (rank 1)

Under `cp_wg_t099_l10` (arrival-norm − 1.0×max(0, 0.99−CF)):
- SCORPIO: 0.7937 (rank 6+)
- WSP: 0.8480 (rank 1)

**Impact:** If the deployment objective includes a completion-rate target ≥ 0.95, SCORPIO is not the recommended policy. WSP, SOF, or VLLMStyleTokenBudget are better choices.

**Resolution:** Completion-penalized metrics added with four (target, lambda) configurations. Rankings under all configurations reported. Safe/unsafe claims updated.

---

## fail_016_near_tie_labels_metric_artifact

**Pattern:** Many near-tie labels under conditional WG were artifacts of all-complete windows, not metric discrimination.

**Status:** Partially resolved.

**Detail:** Under conditional WG: 93% of windows were all-complete (best WG ≥ 0.99). Under arrival-normalized WG: only 64% are all-complete. Near-tie at ε=0.001: 70% (vs 74% old). Meaningful windows: 97 (vs ~84 old).

**Impact:** Selector training had ~84 meaningful windows under old metric, now ~97 under corrected metric. The training quality was slightly understated.

**Resolution:** Near-tie analysis rerun under arrival-normalized WG. Results show slight improvement. Selector conclusions unchanged.

---

## fail_017_selector_label_metric_mismatch

**Pattern:** Selectors trained on conditional WG labels may have learned to prefer high-rejection policies.

**Status:** Partially mitigated.

**Detail:** SCORPIO's high conditional WG made it the best-label winner in 55% of windows. Selectors trained on these labels may have implicitly learned "choose SCORPIO when in doubt." Under arrival-normalized WG, selectors still beat always-SCORPIO (+0.0059 for RF), suggesting the mismatch does not harm selector performance in practice.

**Impact:** Selector training is defensible as-is (non-SCORPIO policies have CF≈1, so their labels are unaffected). SCORPIO-heavy windows add noise but not bias for non-SCORPIO policies.

**Resolution:** Selector scores confirmed valid under arrival-normalized WG. No retraining required.

---

## fail_018_ablation_laxity_filter_critical

**Pattern:** SCORPIO's laxity pre-filter is the single most critical component; removing it causes catastrophic WG failure.

**Status:** Confirmed by ablation.

**Detail:** On targeted discriminative workloads (KV-saturated, overloaded):
- `scorpio_no_laxity_filter`: arrival-norm WG = 0.000 (gap = -0.688 vs full SCORPIO)
- `scorpio_no_rejection`: arrival-norm WG = 0.000 (gap = -0.688)
- `scorpio_no_kv_guard`, `scorpio_no_credit_budget`: gap ≈ 0.0 (negligible)
- `scorpio_deadline_only` (laxity filter alone): arrival-norm WG = 0.6862 (gap = -0.0017 — nearly full performance)

**Interpretation:** SCORPIO's admission throttling (KV guard, credit budget) adds negligible value. The laxity pre-filter alone accounts for nearly all of SCORPIO's discriminative advantage. Without it, requests with near-zero laxity flood the KV cache and the system deadlocks.

**Impact:** SCORPIO's core value is a deadline-based pre-filter, not a sophisticated multi-component admission throttle. The credit budget and KV guard are redundant in the scenarios where SCORPIO matters most. A simplified `scorpio_deadline_only` policy achieves nearly identical performance.

**Resolution:** Documented in ablation summary. No policy changes needed.

---

## fail_019_admission_fraction_not_tracked

**Pattern:** `num_admitted` is not separately tracked in RunMetrics; only `num_completed` and `num_dropped` are available.

**Status:** Acknowledged, will not fix in Phase 2B.14.

**Detail:** For policies without mid-execution drops (all current policies), `admission_normalized_wg ≈ arrival_normalized_wg`. The distinction would matter only for preemptive policies. No fix needed for current policy suite.

**Resolution:** Documented in metric definition audit. Future work: add `num_admitted` tracking if preemptive policies are added.

---

## fail_020_oracle_srtf_exclusion

**Pattern:** oracle_srtf must remain excluded from selector candidates at all times.

**Status:** Confirmed (no regression).

**Detail:** oracle_srtf uses hindsight information (actual output tokens). It is a non-deployable upper bound. Phase 2B.14 adds no oracle candidates to any candidate list.

**Resolution:** Registry check confirms oracle_srtf not in SELECTOR_CANDIDATES or BASELINE_NAMES.
