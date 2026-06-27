# Phase 2C.3 / Labeled Dataset / API Calibration Audit

**Date:** 2026-06-27
**Branch:** phase2c1-real-trace-ingestion-validation
**Tests:** 65/65 new tests pass; 91/91 total (incl. Phase 2C.2 and selector-feature regressions)

---

## 1. Phase 2C.3 — External-Aware Orca Recovery

**Result: Valid negative finding.**

- **Goal:** Determine whether adding external-style policies (orca_style, sarathi_style, splitfuse_style, vllm_style_token_budget, multi_bin_batching, estimated_service_time_first) to the training pool recovers orca's advantage in azure_2023_conv windows.
- **Structural finding:** In the full training pool, orca_style never wins as the best policy in any window. Therefore, the external-aware pool (`external_aware_non_oracle`) has **zero** orca_style training labels after near-tie filtering. The trained DT on this pool is numerically identical to the native non-oracle DT.
- **Label counts (external_aware pool after near-tie filter):** scorpio=68 (88%), admission_control=7 (9%), wsp=2 (3%), orca=0 (0%).
- **Best selector ANWG:** `native_non_oracle_dt` = 0.8063 (vs Phase 2C.2 baseline 0.8021, delta +0.0042).
- **External-style envelope:** 0.8297 (62 windows where envelope > dt_anwg selector). The gap remains.
- **Orca selected by best selector:** 0 times (0%).
- **Conclusion:** Adding external-style policies to the training pool does not recover orca. The external envelope advantage (especially in azure_2023_conv) is not addressable via simple pool expansion; it requires a feature-conditioned regime gate or targeted training data generation for long-prompt + mixed-SLO windows.
- **Phase 2C.2 reproduction check:** All four reference values matched exactly (within 5e-4 tolerance):
  - dt_anwg: 0.802110 ✓
  - always_scorpio: 0.796270 ✓
  - external_style_envelope: 0.829684 ✓
  - external_loss_windows: 62 ✓

### Safe claims
- native_non_oracle_dt outperforms always_scorpio by +0.0059 ANWG on eval.
- Adding external-style policies to training pool does not hurt performance.

### Unsafe claims (do not make without further evidence)
- Orca is categorically worse than scorpio in all regimes (orca beats scorpio 212/611 windows by pairwise ANWG).
- The azure_2023_conv gap is permanent; targeted training could close it.

---

## 2. Gemini API Calibration Infrastructure

**Status: Dry-run only. No live API call made.**

- **Config:** `configs/api_calibration/gemini_minimal_v1.yaml`
- **Script:** `scripts/run_gemini_api_calibration.py`
- **Tests:** 22/22 pass (includes: no SDK import at module level, dry-run works without credentials, live mode refuses without `--allow-live-api`, credentials check before any call, manifest has no secrets).
- **Planned calls:** 24 (3 prompt buckets × 2 output buckets × 2 concurrency groups × 2 repeats).
- **Hard cap:** max_calls = 50; max_prompt_tokens_per_call = 2048; max_output_tokens_per_call = 512; estimated_budget_usd = $0.10.
- **Worst-case estimated cost:** $0.00187 under config (well under $0.10 cap).
- **Live mode requires:** explicit `--allow-live-api` flag AND `GOOGLE_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` in environment (exits with code 4 if absent).
- **Dry-run outputs:** manifest + summary written to `results/api_calibration/` (git-ignored).
- **Mock mode:** `--mock` available for integration testing without real API calls.

### Safe claims
- Live API calls are impossible without the explicit `--allow-live-api` flag.
- Budget cap and call count are validated before any live execution.
- No secrets are embedded in configs, scripts, or manifests.

### Unsafe claims
- Do not run live mode until credentials are reviewed and caps are confirmed sufficient.

---

## 3. Phase 2C Labeled Selector Dataset

**Status: Full dataset generated. No live API call made.**

- **Config:** `configs/phase2c_labeled_selector_dataset.yaml`
- **Script:** `scripts/build_phase2c_labeled_selector_dataset.py`
- **Tests:** 37/37 pass (includes: no live API path, no leaky feature columns, correct ANWG reconstruction, orca_vs_scorpio labels present, exact-prediction and overlap windows flagged, near-tie flags correct, realistic-subset exclusions correct, schema safety metadata, train/val/eval splits preserved, is_azure_conv_like is feature-threshold-based not workload-name-based).
- **Dataset:** `results/phase2c_labeled_selector_dataset/20260627_142404/` (git-ignored).

### Dataset Stats

| Metric | Value |
|---|---|
| Total rows | 611 |
| Train | 245 |
| Val | 41 |
| Eval | 325 |
| Causal feature columns | 17 |
| Near-tie rows (native pool) | 304 |
| azure_conv_like rows (feature-based) | 135 |
| orca_beats_scorpio rows (pairwise) | 212 |
| Phase 2C.3 external-loss rows | 69 |

### Label Safety Tiers

| Tier | Labels |
|---|---|
| safe_for_training | label_best_native_non_oracle_policy, margin_best_native, is_near_tie_native, is_realistic_subset, all is_* regime flags, is_azure_conv_like |
| analysis_only | label_best_external_style_policy, label_orca_beats_scorpio, label_phase2c2_dt_loses_to_external_envelope, label_phase2c3_best_loses_to_external_envelope, external_envelope_anwg, selected_policy_* |
| oracle_like_sensitive | is_exact_prediction_oracle_like |
| external_approximation_sensitive | All external_style pool labels, orca_vs_scorpio labels |

### Key Design Decisions
- **Labels are from simulator only:** ANWG = reward_* × completion_* from Phase 2C.2 evaluation outputs. No live API is used as ground-truth.
- **is_azure_conv_like is feature-based:** `is_long_prompt AND is_mixed_tight_slo` (not workload name), so it fires on any workload matching the azure_conv profile.
- **Phase 2C.2 reference check halts pipeline** if four reference metrics (dt_anwg, always_scorpio, external_envelope, external_loss_windows) don't match within 5e-4 tolerance.
- **api_annotation.enabled: false** enforced at CLI entry; script exits with code 2 if enabled.

### Safe claims
- All 17 feature columns are causal feat_* columns with no reward_, completion_, sel_, or label leakage.
- Train/val splits use only dev_* workloads; eval rows come from real Azure/BurstGPT traces.
- Mock API annotation fields are clearly marked `_mock` and `api_annotation_is_mock=True`; they are not used as labels.

---

## 4. Next Recommended Directions

1. **Pairwise / regret-weighted selector training:** Use `label_best_native_non_oracle_policy` and `pairwise_orca_scorpio_labels.csv` from the labeled dataset to train a regret-weighted selector that deprioritizes near-tie windows (304 rows flagged).

2. **Azure-conv-like synthetic training generation:** Use `is_azure_conv_like` (135 rows) to generate targeted synthetic windows with long-prompt + mixed-SLO characteristics. This could close the external-envelope gap on azure_2023_conv without adding external-style policies to the training pool.

3. **Tiny Gemini live calibration pilot:** After reviewing credentials and confirming caps with the user, run a small live pilot (10 calls, `--max-calls 10`) using the existing infrastructure. This should only happen after confirming `GOOGLE_API_KEY` is available and the $0.10 budget cap is acceptable.

4. **Regime-gated selector:** Implement a two-stage selector that routes azure_conv_like windows to a specialized sub-selector trained on those 135 rows.

---

## 5. Files Committed in This Batch

| File | Purpose |
|---|---|
| configs/phase2c3_external_aware_orca_recovery.yaml | Phase 2C.3 experiment config |
| scripts/run_phase2c3_external_aware_orca_recovery.py | Phase 2C.3 runner |
| tests/test_phase2c3_external_aware_orca_recovery.py | Phase 2C.3 tests (6) |
| configs/api_calibration/gemini_minimal_v1.yaml | Gemini calibration config |
| scripts/run_gemini_api_calibration.py | Gemini dry-run/mock infrastructure |
| tests/test_gemini_api_calibration.py | Gemini tests (22) |
| configs/phase2c_labeled_selector_dataset.yaml | Labeled dataset config |
| scripts/build_phase2c_labeled_selector_dataset.py | Dataset builder |
| tests/test_phase2c_labeled_selector_dataset.py | Dataset builder tests (37) |
| docs/audits/phase2c3_labeled_dataset_and_api_calibration.md | This audit |

**Not committed (git-ignored):**
- results/ (all generated datasets, dry-run outputs, experiment artifacts)
- logs/
- .env / credentials / API keys
