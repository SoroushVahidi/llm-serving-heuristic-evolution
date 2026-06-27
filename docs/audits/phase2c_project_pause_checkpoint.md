# Phase 2C Project Pause Checkpoint

**Date:** 2026-06-27  
**Branch:** `phase2c1-real-trace-ingestion-validation`  
**Latest pushed commit:** `69c80ea` — "Add Phase 2C.3 analysis, API calibration dry-run, and labeled dataset builder"  
**Upstream:** `origin/phase2c1-real-trace-ingestion-validation` — fully synced (0 ahead, 0 behind)  
**Working tree:** Clean (no dirty tracked files, no staged changes)

---

## What Has Been Completed

### Phase 2C.1 — Real-Trace Ingestion & Validation
- Azure 2023 conv + code traces and BurstGPT traces ingested and validated.
- Evaluation runner validates simulator outputs against real-trace distributions.
- Commits: `db6819e`, `a04eb6a`

### Phase 2C.2 — Causal Selector Retraining
- Trained DT and RF selectors on causal feat_* features, corrected-objective ANWG labels.
- Best selector: `dt_anwg`, eval ANWG = **0.8021** (native non-oracle pool).
- `rf_anwg` ANWG ≈ 0.8116 (best learned selector, but close to dt in practice).
- External-style envelope on eval: **0.8297** — gap of ~0.028 remains.
- External-loss windows: **62** (windows where external envelope > dt_anwg selector).
- Commits: `7244444` (+ prior `9213d0a` safeguard fix)
- Artifact: `results/phase2c2_causal_selector_retraining/20260626_194325/` (~1.1 MB)

### Phase 2C.3 — External-Aware Orca Recovery (negative finding)
- **Goal:** Test whether adding external-style policies to training pool recovers orca's advantage in azure_2023_conv windows.
- **Structural finding:** orca_style wins **zero** full-pool training windows → zero orca training labels after near-tie filtering. External-aware DT is numerically identical to native DT.
- **Label distribution (external_aware pool):** scorpio=68 (88%), admission_control=7 (9%), wsp=2 (3%), orca=0 (0%).
- **Best Phase 2C.3 ANWG:** `native_non_oracle_dt` = **0.8063** (delta +0.0042 vs Phase 2C.2, but no orca recovery).
- **Orca selected by best selector:** 0 times.
- **azure_2023_conv gap remains** — external envelope still +0.028 above best selector.
- Phase 2C.2 reference reproduced exactly within tolerance (dt_anwg 0.80211, always_scorpio 0.79627, external_envelope 0.82968, external_loss_windows 62).
- Commit: `69c80ea`
- Artifact: `results/phase2c3_external_aware_orca_recovery/20260626_234942/` (~2.3 MB)
- Failure diagnosis: `results/phase2c3_failure_diagnosis/20260626_phase2c2_analysis/` (~40 KB)

### Gemini API Calibration Dry-Run Infrastructure
- **Status: dry-run only. No live API call made.**
- 24 planned calls (3 prompt × 2 output × 2 concurrency × 2 repeats).
- Hard cap: 50 calls, estimated worst-case cost $0.00187 (under $0.10 budget cap).
- Live mode requires explicit `--allow-live-api` flag and credentials env var.
- Commit: `69c80ea`
- Artifact: `results/api_calibration/gemini_minimal_v1/` (~36 KB, two dry-run manifests)

### Phase 2C Labeled Selector Dataset
- **Status: full dataset generated. No live API call made.**
- Labels derived exclusively from ANWG = reward_* × completion_* (simulator outputs).
- 611 total rows (train=245, val=41, eval=325).
- 17 causal feat_* feature columns (no reward_/completion_/sel_/anwg_ leakage).
- 304 near-tie rows (native pool, margin < 0.005).
- 135 azure_conv_like rows (feature-based: is_long_prompt AND is_mixed_tight_slo).
- 212 orca-beats-scorpio rows (pairwise ANWG).
- 69 Phase 2C.3 external-loss rows (Phase 2C.3 best selector < external envelope).
- Label safety tiers: safe_for_training, analysis_only, oracle_like_sensitive, external_approximation_sensitive.
- Commit: `69c80ea`
- Artifact: `results/phase2c_labeled_selector_dataset/20260627_142404/` (~2.3 MB)

---

## Key Committed Files

| File | Description |
|---|---|
| `configs/phase2c3_external_aware_orca_recovery.yaml` | Phase 2C.3 experiment config |
| `scripts/run_phase2c3_external_aware_orca_recovery.py` | Phase 2C.3 runner |
| `tests/test_phase2c3_external_aware_orca_recovery.py` | Phase 2C.3 tests (6) |
| `configs/api_calibration/gemini_minimal_v1.yaml` | Gemini calibration config |
| `scripts/run_gemini_api_calibration.py` | Gemini dry-run/mock/live infrastructure |
| `tests/test_gemini_api_calibration.py` | Gemini tests (22) |
| `configs/phase2c_labeled_selector_dataset.yaml` | Labeled dataset config |
| `scripts/build_phase2c_labeled_selector_dataset.py` | Dataset builder |
| `tests/test_phase2c_labeled_selector_dataset.py` | Dataset builder tests (37) |
| `docs/audits/phase2c3_labeled_dataset_and_api_calibration.md` | Detailed audit for Phase 2C.3 + calibration + dataset |
| `docs/audits/phase2c_project_pause_checkpoint.md` | This file |

---

## Main Result Summary

### What Is Safe to Claim
- `native_non_oracle_dt` selector (ANWG 0.8021 on eval) outperforms always-scorpio (0.7963) and always-wsp by a reproducible margin.
- Adding external-style policies to the training pool does not hurt the selector and adds no benefit in the current setup.
- `scorpio_style_slo_guard` is the best fixed external-style baseline overall (by mean eval ANWG).
- `orca_style` wins selectively on 212/611 windows by pairwise ANWG, but is not a good always-fixed policy.
- azure_2023_conv is the main failure mode: long-prompt + mixed-SLO windows where external policies outperform the learned selector.
- `is_azure_conv_like` (feature-based: is_long_prompt AND is_mixed_tight_slo) correctly identifies 135 windows matching this profile.
- The labeled dataset captures reliable pairwise orca-vs-scorpio signal for 611 windows.
- All 65 new tests pass; all dry-runs make no live API calls.

### What Is Unsafe to Claim (without further evidence)
- That the +0.0042 ANWG improvement from Phase 2C.2 → Phase 2C.3 is statistically significant (it is not; Phase 2C.3 DTs are identical to Phase 2C.2 DTs).
- That orca recovery is impossible — it may be possible with targeted training data for azure_conv_like windows.
- That the rf_anwg vs dt_anwg gap generalizes beyond the current workload suite.
- That adding more external policies beyond the 7 currently included would close the envelope gap.

---

## Generated Artifacts (Git-Ignored)

All paths below are git-ignored and exist only locally. They must be regenerated if the machine is wiped.

| Path | Size | Description |
|---|---|---|
| `results/phase2c2_causal_selector_retraining/20260626_194325/` | ~1.1 MB | Phase 2C.2 training/eval outputs, trained model files |
| `results/phase2c3_external_aware_orca_recovery/20260626_234942/` | ~2.3 MB | Phase 2C.3 per-window predictions, selector summary, training label counts |
| `results/phase2c3_failure_diagnosis/20260626_phase2c2_analysis/` | ~40 KB | azure_conv failure analysis, top external-loss cases |
| `results/phase2c_labeled_selector_dataset/20260627_142404/` | ~2.3 MB | Full labeled dataset (12 output files) |
| `results/api_calibration/gemini_minimal_v1/` | ~36 KB | Gemini dry-run manifests (2 runs) |
| `logs/phase2c*/` | ~436 KB total | Training and evaluation logs |

**Note:** `results/.gitkeep` is the only tracked file under `results/` — confirmed with `git ls-files results logs`.

---

## Open Risks

1. **Labeled dataset is local-only.** The 611-row dataset in `results/phase2c_labeled_selector_dataset/20260627_142404/` is git-ignored. If regenerated after changes to source data or config, results could shift. The Phase 2C.2 reference check (dt_anwg, always_scorpio, external_envelope, external_loss_windows within 5e-4) guards against this.
2. **Phase 2C.2 result artifacts are local-only.** The trained models (`results/phase2c2_causal_selector_retraining/`) are not committed. Re-running Phase 2C.2 from the committed script will regenerate them, but the exact timestamped path in the labeled dataset config (`configs/phase2c_labeled_selector_dataset.yaml`) hardcodes `20260626_194325`. Update the path if re-running from scratch.
3. **Gemini live mode requires credentials.** `GOOGLE_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` must be set. Not tested; credentials not confirmed on this machine.
4. **Worktree at `.claude/worktrees/phase2b9`.** This is a Claude Code internal worktree on commit `429e96e`. It is not used by Phase 2C work and should not be touched.
5. **Several tmux sessions are idle.** Old sessions (kbs_*, phase2b11-b16) appear to be finished experiments. Safe to leave; not blocking anything.

---

## Commands to Resume

```bash
# 1. Confirm state after returning
cd /home/soroush/llm-serving-heuristic-evolution
git status --short --branch
git log --oneline -5

# 2. Verify labeled dataset builder still works
python scripts/build_phase2c_labeled_selector_dataset.py \
  --config configs/phase2c_labeled_selector_dataset.yaml \
  --dry-run

# 3. Verify Gemini dry-run still works
python scripts/run_gemini_api_calibration.py \
  --config configs/api_calibration/gemini_minimal_v1.yaml \
  --dry-run

# 4. Re-run all Phase 2C tests
/home/soroush/modal-venv/bin/pytest -q \
  tests/test_phase2c3_external_aware_orca_recovery.py \
  tests/test_gemini_api_calibration.py \
  tests/test_phase2c_labeled_selector_dataset.py

# 5. Inspect latest labeled dataset
ls results/phase2c_labeled_selector_dataset/
# (pick the latest timestamped dir)

# 6. To regenerate full dataset
python scripts/build_phase2c_labeled_selector_dataset.py \
  --config configs/phase2c_labeled_selector_dataset.yaml

# 7. (When ready) Gemini live pilot - 10 calls only
# Requires GOOGLE_API_KEY to be set
python scripts/run_gemini_api_calibration.py \
  --config configs/api_calibration/gemini_minimal_v1.yaml \
  --allow-live-api --max-calls 10
```

---

## Recommended Next Steps (in order)

### Immediate (Queries 2–3)
1. **Query 2:** Docs/README/result-claims consistency check. Ensure `docs/result_claims.md` (if it exists) accurately reflects Phase 2C results; no inflated claims about orca recovery.
2. **Query 3:** Final validation and possible PR preparation. Consider whether this branch is ready to merge to main or should remain as a feature branch.

### Near-term Research (Phase 2C.4+)
3. **Pairwise/regret-weighted selector training:** Use `label_best_native_non_oracle_policy` (safe_for_training) and `pairwise_orca_scorpio_labels.csv` from the labeled dataset. Train a regret-weighted or pairwise-margin selector that deprioritizes the 304 near-tie windows. Expected to outperform dt_anwg by exploiting margin information.
4. **Azure-conv-like synthetic training generation:** Generate targeted synthetic windows matching `is_azure_conv_like` profile (long prompt + mixed tight SLO). The 135 real windows may not be enough for a regime-specific sub-selector. Aim for ~500 windows covering the long-prompt regime.
5. **Regime-gated selector:** Implement two-stage selector: route `is_azure_conv_like` windows to a specialized sub-selector (possibly an external-style policy or a dedicated trained model), route other windows to `native_non_oracle_dt`.

### Calibration (When Ready)
6. **Gemini live calibration pilot:** Run `--allow-live-api --max-calls 10` after confirming `GOOGLE_API_KEY` is available and the $0.10 budget cap is acceptable. Use results to calibrate token-rate estimates in config.
