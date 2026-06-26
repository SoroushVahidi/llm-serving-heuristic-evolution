# Phase 2C.1 Evaluation Validity Audit

## Purpose

This note documents validity issues found in the completed Phase 2C.1 full
evaluation (`results/phase2c1_real_trace_ingestion_validation/20260626_144419`)
and the safeguards added afterward.

## 1. Within-window feature lookahead (fixed)

### Problem

The legacy `online_prefix` feature mode used **all requests in the current
window** for token and SLO statistics (mean prompt length, predicted output,
slack, tight-SLO fraction). At the real decision time — the start of the
window — those later arrivals have not happened yet. This is future-arrival
lookahead and invalidates online/deployable selector claims.

Queue / arrival-rate features already used only prefix history up to
`window_start_time`, but token/SLO features did not.

### Fix

- New deployable mode: **`causal`**
  - Features use only requests with `arrival_time <= window_start_time`
    (prefix history plus any request arriving exactly at window start).
- Legacy leaky mode renamed: **`offline_window_lookahead`**
  - Preserves the old `online_prefix` behavior for offline diagnostics only.
  - Config alias `online_prefix` still accepted but deprecated.
- Phase 2C.1 config now defaults to `causal` and rejects non-causal modes.

### Interpretation

| Evaluation | Feature mode used | Safe for deployable claims? |
|------------|-----------------|------------------------------|
| Phase 2C.1 full run (20260626_144419) | `online_prefix` (lookahead) | **No** — upper-bound / optimistic |
| Future Phase 2C.1+ runs | `causal` | **Yes** (subject to other caveats below) |

BurstGPT near-parity results from the lookahead run should be treated as an
optimistic ceiling until re-evaluated under `causal`.

## 2. Oracle-assisted safe-fallback selectors (labeled)

### Problem

`safe_fallback_wsp_margin*` selectors call `SafeFallbackWspSelector`, which
compares the base selector's **realized per-window arrival-normalized WG**
against WSP's realized WG before choosing. That uses post-window reward
information and is not deployable online.

### Fix

- Selectors are classified in `src/llmserveopt/selector/roles.py`.
- Phase 2C.1 metadata records `oracle_assisted_selectors`.
- `deployable_selector_summary.csv` excludes oracle-assisted selectors.
- Full `selector_summary.csv` still includes them as diagnostic upper bounds.

## 3. Completed-only weighted goodput inflation (metrics clarified)

### Problem

Completed-only WG (`reward_*` / `mean_completed_request_quality`) can be high
while completion fraction is low. Ranking selectors on completed-only WG alone
overstates quality when policies drop or fail to complete requests.

### Fix

Summaries now report, per selector:

| Metric | Definition |
|--------|------------|
| `mean_completed_request_quality` | WG among completed requests only |
| `mean_completion_fraction` | Fraction of arrivals completed |
| `mean_arrival_normalized_wg` | completion × completed-only WG (**primary rank metric**) |
| `mean_cp_wg_t095_l05` / `mean_cp_wg_t099_l05` / `mean_cp_wg_t099_l10` | Completion-penalized variants (Phase 2B.15/2B.16) |

Deployable headline tables sort by **`mean_arrival_normalized_wg`**, not
completed-only WG.

### Phase 2C.1 lookahead result example

`regression_anwg` overall: completed-only WG ≈ 0.958 but completion ≈ 0.832,
so ANWG ≈ 0.791 and cp_wg_t099_l10 ≈ 0.630.

## 4. How to read Phase 2C.1 results

### Before this fix (completed run)

- Feature lookahead likely **inflates** learned selector quality vs true online
  deployment, especially on BurstGPT workloads where gaps to best fixed baseline
  were already small.
- `safe_fallback_wsp_margin*` matching `always_scorpio` is an **oracle-assisted
  ceiling**, not deployable selector performance.
- `burstgpt_moderate_exact_prediction` is an oracle-like prediction regime —
  report separately from realistic workloads.
- `azure_2023_conv` failure (selector collapse, large negative gap) is a
  genuine generalization concern independent of lookahead.

### After this fix (future runs)

1. Re-run Phase 2C.1 (or Phase 2C.2) with `feature_mode: causal`.
2. Use `deployable_selector_summary.csv` for headline comparisons.
3. Rank by ANWG; report completion fraction alongside completed-only WG.
4. Keep oracle-assisted and exact-prediction workloads out of deployable claims.

## 5. Remaining risks

- Selectors were **trained** on Phase 2B.13 data built with legacy feature modes;
  causal re-evaluation measures inference-time validity but not retrained models.
- Azure conv distribution shift may still dominate failure even under causal features.
- `burstgpt_scaled_high` first-two-window training overlap remains a separate
  data-leakage concern for training splits (not fixed here).

## Recommended next experiment

Re-run Phase 2C.1 under `causal` feature mode (smoke first, then full tmux run)
without retraining, to quantify how much the lookahead inflated BurstGPT gains.
If causal gaps widen materially, schedule Phase 2C.2 selector retraining on
causal features before making deployable claims.
