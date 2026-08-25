# Joint-240 Guarding / Abstaining Selector v1 — Analysis

**Date:** 2026-08-25  
**Schema:** `joint240_guarded_abstaining_selector_v1.0.0`  
**Experiment dir:** `experiments/joint240_guarded_abstaining_selector_v1/`  
**Preregistration:** `docs/design/JOINT240_GUARDED_ABSTAINING_SELECTOR_V1.md`  
**Parent artifacts:** `experiments/joint240_same_distribution_adaptive_exploitability_v1/`  
**Wall time:** 0.77s (CPU-only, `OMP_NUM_THREADS=1`)

## Verdict (preregistered)

**Labels:** `['GUARDED_SELECTOR_RECOVERS_SBS']`

- Best guarded method: **`util_advantage_guard`**
- Gain vs SBS: **+0.001022** with 95% CI **[-0.000033, 0.002118]**
- Catastrophic regressions: Ascen **67** → best guarded **7** (≈90% relative reduction)

Interpretation: a utility-aware abstaining rule recovers approximately SBS-level mean ANWG (CI includes zero; point estimate slightly positive) and **materially eliminates** Ascen's catastrophic non-SBS mistakes. It does **not** clearly beat SBS (CI does not exclude zero on the positive side only).

## Frozen Table 4 reference (verified)

| Method | Mean ANWG |
|---|---:|
| SBS (`kv_constrained_online`) | 0.3140716695 |
| VBS | 0.3331055037 |
| Ascen (prior OOF) | 0.3059465520 |
| Alive (prior OOF) | 0.2839667616 |

Unguarded Ascen **exactly reproduced** in this run: 0.3059465520 (`match_prior=true`).

## Methods

1. **Unguarded Ascen** — same OOF LogReg multiclass + VAL model selection as Section 4.2.
2. **Max-probability guard** — act iff `max p >= τ`, else SBS.
3. **Margin guard** — act iff `p_(1)-p_(2) >= τ`, else SBS.
4. **Utility-advantage guard** — Ridge per-policy ANWG regressors; act with `argmax pred` iff `pred[p*]-pred[SBS] >= τ`, else SBS.

SBS fallback is **fixed globally** as `kv_constrained_online`.

## Threshold-selection protocol (no held-out leakage)

Per outer OOF fold:
1. train_pool = other 4 folds;
2. inner 80/20 split (`seed = 20260825+100+fold`, identical to joint-240 runner);
3. fit base model on inner train;
4. choose `τ` maximizing **inner-VAL mean ANWG**;
5. refit on full train_pool;
6. apply chosen `τ` once to outer test.

### Chosen thresholds

| fold | tau_maxprob | tau_margin | tau_util_advantage |
|---:|---:|---:|---:|
| 0 | 1.00 | 1.00 | 0.0500 |
| 1 | 0.35 | 0.30 | 0.0200 |
| 2 | 1.00 | 1.00 | 0.0300 |
| 3 | 0.45 | 0.20 | 0.0075 |
| 4 | 0.80 | 0.75 | 0.0050 |

## Leakage checks

| Check | Status |
|---|---|
| Held-out outcomes used to choose τ | **False** |
| Future output length in features | **False** |
| VBS label / held-out best policy as input | **False** |
| SBS fallback fixed globally | **True** (`kv_constrained_online`) |
| τ from train-only inner VAL | **True** |
| 17 allowlisted generator features only | **True** |

## Main OOF results (n=240)

| Method | R_A | Gain vs SBS | Gap vs VBS | GapClosure | Frac→SBS/abstain | Frac beat SBS | Frac < SBS | N catastrophic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SBS_fixed` | 0.314072 | +0.000000 | 0.019034 | 0.0000 | 1.000 | 0.000 | 0.000 | 0 |
| `unguarded_ascen` | 0.305947 | -0.008125 | 0.027159 | -0.4269 | 0.212 | 0.242 | 0.321 | 67 |
| `maxprob_guard` | 0.313813 | -0.000259 | 0.019293 | -0.0136 | 0.738 | 0.054 | 0.071 | 13 |
| `margin_guard` | 0.313374 | -0.000698 | 0.019732 | -0.0367 | 0.829 | 0.042 | 0.046 | 11 |
| `util_advantage_guard` | 0.315094 | +0.001022 | 0.018011 | 0.0537 | 0.796 | 0.075 | 0.033 | 7 |
| `VBS_oracle` | 0.333106 | +0.019034 | 0.000000 | 1.0000 | 0.000 | 0.604 | 0.000 | 0 |
| `Alive_prior_reference` | 0.283967 | -0.030105 | 0.049139 | — | — | — | — | — |

## Bootstrap CIs (B=2000, scenario-paired, seed 20260825)

### Gain vs SBS

| Method | mean | CI95 low | CI95 high |
|---|---:|---:|---:|
| unguarded_ascen | -0.008125 | -0.013493 | -0.002568 |
| maxprob_guard | -0.000259 | -0.002136 | +0.001562 |
| margin_guard | -0.000698 | -0.002885 | +0.001385 |
| util_advantage_guard | +0.001022 | -0.000033 | +0.002118 |

### Delta vs unguarded Ascen (positive = better than Ascen)

| Method | mean | CI95 low | CI95 high |
|---|---:|---:|---:|
| maxprob_guard | +0.007866 | +0.002706 | +0.013001 |
| margin_guard | +0.007427 | +0.002442 | +0.012512 |
| util_advantage_guard | +0.009148 | +0.003709 | +0.014452 |

### Catastrophic-rate delta vs Ascen (negative = fewer catastrophes)

| Method | mean Δ rate | CI95 |
|---|---:|---|
| maxprob_guard | -0.225 | [-0.279, -0.171] |
| margin_guard | -0.233 | [-0.287, -0.179] |
| util_advantage_guard | -0.250 | [-0.312, -0.192] |

## Abstention / fallback

- Max-prob abstain rate: **73.8%** (n=177)
- Margin abstain rate: **82.9%**
- Utility guard chooses SBS: **79.6%**

All guarded methods abstain heavily — this is expected if confidence is poorly calibrated for specialist gains.

## Why Ascen loses (decomposition)

- Ascen chose non-SBS on **189** / 240 scenarios.
- Of those, **77** landed below SBS; **67** were catastrophic (`< SBS−0.01`).
- On those bad non-SBS picks, guards abstained / chose SBS at:
  - maxprob: 77.9%
  - margin: 85.7%
  - util→SBS: 89.6%
- On Ascen catastrophic non-SBS picks: maxprob abstain 80.6%, margin 83.6%.

So most of Ascen's value destruction is **unsafe specialist selection**, and abstention catches a large majority of those mistakes.

## Calibration / pressure

See:
- `calibration_maxprob.csv`, `calibration_margin.csv`
- `diagnostics.json` → `pressure_maxprob` / `pressure_margin` / `pressure_util`

## Comparison takeaway

| Claim | Evidence |
|---|---|
| Unguarded Ascen destroys SBS value | Gain −0.0081; CI entirely below 0; 67 catastrophes |
| Probability/margin guards ≈ recover SBS | Gains ≈ 0 with CIs including 0; catastrophes 13 / 11 |
| Utility-advantage guard slightly ≥ SBS (not significant) | +0.0010; CI includes 0; **7** catastrophes |
| Alive remains worse | −0.0301 vs SBS (prior) |
| Portfolio headroom still largely unexploited | Best guarded still far from VBS (+0.019 headroom) |

## Scientific implication for the reviewer concern

The negative Ascen result is **partly** explained by unsafe non-abstaining selection. A standard SBS-fallback guard removes most catastrophic regressions and brings mean ANWG back to approximately SBS. This weakens a narrative that “selectors cannot avoid harm,” but **does not** show that guarded selection closes the VBS–SBS headroom. The honest summary is:

> Without abstention, Ascen destroys value; with train-only confidence/utility guards, selection can avoid falling materially below SBS, but still fails to exploit portfolio headroom.

## Manuscript

**Not edited.**

## Artifacts

- `summary.json`, `summary.csv`, `bootstrap.json`
- `per_scenario_oof_results.csv`, `per_fold_chosen_thresholds.csv`
- `threshold_grids.json`, `config.json`, `diagnostics.json`, `DONE`
