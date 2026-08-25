# Joint-240 Strong Learned Selector v1 — Analysis (2026-08-25)

**Experiment:** `experiments/joint240_strong_learned_selector_v1/`  
**Design:** `docs/design/JOINT240_STRONG_LEARNED_SELECTOR_V1.md`  
**Design SHA-256:** `6d0459d7e8e78f96a03aed58dd17f5e5928cddd68ec0e33120b56fb350c0c9ed`  
**Parent (unmodified):** `experiments/joint240_same_distribution_adaptive_exploitability_v1/`

## Scientific question

Does a stronger nonlinear, cost-sensitive learned selector recover materially more
of the joint-240 SBS→VBS headroom than logistic `A_scen`?

## Reproduction of parent headline numbers

| Quantity | Value |
|---|---:|
| SBS (`kv_constrained_online`) | 0.314072 |
| VBS | 0.333106 |
| Headroom | 0.019034 |
| Majority | 0.2909 |
| `A_scen` | 0.3059 |
| `A_live` | 0.2840 |

Exact outer folds match parent `split_oof_folds.csv` (seed 20260825). Parent OOF
ANWG columns byte-identical after join.

## Preregistered design (frozen before held-out eval)

- **Formulation:** pooled direct utility regression
  \(f(x,p)\approx\mathrm{ANWG}(x,p)\) with 17 allowlisted scenario features + one-hot policy;
  \(\hat p=\arg\max_p f(x,p)\); evaluate via frozen matrix lookup.
- **Primary model:** `HistGradientBoostingRegressor` (`A_hgb`)
- **Secondary:** `ExtraTreesRegressor` (`A_et`)
- **HP search:** leave-one-train-fold-out among the four non-test outer folds;
  HGB grid size 8; ET grid size 4.
- **Seeds:** model `20260825`; bootstrap `20260826`; \(B=10{,}000\).
- **Success labels:** STRONG / PARTIAL / NO_RECOVERY vs SBS;
  CLEAR_IMPROVEMENT_OVER_A_SCEN if paired CI lower bound \(>0\).

## Main results

| Method | Mean ANWG | Gain vs SBS [95% CI] | GapClosure | Catastrophic | Acc vs VBS |
|---|---:|---|---:|---:|---:|
| SBS | 0.3141 | — | — | — | — |
| VBS | 0.3331 | — | — | — | — |
| Majority | 0.2909 | −0.023 [−0.031, −0.015] | −1.22 | 109/240 | 0.246 |
| `A_scen` | 0.3059 | −0.008 [−0.014, −0.003] | −0.43 | 67/240 | 0.296 |
| `A_live` | 0.2840 | −0.030 [−0.038, −0.023] | −1.58 | 118/240 | — |
| **`A_hgb`** | **0.3145** | **+0.00047 [−0.0023, +0.0033]** | **0.025** | **34/240** | **0.263** |
| `A_et` | 0.3127 | −0.0014 [−0.0049, +0.0020] | −0.073 | 48/240 | 0.246 |

### Pairwise (paired scenario bootstrap)

| Contrast | Mean ΔANWG | 95% CI |
|---|---:|---|
| `A_hgb` − `A_scen` | +0.0086 | **[+0.0038, +0.0139]** |
| `A_hgb` − `A_live` | +0.0306 | [+0.0232, +0.0386] |
| `A_et` − `A_scen` | +0.0067 | **[+0.0021, +0.0120]** |
| `A_hgb` − `A_et` | +0.0019 | [−0.0011, +0.0050] |

### Fold-level `A_hgb` ANWG

| Fold | n | SBS | VBS | `A_scen` | `A_hgb` |
|---:|---:|---:|---:|---:|---:|
| 0 | 50 | 0.270 | 0.287 | 0.267 | 0.270 |
| 1 | 49 | 0.341 | 0.357 | 0.332 | 0.344 |
| 2 | 48 | 0.305 | 0.330 | 0.292 | 0.304 |
| 3 | 47 | 0.343 | 0.360 | 0.327 | 0.345 |
| 4 | 46 | 0.313 | 0.334 | 0.313 | 0.311 |

No single fold dominates; folds 1 and 3 show mild positive `A_hgb`−SBS, fold 4 mild negative.

## Failure analysis (brief)

- **Accuracy vs utility:** `A_hgb` accuracy (26.3%) is *lower* than `A_scen` (29.6%),
  yet mean ANWG is higher and catastrophic regressions fall from 67→34.
  This matches the preregistered motivation: winner classification is not the
  right objective; cost-sensitive utility modeling reduces costly mistakes
  (`mean regret when incorrect` 0.025 vs 0.039 for `A_scen`).
- **Confusion:** heavy collapse toward `kv_constrained_online` (SBS) for several
  VBS winners (ESTF/WFS/LLF rows), consistent with near-SBS utility.
- **When VBS is SBS (`kv_constrained_online`, n=50):** `A_hgb` still incurs
  20 catastrophic regressions — residual underperformance when SBS is already
  optimal.
- **Pressure strata:** mild positive gains at elevated pressures 2–5; negative
  at pressure=1 (n=17).
- **HGB vs ET:** both beat `A_scen` with CI excluding zero; HGB slightly ahead
  of ET (CI includes zero). Conclusions do not hinge on one learner alone for
  the vs-`A_scen` claim.

## LTR relationship (analysis note; not manuscript text)

Fu et al. Learning-to-Rank learns **request-level relative output-length ranking**
for SJF-like scheduling. This experiment learns **scenario-level policy utility**
for portfolio selection among P6. It addresses the review concern about a
stronger learned adaptive *portfolio* baseline; it is **not** a faithful LTR
reimplementation.

## Integrity checks

- 240 unique OOF predictions; each scenario in exactly one outer test fold.
- Zero train/test overlap by nested protocol; HP selection never saw outer test.
- Six policies handled via one-hot + matrix columns.
- Parent `A_scen`/`A_live`/SBS/VBS means unchanged; parent artifact hashes unchanged.
- Unit tests: `tests/test_joint240_strong_learned_selector_v1.py` (8 passed).

## Preregistered interpretation

| Label | Result |
|---|---|
| `A_hgb` vs SBS | **PARTIAL_RECOVERY** (mean gain \(>0\), but CI includes 0 and GapClosure \(0.025 \ll 0.50\)) |
| `A_et` vs SBS | **NO_RECOVERY** |
| CLEAR_IMPROVEMENT_OVER_A_SCEN (`A_hgb`) | **Yes** |
| CLEAR_IMPROVEMENT_OVER_A_SCEN (`A_et`) | **Yes** |

## Manuscript decision (no edit in this task)

**PROMOTE** — integrate as a stronger learned portfolio-selection baseline.

Reasoning:
1. It directly answers the shared review criticism that only logistic `A_scen`
   and constrained `A_live` were tested.
2. Both nonlinear utility models **clearly beat** `A_scen` on ANWG.
3. Opportunity recovery remains essentially **null**: `A_hgb` is statistically
   indistinguishable from SBS (gain CI includes 0) and closes only ~2.5% of
   headroom. That near-null exploitation result, under a stronger learner,
   **strengthens** rather than weakens the exploitability-gap narrative.

Do **not** claim STRONG_RECOVERY. Do **not** claim LTR reproduction.

## Review-criticism mapping

| Criticism | Status |
|---|---|
| Same-dist result only tests lightweight adapters | **PARTIALLY RESOLVED** — stronger nonlinear utility selector now exists and still fails to recover headroom |
| Need evidence beyond logistic classification | **RESOLVED** for portfolio selection |
| Faithful Learning-to-Rank / live learned router / SOTA systems | **UNRESOLVED** (explicitly out of scope here) |
