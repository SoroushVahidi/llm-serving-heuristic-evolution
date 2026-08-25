# Multi-Family Contextual Selector v1 — Step 3 Results and Audit

Date: 2026-08-17

## 0. Scope

Executes the preregistered Step-3 experiment
([`MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md`](../design/MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md))
against the frozen, complete unified utility matrix
(`experiments/unified_utility_matrix_v2/`, `UNIFIED_UTILITY_MATRIX_READY`).
**Scientific question:** can a contextual selector trained on the unified
six-policy utility matrix generalize across mechanism families and to a
held-out family, or does it merely memorize family-specific boundaries?
**Evaluation only** — no mechanism attribution, no composition/synthesis,
no real-vLLM validation, no PSD redesign, no TEST/OOD tuning. Every verdict
gate and threshold was frozen in the design doc before any TEST/LOFO row
was scored; nothing below was adjusted after seeing results.

## A. Launch

- Design: committed and pushed pre-launch (`eb39ff4`).
- Harness: `src/llmserveopt/selector/multifamily_contextual_selector_v1.py`.
  CLI: `scripts/run_multifamily_contextual_selector_v1.py`.
- Tests: `tests/test_multifamily_contextual_selector_v1.py`, 27/27 passing
  — including a regression guard for a real bug caught during
  implementation: `evaluate_predictions` originally derived its "best
  fixed" comparison baseline from the same dataframe being evaluated
  (i.e. from TEST/held-out data), leaking test information into a metric
  meant to be a fair external comparison. Fixed before any TEST/LOFO run;
  `gap_to_best_fixed` is now always computed at the orchestration layer
  from a separately, correctly TRAIN-derived baseline.
- Regression: 92/92 relevant tests pass (65 pre-existing Step-1/2 + 27
  new).
- Smoke (`--smoke`, forest-only, reduced bootstrap): all three regimes
  executed without error; notably already showed the LOFO-Family-A
  failure mode that the full run confirms (§F).
- tmux session `mcs_v1_run`, launch SHA `eb39ff4`. Ran to natural
  completion in under 5 seconds (well inside the health-check window).

## B. Input Confirmation

- 176 scenarios, 33 learnable features, 6 policies (alphabetical:
  `chunked_prefill_small`, `estimated_service_time_first`, `full_prefill`,
  `kv_constrained_online`, `least_laxity_first`, `weighted_fair_share`).
- `mechanism_family` confirmed absent from every feature matrix passed to
  `.fit()` (tested).

## C. Feature Set

33-column learnable allowlist, family-prefixed with explicit
missingness. Missing-value handling: numeric → 0.0 + `<col>__missing`
indicator; categorical → `"__NONE__"`.

## D. Family-Predictability Diagnostic

**Mean accuracy: 100.0%** (5/5 grouped-CV folds, every fold 100%). This is
the expected finding stated in advance (design doc §2): because every
allowlist column is family-prefixed with structural, not random,
missingness, the `<col>__missing` indicators alone perfectly encode family
identity. **This is a property of the MF-PSD v1 schema, not a modeling
artifact.**

## E. Target / Tie Handling

Primary target: exact top-1 winner, alphabetical tie-break on bit-exact
ties (`chunked_prefill_small` preferred over `full_prefill` when tied —
verified by test). Winner distribution over all 176 scenarios:
`estimated_service_time_first` 70, `chunked_prefill_small` 48,
`kv_constrained_online` 31, `weighted_fair_share` 27 (`full_prefill` and
`least_laxity_first` never win outright under this tie-break, since
`full_prefill` always loses the alphabetical tie to `chunked_prefill_small`
wherever they're tied, and `least_laxity_first` never has the single
highest ANWG on any scenario in this matrix).

## F. Regime Results

### Regime A — within-family (in-distribution)

| Family | Best model mean regret | best_fixed | majority |
|---|---|---|---|
| A (n=14 test) | 0.0000 (logreg/tree/forest) | 0.0228 | 0.0228 |
| B (n=8 test) | 0.0000 (logreg/tree/forest/utility) | 0.0488 | 0.0488 |
| C (n=12 test) | 0.0319 (utility_argmax) / 0.0417 (classifiers) | **0.0221 (best_fixed wins)** | 0.0417 |

**Within-family, contextual selection is strong** — perfect (0 regret) on
Family A and B holdouts, and competitive (though not superior to
best-fixed) on Family C. This confirms the matrix contains real,
learnable per-family structure.

### Regime B — multi-family pooled (n=36 test, 12 per family)

| Model | Mean regret | Gap to fixed | ε-optimal acc. | Exact-winner acc. |
|---|---|---|---|---|
| `logreg` | 0.0477 | **+0.0244** | 0.611 | 0.500 |
| `tree` | 0.0504 | +0.0271 | 0.556 | 0.333 |
| `forest` | 0.0463 | +0.0230 | 0.639 | 0.639 |
| `utility_argmax` | 0.0504 | +0.0271 | 0.556 | 0.222 |
| `pairwise` | 0.0518 | +0.0285 | 0.528 | 0.250 |
| `best_fixed` | 0.0233 | 0.0000 | 0.583 | 0.056 |
| **`majority`** | **0.0127** | **−0.0106** | **0.833** | 0.611 |

**Every trained model performs worse than best-fixed, and worse than the
trivial majority baseline**, on this pooled holdout. Bootstrap 90% CI
(group-resampled, 1000 draws): forest mean regret `[0.015, 0.083]`,
best-fixed `[0.008, 0.039]` — overlapping, forest not distinguishable from
(and nominally worse than) best-fixed. Macro-by-family for `forest`:
Family A regret **0.000** (12/12 perfect), Family B regret **0.130**
(much worse than best-fixed's 0.000 there), Family C regret 0.008. Family
B's known near-total policy collapse (`unified_policy_utility_matrix_v1`
§G2/§I) means "always predict the fixed best policy" is *already*
essentially optimal there — any model introducing per-scenario variance
can only hurt on that family, and it does, dragging the pooled average
down enough to erase the real gains visible on Family A/C.

### Regime C — Leave-One-Family-Out (the primary regime)

| Held-out family | Best model mean regret | best_fixed | Win? |
|---|---|---|---|
| A | forest **0.4786** | 0.0767 | **No — 6.2× worse** |
| B | forest 0.0494 | 0.0494 | No — exact tie |
| C | forest **0.0263** | 0.0385 | **Yes** |

**LOFO wins: 1/3.** The Family-A-held-out failure is severe and
mechanistically explicable, not noise: when Family A is entirely excluded
from training, every `feat_A__*` column is constant-missing throughout
TRAIN (all training rows come from B/C, which never populate those
columns), so no model can learn to use them — they carry zero training
signal. At test time on Family A, those same columns are the *only*
columns that actually vary meaningfully within Family A's own scenario
space, so the model effectively has no informative signal about the
held-out family's context, and its predictions collapse toward whatever
its B/C-derived decision boundary defaults to. This is the concrete,
observed failure mode of "the selector identifies family, not mechanism"
that the family-predictability diagnostic (§D) predicts.

## G. Utility-Prediction and Pairwise Baselines

Neither `utility_argmax` (Ridge regression + argmax) nor `pairwise`
(15 pairwise logistic classifiers + beats-count) outperforms the direct
top-1 classifiers in any regime; `pairwise` is the single worst model in
Regime B (mean regret 0.0518, highest of all seven). Pairwise modeling
does **not** outperform direct utility prediction in this experiment.

## H. Shared-Feature Robustness (A↔B only, Family C excluded)

`shared__max_active_sequences` and `shared__stress_control_relationship`
are the only two features with a matching semantic role across two (not
three) families (MF-PSD v1 audit §P item 3). Restricting to just these 2
columns, pooled A∪B (n=104, 70/34 grouped train/test): selector mean
regret 0.0611 vs. best-fixed 0.0122 — **improvement over fixed is
−0.0489 (worse, not better)**. With almost no informative feature signal
remaining, the selector cannot beat a trivial fixed baseline at all — this
is consistent with, not contradictory to, §D/§F: the family-prefixed
feature blocks are carrying essentially all of the model's apparent
in-distribution skill.

## I. Preregistered Verdict Gates — Mechanical Result

| Gate | Result |
|---|---|
| 1. Regime B best model beats best-fixed by ≥20% relative regret | **FAIL** (forest is 98% *worse*, not better) |
| 2. Regime B ε-optimal accuracy beats majority by ≥15pp | **FAIL** (forest is 19pp *worse* than majority) |
| 3. Macro-by-family beats fixed in all 3 families (Regime B) | **FAIL** (Family B loses badly) |
| 4. LOFO wins ≥2/3 held-out directions | **FAIL** (1/3) |
| 5. Shared-feature robustness positive | **FAIL** (−0.049) |

Applying the frozen decision tree (design doc §7) mechanically: gate 1
fails → **`MULTIFAMILY_SELECTOR_NO_GO`**, regardless of the other gates.

## J. Final Verdict

**`MULTIFAMILY_SELECTOR_NO_GO`**

Not softened, not re-derived from a different regime after seeing the
Regime-B result. The gates were defined against Regime B (pooled,
in-distribution) specifically as the entry condition before even
considering LOFO transfer — Regime B itself already fails decisively.

## K. Scientific Interpretation

1. **Does unified contextual selection outperform best fixed?** No, on
   the pooled holdout — every trained model has higher regret than
   best-fixed, and than the trivial majority baseline.
2. **Does it approach the six-policy oracle?** No — the best model's
   pooled regret (0.046) is roughly double best-fixed's own gap to
   oracle (0.023).
3. **Does it generalize within families?** **Yes, strongly** — Regime A
   shows near-perfect (0 regret) performance on Family A and B holdouts.
   This is the most important nuance in these results: the failure is not
   in the models' ability to learn *some* real structure, but in pooling
   and cross-family transfer specifically.
4. **Does it generalize to an unseen mechanism family?** Mostly no — 1/3
   LOFO directions win, and the Family-A failure is severe (6.2× worse
   than best-fixed), with a concrete, verified mechanistic cause (§F).
5. **Is performance driven by family identification?** Yes, strong
   evidence: 100% family-predictability from features alone (§D), and the
   LOFO-A collapse is exactly the failure mode predicted by a
   family-identification-dependent selector losing its family "fingerprint."
6. **Does pairwise modeling outperform direct utility prediction?** No —
   pairwise is the worst model tested.
7. **Is the current MF-PSD v1 dataset/feature schema useful for
   cross-mechanism selection as-is?** Not demonstrated to be. Two distinct,
   separable causes: (a) genuine small-sample noise (12–72 scenarios per
   family per fold is a small-data regime, visible in the wide bootstrap
   CIs), and (b) a structural cause — the feature schema partitions
   almost entirely by family (§H: only 2 of 33 columns have any
   cross-family semantic overlap, and only between 2 of 3 families),
   making it nearly impossible for a selector to demonstrate genuine
   mechanism-level (as opposed to family-level) generalization with this
   representation.
8. **Is mechanism attribution now justified?** **No.** The preregistered
   gate for that decision (a `READY`/`READY_LOW_DIVERSITY`-analogous GO
   signal at Step 3) was not met. Step 4 remains blocked.

## L. Files Created / Modified

- `docs/design/MULTIFAMILY_CONTEXTUAL_SELECTOR_V1.md`
- `src/llmserveopt/selector/multifamily_contextual_selector_v1.py`
- `scripts/run_multifamily_contextual_selector_v1.py`
- `tests/test_multifamily_contextual_selector_v1.py`
- `experiments/multifamily_contextual_selector_v1/multifamily_contextual_selector_v1_results.json`, `run.log`
- This document.
- `docs/current/{RESUME_HERE,NEXT_ACTIONS}.md` (reconciled).

**Not modified:** `experiments/mf_psd_v1/`, `experiments/unified_utility_matrix_v1/`,
`experiments/unified_utility_matrix_v2/`, `experiments/family_c_reconstruction_v1/`,
any historical run directory, any prior audit document.

## M. Exact Next Scientific Action

**Not mechanism attribution** (blocked by this verdict). The most
directly motivated next step, if pursued, would be a **feature-schema
redesign task** — investigating whether a genuinely shared, cross-family
feature representation (e.g. derived request-level statistics computed
identically regardless of family, rather than family-prefixed raw sweep
parameters) could let a selector demonstrate real mechanism-level transfer
without relying on family identification. That is a new, separately
scoped and separately authorized task — **not started here**. No selector
was deployed; no mechanism attribution, composition, or synthesis work was
performed in this task.
