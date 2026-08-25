# Multi-Family Contextual Selector v1 — Preregistration

Date: 2026-08-17

## 0. Scope and Question

Step 3 of the revised roadmap
([`reassessment_composition_hypothesis_20260817.md`](../audits/reassessment_composition_hypothesis_20260817.md)
§O). Scientific question: **can a contextual selector trained on the
unified six-policy utility matrix generalize across mechanism families and
to a held-out family, rather than merely memorizing family-specific
boundaries?** No mechanism attribution, no symbolic distillation, no
composition/synthesis, no real-vLLM validation, no PSD redesign, no
TEST/OOD tuning, no `mechanism_family` as a selector feature.

## 1. Frozen Input

- `experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv`
  — 176 scenarios × 6 anchors, 1,056/1,056 cells populated, verdict
  `UNIFIED_UTILITY_MATRIX_READY`.
- `experiments/mf_psd_v1/mf_psd_scenarios_v1.csv` — 176 scenarios × 44
  columns (10 identity/audit + 34 learnable-feature-allowlist columns,
  family-prefixed with explicit missingness), joined to the matrix on
  `canonical_scenario_id`.
- `experiments/mf_psd_v1/mf_psd_schema_v1.json` — authoritative
  `learnable_feature_allowlist` (33 entries: 9 `feat_A__*`, 20 `feat_B__*`,
  3 `feat_C__*`, re-read directly from the schema file for this task) and
  `forbidden_audit_only_fields` (22 entries, includes `mechanism_family`).
- Six canonical policies, order fixed alphabetically by `canonical_policy_id`:
  `chunked_prefill_small`, `estimated_service_time_first`, `full_prefill`,
  `kv_constrained_online`, `least_laxity_first`, `weighted_fair_share`.

## 2. Feature Handling — Frozen Before Any Modeling

- **X = the 33-column learnable allowlist, exactly.** No `mechanism_family`,
  no scenario/policy identity, no utility column ever enters X.
- **Missing-value handling (explicit, frozen):** every allowlist column is
  empty-string for the 2 families a scenario does not belong to (by
  MF-PSD v1's own design). Numeric columns: missing → `0.0` plus a
  same-named `<col>__missing` binary indicator column (so the imputation
  value itself carries no information the indicator doesn't already make
  explicit). Categorical columns: missing → literal category `"__NONE__"`,
  one-hot encoded. This is a simple, explicit, non-adaptive rule — frozen
  before touching TEST/LOFO data.
- **Family-predictability diagnostic (mandatory, run first):** fit a
  `RandomForestClassifier` (same fixed hyperparameters as model class E
  below) to predict `mechanism_family` (3-class) from X alone, 5-fold
  grouped-by-`group_key` cross-validation on the full 176-scenario pool.
  Report accuracy. **Expected finding, stated in advance:** because every
  allowlist column is family-prefixed with structural (not random)
  missingness, the `<col>__missing` indicators alone are expected to
  predict family at or near 100% accuracy — this is a property of the
  MF-PSD v1 schema, not a modeling artifact, and does not by itself
  invalidate the selector (see §9 verdict logic, `ID_ONLY` branch).
- **Shared-feature-subset robustness check:** the two columns with
  matching semantic role across two (not three) families are
  `feat_A__max_active_sequences`/`feat_B__max_active_sequences` and
  `feat_A__stress_control_relationship`/`feat_B__stress_control_relationship`
  (MF-PSD v1 audit §P item 3 — kept family-prefixed there specifically
  because equivalence was never proven). **No feature is shared across all
  three families in the current MF-PSD v1 schema** — Family C's 3 features
  (`bulk_pressure`, `urgent_arrival_phase`, `urgent_tightness`) have no A/B
  analog. The robustness check therefore restricts to the A↔B shared pair
  (2 columns, unified to `shared__max_active_sequences`,
  `shared__stress_control_relationship`) and is run **only** as an
  A-vs-B-pooled diagnostic, explicitly excluding Family C — this limitation
  is reported directly, not silently worked around, and no feature is
  removed from the main pipeline because of it.

## 3. Target Construction — Frozen

- **Primary target (used for training classifiers): exact top-1 winner
  with deterministic tie-breaking (Option A).** `winner(x) = argmax_p
  ANWG(x, p)`; on a bit-exact tie (`|Δ|<1e-9`, which happens on
  `full_prefill`/`chunked_prefill_small` for 144/176 scenarios by the
  Step-2 ServiceModel-degeneracy finding, plus 1 incidental Family-B tie),
  break by fixed alphabetical `canonical_policy_id` order
  (`chunked_prefill_small` before `full_prefill`) — i.e. `chunked_prefill_small`
  is the recorded label whenever it exactly ties `full_prefill`. This rule
  is arbitrary but fixed and documented, not chosen to favor any result.
- **Secondary reporting metric: epsilon-optimal accuracy (ε=0.01).** A
  prediction is "correct" under this metric if its achieved ANWG is within
  0.01 of the scenario's true best achievable ANWG, regardless of whether
  it matches the primary target's designated single winner. Reported
  alongside exact-winner accuracy for every regime, per the 46% overall
  tie rate making exact-winner accuracy alone potentially misleading.
- **Primary evaluation metric (not the training target): regret.**
  `regret(x) = best_achievable_ANWG(x) - achieved_ANWG(x)`, where
  `achieved_ANWG(x)` is the true ANWG of whichever policy the selector
  actually picked (looked up from the frozen matrix, never predicted) —
  this is the number that matters, not classification accuracy.

## 4. Split Regimes — Frozen

All splits are **group-aware** (by `group_key`, MF-PSD v1's seed-stripped
scenario-config identity) so seed-variant siblings of the same underlying
configuration never cross a split boundary. Given small group counts
(A: 36, B: 8, C: 12 groups), fixed seed = `20260817` for all randomized
group assignment (`numpy.random.default_rng(20260817)`).

**Regime A — within-family grouped holdout.** Per family independently:
groups split roughly 60/20/20 (train/val/test), rounded to whole groups,
minimum 1 group per split. Family B (8 groups) → 5/1/2; Family C (12
groups) → 7/2/3; Family A (36 groups) → 22/7/7.

**Regime B — multi-family pooled holdout.** All 176 scenarios pooled;
groups (56 total across all 3 families — each family's `group_key`s are
already family-prefixed per MF-PSD v1 §N, so no cross-family collision) split
60/20/20 by count, stratified by family so the split's family mix roughly
matches the pool (not a hard requirement, reported either way).

**Regime C — Leave-One-Family-Out (LOFO), the primary regime.** For each
held-out family F ∈ {A, B, C}: train+val = grouped 80/20 split of the
*other two* families' pooled scenarios; test = **all** scenarios of family
F, untouched by any split/selection/tuning decision. The held-out family's
data is never used for model class selection, hyperparameter choice, or
calibration — verified by a dataflow test (§12).

## 5. Baselines and Models — Frozen, Fixed Hyperparameters (no grid search)

Given the sample size (176 scenarios total, per-fold training sets as
small as ~14 groups), a hyperparameter grid search would itself be
overfitting-prone at this scale; one fixed, standard configuration per
model class is used instead — chosen before looking at any TEST/LOFO
result, documented here:

- **A. Best global fixed policy** — computed from TRAIN only, per split.
- **B. Per-family oracle** — audit-only upper bound, never a "beatable"
  target for the GO gate.
- **C. Global six-policy oracle** — audit-only upper bound.
- **D. Majority baseline** — always predicts TRAIN's most frequent
  primary-target winner class.
- **E. Contextual top-1 classifiers** (three, all reported): multinomial
  `LogisticRegression(C=1.0, max_iter=2000)`;
  `DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=20260817)`;
  `RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=20260817)`.
- **F. Per-policy utility regression + argmax**: one
  `Ridge(alpha=1.0, random_state=20260817)` per policy, predict all 6
  ANWG values, select `argmax`.
- **G. Pairwise model** (simple, as scoped): for each of the 15 policy
  pairs, `LogisticRegression(C=1.0, max_iter=2000)` predicts
  `P(policy_i beats policy_j | x)` (label = 1 if
  `ANWG_i - ANWG_j > 0.01` on TRAIN); at inference, each policy's
  predicted-beats-count across its 5 pairs is tallied and the argmax is
  selected.

No neural networks, no AutoML, no LLM-generated features.

## 6. Metrics and Aggregation — Frozen

Per §8's requirement, for every (regime, held-out-family-if-LOFO,
model): mean/median/p95 regret, fraction of scenarios with regret ≤ 0.01,
exact-winner accuracy, epsilon-optimal accuracy, mean achieved ANWG, gap
to best-fixed, gap to oracle. Reported **overall** (micro, every test
scenario weighted equally), **macro-by-family** (per-family metric
averaged with equal family weight, so Family B's 32 easy scenarios cannot
dominate a pooled number), and **per-family** (each family's own numbers
shown separately). Bootstrap 90% CIs (1,000 resamples, group-aware
resampling by `group_key`) reported for the pooled and LOFO mean-regret
numbers.

## 7. Preregistered Verdict Gates — Frozen Numerical Thresholds

Applied **after** all TEST/LOFO evaluation completes, mechanically, no
post-hoc softening:

**`MULTIFAMILY_SELECTOR_GO`** requires **all** of:
1. Best model's pooled-holdout (Regime B) mean regret ≤ 80% of best-fixed
   baseline's mean regret (i.e. ≥20% relative regret reduction).
2. Regime B epsilon-optimal accuracy ≥ 15 percentage points above the
   majority baseline's epsilon-optimal accuracy.
3. Macro-by-family regret improvement over best-fixed holds in **all 3**
   families individually (not just pooled/micro).
4. LOFO (Regime C): mean regret beats best-fixed baseline (computed
   per-held-out-family, using that family's own best-fixed policy) in
   **at least 2 of 3** held-out-family directions.
5. Shared-feature robustness check (§2, A↔B only): pooled A∪B regret
   improvement over best-fixed survives (any positive improvement,
   non-zero) when restricted to the 2 shared-role features — i.e. the
   result is not purely an artifact of family-specific feature blocks.

**`MULTIFAMILY_SELECTOR_ID_ONLY`**: Regime A/B (in-distribution) satisfies
gate 1–2 above, but LOFO (gate 4) fails on **2 or 3** of the 3 held-out
directions — i.e. pooled/within-family performance is real but does not
transfer to a genuinely unseen mechanism family.

**`MULTIFAMILY_SELECTOR_NO_GO`**: gate 1 or 2 fails (pooled selector does
not meaningfully beat best-fixed even in-distribution).

**`MULTIFAMILY_SELECTOR_INCONCLUSIVE`**: none of the above cleanly applies
— e.g. results are highly sensitive to which model class is used, sample
sizes in a LOFO fold are too small (<10 test scenarios) to support any
claim, or the family-predictability diagnostic (§2) is so extreme (>99%)
combined with a materially degraded shared-feature-only result that
in-distribution gains cannot be attributed to anything but family
identification even before checking LOFO.

## 8. Anti-Leakage Guards — Frozen

- Fitting functions accept only `(X_train, y_train)`; a dataflow test
  asserts no function that fits a model ever receives a row whose
  `canonical_scenario_id` is in the current fold's VAL/TEST/held-out-family
  set.
- `mechanism_family`, `canonical_scenario_id`, `source_scenario_id`,
  `group_key`, `seed`, and every one of the 6 utility columns are asserted
  absent from every `X` matrix passed to any `.fit()` call.
- The held-out LOFO family's scenarios are never touched by group-split
  construction, hyperparameter selection (there is none — §5), or
  calibration for that fold.

## 9. Artifacts

`experiments/multifamily_contextual_selector_v1/` — separate from MF-PSD
and both Step-2 matrices. Harness:
`src/llmserveopt/selector/multifamily_contextual_selector_v1.py`. CLI:
`scripts/run_multifamily_contextual_selector_v1.py`. Tests:
`tests/test_multifamily_contextual_selector_v1.py`.
