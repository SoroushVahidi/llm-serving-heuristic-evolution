# Local Branch Status (Canonical)

**This is the canonical handoff document for the local branch
`phase2c-final-selector-improvement`.** It is scoped to this branch only.
For the full project narrative see [PROJECT_STATUS.md](PROJECT_STATUS.md);
for the integration-branch comparison this document summarizes, see below.

**Status as of:** 2026-07-21, local branch polish pass (classification,
ignore-policy fix, duplicate-script marking, documentation, and test
verification of the branch's uncommitted work).

## 1. Branch purpose

This branch continues Phase 2C selector-improvement work: a corrected-split
Selector v2 calibrated pilot generator, an independent leakage audit, a
reusable causal advanced-selector module, and a bounded final
selector-improvement evaluation over the Phase 2C labeled dataset. It does
not include Policy Library v2, policy composition, or structural synthesis
-- see §6.

## 2. Current commit

Base commit before this polish pass: `8087ee1be3af2ef51e072ff37fb18617338ad10a`
("Add operational agent handoff doc"). No upstream configured; this branch
has never been pushed.

## 3. Uncommitted work finalized in this pass

| Area | Files | Disposition |
|---|---|---|
| Selector v2 split-grouping fix | `src/llmserveopt/selector/dataset_v2/splits.py`, `scripts/audit_selector_v2_calibrated_pilot_leakage.py`, `scripts/build_selector_dataset_v2_calibrated_targeted_pilot.py`, `scripts/selector_v2_calibrated_pilot_gates.py`, `tests/test_selector_dataset_v2.py` | Kept as-is; already a single source of truth (see §4). |
| Advanced causal selectors | `src/llmserveopt/selector/advanced.py`, `tests/test_advanced_selector_models.py` | Kept as-is; already well-factored. |
| Local smoke pipeline | `scripts/run_local_e2e_smoke.py`, `tests/test_local_e2e_smoke.py` | Kept as-is. |
| Clean-pilot evaluation | `scripts/evaluate_selector_v2_clean_pilot.py`, `tests/test_selector_v2_clean_pilot_audit.py` | Kept; supersedes `scripts/train_selector_v2_calibrated_prototype.py` (now marked historical -- see §5). |
| Phase 2C final selector-improvement run | `scripts/run_phase2c_final_selector_improvement.py`, `docs/audits/phase2c_final_selector_improvement_audit.md` | Kept; anchored `--dataset-dir`/`--out-dir` defaults to `ROOT` for cwd-independent reproducibility (previously only `evaluate_selector_v2_clean_pilot.py` and `run_local_e2e_smoke.py` did this). |
| Composition diagnostic pilot | `scripts/run_composition_smart_pilot.py` | Kept; explicitly reports the native Wulver-only composition harness as unavailable rather than approximating it -- does not duplicate `origin/wulver-final-integration-20260721`'s `policies/composition.py`. |
| Documentation | `docs/current/README.md`, `docs/result_claims.md`, `docs/current/REPO_ARCHITECTURE_MAP.md`, `docs/current/ROADMAP_GAP_ANALYSIS.md`, `docs/current/WULVER_HANDOFF.md` | Reviewed and updated; added explicit branch-scope notes (§6) and fixed one stale script reference. |
| Ignore policy | `.gitignore` | Added explicit ignore rules for the two large calibrated-pilot CSVs (see §7). |
| vLLM healthcheck log | `experiments/real_llm/vllm_healthcheck_20260703T171021Z/server.log` | Append-only (verified: 8619 insertions, 0 deletions); real historical request log from 2026-07-03, not new server activity -- no vLLM process is currently running. Pre-existing tracked exception per `.gitignore`. |

## 4. Selector v2 split-leakage fix architecture (this branch)

Single source of truth: `src/llmserveopt/selector/dataset_v2/splits.py`.

- `leakage_safe_split_group_key(row)` / `attach_leakage_safe_split_group_keys(rows)`
  -- derives a `split_group_key` that groups real-trace windows by
  `(request_plan_ancestor_id, time_slice_pool)` so that transformed siblings
  of the same underlying raw row range stay atomic across splits.
- `verify_no_cross_split_row_range_overlap(rows)` -- a second, independent
  check that catches raw row-range reuse even if `split_group_key` grouping
  were ever bypassed.
- Both the pilot builder (`build_selector_dataset_v2_calibrated_targeted_pilot.py`),
  the quality gates (`selector_v2_calibrated_pilot_gates.py`), and the
  independent post-hoc audit (`audit_selector_v2_calibrated_pilot_leakage.py`)
  all import these functions rather than reimplementing grouping logic.

**This is a different implementation from commit `c8aee129f553f8dc3ede99eac60d5b14484beb41`**
("Fix Selector v2 real-trace split grouping") on
`origin/wulver-final-integration-20260721`, which fixes the identical bug
without touching `splits.py` (it patches the audit/builder/gates scripts
directly, in fewer lines, with no new `splits.py` functions). **Both fixes
are unreconciled as of this pass** -- per the task that produced this
branch's changes, no merge/cherry-pick was performed. Reconciling them
(pick one, or combine) is required before this branch's pilot-generation
path can be trusted as the single, final implementation; see §8.

## 5. Advanced selector additions

`src/llmserveopt/selector/advanced.py` (all prediction paths are
causal-`feat_*`-only, enforced by `validate_feature_columns`):

- `PolicyRewardRegressorSelector` -- one regressor per policy (Random
  Forest / Extra Trees / HistGradientBoosting), argmax predicted ANWG.
- `PolicyClassifierSelector` -- multiclass policy classifier with optional
  margin/regret sample weighting (`policy_margin_weights`).
- `PairwisePolicyRanker` -- pairwise policy-vs-policy voting classifiers.
- `UncertaintyFallbackSelector` -- falls back to a fixed policy when the
  top-two predicted-score margin is below a threshold.
- `RegimeGatedSelector` -- routes rows matching a gate (e.g.
  `azure_conv_like_gate`) to a specialist selector, others to a default.

These were exercised in `scripts/run_phase2c_final_selector_improvement.py`;
see `docs/audits/phase2c_final_selector_improvement_audit.md` for the full
result (`SELECTOR_STATUS = IMPROVABLE`, strict best selector still the
prior Phase 2C.3 `native_non_oracle_dt`).

## 6. Known gap vs. `origin/wulver-final-integration-20260721`

`git merge-base HEAD origin/wulver-final-integration-20260721` equals this
branch's own commit -- the integration branch is a **clean fast-forward**
from here, 10 commits ahead, zero divergence. It adds, and this branch does
**not** have:

- The 27-policy Policy Library v2 (7 new policies:
  `sola_style_state_aware`, `slai_style_phase_aware`,
  `flow_control_stability`, `kv_constrained_online`,
  `adaptive_chunked_prefill`, `aging_priority`, `weighted_fair_share`).
- `src/llmserveopt/policies/composition.py` and
  `src/llmserveopt/selector/composition_experiment.py` (policy composition).
- `src/llmserveopt/policies/genome.py` and
  `src/llmserveopt/policies/structural_synthesis.py` (typed scheduler
  genome / structural synthesis readiness).
- Its own copy of the split-leakage fix (`c8aee12`, see §4).

No document on this branch claims otherwise; every doc that lists a policy
count says 20, not 27, and none reference `genome.py`, `composition.py`, or
`structural_synthesis.py` as present here.

## 7. Files/artifacts intentionally excluded from git

- `results/`, `logs/` -- fully gitignored (generated outputs).
- `experiments/**/server.log` -- gitignored except two pre-existing tracked
  exceptions (see `.gitignore` comment).
- `experiments/selector_v2_calibrated_pilot_*/full_policy_vectors.csv` and
  `.../window_features.csv` -- newly added ignore rules in this pass (large,
  regeneratable via `build_selector_dataset_v2_calibrated_targeted_pilot.py`;
  the pilot's small provenance/summary/manifest/audit files alongside them
  remain committed normally).

## 8. Tests

Focused: `tests/test_selector_dataset_v2.py`, `test_selector_v2_clean_pilot_audit.py`,
`test_advanced_selector_models.py`, `test_local_e2e_smoke.py`,
`test_selector_v2_candidate_source_of_truth.py`, `test_selector_candidates.py`,
`test_heuristic_dsl_expressions.py`, `test_heuristic_dsl_verifier.py`,
`test_heuristic_dsl_no_leakage.py`: **152 passed**.

Full non-GPU suite (`python3 -m pytest -q -m "not gpu"`): **2508 passed, 2
skipped, 3 deselected (gpu), 0 failed**, 205.3s. The one warning present
(`test_disaggregated_prefill_decode.py`, `UserWarning`) is pre-existing and
unrelated to this branch's changes.

## 9. Exact next synchronization step

1. Reconcile the split-leakage fix (§4): diff this branch's `splits.py`-based
   approach against `c8aee129f553f8dc3ede99eac60d5b14484beb41`'s
   script-level approach; pick or combine one implementation. Do this
   *before* trusting either branch's pilot-generation output as final.
2. Once reconciled, branch future Policy Library v2 / composition /
   structural-synthesis / genome work from
   `origin/wulver-final-integration-20260721` (not from this branch), since
   that is the only branch containing those prerequisites.
3. Do not modify `src/llmserveopt/policies/registry.py` or any composition/
   structural-synthesis file on this branch without first reconciling
   against the integration branch -- active Wulver HPC workflows (Policy
   Frontier Cartography, Policy Library V2 Expanded Frontier, V2 real-OOD
   evaluation) depend on that branch's exact registry and composition code.
