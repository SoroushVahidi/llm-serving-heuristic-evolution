# vLLM-LTR Comparative-Evaluation Recovery — 2026-08-04

Recovery report for the vLLM-LTR comparative-evaluation task, following an
independent audit that found the prior attempt never produced results. This
document records the audit evidence, the diagnosed root causes, the fixes
applied, and (once complete) the recovered experiment's outputs.

## 1. Audit evidence (captured before any code was touched)

**Branch:** `contextual-compositional-heuristics-20260731`
**Starting SHA:** `d64ed68d941e13aa612ce1fb15b96d0f3e68c329`
(`git log -1`: `d64ed68d941e13aa612ce1fb15b96d0f3e68c329 2026-08-04 01:02:44 -0400`)

**Working-tree state at the start of recovery** (identical to the state
found by the preceding independent audit — no drift in between):

```
Changes not staged for commit:
	modified:   baselines/vllm_ltr/adapter/checkpoint_loader.py

Untracked files:
	external/datasets/wildchat.md
	scripts/ingest_wildchat_eval_dataset.py
	scripts/run_vllm_ltr_first_comparative_evaluation.py
	scripts/score_vllm_ltr_eval_dataset.py
	tests/test_vllm_ltr_eval_dataset_scoring.py
	tests/test_wildchat_ingestion.py
```

Local branch was up to date with `origin/contextual-compositional-heuristics-20260731`
(0 ahead / 0 behind) at the start of recovery.

**Exact reproduction command that hung** (the comparative-evaluation
script, one seed, default config):

```
python3 scripts/run_vllm_ltr_first_comparative_evaluation.py --seeds 0
```

Observed behavior: printed progress through 8 of 10 policies for seed 0
(`fifo`, `edf`, `estimated_service_time_first`, `shortest_output_first`,
`weighted_shortest_processing`, `scorpio_style_slo_guard`,
`rule_based_selector`, then the header line for `regression_anwg_selector`)
and then produced no further output for 9+ CPU-minutes before being
killed. **Initial diagnosis suspected the new `vllm_ltr_semantic_reference`
policy** (its header line printed most recently in the earlier
`--seeds 0 1 2` tmux run this recovery inherited) — this was wrong, and
disproven by direct isolation: `VLLMLTRSemanticReferencePolicy` alone
completes all 300 requests in 0.47s. The true hang was one policy earlier:
`regression_anwg_selector` (`SelectorDispatchPolicy` wrapping
`PerPolicyRegressionAnwgSelector`), which the eval script's own print
ordering (name printed *before* the policy runs, not after) made easy to
misattribute. Isolating the exact policy sequence up to and including
`regression_anwg_selector` reproduced the hang standalone.

## 2. Root cause 1 (performance): `PerPolicyRegressionAnwgSelector.predict_one()`

**Measured latency, production artifact**
(`results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib`,
100 estimators/regressor, max_depth=8, 20 candidate-policy regressors):

| | before fix | after fix | speedup |
|---|---|---|---|
| `predict_one()` | 56.24 ms/call | 4.16 ms/call | **13.5x** |
| single `RandomForestRegressor.predict()` call (1 of 20) | 2.81 ms/call | n/a (bypassed) | — |

**Diagnosis.** `predict_one()` → `predict([features])[0]` → for each of the
20 `SELECTOR_CANDIDATES`, a separate top-level `RandomForestRegressor.predict(X)`
call on a **batch of 1 row**. `cProfile` over 300 reps × 20 regressor calls
(6000 calls, 45.69s under profiler overhead) attributed the cost almost
entirely to sklearn's own per-call machinery, not to the actual tree
traversal:

- ~85% of wall time: `sklearn.utils.parallel.py:140 __call__` — the
  per-task wrapper `joblib.Parallel` uses to propagate `sklearn.get_config()`
  into each of the 100 per-tree prediction jobs, paid on *every* forest
  `.predict()` call regardless of `n_jobs`.
- ~28%: `sklearn/tree/_classes.py:509 predict` — each of the 100
  individual `DecisionTreeRegressor.predict()` calls inside the forest
  re-runs `check_is_fitted`, `__sklearn_tags__`/`get_tags` (sklearn 1.6+
  tag system), and `warnings.filterwarnings`/`simplefilter` context
  management, every single call.

This overhead is designed to be amortized over large batches; calling
`.predict()` once per candidate policy, once per simulator step, on a
single row, pays the full fixed cost every time. With 20 regressors called
once per step over the tens of thousands of steps a real long-response
WildChat workload needs to drain, this is what made the run never finish.

**Fix** (`src/llmserveopt/selector/models.py`): `_fast_forest_predict()`
bypasses sklearn's high-level API entirely — it averages each fitted
tree's compiled `tree_.predict()` output directly (the same Cython call
sklearn's internals eventually make), using the *exact* accumulation
sklearn's own `_accumulate_prediction` uses (plain running sum over
estimators in order, divided by `n_estimators` — not `np.mean`, whose
pairwise-summation algorithm can differ in the last bit).

**Equivalence proof.** Verified **bit-exact** (`max abs diff == 0.0`, not
"close"): 20 regressors × 200–4000 random feature vectors each, both in an
ad hoc script and in `tests/test_selector_regression_anwg_fast_path.py`
(`test_fast_path_bit_exact_vs_official_predict`). A per-regressor self-check
(`_verify_fast_forest_predict`) runs automatically at `fit()`/`load()` time;
any regressor that ever fails it (e.g. a future sklearn internals change)
individually falls back to the slow-but-always-correct `reg.predict()`
path rather than silently drifting — `_fast_path_ok` records this per
policy and is asserted `True` for the production artifact in
`test_load_reverifies_fast_path`.

**Actual step count and cost (measured, not just extrapolated).** WildChat
response lengths for this sample: p50=459, p95=1130, max=4591 output
tokens (300 requests). A naive `n_requests × mean_output_tokens /
max_active_sequences` lower bound (≈17,200 steps) badly underestimates the
real cost: instrumenting `predict_one()` call counts directly showed a
**steady, non-hung** rate of ≈227 calls/s (consistent with the measured
4.16 ms/call fast-path cost — i.e. wall-clock is now genuinely dominated
by call *count*, not per-call overhead), but simulated time only advanced
≈22 units in the first 100s of real time — this workload's real total step
count is far larger than the naive estimate (heavy-tailed real response
lengths at `step_size=0.001` drive many more decode steps than a
mean-based estimate suggests). Empirically: **one seed's
`regression_anwg_selector` policy alone did not finish within 900s
(15 min)** even post-fix, confirming the naive ≈72s/seed estimate below
was wrong. The fix is confirmed working correctly (steady progress, no
hang, bit-exact/correct decisions) — the *remaining* cost is genuine
simulator step volume for this real, heavy-tailed workload, not a
residual performance bug. The full 3-seed run was therefore launched as a
long-running background job (tmux, generous 3-hour hard cap) rather than
assumed to finish in a couple of minutes. See §5 for actual observed
runtime.

_Superseded estimate, kept for the record of how the initial (wrong)
sizing was arrived at:_ at 56.24 ms/call, ≈17,200 steps/seed implied
≈1,147s (≈19 min)/seed pre-fix, ≈72s/seed post-fix -- both figures
undercounted real step volume by roughly an order of magnitude, as the
15-minute non-completion above demonstrates.

## 3. Root cause 2 (correctness, found while implementing the performance
fix — separate bug, user-authorized fix)

`_feature_matrix()` (shared by `DecisionTreeSelector`, `RandomForestSelector`,
`PerPolicyRegressionAnwgSelector`) only recognized `feat_`-prefixed dict
keys (the persisted dataset-row column convention). The simulator's live
`llmserveopt.selector.features.extract_features()` — the function
`SelectorDispatchPolicy` actually calls every step, in both this new eval
script and the earlier `scripts/run_vllm_external_baseline_comparison.py`
— returns **bare** (unprefixed) keys. Result: every live-simulator-driven
prediction silently received an all-zero feature vector.

**Verified impact:** 50/50 random real-valued feature draws, fed through
`extract_features()`'s key shape, all resolved to the same policy
(`edf`) — i.e. `regression_anwg_selector`, whenever driven live by a
`SelectorDispatchPolicy`-style wrapper, was not actually responding to
queue/KV/SLO state at all. The same selector correctly diversified across
6 different policies when fed `feat_`-prefixed keys (the offline
evaluation format), so the Phase 2B.16 offline-eval numbers
(`docs/result_claims.md`, 0.9856 arrival_normalized_wg) are not affected —
only the live-simulator dispatch path was broken. `RuleBasedSelector._get()`
already handled both key formats and was unaffected.

This was flagged to the user (not decided unilaterally, per this task's
explicit "do not change the selector's scientific behavior without
direction" instruction) — **user selected "fix the key-mismatch bug"**.
`_feature_matrix()` now mirrors `RuleBasedSelector._get()`'s dual-format
lookup (bare name first, `feat_`-prefixed fallback). Verified post-fix: the
same 50-draw live-key test now diversifies across 6 policies again.
Regression tests: `tests/test_selector_regression_anwg_fast_path.py::TestFeatureKeyFormatBugFix*`.

## 4. Other bugs fixed (see individual commits/diffs for detail)

- **EST/SOF ranking-agreement conflation**
  (`scripts/run_vllm_ltr_first_comparative_evaluation.py`): `est_order` and
  `sof_order` were computed with the identical formula
  (`-predicted_output_tokens`), so "agreement with EST" and "agreement with
  SOF" were always numerically identical — silently hiding that
  `estimated_service_time_first` actually ranks by
  `alpha*prompt_tokens + beta*predicted_output_tokens`
  (`EstimatedServiceTimeFirstPolicy._sort_key`). Fixed to use
  `llmserveopt.policies.scoring.predicted_service_proxy` directly (the real
  policy's own formula) rather than a duplicated local approximation.
  Regression test: `tests/test_vllm_ltr_comparative_evaluation_ranking.py`.

- **Tokenizer truncation** (`baselines/vllm_ltr/adapter/checkpoint_loader.py`):
  confirmed real — `facebook/opt-125m`'s tokenizer ships `model_max_length`
  at HF's ~1e30 "unset" sentinel, so bare `truncation=True` is a no-op; 14
  of the 300 real WildChat prompts exceed the checkpoint's actual
  `max_position_embeddings=2048`. The existing fix (explicit
  `max_length=model.config.max_position_embeddings`) had zero regression
  coverage and the independent GPU fidelity cross-check path
  (`tests/test_vllm_ltr_checkpoint_fidelity_gpu.py`) used a different,
  unbounded tokenizer call. Both fixed: `max_length` is now applied
  identically in both paths; `OPTPredictorHandle` gained a
  `num_prompts_truncated` counter (truncation metadata, incremented via a
  cheap pre-check against the model's real limit, asserted never exceeded
  post-truncation); 5 new GPU-gated tests in
  `TestLongPromptTruncation` (all pass locally, `LLMSERVEOPT_RUN_GPU_TESTS=1`,
  36.64s, 18/18 in that file).

- **False "sorted-hash order" documentation claim**
  (`scripts/ingest_wildchat_eval_dataset.py`): docstrings claimed
  `request_id`s are assigned in `conversation_hash`-sorted order. Verified
  false — `random.Random(seed).sample()` returns a seeded-random
  permutation of its (sorted) input, not the sorted order itself;
  determinism (same seed ⇒ same output) was never actually broken, only
  the ordering claim was wrong. Docstrings corrected; a real, previously
  misleading test (`test_request_ids_assigned_in_sorted_hash_order`) was
  renamed and clarified; a new regression test
  (`test_output_order_is_not_necessarily_hash_sorted`) locks in the
  corrected understanding with a verified non-sorted seed/pool case.

- **Duplicate-prompt reporting**: the real 300-row WildChat sample has 2
  rows (`request_id` 138, 213) sharing byte-identical prompt text under
  different `conversation_hash` values (real WildChat duplication, not an
  ingestion bug). Not deduplicated (no spec requirement to do so, and
  dropping one would silently redefine what `--sample-size` means); now
  recorded explicitly via `duplicate_prompt_summary()` in
  `wildchat_eval_manifest.json`'s new `duplicate_prompt_accounting` field
  (total/unique counts + duplicate groups).

## 5. Experiment (recovered run)

**Machine:** local workstation (`al-khwarizmi`, 20 vCPU, RTX 5060 Ti — GPU
unused for this run; vLLM-LTR scoring was done offline in advance).
**tmux session:** `vllm_ltr_comparison_recovery`.
**Command** (replay, per `run_manifest.json`):

```
python3 scripts/run_vllm_ltr_first_comparative_evaluation.py \
    --pairs-path data/processed/wildchat/wildchat_eval_sharegpt_shaped.json \
    --prompts-path data/processed/wildchat/wildchat_eval_prompts_by_id.json \
    --score-cache-path data/processed/wildchat/vllm_ltr_score_cache.json \
    --selector-artifact results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib \
    --tokenizer facebook/opt-125m --seeds 0 1 2 \
    --output-dir results/vllm_ltr_first_comparative_evaluation
```

**Two runs were executed, not one:**

1. A first full 3-seed run (log `full_run_20260804T154550Z.log`,
   `real 37m57.9s`, `FULL_RUN_EXIT_STATUS=0`) completed successfully with
   the fast-path-fixed selector, confirming the performance fix actually
   resolves the original hang (§2) under the full, un-truncated 300-prompt
   x 3-seed x 10-policy workload — not just in the isolated `predict_one()`
   micro-benchmark. Its outputs are preserved unmodified at
   `results/vllm_ltr_first_comparative_evaluation_v1_no_raw_rows_20260804/`
   for provenance and as a determinism cross-check.
2. Immediately after, `scripts/run_vllm_ltr_first_comparative_evaluation.py`
   was extended to also persist raw per-request outcome rows
   (`request_level_outcomes.csv`, one row per (policy, seed, request_id):
   `policy, seed, request_id, priority, class_id, status, slo_violated`) --
   the run's own in-memory `rows_by_policy` was already being computed for
   `compute_bootstrap_ci()` but was previously discarded after use, which
   would have limited independent re-verification (§6, §10 of the task) to
   re-checking pre-aggregated ratios rather than genuinely raw,
   per-request data. This is a strictly additive change (no simulator,
   policy, or metric-computation code was touched) and was re-run
   identically (same command, same seeds, same config) into the same
   output directory: **this second run's outputs are authoritative** for
   `results/vllm_ltr_first_comparative_evaluation/`. Its own log is
   `full_run_v2_with_raw_rows_<timestamp>.log` in that directory.

Both runs' aggregate `run_metrics.csv` numbers were confirmed identical
(the discrete-event simulator and every policy under test are
deterministic given identical seeds/config), which is itself a useful
correctness cross-check on top of the raw-row addition.

**Every requested policy finished, every seed finished, no crash /
timeout / interruption / silent retry occurred** in either run (both
exited status 0 well under the 3-hour hard cap; `10,800s` cap vs.
`~2,278s` (v1) / see run manifest timestamps (v2) actual wall-clock).

**Prompt count:** 300 unique WildChat rows sampled (2 near-duplicate
prompt-text pairs recorded in `duplicate_prompt_accounting`, not
deduplicated — see §4). **Seeds:** 0, 1, 2 (3 total). **Policies:** the 10
listed in §"Policies compared" of
`docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`.

## 6. Independent re-verification

`scripts/verify_vllm_ltr_comparison_results.py` (new script, written
specifically so this step does not reuse the eval script's own aggregation
functions) recomputes ANWG, completion fraction, a paired bootstrap CI
(resampling (seed, request_id) keys, mirroring the eval script's own
resampling granularity but coded from scratch), pairwise win/tie/loss,
oracle-envelope contribution, per-regime (SLO `class_id`) ANWG, and an
independently re-implemented Spearman rank correlation for the EST/SOF
ranking-agreement cross-check -- entirely from `request_level_outcomes.csv`
(request-level mode) rather than trusting `run_metrics.csv`,
`bootstrap_confidence_intervals.json`, or `ranking_agreement.json`.

**Raw row count check:** expected `300 prompts x 3 seeds x 10 policies =
9,000`; actual `9,000`. Matches.

**Result: zero mismatches found**, after two bugs in the *verifier itself*
were caught and fixed during this process (documented here rather than
silently corrected, since catching your own checker's bugs is exactly what
independent verification is for):

1. The completion-fraction/SLO-violation-rate cross-check initially used a
   `1e-9` tolerance against `run_metrics.csv`'s columns, which are written
   through the eval script's `metrics_to_dict()`/`_fmt()` (rounds to 6
   decimal places) -- every non-terminating ratio (e.g. `1/300`) falsely
   "mismatched" by ~3e-7. Not a bug in the eval script; fixed by widening
   the verifier's tolerance to `6e-7` (just above the rounding's max
   possible error).
2. The ranking-agreement cross-check initially used a strict `1e-6`
   tolerance, but the eval script's `_rank_correlation()` ranks via
   `np.argsort(np.argsort(x))` with **no tie-averaging**, while this
   verifier's independently-implemented Spearman correlation uses
   **average-rank tie-handling** (the textbook-correct convention). Real
   ties exist in `predicted_output_tokens`/EST/SOF scores for this
   dataset, so the two conventions differ by up to ~1.2e-4 -- a genuine
   methodological difference, not a computational error (confirmed by the
   magnitude: far too small and too systematic to be a formula bug, and
   traceable to a specific, named cause). Documented in the verifier's own
   output (`tie_handling_note`) with a tolerance (`5e-3`) set above the
   expected gap.

After both fixes: `independent_verification_report.json` reports
0 completion/SLO-rate mismatches (30/30 match), 0 accounting-identity
violations, 0 ANWG cross-check mismatches (request-level ANWG recomputed
from raw rows exactly matches `run_metrics.csv`'s column to within
floating-point/rounding precision), and `ranking_agreement_cross_check.all_match
= True`.

**Fairness/leakage checks:** every policy in a given seed runs the
identical `Request` list (same `Simulator`, same `SimulatorConfig`, same
trace, same admission opportunities) -- verified by construction (the eval
script builds `requests` once per seed and passes the same list object to
every policy's run). vLLM-LTR's scores were computed offline from prompt
text only (`scripts/score_vllm_ltr_eval_dataset.py`), never from
`actual_output_tokens` or any post-hoc field; `VLLMLTRSemanticReferencePolicy`
is not selector-eligible and was not added to any registry this run. No
future information (`actual_output_tokens`, completion time, SLO outcome)
is available to any policy's `select_action()` at decision time -- this is
an existing, previously-verified invariant of the simulator's
`ObservableState`/`ObservableRequest` split, unchanged by this recovery.

**Final confirmation (post-completion):** the v2 run (§5, the one with
`request_level_outcomes.csv`) completed successfully (`FULL_RUN_V2_EXIT_STATUS=0`,
`real 37m54.5s` -- matching v1's `37m57.9s` almost exactly, a second
independent confirmation of determinism).
`scripts/verify_vllm_ltr_comparison_results.py` run in its full
request-level mode against it: raw row count 9,000/9,000 expected
(`300 prompts x 3 seeds x 10 policies`), 0 ANWG cross-check mismatches
against `run_metrics.csv`, 0 completion/SLO-rate mismatches, 0 accounting-
identity violations, `ranking_agreement_cross_check.all_match = True`, and
the independently-recomputed paired-request bootstrap CI matches the run's
own `bootstrap_confidence_intervals.json` exactly (e.g. the 8 tied
policies' CI is `[0.9901, 1.0000]` under both). Every ANWG figure in
`docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md` is drawn
from this fully-verified v2 run.
