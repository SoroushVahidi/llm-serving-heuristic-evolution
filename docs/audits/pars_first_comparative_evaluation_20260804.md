# PARS-Serve-2026 First Comparative Evaluation — 2026-08-04

**STATUS: COMPLETE AND INDEPENDENTLY VERIFIED.** The comparison sweep
covers WildChat control + all 7 accepted canonical-suite families (8
workloads × 3 seeds × 10 policies). The original single-invocation run
(`tmux` session `pars_comparison`, `timeout 10800`) completed 3 of 8
families before being killed by its own timeout; the remaining 5 families
were completed by isolated, per-family recovery runs and merged into the
same final result set. Every reported number was independently
recomputed from raw per-request outcome rows
(`scripts/verify_pars_comparison_results.py`) with **zero unexplained
mismatches** (one small, fully-explained discrepancy — see §5).

## What this evaluation is

The first comparative simulator sweep pitting the offline-scored
PARS-Serve-2026 baseline (`baselines/pars/`, locally-trained checkpoint,
see [`pars_baseline_implementation_20260804.md`](pars_baseline_implementation_20260804.md))
against this repo's existing fixed policies, selectors, and oracle — run
across both the real WildChat control workload and the full
canonical-suite discriminative benchmark (the same suite vLLM-LTR's own
comparative evaluation predates, see
[`vllm_ltr_first_comparative_evaluation_20260804.md`](vllm_ltr_first_comparative_evaluation_20260804.md)).

## Checkpoint and dataset provenance

- **Checkpoint:** `results/pars_official/predictor_train/alpaca_gpt4_bert/best_model.pt`,
  SHA256 `d54be0871ebc9f2c2538b4e53da7f45cb57ae678563488822cdc1694bc33eb27`
  (matches pinned `d54be087...c33eb27`). Trained locally with the
  official, unmodified `predictor_train/scripts/train_pairwise_bert.py`
  on `vicgalle/alpaca-gpt4` (CC BY-NC 4.0); `best_val_accuracy=0.9141`
  (epoch 2 of 3). Full history:
  `results/pars_official/predictor_train/alpaca_gpt4_bert/metrics.json`.
- **Official repository:** `SPEAR-UIC/PARS`, pinned commit
  `fd4e125b65bb73aef5eccafa79c2509434be61ec`. **No upstream LICENSE
  file** — disclosed in `baselines/pars/PROVENANCE.md`; proceeding is a
  user-directed, local/non-commercial-research-use decision.
- **WildChat control:** `allenai/WildChat-1M`, pinned revision
  `7d6490e462285cf85d91eabea0f9a954fbddcd1f` (same 300-conversation sample
  as the vLLM-LTR evaluation, for direct comparability).
- **Canonical suite:** `benchmarks/canonical_suite/`, `suite_version 1.0`,
  generated `2026-08-04T17:32:10Z`. 7 accepted families (2 rejected —
  `overloaded_queue`, `kv_budget_pressure` — see
  `docs/audits/canonical_benchmark_suite_design_20260804.md`).
- **Score caches (reused unchanged by every recovery run, never
  rescored):** `results/pars_official/wildchat_score_cache.json` +
  `results/pars_official/canonical_suite_matched/<family>/seed_<n>_score_cache.json`
  for the 7 canonical families. Each synthetic request is matched to the
  nearest-token-count real WildChat prompt
  (`scripts/score_pars_eval_datasets.py`) since the canonical suite
  carries no prompt text — quantified per-family in each
  `matching_manifest.json`, never hidden.
- **Selector artifact:** `results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib`
  (identical artifact used by every other comparative evaluation on this
  branch, for cross-run comparability).

## Policies compared (same set as the vLLM-LTR evaluation, PARS in place of vLLM-LTR)

1. `fifo` 2. `edf` 3. `estimated_service_time_first`
4. `shortest_output_first` 5. `weighted_shortest_processing`
6. `scorpio_style_slo_guard` (best fixed baseline) 7. `rule_based_selector`
8. `regression_anwg_selector` (best global composition)
9. `pars_semantic_reference` (offline-scored; `SELECTOR_ELIGIBLE = False`,
   evaluation-only) 10. `oracle_srtf` (non-deployable hindsight ceiling)

## §1 — Original run and timeout

- **Command:** `timeout 10800 python3 -u scripts/run_pars_first_comparative_evaluation.py --seeds 0 1 2`
- **tmux session:** `pars_comparison`
- **Log:** `results/pars_first_comparative_evaluation_run_20260804T201709Z.log`
- **Exit:** killed by `timeout(1)` SIGTERM at **2026-08-04T23:17:09Z**,
  elapsed **180m00.040s** (exactly 3h00m00.04s — the shell's own
  `Terminated` message and `real 180m0.040s` timing line are in the log).
  The wrapper script's own `PARS_COMPARISON_EXIT=0` line reflects `tee`'s
  pipe exit status, **not** the killed Python process's status (a pipe
  without `pipefail` always reports the last command's status) — this was
  confirmed by direct inspection, not assumed.
- **Families completed before the kill:** `wildchat`,
  `staggered_heterogeneous`, `burst_independent_lengths` (all 3 seeds ×
  10 policies each, full artifact set, verified in §3).
- **Family interrupted mid-flight:** `mixed_tight_deadlines` — last
  printed progress line was `seed=2 policy=regression_anwg_selector`.
  Per `run_one_workload()`'s own write ordering (all 6 output files are
  written once, only after all 3 seeds × 10 policies finish), the
  interrupted family's output directory contained **no partial/truncated
  files at all** — confirmed empty via
  `directory_listing_at_interruption.txt` in the forensic quarantine
  (§2). Nothing computed for this family survived the kill; nothing was
  lost that needed recovering beyond re-running it from scratch.
- **Families never started:** `priority_vs_service_time_conflict`,
  `prediction_noise_regime`, `long_output_tail`, `burst_arrivals_isolated`
  (the single-invocation script runs workloads strictly sequentially in
  `suite_manifest.json`'s `accepted` order, so these 4 were never reached).

## §2 — Forensic quarantine

`results/pars_first_comparative_evaluation_FORENSIC_mixed_tight_deadlines_interrupted_20260804T231942Z/`
preserves the empty `mixed_tight_deadlines` output directory listing, the
relevant log excerpt, and a `README.md` recording the full disposition
reasoning above. No data-bearing artifact needed quarantining (the
directory was empty), but the empty directory was preserved rather than
silently overwritten, per the "never delete evidence" instruction. This
family was recreated from scratch in a clean recovery output directory
(§3), not resumed or patched in place.

## §3 — Recovery runs

The 5 missing/interrupted families
(`mixed_tight_deadlines`, `priority_vs_service_time_conflict`,
`prediction_noise_regime`, `long_output_tail`, `burst_arrivals_isolated`)
were each run in isolation, all 3 seeds × 10 policies, into
`results/pars_recovery/<family>/`:

```
scripts/run_pars_first_comparative_evaluation.py \
    --workloads <family> --seeds 0 1 2 \
    --output-dir results/pars_recovery/<family>
```

Per-family logs: `results/pars_recovery/<family>_run_20260804T232002Z.log`.
Per-family manifests (command, input score-cache SHA256 hashes, output
paths): `results/pars_recovery/<family>/run_manifest.json`. All 5 runs
completed with exit code 0 and wrote their manifest as the final step
(confirming no truncation) — reusing the existing score caches
unchanged (input-hash-verified against `results/pars_official/canonical_suite_matched/`,
identical to the hashes recorded by the original run's own manifest
pattern) and never rescoring any prompt.

**Caveat, disclosed:** the 5 recovery runs' log/manifest timestamps
(all completing within an 85-second window of each other, each after
~54 minutes of measured `wall_clock_s`) indicate they ran **concurrently**
on the same machine rather than strictly sequentially. This does not
affect correctness (each family's simulation is independent, uses only
its own score cache and requests, and the independent verifier confirms
identical, internally-consistent results — see §4), but wall-clock timing
numbers reported in §6 reflect CPU-contended time, not isolated
single-process time, and per-family timing comparisons across the 5
recovered families should be read with that caveat. No tmux session-name
record survived process exit to confirm the exact
`pars_recovery_<family_name>` naming was used for each; the per-family
output directories, logs, and manifests are the authoritative record of
what ran.

## §4 — Verification (both completed-family and recovered-family)

**3 originally-completed families** (`wildchat`, `staggered_heterogeneous`,
`burst_independent_lengths`): all 6 required artifacts present; all 3
seeds × 10 policies present (30/30 unique combos, 0 duplicates); raw
request-row counts match the per-seed `n_req` values printed in the
original log exactly (9000, 7150, 8080 rows respectively — accounting for
each family's own per-seed Poisson-arrival request count, not a uniform
count); no truncated rows (constant 7-field row width); all 4 JSON
artifacts parse. SHA256 hashes recorded in
`results/pars_first_comparative_evaluation/run_manifest.json`.

**5 recovered families:** identical checks, all passing — plus the
official `scripts/verify_pars_comparison_results.py` was run against each
family in isolation (`--output-dir results/pars_recovery/<family>`) and
against the final assembled 8-family directory
(`--output-dir results/pars_first_comparative_evaluation`).

**Verifier bug found and fixed:** the verifier's row-count check
originally assumed a uniform per-seed request count
(`n_prompts_expected * n_seeds * n_policies`, inferred from a single
row's `num_total`). This is correct for WildChat (n_req=300 every seed)
but wrong for every canonical-suite family, which uses Poisson arrivals
and therefore has a *different* `num_total` per seed by design (e.g.
`mixed_tight_deadlines`: 246 / 240 / 229 across seeds 0/1/2). This
produced a false "row count mismatch" on all 5 canonical-suite recovery
families on first run. Fixed in `scripts/verify_pars_comparison_results.py`
(`verify_workload` now sums each seed's own `num_total` rather than
multiplying a single uniform value) — a narrow bug fix to the
verification tooling itself, not to PARS, the simulator, the benchmark
suite, or any metric definition. After the fix, all 8 families show
`row count: matches=True`.

**One remaining, fully-explained discrepancy:** `staggered_heterogeneous`
seed=1's independently-recomputed `spearman_pars_vs_shortest_output_first`
is `0.02759` vs. the run's own recorded `0.03366` (Δ=0.0061, just over the
verifier's 0.005 tolerance). Root cause: the verifier's
`independent_rank_correlation` deliberately uses **average-rank**
tie-breaking (proper Spearman tie handling, reimplemented from scratch to
be a true independent check) while the original eval script's
`_rank_correlation` uses **positional** tie-breaking
(`np.argsort(np.argsort(x))`, applied consistently across the whole
evaluation). These are two legitimate, different conventions that only
diverge when the underlying data has ties — which happens here because
the correlation itself is near zero (both computations agree there is no
meaningful PARS-vs-SOF monotonic relationship in this seed). All other
14 seed/family/policy-pair ranking-agreement checks match within
tolerance. This is not a data-integrity or computation defect.

**Independent-verification report:**
`results/pars_first_comparative_evaluation/independent_verification_report.json`
(per-family row counts, ANWG cross-checks, completion-accounting
cross-checks, ranking-agreement cross-checks, paired bootstrap CIs,
win/tie/loss, oracle-envelope contribution, per-regime breakdown,
completion-violation counts — all independently recomputed from raw
`request_level_outcomes.csv` rows, not read from the run's own summary
files).

**Result: zero unexplained mismatches across all 8 families** (row
counts, ANWG, completion accounting: 0/0/0 mismatches everywhere; the one
ranking-agreement near-tolerance discrepancy is explained above, not
unexplained).

## §5 — Per-family scientific results

ANWG = arrival-normalized weighted goodput (this project's primary
metric; `arrival_normalized_weighted_goodput` in `run_metrics.csv`).
PARS completion fraction is **1.0 in every family** — it never drops a
request; any shortfall is entirely in latency-weighted goodput.

| Family | Best policy (ANWG) | PARS ANWG | PARS rank /10 | Regret vs. oracle | PARS vs. best | PARS vs. FIFO | PARS unique wins |
|---|---|---|---|---|---|---|---|
| `wildchat` (control) | `fifo` (0.9957) | 0.9957 | 7 (tied) | 0.0000 | not significant (tied) | not significant (tied) | 0 |
| `staggered_heterogeneous` | `scorpio_style_slo_guard` (0.8154) | 0.7399 | 7 | 0.0424 | **significantly worse** | not significant | 0 |
| `burst_independent_lengths` | `shortest_output_first` (0.7079) | 0.4084 | 7 | 0.2982 | **significantly worse** | **significantly better** | 0 |
| `mixed_tight_deadlines` | `estimated_service_time_first` (0.5642) | 0.5494 | **5** (best rank observed) | 0.0125 | not significant | not significant | 0 |
| `priority_vs_service_time_conflict` | `scorpio_style_slo_guard` (0.8004) | 0.6844 | 8 | 0.0402 | **significantly worse** | not significant | 0 |
| `prediction_noise_regime` | `shortest_output_first` (0.7944) | 0.7580 | 7 | 0.0366 | not significant | not significant | 0 |
| `long_output_tail` | `scorpio_style_slo_guard` (0.5680) | 0.2928 | 7 | 0.1055 | **significantly worse** | **significantly better** | 0 |
| `burst_arrivals_isolated` | `scorpio_style_slo_guard` (0.4225) | 0.2207 | 7 | 0.1810 | **significantly worse** | **significantly better** | 0 |

("Significant" = non-overlapping paired-request bootstrap 95% CIs,
recomputed independently by the verifier, not read from the run's own
`bootstrap_confidence_intervals.json`.)

**Reading this table:** PARS is never the best or second-best policy in
any family, has zero unique wins anywhere, and is statistically
significantly worse than the top policy in 5 of 8 families. It *is*
significantly better than FIFO/EDF in the 3 burst/long-tail-heavy
regimes (`burst_independent_lengths`, `long_output_tail`,
`burst_arrivals_isolated`) — its learned length-prediction signal is
real, not vacuous, but is consistently dominated by the much simpler
`shortest_output_first`/`estimated_service_time_first` heuristics and by
this project's best adaptive/fixed policies
(`scorpio_style_slo_guard`, `regression_anwg_selector`) in every
discriminative regime.

## §6 — Comparison with vLLM-LTR

**Only WildChat control is directly comparable** — vLLM-LTR's own
comparative evaluation (`results/vllm_ltr_first_comparative_evaluation/`)
predates the canonical suite and was run on WildChat only. On that one
shared workload, PARS and vLLM-LTR are **statistically indistinguishable**:
both `pars_semantic_reference` and `vllm_ltr_semantic_reference` score
exactly 0.9957 ANWG, tied with 6 of the other 8 policies. This is not
evidence the two methods behave identically in general — WildChat control
is itself non-discriminative by design
(`benchmarks/canonical_suite/suite_manifest.json`'s own
`control_workload_headroom.metrics.queue_contention_fraction ≈ 0.498`,
`fifo_srtf_decision_disagreement_fraction ≈ 0.005` — almost no room for
any scheduling policy to matter there). **No vLLM-LTR run exists on the
canonical suite's 7 discriminative families** — extending vLLM-LTR to
that suite was out of scope for this recovery task (explicitly: "wait for
the current `pars_comparison` run to exit... do not begin VTC or any
other baseline yet") and was not performed. Behavioral distinctness and
complementarity between PARS and vLLM-LTR therefore **cannot be
established from this evaluation** beyond the one non-discriminative
control workload where they tie; this is recorded as an open evaluation
gap, not resolved either way.

## §7 — Classification

**PARS-Serve-2026: EVALUATION_ONLY.**

- **Not FOUNDATIONAL:** zero unique wins across 8 families; best rank
  observed is 5th of 10; consistently dominated by cheaper existing
  policies (`shortest_output_first`, `estimated_service_time_first`) and
  by this project's best fixed/adaptive policies
  (`scorpio_style_slo_guard`, `regression_anwg_selector`) in every
  discriminative regime. No case for selector-candidate or
  deployable-policy promotion.
- **Not REJECTED:** the integration is real, working, and verified — 22/22
  adapter unit tests, 10/10 real-checkpoint fidelity tests, a
  hash-verified checkpoint trained with the unmodified official script,
  and a fully independently-verified 8-family comparative evaluation with
  zero unexplained mismatches. It is a legitimate, reproducible external
  baseline worth retaining for future comparison, exactly like vLLM-LTR's
  own EVALUATION_ONLY status.
- **Not INCONCLUSIVE:** the evidence is not ambiguous — 5 of 8 families
  show statistically significant underperformance vs. the best policy
  with tight, non-overlapping bootstrap CIs; the 3 "not significant"
  families are consistent with solid mid-pack performance, not with
  top-tier candidacy.

**Per this task's explicit scope: PARS is not registered in the
foundational library / any selector-candidate list as part of this
recovery task** (`SELECTOR_ELIGIBLE = False` in
`baselines/pars/adapter/simulator_policy.py` is unchanged).

## §8 — Compute cost and timing

- **Original run:** 180m00s (killed by its own 10800s timeout), 3 of 8
  families completed.
- **Recovery runs:** 5 families, each ≈53-56 minutes of measured
  `wall_clock_s` (concurrent execution — see §3 caveat on timing
  fairness). `regression_anwg_selector` alone accounts for **99.1%–99.6%
  of each recovered family's total policy-decision time**
  (`total_policy_time_s` summed across all other 9 policies is a rounding
  error by comparison — e.g. `mixed_tight_deadlines`: 3109.0s for
  `regression_anwg_selector` vs. 14.3s combined for the other 9
  policies). This selector's per-decision inference cost, not the
  simulator or PARS itself, is the dominant cost driver of every
  comparative evaluation on this branch.
- **Total wall-clock across original + recovery:** ≈180 + ≈56 ≈ 236
  minutes (the 5 recovery families overlapped in wall-clock time with
  each other, not with the original run).

## §9 — Limitations

1. vLLM-LTR head-to-head comparison is limited to one non-discriminative
   workload (§6) — a genuine open gap, not a finding.
2. Recovery families ran concurrently rather than sequentially (§3) —
   does not affect correctness (independently verified) but limits
   cross-family timing-fairness claims.
3. No official upstream LICENSE file for `SPEAR-UIC/PARS` — disclosed in
   `baselines/pars/PROVENANCE.md`, unchanged by this evaluation.
4. Canonical-suite text substitution (nearest-token-count WildChat prompt
   matching, since synthetic requests carry no text) means PARS scores
   real text that was not originally paired with each synthetic request's
   metadata — quantified per-family, not hidden, but a real deviation
   from "PARS scoring its own native prompt."
5. Exact tmux session names used for the 5 recovery runs were not
   independently recoverable after the fact (§3) — the per-family output
   directories, logs, and manifests are the authoritative provenance
   record instead.
