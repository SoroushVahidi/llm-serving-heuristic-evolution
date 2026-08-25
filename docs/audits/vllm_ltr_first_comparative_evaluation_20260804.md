# vLLM-LTR First Comparative Evaluation — 2026-08-04

**STATUS: COMPLETE AND INDEPENDENTLY VERIFIED.** The comparison sweep
below completed successfully (two full 3-seed runs — see
[`vllm_ltr_comparative_evaluation_recovery_20260804.md`](vllm_ltr_comparative_evaluation_recovery_20260804.md)
§5 for why there were two) and every reported number was independently
recomputed from raw per-request outcome rows
(`scripts/verify_vllm_ltr_comparison_results.py`) with zero mismatches
after fixing two bugs found in the verifier itself (§6 of the recovery
doc). This run followed an earlier attempt that never produced results at
all (a selector performance bug that made the comparison never finish);
the recovery doc has the full diagnosis and fix.

## What this evaluation is

The first comparative simulator sweep pitting the offline-scored vLLM-LTR
baseline (`baselines/vllm_ltr/`, official checkpoint, see
[`vllm_ltr_baseline_audit_20260804.md`](vllm_ltr_baseline_audit_20260804.md))
against this repo's existing fixed policies, selectors, and oracle — the
first time vLLM-LTR is evaluated on real prompt text rather than left as an
evaluation-ready scaffold.

## Dataset

WildChat-1M (`allenai/WildChat-1M`, official AI2 dataset, ODC-BY license,
ungated), pinned revision `7d6490e462285cf85d91eabea0f9a954fbddcd1f`. Full
selection rationale and provenance: [`external/datasets/wildchat.md`](../../external/datasets/wildchat.md).
300 single-turn, English, non-toxic, non-redacted conversations sampled
deterministically (seed=20260804) from a 100,000-row scan. Prompt token
lengths (facebook/opt-125m tokenizer): min=2, p50=62, p95=1691, max=11,180.
Real first-response text is used only for the same real-actual-output-tokens
role `workloads/sharegpt.py` already gives ShareGPT responses (ground-truth
decode length for the simulator, never exposed to deployable policies).

## Checkpoint

`LLM-ltr/OPT-Predictors`, regression variant
(`opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32`), pinned revision
`39df2b41ffe88d5ed967c6035d3838b5b5960379`. Hash-verified, architecture-verified,
bit-exact semantic-equivalence-verified: [`vllm_ltr_baseline_audit_20260804.md`](vllm_ltr_baseline_audit_20260804.md).
Scored offline (`scripts/score_vllm_ltr_eval_dataset.py`) — prompt text
only, cached and prompt-hash-integrity-checked before use.

## Policies compared

1. `fifo`
2. `edf`
3. `estimated_service_time_first`
4. `shortest_output_first`
5. `weighted_shortest_processing`
6. `scorpio_style_slo_guard` (best fixed baseline)
7. `rule_based` (`RuleBasedSelector`, current hard selector)
8. `regression_anwg` (`PerPolicyRegressionAnwgSelector`, best global composition)
9. `vllm_ltr_semantic_reference` (offline-scored; `SELECTOR_ELIGIBLE = False`,
   evaluation-only per the existing scope boundary)
10. `oracle_srtf` (non-deployable hindsight ceiling)

vLLM-LTR is evaluation-only for this run: not registered in any policy
registry, selector-candidate list, or CC4/CC5 training data
(`baselines/vllm_ltr/adapter/simulator_policy.py::SELECTOR_ELIGIBLE = False`).

## Why the first attempt didn't produce results

Summarized here; full diagnosis in the recovery doc. The run appeared to
hang indefinitely on the `regression_anwg_selector` policy. Root cause:
`PerPolicyRegressionAnwgSelector.predict_one()` cost ~56ms/call (sklearn
per-call overhead across 20 separate single-row `RandomForestRegressor.predict()`
calls), and this real long-response workload needs on the order of 17,000+
simulator steps per seed — making the run take tens of minutes per seed,
never finishing before being interrupted. A separate, pre-existing
correctness bug was also found and fixed (user-authorized) along the way:
the same selector silently received all-zero features when driven live by
the simulator (bare-key vs. `feat_`-prefixed key mismatch), so its
`regression_anwg` column was previously going to reflect a constant `edf`
dispatch rather than the selector's real trained behavior.

## Results

Mean ANWG (arrival_normalized_weighted_goodput) across 3 seeds, with the
paired bootstrap 95% CI (independently recomputed from raw per-request
rows — §6 of the recovery doc):

| Policy | ANWG (mean) | 95% CI | num_completed / num_dropped / num_slo_violated (seed 0, 1, 2) |
|---|---|---|---|
| fifo | 0.9957 | [0.9957, 0.9958] | 300/0/1, 300/0/1, 300/0/1 |
| edf | 0.9957 | [0.9957, 0.9958] | 300/0/1, 300/0/1, 300/0/1 |
| estimated_service_time_first | 0.9957 | [0.9957, 0.9958] | 300/0/1, 300/0/1, 300/0/1 |
| shortest_output_first | 0.9957 | [0.9957, 0.9958] | 300/0/1, 300/0/1, 300/0/1 |
| weighted_shortest_processing | 0.9957 | [0.9957, 0.9958] | 300/0/1, 300/0/1, 300/0/1 |
| rule_based_selector (hard selector) | 0.9957 | [0.9957, 0.9958] | 300/0/1, 300/0/1, 300/0/1 |
| **vllm_ltr_semantic_reference** | **0.9957** | **[0.9957, 0.9958]** | 300/0/1, 300/0/1, 300/0/1 |
| **oracle_srtf** (non-deployable ceiling) | **0.9957** | **[0.9957, 0.9958]** | 300/0/1, 300/0/1, 300/0/1 |
| scorpio_style_slo_guard (best fixed) | 0.9743 | [0.9701, 0.9788] | 295/5/0, 294/6/0, 294/6/1 |
| regression_anwg_selector (best global composition) | 0.9743 | [0.9621, 0.9844] | 297/3/1, 298/2/4, 300/0/8 |

**The central, load-bearing result: `oracle_srtf` — the non-deployable,
hindsight-optimal reordering — is statistically tied with plain `fifo`.**
Every ordering-only policy (`fifo`/`edf`/`estimated_service_time_first`/
`shortest_output_first`/`weighted_shortest_processing`/`rule_based_selector`/
`vllm_ltr_semantic_reference`/`oracle_srtf`) produced the **exact same**
per-seed outcome: all 300 requests completed, exactly 1 SLO violation
(seed 2's `scorpio_style_slo_guard` also has 1, but via a different
mechanism — see below). Since even perfect hindsight reordering cannot
avoid that one violation, it is not a scheduling-order problem at all in
this workload/config combination (Poisson arrival_rate=10, `max_active_sequences=8`,
`max_kv_tokens=131072`, real WildChat prompt/response lengths) — most
likely a single request whose own minimum service time already exceeds its
SLO deadline regardless of when it is admitted. **This means the tested
regime has zero reorderable headroom**, and no ordering policy — including
the theoretical optimum — can be distinguished from `fifo` in it.

**vLLM-LTR is exactly as good as `fifo` and the oracle in this regime, and
neither better nor worse.** It cannot be said to "win" or "lose" against
anything here, because nothing had room to win.

**The two policies that score measurably lower both do so via
admission-control drops or mispredicted dispatch, not ordering:**
- `scorpio_style_slo_guard` actively drops 5-6 of 300 requests per seed
  (its own admission guard rejecting them) while achieving 0 SLO
  violations among the ones it does admit — but since `fifo` already gets
  away with only 1 unavoidable violation, guarding against violations by
  dropping 5-6x that many requests is a net loss under ANWG (a drop and an
  SLO-missed completion cost the same — zero numerator credit, full
  denominator weight — so trading 1 violation for 5-6 drops is strictly
  worse).
- `regression_anwg_selector` (the persisted "best global composition"
  selector, live-dispatched per simulator step) drops fewer requests
  (2-3, one seed 0) but incurs *more* SLO violations than `fifo` (1, 4, 8
  across the three seeds) — see "Behavior" below for why: it is not
  behaving like a diverse ensemble here, it is overwhelmingly (~92% of
  live dispatch decisions) just re-choosing `scorpio_style_slo_guard`.

## Statistical analysis

- **Confidence intervals:** paired bootstrap (2000 replicates, resampling
  (seed, request_id) pairs) — see table above. The 8 tied policies' CIs
  are all `[0.9957, 0.9958]`, essentially a point estimate (very little
  seed-to-seed variance, expected since their raw outcomes are bit-identical
  across seeds). `scorpio_style_slo_guard` and `regression_anwg_selector`
  have visibly wider, non-overlapping-with-the-rest CIs.
- **Pairwise win/tie/loss** (per-seed comparison across all 3 seeds, 45
  total pairs per policy): the 8 tied policies each record 6 wins / 21
  ties / 0 losses (winning only against `scorpio_style_slo_guard` and
  `regression_anwg_selector`, tying every other tied policy every seed).
  `scorpio_style_slo_guard` and `regression_anwg_selector` each record 25
  losses / 1 tie / 1 win (they tie/beat each other once, in the one seed
  where `regression_anwg_selector`'s dispatch happened to avoid enough
  violations to match or edge out `scorpio_style_slo_guard`).
- **Unique wins:** zero for every policy — with 8 of 10 policies tied
  exactly every seed, no policy is ever the *sole* best performer in any
  seed.
- **Oracle-envelope contribution:** the gap between `oracle_srtf` and the
  worst policy (`scorpio_style_slo_guard`/`regression_anwg_selector`,
  0.9743) is 0.0214 ANWG. The best non-oracle policy (any of the 8 tied
  ones, all equal to the oracle) closes **100%** of that gap — trivially,
  since it *is* the oracle's value. This is a degenerate case of the
  metric (a real, contentful oracle-envelope measurement needs a regime
  where the oracle actually beats fifo) rather than a meaningful measure
  of composition quality here.

## Behavior

- **Ranking agreement:** Spearman correlation between vLLM-LTR's score
  order and each SJF-proxy policy's own real ranking rule (computed via
  `llmserveopt.policies.scoring.predicted_service_proxy` for EST, raw
  `predicted_output_tokens` for SOF — the ranking-agreement conflation bug
  described in the recovery doc, fixed, is what makes this comparison
  meaningful rather than two copies of the same number):
  - vs. `estimated_service_time_first`: 0.353 (seed 0), 0.402 (seed 1),
    0.383 (seed 2) — mean ≈0.38.
  - vs. `shortest_output_first`: 0.430 (seed 0), 0.475 (seed 1), 0.462
    (seed 2) — mean ≈0.46.
  - Both are moderate positive correlations, clearly not near 1.0 (which
    would mean vLLM-LTR just reproduces an existing hand-coded proxy) and
    clearly not near 0 (which would mean its ranking is unrelated to
    request size at all). vLLM-LTR agrees somewhat more with SOF
    (output-length-only) than with EST (prompt+output-length), across all
    3 seeds consistently.
- **Unique wins:** zero, as above — this regime never lets any policy show
  a unique advantage.
- **Strongest workload regime observed:** none differentiated — the
  workload as configured (300 requests, Poisson arrival_rate=10,
  8-sequence concurrency, ample 131,072-token KV budget) is essentially
  uncontended for every ordering-only policy; this is itself the main
  finding, not a null result to explain away.
- **Weakest workload regime observed:** the same regime is where
  admission-control-style policies (`scorpio_style_slo_guard`, and by
  inheritance `regression_anwg_selector`) actively *underperform* simple
  FIFO, because they pay a real cost (drops/mispredicted violations) to
  guard against a violation rate (1/300) that is already at its
  irreducible floor.
- **Behavioral diversity (dispatch histograms, live per-step selector
  dispatch across all 3 seeds):**
  - `rule_based_selector` (hard selector): 87,070 of ~93,000 decisions to
    `slo_slack_score`, with `weighted_shortest_processing` (5,491),
    `sarathi_style` (124), `edf` (244), `estimated_service_time_first`
    (69) as minor components. Entropy 0.373 bits (max possible 2.32 bits
    at this candidate-set size) — heavily concentrated, but its dominant
    choice (`slo_slack_score`) is evidently benign here (ties `fifo`).
  - `regression_anwg_selector`: **452,796 of ~491,714 decisions (≈92%) to
    `scorpio_style_slo_guard`**, with `edf` (37,608), `admission_control`
    (807), `weighted_shortest_processing` (435), `fifo` (25),
    `multi_bin_batching` (24), `least_laxity_first` (19) as minor. Entropy
    0.419 bits (max 2.807). This is the direct mechanistic explanation for
    why `regression_anwg_selector`'s ANWG degradation pattern mirrors
    `scorpio_style_slo_guard`'s: when driven live per simulator step (as
    opposed to the offline, windowed evaluation format it was trained and
    validated on — Phase 2B.16, `docs/result_claims.md`), it overwhelmingly
    re-derives the same choice as `scorpio_style_slo_guard`. This is a
    genuine regime/feature-distribution mismatch between the selector's
    training data and this live real-text workload, not a new bug (the
    feature-key bug that *would* have caused a different, worse failure
    mode — constant all-zero features collapsing to one fixed policy
    regardless of state — was found and fixed before this run; see the
    recovery doc §3).
- **Overlap with existing foundational heuristics:** vLLM-LTR's ranking is
  moderately-but-not-strongly correlated with the existing `EST`/`SOF`
  proxies (§ above) — it is not a redundant restatement of either. Whether
  that distinct ranking translates into measurable ANWG benefit is
  untested by this run (the regime gives no ordering policy room to show
  benefit at all).
- **Per-regime (SLO class) breakdown** (independently recomputed directly
  from `request_level_outcomes.csv`'s `class_id` field, pooled across all
  3 seeds — `data/processed/wildchat/`'s default augmentation assigns
  three classes: `interactive` (2.0s slack), `standard` (6.0s), `batch`
  (20.0s)): **all degradation is concentrated in the `interactive`
  (tightest-slack) class.** The 8 tied policies score ANWG=1.0000 on
  `standard` and `batch` but 0.9933 on `interactive`; `scorpio_style_slo_guard`
  and `regression_anwg_selector` score 1.0000 on `standard`/`batch` too,
  but drop to 0.9600 on `interactive` — a much larger relative penalty
  than the 8 tied policies see there. This directly confirms §"Results"'s
  reading: the one irreducible violation among the 8 tied policies (and
  the disproportionate extra cost `scorpio_style_slo_guard`/
  `regression_anwg_selector` pay) is concentrated exactly where slack is
  tightest relative to service time — consistent with an individually
  infeasible deadline (service time already exceeding the 2.0s
  `interactive` slack for at least one long-output request), not a
  cross-class scheduling-order effect that a different `standard`/`batch`
  admission order could have fixed.

## Strengths, weaknesses, limitations

**Strengths of this evaluation:** real prompt/response text (not
synthetic), the official checkpoint (hash- and architecture-verified,
bit-exact independent recomputation), identical request lists across every
policy per seed (fairness by construction), a genuine performance fix
proven bit-exact-equivalent before use, a real correctness bug found and
fixed along the way, and independent re-verification from raw per-request
rows that caught two bugs in its own verifier before being trusted.

**Weaknesses / limitations:**
- **The chosen workload configuration does not stress-test scheduling
  order.** This is the single most important limitation: with the oracle
  tying `fifo`, this run cannot distinguish "vLLM-LTR is genuinely useless
  as a scheduler" from "this workload never gives any scheduler a chance
  to matter." A follow-up with either a higher Poisson arrival rate, lower
  `max_active_sequences`, or tighter SLO deadlines is needed before any
  claim about vLLM-LTR's *scheduling value* (as opposed to its *ranking
  distinctness*, which this run does establish) can be made.
- Only 300 unique prompts (2 duplicate pairs, not deduplicated — see
  recovery doc §4) and 3 seeds; small-sample effects are visible in how
  tight the 8 tied policies' CIs are (near-zero variance because their raw
  outcomes are literally identical across seeds, not because 3 seeds is a
  large sample).
- `regression_anwg_selector`'s underperformance here is a known regime
  mismatch (its offline training/validation objective vs. this live,
  real-text, low-contention regime), not a re-litigation of Phase 2B.16's
  offline-eval result (`arrival_normalized_wg=0.9856` there remains
  correct and unaffected — see recovery doc §3).

## Final recommendation

**Classification: EVALUATION_ONLY.** vLLM-LTR (`vllm_ltr_semantic_reference`)
remains an offline-scored, evaluation-only baseline — **not** registered
as a selector candidate, policy-library member, or composition-training
input by this run (per the task's explicit scope boundary,
`SELECTOR_ELIGIBLE = False` unchanged). It **does** contribute a genuinely
distinct *ranking* (moderate, not near-1.0, Spearman agreement with the
existing EST/SOF proxies), so it is not simply redundant with them
behaviorally. Whether that distinct ranking is a genuinely distinct
*behavioral niche worth foundational-library entry* is **not established
by this run** — the tested regime gives no policy, including the
theoretical oracle, any headroom to demonstrate scheduling value at all.
**Foundational-library eligibility: not yet assessable; do not register
in this task**, per the explicit instruction not to make that decision
here regardless.

**Central conclusion:** this evaluation successfully recovered from a
performance bug that previously made it never finish, ran to completion
twice (confirming determinism), and was independently re-verified from raw
data with zero unexplained discrepancies. The clean, well-supported result
is a *regime-specific null finding*: in this workload, scheduling order
does not matter (the oracle itself proves this), so vLLM-LTR's real,
measurably-distinct ranking behavior has no opportunity to show measurable
benefit here — and the two policies that do score lower do so via
admission-control side effects, not ordering.

**Exact next baseline:** none — per this task's explicit instruction, do
not begin VTC, PARS, SSJF, ATHENA, or any other new baseline in this task.

**Exact next action (recommended, not started here):** re-run this same
comparison under a workload configuration engineered to actually stress
scheduling order — e.g. a higher Poisson `arrival_rate`, a lower
`max_active_sequences`, and/or tighter synthetic SLO deadlines — so that
`oracle_srtf` separates from `fifo` and the comparison can test whether
vLLM-LTR's distinct ranking translates into measurable ANWG benefit under
real contention. Until that regime is tested, vLLM-LTR's scheduling value
(as distinct from its ranking distinctness, already established) remains
untested, not negative.
