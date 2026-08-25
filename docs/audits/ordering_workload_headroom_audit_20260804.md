# Ordering Workload-Headroom Audit — 2026-08-04

**Scope note:** this is a design-only, parallel, isolated audit run
alongside (and strictly without disturbing) the vLLM-LTR comparative-
evaluation recovery task, which was independently finishing a separate
3-seed run in tmux session `vllm_ltr_comparison_recovery` while this audit
was produced. Nothing in this document edits, reruns, or depends on that
task's files, results directory, or in-progress audit docs. See the
isolation record at the end of this document.

## 1. Diagnosis: why the current WildChat comparison gives FIFO/oracle no room to differ

The real WildChat-1M evaluation workload
(`data/processed/wildchat/wildchat_eval_sharegpt_shaped.json`, seed 0,
300 requests, `ShareGPTConversionConfig(arrival_mode="poisson",
arrival_rate=10.0)`, default `AugmentationConfig` SLO classes
`interactive`/`standard`/`batch` with `slo_slack` 2.0/6.0/20.0s) was
analyzed with the new `scripts/check_ordering_workload_headroom.py`
(read-only reuse of the same ingestion path the live comparison uses,
against the same already-committed data file — no interaction with the
running comparison's process or its own copy of that data):

| Metric | Value |
|---|---|
| `n_requests` | 300 |
| `n_decision_steps_with_nonempty_queue` | 1,678 |
| **`queue_contention_fraction`** (≥2 requests queued at decision time) | **0.498** |
| **`fifo_srtf_decision_disagreement_fraction`** | **0.0054** |
| `service_time_cv` (actual_output_tokens, diagnosis-only) | 0.881 |
| `deadline_slack_cv` | 1.015 |
| `prompt_predicted_output_correlation` | 0.122 |
| `priority_distribution` | {1.0: 46, 2.0: 100, 3.0: 154} |
| `fifo` / `oracle_srtf` ANWG | 0.9958 / 0.9958 (bit-identical) |
| `fifo` / `oracle_srtf` completion_fraction | 1.0 / 1.0 |
| `fifo` / `oracle_srtf` slo_violation_rate | 0.00333 / 0.00333 |

**The central finding: real queue contention exists (49.8% of decisions
have ≥2 requests waiting), but admission order almost never changes WHICH
requests get admitted (0.54% disagreement between FIFO's actual admission
set and a same-snapshot SRTF-ordered admission pass, computed with the
identical feasibility constraints).** This is the precise mechanism, not
just an observation: `max_active_sequences=8` and `max_kv_tokens=131072`
against this workload's prompt-length profile (p50=62, p95=1691 tokens)
are generous enough that essentially every request that is ever queued
gets admitted within the very next decision or two regardless of which
order the queue is processed in — contention without genuine exclusion.

**Distinguishing the categories the task asks to separate:**
- **Not** "no decision opportunity" — decisions with ≥2 queued requests
  are common (49.8%).
- **Is** "decisions occur but have no metric impact" — this is the
  dominant mechanism: admission is contended in the sense of "multiple
  requests waiting," but not in the sense of "admitting one precludes
  admitting another soon after," so reordering essentially never changes
  a request's ultimate completion/SLO outcome.
- **Also present, secondarily:** "workload too underloaded" relative to
  its own deadline slack — `deadline_slack_cv=1.015` with slack values of
  2–20s vs. typical single-request service times on the order of
  0.1–0.5s means almost every request can absorb a large amount of
  queueing delay and still meet its deadline. The one violation each seed
  reliably showed in the real 3-seed run (see
  `docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`) is most
  consistent with an individually-infeasible outlier (a request whose own
  service time — e.g. the real WildChat output-length max of 4,591
  tokens ≈ 4.6s of decode time — already exceeds its assigned slack
  regardless of admission order), not a queueing-delay effect ordering
  could fix. This is a *different* "no headroom" mechanism than
  under-contention and is why even the **hindsight-optimal** `oracle_srtf`
  cannot do better than `fifo`: no reordering fixes an individually
  infeasible deadline.
- **Not** "metric too insensitive" or "simulator semantics suppress
  ordering effects" — §6 below (the validation config) proves the exact
  same metric, on the exact same simulator, shows an 11–50 percentage
  point FIFO/oracle ANWG gap once concurrency and deadline slack are
  recalibrated. The metric and simulator are both fully capable of
  showing ordering value; this specific workload's parameters just don't
  create the conditions for it.
- **Not** "service estimates insufficiently diverse" — `service_time_cv
  = 0.881` is substantial diversity; the issue is capacity/slack, not a
  flat service-time distribution.

## 2. Ordering-headroom metric and smoke gate

**Metrics** (all computed by `scripts/check_ordering_workload_headroom.py`,
which runs only `fifo` / `estimated_service_time_first` /
`shortest_output_first` / `oracle_srtf` — no learned selector, no vLLM-LTR
checkpoint inference, no GPU):

- `fifo_srtf_anwg_gap` = `oracle_srtf` ANWG − `fifo` ANWG
- `fifo_srtf_completion_gap`, `fifo_srtf_slo_violation_gap`
- `queue_contention_fraction` — fraction of nonempty-queue decision steps
  with ≥2 queued requests
- `fifo_srtf_decision_disagreement_fraction` — fraction of nonempty-queue
  decision steps where a same-snapshot SRTF-ordered admission pass (using
  each request's real `actual_output_tokens`, diagnostic-only, computed
  locally via the exact same round-robin/feasibility logic
  `FIFOPolicy`/`BasePolicy._feasible_on_gpu` use — never exposed to any
  policy) admits a **different set** of request_ids than FIFO's real
  decision
- `service_time_cv`, `deadline_slack_cv` (coefficient of variation)
- `prompt_predicted_output_correlation` (Pearson r)
- `all_four_policies_bit_identical_anwg` — direct detector of the exact
  degenerate pattern found in the WildChat control

**Smoke gate (PASS requires ALL of):**

| Check | Threshold | Justification |
|---|---|---|
| `disagreement_fraction_nonzero` | `> 0` | SRTF and FIFO must actually choose differently at least once — the WildChat control's own 0.54% (nonzero but tiny) shows this alone is too weak a bar by itself, hence the next two |
| `anwg_gap_meaningful` | `fifo_srtf_anwg_gap >= 0.01` | 1 percentage point — an order of magnitude above the WildChat control's exact 0.0 gap; small enough to not require an extreme regime, large enough to not be noise |
| `queue_contention_sufficient` | `queue_contention_fraction >= 0.05` | at least 5% of decisions see a real multi-request choice (WildChat control's own 49.8% shows contention alone is easy to get — this check mainly guards against a workload with almost no simultaneous arrivals at all) |
| `not_degenerate_tie` | `all_four_policies_bit_identical_anwg == False` | direct, cheap detector for the exact failure pattern this audit exists to avoid recreating |

Any workload failing this gate should not be promoted to an expensive
multi-policy (10-policy, multi-seed, vLLM-LTR-inference-included)
comparison sweep without first being retuned.

## 3. Candidate workload families

Eight families were designed (`configs/workload_headroom_candidates/`),
covering the task's required set. Each YAML documents its own `purpose`
and `scientific_purpose` field. First-draft parameters (arrival rate,
concurrency, and — critically — SLO slack) were calibrated using the same
multi-second slack style as the WildChat control's `AugmentationConfig`
defaults; **this reproduced the same null result across all 8** (see §5,
first pass). The families were then recalibrated using a lesson directly
derived from §1's diagnosis: **SLO slack must be set relative to a
TYPICAL single request's own service time** (here, `output_mean` decode
steps × `step_size=0.001s`), not to an arbitrary multi-second constant —
because loose slack relative to service time is precisely the mechanism
that suppressed headroom in the WildChat control. Concurrency
(`max_active_sequences`) was also tightened (2–3 instead of 8) and arrival
rate raised toward the resulting capacity ceiling.

1. **`1_staggered_heterogeneous`** — wide output-length spread, tight
   concurrency (2): tests head-of-line blocking from a long job occupying
   a scarce slot.
2. **`2_burst_independent_lengths`** — bursty arrivals, prompt/output
   lengths drawn independently: tests EST-vs-SOF ranking disagreement
   translating into an outcome difference.
3. **`3_mixed_tight_deadlines`** — three SLO classes spanning 0.04–0.5s
   slack (vs. a ~0.08s typical service time): tests SRTF-vs-deadline
   tradeoffs.
4. **`4_priority_vs_service_time_conflict`** — priority independent of
   request size, tight concurrency: tests priority-vs-size conflicts (WSP
   territory; not directly run by this cheap 4-policy checker).
5. **`5_overloaded_queue`** — arrival rate exceeding sustainable
   throughput: tests admission-control value under genuine overload.
6. **`6_prediction_noise_regime`** — `prediction_noise_rel=0.6`: isolates
   "value of a better predictor" from "value of ordering at all."
7. **`7_long_output_tail`** — Pareto output distribution: tests
   tail-heaviness as a distinct headroom source from arrival structure.
8. **`8_kv_budget_pressure`** — large prompts, `max_kv_tokens` tightened
   to 2048 with `max_active_sequences` left generous (8, matching the
   WildChat control): isolates KV-capacity feasibility from sequence-count
   feasibility as a distinct admission constraint.

## 4. Smoke checker

`scripts/check_ordering_workload_headroom.py` (new, isolated; does not
import, call, or modify `scripts/run_vllm_ltr_first_comparative_evaluation.py`,
`src/llmserveopt/selector/`, or `baselines/vllm_ltr/`). Supports
`--preset NAME` (any `llmserveopt.workloads.synthetic` preset),
`--config path.yaml` (a `workload_headroom_candidates`-shaped file),
`--wildchat-control` (the real committed WildChat data, reproduced via a
direct, read-only call to `convert_sharegpt_to_requests` — the same
underlying conversion function the live comparison uses, not a duplicate
reimplementation of it), `--dry-run`, `--json`, `--output PATH`, and a
deterministic `--seed`. Runs only `fifo` / `estimated_service_time_first`
/ `shortest_output_first` / `oracle_srtf` (`llmserveopt.policies.registry.make_policy`
and `llmserveopt.policies.oracle.build_oracle` — existing, unmodified) —
no learned selector, no vLLM-LTR checkpoint, no GPU.

## 5. Smoke results

**First pass** (multi-second slack, as in the WildChat control):

| Workload | ANWG gap | disagreement | contention | tied? | Gate |
|---|---|---|---|---|---|
| `wildchat_control` (n=300) | 0.0000 | 0.0054 | 0.498 | yes | **FAIL** |
| `1_staggered_heterogeneous` (n=84) | 0.0000 | 0.0000 | 0.000 | yes | FAIL |
| `2_burst_independent_lengths` (n=213) | 0.0000 | 0.0216 | 0.807 | yes | FAIL |
| `3_mixed_tight_deadlines` (n=104) | 0.0000 | 0.0000 | 0.000 | yes | FAIL |
| `4_priority_vs_service_time_conflict` (n=104) | 0.0000 | 0.0022 | 0.066 | yes | FAIL |
| `5_overloaded_queue` (n=185) | **−0.0108** | 0.0079 | 0.770 | no | FAIL |
| `6_prediction_noise_regime` (n=104) | 0.0000 | 0.0028 | 0.121 | yes | FAIL |
| `7_long_output_tail` (n=124) | 0.0000 | 0.0023 | 0.613 | yes | FAIL |
| `8_kv_budget_pressure` (n=104) | 0.0000 | 0.0000 | 0.000 | yes | FAIL |

All 8 first-draft families reproduced the WildChat control's null
pattern — confirming the diagnosis (loose slack was the dominant cause,
not something specific to real WildChat text) rather than refuting it.

**Recalibrated pass** (slack ≈ 1–2× typical single-request service time,
tighter concurrency, arrival rate raised toward capacity):

| Workload | ANWG gap | disagreement | contention | tied? | Gate | Runtime |
|---|---|---|---|---|---|---|
| `1_staggered_heterogeneous` (n=246) | **0.0244** | 0.0089 | 0.597 | no | **PASS** | 0.54s |
| `2_burst_independent_lengths` (n=288) | **0.5035** | 0.0227 | 0.942 | no | **PASS** | 0.90s |
| `3_mixed_tight_deadlines` (n=246) | **0.0239** | 0.0093 | 0.512 | no | **PASS** | 0.57s |
| `4_priority_vs_service_time_conflict` (n=246) | **0.0302** | 0.0089 | 0.623 | no | **PASS** | 0.62s |
| `5_overloaded_queue` (n=185, unchanged) | −0.0108 | 0.0079 | 0.770 | no | FAIL | 1.16s |
| `6_prediction_noise_regime` (n=246) | 0.0041 | 0.0090 | 0.477 | no | FAIL | 0.58s |
| `7_long_output_tail` (n=216) | **0.2500** | 0.0091 | 0.886 | no | **PASS** | 1.18s |
| `8_kv_budget_pressure` (n=216) | 0.0000 | 0.0000 | 0.992 | no | FAIL | 11.64s |
| `9_validation_aggressive` (diagnostic only, not a recommended family) | **0.1119** | 0.0123 | 0.615 | no | **PASS** | 0.67s |

**5 of 8 families pass after recalibration.** The 3 that still fail are
each independently informative, not just "needs more tuning":

- **`5_overloaded_queue` (FAIL, negative gap):** `fifo` (0.9730) actually
  *beats* `oracle_srtf` (0.9622) here. This is a real, expected scheduling-
  theory result, not a bug: `oracle_srtf` (`OracleShortestJobFirstPolicy`)
  is a hindsight-**shortest-job-first** ceiling — optimal for minimizing
  mean flow time, but **not** guaranteed optimal for this repo's
  priority-weighted, deadline-based ANWG objective. Under genuine overload
  with uniform priority, greedily admitting short jobs first can starve a
  request that arrived early and could have met its own deadline if
  served promptly, converting an avoidable violation into a real one —
  something FIFO's simple arrival-order admission doesn't do here.
  **Caveat for any future use of `oracle_srtf` as an "oracle": it is a
  standard, well-motivated SJF benchmark, not a literal ANWG-optimal
  ceiling.** `shortest_output_first` (which uses *predicted*, not actual,
  length) scored 0.9568 — between fifo and oracle, consistent with this
  read.
- **`6_prediction_noise_regime` (FAIL, gap=0.0041):** below the 0.01 gate
  threshold even after recalibration — high prediction noise
  (`prediction_noise_rel=0.6`) alone, without also tightening
  concurrency/slack further, did not reliably translate into an ANWG
  difference in this pass. This family needs one more tuning iteration
  (likely: tighter slack still, since 0.15s here may remain too loose
  once `output_sigma=0.8` spreads actual lengths further than the other
  families) before being used in the next full comparison — flagged
  honestly as unresolved rather than force-fit to a PASS.
- **`8_kv_budget_pressure` (FAIL, gap=0.0000, disagreement=0.0000,
  contention=0.992):** the most interesting failure. Queue contention is
  nearly total (99.2% of decisions have ≥2 requests waiting — the highest
  of any family, including the validated one), yet **FIFO and SRTF admit
  the exact identical set every single time.** This means large-prompt KV
  exhaustion here blocks admission in an order-INDEPENDENT way: with
  `max_kv_tokens=2048` and `prompt_mean=800`, at most 1–2 requests fit
  concurrently regardless of which one is tried first, so "which large
  request goes first" doesn't change who eventually gets admitted, only
  exactly when. This is genuine evidence that isolating a KV-only
  bottleneck (leaving `max_active_sequences` generous) is a qualitatively
  different — and here, harder-to-expose — headroom source than sequence-
  count scarcity; family 8 likely needs an intentionally heterogeneous
  prompt-size mix (some requests that fit KV budget alone, some that
  don't) rather than a uniformly large-prompt distribution, so admission
  order actually determines *which* combination of requests fits.

**`9_validation_aggressive`** (not one of the 8 recommended families, kept
as a diagnostic-only calibration reference) confirms the checker and gate
are not just always-failing: a deliberately aggressive configuration
(slack tightly calibrated to service time, concurrency=2, arrival rate
near the resulting throughput ceiling) shows an 11.2-percentage-point
FIFO/oracle ANWG gap, real disagreement, real contention, and no tie. This
is the calibration reference the 8 families' recalibration was modeled on.

## 6. Selected future suite

For the next vLLM-LTR comparison (not started in this task), recommend
running the **5 passing families** — `1_staggered_heterogeneous`,
`2_burst_independent_lengths`, `3_mixed_tight_deadlines`,
`4_priority_vs_service_time_conflict`, `7_long_output_tail` — as the core
discriminative suite, since together they cover:

- **Learned-ranking value** (`2`, `6` once fixed): independent prompt/
  output lengths and prediction noise are exactly where a learned ranker
  should differentiate from a naive proxy.
- **Service-time ranking** (`1`, `7`): head-of-line blocking and tail
  heaviness both directly test SRTF-style value.
- **Deadline ranking** (`3`): tight/medium/loose slack heterogeneity.
- **Priority handling** (`4`): priority independent of size.
- **Overload behavior** (`5`, once its interpretation — see §5 — is
  incorporated rather than treated as a simple pass/fail): specifically
  useful as a regime where admission-control-style guards (which measurably
  *hurt* ANWG in the WildChat control — see the recovery doc's finding
  that `scorpio_style_slo_guard` drops 5–6/300 requests for no benefit
  there) might plausibly help instead.
- **Robustness to prediction error** (`6`, needs one more tuning pass
  before use).

**The current WildChat workload should be retained as the negative/control
workload**, not discarded: it is real prompt/response text (the other 7
families are synthetic), and its own well-characterized null result (zero
reorderable headroom, `oracle_srtf` ties `fifo` exactly) is itself a
useful, reproducible baseline — any future workload-design regression can
be checked against "does this still look like the WildChat control" as a
sanity floor. It also remains the only family in this comparison that
verifies vLLM-LTR's real-text ranking behavior (Spearman agreement with
EST/SOF ≈0.35–0.48) is genuine and not an artifact of synthetic length
distributions.

**Expected policy distinctions in this suite:** `1`/`7` should separate
`fifo` from `{est, sof, oracle_srtf}`; `2`/`6` should separate `est` from
`sof` (not just tie them, unlike the WildChat control where their ANWG was
identical); `3` should motivate adding `edf`/`least_laxity_first` to a
follow-up run (not in this cheap checker's 4-policy set); `4` should
motivate adding `weighted_shortest_processing`; `5` should be the regime
where `scorpio_style_slo_guard`-style admission control is finally tested
under conditions where it could plausibly help, unlike the WildChat
control where it only hurt.

## 7. Limitations

- Single seed (seed=0) per family in this smoke pass — sufficient to
  establish gate PASS/FAIL directionally, not to report a publication-
  grade multi-seed comparison (that is exactly what the recommended next
  full sweep, §8, is for).
- `oracle_srtf` is not a literal ANWG-optimal ceiling (§5's `5_overloaded_queue`
  finding) — any future full comparison should treat it as a standard,
  well-motivated SJF benchmark, not blindly as "the best possible outcome
  under this metric."
- `fifo_srtf_decision_disagreement_fraction` is computed against a
  same-snapshot SRTF re-ordering using FIFO's own trajectory only (not a
  true lockstep comparison against SRTF's own diverging trajectory, which
  would require either running both policies through identical states or
  a more complex counterfactual simulation) — a deliberate, documented
  simplification appropriate for a cheap pre-screening gate, not a
  substitute for the full comparison's real multi-policy run.
- `6_prediction_noise_regime` and `8_kv_budget_pressure` remain unresolved
  (documented as FAIL with a specific hypothesis for the next tuning
  iteration each), not silently dropped or force-fit.
- This audit used only the cheap 4-policy checker; it makes no claim about
  `scorpio_style_slo_guard`, `rule_based_selector`, `regression_anwg_selector`,
  or `vllm_ltr_semantic_reference`'s behavior on any of these candidate
  workloads — that is explicitly out of scope for a smoke-only, GPU-free,
  no-learned-inference gate.

## 8. Exact next experiment (recommended, not started here)

Re-run the vLLM-LTR comparison (per
`docs/audits/vllm_ltr_first_comparative_evaluation_20260804.md`'s own
"Exact next action") using the 5 passing families above (plus a fixed
`6_prediction_noise_regime` once retuned) as new workloads, alongside the
existing WildChat control, across the full 10-policy set and multiple
seeds — this is the concrete way to test whether vLLM-LTR's already-
established distinct ranking (moderate, not near-1.0, Spearman agreement
with EST/SOF) translates into measurable ANWG benefit once the workload
actually has reorderable headroom, which the WildChat-only comparison
could not test.

---

## Isolation record

- **Running comparison disturbed:** No. Verified via `tmux capture-pane`
  checks throughout this audit (before writing any file, after each major
  step, and immediately before writing this document) — the
  `vllm_ltr_comparison_recovery` session's own process was still
  progressing through its 3-seed run each time, untouched.
- **Files created (all new, isolated paths):**
  `scripts/check_ordering_workload_headroom.py`,
  `configs/workload_headroom_candidates/{1..8}_*.yaml`,
  `docs/audits/ordering_workload_headroom_audit_20260804.md` (this file).
  A 9th diagnostic-only config
  (`9_validation_aggressive.yaml`) was kept under `/tmp/` (scratch), not
  committed to `configs/`, since it exists only to calibrate the other 8,
  not as a recommended family.
- **Shared files modified:** none.
  `scripts/run_vllm_ltr_first_comparative_evaluation.py`, selector code,
  vLLM-LTR adapter code, simulator code, and the other task's audit docs
  were not opened for editing by this task (read-only imports of
  `llmserveopt.workloads.sharegpt.convert_sharegpt_to_requests` and the
  registry/oracle/simulator modules only — all pre-existing, unmodified).
- **Git commits:** none (per this task's explicit instruction).
- **Experiments run:** 1 exploratory (`make_small_debug_trace`) + 1
  control (`wildchat_control`) + 8 candidates × 2 passes + 1 validation
  config × 2 passes = 22 `check_ordering_workload_headroom.py` invocations,
  each running exactly 4 cheap policies = **80 total simulator
  executions**, well under the "fewer than a few hundred" budget. No GPU
  used. Total wall-clock for all simulator executions: ≈25s (the sum of
  each run's reported `runtime_s`); total wall-clock for this entire
  audit task (including config authoring and this document): well under
  the main task's own ~38-minute run, and did not extend it.
