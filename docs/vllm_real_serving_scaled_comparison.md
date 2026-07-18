# vLLM Real-Serving Scaled External-Baseline + Selector Comparison

> **Status: superseded (selector arm only).** The `PolicyNeverAdmitted`
> confound described below has been fixed and re-run; see
> [`docs/vllm_real_serving_scaled_comparison_corrected.md`](vllm_real_serving_scaled_comparison_corrected.md)
> for the corrected selector-arm results. The baseline arms here remain
> valid and are retained for historical context.

**Status: completed but caveated.** This is a real, multi-regime, higher-volume
comparison of this project's admission/scheduling policies **and its corrected-
objective selector** run as external admission controllers on top of a real,
running vLLM server. The baseline arms are usable. **The selector arm is
confounded** and must be read with the `PolicyNeverAdmitted` caveat below — do
not draw an "our selector is intrinsically worse than FIFO" conclusion from it.

Supersedes the 24-req/policy smoke run in
`docs/vllm_real_serving_external_baseline_pilot.md` in scale and coverage, and
is the first real-vLLM run to include the wired corrected-objective selector.

## Experiment directory

`experiments/real_llm/vllm_scaled_comparison_20260703T203640Z/`

Created 2026-07-03 16:36 UTC-4, completed 2026-07-03 17:30 UTC-4
(`run_config.json.run_status == "completed"`).

## Model / server details

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Server: real vLLM at `http://127.0.0.1:8001` (OpenAI-compatible HTTP API),
  launched `--gpu-memory-utilization 0.5 --max-model-len 4096 --enforce-eager`
  in the isolated `/home/soroush/.venvs/vllm_baseline_pilot` venv
  (see `experiments/real_llm/vllm_healthcheck_20260703T171021Z/`).
- No hosted API (Cohere / Gemini / OpenAI / Azure / Fireworks / CloudRift) was
  called anywhere in this run. The only network target was the local vLLM
  server.
- This measures **external admission control layered on top of vLLM** — a
  client-side controller choosing which waiting request gets the next of
  `concurrency_level` in-flight slots. It does **not** observe or replace
  vLLM's own internal batching/KV scheduler (see the pilot doc's "what this
  does not test").

## Request counts

- Grid: prompt buckets `short, medium, long` × target output tokens
  `64, 128, 256` × concurrency `1, 2, 4, 8` × `5` requests/cell × `3` arrival
  regimes (`steady_moderate, bursty_tight, overloaded_mixed_priority`).
- **540 requests per policy** (180 per regime), **7 policies → 3780 total
  measured requests** (`requests.jsonl`, `summary.json.total_records == 3780`),
  under the `--max-total-requests 4000` hard cap.
- One fixed plan (seed `20260703`) reused identically for every policy.
- Plus 2 warm-up requests (excluded from all metrics).

## Policies compared

`vllm_direct`, `fifo`, `edf`, `least_laxity_first`,
`estimated_service_time_first`, `shortest_output_first`, `selector`.

The selector is the corrected-objective artifact
`results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib`
(`PerPolicyRegressionAnwgSelector`, objective `arrival_normalized_wg`),
loaded with `--require-our-method` and manifest-verified before the run
(objective `arrival_normalized_wg`, class `PerPolicyRegressionAnwgSelector`).

## Objective

`arrival_normalized_weighted_goodput = completion_fraction × conditional_WG`,
where `conditional_WG = Σ(priority · slo_met) / Σ(priority)` over **completed**
requests and `completion_fraction = n_completed / n_total`. Failed / timed-out /
dropped requests correctly count as **zero** via the completion-fraction
multiplier (the Phase 2B.14-corrected convention).

## Overall weighted goodput (all regimes pooled)

| Rank | Policy | n_completed / n_total | Arrival-norm. WG | Conditional WG | SLO viol. (completed) | Mean TTFT (s) | Mean server latency (s) | Req/s |
|---|---|---|---|---|---|---|---|---|
| 1 | fifo | 540/540 | **0.2440** | 0.2440 | 0.6148 | 0.0190 | 1.467 | 1.157 |
| 2 | shortest_output_first | 540/540 | 0.2432 | 0.2432 | 0.6167 | 0.0193 | 1.486 | 1.142 |
| 3 | vllm_direct | 540/540 | 0.2414 | 0.2414 | 0.6185 | 0.0198 | 1.496 | 1.139 |
| 4 | edf | 540/540 | 0.2303 | 0.2303 | 0.6481 | 0.0187 | 1.450 | 1.169 |
| 5 | least_laxity_first | 540/540 | 0.2277 | 0.2277 | 0.6500 | 0.0191 | 1.468 | 1.157 |
| 6 | **selector** | **481/540** | **0.2274** | **0.2553** | 0.6154 | 0.0202 | 1.423 | 1.149 |
| 7 | estimated_service_time_first | 540/540 | 0.2235 | 0.2235 | 0.6556 | 0.0194 | 1.485 | 1.148 |

Note the selector's **conditional** WG (0.2553, quality among the requests it
actually served) is the **highest** of any policy — but its
`completion_fraction` is only 481/540 = 0.891 because of the 59 dropped
requests, which drags its arrival-normalized WG down to 6th. This is the
central confound (see "PolicyNeverAdmitted" below).

## Weighted goodput per policy and regime (arrival-norm. WG)

| Policy | steady_moderate | bursty_tight | overloaded_mixed_priority |
|---|---|---|---|
| fifo | 0.4512 | 0.1962 | 0.1368 |
| shortest_output_first | 0.4512 | 0.1962 | 0.1346 |
| vllm_direct | 0.4512 | 0.1935 | 0.1325 |
| edf | 0.4787 | 0.1774 | 0.0983 |
| least_laxity_first | 0.4665 | 0.1720 | 0.1047 |
| estimated_service_time_first | 0.4665 | 0.1640 | 0.1004 |
| **selector** | 0.4535 | 0.1893 | **0.0843** |

Selector completion per regime: steady 175/180, bursty 164/180, overloaded
142/180. All 59 drops are concentrated in the two tight regimes
(bursty_tight + overloaded_mixed_priority), where the selector routed to
`scorpio_style_slo_guard` and that policy shed load.

## Selector completion / drop counts

- Completed: **481 / 540** (89.1%).
- Dropped: **59** — all with `status="dropped"`, `error_type="PolicyNeverAdmitted"`.
- No `error`/`timeout` (network/dispatch) failures: `errors.jsonl` is 100%
  `PolicyNeverAdmitted`.
- Every one of the 59 drops occurred while the selector's chosen sub-policy was
  `scorpio_style_slo_guard` (confirmed from `errors.jsonl`:
  `(PolicyNeverAdmitted, scorpio_style_slo_guard) → 59`).

Selector sub-policy choice distribution over its 540 decisions
(`selector_chosen_policy`): `edf` 135, `admission_control` 124,
`scorpio_style_slo_guard` 121, `fifo` 112, `weighted_shortest_processing` 27,
`least_laxity_first` 20, `multi_bin_batching` 1. Of the load-capable sub-policies
the selector chose, **only** `scorpio_style_slo_guard` produced drops;
`admission_control` (default `laxity_threshold=inf`, no filtering) and
`weighted_shortest_processing` (pure reordering) produced **zero** drops.

## Explanation of `PolicyNeverAdmitted`

`PolicyNeverAdmitted` is **not** a missing-adapter or network error. All seven
policies (and every sub-policy the selector can emit) are constructible via
`make_policy()` and use only online-observable request fields.

What happened: `scorpio_style_slo_guard` is an SLO-guard admission controller.
Its laxity filter (`slo_deadline − now − est_service`) **deliberately declines**
requests it judges cannot meet their deadline. In the two tight regimes,
requests queued behind others reach negative laxity (their real wall-clock
deadline has effectively passed) before the guard would admit them. The external
harness's loop, on finding *nothing in flight, nothing pending, and a policy
that admits nothing*, records the remaining waiting requests as
`status="dropped"` / `PolicyNeverAdmitted` (rather than silently losing them),
counting each as zero-credit SLO violations.

This is a genuine **load-shed by a load-shedding policy**, correctly executed —
but the label `PolicyNeverAdmitted` implied a harness failure, and the *effect*
on the comparison is a confound: the always-admitting baselines (FIFO, EDF, …)
run those same doomed requests to completion (counting them as
completed-but-SLO-violated), while the selector-via-SCORPIO drops them
(counting them as not-completed). Under arrival-normalized WG, a truly-doomed
request is neutral either way, but SCORPIO also throttled/reordered some
*meetable* requests into misses, and the `completion_fraction` penalty makes the
selector's aggregate number look worse than a like-for-like admission comparison
would. See `docs/vllm_real_serving_external_baseline_pilot.md` and the harness
fix (preflight + explicit load-shed labeling) for the remediation.

## Decision-divergence summary

From `decision_divergence.csv` / `selector_vs_baselines_examples.md`
(real, independently-executed admission orders over the identical plan, not
shadow evaluation):

- 648 cells compared (selector vs. each of the 6 baselines).
- **186 requests** where the selector's real SLO outcome differed from a
  baseline's real SLO outcome for the same `request_id`.
- Divergence is genuine: the selector reorders and (via SCORPIO) sheds, so its
  admission order and per-request SLO outcomes differ measurably from every
  fixed baseline. Both directions occur (selector meets where a baseline
  misses, and vice-versa).

## Bootstrap confidence intervals

From `bootstrap_confidence_intervals.csv` (paired bootstrap over `request_id`,
n_boot=2000). Every `selector − baseline` paired-difference CI **includes zero**:

| Contrast | Point | 95% CI |
|---|---|---|
| selector − edf | −0.0147 | [−0.0590, +0.0294] |
| selector − estimated_service_time_first | −0.0078 | [−0.0551, +0.0388] |
| selector − fifo | −0.0279 | [−0.0739, +0.0198] |
| selector − least_laxity_first | −0.0123 | [−0.0605, +0.0341] |
| selector − shortest_output_first | −0.0279 | [−0.0758, +0.0193] |
| selector − vllm_direct | −0.0253 | [−0.0717, +0.0211] |

No pairwise difference between the selector and any baseline is statistically
distinguishable from zero at the 95% level.

## Paper-safe claims

- "In a real multi-regime vLLM external-admission comparison (3780 requests,
  7 policies, 3 arrival regimes, `Qwen2.5-0.5B-Instruct`), FIFO,
  shortest-output-first, and vLLM-direct achieved the highest
  arrival-normalized weighted goodput as executed (0.2440 / 0.2432 / 0.2414),
  ahead of the selector as executed (0.2274)."
- "The selector's result in this run is confounded by 59/540 (10.9%) dropped
  requests, all caused by the selector routing to `scorpio_style_slo_guard`,
  which load-sheds requests it judges unmeetable; the selector's conditional
  quality among served requests (0.2553) was the highest of any policy."
- "No selector-vs-baseline arrival-normalized WG difference is statistically
  distinguishable from zero (all paired bootstrap 95% CIs include zero)."
- "All policies used only online-observable request fields; no policy accessed
  realized output length; no hosted API was called."
- "This measures external admission control on top of vLLM, not vLLM's own
  internal scheduler."

## Unsafe claims (do NOT make)

- ❌ "The selector is intrinsically worse than FIFO / SOF / vLLM-direct." Its
  action space includes a load-shedding sub-policy whose declines the harness
  recorded as drops; the comparison is not like-for-like until the selector's
  full action space is executed on equal admission terms (fix implemented) or
  the selector is retrained/restricted to a non-shedding executable action set.
- ❌ "The selector meaningfully beats any baseline." All CIs include zero.
- ❌ "This shows how our method performs against vLLM's scheduler." It does not
  observe vLLM's internal scheduler.
- ❌ "The 59 drops are network/harness failures." They are intentional
  load-shedding by `scorpio_style_slo_guard`, correctly executed.

## Conclusion

- **Baseline results are usable.** All six fixed policies completed 540/540 with
  zero network/timeout failures; their arrival-normalized WG ranking is a valid
  real-vLLM external-admission result.
- **The selector result is caveated / flawed.** 59/540 requests were dropped
  because the selector routed to a load-shedding admission policy
  (`scorpio_style_slo_guard`) whose declines the harness recorded as
  `PolicyNeverAdmitted`. The selector's aggregate number therefore **understates
  a like-for-like comparison** and must not be read as "the selector is worse."
- **Hosted (Cohere/Gemini) replication should wait** until the selector
  action-space / harness starvation issue is fixed and validated. It was not
  started in this run.

## Fix status (see harness changes)

`scripts/run_vllm_external_baseline_comparison.py` now:
1. runs a **selector action-space preflight** that enumerates every label the
   selector can emit over the request plan and aborts **before any live
   request** if any label is not dispatchable (and `--require-our-method` fails
   fast on an unsupported label);
2. labels intentional load-shed declines explicitly
   (`error_type="PolicyDeclinedAdmission"`) and reserves `PolicyNeverAdmitted`
   for the (now preflight-prevented) missing-adapter case;
3. writes `selector_action_space_preflight.json` into each run directory.

A **tiny 24-req/policy validation rerun** (loose-deadline pilot config) is the
gate before any full scaled rerun; the full 3780-request scaled rerun is **not**
to be re-executed until explicitly instructed.
