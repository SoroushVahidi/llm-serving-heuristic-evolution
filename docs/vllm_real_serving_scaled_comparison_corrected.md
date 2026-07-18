# vLLM Real-Serving Scaled Comparison — Corrected Rerun

**Status: completed.** This is the corrected full-scale (3780-request) rerun
of `docs/vllm_real_serving_scaled_comparison.md`, executed after the selector
action-space preflight fix that document's "Fix status" section describes.
The earlier run's `PolicyNeverAdmitted` confound (selector-emitted labels
with no dispatchable adapter) is **not present here**: every per-policy row
in this run shows `n_never_admitted: 0`; every selector drop is a legitimate,
labeled `PolicyDeclinedAdmission` (intentional load-shed), not a harness bug.

This document supersedes `docs/vllm_real_serving_scaled_comparison.md` for
the *selector* arm specifically. The baseline arms in both runs are
consistent (same request plan generation, same policies, same harness).

## What this is, and what it is not

- **Real, external execution.** Every measured request in this run is a real
  HTTP `/v1/completions` call to a real vLLM server (see Model/server
  details below) — not a simulation, not a shadow evaluation.
- **Client-side admission control, not vLLM scheduling.** The policies
  compared here (`fifo`, `edf`, `least_laxity_first`,
  `estimated_service_time_first`, `shortest_output_first`, `vllm_direct`,
  and the `selector` meta-policy) decide only *when* a waiting request is
  admitted into a fixed client-side concurrency budget. Once admitted, the
  request is handed to vLLM exactly as any other HTTP client would.
- **vLLM's internal scheduler is a black box here.** This run observes
  nothing about vLLM's own continuous-batching/PagedAttention/KV-cache
  scheduling decisions. `vllm_direct` (strict arrival-order admission, no
  `ObservableState`/`Action` machinery) is the closest thing to "no extra
  admission logic," not a proxy for vLLM's internals.
- **This is NOT a faithful vLLM scheduler reproduction**, and does not claim
  to be one anywhere in the harness or this document.

## Experiment directory

`experiments/real_llm/vllm_scaled_fixed_selector_comparison_20260704T011204Z/`

- Generated: 2026-07-04T01:12:11 UTC (`reproducibility.md`)
- Git commit recorded in the run: `e74d29ce60d995318233a3aaa2bf97c2a7e3e86e`
  ("Fix vLLM selector action-space validation")
- **Git dirty at run time: `true`.** The run executed against an uncommitted
  working tree that included (among other things) the then-unstaged
  loss-case-reporting code later committed as `ad268f6` in this repository's
  history, plus unrelated cleanup later committed as `ea3fac9`/`b6afd6e`/
  `63b0358`/`15d7b04`. The harness code that *produced* this run's
  `loss_cases.*` files is the same code now on `main`/this branch at those
  commits — nothing about the loss-case logic changed between the dirty
  run and its eventual commit.
- Selector artifact: `results/corrected_selector_artifact_regression_anwg/regression_anwg_selector.joblib`
  (`regression_anwg`, per-policy `RandomForestRegressor`, argmax at predict
  time; feature-only, no oracle/hindsight fields — see the artifact's own
  `manifest.json` embedded in `run_config.json.selector_manifest`).

## Model / server configuration

Recoverable from this run's own files:

- Model: `Qwen/Qwen2.5-0.5B-Instruct` (`run_config.json`, `server_status.json`)
- Server URL: `http://127.0.0.1:8001` (local; `run_config.json.server_url`)
- `server_status.json` (a real `/v1/models` response at run time) confirms
  `max_model_len: 4096` for the served model.

**Not recoverable from this run's own manifest** (a real reproducibility
gap, since closed for *future* runs by commit `63b0358`, which this run
predates — its metadata is not retroactively reconstructed here):

- The vLLM server package version. Known only from *separate*, contemporaneous
  environment inspection (not from this run's own files): the same
  `.venvs/vllm_baseline_pilot` server observed live during this repository's
  audit work reports vLLM **0.24.0**. This is circumstantial, not proof this
  exact run used that exact build — treat as a plausible but unverified
  data point, not a recorded fact of the run.
- The server launch command/flags (`--gpu-memory-utilization`,
  `--max-model-len`, `--enforce-eager`, etc.). These are documented
  separately for the *healthcheck* server in
  `experiments/real_llm/vllm_healthcheck_20260703T171021Z/`, which is the
  same long-lived server process this run's `--server-url` points at, but
  this run's own manifest does not itself record those flags.

## Scenario / workload

One fixed request plan (seed `20260703`), reused identically across all 7
policies so outcome differences are attributable to admission order, not
workload variance:

- 3 arrival regimes: `steady_moderate`, `bursty_tight`, `overloaded_mixed_priority`
- × 3 prompt buckets: `short`, `medium`, `long`
- × 3 target output lengths: `64`, `128`, `256` tokens
- × 4 concurrency levels: `1`, `2`, `4`, `8`
- × 5 requests/cell = 108 cells → **540 requests per policy**
- **7 policies → 3,780 measured requests** (`requests.jsonl`), plus 2
  excluded warm-up requests (JIT-compilation absorption, `warmup_summary.md`)

## Policy portfolio compared

Only the harness-wired subset of the full 20-policy deployable registry —
`vllm_direct`, `fifo`, `edf`, `least_laxity_first`,
`estimated_service_time_first`, `shortest_output_first`, plus the `selector`
meta-policy (dispatching to whichever of 5 sub-policies it chose per window:
`edf`, `fifo`, `admission_control`, `scorpio_style_slo_guard`,
`least_laxity_first`, per `selector_action_space_preflight.json`'s
`labels_emitted_over_plan`). The remaining ~13 registered policies, and the
LLM-generated-heuristic shortlist, are not wired against a real server (see
`not_wired_policies` in `run_config.json`).

## Results

### Fixed-baseline performance (all completed 540/540)

| Policy | Arrival-norm. WG | 95% bootstrap CI |
|---|---|---|
| `shortest_output_first` | 0.2432 | [0.2100, 0.2781] |
| `fifo` | 0.2414 | [0.2081, 0.2753] |
| `vllm_direct` | 0.2397 | [0.2068, 0.2740] |
| `edf` | 0.2277 | [0.1967, 0.2624] |
| `estimated_service_time_first` | 0.2277 | [0.1964, 0.2614] |
| `least_laxity_first` | 0.2235 | [0.1910, 0.2594] |

### Selector performance

| Metric | Value |
|---|---|
| Completed | **483 / 540 (89.4%)** |
| Declined (legitimate load-shed, `PolicyDeclinedAdmission`) | 57 |
| Never-admitted (harness/adapter bug) | **0** |
| Arrival-normalized WG (point estimate) | **0.2241** — lowest of the 7 policies |
| Conditional WG (over the 483 completed only) | **0.2505** — highest of the 7 policies |

The selector's per-request quality *given completion* is the best of any
policy in this run; its arrival-normalized score is the worst, because it is
the only policy that declined any admissions at all under this specific
fixed plan (all 6 fixed baselines completed 100%).

*Note: `bootstrap_confidence_intervals.csv`'s own standalone point estimate
for the selector is 0.2123, not 0.2241 — the bootstrap and the direct
per-policy metric use separately-implemented WG calculations
(`compute_bootstrap_ci._wg_for_ids` vs. `compute_policy_metrics`) that agree
for every fixed baseline in this run but diverge slightly for the selector.
Not investigated further here; the qualitative conclusions below (selector
lowest of 7, all pairwise CIs include zero) hold under either number.*

### Statistical significance (2,000-resample paired bootstrap, selector − baseline)

| Comparison | Point estimate | 95% CI | Includes zero? |
|---|---|---|---|
| selector − fifo | −0.0287 | [−0.0746, +0.0185] | yes |
| selector − shortest_output_first | −0.0313 | [−0.0775, +0.0151] | yes |
| selector − vllm_direct | −0.0269 | [−0.0733, +0.0187] | yes |
| selector − edf | −0.0154 | [−0.0592, +0.0302] | yes |
| selector − estimated_service_time_first | −0.0154 | [−0.0625, +0.0317] | yes |
| selector − least_laxity_first | −0.0113 | [−0.0584, +0.0348] | yes |

**Every pairwise CI includes zero.** This single run does not provide
statistically significant evidence that the selector is worse (or better)
than any individual fixed baseline.

### Loss-case report (149 request×baseline pairs where the selector's real outcome lost)

| Baseline | Loss count |
|---|---|
| fifo | 31 |
| shortest_output_first | 31 |
| vllm_direct | 30 |
| estimated_service_time_first | 20 |
| edf | 19 |
| least_laxity_first | 18 |

Reason taxonomy (deterministic, heuristic, **not a causal analysis** — see
`_classify_loss_reason` and its own docstring caveat):

| Reason | Count | % |
|---|---|---|
| long-output underestimation | 146 | 98.0% |
| different ordering | 2 | 1.3% |
| high-priority request missed | 1 | 0.7% |

"Long-output underestimation" means the actual generated length for that
request exceeded 1.2× its planned `target_output_tokens` (the harness plans
against a target but allows generation up to `2×target`, at
`temperature=0.0`, so real per-request output length can and does vary).
Example: request 320 (bucket `long`, target 128) actually generated 256
tokens under the selector's chosen `admission_control` sub-policy, missing
its SLO deadline while several fixed baselines happened to complete the same
request without violating it (`loss_cases.csv`, `selector_vs_baselines_examples.md`).
This pattern dominating 98% of losses is consistent with output-length
variance/model behavior driving most losses in this run, rather than a
scheduling-order defect — but the taxonomy explicitly disclaims causal
attribution, and this should be read as a triage label, not a diagnosis.

### Decision divergence

648 (policy-pair, cell) comparisons; 176 produced a different SLO outcome
for the same request_id between the selector and a baseline
(`decision_divergence.csv`, `selector_vs_baselines_examples.md`).

## Safe claims

- This run is a real, external, HTTP-based comparison against a live vLLM
  server (OpenAI-compatible `/v1/completions`), not a simulation.
- The Phase 2C selector action-space confound from the earlier run
  (`docs/vllm_real_serving_scaled_comparison.md`) is fixed here: 0
  `PolicyNeverAdmitted` cases across all 7 policies.
- Under this specific fixed plan (one seed, one regime/bucket/length/
  concurrency grid), the selector's point-estimate arrival-normalized WG is
  the lowest of the 7 policies compared, driven by a lower completion
  fraction (89.4% vs. 100%) from legitimate load-shedding declines, not
  adapter/harness failures.
- None of the 6 pairwise (selector − baseline) differences are statistically
  significant at 95% in this single run.
- 98% of the 149 per-request losses are tagged (heuristically, non-causally)
  as driven by actual output length substantially exceeding the planned
  target for that request.
- This measures client-side admission-order effects layered on top of vLLM;
  it does not observe or reproduce vLLM's own internal batching/scheduling.

## Unsafe claims

- "The selector is worse than FIFO/EDF/etc." as a general, statistically
  significant conclusion — the bootstrap CIs all include zero, and this is
  one seed/one plan.
- "This proves the selector doesn't work" — its conditional WG (given
  completion) is the *best* of all 7 policies in this same run; the effect
  is a completion-fraction trade-off, not uniformly worse quality.
- Any claim about vLLM's own internal scheduler behavior — this harness
  never observes it.
- Any claim that this run used a specific vLLM server version — not
  recorded in this run's own metadata (see Model/server configuration).
- Any causal interpretation of the loss-reason taxonomy ("the scheduler
  caused these losses") — it is an explicitly heuristic triage label.
- Generalizing this single real-server run to the selector's broader
  validated performance (e.g. the 174-window fresh offline-simulator
  validation where `regression_anwg` beat always-SCORPIO, `docs/result_claims.md`)
  — different evaluation regime (offline simulator vs. real server), not
  directly comparable.

## Related documents

- [vllm_real_serving_scaled_comparison.md](vllm_real_serving_scaled_comparison.md) — the earlier, confounded run this supersedes
- [vllm_real_serving_external_baseline_pilot.md](vllm_real_serving_external_baseline_pilot.md) — the original small pilot
- [result_claims.md](result_claims.md) — offline-simulator selector validation claims (different evaluation regime)
