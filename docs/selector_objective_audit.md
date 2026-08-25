# Selector Objective Audit

This audit freezes the historical metric semantics, adds an explicit
arrival-normalized metric, and explains why Selector Dataset v2 should not be
generated at scale until the manuscript-primary objective is updated.

## Current Implementation

`src/llmserveopt/core/metrics.py` historically computes:

```text
weighted_goodput =
  sum(weight_i * 1[completed_i and completion_time_i <= deadline_i])
  ----------------------------------------------------------------
  sum(weight_i for completed requests only)
```

where `weight_i = request.priority` when positive and `1.0` otherwise.

Exact semantics:

- Numerator: priority weight for completed requests that meet SLO.
- Denominator: priority weight for completed requests only.
- Rejected/dropped requests: excluded from numerator and denominator.
- Unfinished requests: excluded from numerator and denominator.
- Completed SLO misses: included in denominator, zero numerator.
- No completed requests: `RunMetrics.weighted_goodput = NaN`; Dataset v2
  preserves a backward-compatible zero for zero-completion windows so those
  windows are not silently omitted from objective means.

This field is preserved for historical reproducibility. Its scientifically
accurate name is:

```text
conditional_weighted_slo_attainment
```

or, less formally:

```text
completed-request weighted SLO success rate
```

`priority_weighted_slo_goodput` remains a historical alias of
`weighted_goodput`; it should not be used as the manuscript-primary system
goodput name without the qualifier above.

## Corrected Metric

The new system-level metric is:

```text
arrival_normalized_weighted_goodput =
  sum(weight_i * 1[completed_i and completion_time_i <= deadline_i])
  ----------------------------------------------------------------
  sum(weight_i for all arriving requests)
```

Required semantics:

- Rejected/dropped requests contribute denominator weight and zero numerator.
- Unfinished requests contribute denominator weight and zero numerator.
- SLO-missed completions contribute denominator weight and zero numerator.
- Priority weights use the same positive-priority-or-1.0 fallback as the
  historical metric.
- Empty traces return `NaN`; non-empty traces with no successes return `0.0`.

The code also records:

```text
weighted_completion_fraction =
  sum(weight_i for completed requests)
  ------------------------------------
  sum(weight_i for all arriving requests)
```

For nonzero denominators:

```text
conditional_weighted_slo_attainment * weighted_completion_fraction
  == arrival_normalized_weighted_goodput
```

This algebra is exact because both factors share the completed-request weight
denominator.

## Hand-Check Example

All priorities are 1.0.

| Policy | Arrivals | Completed | SLO-met completions | Historical `weighted_goodput` | Completion fraction | Arrival-normalized WG | Rejection fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 100 | 100 | 80 | 0.80 | 1.00 | 0.80 | 0.00 |
| B | 100 | 20 | 20 | 1.00 | 0.20 | 0.20 | 0.80 |

The historical metric ranks B above A even though B leaves 80% of the offered
workload unserved. The corrected metric ranks A above B.

## Simulator Rejection Semantics

In the simulator, normal monolithic policies do not emit a permanent
"reject-with-response" action. A policy can skip requests in the waiting queue;
those requests may be admitted later. At termination, requests still waiting,
mid-transfer, or mid-relocation after the drain window are counted as dropped:

```text
dropped = waiting + migrating + relocating
```

Dropped requests are permanently unserved in the resulting `RunMetrics`. They
are not retried, completed, or transferred elsewhere after metric computation.
Invalid admissions are skipped with warnings; they are not successful service.

Therefore, a system-level utility for "handling the workload offered to it"
must keep all arrivals in the denominator or otherwise explicitly constrain
completion/rejection.

## Literature Check

Primary-source context supports the correction:

- DistServe defines per-GPU goodput as the maximum request rate that can be
  served while adhering to an SLO attainment goal; this is a sustainable
  arrival-rate notion, not success conditional on completed requests:
  https://arxiv.org/abs/2401.09670 and
  https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- SCORPIO defines a sequence of arriving user requests, a subset that is
  SLO-compliant, and reports system goodput/SLO adherence for heterogeneous
  SLOs; it also compares cumulative SLO-met requests on real traces:
  https://arxiv.org/abs/2505.23022
- "Revisiting SLO and Goodput Metrics in LLM Serving" explicitly critiques
  metrics where dropping a request that will miss SLO can improve goodput:
  https://arxiv.org/abs/2410.14257
- Mooncake treats overload rejection as an explicit system mechanism: it
  develops prediction-based early rejection to reduce wasted work under
  overload, rather than treating rejected requests as free successes:
  https://arxiv.org/abs/2407.00079

## Dataset v2 Plumbing

Future Dataset v2 scenario-policy rows preserve:

- `metric_weighted_goodput`: historical conditional weighted SLO attainment.
- `metric_arrival_normalized_weighted_goodput`: corrected all-arrivals
  denominator.
- `metric_weighted_completion_fraction`: weighted completion denominator
  factor.
- `metric_completion_fraction`: unweighted completed/arrivals fraction.
- `metric_rejection_rate` and `metric_rejection_fraction`: dropped/arrivals.
- `metric_slo_attainment`: unweighted SLO success among completed requests.
- `metric_request_throughput` and `metric_token_throughput`.

Historical CSV/JSON semantics are not changed.

## Existing Pilot Re-Scoring

No simulations were rerun. Existing Dataset v2 pilot CSVs were rescored with:

```bash
python scripts/audit_selector_objectives.py \
  --input \
    results/selector_dataset_v2/redesigned_pilot/selector_dataset_v2_redesigned_pilot.csv \
    results/selector_dataset_v2/targeted_discovery_pilot/selector_dataset_v2_redesigned_pilot.csv \
    results/selector_dataset_v2/targeted_discovery_pilot_strict_small/selector_dataset_v2_redesigned_pilot.csv \
  --output results/selector_dataset_v2/objective_audit/objective_rescore_report.json
```

Important limitation: these are existing generated rows. For older rows, the
stored `arrival_normalized_weighted_goodput` was computed as
`completion_fraction * weighted_goodput`, which is exact when all weights are
equal or completed-request weights match arrival weights. New simulator runs
now compute the corrected weighted denominator directly.

### 307-Window Targeted Diagnostic

| Objective | Best fixed | Oracle headroom | Strong fraction | Near-tie fraction | All-equivalent fraction | SCORPIO wins | vLLM wins | Sarathi wins | Mean winner CF | Mean winner rejection |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical conditional WG | `scorpio_style_slo_guard` | 0.000814 | 27.36% | 28.66% | 43.97% | 86 | 157 | 20 | 0.7568 | 0.2432 |
| Arrival-normalized WG | `edf` | 0.006577 | 4.89% | 73.62% | 20.85% | 14 | 129 | 52 | 0.9634 | 0.0366 |
| Completion-adjusted WG | `edf` | 0.006577 | 4.89% | 73.62% | 20.85% | 14 | 129 | 52 | 0.9634 | 0.0366 |
| SLO-success throughput | `admission_control` | 0.349274 | 13.68% | 75.57% | 10.75% | 2 | 99 | 14 | 0.9908 | 0.0092 |
| Constrained ANWG, CF>=0.8, rejection<=0.2 | `scorpio_style_slo_guard` is not reliably meaningful as a fixed-policy statistic unless eligible in every window; per-window winners are diverse | 0.0-0.01 range across grids | low | high | high | 1-4 | 142 | 52 | 0.9958 | 0.0042 |

Under the historical metric, winner completion is far lower and winner
rejection is far higher. Under arrival-normalized and constrained objectives,
faithful baselines and deadline/throughput policies reappear.

### Redesigned 51-Window Pilot

Historical conditional WG:

- best fixed: `scorpio_style_slo_guard`
- oracle headroom: 0.051779
- wins: SCORPIO 27, vLLM 20, Sarathi 2, EDF 2
- mean winner completion: 0.4134
- mean winner rejection: 0.5866

Arrival-normalized / completion-adjusted WG:

- best fixed: `orca_style`
- oracle headroom: 0.025841
- wins: Sarathi 12, vLLM 9, Orca 9, WSP 6, SOF 4, EDF 3,
  multi-bin 3, FIFO 2, admission-control 2, SCORPIO 1
- mean winner completion: 0.9992
- mean winner rejection: 0.0008

### Strict Prefill Diagnostic

The strict 9-window prefill-heavy diagnostic shows why final selector data
should wait for objective correction:

- historical strong winners: SCORPIO 2/2
- arrival-normalized winners: Sarathi 8, vLLM 1
- historical winner mean completion: 0.4074
- arrival-normalized winner mean completion: 1.0

## Selective-Service Advantage

`SELECTIVE_SERVICE_ADVANTAGE` flags pairwise cases where policy A beats policy
B under historical conditional WG, has materially lower completion or higher
rejection, and loses or ties under arrival-normalized WG.

Across the three rescored pilot CSVs:

- total pairwise reversals: 1534
- advantaged policies:
  - `scorpio_style_slo_guard`: 1383
  - `vllm_faithful`: 205
  - `sarathi_faithful`: 8

The dominant reversal mechanism is selective service by SCORPIO. A
representative BurstGPT admission-pressure window has SCORPIO beating vLLM by
0.0536 historical WG while losing by -0.1131 ANWG, with completion fraction
lower by -0.1667 and rejection higher by +0.1667.

## Constraint Sensitivity

Exploratory constrained ANWG grid:

- completion minimum: 0.5, 0.7, 0.8, 0.9
- maximum rejection: 0.5, 0.3, 0.2, 0.1

On the 307-window targeted diagnostic, tightening rejection limits from 0.5 to
0.1 reduces SCORPIO constrained wins from 3 to 1, while vLLM stays high
(about 140-142 wins) and Sarathi remains at 52. Raising completion minimum to
0.9 has little additional effect because the per-window winners under ANWG
already mostly complete nearly all requests.

Do not hard-code beta/rho from this pilot. The stable finding is directional:
constraints suppress selective-service wins and expose non-SCORPIO policies.

## Historical Claims Impact

Repository search found 401 references to `weighted_goodput`,
`priority_weighted_slo_goodput`, `arrival_normalized_wg`, or related goodput
claims across docs, scripts, and selector code.

Classification:

| Area | Classification | Rationale |
|---|---|---|
| `src/llmserveopt/core/metrics.py` historical `weighted_goodput` | HISTORICAL_SAFE | Implementation preserved exactly; docs now clarify conditional semantics. |
| Historical selector v1 labels (`src/llmserveopt/selector/labels.py`, `src/llmserveopt/selector/dataset.py`, `docs/selector.md`) | NEEDS_REINTERPRETATION | Labels optimize completed-request conditional quality, not system-level goodput. Do not use as final Selector v2 target. |
| `docs/problem_formulation.md` and `docs/llm_heuristic_dsl.md` formula claims | POTENTIALLY_MISLEADING | Formula text omits the completed-only implementation denominator. Needs a focused rewrite before manuscript use. |
| Phase 2B.10-2B.13 SCORPIO/selector claims | NEEDS_RECOMPUTATION or NEEDS_REINTERPRETATION | SCORPIO-heavy gains may reflect selective-service advantage. |
| Phase 2B.14/2B.15/2B.16 corrected-objective docs/scripts | HISTORICAL_SAFE | They already identify completed-only WG and use arrival-normalized variants. |
| External-baseline integration docs that say ANWG is not in `RunMetrics` | NEEDS_RENAMING_ONLY | Now outdated by this audit; historical artifacts remain unchanged. |
| Plot/table scripts reading `priority_weighted_slo_goodput` | HISTORICAL_SAFE for reproduction, NEEDS_RENAMING_ONLY for paper labels | They should label old results as conditional weighted SLO attainment. |
| Dataset v2 pilot scripts that summarize `weighted_goodput` | NEEDS_REINTERPRETATION | Keep diagnostics, but final data generation should use corrected/constrained targets. |

Do not mass-edit old phase documents in this task. Historical artifacts must
remain reproducible and interpretable.

## Recommendation

Recommended manuscript-primary objective:

```text
arrival-normalized priority-weighted SLO goodput
```

This answers: "What does it mean for a scheduler to best handle the workload
offered to it?" A scheduler earns utility for offered requests that it actually
serves before deadline, weighted by priority. Rejected, dropped, unfinished,
and SLO-missed arrivals are zero utility but remain in the denominator.

Recommended secondary metrics:

- completion fraction and weighted completion fraction
- rejection fraction
- SLO attainment among completed requests
- request/token throughput
- p95 latency, p95 TTFT, p95 TPOT/TBT
- constrained ANWG sensitivity over completion/rejection requirements
- Pareto view over ANWG, completion, rejection, and latency

Recommended selector target:

- preserve the full policy-performance vector
- train regret-aware or per-policy utility-regression models on
  `arrival_normalized_weighted_goodput`
- report constrained ANWG and Pareto-aware diagnostics
- avoid single-label classification on near-tie windows

Selector v2 data generation can resume only after confirming that corrected
ANWG or a constrained ANWG target produces meaningful multi-policy
specialization and nontrivial oracle headroom on a pilot.

## Backward Compatibility

Preserved:

- old `RunMetrics.weighted_goodput`
- old `priority_weighted_slo_goodput` alias
- old CSV/JSON field semantics
- historical selector label code

Added:

- `RunMetrics.arrival_normalized_weighted_goodput`
- `RunMetrics.weighted_completion_fraction`
- `RunMetrics.conditional_weighted_slo_attainment` semantic alias
- Dataset v2 fields for corrected ANWG, weighted completion fraction, and
  rejection fraction
- objective-audit utilities for constrained ranking and reversal detection

Historical reproducibility is preserved.
