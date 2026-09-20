# FRESH_PRODUCTION_LATENCY_HEADROOM_CONFIRMATORY_V1

Status: preregistration design freeze. Do not execute support mapping or causal
labeling in the design task.

## Methodological Correction

The existing Phase-D2 latency design is reclassified as:

`EXISTING_D2_STATUS = POST_HOC_OBJECTIVE_REANALYSIS_DESIGN`

Reason: mean latency was selected after the Phase-D V1 continuation table had
already shown that latency varies. That same-data analysis can characterize
mechanisms and compare ANWG saturation with latency-sensitive objectives, but it
is not a fresh confirmation.

## Confirmatory Question

On previously unused production-derived trace windows, when realistic resource
pressure creates SBS-vs-P6 canonical action disagreement, does forcing a
non-SBS action once and then reverting to SBS produce positive causal headroom
for mean request latency?

## Fresh Windows

Use the frozen window universe in
`experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_WINDOW_UNIVERSE_V1.json`.

Selection rule:

1. Load each local `public_trace_corpus_v1` source.
2. Partition records into full contiguous 200-request windows by row order.
3. Exclude every window used in Phase A/B/D.
4. Select the first 20 remaining full windows per workload by ascending window index.
5. Verify zero overlap by `source_record_id`.

Frozen selected windows:

- Azure 2023 code: `1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39`
- Azure 2023 conversation: `1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26`
- BurstGPT: `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20`

Overlap with Phase A/B/D by source request identity is zero.

## Support Stage

Fresh support mapping precedes causal labeling.

Use the Phase-B V2 one-axis pressure grid:

- arrival multiplier: `{0.5, 1, 2, 4, 8}`
- KV capacity: `{8000000, 240000, 120000, 60000, 32000, 16000, 8000}`
- active-sequence capacity: `{512, 64, 32, 16, 8, 4}`

Use the same one-axis isolation, validity classes, canonical SBS-vs-P6
disagreement detection, and denominator conventions as Phase B V2.

Do not change the grid based on existing Phase-D V1 latency signs or magnitudes.

## Regime Selection

After fresh support mapping, mechanically select regimes by workload and axis:

1. first valid pressure setting where canonical disagreement appears;
2. first sustained-disagreement pressure setting under the Phase-B sustained rule;
3. strongest valid preregistered pressure setting before invalidity/truncation.

Deduplicate when these are the same condition. Arrival regimes remain structural
controls if no canonical disagreement appears.

Do not select regimes based on latency outcomes.

## Causal Population

The fresh causal population is all canonical disagreement states in the
mechanically selected fresh regimes.

If reference plus unique alternative continuations are `<= 50000`, label
exhaustively. If above that threshold, use deterministic stratified sampling by:

- workload;
- axis;
- regime stage;
- source window.

Sampling must be outcome-blind and use seed `20260920`.

## Intervention Semantics

For each disagreement state `s` and each unique non-SBS canonical action `a`:

- force exactly action `a` once at `s`;
- immediately revert to fixed SBS / `kv_constrained_online`;
- preserve future arrivals;
- preserve simulator configuration and random/common-random state;
- compare against an SBS reference branch at the same state.

No policy-forever switch, selector training, learned continuation, new
scheduler, or portfolio extension is permitted.

## Primary Metric

Use continuation-population mean request latency, not the historical simulator
aggregate that includes pre-intervention completed requests.

For state `s`, define the continuation request population \(R_s\) as every
source-window request that has not completed before the intervention state. This
includes active, queued, and future requests. The same \(R_s\) must be used for
the SBS reference and every counterfactual branch.

For each request \(i \in R_s\):

\[
latency_i = completion\_time_i - arrival\_time_i
\]

Primary branch advantage:

\[
A_{LAT}(s,a) = L_{SBS}(s) - L_{CF}(s,a)
\]

where \(L\) is the mean of `latency_i` over \(R_s\). Positive means the
alternative action reduces mean latency.

State-level oracle latency headroom:

\[
H_{LAT}(s) = max(0, max_a A_{LAT}(s,a))
\]

Beneficial state:

\[
B_{LAT}(s)=1 \iff max_a A_{LAT}(s,a) > 0
\]

Units: seconds.

If a branch has unfinished requests or horizon truncation, classify it under the
frozen validity rules and do not silently treat missing latency as zero.

## Primary Endpoint And Confirmation Rule

Primary endpoint:

`mean_s H_LAT(s)` over all causal disagreement states in selected fresh regimes,
with zero-headroom states included.

Clustered inference:

- cluster unit: faithful source window;
- bootstrap replicates: `2000`;
- seed: `20260920`;
- percentile `95%` CI;
- minimum contributing windows for CI: `5`.

Confirm:

`LATENCY_HEADROOM_CONFIRMED` iff:

1. primary mean oracle latency headroom is greater than `0`; and
2. the lower endpoint of the clustered 95% CI is greater than `0`.

If fewer than five independent source windows contribute:

`INSUFFICIENT_INDEPENDENT_WINDOWS_FOR_CONFIRMATION`

Do not use state-level IID bootstrap.

## Secondary Metrics

Report, without changing the primary endpoint:

- `P(B_LAT | D)`;
- mean and median positive latency headroom;
- fraction all-harmful, all-zero, and mixed states;
- action-level positive/negative/zero rates;
- relative latency headroom:
  \[
  R_{LAT}(s,a) = (L_{SBS}(s) - L_{CF}(s,a)) / L_{SBS}(s)
  \]
  where denominator is positive;
- state-level normalized oracle headroom:
  \[
  H_{REL}(s)=max(0,max_a R_{LAT}(s,a))
  \]
- p95 latency as secondary;
- ANWG as secondary continuity with Phase-D V1.

Do not choose whichever metric looks better after outcomes are visible.

## Claim Boundaries

Positive fresh confirmation supports:

Under production-derived resource-constrained replay, canonical scheduling
disagreement can contain positive one-step SBS-relative latency headroom.

It does not support:

- production deployment improvement;
- general superiority of P6;
- learned-selector success;
- all-scheduler generalization.

A null result is also valid and would imply that trajectory changes do not
robustly transfer to positive oracle latency headroom on fresh windows.

## Old Phase-D Latency Analysis

Old Phase-D V1 latency analysis is allowed only after this protocol is committed
and pushed, and only as:

`PHASE_D_V1_POST_HOC_LATENCY_REANALYSIS`

Its role is mechanism characterization, ANWG-vs-latency comparison, and
hypothesis generation. It cannot determine this fresh design.

## Data Sufficiency

The local corpus is sufficient for a fresh independent confirmation using the
same three workload classes:

- Azure 2023 code: `24` unused full windows exist; `20` are frozen.
- Azure 2023 conversation: `76` unused full windows exist; `20` are frozen.
- BurstGPT: `7001` unused full windows exist; `20` are frozen.

No Phase-A/B/D request identity appears in the fresh set.

## Additional Trace Planning

A fourth modern trace could still be valuable for the FGCS real-world-evidence
gate, but it is not needed to freeze this confirmatory study. Planning rank:

1. Azure 2024 or another newer Azure-derived trace, if request timestamps and
   token information are locally available under compatible terms.
2. A public agentic/coding trace with faithful arrival timing and token counts,
   because it would broaden workload class coverage.
3. Bailian/Qwen-style or Mooncake/Kimi-style traces, if license, timestamp
   fidelity, and replay compatibility can be established.

Do not add a fourth trace silently. It should be a separately versioned
workload-extension task if pursued.

## FGCS Readiness Forecast

Current internal readiness remains `78/100` with contribution-strength
confidence `68%`. This design alone does not increase the score. A successful
fresh latency confirmation could resolve the causal-headroom uncertainty for a
latency objective and materially improve technical depth, practitioner value,
and confidence. Remaining likely requirements would still include bounded
real-system validation, manuscript restructuring, and possibly broader workload
coverage.
