# Selector Dataset v2

Selector Dataset v2 is the replacement methodology for training and evaluating
the next-generation policy selector. It is topology-aware, leakage-constrained,
regret-aware, and preserves the full policy-performance vector for each
scenario/window rather than reducing construction output to one winner label.

This work does **not** train the final selector.

Objective note: the historical field named `weighted_goodput` is preserved for
reproducibility, but it is a completed-request conditional SLO-attainment metric.
Use `arrival_normalized_weighted_goodput` for Selector Dataset v2 target
construction. See `docs/selector_objective_audit.md` for the formula audit,
rejection semantics, and manuscript recommendation.

External-validity note: GPU validation of the monolithic faithful baselines is
tracked in `docs/gpu_external_validity_audit.md`. Large-scale Dataset v2
generation should not resume until vLLM/Sarathi advantage regimes are either
validated against GPU behavior or explicitly scoped as simulator limitations.

## Motivation

The historical selector datasets are insufficient for final manuscript claims:

- They contain too few windows for learning-curve-backed selector evaluation.
- Later selector datasets contain many near-tie or all-complete windows.
- Always-SCORPIO was nearly as good as the per-window oracle in held-out tests.
- The five faithful external baselines are absent from historical selector data.
- Existing selector features are too narrow for topology-specific selectors.
- Several legacy datasets use `online_prefix`, which is now explicitly treated as
  `offline_window_lookahead` and is not deployable for selector claims.

## Historical Dataset Audit

Classification legend:

- `REUSE_AS_TRAINING`: acceptable as final selector training data.
- `REUSE_AS_AUXILIARY`: useful for auxiliary analyses, calibration, or coverage design.
- `REUSE_AS_REGRESSION_ONLY`: useful only to ensure old behavior remains reproducible.
- `RETIRE_FROM_FINAL_SELECTOR`: do not use for final selector training/evaluation.
- `NEEDS_REVIEW`: requires manual inspection before reuse.

| Dataset | Scenarios | Windows | Policy portfolio | Topology | Workload sources | Synthetic vs real | Seeds | Split methodology | Label distribution | Near-tie rate | All-complete rate | Top-policy concentration | Faithful external baselines | Leakage risks | Useful? | Classification |
|---|---:|---:|---|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|
| Phase 2A.2 smoke selector | ~1 | 4 | 16 historical deployable policies | Monolithic | Synthetic smoke | Synthetic | config seed | none/smoke | best_fit 4/4 | 100% | n/a | 100% | No | `online_prefix` lookahead; tiny | Regression only | `REUSE_AS_REGRESSION_ONLY` |
| Phase 2A.3 train | multiple config workloads | 19 | 16 historical policies | Monolithic | Synthetic config workloads | Synthetic | config seeds | separate config files | multi_bin_batching 6, shortest_output_first 5, edf 5, others 3 | 78.9% | n/a | 31.6% | No | `online_prefix` lookahead; tiny; derived configs may cross splits | Regression only | `REUSE_AS_REGRESSION_ONLY` |
| Phase 2A.3 validation | multiple config workloads | 8 | 16 historical policies | Monolithic | Synthetic config workloads | Synthetic | config seeds | separate config files | best_fit 3, shortest_output_first 3, edf 1, weighted_shortest_processing 1 | 87.5% | n/a | 37.5% | No | `online_prefix` lookahead; tiny | Regression only | `REUSE_AS_REGRESSION_ONLY` |
| Phase 2A.3 test | multiple config workloads | 9 | 16 historical policies | Monolithic | Synthetic config workloads | Synthetic | config seeds | separate config files | multi_bin_batching 4, weighted_shortest_processing 2, others 3 | 55.6% | n/a | 44.4% | No | `online_prefix` lookahead; tiny | Regression only | `REUSE_AS_REGRESSION_ONLY` |
| Phase 2A.4 train | multiple config workloads | 30 | 18 historical policies | Monolithic | Synthetic config workloads | Synthetic | config seeds | separate config files | multi_bin_batching 11, shortest_output_first 8, edf 5, others 6 | 66.7% | n/a | 36.7% | No | `online_prefix` lookahead; tiny | Regression only | `REUSE_AS_REGRESSION_ONLY` |
| Phase 2A.4 validation | multiple config workloads | 13 | 18 historical policies | Monolithic | Synthetic config workloads | Synthetic | config seeds | separate config files | best_fit 8, shortest_output_first 3, edf 1, estimated_service_time_first 1 | 92.3% | n/a | 61.5% | No | `online_prefix` lookahead; tiny | Regression only | `REUSE_AS_REGRESSION_ONLY` |
| Phase 2A.4 test | multiple config workloads | 9 | 18 historical policies | Monolithic | Synthetic config workloads | Synthetic | config seeds | separate config files | multi_bin_batching 4, weighted_shortest_processing 2, others 3 | 55.6% | n/a | 44.4% | No | `online_prefix` lookahead; tiny | Regression only | `REUSE_AS_REGRESSION_ONLY` |
| Phase 2B.9 robustness | n/a | summary-only in current tree | historical selector variants | Monolithic | robustness configs | Synthetic | config seeds | dev/heldout summaries | per-window CSV absent in current tree | n/a | n/a | n/a | No | cannot reconstruct full rows from current files | Audit context only | `NEEDS_REVIEW` |
| Phase 2B.12 diversity | 12+ workloads | 172 | 20 historical policies | Monolithic | diversity synthetic workloads | Synthetic | config seeds | dev/heldout summaries | scorpio_style_slo_guard 79, admission_control 29, best_fit 14, edf 14, shortest_output_first 13 | 74.4% | n/a | 45.9% | No | label-only framing; near ties dominate | Auxiliary diversity signal | `REUSE_AS_AUXILIARY` |
| Phase 2B.13 selector training and suspicion audit | 12+ workloads | 319 | 20 historical policies | Monolithic | diversity synthetic workloads | Synthetic | config seeds | dev/heldout summaries | scorpio_style_slo_guard 176, admission_control 37, best_fit 28, edf 20, multi_bin_batching 19 | 60.5% | 33.5% | 55.2% | No | many near-tie/all-complete windows; no external baselines | Auxiliary/negative evidence | `REUSE_AS_AUXILIARY` |
| Phase 2B.13 after diversity | 12+ workloads | 256 | 20 historical policies | Monolithic | diversity synthetic workloads | Synthetic | config seeds | dev/heldout summaries | scorpio_style_slo_guard 112, admission_control 37, best_fit 36, edf 20, multi_bin_batching 18 | 75.0% | n/a | 43.8% | No | many near ties; label-only | Auxiliary only | `REUSE_AS_AUXILIARY` |
| Phase 2B.15 corrected objective | n/a | 20 policy aggregate rows | 20 historical policies | Monolithic | inherited B13/B16 data | Synthetic | inherited | train/val/test summary | policy table, not window dataset | n/a | n/a | n/a | No | not a scenario/window dataset | Regression/objective audit | `REUSE_AS_REGRESSION_ONLY` |
| Phase 2B.16 fresh corrected validation | multiple fresh workloads | 174 | 20 historical policies | Monolithic | fresh validation synthetic workloads | Synthetic | config seeds | fresh heldout | scorpio_style_slo_guard 71, admission_control 28, best_fit 20, edf 20, multi_bin_batching 10 | 83.3% | 42.5% | 40.8% | No | near-tie/all-complete dominated | Auxiliary/negative evidence | `REUSE_AS_AUXILIARY` |
| Phase 2C labeled selector | 611 rows | 611 | 20 historical policies plus pairwise labels | Monolithic | mixed real/synthetic selector rows | Mixed | generated manifest | train/val/eval files | selected_policy empty 286, scorpio 241, fifo 49, edf 35 | n/a | 17.2% | 39.4% over all rows | No | label task differs; blank labels; no new baselines | Auxiliary only | `REUSE_AS_AUXILIARY` |
| Phase 2C1 real trace validation | 6 workloads | 325 | 20 historical policies | Monolithic | BurstGPT, Azure 2023 | Real traces with synthetic SLO/priority/prediction overlay | manifest/config | real-trace validation | scorpio_style_slo_guard 309, multi_bin_batching 11, admission_control 3, edf 1, shortest_output_first 1 | 25.2% | 0.0% | 95.1% | No | external baselines absent; single-policy dominance | Real-trace auxiliary/regression | `REUSE_AS_AUXILIARY` |
| Phase 2C2 causal selector retraining | mixed rows | 286 train rows | 20 historical policies | Monolithic | inherited B13/B16/real-trace rows | Mixed | inherited | causal train/val | scorpio_style_slo_guard 155, admission_control 33, best_fit 27, multi_bin_batching 19, edf 17 | 59.8% | 36.7% | 54.2% | No | no faithful external baselines; many near ties | Auxiliary only | `REUSE_AS_AUXILIARY` |
| Phase 2C3 external-aware ORCA recovery | smoke outputs only | per-window file absent at audited path | recovery labels, not selector-v2 portfolio | Monolithic | Azure-derived diagnosis | Mixed | smoke timestamp | smoke | unavailable from current path | n/a | n/a | n/a | No | incomplete artifact path | Review only | `NEEDS_REVIEW` |

Conclusion: no historical dataset is classified `REUSE_AS_TRAINING` for the
final selector. The useful material is methodology, failure evidence, real-trace
ingestion code, and regression fixtures.

## Dataset Unit and Schema

The fundamental flattened row is:

```text
scenario/window x topology x policy
```

Each `WindowRecordV2` contains:

- identifiers: `scenario_id`, `scenario_family_id`, `dataset_family`,
  `source_trace`, `temporal_block_id`, `seed`, `topology_class`,
  `resource_configuration_id`, `window_id`
- leakage-safe selector features with a `feat_` prefix
- one `PolicyOutcomeVector` per compatible policy
- discriminativeness records per objective
- regret records per objective/policy

The machine-readable manifest is `DatasetManifestV2` in
`src/llmserveopt/selector/dataset_v2/schema.py`. A generated dataset must include
at least:

- `schema_version`
- `topology_class`
- `candidate_policies`
- `feature_names`
- `objectives`
- scenario/window/policy-evaluation counts
- scenario families, sources, seeds
- split group key and split counts
- quality gate results
- generation configuration

## Leakage Prevention

Feature extraction for Dataset v2 is implemented in
`src/llmserveopt/selector/dataset_v2/features.py`.

Rules:

- Never use `actual_output_tokens` as a selector feature.
- Never use future arrivals inside the current window.
- Never use post-hoc latency/completion/SLO outcomes as selector features.
- Identifier/provenance columns are retained in rows but are not model features.
- Model feature columns are only the explicit `feat_` columns.
- Group-aware splits operate on source/trace/family groups, not individual rows.

Tests prove:

- changing `actual_output_tokens` does not change features
- mutating later within-window requests does not change current features
- held-out trace/source identifiers are excluded from model feature columns
- OOD groups cannot appear in non-OOD splits

## Features

Dataset v2 feature families are causal and online-available:

- arrival/load: recent and prefix arrival-rate estimates, inter-arrival CV,
  burstiness, queue length, recent queue growth, active sequence count,
  saturation estimate
- prompt: mean, median, p90, p95, variance, CV
- predicted output: mean, median, p90, p95, variance, CV
- SLO: tight-SLO fraction, mean slack, p10 slack, minimum slack, recent SLO
  violation rate when legitimately available
- priority: mean, p90, high-priority fraction, priority-class count
- resource: GPU count, KV capacity, block size, sequence capacity, token budget
- monolithic: aggregate KV utilization, active batch size when available
- disaggregated: role counts, prefill/decode/bridge queue and utilization fields
  when infrastructure exposes them
- multi-instance: instance count, load/KV imbalance, incoming migration count,
  migration pressure when infrastructure exposes them

Unavailable topology metrics are stored as missing/`None`, not zero.

## Full Policy Outcomes

For every compatible policy, Dataset v2 preserves:

- weighted goodput
- arrival-normalized weighted goodput
- completion fraction
- SLO attainment and violation rate
- request/token throughput
- mean, p50/median, p95, p99 latency where available
- mean, p50, p95, p99 TTFT/TPOT/TBT where available
- admission/rejection rates and dropped request counts
- preemption, swap, migration event counts
- policy-decision overhead
- simulation wall time
- resource GPU count
- disaggregated per-role and queue statistics when infrastructure supports them

Unavailable values are `None` plus `available_metrics` metadata.

## Objectives and Regret

The dataset does not reduce immediately to one classification target. It
preserves `score(scenario, policy)` and computes best policy, ranking, tie set,
winner margin, top-2 gap, and regret for:

- weighted goodput
- arrival-normalized weighted goodput
- p95 latency
- SLO attainment
- request throughput

Regret is sign-aware:

```text
regret(s, p) = score(best compatible policy for s) - score(p)
```

For lower-is-better metrics, the sign is reversed so regret remains nonnegative.

## Near-Tie Handling

Every window/objective records:

- best score
- second-best score
- absolute winner margin
- relative winner margin
- max-min policy spread
- tie set
- class: `STRONGLY_DISCRIMINATIVE`, `MODERATELY_DISCRIMINATIVE`, `NEAR_TIE`,
  or `ALL_COMPLETE_OR_EFFECTIVELY_TIED`

Near-tie windows are realistic and must remain in final evaluation, but they
must not dominate training. Training should use regret-aware weights and
stratified sampling/reporting by discriminativeness. Alphabetical tie-breaking
is not meaningful ground truth.

## Workload Sources

Local available sources:

- BurstGPT raw CSV and processed JSONL are present under `data/raw/burstgpt/`
  and `data/processed/burstgpt/`.
- Azure LLM 2023 code/conversation raw CSV and processed JSONL are present under
  `data/raw/azure/` and `data/processed/azure/`.
- Synthetic workload generator is present in `src/llmserveopt/workloads/synthetic.py`.
- ShareGPT conversion code and tests exist, but the raw ShareGPT file is not
  present locally.

Recommended future acquisitions, without silent download:

- Azure 2024/2025 LLM/LMM traces from `https://github.com/Azure/AzurePublicDataset`
- Mooncake/Kimi traces from `https://github.com/kvcache-ai/Mooncake`
- ServeGen from `https://github.com/alibaba/ServeGen`
- TraceLab from `https://github.com/uw-syfi/TraceLab.git`

For each source, the manifest records real fields, synthesized fields, schema,
license note, official URL, and local acquisition state. Real timestamps/token
lengths are never conflated with synthetic SLOs/priorities/prediction noise.

## Scenario-Family Design

Dataset v2 combines:

- real production trace scenarios: temporal slices/windows from BurstGPT and
  Azure 2023 locally, with additional sources only after explicit acquisition
- real-distribution synthetic scenarios: seeded lognormal/heavy-tail scenarios
  using existing workload generators
- controlled stress scenarios: low load, moderate load, near saturation,
  overload, burst overload, KV pressure, prefill-heavy, decode-heavy, mixed
  short/long jobs, high prediction noise, tight SLOs, mixed priorities, rapid
  shifts

The design is coverage-aware, not a naive Cartesian product.

## Splits

Required splits:

- `TRAIN`
- `VALIDATION`
- `ID_TEST`
- `OOD_TEST`

Splits are group-aware. Candidate grouping units include source trace,
temporal block, scenario family, base synthetic distribution, and request-plan
ancestor. Derived variants of a base trace must not casually appear across
train and test. OOD should hold out at least one full workload-source family
when source coverage permits.

## Topology and Candidate Policy

Phase 1 prioritizes the monolithic selector dataset because it has enough
legitimate within-class candidates.

Monolithic candidates:

- `vllm_faithful`
- `sarathi_faithful`
- `fifo`
- `edf`
- `scorpio_style_slo_guard`
- `orca_style`
- `slo_slack_score`
- `admission_control`
- `weighted_shortest_processing`
- `shortest_output_first`
- `estimated_service_time_first`
- `best_fit`
- `multi_bin_batching`

> **Superseded for the trainable action space:** the list above was this
> design doc's original general monolithic candidate pool. The current
> approved Selector v2 *trainable* action space is the narrower **Option B**
> scope decided in `docs/selector_v2_faithful_baseline_scope_audit.md`: the
> 8 historical-monolithic policies only (`fifo`, `edf`,
> `scorpio_style_slo_guard`, `admission_control`,
> `weighted_shortest_processing`, `estimated_service_time_first`,
> `best_fit`, `multi_bin_batching`). `vllm_faithful`, `sarathi_faithful`,
> and `vllm_chunked_prefill_faithful` (added after this list was written)
> are confirmed genuinely dominated under ANWG and are evaluated
> separately, never trained on. `orca_style`, `slo_slack_score`, and
> `shortest_output_first` are also excluded from the trainable set under
> Option B. The in-code candidate resolver at
> `src/llmserveopt/selector/dataset_v2/candidates.py` has not yet been
> reconciled with this narrower scope (tracked as follow-up work); the
> actual current pipeline's candidate set lives in
> `src/llmserveopt/selector/dataset_v2/calibrated_targeted_pilot.py`.

Inclusion criteria:

- compatible with monolithic shared-queue topology
- online deployable, not an oracle
- scientifically distinct scheduling/admission behavior
- either a faithful external baseline or a strong/diagnostic historical policy
- length-specialist and placement/batching policies are included in the
  redesigned pilot because v1 did not sufficiently expose KV/decode/resource
  bottlenecks

Disaggregated and migratory topologies are represented in schema/infrastructure,
but no final selector should be trained until there are enough legitimate
within-class candidates and scenarios.

## Pilot v1 Failure Analysis

The first Dataset v2 pilot produced 260 windows and 2,340 policy evaluations.
Using the original summary, 78.46% of windows had every policy complete every
request. Under the stricter practical-equivalence threshold introduced here,
90.0% of windows are weighted-goodput equivalent or all-complete/effectively
tied.

Quantitative causes from `results/selector_dataset_v2/pilot/failure_analysis.json`:

- KV pressure was negligible in equivalent windows: p50 `pred_output_p95 /
  kv_capacity` was 0.0078, p90 was 0.0279.
- Service-token pressure did not translate into weighted-goodput differences
  because the v1 pilot used the default service model with instantaneous
  prefill; p50 prompt/token-budget pressure was only 0.623.
- Weighted goodput saturated: 234/260 windows had zero practical WG spread.
- Differentiation appeared in non-primary metrics more often than in WG:
  arrival-normalized WG and throughput differentiated 56 windows, p95 latency
  37, SLO attainment and WG only 26.
- Strong WG windows were concentrated in KV/mixed/decode families and 25/25
  strong windows were won by `scorpio_style_slo_guard`.
- Sarathi's intended chunked-prefill advantage was mostly invisible because
  `ServiceModel(enable_prefill_modeling=False)` makes prefill instantaneous.

Conclusion: the v1 pilot was not merely too small; it was structurally
under-stressed for the primary weighted-goodput objective and did not expose
topology-specific mechanisms.

## Scenario Redesign

The redesigned scenario taxonomy is implemented in
`src/llmserveopt/selector/dataset_v2/scenario_redesign.py`.

Targeted bottleneck classes:

- `admission_pressure`: overloaded tight-SLO and mixed-priority regimes.
- `kv_pressure`: low KV capacity with long/high-variance outputs.
- `prefill_heavy`: long prompts, short outputs, constrained prefill token
  budget, and `enable_prefill_modeling=True`.
- `decode_heavy`: long-running decode occupancy with output variance.
- `slo_heterogeneous`: mixed tight/loose SLOs and heterogeneous priorities.
- `prediction_noise`: exact/noisy/biased predicted output lengths.
- `bursty_transient`: short queue shocks and recovery windows.
- `resource_scarcity`: independently varied sequence cap, token budget, and KV
  capacity.

Real-trace stress transforms are applied to local BurstGPT and Azure 2023
traces:

- time compression/expansion
- burst amplification
- SLO scaling
- prediction-noise and underprediction transforms
- resource scaling

Every transformed real-trace variant keeps a shared `request_plan_ancestor_id`,
so split assignment can hold siblings together.

## Adaptive Search

The redesigned pilot script is:

```bash
python scripts/build_selector_dataset_v2_redesigned_pilot.py \
  --output-dir results/selector_dataset_v2/redesigned_pilot
```

The search loop:

1. samples or enumerates a candidate bottleneck scenario
2. runs a core policy subset for search-time cost control
3. computes primary-objective spread, winner identity, and local oracle headroom
4. retains trials into either `REPRESENTATIVE_POOL` or `DISCRIMINATIVE_POOL`
5. skips redundant trials once one winner dominates the retained pool
6. rebuilds retained trials with the full monolithic candidate set

The search is deterministic for fixed seeds. It deliberately refuses to pad the
dataset with redundant SCORPIO-only windows just to hit a row-count target.

## Thresholds

Corrected-objective thresholds use practical significance rather than purely
relative numerical thresholds. For the primary
`arrival_normalized_weighted_goodput` objective, the utility is on a 0-1 scale
over weighted arrivals:

- equivalent/all-complete: max-min objective spread <= 0.002
- near tie: top-2 gap <= 0.002 or relative gap < 0.5%
- moderate: non-equivalent but top-2 gap < 0.02 and relative gap < 3%
- strong: top-2 gap >= 0.02 or relative gap >= 3%

These thresholds avoid treating tiny floating-point or tie-break differences as
selector ground truth.

## Redesigned Pilot Result

The first redesigned run was intentionally stopped from padding with redundant
SCORPIO-heavy scenarios. It retained 51 windows and 663 policy evaluations, so
it is a diagnostic pilot, not a target-size dataset.

Redesigned pilot statistics:

- all-complete/effectively tied fraction: 23.53%
- near-tie fraction: 21.57%
- moderately discriminative fraction: 0.0%
- strongly discriminative fraction: 54.90%
- global best fixed policy: `scorpio_style_slo_guard`
- global best fixed WG: 0.9411764706
- per-scenario oracle WG: 0.9929557008
- oracle headroom: 0.0517792302
- discriminative oracle headroom: 0.0678571429

Win distribution:

- `scorpio_style_slo_guard`: 27
- `vllm_faithful`: 20
- `sarathi_faithful`: 2
- `edf`: 2

Discriminative-window wins after correcting the gate to exclude near/equivalent
windows:

- `scorpio_style_slo_guard`: 26
- `vllm_faithful`: 2

Quality gates passed:

- all-complete fraction below 40%
- moderate+strong discriminative fraction increased
- nontrivial overall oracle headroom
- nontrivial discriminative oracle headroom
- real-trace representation preserved

Quality gates failed:

- only 51 retained windows, below the 500-1,000 target
- fewer than 3 policies win genuinely discriminative windows
- faithful baselines do not yet win enough discriminative windows
- strong-window top-policy share is 92.86%, above the 85% cap
- OOD split was not defined because the retained set did not keep the Azure
  ancestor needed for OOD holdout

Policy-specialization observations from
`results/selector_dataset_v2/redesigned_pilot/policy_specialization.json`:

- `scorpio_style_slo_guard` wins strongly under tight-SLO, high-burst,
  high-load windows, especially admission-pressure and bursty-transient cases.
- `vllm_faithful` wins many easy/representative or resource-scarcity windows,
  but most are equivalent or near-tie; only 2 are strongly discriminative.
- `sarathi_faithful` appears in long-prompt/resource-scarcity regions, but its
  wins are near ties in this run.
- `edf` appears only in near-tie burst/noise windows.

The redesigned methodology improved headroom and reduced all-complete windows,
but the dataset is still not ready for large-scale generation.

## Targeted Counterexample Discovery

Follow-up targeted discovery added deterministic counterexample families for
underrepresented policies. These are search hypotheses, not labels:

- `counterexample__sarathi_faithful__*`: long prompts, moderate outputs,
  prefill modeling enabled, decode-first scheduling, varied step token budget
  and max prefill chunk size.
- `counterexample__vllm_faithful__*`: decode-heavy, high-variance outputs,
  varied KV capacity and sequence cap under looser SLOs.
- `counterexample__deadline_policy__*`: heterogeneous deadlines and mixed
  priorities under moderate overload for EDF, SLO-slack, admission-control,
  WSP, and ESTF-style policies.

Adaptive retention was tightened after an audit found that the first targeted
run retained "target-policy" trials when the target policy only won
all-complete or near-tie windows. The corrected retention path uses
strong-window winner counts separately from overall winner counts.

Two diagnostic runs are recorded under `results/selector_dataset_v2/`:

- `targeted_discovery_pilot`: permissive retention, 307 windows.
- `targeted_discovery_pilot_strict_small`: corrected strict retention, 9
  windows, one seed, compact target search.

Primary weighted-goodput result from the permissive targeted run:

- windows: 307
- weighted-goodput all-complete/equivalent fraction: 43.97%
- near-tie fraction: 28.66%
- strongly discriminative fraction: 27.36%
- strong-window winners: `scorpio_style_slo_guard` 84/84
- global best fixed: `scorpio_style_slo_guard`
- global best fixed WG: 0.9967426710
- per-window oracle WG: 0.9975570033
- oracle headroom: 0.0008143322
- discriminative oracle headroom: 0.0

Corrected strict targeted search did not find a strong primary-objective
Sarathi or vLLM counterexample. It retained only one prefill-heavy trial:

- windows: 9
- strong-window winners: `scorpio_style_slo_guard` 2/2
- global best fixed: `sarathi_faithful`
- global best fixed WG: 0.9546964193
- per-window oracle WG: 1.0
- oracle headroom: 0.0453035807

That strict run is scientifically useful because it shows Sarathi can be the
best fixed policy in a prefill-heavy region, but the primary objective still
does not produce strong per-window Sarathi labels. The strong windows are
SCORPIO wins caused by selective completion/rejection.

### SCORPIO Dominance Evidence

For 84 strongly discriminative SCORPIO wins in the targeted run, compared to
the second-best weighted-goodput policy:

- weighted-goodput gap mean: 0.3046
- completion-fraction gap mean: -0.4296
- rejection-rate gap mean: +0.4296
- SLO-attainment gap mean: +0.2490
- p95 TTFT gap mean: -0.2518 s
- p95 TPOT/TBT gap mean: approximately 0
- p95 latency gap mean: -0.3148 s
- median KV pressure: 0.0803
- median token-budget pressure: 0.3985

Interpretation: under the current primary weighted-goodput metric, SCORPIO's
strong wins are primarily selective-service wins. It rejects or leaves many
more requests incomplete, but its completed subset has better SLO attainment
and lower TTFT/latency. This is not just a workload-generator bias toward
admission pressure: the same targeted run contains many vLLM/Sarathi wins in
near-tie/equivalent windows and many non-SCORPIO winners under secondary
objectives.

### Objective Sensitivity

The primary objective remains unchanged for now, but the targeted run strongly
suggests that the current weighted-goodput implementation structurally favors
rejection-heavy policies because completed-request SLO goodput is not penalized
enough by arrivals that were dropped or never completed.

Secondary objective audit on `targeted_discovery_pilot`:

- weighted_goodput: strong wins = `scorpio_style_slo_guard` 84; oracle
  headroom = 0.000814
- arrival-normalized / completion-adjusted weighted goodput: best fixed =
  `edf`; all-window wins include `vllm_faithful` 129, `sarathi_faithful` 52,
  `fifo` 72, `edf` 30; strong wins include `scorpio_style_slo_guard` 11,
  `shortest_output_first` 2, `admission_control` 1; oracle headroom = 0.00658
- request throughput: best fixed = `fifo`; strong wins include
  `admission_control` 14, `estimated_service_time_first` 5,
  `shortest_output_first` 2, `weighted_shortest_processing` 1,
  `scorpio_style_slo_guard` 1; oracle headroom = 0.30745
- p95-latency-constrained goodput: strong wins include
  `scorpio_style_slo_guard` 178, `vllm_faithful` 28,
  `estimated_service_time_first` 4, `admission_control` 3; oracle headroom =
  2.31966

The objective audit in `docs/selector_objective_audit.md` resolved this: new
Selector Dataset v2 generation uses `arrival_normalized_weighted_goodput` as
the manuscript-primary selector objective while preserving historical
`weighted_goodput`/`conditional_weighted_slo_attainment` for reproduction.

## Corrected-Objective Pilot

The corrected-objective pilot was generated with:

```bash
python scripts/build_selector_dataset_v2_redesigned_pilot.py \
  --output-dir results/selector_dataset_v2/corrected_objective_pilot \
  --target-windows 260 \
  --sampled-candidates 36 \
  --counterexamples-per-target 18 \
  --seeds 11 17 \
  --max-real-requests 96 \
  --window-size 12 \
  --search-drain-steps 1500 \
  --drain-steps 8000
```

This is a CPU-only pilot and does not train a selector. The first attempted
360-window run was interrupted before writing artifacts because the search pass
was too slow for pilot iteration. The retained 266-window run is the reported
artifact.

Primary objective:

```text
arrival_normalized_weighted_goodput
```

Corrected pilot statistics:

- windows: 266
- policy evaluations: 3,458
- source traces: `burstgpt`, `azure_llm_2023`, `synthetic`
- split groups: `request_plan_ancestor_id`
- splits: TRAIN 1,976 rows; VALIDATION 767; ID_TEST 507; OOD_TEST 208
- OOD group: `real_trace__azure_2023_conv`
- all-complete/effectively tied fraction: 11.65%
- near-tie fraction: 76.69%
- moderately discriminative fraction: 0.0%
- strongly discriminative fraction: 11.65%
- global best fixed policy: `weighted_shortest_processing`
- global best fixed ANWG: 0.4297888538
- per-window oracle ANWG: 0.4412524070
- oracle headroom: 0.0114635532
- discriminative oracle headroom: 0.0437143093
- random-policy baseline ANWG: 0.4090268268
- simple rule-selector baseline ANWG: 0.3037251687

Primary-objective win distribution:

- `vllm_faithful`: 173
- `fifo`: 28
- `scorpio_style_slo_guard`: 25
- `edf`: 13
- `shortest_output_first`: 9
- `sarathi_faithful`: 8
- `admission_control`: 4
- `weighted_shortest_processing`: 3
- `orca_style`: 2
- `multi_bin_batching`: 1

Strong-window win distribution:

- `scorpio_style_slo_guard`: 24
- `admission_control`: 3
- `weighted_shortest_processing`: 2
- `multi_bin_batching`: 1
- `shortest_output_first`: 1

Objective stability:

- completion-adjusted WG differs from ANWG in 1/266 windows.
- historical conditional `weighted_goodput` differs from ANWG in 221/266
  windows and gives SCORPIO 237 total wins, with winner mean completion
  fraction 0.4596 and rejection fraction 0.5404.
- SLO-success throughput differs from ANWG in 80/266 windows; its best fixed
  policy is `orca_style`, with oracle headroom 0.15346.
- constrained ANWG with completion >= 0.8 and rejection <= 0.2 gives
  `vllm_faithful` 182 total wins and `sarathi_faithful` 8, but strong wins are
  still historical-policy wins.

Policy-specialization observations under ANWG:

- `vllm_faithful` wins many admission-pressure, bursty-transient,
  prediction-noise, and resource-scarcity windows, but all 173 wins are
  near-tie or all-equivalent under the current practical thresholds.
- `sarathi_faithful` wins 8 prediction-noise/long-prompt windows, also all
  near-tie. The current monolithic simulator still does not expose a strong
  chunked-prefill advantage against the full candidate set.
- `scorpio_style_slo_guard` wins strongly in decode-heavy and KV-pressure
  windows with long outputs, low prompt lengths, and tighter slack.
- `admission_control`, `weighted_shortest_processing`, `multi_bin_batching`,
  and `shortest_output_first` provide genuine non-SCORPIO strong wins, mostly
  under bursty, KV/resource-pressure, and admission-pressure windows.

Corrected-objective quality gates passed:

- all-equivalent fraction below 40%
- at least three policies win strongly discriminative windows
- no single policy exceeds 85% of strongly discriminative wins
- overall oracle headroom >= 0.01
- discriminative oracle headroom >= 0.03
- real-trace representation exists
- OOD split is defined

Corrected-objective quality gates failed:

- faithful external baselines have many total ANWG wins but 0 strongly
  discriminative wins after full-candidate evaluation
- moderate+strong discriminative fraction is only 11.65%, below the 35%
  search target
- `sarathi_faithful` meaningful wins remain near-tie rather than strong

Conclusion: the corrected objective makes a selector more plausible than the
historical metric because oracle headroom is nonzero and several non-SCORPIO
policies have strong wins. However, the current retained pilot is not ready for
large-scale generation because faithful external baselines do not yet win
strongly discriminative windows. Do not scale this dataset until either
faithful-baseline strong regions are discovered or the absence of such regions
is accepted as a reported scientific finding.

## Source Acquisition Plan

| Source | Official source | License | Approximate size | Real fields | Synthetic fields needed | Expected usefulness |
|---|---|---|---|---|---|---|
| Azure LLM 2024 | `https://github.com/Azure/AzurePublicDataset` and `AzureLLMInferenceDataset2024.md` | CC BY 4.0 per repository license | one-week sample; inspect file sizes before download | arrival timestamps, input tokens, output tokens | SLOs, priority, prediction noise | High: stronger OOD temporal/source coverage for monolithic selector |
| Azure LMM 2025 | `https://github.com/Azure/AzurePublicDataset` and `AzureLMMInferenceDataset2025.md` | CC BY 4.0 per repository license | one-week multimodal sample; inspect before download | timestamps, image count, input/output tokens | text-only projection, SLOs, priority, prediction noise | Medium: useful OOD, but multimodal fields need careful projection |
| Mooncake/Kimi | `https://github.com/kvcache-ai/Mooncake`, `FAST25-release/traces/` | verify repository license before redistribution | JSONL trace; inspect before download | arrivals, token counts, anonymized/remapped KV block-hash structure | SLOs, priorities, monolithic projection unless prefix cache is modeled | High later: especially useful once prefix/KV-cache features are modeled |
| ServeGen | `https://github.com/alibaba/ServeGen` and NSDI 2026 paper page | verify repository license before use | generator code; no GitHub releases observed during audit | generator parameters from Alibaba workload characterization, not raw production rows | project SLOs/priorities and generated request plans | High for coverage-aware synthetic generation without naive grids |
| TraceLab | `https://github.com/uw-syfi/TraceLab` | code Apache-2.0; public trace dataset CC BY 4.0 per repository | public normalized JSONL and DuckDB release; inspect release size | coding-agent sessions, LLM steps, tool-call structure | serving-window extraction, SLOs, priorities, token normalization | Medium/high OOD for agentic-serving selector evaluation |

### Local Source Provenance

Current local real traces:

- BurstGPT raw `data/raw/burstgpt/BurstGPT_1.csv`
  - sha256: `46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a`
  - real fields: arrival timestamps, prompt/request tokens, response tokens
  - synthetic fields: SLOs, priority classes, predicted output tokens
- Azure 2023 code raw `data/raw/azure/AzureLLMInferenceTrace_code_2023.csv`
  - sha256: `54e9a6d2a4bd06ba1e060304b900abbc74cbea53de96506e60fe5bb4f2277fb6`
  - processed sha256:
    `49a1aec622c8503872504ae5fe631d34128b034a73f9655153da5f9031365173`
- Azure 2023 conversation raw
  `data/raw/azure/AzureLLMInferenceTrace_conv_2023.csv`
  - sha256: `2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8`
  - processed sha256:
    `5de02a43248667ff3dba389c23492c1e1a5896e7a106b110a84ceacf3c7b804a`

All real-trace stress variants preserve `request_plan_ancestor_id`; variants
of one raw ancestor must not be split casually across train/OOD boundaries.

## Pilot and Learning-Curve Targets

Pilot targets:

- 250 informative windows
- 500 windows
- 1,000 windows
- 2,000 windows
- 5,000+ windows

Final size should be justified by validation regret, held-out regret,
learning-curve saturation, label/policy diversity, scenario-family coverage,
and nontrivial regret headroom over the best fixed policy. Large near-tie
counts do not count as informative coverage.

The CPU-only pilot script is:

```bash
python scripts/build_selector_dataset_v2_pilot.py --output-dir results/selector_dataset_v2/pilot
```

It writes a small full-outcome dataset, `manifest.json`, `pilot_summary.json`,
and `workload_source_manifest.json`. It does not train a selector.

## Quality Gates

Before large-scale generation:

- leakage tests pass
- no request duplication/loss in targeted policy runs
- deterministic reproducibility tests pass
- at least three policies have meaningful wins, not tie-break wins
- no single policy dominates more than 85% of strongly discriminative windows
  unless reported as a scientific finding
- substantial nonzero regret headroom over the globally best fixed policy
- real-trace representation exists
- controlled-stress representation exists
- OOD split is defined

If gates fail, redesign scenario coverage before generating more data.
