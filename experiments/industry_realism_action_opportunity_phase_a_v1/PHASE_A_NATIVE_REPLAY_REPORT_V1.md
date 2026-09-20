# Industry Realism Action Opportunity Phase A V1

Status: COMPLETE

Scientific scope: native/faithful replay support characterization only. This run did not execute causal labeling, selector training, synthetic stress, resource-pressure sweeps, or closed-loop learned scheduling.

## Frozen Inputs

- Design commit: `7457a6dc12dd18bd923b42da4952e38fb61582f7`
- Phase-A source/config freeze commit: `f394d73119d5d4039ee2e1dc38482071db2bd493`
- Workloads: `azure_2023_code`, `azure_2023_conv`, `burstgpt`
- Faithful windows: 20 per workload, 60 total
- Augmented windows used: no
- SBS policy: `kv_constrained_online`
- P6 portfolio: `full_prefill`, `chunked_prefill_small`, `estimated_service_time_first`, `weighted_fair_share`, `least_laxity_first`, `kv_constrained_online`
- Native GPU config: 1 GPU, 512 active sequences, 512 max batch tokens, 8,000,000 KV tokens
- Service config: `step_size=0.001`, prefill modeling enabled, `prefill_cost_per_token=1.0`, `step_token_budget=512`, decode/prefill contention enabled, `decode_first=false`

## Replay Semantics

The faithful public-trace view preserves request order, relative interarrival spacing, native prompt token counts, native output token counts, original burstiness within deterministic 200-request windows, and source window boundaries. Each window is rebased so first arrival is `0.0`. Missing prompt/output token counts are filled with `1` and clipped to a lower bound of `1`. The faithful simulator view assigns uniform priority, default class, and a broad synthetic SLO deadline; these fields are documented because some full-P6 policies can read them when queried on faithful states.

## Primary Results

Primary metric: `disagreement_rate = disagreement_states / total_SBS_decision_states`.

| Workload | Requests | SBS decisions | Disagreement states | Disagreement rate | Windows with disagreement |
|---|---:|---:|---:|---:|---:|
| Azure 2023 code | 4,000 | 99,992 | 0 | 0.000000 | 0/20 |
| Azure 2023 conversation | 4,000 | 461,985 | 0 | 0.000000 | 0/20 |
| BurstGPT | 4,000 | 440,461 | 0 | 0.000000 | 0/20 |

All five non-SBS P6 policy queries mapped to the same canonical action as SBS for every live SBS decision state.

## Action Collapse

| Workload | Structurally single-canonical-action states | Policy-ranking proxy difference with canonical collapse | True canonical disagreement |
|---|---:|---:|---:|
| Azure 2023 code | 99,992 | 194 | 0 |
| Azure 2023 conversation | 461,985 | 22 | 0 |
| BurstGPT | 440,461 | 462 | 0 |

The proxy-collapse column marks states where simple online ranking proxies would distinguish pending requests, but the actual canonical P6 action output still matched SBS. It is not causal headroom evidence.

## Resource Pressure

| Workload | Max active seqs | Mean active seqs | Max queue | Mean queue | Max KV util | Near/binding states |
|---|---:|---:|---:|---:|---:|---:|
| Azure 2023 code | 9 | 1.360 | 4 | 0.040 | 0.002716 | 0 |
| Azure 2023 conversation | 7 | 1.704 | 2 | 0.009 | 0.001728 | 0 |
| BurstGPT | 31 | 1.364 | 31 | 0.009 | 0.003802 | 0 |

No active-sequence capacity, KV-capacity, or token-budget binding state was observed under the frozen native replay configuration.

## Interpretation

Phase A establishes that these three locally available Tier-1 production traces, under the existing faithful native replay baseline, do not naturally expose canonical SBS-vs-P6 action opportunity. This is a support/prevalence null result, not evidence that alternative schedulers have no causal value in general.

The low pressure telemetry indicates the result is most consistent with effectively unconstrained or low-contention native replay, with canonical action collapse in the few states where simple ranking proxies differ. Phase B should therefore preserve the same request traces while varying system pressure axes one at a time.

## Phase B Plan

Designed but not executed:

- Arrival pressure multiplier: `0.5, 0.75, 1.0, 1.25, 1.5, 2.0`
- KV capacity fraction of native: `1.0, 0.5, 0.25, 0.125`
- Max active sequence fraction of native: `1.0, 0.5, 0.25, 0.125`

The Phase B grid preserves real request identity, ordering, prompt/output lengths, and window membership. It remains forbidden to use augmented windows, synthetic scenarios, selector training, or causal labeling during the support sweep.

## FGCS Readiness Checkpoint

- Scientific novelty: 12/20
- Industry realism: 11/20
- Technical depth: 8/20
- Experimental rigor: 12/20
- Practitioner value: 4/10
- Reproducibility/community value: 8/10
- Total: 55/100
- Contribution-strength confidence: 45%

Hard gates remain incomplete for systems-regime characterization and causal headroom. The smallest meaningful next phase is the preregistered Phase B one-axis pressure sweep over the same faithful traces.
