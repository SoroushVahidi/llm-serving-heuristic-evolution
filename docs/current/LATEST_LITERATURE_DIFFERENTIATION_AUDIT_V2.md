# Latest Literature Differentiation Audit V2

Audit date: 2026-09-20

## Current literature boundary

The rewrite integrates ServeGen, LMetric, OpenTela, A Year in LLM Serving,
Libra, Llumnix, MorphServe, Strata, Mooncake, FastServe, Sarathi-Serve,
DistServe, SOLA, BurstGPT, Azure, and DynamoLLM.

ServeGen and the longitudinal Chutes trace study establish that realistic
production workload characterization is an important existing contribution.
OpenTela establishes an operational decentralized serving/control-plane system.
LMetric is the closest scheduling comparison: it combines KV-aware and
load-aware indicators and analyzes failure conditions. MorphServe and Strata
show pressure-aware model/cache adaptation; Mooncake, Libra, Llumnix, SOLA,
FastServe, Sarathi-Serve, DistServe, and DynamoLLM cover modern resource,
phase, preemption, routing, or cluster-control mechanisms.

The manuscript therefore makes no generic novelty claim about realistic
workloads, KV-aware scheduling, adaptive serving, or production deployment.
Its narrower distinction is the measurement chain:

`resource binding -> canonical executable disagreement P(D) ->
default-relative one-step causal headroom -> later predictability gate`.

The audit found no closest-work result that reports this exact canonical
action prevalence plus one-step default-relative intervention in LLM serving.
This is a scoped moderate-novelty claim, not a firstness claim over the
serving literature.

## Citation sources

- ServeGen: arXiv:2505.09999 / NSDI 2026.
- LMetric: arXiv:2603.15202.
- OpenTela: OSDI 2026, pp. 1821--1838.
- A Year in LLM Serving: arXiv:2608.13573.
- MorphServe: arXiv:2506.02006.
- Strata: arXiv:2508.18572.
- DynamoLLM: HPCA 2025, DOI 10.1109/HPCA61900.2025.00102.

The bibliography records these entries and the manuscript states the
differentiation without claiming that related systems ignore pressure,
production traces, or failure conditions.
