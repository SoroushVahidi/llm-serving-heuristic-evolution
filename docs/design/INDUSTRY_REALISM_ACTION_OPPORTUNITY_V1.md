# INDUSTRY_REALISM_ACTION_OPPORTUNITY_V1

Status: planning/audit design, not a preregistered executable protocol.
Created for the post-V2-confirmation pivot on 2026-09-19.

## Scientific Record Preserved

The SBS Override V2 selector remains frozen as a completed negative
confirmatory result. Its development crossfit signal was positive, but the
one-shot fresh confirmation produced a negative primary mean gain and a
scenario-clustered confidence interval crossing zero. V2 must not be presented
as a successful generalizing selector.

This phase does not create V3, train another selector, retune V2, or attempt to
rescue the failed confirmation. It asks a more upstream question:

Under realistic LLM-serving workloads and resource constraints, how often does
a strong default scheduler actually have meaningful decision-level
alternatives, how much causal headroom exists when those alternatives occur,
and under what operating conditions does scheduler adaptivity become
practically relevant?

## Core Decomposition

Let `D` be a live state where SBS and at least one alternative scheduler induce
different canonical actions. Let `B` mean at least one available non-SBS action
has positive SBS-relative one-step causal advantage. Let `G` be SBS-relative
causal gain.

The study separates four questions:

1. Opportunity prevalence: `P(D)`.
2. Causal headroom: `P(B | D)` and `E[max_a A_SBS(s,a) | D]`.
3. Predictability: whether useful alternatives can be predicted from online
   information. The failed V2 fresh confirmation is evidence that this is
   nontrivial. No new predictor is trained in this phase.
4. Practical system value: whether the headroom is large, frequent, stable,
   and broadly distributed enough to justify serving-system complexity.

## Evidence Taxonomy

TIER 1 -- NATURAL PRODUCTION TRACE: directly derived from real serving traffic
with request-level structure. These can support prevalence claims for similar
serving traffic, subject to simulator fidelity.

TIER 2 -- PRODUCTION-DERIVED / RECONSTRUCTED: empirical production
distributions or published behavior without raw request-level traffic. These
can support plausibility and robustness checks, not primary prevalence claims.

TIER 3 -- CONTROLLED TRANSFORMATION OF REAL TRACE: real trace structure
preserved while system environment changes, such as arrival-rate scaling,
KV-capacity reduction, GPU-count/capacity change, or active-sequence limits.
These can support operating-regime transition claims if every pressure point is
reported, including zero-support regimes.

TIER 4 -- FULLY SYNTHETIC / GENERATED: no direct request-level production
provenance. These are mechanism probes, not primary industry-prevalence
evidence.

Primary paper evidence should prioritize Tier 1 native replay and Tier 3
system-pressure-transformed replay. Tier 4 can explain mechanisms, but not
stand alone as industry realism.

## Primary Characterization Question

How does canonical SBS-vs-portfolio action-disagreement prevalence change as
production-derived workloads move from native replay into progressively
resource-constrained serving regimes?

Primary outcome:

```text
disagreement_rate = canonical_disagreement_states / total_SBS_decision_states
```

This is descriptive characterization, not selector evaluation and not
hypothesis testing.

## Workload Classes

Assign a class only when supported by local trace provenance:

- Azure 2023 code: code.
- Azure 2023 conversation: ordinary chat / conversation.
- BurstGPT: mixed public serving traffic with model identifier available.
- Azure 2024: production LLM trace candidate; locally referenced in support
  artifacts but raw local availability must be reconfirmed before Phase A.
- Bailian/Qwen: conversational/multi-turn Qwen service trace when raw JSONL is
  available.
- Mooncake/Kimi: conversation and tool-agent trace candidate with prefix-block
  hashes, not locally acquired in the current corpus.
- AgentPerfBench: real-system aggregate validation source, not a per-request
  workload trace.
- joint240 generator-derived scenarios: Tier 4 mechanism corpus.
- Family-A and related controlled stress traces: Tier 4 mechanism corpora.

Do not fabricate agentic, reasoning, multimodal, or prefix-cache-heavy classes
for sources that lack the relevant fields locally.

## Null Result Reassessment

Previous natural OOD support scans should not be read as selector failure. They
mostly answer a prior support question: did the replayed workload/regime expose
canonical SBS-vs-P6 action alternatives at all?

Observed local evidence:

- Azure 2024 support scan: 291,211 decision states, zero disagreements,
  choice-state rate 0.0116, binding-capacity choice rate 0.0, mean KV
  utilization 0.192. Interpretation: `REPLAY_EFFECTIVELY_UNCONSTRAINED`.
- Bailian/Qwen support scan: 503,059 decision states, zero disagreements,
  choice-state rate 0.000565, binding-capacity choice rate 0.0, mean KV
  utilization 0.0687. Interpretation: `WORKLOAD_LIKELY_NON_CONTENTIOUS`.
- joint240 generated holdout: 144,310 decision states, 2,862 disagreements,
  disagreement rate 0.0198, 78/80 scenarios with disagreement, binding-capacity
  choice rate 0.643. Interpretation: generated pressure regime, not natural
  production prevalence.
- Azure 2023 and BurstGPT public replay smoke/native artifacts showed little
  or no policy separation in native replay; load-scaling v1 was invalidated by
  a simulation-horizon bug, while v2 fixed that bug and passed the lambda=1
  gate. Interpretation for native replay: `REPLAY_EFFECTIVELY_UNCONSTRAINED`
  until corrected support scans are run.

Zero disagreement is scientifically meaningful: it may indicate SBS is usually
sufficient in that workload/regime, or that the replay lacks the pressure
conditions under which scheduler choice matters. It should be reported, not
discarded.

## Operating Pressure Axes

Prefer axes that change the serving environment while preserving the empirical
request trace:

- Arrival-rate multiplier: supported by `public_replay_load_scaling_v2`; fixed
  grid currently exists at 1, 2, 4, 8, 16, 32, 64, 128. A smaller industry
  grid such as 0.5, 0.75, 1, 1.25, 1.5, 2 can be used only if preregistered
  before execution.
- KV capacity: supported structurally through `GPUConfig.max_kv_tokens`;
  recommended as a Phase-B axis after smoke validation.
- Maximum active sequences: supported through `GPUConfig.max_active_sequences`;
  recommended because it is physically interpretable and operator-controlled.
- Prefill/decode capacity: supported through `ServiceModel.step_token_budget`,
  `max_prefill_chunk_tokens`, and decode-prefill contention flags; use only
  with explicit simulator-fidelity caveats.
- GPU/server count: supported by multiple `GPUConfig` entries, but policy
  semantics should be smoke-tested before inclusion.
- Context-window limits: not a clean replay axis unless request filtering or
  truncation is explicitly treated as workload transformation.
- Prefix reuse/cache availability: data fields exist in Bailian and Mooncake
  adapters, but current SBS-vs-P6 replay does not yet model prefix-cache reuse
  as a primary online scheduling state. Treat as a blocker for prefix-heavy
  claims.
- SLO limits: available only as project-synthesized overlays for most public
  traces. Keep out of the primary opportunity-prevalence endpoint unless
  clearly labeled controlled annotation.

## Trace Transformation Types

NATIVE_TRACE: the request identities, order, timing, token lengths, sessions,
and available metadata are replayed as provided or as already canonicalized.

SYSTEM_PRESSURE_TRANSFORMED_TRACE: request identities, order, token lengths,
session structure, and content/provenance remain fixed while the serving
environment changes, such as arrival compression, KV capacity, active-sequence
cap, or GPU count.

WORKLOAD_TRANSFORMED_TRACE: prompt/output lengths, request mix, session
structure, or prefix reuse are altered. These are secondary and must be labeled
as such.

FULLY_SYNTHETIC: all request structure is generated. These are mechanism
exploration only.

## Phased Matrix

Phase A -- Native replay. For every usable Tier-1 trace, run unchanged workload
under a justified baseline serving configuration and measure support only:
SBS decision states, canonical disagreement states, disagreement rate, pressure
telemetry, and zero-support evidence.

Phase B -- One-axis resource-pressure sweep. Preserve each trace and vary one
environment axis at a time. Recommended order: arrival pressure, active
sequence cap, KV capacity, then prefill/decode token budget. Report every grid
point, including zero-support points.

Phase C -- Joint realistic regimes. Use a small predetermined sparse grid over
the axes that Phase A/B design identifies as operator-relevant. Do not select
combinations because early outcomes look favorable.

Phase D -- Causal headroom. After support mapping, run expensive SBS-relative
one-step counterfactual labeling on regimes chosen by frozen coverage criteria:
native if nonzero support, first-emergence pressure, mid-pressure, high-pressure,
and a zero/near-zero support control where feasible.

Phase E -- Predictability. Only after support and headroom are characterized,
consider a learned online predictor. This phase is explicitly out of scope for
this design task.

## Metrics

Support metrics:

- total SBS decision states;
- canonical disagreement states;
- disagreement rate;
- distinct alternative canonical actions;
- fraction of scenarios/windows with any disagreement.

Headroom metrics, only where one-step causal labeling is feasible:

- fraction of disagreement states with any beneficial alternative;
- oracle best one-step gain;
- mean and median best SBS-relative gain;
- harmful-alternative prevalence;
- zero-effect prevalence;
- distribution of action advantages.

Concentration metrics:

- disagreement concentration across windows/scenarios;
- top 1%, 5%, and 10% workload segments contributing disagreement;
- headroom concentration;
- relationship to pressure telemetry.

Context metrics:

- queue length;
- active sequence count;
- KV utilization;
- arrival intensity;
- prompt/output length statistics;
- prefix/cache reuse when genuinely present and modeled;
- load-regime labels.

## Anti-Cherry-Picking Rules

Every preregistered workload, native condition, pressure point, and support
scan must be reported. Zero-support and negative/null headroom regimes remain
in the result tables. Phase-D labeling regimes must be chosen by frozen
coverage rules, not by favorable observed gains. Secondary diagnostics cannot
be used to hide a null primary opportunity map.

## Practitioner Output Target

The final output should be an operating-regime map:

```text
workload_class | native_disagreement | pressure_threshold | beneficial_headroom | practical_implication
```

The aim is to tell an inference-serving engineer when the default scheduler is
sufficient, when scheduler choice matters, what observable pressure predicts
that transition, and whether adaptive complexity is justified.

## Distinction From The JSC Paper

The separate Journal of Supercomputing line asks which scheduler ranks best
across scenarios and how portable scheduler rankings are. This FGCS-oriented
line asks when a strong default has different feasible actions available, how
often the default is locally causally suboptimal, which workload/system
conditions create that headroom, and whether intervention is worth its
complexity. The failed V2 confirmation is not an embarrassment to bury; it is
evidence that prediction is hard and that opportunity/headroom must be mapped
before promising adaptive scheduling gains.

## Revised Paper Thesis

Working hypothesis, not a conclusion:

Strong default LLM schedulers exhibit highly regime-dependent local headroom.
Under many production-derived workloads, scheduler alternatives collapse to
the same canonical action; actionable complementarity emerges primarily under
identifiable resource-pressure regimes, and apparent development-time
exploitability need not transfer to fresh workloads.

Neutral title candidates:

- When Does Scheduler Choice Matter in LLM Serving?
- Action Opportunity Under Realistic LLM Serving Pressure
- Mapping Scheduler Headroom in Production-Derived LLM Workloads
- From Scheduler Rankings to Actionable Serving Regimes
- Measuring Local Scheduler Opportunity in LLM Inference Systems

## Readiness

This design is ready for a bounded Phase-A implementation task if that task
only freezes native-replay support scanning and smoke validation. It is not
ready for broad sweeps, causal labeling, selector training, or closed-loop
learning.
