# Project Status

**Canonical current-state document.** Historical reports remain valuable, but
this file is the source of truth for deciding what to do next.

**Status date:** 2026-07-22
**Current integration branch:** `wulver-final-integration-20260721`
**Current integration commit before this polish pass:** `e8bd759b6cdaa8a05096b0ceeb1c7684cfa07302`

## Project Objective

The long-term goal is not just a 27-way policy selector. The intended
contribution is a state-conditioned mechanism that estimates the suitability or
module credit of scheduling algorithms/components for a serving scenario, then
uses that evidence to combine or synthesize a new deployable scheduling policy
and compare it fairly against fixed, adaptive, and external baselines.

The current repository supports:

- a deterministic LLM-serving simulator;
- 27 deployable V2 scheduling policies;
- leakage-safe full-information policy-vector datasets;
- selector/suitability models;
- typed rank composition and component-wise prototypes;
- a scheduler genome and structural-synthesis harness;
- module-intervention/credit data and analysis infrastructure.

## What We Know Now

### Established

- Policy Library V2 contains **27 deployable policies**: the historical
  20-policy library plus 7 simulator-compatible V2 policies.
- Synthetic/frontier V2 expansion was real but modest.
- The real-trace OOD V2 library audit showed strong oracle-envelope expansion:
  V1 oracle ANWG `0.251666`, V2 oracle ANWG `0.260571`, absolute gain
  `0.008904`, relative gain about `3.54%`, 95% CI `[0.008191, 0.009646]`.
- The 27-policy selector/regret benchmark completed with
  `SELECTOR_V2_27_STATUS = STRONG` in its own report, but that status needs
  the caveat below: the learned top-1 selector still did not meaningfully
  capture the V1-to-V2 oracle-envelope gain on held-out OOD.
- The selector beats some fixed baselines on some splits and produces useful
  suitability/ranking signals, but it remains substantially below the oracle.
- Single-module interventions produced sparse but real positive transfer. Some
  children beat both parents, and a small number expanded the 27-policy
  envelope.
- Synthetic SLO/deadline augmentation produced useful incremental diversity:
  EDF, SCORPIO-style SLO/admission behavior, laxity, and admission-control
  competence became more visible under tight and heterogeneous SLO pressure.

### Negative But Valuable Findings

- Naive/native rank or component-wise composition did not beat discrete
  selection or expand the frontier in the Wulver pilot.
- Pairwise module intervention did not expand the 27-policy envelope beyond the
  single-module signal.
- Module-credit learning remains weak/generalization-limited. The overnight
  10,000-trial search improved modeling but did not make unrestricted
  structural synthesis ready.
- SwissAI staging showed novel KV/cache/reuse feature-space coverage, but the
  512-window x 27-policy evaluation saturated the ANWG objective and produced
  **zero strict V2 marginal oracle gain**. FIFO/simple behavior dominated near
  ties. The failed SwissAI final-report stage was a reporting bug around
  `kv_proxy_p95`; the complete policy-vector matrix exists and is usable.
- TraceLab staging showed high long-context, agentic, prefix-reuse, and
  tool-use novelty, but the 512-window x 27-policy evaluation also saturated
  ANWG and produced **zero strict V2 marginal oracle gain**.
- Raw workload novelty alone does not guarantee policy separation under the
  current simulator/objective.

### Current Bottleneck

The primary bottleneck is no longer lack of datasets or generic selector model
choice. The current simulator/objective often fails to translate important
workload differences, especially KV/cache reuse, long context, and
prefill/decode structure, into sufficiently different resource pressure and
policy rewards. This causes ANWG saturation and weak policy separation, making
both selector learning and future combiner/module-credit learning unreliable.

The simulator discriminative-power audit at:

`/mmfs1/project/ikoutis/sv96/llmserveopt-data/simulator_discriminative_audit_20260722T223236Z`

found:

- TraceLab reward saturation near 1: mean ANWG `0.998822`, p5 `1.000000`,
  fraction ANWG > `0.999` equal to `0.988`.
- SwissAI strong saturation: mean ANWG `0.991726`, p5 `0.979208`, fraction
  ANWG > `0.999` equal to `0.361`.
- V2 real-OOD was not saturated: mean ANWG `0.169498`, no rewards > `0.99`,
  and many meaningful policy margins.
- V2 real-OOD had 21 distinct deterministic winners and effective winner
  classes about `10.12`.
- TraceLab had effective winner classes about `1.12`; SwissAI had `1.00`.
- `KV_CACHE_COUPLING_VERDICT = WEAK_DIRECT_COUPLING`.
- `PREFILL_DECODE_COUPLING_VERDICT = PARTIAL_AND_WEAK_UNDER_CURRENT_WORKLOADS`.
- `SLO_DEADLINE_SENSITIVITY_VERDICT = STRONG_WHEN_TIGHT_HETEROGENEOUS_WEAK_WHEN_NEUTRAL`.
- `COMBINER_TRAINING_SIGNAL = WEAK`.
- `COMBINER_EVALUATION_READINESS = NEEDS_SIMULATOR_FIX`.

## Current Interpretation

The V2 action space is valuable: real-OOD oracle evidence supports that.
However, the simulator/reward system is not yet consistently discriminative
enough to train or fairly evaluate a state-conditioned combiner/synthesizer
across the new workload families. Additional generic datasets, broader selector
model sweeps, or unrestricted synthesis would mostly add data volume without
fixing the weak causal coupling between workload features and simulated
resource pressure.

## Authoritative Current Decisions

- Do not launch more generic dataset collection as the next major step.
- Do not retrain the 27-policy selector as the next major step.
- Do not launch broad structural synthesis or unrestricted module recombination.
- Treat SwissAI and TraceLab as important negative evidence about the simulator
  and objective, not as failed or useless datasets.
- Treat SLO augmentation as synthetic training/regime-probing evidence, not as
  natural real-OOD evidence.
- Prioritize bounded simulator calibration and discriminative-power validation.

## Next Experiment

The next major task should be a bounded simulator calibration and
pressure-validation pass:

- strengthen and validate KV/cache coupling;
- model prefix reuse effects on prefill/service cost where scientifically
  justified;
- strengthen KV occupancy and resource-pressure semantics;
- validate prefill/decode contention;
- validate capacity/overload pressure;
- calibrate SLO feasibility effects;
- audit ANWG ceiling behavior and decide whether an auxiliary objective is
  needed for saturated regimes.

Only after that should the project rerun bounded policy-vector subsets and then
retrain suitability/module-credit models.

## Repository Organization Status

Use `docs/current/README.md` as the navigation entry point. Current docs are
organized around:

- repository architecture;
- selector status;
- policy library status;
- frontier and experiment evidence;
- composition/synthesis implementation status;
- roadmap and gap analysis;
- protected Wulver experiment roots.

Historical reports in `docs/audits/`, `docs/`, and `experiments/` remain
scientific provenance. Do not delete them merely because later results changed
the interpretation.
