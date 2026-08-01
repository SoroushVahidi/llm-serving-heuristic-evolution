# Contextual Compositional Heuristics Roadmap

```yaml
canonical_branch: contextual-compositional-heuristics-20260731
current_phase: CC1
current_status: COMPLETE
next_action: do not start CC2; Query 5 should redesign or strengthen CC1 workload discriminativeness before any primitive refactor
roadmap_version: 1
```

Authoritative branch: `contextual-compositional-heuristics-20260731`

Starting base commit: `775147beec997b14039bbaa088d17630a32156cf`

Roadmap establishment commit: created in Query 2; verify with `git log -1`.

Current date: 2026-07-31

## Status Table

| Phase | Purpose | Status | Entry condition | Exit condition | Canonical evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| CC0 | Repository and evidence stabilization | COMPLETE | Query 1 branch established | Roadmap, decisions, navigation, issues, and checker exist | This roadmap; [branch marker](CONTEXTUAL_COMPOSITION_BRANCH.md); [Query 1 report](audits/contextual_composition_query1_sync_report_20260731.md); [Query 2 report](audits/contextual_composition_query2_roadmap_report_20260731.md) | Maintain links only |
| CC1 | Composition opportunity experiment | COMPLETE | CC0 complete | Composition opportunity gap measured with true simulator execution | Issue [#1](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1); [CC1 specification](experiments/cc1_composition_opportunity_spec.md); [Query 4 results](audits/contextual_composition_query4_cc1_results_20260731.md) | Negative gate: do not start CC2; redesign workload discriminativeness |
| CC2 | Canonical primitive interface | BLOCKED | CC1 decision gate passes or is explicitly revised | Representative policies reproduced from primitive configurations | Issue [#2](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/2) | Wait for CC1 |
| CC3 | Compositional DSL and verifier | BLOCKED | CC1 and CC2 gates pass | Verified deterministic composition programs pass all safety tests | Issue [#3](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/3) | Wait for CC2 |
| CC4 | Offline oracle composition dataset | BLOCKED | CC1-CC3 gates pass | Oracle dataset shows reproducible composition signal | Issue [#4](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/4) | Wait for CC3 |
| CC5 | Contextual composition predictor | BLOCKED | CC4 signal gate passes | Deployable predictor beats fixed, hard selector, and global composition with fallback | Issue [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5) | Wait for CC4 |
| CC6 | Dynamic adaptation and stability | PLANNED | CC5 deployable model gate passes | Adaptation improves changing regimes without instability | Issue [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6) | Wait for CC5 |
| CC7 | Counterexample-guided hardening | PLANNED | CC6 stable adaptation or explicit static-only scope | No critical supported-envelope failures remain | Issue [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6) | Wait for CC6 |
| CC8 | Real-trace and real-serving validation | PLANNED | CC7 hardening gate passes | Clean staged validation through real-serving evidence | Issue [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6) | Wait for CC7 |

Allowed status values: `COMPLETE`, `IN PROGRESS`, `NEXT`, `BLOCKED`,
`PLANNED`, `PAUSED`, `INVALIDATED`.

## Objective

Given causal workload context, predict a sparse, safe composition of reusable
scheduling primitives, compile that composition into a verified deterministic
heuristic, and execute it only when it is expected to improve over a robust
fallback.

This branch is scoped to the contextual-compositional heuristic research path.
It does not make this branch authoritative for unrelated historical work,
dataset staging, external baseline fidelity, or earlier selector-only phases.

## Scope Boundaries

In scope:

- offline composition-opportunity measurement;
- primitive interface design after evidence justifies it;
- verified deterministic composition;
- uncertainty, abstention, fallback, and switching stability;
- simulator, real-trace, and real-serving validation in that order.

Out of scope until gated:

- broad DSL redesign before CC1/CC2;
- broad policy refactoring before CC1;
- contextual predictor training before CC4;
- live hosted APIs without explicit opt-in and hard cost caps;
- real-vLLM claims before simulator and real-trace evidence is clean.

## Relationship To Existing Systems

Existing hard selector:

- The hard selector remains a required comparison baseline.
- CC1 must compare against the learned hard selector before claiming
  composition opportunity.
- A contextual composition model must later outperform best fixed, hard
  selector, and best global composition on held-out data.

JSON DSL and verifier:

- The current DSL/verifier is the safety foundation.
- CC3 may extend it only after CC1 and CC2 clarify composition semantics.
- Runtime compositions must compile deterministically and pass static/resource
  safety verification before execution.

LLM-generated heuristics:

- LLM-generated structures remain offline candidates only.
- No runtime scheduling path may require a live LLM call.
- Numerical constants proposed by an LLM are not accepted as optimal without
  search or evaluation.

Simulator, real traces, hosted APIs, and real-vLLM:

- Simulator execution is the first authority for CC1-CC7.
- Real-trace-derived simulation is required before real-serving claims.
- Hosted APIs require explicit opt-in and hard cost caps.
- Real-vLLM validation is staged late and cannot use known-confounded runs as
  clean evidence.

## Canonical Architecture

```text
Recent workload observations
          ↓
Causal feature extraction
          ↓
OOD / uncertainty assessment
          ↓
Contextual composition model
          ↓
Sparse weights and parameters over verified primitives
          ↓
DSL template instantiation
          ↓
Static and resource-safety verification
          ↓
Commitment / hysteresis controller
          ↓
Deterministic scheduler
          ↓
Outcome logging for offline retraining
```

Offline training:

- builds causal feature datasets;
- searches composition parameters only on training/development data;
- trains predictors only after oracle composition signal is established.

Offline composition search:

- executes candidate mixtures in the simulator;
- records true outcomes, regret surfaces, sparsity, and complementarity;
- never estimates mixture performance by arithmetic over policy reward vectors
  when actions could interact.

Online inference:

- consumes recent causal observations;
- predicts sparse weights or abstains;
- instantiates a verified deterministic composition;
- applies fallback and commitment rules.

Optional future online adaptation:

- may update composition per workload window or stable regime only;
- must use minimum commitment, hysteresis, and rollback/fallback;
- must not use uncontrolled per-step switching.

No live LLM call is required during runtime scheduling.

## Phase CC0 - Repository And Evidence Stabilization

Status after Query 2: COMPLETE.

Completed:

- authoritative branch established:
  `contextual-compositional-heuristics-20260731`;
- local audit preserved and synchronized with an addendum:
  [local branch audit](audits/local_branch_compositional_path_audit_20260731.md);
- Query 1 synchronization report committed:
  [Query 1 report](audits/contextual_composition_query1_sync_report_20260731.md);
- this roadmap established;
- decision log established:
  [decision log](contextual_composition_decisions.md);
- start-here path established:
  [start here](START_HERE_CONTEXTUAL_COMPOSITION.md);
- historical/current distinction documented in navigation;
- GitHub issue structure established:
  [#1](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1),
  [#2](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/2),
  [#3](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/3),
  [#4](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/4),
  [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5),
  [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6).

## Phase CC1 - Composition Opportunity Experiment

Goal:

Determine whether true policy or primitive composition can outperform
per-window hard selection.

Approved specification:

- [CC1 composition opportunity specification](experiments/cc1_composition_opportunity_spec.md)

Status after Query 4: COMPLETE with `STOP_OR_REDESIGN`.

Query 4 implemented and ran the approved true simulator-executed weighted Borda
rank-aggregation experiment. The full local run executed 200 simulator runs
over 10 windows, including two real-trace-derived OOD windows. The measured
composition-opportunity gap was `0.0`; all four evaluation windows were
near-ties under the fixed-policy margin threshold; best global mixture was the
one-hot WSP mixture; oracle mixture matched oracle fixed. The CC1 decision gate
did not pass, and CC2 remains blocked.

Canonical result report:

- [Query 4 CC1 results](audits/contextual_composition_query4_cc1_results_20260731.md)

Required work:

- define one minimal compatible composition interface;
- expose normalized request ranks or scores for a small representative policy
  subset;
- implement true simulator-executed weighted composition;
- compare:
  - best fixed policy;
  - learned hard selector;
  - oracle best fixed policy per window;
  - best global mixture;
  - oracle best mixture per window;
- calculate the composition opportunity gap:

```text
oracle per-window mixture performance
- oracle per-window fixed-policy performance
```

Required decision gate:

Proceed to full primitive refactoring only if composition provides a meaningful,
reproducible gain on non-near-tie windows or strategically important regimes.

Initial revisable thresholds:

- aggregate ANWG gain of at least `0.005`; or
- regime-specific gain of at least `0.01` on a known failure regime;
- without materially reducing completion fraction or violating safety
  constraints.

Canonical issue: [#1](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1).

## Phase CC2 - Canonical Primitive Interface

Goal:

Define reusable and semantically compatible scheduling primitives.

Required primitive families:

- deadline urgency;
- laxity;
- prompt length;
- predicted output length;
- estimated service time;
- priority;
- queue age;
- KV pressure;
- GPU projected load;
- feasibility;
- admission risk;
- prefill pressure;
- token budget;
- fairness or starvation prevention.

Separate interfaces:

- request ranking;
- admission gating;
- GPU placement;
- batching and token-budget parameters.

Required decision gate:

Existing representative policies must be reproducible approximately or exactly
from primitive configurations, with equivalence tests.

Canonical issue: [#2](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/2).

## Phase CC3 - Compositional DSL And Verifier

Goal:

Extend the DSL to express named, bounded, parameterized compositions.

Required constructs:

- named primitive references;
- weighted sums;
- sparse top-k mixtures;
- conditional branches;
- admission gates;
- placement scores;
- externally supplied bounded parameters;
- deterministic tie-breaking;
- explicit safe fallback.

Verifier requirements:

- bounded coefficients;
- bounded number of active primitives;
- normalized-feature requirements;
- no future or oracle information;
- execution-cost limits;
- no contradictory gates;
- guaranteed fallback;
- stable serialization and hashing.

Required decision gate:

All compositions must compile deterministically and pass verifier,
equivalence, property, and adversarial tests.

Canonical issue: [#3](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/3).

## Phase CC4 - Offline Oracle Composition Dataset

Goal:

For each training window, search for high-quality composition parameters
through true simulator execution.

Required comparisons:

- global composition;
- per-regime composition;
- per-window oracle composition;
- hard-selector oracle;
- best fixed policy.

Required outputs:

- optimal or near-optimal weights;
- regret surfaces;
- sparsity;
- stability across seeds;
- component complementarity;
- uncertainty or ambiguity for near-tie windows.

Required decision gate:

The oracle composition dataset must show sufficient signal and reproducibility
to justify contextual prediction.

Canonical issue: [#4](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/4).

## Phase CC5 - Contextual Composition Predictor

Goal:

Predict composition parameters from causal workload features.

Required baseline models:

- linear or ridge;
- decision tree;
- random forest;
- gradient boosting if already available;
- per-parameter regression;
- direct regret model;
- ranking-based or pairwise model;
- sparse gating model.

Required safeguards:

- held-out workload families;
- unseen seeds;
- no reward leakage;
- uncertainty estimates;
- OOD detection;
- abstention;
- robust fallback.

Required decision gate:

The deployable contextual composition model must outperform:

- best fixed policy;
- hard selector;
- best global composition;

on held-out data, with uncertainty-aware safety behavior.

Canonical issue: [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5).

## Phase CC6 - Dynamic Adaptation And Stability

Goal:

Allow controlled per-window updates rather than per-step uncontrolled
switching.

Required mechanisms:

- minimum commitment horizon;
- hysteresis margin;
- exponential smoothing;
- switch-cost accounting;
- rollback or fallback;
- regime-transition tests.

Required decision gate:

Dynamic adaptation must improve changing-regime workloads without instability,
excessive switching, or meaningful scheduling overhead.

Canonical issue: [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6).

## Phase CC7 - Counterexample-Guided Hardening

Goal:

Use adversarial and failure-mined workloads to repair compositions and
predictors.

Required workload families:

- long-prompt mixed-tight-SLO;
- extreme burstiness;
- KV saturation;
- high prediction noise;
- priority inversion;
- abrupt regime shifts;
- starvation cases;
- infeasible workloads;
- selective-admission traps.

Required decision gate:

No unresolved critical safety or correctness failure remains in the supported
operating envelope.

Canonical issue: [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6).

## Phase CC8 - Real-Trace And Real-Serving Validation

Goal:

Validate in increasingly realistic settings.

Required order:

1. synthetic held-out;
2. held-out real-trace-derived simulation;
3. shadow real-vLLM comparison;
4. controlled active real-vLLM run;
5. hosted-provider experiments only where scientifically appropriate.

Required decision gate:

Do not make production or general superiority claims without clean,
unconfounded real-serving evidence.

Canonical issue: [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6).

## Research Invariants

1. Primary objective is arrival-normalized weighted goodput unless a later
   documented decision changes it.
2. Completion fraction and number of arrivals must always be reported.
3. Completed-request-only weighted goodput cannot be used as the sole
   optimization objective.
4. Real-trace workloads cannot enter training splits used for their own
   reported evaluation.
5. Causal deployable features must be separated from oracle or diagnostic
   fields.
6. LLM-generated structures must be verified before execution.
7. Numerical constants proposed by an LLM are not accepted as optimal without
   search or evaluation.
8. Oracle-assisted models must be clearly separated from deployable models.
9. Near-tie windows must be reported and not allowed to dominate
   interpretations.
10. New methods must be compared against the best fixed baseline, hard
    selector, and relevant global composition.
11. Real-vLLM results with known confounding bugs cannot be used as clean
    evidence.
12. Live APIs require explicit opt-in and hard cost caps.
13. Runtime scheduling must remain deterministic for a fixed context and
    composition.
14. Every phase must update roadmap status and canonical evidence links.

## Roadmap Update Protocol

Every future implementation query must:

1. verify branch and upstream;
2. read the roadmap before acting;
3. identify the current `NEXT` phase;
4. avoid work in blocked phases unless explicitly authorized;
5. update:
   - phase status;
   - evidence links;
   - decisions;
   - unresolved risks;
   - next action;
6. add or update an audit report;
7. run relevant tests;
8. commit and push;
9. leave the branch clean and synchronized.

When a phase gate changes, update this roadmap and the related GitHub issue in
the same query if possible.

## Historical And Caveated Evidence

Historical docs remain provenance, not trash. Do not delete them merely because
the current branch supersedes their "next step" language.

Current contextual-composition readers should treat these as the canonical
branch-local entry points:

- [Start Here](START_HERE_CONTEXTUAL_COMPOSITION.md)
- this roadmap
- [Decision Log](contextual_composition_decisions.md)
- [Branch Marker](CONTEXTUAL_COMPOSITION_BRANCH.md)
- [Local Branch Audit](audits/local_branch_compositional_path_audit_20260731.md)
- [Query 2 Roadmap Report](audits/contextual_composition_query2_roadmap_report_20260731.md)

Historical project status remains under `docs/current/` and the broader
historical index remains [docs/README.md](README.md).

Safe-claim guidance remains [result_claims.md](result_claims.md). Contextual
composition claims must follow the invariants above and must distinguish
historical simulator claims, corrected-objective claims, real-trace claims, and
real-serving claims.
