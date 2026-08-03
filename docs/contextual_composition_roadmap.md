# Contextual Compositional Heuristics Roadmap

```yaml
canonical_branch: contextual-compositional-heuristics-20260731
current_phase: CC5
current_status: IN PROGRESS
next_action: the CC4b/CC5 retry completed with verdict REGIME_SPECIFIC_ONLY (predictor beats best fixed policy and is competitive with the hard selector on 76 held-out windows, but does not clearly beat best_global_composition); CC6 remains not queued; next step is addressing the uncertainty-method gap (no LOWO-CV-selected model has supported ensemble uncertainty across two retries) then a per-regime regret breakdown, per the CC4b/CC5 retry report section 10
roadmap_version: 7
```

Authoritative branch: `contextual-compositional-heuristics-20260731`

Starting base commit: `775147beec997b14039bbaa088d17630a32156cf`

Roadmap establishment commit: created in Query 2; verify with `git log -1`.

Current date: 2026-07-31

## Status Table

| Phase | Purpose | Status | Entry condition | Exit condition | Canonical evidence | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| CC0 | Repository and evidence stabilization | COMPLETE | Query 1 branch established | Roadmap, decisions, navigation, issues, and checker exist | This roadmap; [branch marker](CONTEXTUAL_COMPOSITION_BRANCH.md); [Query 1 report](audits/contextual_composition_query1_sync_report_20260731.md); [Query 2 report](audits/contextual_composition_query2_roadmap_report_20260731.md) | Maintain links only |
| CC1 | Composition opportunity experiment | COMPLETE | CC0 complete | Composition opportunity gap measured with true simulator execution | Issue [#1](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1); [CC1 specification](experiments/cc1_composition_opportunity_spec.md); [Query 4 results](audits/contextual_composition_query4_cc1_results_20260731.md); [Query 5 discriminativeness review](audits/contextual_composition_query5_discriminativeness_review_20260731.md) | Complete; CC1b gate passed |
| CC2 | Canonical primitive interface | COMPLETE | CC1b decision gate passed | Representative policies reproduced from primitive configurations | Issue [#2](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/2); [architecture doc](architecture/contextual_composition_primitives.md); [CC2 primitive interface report](audits/contextual_composition_cc2_primitive_interface_report_20260802.md) | Complete; CC2 equivalence gate passed (6/7 EXACT, 1/7 documented APPROXIMATE) |
| CC3 | Compositional DSL and verifier | COMPLETE | CC1 and CC2 gates pass | Verified deterministic composition programs pass all safety tests | Issue [#3](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/3); [architecture doc](architecture/contextual_composition_dsl.md); [CC3 DSL/verifier report](audits/contextual_composition_cc3_dsl_verifier_report_20260803.md) | Complete; CC3 exit gate passed (8/8 required constructs, 447 focused+regression tests, legacy compatibility preserved). CC4 remains BLOCKED pending explicit authorization. |
| CC4 | Offline oracle composition dataset | COMPLETE | CC1-CC3 gates pass | Oracle dataset shows reproducible composition signal | Issue [#4](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/4); [CC4 oracle dataset report](audits/contextual_composition_cc4_oracle_dataset_report_20260803.md) | Complete; CC4 exit gate passed (12 windows/34 candidates/408 executions, 0 rejected, reproducible+resumable, 66.7% evaluation-window composition-oracle gain, completion constraints hold on all windows). CC5 remains queued pending explicit authorization. |
| CC5 | Contextual composition predictor | IN PROGRESS | CC4 signal gate passes | Deployable predictor beats fixed, hard selector, and global composition with fallback | Issue [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5); [CC5 predictor report](audits/contextual_composition_cc5_predictor_report_20260803.md) (first attempt); [CC4b/CC5 retry report](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md) (completed) | First attempt: verdict `INCONCLUSIVE` at n=6 held-out windows. **Retry complete** (76 held-out windows, quality gates passed): verdict `REGIME_SPECIFIC_ONLY` -- predictor clearly beats best fixed policy (0.4006 vs 0.3895 ANWG) and is competitive with the hard selector (0.3938), but does not clearly beat `best_global_composition` (0.4025, within the bootstrap CI overlap). Exit gate NOT fully passed; CC6 not queued. Exact remaining task: address the uncertainty-method gap (no LOWO-CV-selected model across two retries has supported real ensemble uncertainty), then a per-regime regret breakdown -- see the retry report section 10. |
| CC6 | Dynamic adaptation and stability | BLOCKED | CC5 deployable model gate passes | Adaptation improves changing regimes without instability | Issue [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6) | Blocked on the CC5 retry decision gate (not yet resolved) |
| CC7 | Counterexample-guided hardening | BLOCKED | CC6 stable adaptation or explicit static-only scope | No critical supported-envelope failures remain | Issue [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6) | Blocked on CC6 |
| CC8 | Real-trace and real-serving validation | BLOCKED | CC7 hardening gate passes | Clean staged validation through real-serving evidence | Issue [#6](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/6) | Blocked on CC7 |

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

Status after Query 5: COMPLETE with CC1b `PROCEED`.

Query 5 diagnosed the Query 4 result as nondiscriminative rather than a
mixture-accounting bug: CC1 windows were under capacity or had SLO slack much
looser than observed latencies, so oracle fixed and oracle mixture both
reached ANWG `1.0` on every evaluation window. Query 5 added and ran a compact
CC1b discriminative suite using true simulator-executed weighted Borda
composition with tighter SLOs, prefill contention enabled, a step-`0.25`
top-2 weight grid, and a fixed-policy-spread gate before mixture evaluation.
The CC1b full local run executed 440 simulator runs over 11 windows. On the
four held-out evaluation windows, all were non-near-ties; the non-near-tie
composition-opportunity gap was `0.0167735`, best regime gain was `0.05`, and
completion impact was `0.0`. The CC1b decision gate passed, so CC2 is now
`NEXT`.

Canonical discriminativeness report:

- [Query 5 discriminativeness review](audits/contextual_composition_query5_discriminativeness_review_20260731.md)

Pause and resume records:

- [Query 6 pause checkpoint](audits/contextual_composition_pause_checkpoint_20260731.md)
- [Resume guide](RESUME_CONTEXTUAL_COMPOSITION.md)
- [Query 6 pause report](audits/contextual_composition_query6_pause_report_20260731.md)
- [Query 7 final pause-readiness report](audits/contextual_composition_query7_final_pause_readiness_20260731.md)

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

Status after Query 8 (this implementation): `COMPLETE`.

Query 8 implemented `src/llmserveopt/policies/primitives.py` (a 28-entry
canonical registry across the five required families: RANKING, ADMISSION,
PLACEMENT, BATCHING, RESOURCE_GUARD) and
`src/llmserveopt/policies/primitive_reconstructions.py` (seven
representative-policy reconstructions built only from those primitives).
The equivalence gate passed: `fifo`, `edf`,
`weighted_shortest_processing`, `estimated_service_time_first`,
`best_fit` (placement-oriented), and `admission_control`
(admission-oriented) reproduce their originals EXACTLY across synthetic
states, randomized fuzz, and full simulator-trace runs;
`scorpio_style_slo_guard` reproduces APPROXIMATELY (0 observed
mismatches across all tested fixtures, documented as approximate per the
CC2 exit-gate instruction rather than claimed as a formal guarantee).
107 new tests were added (42 registry/typed-behavior tests, 65
equivalence tests); the full existing test suite continues to pass. The
DSL (`src/llmserveopt/heuristics/`) was not modified.

Canonical evidence:

- [Architecture: CC2 canonical scheduling primitive interface](architecture/contextual_composition_primitives.md)
- [CC2 primitive interface report](audits/contextual_composition_cc2_primitive_interface_report_20260802.md)

The first CC2 implementation task after the pause was:

Define the canonical primitive interface for ranking, admission, placement,
batching, and resource guards, then add representative-policy equivalence tests.
Do not extend the DSL yet.

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

**Gate result (2026-08-03): PASSED.** See the
[CC3 DSL/verifier report](audits/contextual_composition_cc3_dsl_verifier_report_20260803.md)
for the full construct-by-construct evidence, backward-compatibility
findings, and the two documented (non-blocking) unresolved risks. CC4 is not
begun by this same query even though the gate passed; a separate query must
explicitly authorize it after reading that report.

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

**Gate result (2026-08-03): PASSED.** See the
[CC4 oracle dataset report](audits/contextual_composition_cc4_oracle_dataset_report_20260803.md)
for full search-design, quality-statistic, and limitations evidence. CC5 is
not begun by this same query even though the gate passed; a separate query
must explicitly authorize it after reading that report.

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

**Gate result (2026-08-03): NOT PASSED. Verdict: `INCONCLUSIVE`.** See the
[CC5 predictor report](audits/contextual_composition_cc5_predictor_report_20260803.md)
for full evidence. The trained predictor (KNN regret regressor + OOD-gated
fallback) ties the best fixed policy on the 6 CC4 evaluation windows (mean
ANWG 0.2306 vs 0.2310; bootstrap 95% CIs ~[0.10, 0.37] on both) and is
beaten by `best_global_composition` (0.2633). This is judged a data-scarcity
finding, not a methodology failure: n=6 held-out windows cannot
statistically distinguish these methods at any interesting effect size, and
the top 4 cross-validated model candidates were separated by less than the
CV noise floor. CC6 is **not** queued as a result. Exact remaining task:
expand the CC4 dataset (more windows, more per regime) before retraining;
no CC5 pipeline code changes are anticipated to be required.

**Retry complete (2026-08-03). Verdict: `REGIME_SPECIFIC_ONLY`.** The
exact remaining task above was carried out: a targeted CC4b oracle-dataset
expansion (`configs/cc4b_oracle_composition_expansion.yaml`) grew the
held-out set from 6 to 76 windows (30 development windows; quality gates
all passed -- see `scripts/check_cc4b_quality_gates.py`), reusing CC4's
engine and DSL/compiler/verifier infrastructure and CC5's own pipeline
completely unchanged. Result: the predictor (gradient_boosting regret
regression) now **clearly beats best fixed policy** (0.4006 vs 0.3895 ANWG)
and is **competitive with the existing hard selector** (0.3938), but does
**not clearly beat `best_global_composition`** (0.4025 -- within the
bootstrap CI overlap on both sides). This is a materially more informative
result than the first attempt's `INCONCLUSIVE` (the fixed-policy question
is now resolved), not a repeat of it. CC6 remains **not** queued. Exact
remaining task: address the uncertainty-method gap (across two independent
retries, leave-one-window-out model selection has never chosen the one
model family this pipeline's ensemble-disagreement uncertainty estimator
supports), then a per-regime regret breakdown to determine whether the
predictor's value is regime-concentrated. See the
[CC4b/CC5 retry report](audits/contextual_composition_cc4b_cc5_retry_report_20260803.md)
for full evidence.

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

## Current Scientific Interpretation

This section summarizes what the evidence gathered so far actually means,
kept separate from the phase-gate mechanics above.

* **CC1b proved composition opportunity exists in discriminative
  workloads.** The original CC1 suite was nondiscriminative; the
  strengthened CC1b suite found a true simulator-executed weighted-Borda
  composition opportunity and cleared the `PROCEED` gate (non-near-tie
  opportunity gap 0.0167735). See
  [Query 5 discriminativeness review](audits/contextual_composition_query5_discriminativeness_review_20260731.md).
* **CC4 produced simulator-grounded oracle composition labels.** 12
  windows, 34 verified candidates, 408 true simulator executions, 0 reward-
  vector interpolation anywhere. A composition-family candidate was the
  oracle winner on 66.7% of CC4's own held-out windows. See the
  [CC4 oracle dataset report](audits/contextual_composition_cc4_oracle_dataset_report_20260803.md).
* **The first CC5 run was technically correct but statistically
  inconclusive.** The pipeline (targets, models, uncertainty/OOD/fallback,
  evaluation, verdict logic) ran deterministically and passed all 22 of
  its focused tests. With only 6 held-out windows, `best_global_composition`
  (mean ANWG 0.2633) beat the trained predictor (0.2306), which in turn
  tied `best_fixed_policy` (0.2310) -- differences the bootstrap 95% CIs
  (roughly [0.10, 0.40] on every method) cannot statistically support as
  real. See the
  [CC5 predictor report](audits/contextual_composition_cc5_predictor_report_20260803.md).
* **The CC4b expansion (in progress) tests whether that outcome was caused
  by insufficient and insufficiently diverse data**, not by a genuine
  absence of adaptive value. It reuses CC4's exact candidate search (same
  34 candidates) so the comparison is apples-to-apples, and reruns CC5's
  exact, unchanged pipeline against the larger dataset.
* **The decisive comparison** the retry is designed to resolve is:

  > **contextual composition predictor** vs. **best global verified
  > composition**

  with three possible outcomes:

  1. **Predictor wins** -> contextual adaptation is justified; proceed
     toward CC6 (controlled per-window adaptation).
  2. **Global composition wins** -> composition itself is useful (a single
     well-chosen verified composition beats every fixed policy and the
     hard selector), but machine-learned contextual adaptation may not be
     worth its complexity given current data; the roadmap would need a
     documented decision before any further predictor investment.
  3. **Hard selector wins** -> even primitive-level composition may not
     justify its added complexity over a much simpler per-regime lookup;
     this would be the strongest signal to reconsider the CC5+ research
     direction entirely, not just retune it.

  None of these three outcomes is presupposed; the retry report records
  whichever one the expanded held-out evidence actually supports.

  **Actual result (2026-08-03, n=76 held-out windows):** none of the three
  outcomes above obtained cleanly -- the evidence is genuinely mixed.
  Outcome 3 (hard selector wins) is ruled out: the predictor beats the
  hard selector. Between outcomes 1 and 2, the data leans toward outcome 2
  (global composition is very strong, 0.4025 vs the predictor's 0.4006 ANWG)
  but the gap is within the bootstrap CI overlap, so outcome 1 is not ruled
  out either -- what the retry *did* resolve is that the predictor clearly
  beats naive fixed-policy selection (0.3895), which the first attempt at
  n=6 could not distinguish from a tie. The roadmap's working interpretation
  is: composition (of some form -- fixed, global, or contextually-selected)
  is clearly valuable; whether *contextual* adaptation specifically adds
  value over a single well-chosen global composition remains open and is
  the CC4b/CC5 retry report's exact next research step (address the
  uncertainty-method gap, then a per-regime regret breakdown).

## Future Research Directions -- Not Yet Implemented

**These are ideas under consideration for phases beyond the current
roadmap. None of them exist in this repository today -- no code, no
experiments, no partial implementation.** They are recorded here so that
future roadmap decisions have a place to point to, not because any of them
are scheduled or authorized.

* **Envelope-aware policy usefulness** -- characterizing a policy's value
  as a function of the operating envelope (load, SLO tightness, mix) it is
  actually useful within, rather than a single scalar score.
* **Regret-profile complementarity** -- identifying which primitives/
  compositions have *complementary* regret profiles (each wins where the
  others lose) as a principled basis for a richer composition search,
  beyond CC4's current independent-candidate grid.
* **Behavioral embeddings** -- learned vector representations of a
  policy's scheduling behavior (e.g. from its decision traces), as a
  similarity/complementarity signal for composition or selection.
* **Typed module-level crossover** -- genome-style recombination of
  scheduling modules (admission/ranking/placement/batching) with type
  compatibility constraints, extending `policies/genome.py` beyond its
  current conservative scope.
* **QD/MAP-Elites-style library expansion** -- quality-diversity search
  over the composition space to build a diverse library of behaviorally
  distinct, individually strong compositions, rather than a single
  best-of search.
* **LLM-guided symbolic scheduler synthesis** -- using an LLM to propose
  candidate DSL structures for CC3's verifier to check and CC4's simulator
  to evaluate (CC4's report already sketches a clean, cost-capped,
  cache-deduplicated integration point for this that was never exercised
  live in any run to date).
* **Symbolic distillation from a dynamic teacher** -- once CC6 (dynamic
  adaptation) exists, distilling its learned per-window behavior back into
  a simpler, auditable symbolic (DSL) form.

None of the above should be read as implemented, scheduled, or endorsed by
this roadmap's phase gates -- they require their own future decision gate
and roadmap entry before any implementation work begins, exactly like every
CC-numbered phase above.

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
