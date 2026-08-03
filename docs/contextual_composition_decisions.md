# Contextual Composition Decision Log

Status vocabulary: `accepted`, `superseded`, `provisional`.

## CCD-001: Use The Synchronized Contextual-Composition Branch As Authoritative

Date: 2026-07-31

Status: accepted

Decision: Use `contextual-compositional-heuristics-20260731` as the
authoritative branch for the contextual-compositional heuristic research path.

Rationale: Query 1 synchronized
`reality-grounded-dataset-expansion-20260724` with GitHub, preserved the local
audit, created the new branch from synchronized commit
`775147beec997b14039bbaa088d17630a32156cf`, committed the branch marker, and
pushed upstream.

Consequences: Future contextual-composition work starts on this branch unless a
later decision supersedes this one. Historical branch/status docs remain
provenance but are not the branch authority for this path.

Related files or evidence:

- `docs/CONTEXTUAL_COMPOSITION_BRANCH.md`
- `docs/audits/contextual_composition_query1_sync_report_20260731.md`
- `docs/contextual_composition_roadmap.md`

## CCD-002: Measure The Composition Opportunity Gap Before Large Refactoring

Date: 2026-07-31

Status: accepted

Decision: CC1 must measure the true simulator-executed composition opportunity
gap before broad primitive refactoring, DSL extension, or predictor training.

Rationale: Existing policy reward vectors can show fixed-policy and hard
selector performance but cannot prove weighted-mixture behavior because mixed
actions can interact. The audit found native composition prototypes but no
decisive local true-execution opportunity measurement.

Consequences: At roadmap establishment time, CC1 was the only `NEXT` phase and
CC2-CC8 remained blocked or planned until the CC1 decision gate passed or was
explicitly revised. CCD-010 and CCD-011 record the current post-CC1b state.

Related files or evidence:

- `docs/audits/local_branch_compositional_path_audit_20260731.md`
- `docs/contextual_composition_roadmap.md`
- GitHub issue #1

## CCD-003: Prefer Compatible Primitive Composition Over Averaging Incompatible Final Actions

Date: 2026-07-31

Status: accepted

Decision: Composition should combine compatible primitive scores, ranks,
admission gates, placement scores, or bounded parameters, not arbitrary final
policy actions or incomparable raw scores.

Rationale: Existing policies expose different semantics: ranks, private scores,
binary admission behavior, placement choices, and full actions. Averaging final
actions or unnormalized scores can be semantically invalid.

Consequences: CC1 starts with a minimal compatible subset. CC2 must define
semantic contracts for primitive outputs before broad refactoring.

Related files or evidence:

- `src/llmserveopt/policies/composition.py`
- `src/llmserveopt/policies/score_aggregation.py`
- `src/llmserveopt/policies/capabilities.py`
- `docs/current/POLICY_COMPOSITION_READINESS.md`

## CCD-004: Use Arrival-Normalized Weighted Goodput As The Primary Objective

Date: 2026-07-31

Status: accepted

Decision: Use `arrival_normalized_weighted_goodput` as the primary optimization
and comparison objective unless a later documented decision changes it.

Rationale: Completed-request-only weighted goodput can reward selective
dropping/rejection and cannot be the sole system-level objective.

Consequences: All contextual-composition reports must also include completion
fraction and number of arrivals. Completed-request-only WG may be reported only
as a secondary conditional-quality metric.

Related files or evidence:

- `docs/selector_objective_audit.md`
- `docs/result_claims.md`
- `src/llmserveopt/core/metrics.py`
- `docs/contextual_composition_roadmap.md`

## CCD-005: Treat Uncertainty, Abstention, Fallback, And Switching Stability As Core Requirements

Date: 2026-07-31

Status: accepted

Decision: Uncertainty estimates, OOD detection, abstention, robust fallback,
minimum commitment, and hysteresis are core requirements for deployable
contextual composition.

Rationale: The existing selector and composition evidence includes near-ties,
OOD sensitivity, and possible instability from context-dependent switching.
Safety behavior must be designed into the method rather than added at the end.

Consequences: CC5 and CC6 cannot pass their gates without safety behavior.
Earlier phases must record enough ambiguity and near-tie evidence to support
later abstention decisions.

Related files or evidence:

- `src/llmserveopt/selector/advanced.py`
- `src/llmserveopt/policies/composition.py`
- `docs/contextual_composition_roadmap.md`

## CCD-006: Do Not Extend The DSL Before The Minimal Interface And Opportunity Experiment

Date: 2026-07-31

Status: accepted

Decision: Do not extend the DSL/verifier for contextual compositions until CC1
and CC2 clarify which primitive semantics and parameter bounds are needed.

Rationale: Extending the DSL before measuring opportunity and defining
compatible primitive outputs risks enshrining the wrong abstractions.

Consequences: CC3 is blocked on CC1 and CC2. Query 2 must not implement DSL
extensions.

Related files or evidence:

- `src/llmserveopt/heuristics/dsl_schema.py`
- `src/llmserveopt/heuristics/verifier.py`
- `docs/contextual_composition_roadmap.md`

## CCD-007: Keep Runtime Free Of Required Live LLM Calls

Date: 2026-07-31

Status: accepted

Decision: Runtime scheduling must not require live LLM calls.

Rationale: Scheduling needs deterministic, bounded, low-overhead behavior.
LLM-generated structures can be used offline only after verification and
evaluation.

Consequences: Runtime inference may load a trained local model or deterministic
parameters, but no live generation request is part of the scheduler path.

Related files or evidence:

- `src/llmserveopt/llm_generation/`
- `src/llmserveopt/heuristics/`
- `docs/contextual_composition_roadmap.md`

## CCD-008: Separate Historical, Corrected-Objective, Real-Trace, And Real-Serving Claims

Date: 2026-07-31

Status: accepted

Decision: Contextual-composition evidence must clearly separate historical
simulator claims, corrected-objective claims, real-trace-derived simulation
claims, hosted-API claims, and real-vLLM claims.

Rationale: The repository contains several generations of results with
different objectives, datasets, branches, and confounders. Mixing them creates
unsafe scientific claims.

Consequences: Roadmap evidence links and future audits must label which claim
class each result supports. Known-confounded real-vLLM selector results cannot
serve as clean validation evidence.

Related files or evidence:

- `docs/result_claims.md`
- `docs/audits/local_branch_compositional_path_audit_20260731.md`
- `docs/current/EXPERIMENT_INDEX.md`
- `docs/contextual_composition_roadmap.md`

## CCD-009: Stop CC2 Until CC1 Has Discriminative Composition Evidence

Date: 2026-07-31

Status: superseded

Decision: The Query 4 CC1 full local run does not justify starting CC2
primitive-interface refactoring. Treat the result as `STOP_OR_REDESIGN` and
keep CC2 blocked.

Rationale: The approved true simulator-executed weighted Borda experiment ran
200 simulator executions over the required five-policy subset and representative
local workloads, including two real-trace-derived OOD windows. The oracle
mixture matched oracle fixed with composition-opportunity gap `0.0`; every
evaluation window was a near tie under the fixed-policy margin threshold; best
global mixture collapsed to the one-hot WSP mixture.

Consequences: Query 5 should not begin CC2 or CC3. It should either redesign a
bounded CC1 workload/discriminativeness check or document a pause/stop decision
for the contextual-composition path.

Related files or evidence:

- `docs/audits/contextual_composition_query4_cc1_results_20260731.md`
- `results/cc1_composition_opportunity/query4_full_20260731/manifest.json`
- GitHub issue #1

## CCD-010: Continue To CC2 After CC1b Discriminative Evidence

Date: 2026-07-31

Status: accepted

Decision: Continue the contextual-composition path to CC2. Query 6 should
define the canonical primitive interface and representative-policy equivalence
tests. Do not begin CC3 DSL work until CC2 passes its gate.

Rationale: Query 5 found that the Query 4 `STOP_OR_REDESIGN` result was caused
by nondiscriminative workloads, not reward-vector interpolation or oracle
accounting bugs. The original CC1 windows had high simulated capacity, short
windows, permissive drain behavior, and SLO slack much looser than observed
latencies, so several fixed policies completed all requests within SLO and
tied at ANWG `1.0`. The CC1b suite retained true simulator execution but
tightened the scientific setting: fixed-policy spread was required before
mixture evaluation; service prefill/contention was enabled; SLO slack was set
near the observed latency scale; and held-out windows covered overload,
long-prompt mixed tight SLOs, burst transitions, and Azure-conversation-like
OOD traffic. CC1b measured a non-near-tie composition-opportunity gap of
`0.0167735`, best regime gain of `0.05`, and completion impact of `0.0`.

Consequences: CC2 is now `NEXT`. CC2 may define typed primitive interfaces for
ranking, admission, placement, batching, and resource guards, but Query 6 must
not implement CC3 DSL extensions or contextual predictor training.

Related files or evidence:

- `docs/audits/contextual_composition_query5_discriminativeness_review_20260731.md`
- `configs/cc1b_composition_discriminative.yaml`
- `results/cc1b_composition_discriminative/query5_cc1b_full_20260731/manifest.json`
- GitHub issue #1

## CCD-011: Pause After CC1b Before CC2 Implementation

Date: 2026-07-31

Status: accepted

Decision: Intentionally pause the contextual-composition branch after the CC1b
`PROCEED` decision and before implementing CC2. Keep CC2 as the single `NEXT`
phase, but make the next repository action a final resume-readiness pass rather
than primitive-interface implementation.

Rationale: CC1b established enough evidence to enter CC2, but the branch needs
a stable checkpoint so future work can resume without re-auditing Query 1-5
documents, local-only result paths, issue state, and validation commands.

Consequences: Query 7 performed final repository polish, consistency cleanup,
and resume-readiness verification without implementing CC2. After the pause is
lifted, the exact CC2 task is to define the canonical primitive interface for
ranking, admission, placement, batching, and resource guards, then add
representative-policy equivalence tests. Do not extend the DSL yet.

Related files or evidence:

- `docs/audits/contextual_composition_pause_checkpoint_20260731.md`
- `docs/RESUME_CONTEXTUAL_COMPOSITION.md`
- `docs/audits/contextual_composition_query6_pause_report_20260731.md`
- GitHub issue #2

## CCD-012: CC2 Primitive Interface Complete, Proceed To CC3

Date: 2026-08-02

Status: accepted

Decision: Mark CC2 (canonical scheduling primitive interface) COMPLETE and
proceed to CC3 (compositional DSL and verifier). Do not begin CC4 offline
oracle dataset generation, CC5 predictor training, or any later phase.

Rationale: `src/llmserveopt/policies/primitives.py` implements a 28-entry
canonical registry across the five required families (RANKING, ADMISSION,
PLACEMENT, BATCHING, RESOURCE_GUARD), covering every primitive named in the
roadmap (deadline urgency, laxity, prompt length, predicted output length,
estimated service time, priority, queue age, KV pressure, projected GPU
load, admission risk, prefill pressure, fairness/starvation prevention) plus
supporting primitives needed for exact reconstruction (request-id
tie-break, system overload guard, admission credit budget, and others).
`src/llmserveopt/policies/primitive_reconstructions.py` reconstructs seven
representative policies (fifo, edf, weighted_shortest_processing,
estimated_service_time_first, best_fit, admission_control,
scorpio_style_slo_guard) using only these primitives. Six of seven are
EXACT (verified via 65 equivalence tests spanning synthetic single/multi-GPU
states, a 60-trial-per-policy randomized fuzz suite restricted to
physically valid states, 3-seed full simulator-trace runs compared on
completion fraction/ANWG/weighted goodput, and deterministic-replay
checks); `scorpio_style_slo_guard` is documented APPROXIMATE (0 observed
mismatches across all fixtures, but composed via primitive calls rather
than the original's single inline computation, so no formal
floating-point-identity proof is claimed). The DSL
(`src/llmserveopt/heuristics/`) was not modified, per the CC2 scope
boundary. 42 additional registry/typed-behavior tests confirm unique
names, all five families represented, causal-only inputs, deterministic
enforcement, and explicit unsupported-parameter/non-positive-capacity
errors.

Consequences: CC3 is now `NEXT`. CC3 may extend the DSL/verifier to expose
named references to the CC2 primitive registry, using
`PrimitiveSpec.param_bounds`/`compatible_families` as the source of truth
for verifier bounds and family-mixing rules (see the architecture doc's
"How CC3 Will Later Expose These Primitives" section). CC4-CC8 remain
blocked until CC3's gate passes.

Related files or evidence:

- `docs/architecture/contextual_composition_primitives.md`
- `docs/audits/contextual_composition_cc2_primitive_interface_report_20260802.md`
- `src/llmserveopt/policies/primitives.py`
- `src/llmserveopt/policies/primitive_reconstructions.py`
- `tests/test_primitive_interface.py`
- `tests/test_primitive_reconstructed_policies.py`
- GitHub issue #2 (closing), issue #3 (active)

## CCD-013: CC3 Compositional DSL Complete, CC4 Queued But Not Started

Date: 2026-08-03

Status: accepted

Decision: Mark CC3 (compositional DSL and verifier) COMPLETE. Do not begin
CC4 (offline oracle composition dataset) in this same query even though
CC3's exit gate passed; CC4 becomes the single `NEXT` phase, queued for a
future, explicitly authorized query.

Rationale: `src/llmserveopt/heuristics/primitive_bridge.py` (new) is a
read-only adapter over `policies/primitives.py`, exactly the "separate
compiler-facing adapter module" the CC2 architecture doc anticipated. All 8
roadmap-required constructs (named primitive references, weighted sums,
sparse top-k mixtures, conditional branches, admission gates, placement
scores, externally supplied bounded parameters, deterministic
tie-breaking) plus explicit safe fallback are implemented, each with a
runnable example (`heuristics/primitive_composition_examples.py` +
`configs/heuristics/examples/*.json`) and focused tests
(`tests/test_contextual_composition_cc3_dsl.py`, 45 tests). Primitive/param
references are verified against the as-authored document, then lowered by
the compiler into ordinary `var` nodes, so `heuristics/expressions.py`
needed zero new coupling to `primitives.py`. One real backward-compatibility
regression was found and fixed during implementation (an overly broad
`on_no_admits`-required rule that broke all 6 genome-derived deployable
policies using pre-CC3 admission conditions; fixed by scoping the
requirement to only the new `primitive_gate` construct) -- confirmed fixed
by a full `test_policy_genome_coverage.py` re-run (122 passed). 447
focused+regression tests and the full non-live suite (3047 passed, 5
skipped, 1 pre-existing unrelated failure confirmed via `git stash` to
predate this branch's CC3 work) pass. Every pre-CC3 canonical example and
genome-derived heuristic verifies, compiles, and scores identically to
before.

Consequences: CC4 is now `NEXT` (queued, not started). A future query must
read `docs/architecture/contextual_composition_dsl.md` and the CC3
DSL/verifier report before sampling/mutating over the CC3-exposed
primitive-reference surface (`CompiledHeuristic.primitive_refs`/
`placement_keys`/`admission_budget_spec`/`param_declarations`) to generate
candidate compositions for CC4. CC5-CC8 remain blocked until CC4's gate
passes. Two non-blocking scope boundaries are documented for CC4 to be
aware of: `admission_budget`'s per-step cap does not apply when
`on_no_admits: safe_fallback` delegates a whole step, and `placement.keys`
composes lexicographically rather than via a weighted/normalized blend.

Related files or evidence:

- `docs/architecture/contextual_composition_dsl.md`
- `docs/audits/contextual_composition_cc3_dsl_verifier_report_20260803.md`
- `src/llmserveopt/heuristics/primitive_bridge.py`
- `src/llmserveopt/heuristics/primitive_composition_examples.py`
- `tests/test_contextual_composition_cc3_dsl.py`
- GitHub issue #3 (closing), issue #4 (queued, not started)

## CCD-014: CC4 Oracle Composition Dataset Complete, CC5 Queued But Not Started

Date: 2026-08-03

Status: accepted

Decision: Mark CC4 (offline oracle composition dataset) COMPLETE. Do not
begin CC5 (contextual composition predictor) in this same query even though
CC4's exit gate passed; CC5 becomes the single `NEXT` phase, queued for a
future, explicitly authorized query.

Rationale: `src/llmserveopt/experiments/cc4_oracle_composition_dataset.py`
(new) reuses CC1's workload-window construction, GPU/service-model
construction, and git-state capture verbatim, and executes every candidate
(fixed policies, the CC1b weighted-Borda baseline replayed unchanged,
bounded weighted-primitive-mixture and sparse-top-k DSL mixtures,
admission-gate and placement variants) through the same `run_policy` entry
point CC1 uses -- no new simulator-invocation code, no reward-vector
interpolation anywhere in the pipeline. 12 workload windows cover every
required regime category (underloaded, saturated, mixed SLO, long
prompt/output, burst transition, KV pressure, prediction noise, priority
conflict, selective-admission trap, Azure-conversation-like, BurstGPT-derived)
across TRAIN/VALIDATION/ID_TEST/OOD_TEST splits; 34 candidates were
generated and 0 rejected by the CC3 verifier; 408 true simulator executions
completed in ~4.5 minutes locally. Reproducibility was verified twice (an
interrupt-and-resume cycle via `CC4TrialStore`'s append-only checkpoint, and
an independent from-scratch re-run reproducing a byte-identical verdict). A
composition-family candidate is the oracle winner on 4/6 (66.7%) held-out
evaluation windows; completion-fraction constraints hold on all 12 windows;
`admission_gate_variant` candidates show the lowest mean regret of any
family. 20 new focused tests pass, including a real (non-mocked)
resume/reproducibility integration test. Two real issues were found and
fixed during development (not left as known bugs): a BurstGPT real-trace
`arrival_time_scale` misconfiguration that caused an ~12.5M-step simulator
hang, and a dev/eval split-separation gap where `development_splits`/
`evaluation_splits` were collected in config but never actually used to
scope the dataset-level verdict (fixed to mirror CC1's own dev/eval
separation, so TRAIN-window signal cannot certify the held-out claim).

Consequences: CC5 is now `NEXT` (queued, not started). A future query must
read `docs/audits/contextual_composition_cc4_oracle_dataset_report_20260803.md`
(its "Exact CC5 Entry Condition" section) before training against
`oracle_labels.parquet`/`regret_matrix.parquet`/`causal_features.parquet`,
fitting only on `development_splits` windows and reserving
`evaluation_splits` windows exclusively for the reported validation claim.
CC6-CC8 remain blocked until CC5's gate passes. Non-blocking scope
boundaries for a future CC4 iteration or for CC5 to be aware of:
`admission_budget` was not searched at all in this dataset (deliberately,
per CC3's own documented risk about combining it with
`on_no_admits: safe_fallback`); `fallback_activated_last_step` is a
best-effort last-step flag, not a full-run aggregate (CC3's
`HeuristicPolicy.last_trace` does not accumulate across a run); one
evaluation window (`cc4_burstgpt_derived_ood_test`) is near-collapse
(0.0375 completion fraction across every candidate) and carries little
discriminative signal -- flagged, not excluded.

Related files or evidence:

- `docs/audits/contextual_composition_cc4_oracle_dataset_report_20260803.md`
- `configs/cc4_oracle_composition_dataset.yaml`
- `src/llmserveopt/experiments/cc4_oracle_composition_dataset.py`
- `scripts/run_cc4_oracle_composition_dataset.py`
- `tests/test_cc4_oracle_composition_dataset.py`
- `results/cc4_oracle_composition_dataset/20260803T170735Z/` (local, untracked)
- GitHub issue #4 (closing), issue #5 (queued, not started)
