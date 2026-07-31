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

Consequences: CC1 is the only `NEXT` phase. CC2-CC8 remain blocked or planned
until the CC1 decision gate passes or is explicitly revised.

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
