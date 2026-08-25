# Contextual Composition Query 2 Roadmap Report - 2026-07-31

Repository: `/home/soroush/llm-serving-heuristic-evolution`

Branch: `contextual-compositional-heuristics-20260731`

Starting commit: `265ddcab073698968bcef9d5e32386e4b6721052`

## 1. Starting State

Verified before edits:

- current branch: `contextual-compositional-heuristics-20260731`
- HEAD: `265ddcab073698968bcef9d5e32386e4b6721052`
- upstream: `origin/contextual-compositional-heuristics-20260731`
- ahead/behind: `0 / 0`
- working tree: clean

Verified Query 1 artifacts existed:

- `docs/CONTEXTUAL_COMPOSITION_BRANCH.md`
- `docs/audits/local_branch_compositional_path_audit_20260731.md`
- `docs/audits/contextual_composition_query1_sync_report_20260731.md`

## 2. Existing Documentation Reviewed

Reviewed likely roadmap/status/navigation surfaces, including:

- `README.md`
- `docs/README.md`
- `docs/roadmap.md`
- `docs/research_status.md`
- `docs/result_claims.md`
- `docs/current/README.md`
- `docs/current/PROJECT_STATUS.md`
- `docs/current/RESUME_HERE.md`
- `docs/current/EXPERIMENT_INDEX.md`
- `docs/current/POLICY_COMPOSITION_READINESS.md`
- `docs/current/COMPOSITION_IMPLEMENTATION_STATUS.md`
- Query 1 branch, audit, and sync-report documents

Findings:

- Several documents intentionally preserve older branch/status authorities.
- `docs/current/README.md` still names historical integration branch
  `wulver-final-integration-20260721`.
- `docs/roadmap.md` is a historical numbered-phase roadmap plus Selector v2
  bridge, not the contextual-composition roadmap.
- `docs/research_status.md` is already labeled historical.
- `docs/result_claims.md` remains the safe-claim authority but mixes multiple
  result generations, so contextual composition must distinguish claim classes.

No historical documents were deleted or broadly rewritten.

## 3. Canonical Files Created

Created:

- `docs/contextual_composition_roadmap.md`
- `docs/contextual_composition_decisions.md`
- `docs/START_HERE_CONTEXTUAL_COMPOSITION.md`
- `scripts/check_contextual_composition_status.py`

Updated:

- `README.md`
- `docs/README.md`
- `docs/current/README.md`
- `docs/roadmap.md`
- `docs/CONTEXTUAL_COMPOSITION_BRANCH.md`

## 4. Roadmap State

Canonical branch:
`contextual-compositional-heuristics-20260731`

Roadmap version: `1`

Current phase: `CC1`

Current status: `NEXT`

Next action:
implement the minimal true simulator-executed composition-opportunity
experiment.

Phase status after Query 2:

- CC0: COMPLETE
- CC1: NEXT
- CC2-CC5: BLOCKED
- CC6-CC8: PLANNED

## 5. Decision Log

Created `docs/contextual_composition_decisions.md` with accepted decisions:

- CCD-001: use the synchronized contextual-composition branch as authoritative
- CCD-002: measure the composition opportunity gap before large refactoring
- CCD-003: prefer compatible primitive composition over averaging incompatible
  final actions
- CCD-004: use arrival-normalized weighted goodput as the primary objective
- CCD-005: treat uncertainty, abstention, fallback, and switching stability as
  core requirements
- CCD-006: do not extend the DSL before the minimal interface and opportunity
  experiment
- CCD-007: keep runtime free of required live LLM calls
- CCD-008: separate historical simulator claims, corrected-objective claims,
  real-trace claims, and real-serving claims

## 6. GitHub Issues And Labels

Open issues were inspected before creation; none existed.

Created labels:

- `contextual-composition`
- `research`
- `experiment`
- `architecture`
- `blocked`
- `next`

Created issues:

- #1: CC1: Measure the composition opportunity gap
- #2: CC2: Define the canonical scheduling primitive interface
- #3: CC3: Extend the DSL and verifier for safe compositions
- #4: CC4: Build the simulator-derived oracle composition dataset
- #5: CC5: Train the contextual composition predictor
- #6: CC6-CC8: Dynamic adaptation, hardening, and real-serving validation

Issue numbers are cross-linked from the roadmap.

## 7. Validation

Roadmap consistency:

```bash
python scripts/check_contextual_composition_status.py
```

Result: passed.

Additional validation is recorded in the final Query 2 terminal summary.

## 8. Unresolved Issues

- The roadmap establishment commit SHA is intentionally not embedded in the
  roadmap because doing so would make the commit hash self-referential.
- Historical docs still contain older branch and phase references by design;
  navigation now redirects contextual-composition contributors to the new
  scoped roadmap.
- No method implementation, DSL extension, predictor, real-vLLM job, hosted API
  call, or new experiment was started in Query 2.

## 9. Starting Point For Query 3

Query 3 should start from branch
`contextual-compositional-heuristics-20260731` after the Query 2 commit is
pushed and synchronized.

Query 3 should continue organization and polishing work without implementing
heuristic primitives, weighted mixtures, DSL extensions, contextual predictors,
or new experiments unless the sequence instructions explicitly change.
