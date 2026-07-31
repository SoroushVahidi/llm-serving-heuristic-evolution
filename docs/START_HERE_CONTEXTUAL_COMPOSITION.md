# Start Here: Contextual Composition

Authoritative branch: `contextual-compositional-heuristics-20260731`

Current phase: `CC1 - Composition opportunity experiment`

Exact next action: implement the minimal true simulator-executed
composition-opportunity experiment described in
[contextual_composition_roadmap.md](contextual_composition_roadmap.md) and
GitHub issue
[#1](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1).

## Read In This Order

1. `docs/START_HERE_CONTEXTUAL_COMPOSITION.md`
2. `docs/contextual_composition_roadmap.md`
3. `docs/contextual_composition_decisions.md`
4. `docs/CONTEXTUAL_COMPOSITION_BRANCH.md`
5. `docs/audits/local_branch_compositional_path_audit_20260731.md`
6. `docs/audits/contextual_composition_query2_roadmap_report_20260731.md`
   or the latest later contextual-composition audit report

## What Not To Do Yet

Do not implement broad heuristic primitives, policy-composition refactors, DSL
extensions, contextual predictors, dynamic adaptation, real-vLLM jobs, hosted
API experiments, or large simulator sweeps before the roadmap phase gate allows
them.

## How To Update Status

Every future implementation query must:

- verify branch and upstream;
- read the roadmap;
- identify the current `NEXT` phase;
- update roadmap status, evidence links, decisions, risks, and next action;
- add or update an audit report;
- run relevant tests;
- commit and push;
- leave the branch clean and synchronized.

Run the consistency checker before committing:

```bash
python scripts/check_contextual_composition_status.py
```

## Historical Status And Safe Claims

Historical project status lives in `docs/current/` and the full historical index
is `docs/README.md`. Those documents remain provenance but may describe older
branches or phases.

Safe claim guidance lives in `docs/result_claims.md`. Contextual-composition
claims must distinguish historical simulator claims, corrected-objective claims,
real-trace-derived claims, hosted-API claims, and real-vLLM claims.
