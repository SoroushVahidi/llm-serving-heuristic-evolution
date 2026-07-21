# Wulver Branch Lineage Audit

Generated during Query 1 cleanup audit on 2026-07-21.

## Source-of-Truth Summary

`wulver-policy-composition-readiness` was the best current integration branch at the end of Query 1 because it contained the latest local commit lineage and the uncommitted Policy Library v2, composition, native pilot, and structural synthesis work in one worktree.

Query 2 created a separate final integration worktree and branch so active SLURM workflows that `cd` into the original checkout can continue without source-tree or `HEAD` changes:

```text
FINAL_INTEGRATION_BRANCH = wulver-final-integration-20260721
FINAL_INTEGRATION_WORKTREE = /mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-final-integration
```

Recommended final integration base:

```text
RECOMMENDED_FINAL_INTEGRATION_BASE = wulver-final-integration-20260721, created from c8aee129f553f8dc3ede99eac60d5b14484beb41 and populated with the audited Wulver-only work
```

Do not reset either Wulver worktree to a remote branch. `origin/repo-polish-query5-final-verification` is the closest complete remote baseline, but it lacks local commit `c8aee12` and all Wulver-only policy/composition/synthesis work.

## Branch Tips

| Branch | Tip | Upstream | Notes |
| --- | --- | --- | --- |
| `wulver-final-integration-20260721` | starts at `c8aee12`; Query 2 commits on top | none | Dedicated final integration branch in a separate worktree. |
| `wulver-policy-composition-readiness` | `c8aee12` | none | Current branch; contains the working tree with composition and structural synthesis changes. |
| `wulver-policy-library-v2-frontier` | `c8aee12` | none | Same tip as current branch; Policy Library v2 code is currently uncommitted in the shared worktree. |
| `wulver-selector-v2-overnight-scale` | `c8aee12` | `origin/repo-polish-query5-final-verification` | Same tip as current branch; local commit fixes Selector v2 real-trace split grouping. |
| `origin/repo-polish-query5-final-verification` | `8087ee1` | remote | Closest remote baseline; one commit behind local Wulver branch tips. |
| `origin/wulver-runtime-validation-benchmark-pack` | `57e4bb1` | remote | Older runtime-validation branch. |
| `origin/main` | `277e535` | remote | Older than the Wulver validation/Selector v2 lineage. |

## Local Commit Lineage

The relevant recent lineage is:

```text
c8aee12 Fix Selector v2 real-trace split grouping
8087ee1 Add operational agent handoff doc
45e7610 Stop idle vLLM process, execute safe local artifact cleanup, reconcile status docs
ca6654d Mark unregistered historical policy prototype
db9c369 Classify historical configs and local artifacts
93f5cc9 Clarify experiment and local-result storage policy
```

## Fragmentation Assessment

The named Wulver branches are not diverged by committed history today; they all point to `c8aee12`. Work is fragmented by topic in the dirty worktree, not by separate committed branch tips.

Branches requiring preservation/integration before final push:

- `wulver-selector-v2-overnight-scale`: preserve local commit `c8aee12`.
- `wulver-policy-library-v2-frontier`: preserve Policy Library v2 source/tests/docs/scripts currently uncommitted.
- `wulver-policy-composition-readiness`: preserve composition harness, native pilot scripts, structural synthesis harness, and Query 1 audit docs currently uncommitted.

## Integration Guidance for Query 2

Use `wulver-final-integration-20260721` as the integration branch and organize commits by coherent topic:

1. Policy Library v2 policies, registry wiring, tests, and experiment script.
2. Composition readiness/harness code, tests, sbatch scripts, and docs.
3. Structural synthesis readiness code, tests, sbatch script, and docs.
4. Query 1 cleanup/audit docs and docs/current navigation updates.

Do not attempt blind merges between the local Wulver branches because their committed tips are identical and the meaningful work is uncommitted.
