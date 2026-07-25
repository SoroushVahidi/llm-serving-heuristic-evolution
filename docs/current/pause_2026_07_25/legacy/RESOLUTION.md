# Legacy Worktree Resolution (Part 3)

## Worktree
- Path: `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution`
- Branch: `wulver-policy-composition-readiness`
- HEAD: `c8aee129f553f8dc3ede99eac60d5b14484beb41`
- Upstream: none configured
- Remote branch `origin/wulver-policy-composition-readiness`: **absent**

## Ancestry
`c8aee12` **is an ancestor** of both:
- `wulver-final-integration-20260721` / `origin/wulver-final-integration-20260721`
- `reality-grounded-dataset-expansion-20260724` (local tip)

Therefore the committed legacy tip is already in the durable integration lineage once Part 3 pushes complete.

## Dirty / untracked classification (2026-07-25)
Byte comparison against the canonical integration tip showed:

- **Already preserved identical:** majority of Policy Library v2 / composition / synthesis source, tests, tools, and JSON schemas present as untracked copies in the legacy worktree.
- **Differs but older/superseded drafts:** several docs and older revisions of `composition.py` / `structural_synthesis.py` where the integration tip is larger and newer (includes post-pause operator/tracing work). Unique legacy lines are outdated status text (e.g. “sweep still running”) or earlier API shapes, not newer unique science.
- **No unique reusable newer source/tests** requiring a new commit on the legacy branch.
- **No unique newer documentation** that should replace current handoff docs.

## Actions taken
1. Documented classification here (and `classification.tsv`).
2. **Did not** commit dirty legacy drafts onto `wulver-policy-composition-readiness` (would misrepresent readiness / regress content).
3. **Did not** push `wulver-policy-composition-readiness` (no unique unreproducible tip content beyond already-ancestral `c8aee12`).
4. **Left the worktree in place** (optional removal deferred; prefer leave documented over deletion).
5. **Did not delete** the local branch.

## Part 2 inventory
See `../LEGACY_WORKTREE_INVENTORY.md`.
