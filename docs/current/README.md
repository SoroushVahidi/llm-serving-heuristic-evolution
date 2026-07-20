# Start Here

This is the canonical current-state documentation set. If you are a new
agent or contributor picking up this repository, read these seven documents
in this order:

1. **[PROJECT_STATUS.md](PROJECT_STATUS.md)** -- the single authoritative
   "where things stand today" document: completed milestones, current
   Selector v2 status, known issues, active protected artifacts/processes,
   scientific blockers, next action.
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** -- code architecture: module map,
   key abstractions, execution flow.
3. **[BASELINES.md](BASELINES.md)** -- the exact policy/baseline inventory:
   20 historical policies, the 8-policy Selector v2 action space (Option B),
   6 faithful external baselines and why they're evaluation-only.
4. **[SELECTOR_V2.md](SELECTOR_V2.md)** -- the full Selector v2 research
   narrative in chronological order, from the objective bug fix through the
   most recent pilot's mixed held-out result.
5. **[EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md)** -- what's
   actually committed to git vs. local-only, and the status of every major
   result family.
6. **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** -- environment setup, test
   running, local-GPU vs. Wulver-A100 workflows.
7. **[NEXT_STEPS.md](NEXT_STEPS.md)** -- the exact next recommended research
   step, with explicit stop conditions.

## What this set is (and isn't)

These seven documents synthesize and link to ~75 detailed design/audit
documents elsewhere in `docs/` -- they do not replace them. When a detailed
doc and one of these disagree on a specific number or claim, treat the
detailed doc as authoritative for that specific claim and consider this set
out of date (and worth fixing).

`docs/README.md` is the full legacy documentation index (historical +
current, ~75 documents). This `docs/current/` set is the fast path; the
legacy index is the exhaustive one.

`docs/research_status.md` and `docs/roadmap.md` are retained for historical
compatibility but are **not** the current-status authority --
[PROJECT_STATUS.md](PROJECT_STATUS.md) is.
