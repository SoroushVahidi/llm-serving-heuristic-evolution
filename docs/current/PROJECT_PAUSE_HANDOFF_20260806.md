# Project Pause Handoff — 2026-08-06

**Why paused:** the user is temporarily switching to another project and
wants this repository left in a durable, self-explanatory, safely-resumable
state before stepping away — not a response to any failure or incident.

---

## Authoritative branch / SHA

| Field | Value |
|---|---|
| Branch | `contextual-compositional-heuristics-20260731` |
| Remote | `origin/contextual-compositional-heuristics-20260731` |
| SHA at pause (verify with `git rev-parse HEAD`) | Will equal the SHA of this query's own commit — see `docs/audits/project_pause_documentation_query3_20260806.md` for the exact value |
| Expected state on resume | Working tree clean, 0 ahead / 0 behind `origin` |

---

## What was completed immediately before pausing

This pause was preceded by a deliberate four-query reconciliation sequence
(2026-08-06), not by the research work itself stopping mid-task:

1. **Query 1 (audit):** full repository/worktree/branch/uncommitted-work
   inventory; confirmed the repository was already git-clean; found a
   duplicate, superseded dirty worktree (`phase2b9`); found an uncommitted
   Llumnix research audit sitting in `/tmp`; found `docs/BASELINE_STATUS.md`
   stale for Apt-Serve and Llumnix.
2. **Query 2 (reconciliation):** diagnosed the Wulver SSH/Kerberos
   authentication failure precisely (without resolving it); integrated the
   `/tmp` Llumnix audit into the repository with independent re-verification
   (188/188 tests re-run, not just re-read); corrected
   `docs/BASELINE_STATUS.md`; added 5 decision-log entries; committed and
   pushed (`e413ba1`).
3. **Query 3 (this query, documentation):** consolidated multiple competing
   "start here" documents into one canonical entry point
   (`docs/current/RESUME_HERE.md`); created a stable project map, a
   per-workstream status table, an ordered next-actions list, and a
   scientific-decisions summary; reconciled stale status text across
   `README.md`, `docs/README.md`, `docs/current/README.md`.

The actual research work these queries are reconciling around — CC0–CC5
(complete), the Apt-Serve audit, and the Llumnix audit — was completed in
the days immediately prior (2026-08-03 through 2026-08-06); see
`docs/current/RESUME_HERE.md` §C for the summary.

---

## Unresolved local tasks

See `docs/current/NEXT_ACTIONS.md` §"IMMEDIATE LOCAL ACTIONS" for the full,
ordered list. In brief: Llumnix stress-test coverage, Llumnix comparative
evaluation, independent re-verification, classification, then an
external-baseline sufficiency checkpoint before CC6 is even reconsidered.

## Deferred Wulver work

**No Wulver access should be attempted as part of this local finalization
sequence.** Wulver reconciliation will happen later from a direct Wulver
login, not from this workstation. See `docs/current/NEXT_ACTIONS.md`
§"DEFERRED WULVER ACTIONS" and `docs/current/RESUME_HERE.md` §E/§H for the
exact ordered steps once that login is available: authenticate, inspect the
Apt-Serve probe state, submit the prepared CPU probe if not already
submitted, inspect `squeue`/`sacct`, collect compact logs/manifests, decide
Strategy C vs. D from executed evidence, commit/push, then synchronize this
branch.

---

## Known blockers

- **Wulver SSH/Kerberos authentication fails** even with a locally valid
  ticket and a successfully obtainable service ticket (`kvno` succeeds).
  Diagnosed in detail (correct alias/username, GSSAPI exchange rejected
  server-side, clock sync ruled out, one unconfirmed lead around duplicate
  ticket-cache entries) but not resolved — this needs either an interactive
  `kdestroy && kinit` cycle by the user or HPC-support involvement, neither
  of which a non-interactive local session can do. No Kerberos secrets,
  passwords, or Duo details are recorded anywhere in this repository's
  documentation, per standing project convention.
- **Apt-Serve Strategy C vs. D is undetermined** — blocked entirely on the
  above.

## Query 4 final closure facts (added 2026-08-06, after this note's initial version)

- **Worktree cleanup completed.** `.claude/worktrees/phase2b9` was removed
  (`git worktree remove --force` + `git worktree prune`) after re-verifying
  it was safe byte-for-byte, not just re-stating Query 1/2's conclusion —
  see `docs/audits/project_pause_final_closure_query4_20260806.md` for the
  exact evidence (a minor correction there: the two modified tracked files
  were superseded earlier drafts, not literally byte-identical as
  previously summarized; the safety conclusion is unchanged). Its local
  branch (`worktree-phase2b9`) was intentionally left in place.
- **Full non-live/non-GPU test suite result:** 3455 passed, 1 failed (see
  next bullet), 17 skipped, 21 deselected, ~6 minutes.
- **One known, pre-existing, unrelated failure remains:**
  `tests/test_decode_prefill_contention_execution.py::TestLegacyModeUnchanged::test_existing_yaml_configs_do_not_set_new_field`,
  latent since 2026-08-03 (a CC4b research config sets a key the test's
  guard logic, written 2026-07-20, didn't anticipate) — confirmed via git
  history to be untouched by any commit in this pause sequence. Not fixed
  in this query (out of scope); a future query should apply a one-line test
  fix. Full causal analysis in
  `docs/audits/project_pause_final_closure_query4_20260806.md`.
- **No active local jobs** at Query 4's close — confirmed via tmux/process
  sweep, same result as Queries 1–3.
- **Final clean/sync confirmation:** working tree clean, 0 ahead / 0 behind
  `origin/contextual-compositional-heuristics-20260731` after this query's
  commit(s) — verify live with `git status --short --branch`, do not trust
  a cached SHA from this document.

## Known simulator limitations

- Sarathi-Serve's real-hardware decode-protection mechanism does not
  reproduce inside this simulator under FCFS-strict admission (structural,
  documented, not being chased further — see `docs/current/SCIENTIFIC_DECISIONS.md`).
- Llumnix's migration-bandwidth and concurrent-transfer-contention behavior
  are only partially representable (flat delay scalar, no true bandwidth or
  shared-link-contention model) — two small, localized extensions
  (~150–300 LOC total) would close this if ever needed.
- The simulator's default decode rate is known to run substantially faster
  than hosted real-LLM providers in the Cohere/Gemini pilot comparison, with
  no TTFT analogue — documented, not treated as a blocking defect.

## Active jobs at pause time

**None.** No tmux sessions, no local Python/SLURM processes related to this
project, no Wulver job (as far as verifiable from this workstation — see
"deferred Wulver work" above for why that specific claim is bounded). No
file in the repository tree changed outside of this reconciliation
sequence's own commits.

## Worktrees requiring Query 4 cleanup

**Done.** `.claude/worktrees/phase2b9` was removed in Query 4 — see "Query 4
final closure facts" above. No worktrees remain except the main one.

---

## Exact resumption procedure

```bash
cd /home/soroush/llm-serving-heuristic-evolution
git fetch origin
git status --short --branch          # expect: clean, 0 ahead / 0 behind
less docs/current/RESUME_HERE.md     # read this first, in full
```

Then read, in order: `docs/current/RESUME_HERE.md` →
`docs/current/PROJECT_MAP.md` (navigation) →
`docs/current/WORK_STATUS.md` (per-workstream table) →
`docs/current/NEXT_ACTIONS.md` (what to actually do next).

## First three tasks after return

1. Re-verify this handoff note is still accurate (`git log` since this
   commit — if anything changed, this note is stale and
   `docs/current/RESUME_HERE.md` should be re-checked against current state
   before trusting either).
2. Pick up **"Llumnix stress-test coverage and first comparative
   evaluation"** (`docs/current/RESUME_HERE.md` §G) — the cheapest,
   most load-bearing local gap identified in this pause sequence.
3. Separately, when a direct Wulver login is available: **"Execute Apt-Serve
   Strategy C CPU probe and decide Strategy C vs. D"**
   (`docs/current/RESUME_HERE.md` §H) — this does not need to happen before
   task 2, and does not need to happen from this workstation.
