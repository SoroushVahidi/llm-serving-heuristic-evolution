# Project Pause Reconciliation — Query 2 of 4

**Date:** 2026-08-06
**Scope:** Reconciliation and documentation only. No new baseline started, no
Apt-Serve/Llumnix implementation performed, no CC5/CC6 changes, no new
comparative experiments run, no worktree removed. Follows
`docs/audits/project_pause_reconciliation_query1_20260806.md`.

---

## Starting and ending state

| Field | Value |
|---|---|
| Starting SHA | `f967c095826900aed0eb0326d3d1f3ea60936261` |
| Remote at start | Identical (0 ahead/0 behind) — `git fetch origin` at the start of this query brought no new commits |
| Files changed | 5 (3 modified, 2 new — all documentation) |
| New code/config/test changes | None |

---

## Apt-Serve/Wulver evidence

No SLURM job has ever been submitted (re-confirmed: commit message, empty
`results/wulver_imports/`, no job ID in git history or docs). This query's
contribution is a more precise authentication diagnosis than Query 1 had:

- Query 1's SSH attempt used the bare FQDN (`login02.tartan.njit.edu`), which
  bypasses `~/.ssh/config`'s `Host login02` alias (`User sv96`) and silently
  authenticated as the wrong local Linux user (`soroush`). This query retried
  correctly via the `login02` alias as `sv96` — **still failed**, confirming the
  problem is not username selection.
- `klist` / `kvno host/login02.tartan.njit.edu` confirm a Kerberos ticket-granting
  ticket is valid and that a service ticket for the Wulver login node **is**
  obtainable (`kvno = 41`). Kerberos credential acquisition itself works.
- `timedatectl` confirms NTP sync — clock skew is ruled out as a cause.
- `ssh -vvv` shows the client correctly offering `gssapi-with-mic`; the server
  responds by re-listing the same continuable-methods set (i.e., rejecting the
  attempt) twice, then exhausts authentication methods. Server-side rejection of
  a client that has valid credentials and correct addressing is the residual,
  unexplained fact.
- One unconfirmed lead: the local ticket cache holds two service-ticket entries
  for the same host principal (one with an apparently empty/referral realm, one
  fully qualified) — plausible but not verified as the cause, since confirming it
  would require another live authentication attempt, which this task's
  instructions say to avoid repeating without cause.
- **No further authentication attempts were made beyond what was needed for this
  diagnosis** (2 attempts this query, 1 in Query 1 — 3 total across the sequence).

**Strategy C/D status: unchanged — explicitly unresolved.** No code-reading-based
guess was made (`CCD-022`). **Evidence class for "no remote job exists":**
verified local + verified remote git evidence (both show nothing); Wulver-side
squeue/sacct could not be queried directly (auth failure), so **a Wulver-side
Claude session having submitted and completed a job entirely outside this
workstation's visibility remains formally unresolved** (`CCD-023`) — this
snapshot does not claim otherwise.

**Exact next action:** resolve the Wulver SSH/GSSAPI failure (most likely
requires an interactive `kdestroy && kinit` cycle by the user, or HPC-support
involvement for server-side GSSAPI/keytab issues) — this is a prerequisite for
literally all further Apt-Serve work.

---

## Llumnix audit integration

`/tmp/llumnix_official_artifact_audit_20260806.md` (52,878 bytes, produced
2026-08-05 23:33 by a separate, parallel research pass) was read in full and
independently re-verified against the current repository before being
committed:

| Claim in the `/tmp` audit | Independent re-verification this query | Result |
|---|---|---|
| `llumnix_faithful.py` exists, faithful reimplementation | `find`/`grep` confirmed: `src/llmserveopt/policies/llumnix_faithful.py`, registered in `external_baselines_registry.py` | Confirmed |
| Official pin `alibaba/llm-scheduling-artifact@a908243`, Apache-2.0 | Cross-checked against `docs/llumnix_faithful_scheduler_reference.md` (pre-existing, 2026-07-18) and the registry's embedded pin string | Confirmed |
| 36 fidelity tests, all pass | `pytest tests/test_llumnix_faithful_scheduler.py --collect-only` → 36 collected; `pytest tests/test_llumnix_faithful_scheduler.py tests/test_external_baseline_integration.py` → **188 passed, 0 failed** (ran the full pair since the Llumnix tests include regression checks against 5 other faithful baselines) | Confirmed, and strengthened (actual pass, not just collection) |
| No comparative evaluation exists | Searched `experiments/`, `docs/audits/`, `results/` for Llumnix — none found | Confirmed |
| No stress-test catalog entries | Checked `configs/stress_tests/algorithm_stress_test_catalog.yaml`, `docs/research/algorithm_stress_tests/STRESS_TEST_CATALOG.md` — zero entries; one candidate-identification row in `ALGORITHM_INVENTORY_20260805.md` | Confirmed |
| `docs/BASELINE_STATUS.md` is stale | `git log -1 -- docs/BASELINE_STATUS.md` shows last touch was an unrelated Sarathi commit (`74c7fc8`), predating nothing Llumnix-related; the row said "Unverified in this pass... Not prioritized" | Confirmed |
| `docs/current/BASELINES.md` reflects a more advanced state | Read the file directly — already says "Execution-health clean" for `llumnix_faithful`, correctly, since 2026-07-23 | Confirmed |

All claims held up. The document was copied into
`docs/audits/llumnix_official_artifact_audit_20260806.md` with: the
machine-local "Isolation" line's absolute path removed; a new integration note
added at the top recording the independent re-verification performed this
query (including the strengthened 188/188 pass result); the closing
terminal-style "PARALLEL AUDIT RESULT" block replaced with prose §17/§18/Summary
sections matching this repository's existing audit-doc conventions; every other
factual claim, citation, and classification retained unedited.

---

## Status corrections

- **`docs/BASELINE_STATUS.md`:** Llumnix row rewritten with full paper/pin/license
  citation, implementation status `Complete`, fidelity `Faithful`, evaluation
  status `Not run`, classification `UNESTABLISHED pending evaluation`. Apt-Serve
  row rewritten with paper/pin/license citation, implementation status
  `Not implemented` (unchanged — still true) but corrected from "Not
  prioritized" to reflect the actual audit + probe-prep work done, Wulver
  status `REMOTE_STATE_UNVERIFIED` with the exact blocker attached.
- **`docs/external_baseline_decision.md`:** added a dated update note above
  the historical "Cite Only / Out of Scope" table (§D) flagging that its
  Sarathi-Serve and Llumnix rows are superseded by later work, without rewriting
  the historical table itself — it remains an accurate record of the original
  first-paper scoping decision.
- **`docs/current/BASELINES.md`, `docs/baselines.md`:** no changes — both
  already correctly describe `llumnix_faithful`. Checked for stale Apt-Serve
  mentions — none exist in either file (Apt-Serve had no prior mentions there
  to correct).
- **`docs/roadmap.md`:** checked — only references Llumnix in a doc-index list
  (`docs/*_faithful_scheduler_reference.md`), not a status claim; no change
  needed.
- **Historical, dated audit docs** (`docs/audits/apt_serve_official_artifact_audit_20260805.md`,
  `docs/audits/branch_and_pars_readiness_audit_20260804.md`,
  `docs/research/algorithm_stress_tests/ALGORITHM_INVENTORY_20260805.md`) still
  contain "Not implemented"/"Not prioritized" language for Apt-Serve — **left
  unedited**, since these are point-in-time snapshots that were accurate when
  written (the 2026-08-05 Apt-Serve audit doc explicitly says so itself, in its
  own §13: "not stale or incorrect... a future implementation task should
  update it"), analogous to not rewriting prior decisions in a decision log.
  `docs/BASELINE_STATUS.md` is the living, corrected index; these are provenance.

---

## Decision-log additions

Five new entries added to `docs/contextual_composition_decisions.md`
(the only decision log in this repository), `CCD-020` through `CCD-024`, with
an explicit scope note that they are not CC-roadmap-phase decisions but this
branch's only available decision log:

- **CCD-020** — recognize the existing Llumnix faithful implementation instead of treating it as green-field
- **CCD-021** — passing fidelity tests are not comparative validation (general principle, not Llumnix-specific)
- **CCD-022** — preserve Apt-Serve Strategy C/D as unresolved without executed evidence
- **CCD-023** — remote Wulver state must be verified independently of local visibility
- **CCD-024** — reconcile and document before further baseline implementation

No prior decisions (`CCD-001` through `CCD-019`) were modified.

---

## Duplicate worktree

Re-inspected `.claude/worktrees/phase2b9` (branch `worktree-phase2b9`, HEAD
`429e96e`). Re-confirmed unchanged from Query 1: all 6 dirty/untracked paths
carry identical 2026-06-25 19:39-19:48 timestamps and are byte-present on the
already-merged `phase2b9-selector-robustness-and-suite-freeze` branch (commit
`5fe977b`, ~15 minutes later the same evening). Recorded in
`docs/current/PROJECT_SNAPSHOT_20260806.md`'s closing section as safe for
Query 4 to remove after final validation. **Not removed this query.**

---

## Validation

| Check | Result |
|---|---|
| `python -m compileall -q src scripts tests` | Clean, no errors |
| `python scripts/check_contextual_composition_status.py` | PASSED |
| `pytest --collect-only -q` | 3488 tests collected, 0 errors (unchanged from Query 1 — no code touched) |
| Focused: `pytest tests/test_llumnix_faithful_scheduler.py tests/test_external_baseline_integration.py` | **188 passed, 0 failed** |
| `git diff --check` | No whitespace errors |
| `git diff --stat` | 3 files modified (190 insertions, 4 deletions), 2 new files, all documentation |
| Private-path scan on new/changed files | One absolute path found and fixed (`docs/current/PROJECT_SNAPSHOT_20260806.md`, made repository-relative); one `/tmp` reference retained intentionally as historical provenance narrative (not a live dependency) in the decision log |
| Secret scan on new/changed files | None found |
| `--resume-readiness` | Deferred to after commit, per task instructions (run only once the tree is clean post-commit) |

---

## Unresolved items for Query 3 and Query 4

- Wulver SSH/GSSAPI authentication is still broken — needs either user-run
  interactive Kerberos ticket-cache cleanup or HPC-support involvement; not
  resolvable by further non-interactive diagnosis.
- Apt-Serve Strategy C/D remains genuinely undetermined pending actual Wulver
  execution.
- Llumnix comparative evaluation (Phase F) has not been run — cheapest
  remaining gap identified across this week's two baseline audits.
- The `phase2b9` duplicate worktree remains present, unremoved, for Query 4.
- Query 3 should decide whether `docs/current/RESUME_HERE.md` (which still
  frames its "first three actions" around the 2026-07-25 pause rather than the
  current CC6-restricted/Apt-Serve-blocked state) should be superseded,
  updated, or explicitly marked historical in favor of
  `docs/START_HERE_CONTEXTUAL_COMPOSITION.md` — both currently exist and a
  future resuming session could be pointed at either.
