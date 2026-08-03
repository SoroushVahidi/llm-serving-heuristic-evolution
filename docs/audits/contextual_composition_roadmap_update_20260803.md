# Roadmap Update Report: CC4b/CC5 Retry Status Documentation Pass

Date: 2026-08-03
Branch: `contextual-compositional-heuristics-20260731`
Starting SHA: `c17208079ef50368103f1feca992ac91f52ff4cb`
This is a **documentation-only** query: no scheduling code, model code,
experiment logic, or configs were modified, and the running CC4b/CC5 retry
job was not stopped, restarted, or otherwise interfered with.

## 1. Running-Job Status (Observed, Not Modified)

At the start of this query, the CC4b dataset build (tmux session
`cc4b_cc5_retry`, PID 3093543) was running at 97-98% progress
(3540-3604/3604 simulator executions). It was checked read-only at several
points during this documentation pass and was observed to **complete
naturally** (3604/3604, `heartbeat.json` stage `complete`,
`manifest.json` written) partway through this query -- this was not caused
by any action taken here; it was already most of the way done when this
query began. No files under `results/cc4b_oracle_composition_expansion/`
or `logs/cc4b_cc5_retry_*.log` were edited by this query.

Per this query's explicit scope, the next steps for that dataset (quality
gates, CC5 rerun, retry-verdict interpretation, and the
`contextual_composition_cc4b_cc5_retry_report_20260803.md` report itself)
are **not** performed here -- they remain queued as separate follow-up work
(already tracked from the query that launched the CC4b build). Every
cross-reference to `contextual_composition_cc4b_cc5_retry_report_20260803.md`
added by this documentation pass is a forward-looking link to a report that
does not exist yet as of this commit; it will be created when that
follow-up work runs.

## 2. Documents Read

`docs/START_HERE_CONTEXTUAL_COMPOSITION.md`,
`docs/RESUME_CONTEXTUAL_COMPOSITION.md`,
`docs/contextual_composition_roadmap.md`,
`docs/contextual_composition_decisions.md`,
`docs/CONTEXTUAL_COMPOSITION_BRANCH.md`,
`docs/audits/contextual_composition_cc5_predictor_report_20260803.md`,
`docs/audits/contextual_composition_cc4_oracle_dataset_report_20260803.md`,
`configs/cc4b_oracle_composition_expansion.yaml`, the CC4b checkpoint/
heartbeat files, `logs/cc4b_cc5_retry_*.log`, GitHub issues #1-#6,
`README.md`, `docs/README.md`, `docs/current/README.md`, `docs/roadmap.md`.
No historical audit report was rewritten; all edits either updated
canonical "current state" documents or appended new sections/paragraphs
preserving prior text as history.

## 3. Documents Updated

* `docs/contextual_composition_roadmap.md` -- YAML marker (`current_status:
  IN PROGRESS`, `roadmap_version: 6`, new `next_action`); status table
  (CC5 -> `IN PROGRESS`, CC6/CC7/CC8 -> `BLOCKED`, with evidence links to
  both the CC5 report and the forthcoming CC4b/CC5 retry report); CC5 phase
  section (added a "Retry in progress" paragraph after the preserved
  first-attempt gate result); two new sections: **Current Scientific
  Interpretation** (summarizing CC1b/CC4/CC5-attempt-1 findings and the
  three possible CC4b/CC5-retry outcomes) and **Future Research Directions
  -- Not Yet Implemented** (envelope-aware usefulness, regret-profile
  complementarity, behavioral embeddings, typed module-level crossover,
  QD/MAP-Elites library expansion, LLM-guided symbolic synthesis, symbolic
  distillation -- explicitly labeled as unimplemented).
* `docs/START_HERE_CONTEXTUAL_COMPOSITION.md` -- new "Active Task Right
  Now" section (tmux session, exact commands, checkpoint/log paths, what
  not to start in parallel, exact next decision, active issue) placed
  first so a new contributor understands state in under two minutes;
  updated "Exact next task" and "What Not To Do Yet".
* `docs/RESUME_CONTEXTUAL_COMPOSITION.md` -- new CC4b/CC5-retry checkpoint
  SHA placeholder line; "Current Phase" updated to `IN PROGRESS`; "Exact
  Next Implementation Task" now points at the (forthcoming) retry report
  first; new "CC4b/CC5 Retry Evidence" section; GitHub section updated
  (issue stays open, issue #6 dependency noted correct).
* `docs/CONTEXTUAL_COMPOSITION_BRANCH.md` -- new "Current high-level
  status" lead paragraph describing the in-progress retry; Query Sequence
  extended with Query 12 (IN PROGRESS); Guardrail and Next Action rewritten
  to point at the retry report and forbid parallel runs.
* `docs/contextual_composition_decisions.md` -- new **CCD-016** entry (see
  §4).
* `README.md`, `docs/README.md`, `docs/current/README.md`,
  `docs/roadmap.md` -- the contextual-composition pointer block in each
  was stale ("intentionally paused after CC1b and before CC2
  implementation" -- inaccurate since CC2, CC3, CC4, and CC5's first
  attempt all happened since). Updated to state CC1-CC4 complete, CC5 `IN
  PROGRESS`, with links to the CC5 report, the (forthcoming) CC4b/CC5
  retry report, and issue #5 -- without duplicating roadmap content.
* `scripts/check_contextual_composition_status.py` and
  `tests/test_contextual_composition_status_checker.py` -- see §5.

## 4. Decision Recorded

**CCD-016**: approves the CC4b targeted expansion as the response to
CCD-015's `INCONCLUSIVE` finding -- rerun CC5's pipeline completely
unchanged against the expanded dataset before any model redesign; CC6
stays blocked until the retry verdict resolves the CC5 gate. Full text in
`docs/contextual_composition_decisions.md`.

## 5. Checker Changes

`scripts/check_contextual_composition_status.py`:

* `REQUIRED_MARKER` and `check_status_table`'s `expected` status-table
  dict updated for CC5 `IN PROGRESS` / CC6-CC8 `BLOCKED`.
* **Bug found and fixed**: `check_status_table`'s old invariant only ever
  counted phases with literal status `"NEXT"`; generalized to `active_count`
  (phases with status in `{"NEXT", "IN PROGRESS"}` must total exactly 1),
  since CC5 legitimately carries `IN PROGRESS` now, not `NEXT`.
* **A second, related bug found and fixed**: `check_no_cc2_in_progress()`'s
  forbidden-pattern list included two CC2-*un*anchored patterns (a bare
  `"current_status: IN PROGRESS"` / `` "Current status: `IN PROGRESS`" ``)
  that would have falsely fired against every canonical document the
  moment CC5's own legitimate `IN PROGRESS` status landed. Fixed by
  removing the two unanchored patterns and keeping only the three
  CC2-specific ones (`"CC2 is IN PROGRESS"`, `"CC2 has started"`, the CC2
  table-row pattern). Caught by actually running the checker against the
  edited docs, not by inspection.
* New checks, wired into `main()`: `check_no_cc6_active()` (CC6 must never
  be marked active while CC5 is unresolved -- both a table-row check via
  `check_status_table` and a prose-pattern check via this new function);
  `check_start_here_and_resume_name_same_current_task()`; `
  check_future_work_labeled()` (roadmap must contain the "Future Research
  Directions -- Not Yet Implemented" heading and its "not yet implemented"
  disclaimer); `check_active_issue_referenced()`; `check_cc4b_retry_linked()`
  (every canonical file must reference `cc4b_cc5_retry` or
  `cc4b_oracle_composition_expansion`); `check_no_stale_final_cc5_verdict_claim()`
  (guards against a canonical doc reverting to describing the first,
  INCONCLUSIVE CC5 attempt as final).
* Two now-stale required-string checks inside `check_resume_readiness_extra()`
  (checking for the pre-retry `current_status: NEXT` / `"must be
  **retried**, not begun fresh"` phrasing) were updated to match the new
  text.

`tests/test_contextual_composition_status_checker.py`: renamed/updated
`test_roadmap_links_cc1b_report_and_has_cc5_next` ->
`..._has_cc5_active` (checks `IN PROGRESS` + exactly-one-active
invariant); fixed the same CC2-in-progress false-positive bug in the
test's own forbidden-pattern list; added `test_canonical_docs_do_not_make_cc6_active`,
`test_start_here_and_resume_name_same_current_task`,
`test_future_work_is_labeled_not_implemented`,
`test_no_canonical_doc_claims_first_cc5_result_is_final`,
`test_canonical_docs_reference_active_issue_and_cc4b_retry` (5 new tests).

## 6. Issue Updates

GitHub issue #5: commented with CC4b build completion (3604/3604), the
reason for expansion, the quality gates that must pass before retraining,
the exact post-build sequence, and an explicit statement that CC6 remains
blocked. Left **open** (not closed -- the retry verdict is not yet
determined). Issue #6 reviewed: its body already correctly states
"Blocked on CC5 and its deployable-model decision gate" -- no update
needed.

## 7. Unresolved Risks

* The CC4b/CC5 retry report this documentation pass links to
  (`contextual_composition_cc4b_cc5_retry_report_20260803.md`) does not
  exist yet -- every link to it is forward-looking. `check_cc4b_retry_linked()`
  checks for the substring `cc4b_cc5_retry`/`cc4b_oracle_composition_expansion`
  (already satisfied via the tmux-session name and config path), not for
  the report file's own existence, so the checker will not fail before
  that report is written; once it is, the roadmap/START_HERE/RESUME status
  table and "Active Task Right Now" sections should be updated again to
  reflect the actual verdict rather than "in progress."
* This pass could not run the CC4b quality gates or the CC5 rerun (out of
  scope by explicit instruction), so the actual retry verdict remains
  unknown as of this commit -- the roadmap's "Current Scientific
  Interpretation" section states three possible outcomes without
  presupposing which one obtains.
* `roadmap_version` was bumped to 6 for this documentation-only change;
  if the follow-up query that completes the CC4b/CC5 retry also touches
  the roadmap version, confirm it bumps forward from 6, not backward.

## 8. Exact Next Action After The Running Experiment Completes

The CC4b build has already completed (see §1). The next action, out of
scope for this query, is:

1. Run `scripts/check_cc4b_quality_gates.py results/cc4b_oracle_composition_expansion/<TS>`
   and confirm all gates pass (held-out count >=50, non-near-tie held-out
   >=20, no single oracle-composition family dominating >70%, split
   integrity, completion-accounting consistency).
2. If gates pass, rerun CC5's existing, unchanged pipeline:
   `scripts/run_cc5_contextual_predictor.py --dataset-dir
   results/cc4b_oracle_composition_expansion/<TS> --full-run`.
3. Interpret the verdict against the decisive comparison (contextual
   predictor vs. best global verified composition) per the roadmap's
   "Current Scientific Interpretation" section, write
   `docs/audits/contextual_composition_cc4b_cc5_retry_report_20260803.md`,
   and update the roadmap/START_HERE/RESUME/decisions/branch-marker/issue
   #5 to reflect the actual outcome -- queuing CC6 only if the verdict is
   `PROCEED`.
