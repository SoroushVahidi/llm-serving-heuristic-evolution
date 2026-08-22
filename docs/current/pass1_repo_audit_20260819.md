# Repository Polish — Pass 1 Audit (2026-08-19)

Read-only hygiene/status/roadmap audit. No files modified, no git state changed,
no commits made, no running jobs touched. Produced for Pass 2/Pass 3 planning only.

---

## A. Executive Summary

The scientific lineage (WS-P policy separation → composition falsification →
MF-PSD → unified utility matrix → three selector NO_GOs → cross-family
transfer reassessment → hierarchical regime router → live re-evaluation →
Family-B live replication prep) is real, well-audited, and internally
consistent at the `docs/audits/*` layer. The problem is entirely in the
**living-doc layer**: the four files that are supposed to be authoritative
current-state pointers (`README.md`, `docs/PROJECT_MAP.md`,
`docs/current/WORK_STATUS.md`, `docs/current/EXPERIMENT_INDEX.md`) are stale
by between 2 days and roughly one month, in that order of severity, while
`docs/current/RESUME_HERE.md`/`NEXT_ACTIONS.md` are current to within one
commit. `docs/PROJECT_MAP.md` — the doc explicitly ranked #2 in the
documentation-authority list, above `RESUME_HERE.md` — is the most stale of
the tier-1 docs (frozen at "after MF-PSD v1 build," i.e. it predates Step
2/Step 3, all three NO_GOs, hierarchical routing, and the live re-evaluation
entirely). This inversion (lower-authority doc more current than
higher-authority doc) is the single highest-value Pass-2 fix.

**Most urgent finding, outside the audit's normal scope but too important to
omit:** a second, concurrent Claude Code session is actively writing to this
same working tree right now, building its own colliding version of the
public-trace-corpus work under different paths (`src/llmserveopt/workloads/public_trace_corpus.py`,
`scripts/build_public_trace_corpus_v1.py`, `tests/test_public_trace_corpus_v1.py`,
`docs/design/PUBLIC_TRACE_CORPUS_V1.md`, `data/public_trace_corpus_v1/`), and
a separate, currently-running background process (`src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py`,
pid 80136) for a `DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1` task that has no
matching entry anywhere in `docs/current/`. Neither of these is part of this
audit's own scope, but their existence — and the apparent disappearance of a
different in-progress session's own uncommitted `public_trace_corpus` package
files from disk — is a live coordination hazard that should be surfaced to
the user immediately, independent of Pass 2/3 sequencing. See §C and §O.

No secrets, tracked `.env`, or tracked raw third-party data were found. One
large tracked CSV (~30MB) exists but appears intentional (canonical per-step
telemetry evidence for the online-regime-signal study), not accidental.

---

## B. Current Branch / HEAD / Upstream

| Field | Value |
|---|---|
| Repository root | `<repo-root>` |
| Branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `9d8f997fb2c7d29891e5a5ef5da1558e364d3c7d` |
| HEAD subject | `feat: prepare Family-B live replication harness` |
| HEAD date | 2026-08-19 00:29:17 -0400 |
| Upstream | `origin/contextual-compositional-heuristics-20260731` |
| Ahead/behind | 0 / 0 (up to date with upstream at audit time) |
| Remote | `origin` = `https://github.com/SoroushVahidi/llm-serving-heuristic-evolution.git` (fetch+push) |
| Tags | `pause-2026-07-25` (only tag) |
| Stashes | none |
| Worktrees | only the main worktree (`worktree-phase2b9` is a **branch name**, not an actual `git worktree`) |

The current branch does contain the full recent scientific lineage described
in `RESUME_HERE.md` through the `9d8f997` HEAD (verified via `git log`
subject and the audit-doc cross-references in §J); it is 1 commit ahead of
what `RESUME_HERE.md`'s own prose describes (see §H).

---

## C. Running-Job Protection Inventory

`tmux list-sessions` → **no server running** (no tmux sessions at all).

Relevant processes found via `ps aux`:

| PID | Command | Started | Status | Scientifically important? | Must-not-touch path |
|---|---|---|---|---|---|
| 80131 / 80136 | `python3 -c "... llmserveopt.analysis.decision_criticality_timescale_trainval_v1 ... fit_frozen_models() / run_scenario_diagnostic(...)"` | 14:24:50 | Running (80136 at ~113% CPU) | Unknown to this audit — no corresponding entry exists anywhere in `docs/current/`; appears to belong to a different, concurrent session/task not described in this audit's brief | `src/llmserveopt/analysis/`, `docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md`, whatever it reads under `experiments/mf_psd_v1/` (read-only) |
| 1730 | `uvicorn njit_auditor.web_app:app` | Aug 18 | Long-running, unrelated web service (`njit_auditor`) | Not part of this repo's experiment lineage | N/A |
| 1562 | `unattended-upgrade-shutdown` | Aug 18 | System process, unrelated | N/A | N/A |

**Public-trace corpus build status:** no tmux session or background build
process for a public-trace corpus exists at the OS level. However, the git
working tree shows an **in-progress, uncommitted, and apparently
double-implemented** version of this work (see §D, §O). Per the audit brief,
this is a separate, still-in-flight coordinator task; this Pass-1 audit does
not judge its code, only reports its presence and the file-collision risk.

**Action taken:** none. No process was killed, restarted, attached-to, or
polled repeatedly. No file was moved or deleted.

---

## D. Working-Tree State

```
git status --short
 M experiments/family_b_balanced_replication_v1/run_smoke_synthetic_results.json
 M experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json
?? data/public_trace_corpus_v1/
?? docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md
?? docs/design/PUBLIC_TRACE_CORPUS_V1.md
?? scripts/build_public_trace_corpus_v1.py
?? src/llmserveopt/analysis/
?? src/llmserveopt/workloads/public_trace_corpus.py
?? tests/test_public_trace_corpus_v1.py
```

- **Staged changes:** none.
- **Unstaged (tracked-file) changes:** 2 files, both re-run provenance
  timestamp/`git_head_sha` bumps (see below) — not scientific content
  changes.
- **Untracked files:** 7 paths, all new work-in-progress from concurrent
  sessions (public-trace-corpus attempt #2 and the decision-criticality
  analysis task); none of these are mine to judge in this audit.
- **Ignored-but-relevant generated files observed:** `run.log`, `crash.log`
  (both root-level, correctly matched by `*.log`), `.coverage` (correctly
  ignored), `__pycache__/` (correctly ignored).

**The two modified tracked files** are self-updating provenance stamps
(`git_head_sha`, `run_timestamp_utc`) inside JSON result files that get
rewritten every time their owning script/test is re-executed
(`experiments/family_b_balanced_replication_v1/run_smoke_synthetic_results.json`,
`experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json`).
Diff inspected: only the two provenance fields changed, no metric/result
values changed. Not cleaned per the read-only mandate; reported as found.

**Does the branch contain everything expected from the recent scientific
lineage?** Yes for the tracked/committed lineage — `git log` confirms all
commits described in `RESUME_HERE.md` through `9d8f997` are present, and
`docs/audits/*` dates line up with the commit lineage. The **untracked**
in-progress public-trace-corpus files are a separate, currently-unresolved
question (§O).

---

## E. Branch / Worktree Hygiene Findings

51 local branches, 30 remote branches. `git branch --merged HEAD` shows
**49 of 51 local branches are already fully merged** into the current
branch. Only 2 are not merged:

| Branch | Unique commits (not in current) | Classification |
|---|---|---|
| `phase2b13-selector-training-after-diversity` | 1 (`2474c07`) | `NEEDS_REVIEW` — sibling of the merged `phase2b13-selector-training-and-suspicion-audit`; likely an earlier, superseded attempt at the same phase, but not verified line-by-line in this pass |
| `phase2c-final-selector-improvement` | 3 (`31f5730`, `0703d5d`, `d4736bb`) | `NEEDS_REVIEW` — "Add causal advanced-selector formulations and Phase 2C evaluation tooling" is a nontrivial, possibly still-relevant unmerged feature; do not archive without checking whether its content was independently reimplemented later |

All other 49 merged branches (phase2a1…phase2b16, selector-v2-*,
selector-dataset-v2*, baseline-*-faithful, backup/*, repo-polish-query*,
`main`, `worktree-phase2b9`, `wulver-*`, `external-baseline-integration`,
`fix-side-effecting-scripts`, `simulator-decode-prefill-contention-fix`,
`reality-grounded-dataset-expansion-20260724`): `SAFE_CANDIDATE_FOR_LATER_ARCHIVE_OR_DELETE`
— every commit they contain is already reachable from the current branch tip,
so deleting the branch *pointer* (not the commits) is history-safe. `backup/*`
branches in particular are named as one-off safety snapshots and are the
clearest archive candidates. `main` itself is merged but should be
**`KEEP_ACTIVE`** regardless (it is the repository's nominal default branch,
per memory it is known to be stale relative to phase branches — that is a
separate, pre-existing, already-known condition, not new).

No worktrees exist beyond the primary one. No stashes.

**This pass did not delete or archive any branch**, per instructions.

---

## F. Top-Level Organization Findings

| Path | Classification | Notes |
|---|---|---|
| `p2_config.yaml`, `p3_chunk_control.py`, `p5_analysis_chunk_comp.py`, `p7_runner.py`, `p8_test_runner.py` | `LEGACY_BUT_REFERENCED` | All 5 are **git-tracked** (commits `bbe9339`, `16be179`), cross-reference each other by filename, and are referenced from `docs/CONTEXTUAL_COMPOSITION_BRANCH.md`, `docs/audits/family_b_v2_prefill_control_composition_falsification_20260817.md`, `configs/prefill_control_composition_smoke_v2.yaml`, and `tests/test_prefill_control_composition_v2.py`. This is the real, still-cited Family-B v2 PrefillControl composition-falsification pipeline, just placed at repo root instead of `scripts/`. Safe to leave; a later move into `scripts/` would require updating ≥5 cross-references and is `NEEDS_CARE`, not `NEEDS_REVIEW`. |
| `run.log` | `GENERATED` | Untracked, correctly `.gitignore`d (`*.log`). Default output path used by `p7_runner.py` and referenced by name in several `experiments/*/README.md` files and `tests/test_policy_separation_sobol_pilot.py`. Harmless. |
| `crash.log` | `GENERATED` | Untracked, 0 bytes, correctly ignored. Referenced (as evidence of "no crash") by `docs/audits/hierarchical_regime_router_live_reeval_v1_20260818.md`. Harmless, but a 0-byte file sitting at repo root is easy to mistake for clutter — candidate for a one-line note in Pass 2 docs, not deletion. |
| `opencode.json` | `JUSTIFIED_EXCEPTION` | Untracked; deliberately excluded via `.git/info/exclude` (machine-local exclude, not shared `.gitignore` — correct, since this is a local tool config for a different CLI ("opencode"), not something every clone should ignore). No action needed. |
| `.coverage` | `GENERATED` | Tracked-looking but actually untracked+ignored; stray pytest-cov artifact at root. Harmless, already ignored. |
| `.local_data/` | `GENERATED` | Ignored via `.gitignore`; local scratch root, consistent with its name. |
| `__pycache__/`, `.pytest_cache/` | `GENERATED` | Correctly ignored. |
| `.env.example` | `CANONICAL` | Tracked, referenced by `docs/api_provider_setup.md` and `docs/current/REPRODUCIBILITY.md`. Correct as-is. |
| `data/`, `dataset_staging/`, `external/`, `benchmarks/`, `baselines/`, `configs/`, `experiments/`, `results/`, `scripts/`, `src/`, `tests/`, `tools/`, `docs/` | `CANONICAL` | Standard, expected top-level layout matching `README.md`'s own "Repository Layout" section. |
| `logs/` | `CANONICAL` (gitignored) | Matches `README.md`'s documented layout ("local runtime logs; gitignored"). |

No unexplained/unreferenced stray root files were found beyond the ones
listed. Nothing here was moved.

---

## G. Experiment-Directory Classification

42 top-level entries under `experiments/`. Full individual-file review of all
42 was out of scope for a 1-pass audit at reasonable cost; classification
below is evidence-based (git-tracked status, cross-reference from
`docs/current/RESUME_HERE.md` / `docs/audits/*`, file contents/sizes) but not
exhaustive for every directory.

**CANONICAL_CURRENT** (directly cited by the current audit lineage in
`RESUME_HERE.md`, all git-tracked, all `KEEP`):
`family_b_balanced_replication_v1/`, `hierarchical_regime_router_live_reeval_v1/`,
`hierarchical_regime_router_v1_smoke/`, `hierarchical_regime_router_v1_test_evaluation/`,
`hierarchical_router_live_harness_v1_smoke/`, `cross_family_transfer_wellposedness_reassessment_v1/`,
`online_regime_signal_feasibility_v1/`, `mechanism_choice_target_feasibility_v1/`,
`mf_psd_v1/`, `unified_utility_matrix_v1/`, `unified_utility_matrix_v2/`,
`multifamily_contextual_selector_v1/`, `shared_cross_family_features_v1/`,
`kv_composition_falsification_v1_20260817T172446Z/`,
`kv_composition_safety_refinement_v1_hysteresis/`,
`kv_pressure_pilot_v1_20260817T162650Z/`, `kv_pressure_pilot_v2_20260817T165053Z/`,
`family_c_reconstruction_v1/`, `prefill_control_composition_v2_20260817T154633Z/`,
`policy_separation_prefill_decode_pilot_v1_20260817T020803Z/`,
`policy_separation_prefill_decode_pilot_v2_20260817T024204Z/`,
`policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/`,
`policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/`,
`policy_separation_sobol_pilot_20260816T183600Z_1182183/`,
`estf_wfs_composition_falsification_v1_20260816T222108Z/`.

**FROZEN_HISTORICAL** (superseded-but-preserved-on-purpose, explicitly named
as such in `RESUME_HERE.md`, `KEEP`): the v1 pilots superseded by v2 (Family
A v1, Family B v1, KV-pressure v1) are already covered above as v1/v2 pairs;
`gpu_external_validity/`, `runtime_validation_benchmark_pack/`, `real_llm/`,
`selector_v2_*` (calibrated_pilot, contention_frontier_search,
faithful_baseline_scope_audit, overnight, slo_calibrated_frontier_search),
`baseline_comparison/` — all predate the current MF-PSD/hierarchical-routing
lineage and are cited as historical evidence elsewhere in `docs/audits/` or
`docs/BASELINE_STATUS.md`. `KEEP`.

**ABANDONED_DEBUG / TEMPORARY** (high-confidence, concrete finding):

| Path | Reason | Referenced anywhere? | Git-tracked? | Scientifically important? | Recommended |
|---|---|---|---|---|---|
| `experiments/prefill_control_composition_v2_20260817104112Z/` | Contains only a 197-byte `run.log`, no results. Malformed timestamp (missing `T` separator) vs. the canonical sibling `..._20260817T154633Z/`. | No | No | No — the real run is the `T154633Z` sibling, which *is* tracked and cited | `DELETE_LATER_WITH_AUTHORIZATION` (or `GITIGNORE` if this pattern recurs from a known launcher bug) |
| `experiments/prefill_control_composition_v2_20260817180000Z/` | Same pattern, 392-byte `run.log` only. | No | No | No | `DELETE_LATER_WITH_AUTHORIZATION` |
| `experiments/prefill_control_composition_smoke_v3_20260817T000622Z/` | 91-byte `run.log` only, no other content. | No | No | No | `DELETE_LATER_WITH_AUTHORIZATION` |

These three are the clearest "duplicate/failed retry" clutter in
`experiments/`. All three are **untracked** (confirmed via
`git ls-files | wc -l` = 0 for each), so removing them later is fully
git-history-safe — no scientific evidence is at risk. Not deleted in this
pass.

**SETUP_ONLY / UNKNOWN** (not independently verified this pass, flag for
Pass-2 spot check rather than asserted): `baseline_comparison/` (small, old,
June 10) — likely an early scaffold predating the current baseline registry;
recommend a quick content check before any action.

**Two loose root-level `.log` files inside `experiments/`** (not inside a
timestamped subdirectory): `experiments/family_c_reconstruction_v1_launch.log`,
`experiments/multifamily_contextual_selector_v1_launch.log`,
`experiments/unified_utility_matrix_v1_launch.log` — these are tracked
(check with `git ls-files` recommended in Pass 2) launch logs sitting
alongside their matching directories; harmless but slightly inconsistent
with the "one directory per experiment" convention used everywhere else.
`NEEDS_REVIEW`, not urgent.

---

## H. Living-Doc Consistency Table

| Document | Current role | Last scientific state represented | Actual current state | Stale? | Duplicative? | Recommended Pass-2 action |
|---|---|---|---|---|---|---|
| `README.md` | Public overview / entrypoint #1 | Apt-Serve Phase G analysis ("most recent major local experiment") | 6+ major completed milestones later (MF-PSD, unified matrix, 3 selector NO_GOs, hierarchical routing, live re-eval, Family-B replication prep) | **Yes — severely** (predates the entire Aug-17-19 lineage) | No (unique role) | Rewrite "Current Checkpoint" section to point at `RESUME_HERE.md` for state rather than embedding a snapshot that will re-rot; keep README as navigation-only |
| `docs/PROJECT_MAP.md` | Canonical long-term roadmap, ranked **#2** in documentation authority | "Last reconciled: 2026-08-17, after MF-PSD v1 build" | Predates Step 2, Step 3 (`MULTIFAMILY_SELECTOR_NO_GO`), shared-feature NO_GO, mechanism-target NO_GO, cross-family reassessment, hierarchical router (design→impl→TEST→live re-eval), Family-B replication prep | **Yes — most stale tier-1 doc**, despite being the highest-authority roadmap doc after README | Overlaps `docs/current/RESUME_HERE.md`'s "what's next" content | Update WS-F/WS-H/WS-I rows and "Last reconciled" date to current HEAD; this is the single highest-priority Pass-2 doc edit |
| `docs/current/RESUME_HERE.md` | Shortest operational entrypoint | Through `HIERARCHICAL_ROUTER_NO_GO` formal live re-eval gate-scoring | Missing only the most recent commit (`9d8f997`, "prepare Family-B live replication harness") and the still-in-flight public-trace-corpus / decision-criticality work | **Marginally stale** (~1 commit behind HEAD) | Overlaps `NEXT_ACTIONS.md` substantially | Add one short section for Family-B live replication prep + a pointer that public-trace-corpus/decision-criticality work is in-flight, uncommitted |
| `docs/current/NEXT_ACTIONS.md` | Prioritized next actions | Same as `RESUME_HERE.md` (mirrors its P0 content closely) | Same gap as `RESUME_HERE.md` | Marginally stale | **Yes — large duplicative overlap with `RESUME_HERE.md`** (near-identical prose blocks) | Consider converting to a short pointer + delta-only next-actions list rather than restating the full narrative |
| `docs/current/WORK_STATUS.md` | "Detailed current status table" (self-described operational companion) | MF-PSD v1 only; table literally ends before Step 2 | Missing Step 2, Step 3, both later NO_GOs, cross-family reassessment, hierarchical routing, live re-eval, Family-B replication | **Yes — ~2 days / 8+ milestones behind** | Table format is unique (not fully duplicative) — worth keeping as a format, just needs new rows | Append rows for every milestone from "Step 2" through "Family-B live replication prep"; this is the doc most naturally suited to carry the canonical status table from §J below |
| `docs/current/EXPERIMENT_INDEX.md` | Durable experiment-artifact index | "Generated 2026-07-21... extended 2026-07-25" | Contains **zero** entries for anything after 2026-07-25 — missing all of WS-P Family B/C, composition falsifications, MF-PSD, unified matrix, hierarchical routing, live re-eval | **Yes — ~1 month stale**, the most stale of all audited docs | No (unique index role) | Append rows for the ~15+ missing experiment directories identified in §G; largest single Pass-2 content addition |
| `docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md` | Protected-path list for cluster jobs | "Refreshed during the 2026-07-22 repository polish pass"; states "No currently running project jobs were observed" | Almost certainly inaccurate now (references SLURM `squeue` state from July 22, references only cluster/SwissAI-era job roots) | **Yes — ~1 month stale** | No | Needs a fresh cluster job check + a companion **local-protected-paths** section (this doc currently only covers `/mmfs1/...` cluster roots, not the local experiment directories added since) |

---

## I. Roadmap / Documentation Contradictions

1. **Authority inversion (most important):** `docs/PROJECT_MAP.md` is
   explicitly ranked above `docs/current/RESUME_HERE.md` in the
   documentation-authority list stated in both `README.md` and
   `docs/PROJECT_MAP.md` itself — yet it is the more stale of the two by a
   wide margin. A reader following the stated authority order would stop at
   the wrong (stale) picture of WS-F/WS-H status.
2. **`docs/PROJECT_MAP.md` WS-F row says** "Multi-family selector training is
   Step 3 of the revised roadmap ... not started" — this is **false** as of
   HEAD; Step 3 ran to completion and produced `MULTIFAMILY_SELECTOR_NO_GO`
   (per `RESUME_HERE.md` and `docs/audits/multifamily_contextual_selector_v1_20260817.md`).
3. **`docs/PROJECT_MAP.md` WS-H row says** "COMPOSITION_DEMOTED... deferred"
   — still directionally correct, but doesn't mention that composition was
   subsequently *specifically tested* three times (ESTF/WFS, PrefillControl,
   KV-pressure) with two `SELECTION_SUFFICIENT_FOR_THIS_PAIR` verdicts and
   one `KV_COMPOSITION_INCONCLUSIVE`.
4. **`README.md`'s "Current Checkpoint"** still frames Apt-Serve Phase G as
   the most recent major experiment and as "the canonical next task" driver
   — both now superseded framings. No stale-Apt-Serve-as-primary claim was
   found to be *asserted as current* anywhere else, so this is isolated to
   README, not systemic.
5. **No stale "universal selector is the active goal" framing was found** —
   `RESUME_HERE.md`/`NEXT_ACTIONS.md` correctly and prominently state the
   NO_GOs. This particular risk named in the task brief is **not present**.
6. **No stale "live harness is unbuilt" or "live re-evaluation is pending"
   framing was found** anywhere — `RESUME_HERE.md` is fully current on this
   point specifically (the harness and live re-eval are its most recent
   content). This risk is also **not present**.
7. **Family-B evaluation described inaccurately?** Partially — `RESUME_HERE.md`
   correctly and repeatedly flags "Family B got 0 TEST scenarios" as an
   unresolved gap, and the newest commit (`9d8f997`, "prepare Family-B live
   replication harness") is exactly the in-progress fix for this — but that
   commit is not yet described in prose anywhere (see §H).
8. **Public-trace reuse missing from roadmap:** confirmed — a
   repo-wide `grep -rli` for "public.trace" and "decision.criticality" across
   `docs/current/*.md`, `README.md`, and `docs/PROJECT_MAP.md` returned
   **zero matches**. Both in-flight tasks are entirely undocumented at the
   living-doc layer (expected, since neither is committed yet).
9. **New-policy-synthesis goal:** present but understated. `README.md`'s
   "Research Objective" diagram and `docs/PROJECT_MAP.md`'s "North Star" both
   describe composition/synthesis as the long-run target, but neither
   currently reflects the sharper reframing the task brief asks for (new
   input → find where policies fail/disagree → synthesize a new policy →
   validate on simulator + real LLM). This is a genuine content gap, not a
   contradiction — see §K/§L for exact Pass-2 content.

---

## J. Canonical Current Scientific Status Table

| Workstream | Status | Verdict | Canonical artifact | Canonical commit (best identified) | State | Next dependency |
|---|---|---|---|---|---|---|
| Family A (fairness/starvation) v1 | Complete (diagnostic) | `REDESIGN_REQUIRED` for corpus use | `experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/` | — | FROZEN | superseded by v2 |
| Family A v2 | Complete | `USEFUL_BUT_NEEDS_REFINEMENT` | `experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/` | — | FROZEN | feeds MF-PSD |
| Family B v1 | Complete | `USEFUL_BUT_NEEDS_REFINEMENT` / `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED` | `experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z/` | — | FROZEN | superseded by v2 |
| Family B v2 | Complete | `FAMILY_B_COMPOSITION_READY` | `experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/` | — | FROZEN | feeds MF-PSD |
| PrefillControl composition falsification | Complete | `SELECTION_SUFFICIENT_FOR_THIS_PAIR` | `experiments/prefill_control_composition_v2_20260817T154633Z/` | — | FROZEN | none queued |
| Family C v1 (KV-pressure) | Complete | `KV_FAMILY_USEFUL_NEEDS_REFINEMENT` | `experiments/kv_pressure_pilot_v1_20260817T162650Z/` | — | FROZEN | superseded by v2 |
| Family C v2 (KV-pressure) | Complete | `KV_FAMILY_COMPOSITION_READY` | `experiments/kv_pressure_pilot_v2_20260817T165053Z/` | — | FROZEN | feeds MF-PSD |
| KV composition falsification | Complete | `KV_COMPOSITION_INCONCLUSIVE` (G7 safety fails) | `experiments/kv_composition_falsification_v1_20260817T172446Z/` | — | FROZEN | narrowly-rescoped transition-aware child (not started, not authorized) |
| KV v2 reproducibility forensic | Complete | `REPRODUCIBILITY_GAP_BOUNDED` (root cause not found) | `docs/audits/kv_v2_reproducibility_forensic_20260817.md` | — | FROZEN | none required |
| Composition (higher-level reassessment) | Complete | `COMPOSITION_DEMOTED` | `docs/audits/reassessment_composition_hypothesis_20260817.md` | `dc5757b` | FROZEN | set revised roadmap |
| MF-PSD v1 | Complete | `MF_PSD_READY` | `experiments/mf_psd_v1/` | — | FROZEN | fed Step 2 |
| Unified utility matrix (Step 2) | Complete | `UNIFIED_UTILITY_MATRIX_READY` | `experiments/unified_utility_matrix_v2/` (**missing build manifest — see §M**) | — | FROZEN | fed Step 3 |
| Flat/pooled multi-family selector (Step 3) | Complete | `MULTIFAMILY_SELECTOR_NO_GO` | `experiments/multifamily_contextual_selector_v1/` | — | FROZEN | motivated shared-feature redesign |
| SHARED_CORE_V1 feature schema | Complete | `SHARED_FEATURE_SCHEMA_NO_GO` | `experiments/shared_cross_family_features_v1/` | — | FROZEN | motivated mechanism-choice redesign |
| Mechanism-choice target redesign | Complete | `MECHANISM_TARGET_NO_GO` | `experiments/mechanism_choice_target_feasibility_v1/` | — | FROZEN | motivated cross-family reassessment |
| Cross-family transfer well-posedness reassessment | Complete | `CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY` | `experiments/cross_family_transfer_wellposedness_reassessment_v1/` | — | FROZEN | motivated hierarchical router design |
| Online regime-signal feasibility | Complete | `ONLINE_REGIME_SIGNALS_READY` | `experiments/online_regime_signal_feasibility_v1/` | — | FROZEN | fed hierarchical router Stage-1 |
| Hierarchical regime router (TEST, offline majority-vote) | Complete | `HIERARCHICAL_ROUTER_NO_GO` | `experiments/hierarchical_regime_router_v1_test_evaluation/` | `2923087` (impl) | FROZEN | motivated live harness |
| Live per-step simulation harness | Complete | `LIVE_HIERARCHICAL_HARNESS_READY` | `experiments/hierarchical_router_live_harness_v1_smoke/` | `723a39c` | FROZEN | enabled live re-eval |
| Hierarchical router live re-evaluation | Complete (formally gate-rescored) | `HIERARCHICAL_ROUTER_NO_GO` (formal) | `experiments/hierarchical_regime_router_live_reeval_v1/live_reeval_results.json` | `9fde981` (run), `ed74276` (fix) | FROZEN | Family-B-specific live eval OR higher-level reassessment |
| Family-B balanced live replication | **In progress (prep only)** | not yet run | `experiments/family_b_balanced_replication_v1/` | `9d8f997` | **RUNNING/PREP — HEAD commit** | run + score against frozen gates |
| Decision-criticality/timescale analysis | **Unknown to committed docs; active background process observed** | n/a | `src/llmserveopt/analysis/decision_criticality_timescale_trainval_v1.py` (uncommitted) | uncommitted | **RUNNING (pid 80136)**, undocumented | belongs to a concurrent, out-of-scope task — see §O |
| Public-trace corpus v1 | **In progress, two colliding uncommitted implementations observed** | n/a | `data/public_trace_corpus_v1/` (uncommitted) | uncommitted | **IN PROGRESS**, undocumented, collision risk | belongs to a concurrent, out-of-scope task — see §O |
| New-policy synthesis | Not started | n/a | n/a | n/a | NOT_STARTED | depends on public-trace corpus → replay → decision-criticality layers above |

---

## K. Current Project North Star (evidence-based reconstruction)

From `README.md`, `docs/PROJECT_MAP.md`, and the audit trail, the durable
objective as currently practiced (not merely as originally written) is:

```
policy-separating workloads (synthetic, families A/B/C)
  -> complementary policy library
  -> contextual selection (multi-family) [NO_GO x3 at flat/pooled level]
  -> hierarchical regime routing [NO_GO at TEST + live]
  -> mechanism attribution (blocked pending a routing success)
  -> bounded envelope
```

This is real and evidenced, but it is a **narrower, more defensive** north
star than the one the task brief asks Pass 2 to make explicit: the brief's
framing —

```
new input/workload/state
  -> identify where existing policies fail or disagree
  -> identify mechanism/decision-critical structure
  -> create or evolve a NEW scheduling policy tailored to the input
  -> validate on simulator + real LLM serving
```

— is **consistent with, but not yet stated as, the project's north star** in
any committed doc. `README.md`'s diagram already contains "structural
composition / symbolic synthesis" as a step, but frames it as downstream of
DSL/AST composition specifically, not as "evolve a new policy from
decision-critical state structure" generally. This is the exact gap Pass 2
should close (see §L/§U).

---

## L. Current Dataset Strategy (evidence-based)

What exists today, verified against `src/llmserveopt/workloads/` and
`docs/current/REAL_DATASET_EXPANSION_STATUS.md`/`docs/dataset_workload_decision.md`
(not re-audited in full depth this pass, but cross-checked against
`docs/current/PROJECT_MAP.md`'s Datasets section and
`docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md`):

- **Real traces already ingested with loaders/tests:** BurstGPT (MIT,
  full CSV present locally), Azure 2023 conv+code (CC-BY-4.0, present
  locally), Bailian/Qwen (Apache-2.0, staged on cluster per
  `docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md` history), Mooncake
  (code Apache-2.0, **data license `NOT_EXPLICITLY_SPECIFIED` — internal-only,
  not redistributable**, per `docs/current/PROJECT_MAP.md`'s explicit
  warning), WildChat (ODC-BY, selected/used for vLLM-LTR comparison),
  ShareGPT (license unclear, deliberately not ingested).
- **SwissAI and TraceLab are already staged and swept on the cluster**
  (`swissai_trace_staging_20260722T172215Z`, `swissai_v2_policy_sweep_20260722T184451Z`,
  `tracelab_staging_20260722T192050Z`, `tracelab_v2_policy_sweep_20260722T214129Z`
  — all listed `COMPLETE` in `ACTIVE_EXPERIMENT_PROTECTED_PATHS.md`, with
  `docs/result_claims.md` already recording their finding: "SwissAI and
  TraceLab add raw workload novelty, but their completed 512-window x
  27-policy sweeps saturated ANWG and produced zero strict V2 marginal
  oracle gain under the current simulator/objective." **This is directly
  relevant to the concurrent public-trace-corpus task and is not currently
  visible from that task's own design doc** (out of scope to fix here, but
  flagged).
- **The distinction the task brief asks about** (raw workload data vs.
  derived policy-separation data vs. real-LLM validation subset vs.
  new-policy-synthesis goal) **is not yet explicit anywhere** in
  `docs/current/` — the closest existing material is `docs/data_field_provenance.md`'s
  "online-visible vs. simulator-only-hidden" distinction, which is a related
  but different axis (observability, not layer-of-derivation). Pass 2 should
  add this as new content, not a correction to existing wrong content.

---

## M. New-Policy-Synthesis Path (what Pass 2 should write, not edit yet)

The exact 10-step path from the task brief is **not yet written anywhere**
in the repository. It should be added as new content (recommended location:
a new section in `docs/PROJECT_MAP.md`, since that is the doc ranked above
`RESUME_HERE.md` in documentation authority and is also the most stale — killing
two problems with one edit). Content to add, verbatim per the brief's intent:
reuse public traces → normalize into a corpus → replay under multiple
policies → record scenario-level utility → record state-level
actions/disagreements → record counterfactual consequences → identify
decision-critical states → use as synthesis training signal → validate
synthesized policies on real-LLM serving (Cohere/CloudRift reserved for that
last stage only, never for coding/search). This should explicitly
cross-reference the existing selector NO_GO lineage (§J) as the reason a
flat/pooled selector was abandoned in favor of this decision-critical-state
approach — otherwise a future reader will not understand why the project
pivoted from "train a selector" to "find decision-critical states and
synthesize."

---

## N. Documentation Duplication Findings

| Cluster | Members | Recommendation |
|---|---|---|
| Operational handoff narrative | `docs/current/RESUME_HERE.md`, `docs/current/NEXT_ACTIONS.md` | `RESUME_HERE.md` = `KEEP_AS_CANONICAL`; `NEXT_ACTIONS.md` = `KEEP_AS_SHORT_POINTER` (currently near-duplicates full prose; should shrink to a delta/priority list referencing `RESUME_HERE.md` for narrative) |
| Roadmap/map docs | `docs/PROJECT_MAP.md` (research roadmap) vs. `docs/current/PROJECT_MAP.md` (code-location map) | Both `KEEP_AS_CANONICAL` — they explicitly disclaim confusion with each other in their own headers and serve genuinely different purposes ("why does code exist" vs. "where is code"). Not truly duplicative, just confusingly named; a Pass-2 rename is out of scope per this task's "no moves" rule but worth flagging for a future pass. |
| Status snapshots | `docs/current/PROJECT_HANDOFF_2026-07-23.md`, `docs/current/PROJECT_SNAPSHOT_20260806.md`, `docs/current/PROJECT_PAUSE_HANDOFF_20260806.md`, `docs/current/PAUSE_PROVENANCE_2026-07-23.md` | All `MARK_SUPERSEDED` candidates — these are point-in-time snapshots from before the current lineage; not verified line-by-line this pass, but their filenames/dates alone place them well before MF-PSD. Recommend a one-line "superseded by RESUME_HERE.md as of <date>" banner at the top of each rather than deletion (they may still hold provenance value). |
| Composition-specific docs | `docs/current/COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md`, `docs/current/COMPOSITION_IMPLEMENTATION_STATUS.md`, `docs/current/POLICY_COMPOSITION_READINESS.md`, `docs/current/COMPOSITION_EXPERIMENT_DESIGN.md` | Likely `MARK_SUPERSEDED` given `COMPOSITION_DEMOTED` — not independently re-read this pass; recommend Pass 2 add a superseded-banner sweep across this cluster referencing `docs/audits/reassessment_composition_hypothesis_20260817.md` rather than editing each file's body |
| Experiment index / status table | `docs/current/EXPERIMENT_INDEX.md` vs. `docs/current/WORK_STATUS.md` | Not duplicative (index vs. status table are different shapes) but both stale — `KEEP_AS_CANONICAL` for both, both need the same set of ~15 new rows appended (§H) |

---

## O. Protected / Frozen Path List

```
docs/design/**                                          (frozen preregistrations, once committed)
docs/audits/**                                           (immutable point-in-time audit trail)
configs/hierarchical_regime_router_v1_gates.json          (frozen gate thresholds)
experiments/mf_psd_v1/**                                  (MF_PSD_READY canonical dataset)
experiments/unified_utility_matrix_v1/**                  (frozen Step-2 v1; superseded but not rewritten)
experiments/unified_utility_matrix_v2/**                  (canonical Step-2 dense matrix — NOTE: missing build manifest, §Q, but still frozen content)
experiments/family_c_reconstruction_v1/**                 (CURRENT_RECONSTRUCTED_FAMILY_C_V1 ground truth)
experiments/kv_pressure_pilot_v1_20260817T162650Z/**       (frozen v1, superseded by v2, not rewritten)
experiments/kv_pressure_pilot_v2_20260817T165053Z/**       (frozen KV_FAMILY_COMPOSITION_READY evidence)
experiments/kv_composition_falsification_v1_20260817T172446Z/** (frozen KV_COMPOSITION_INCONCLUSIVE evidence)
experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/** (frozen v1; do not rewrite Job 1182306 CSV rows, per explicit RESUME_HERE.md instruction)
experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/**
experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z/** (frozen v1)
experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/**
experiments/prefill_control_composition_v2_20260817T154633Z/**
experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/**
experiments/multifamily_contextual_selector_v1/**
experiments/shared_cross_family_features_v1/**
experiments/mechanism_choice_target_feasibility_v1/**
experiments/cross_family_transfer_wellposedness_reassessment_v1/**
experiments/online_regime_signal_feasibility_v1/**         (127,319-row telemetry CSV — also the largest tracked file, §Q)
experiments/hierarchical_regime_router_v1_test_evaluation/**
experiments/hierarchical_router_live_harness_v1_smoke/**
experiments/hierarchical_regime_router_live_reeval_v1/**   (currently has an untracked provenance-timestamp diff, §D — content unmodified)
experiments/family_b_balanced_replication_v1/**            (currently ACTIVE prep work, HEAD commit — do not touch)
data/raw/burstgpt/BurstGPT_1.csv, data/raw/azure/*.csv     (raw source caches — regeneratable but should not be silently overwritten)
```

**Additionally, as of this audit, NOT to be touched by Pass 2 because they
belong to concurrent, out-of-scope work (see §C):**
```
src/llmserveopt/analysis/**                    (uncommitted, active background process)
docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md
data/public_trace_corpus_v1/**                 (uncommitted)
docs/design/PUBLIC_TRACE_CORPUS_V1.md          (uncommitted)
src/llmserveopt/workloads/public_trace_corpus.py (uncommitted)
scripts/build_public_trace_corpus_v1.py        (uncommitted)
tests/test_public_trace_corpus_v1.py           (uncommitted)
```

---

## P. `.gitignore` / Local-Exclude Findings

- Logs (`*.log`, `logs/`, `*.out`, `*.err`), pycache, coverage, tool caches
  (`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`), and raw external
  datasets (`data/raw/*`, `data/processed/*`, `dataset_staging/`,
  `hf_cache/`, `llmserveopt-data/`) are all correctly handled.
- **`*.jsonl` is globally ignored**, but the repo has ~25+ pre-existing
  **tracked** `.jsonl` files (`tests/fixtures/{bailian,mooncake}_tiny.jsonl`,
  most of `experiments/real_llm/*/requests.jsonl` and `errors.jsonl`,
  `experiments/family_c_reconstruction_v1/*.jsonl`,
  `experiments/gpu_external_validity/*/requests.jsonl`). This is git's
  normal "already-tracked files aren't affected by a later ignore rule"
  behavior, and one instance of this exact pattern (`server.log`) is already
  explicitly documented as a "pre-existing exception" in the `.gitignore`
  comments — but the broader `*.jsonl` case is **not** similarly documented.
  Flag only; **do not** `git rm --cached` any of these (that would untrack
  real evidence files) — a Pass-2 documentation comment explaining the
  pattern is the safe fix.
- `.git/info/exclude` (not `.gitignore`) carries machine-local excludes
  (`opencode.json`, various `.claude/` runtime state) — correctly
  machine-scoped rather than imposed on all clones; not a problem.
- No scientifically important file was found hidden by an over-broad ignore
  rule.
- No generated junk was found that is *not* currently ignored, other than
  the three abandoned-debug experiment stubs in §G (which are directory
  clutter, not gitignore-pattern gaps — `*.log`-only directories are
  already correctly excluded from git tracking, they just still exist on
  disk).

---

## Q. Generated-Artifact Clutter

- `.coverage` (root) — stray, ignored, harmless; `IGNORE`.
- `crash.log` (root, 0 bytes) — ignored, referenced by one audit doc as
  evidence of "no crash"; `KEEP` (do not delete — it's cited).
- `run.log` (root) — ignored, default output path; `IGNORE`.
- Three `ABANDONED_DEBUG` experiment stub directories (§G) — untracked,
  `DELETE_LATER` (with authorization).
- **Largest tracked file in the repo:** `experiments/online_regime_signal_feasibility_v1/online_regime_telemetry_v1.csv`
  at ~29.9MB (127,319-row per-step telemetry). This is committed, not
  ignored, and appears intentional (it's the canonical evidence artifact for
  the `ONLINE_REGIME_SIGNALS_READY` verdict, explicitly described as such in
  `RESUME_HERE.md`). Flagging only because it is unusually large for this
  repo's normal committed-artifact size envelope (next-largest is ~4.3MB);
  worth a conscious Pass-2/3 decision (keep as-is vs. compress vs. move to
  a summary+regenerate-on-demand pattern) rather than an oversight fix.
- No JUnit XML files, no stray checkpoint/model binaries, were found tracked
  or untracked at notable size.

---

## R. Provenance / Manifest Debt

| Item | Correct historical file? | Recommended classification |
|---|---|---|
| `hierarchical_regime_router_live_reeval_v1/launch_manifest.json`'s `git_sha` does not resolve to a real git object (short-SHA corruption) | Yes — self-disclosed in `RESUME_HERE.md` and the live-reeval audit as a known, transparently-documented defect | `LEAVE_AND_DOCUMENT` — already documented in the audit; do not rewrite the historical manifest |
| `fitted_model_hashes.json` (same experiment) omits a `KV_MEMORY_PRESSURE` model hash | Yes — same self-disclosure | `LEAVE_AND_DOCUMENT` |
| `experiments/unified_utility_matrix_v2/` has no `unified_utility_matrix_build_manifest_v2.json` (v1 has one, v2 does not) | Confirmed this pass via direct directory listing | `GENERATE_NEW_MANIFEST_FROM_EXISTING_ARTIFACTS` — the v2 CSVs and their build inputs (Family A/B/C v2 sources, Family C reconstruction) are all still present and byte-identifiable; a manifest can be reconstructed without rerunning anything. Reasonable Pass-2 scope. |
| Loose root logs (`run.log`, `crash.log`) | N/A (generated, not historical evidence per se) | `DO_NOT_TOUCH` beyond documenting (both already gitignored and harmless) |
| Inconsistent experiment naming (`prefill_control_composition_v2_20260817104112Z` vs. `..._20260817T154633Z`) | N/A — these are the abandoned stubs from §G, not real historical files | `DO_NOT_TOUCH` as historical evidence (there is none to preserve); safe for eventual deletion once explicitly authorized |
| Missing result manifests/checksums elsewhere | Not exhaustively re-audited this pass beyond the two items above; `docs/audits/kv_v2_reproducibility_forensic_20260817.md` already documents a broader historical-reproducibility gap for the KV v1/v2 lineage specifically | `LEAVE_AND_DOCUMENT` (already is) |
| Old setup-only directories | The three §G stub directories are the only concrete instances found | Covered above |

---

## S. Security / Large-File / Secrets Findings

- No API keys, tokens, AWS-style credentials, or PEM/private-key blocks
  found via pattern grep across `*.py`, `*.md`, `*.json`, `*.yaml`, `*.env*`.
- No tracked `.env` or `.env.*` file (only the intentional `.env.example`
  template).
- No `credentials.json`, `secrets.*`, or `*.json.secret` files tracked.
- No raw third-party dataset files (BurstGPT/Azure/etc.) are tracked in git
  — `data/raw/**` is correctly untracked/ignored.
- Largest tracked file: ~29.9MB CSV (§Q) — not a binary/model artifact, a
  legitimate CSV; not a security concern, only a size-hygiene note.
- No large binary/model files (`.bin`, `.safetensors`, `.pt`, `.ckpt`) found
  tracked.

**Severity: none found requiring action.**

---

## T. Safe Cleanup Actions for Pass 2

**CATEGORY A — Documentation reconciliation (safe, recommended):**
- Update `docs/PROJECT_MAP.md`'s WS-F/WS-H/WS-I rows and "Last reconciled"
  date to reflect Step 2 → Step 3 → both NO_GOs → cross-family reassessment
  → hierarchical router → live re-eval → Family-B replication prep.
- Add a short "Family-B live replication (in progress)" section to
  `docs/current/RESUME_HERE.md` and `NEXT_ACTIONS.md`.
- Append ~15 missing rows to `docs/current/EXPERIMENT_INDEX.md` covering
  everything from MF-PSD through Family-B replication prep.
- Append matching rows to `docs/current/WORK_STATUS.md`'s status table
  (reuse the table built in §J of this audit as the direct source).
- Rewrite `README.md`'s "Current Checkpoint" section to point at
  `RESUME_HERE.md` for state rather than embedding a snapshot.
- Add superseded-banners to the `MARK_SUPERSEDED` cluster identified in §N.
- Add the new-policy-synthesis 10-step path (§M) to `docs/PROJECT_MAP.md`.
- Refresh `docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md` with a current
  cluster-job check plus a new local-paths section covering everything in §O.

**CATEGORY B — Organization/pointers (safe, recommended):**
- Shrink `NEXT_ACTIONS.md`'s duplicated narrative into a pointer + delta
  list.
- Add a short note near `p2_config.yaml`/`p3_chunk_control.py`/etc. (e.g. in
  `docs/CONTEXTUAL_COMPOSITION_BRANCH.md`) acknowledging they live at repo
  root by historical accident, without moving them.

**CATEGORY C — gitignore/local-artifact hygiene (safe, recommended):**
- Add a one-line comment near the `*.jsonl` gitignore rule documenting the
  same "pre-existing tracked files stay tracked" exception already used for
  `server.log`.

**CATEGORY D — experiment directory cleanup (requires care):**
- Delete (with explicit authorization) the three untracked
  `ABANDONED_DEBUG` stub directories in §G. Reversible in the sense that
  nothing scientific is lost (confirmed 0 git-tracked files in each), but
  still requires explicit go-ahead per the task's own rules.
- Reconstruct `unified_utility_matrix_build_manifest_v2.json` from existing
  artifacts (§R) — low risk, additive only, but touches the "frozen"
  experiments directory, so treat as Category D, not A.

**CATEGORY E — branch cleanup (requires explicit later authorization):**
- Archive/delete the 49 fully-merged local branch pointers (§E) — 100%
  commit-safe (all reachable from current HEAD) but branch deletion itself
  needs explicit user authorization per policy.
- Review (don't yet act on) the 2 unmerged branches
  (`phase2b13-selector-training-after-diversity`,
  `phase2c-final-selector-improvement`) before any decision.

**CATEGORY F — historical/provenance artifacts (document only, never modify):**
- Everything in §O's protected-path list.
- The known provenance defects in §R (`git_sha` corruption, missing KV
  model hash) — already correctly left as historical evidence; no action
  beyond what's already been done.

---

## U. Updated Guide-Map Structure (Pass-2 proposal, not implemented)

```
README.md                                    <- navigation + link to current state, no embedded snapshot
  -> docs/PROJECT_MAP.md                     <- canonical roadmap (needs date/content refresh, §I/§T)
       -> new-policy-synthesis 10-step path added here (§M)
  -> docs/current/RESUME_HERE.md             <- shortest operational entrypoint (needs 1-commit-gap fix)
  -> docs/current/NEXT_ACTIONS.md            <- shrink to pointer + delta list (§N)
  -> docs/current/WORK_STATUS.md             <- detailed status table (append §J rows)
  -> docs/current/EXPERIMENT_INDEX.md        <- durable artifact index (append ~15 rows)
  -> docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md  <- refresh cluster-job state + local paths
  -> docs/audits/*                            <- frozen, immutable, never edited
  -> docs/design/*                            <- frozen once preregistered, never edited
```

This structure already exists in name; Pass 2's job is content refresh, not
restructuring. Answers to the 10 questions posed in the task brief map
directly onto this hierarchy: (1) `README.md`+`docs/PROJECT_MAP.md` north
star; (2) `docs/audits/*` falsification/demotion trail, summarized in
`RESUME_HERE.md`; (3) §C of this audit / a refreshed
`ACTIVE_EXPERIMENT_PROTECTED_PATHS.md`; (4) §L dataset strategy, to be added
to `docs/PROJECT_MAP.md`; (5) `RESUME_HERE.md`'s "Exact Next Tasks"; (6) same;
(7) §J / `EXPERIMENT_INDEX.md`; (8) §O; (9) `RESUME_HERE.md`'s "Resume
commands"; (10) §M.

---

## V. Exact Files Pass 2 Should Edit/Create

**Edit:**
- `README.md` (Current Checkpoint section)
- `docs/PROJECT_MAP.md` (WS-F/WS-H/WS-I rows, "Last reconciled" date, new
  synthesis-path section)
- `docs/current/RESUME_HERE.md` (append Family-B replication prep section)
- `docs/current/NEXT_ACTIONS.md` (shrink duplication)
- `docs/current/WORK_STATUS.md` (append ~15 status rows)
- `docs/current/EXPERIMENT_INDEX.md` (append ~15 experiment rows)
- `docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md` (refresh)
- `.gitignore` (one-line comment near `*.jsonl`)

**Create:**
- `experiments/unified_utility_matrix_v2/unified_utility_matrix_build_manifest_v2.json`
  (reconstructed from existing artifacts — Category D, do with care)
- Optionally, superseded-banner insertions at the top of the `MARK_SUPERSEDED`
  cluster in §N (small edits, arguably "edit" not "create")

**Mark superseded (banner only, no content deletion):**
- `docs/current/PROJECT_HANDOFF_2026-07-23.md`, `PROJECT_SNAPSHOT_20260806.md`,
  `PROJECT_PAUSE_HANDOFF_20260806.md`, `PAUSE_PROVENANCE_2026-07-23.md`,
  `COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md`,
  `COMPOSITION_IMPLEMENTATION_STATUS.md`, `POLICY_COMPOSITION_READINESS.md`,
  `COMPOSITION_EXPERIMENT_DESIGN.md` (verify each individually before
  banner-marking; not all were re-read in full this pass)

**Move:** none.

**Ignore (add/adjust patterns):** none required beyond the one gitignore
comment above.

**Leave completely untouched:**
- Everything in §O (both the long-standing protected list and the
  concurrent-work list).
- The 3 `ABANDONED_DEBUG` stub directories (document, don't delete, until
  explicitly authorized).
- All branches (document, don't delete, until explicitly authorized).

**Directories to archive/delete, if any:** none in Pass 2 without explicit
authorization; the 3 stub directories and the 49 merged branches are the
only candidates, both deferred to Category D/E.

---

## W. Exact Files/Directories Pass 2 Must Not Touch

See §O in full. Highlights: all `docs/audits/**`, all `docs/design/**` once
committed, `experiments/mf_psd_v1/**`, `experiments/unified_utility_matrix_v1/**`
and `_v2/**` (content, not the missing manifest — that's additive), all
frozen Family A/B/C v1 and v2 pilot directories, `experiments/family_b_balanced_replication_v1/**`
(active prep work at HEAD), and everything currently uncommitted that
belongs to the concurrent public-trace-corpus / decision-criticality work.

---

## X. Items Requiring Later Explicit Authorization

- Deleting/archiving any of the 49 fully-merged local branches.
- Deleting the 3 untracked `ABANDONED_DEBUG` experiment stub directories.
- Any decision about the concurrent public-trace-corpus file collision
  (§C/§O) — this is not this audit's or Pass 2's call; it needs the user or
  the coordinating session to resolve which implementation survives.
- Reviewing/archiving the 2 unmerged branches
  (`phase2b13-selector-training-after-diversity`,
  `phase2c-final-selector-improvement`) before any action.
- Any decision to compress/relocate the ~30MB tracked telemetry CSV (§Q).

---

## Y. Recommended Pass-2 Commit Structure

Given the "no broad edits yet" mandate is lifted in Pass 2 but scope should
stay tight: one commit per logical unit is cleaner than one giant commit,
given how many independent docs are touched —

1. `docs: refresh docs/PROJECT_MAP.md to current HEAD state` (Category A,
   the single highest-value change)
2. `docs: append missing experiment rows to WORK_STATUS.md and EXPERIMENT_INDEX.md`
3. `docs: add Family-B live replication prep to RESUME_HERE.md/NEXT_ACTIONS.md; shrink NEXT_ACTIONS.md duplication`
4. `docs: refresh README.md Current Checkpoint; refresh ACTIVE_EXPERIMENT_PROTECTED_PATHS.md`
5. `docs: mark superseded-banner cluster (composition/pause snapshot docs)`
6. `chore: document *.jsonl pre-existing-tracked-files exception in .gitignore`
7. (Category D, separate/optional, only with authorization) `chore: reconstruct unified_utility_matrix_v2 build manifest`

Do not combine branch/directory deletions (Category D/E) into the same
commits as documentation work, since those need separate explicit
authorization and a different review lens.

---

## Z. Exact Single Next Action

Before Pass 2 touches anything: **resolve the concurrent public-trace-corpus
file collision (§C/§O)** — determine which of the two in-progress,
uncommitted implementations (the parent session's original
`src/llmserveopt/workloads/public_trace_corpus/{__init__,schema,adapters}.py`
+ `tests/test_public_trace_corpus_schema.py` + `scripts/data/build_public_trace_corpus_v1.py`,
versus the currently-on-disk single-file `src/llmserveopt/workloads/public_trace_corpus.py`
+ `scripts/build_public_trace_corpus_v1.py` + `tests/test_public_trace_corpus_v1.py`)
is the one to keep, since only the latter currently exists on disk. This is
orthogonal to Pass 2's documentation-reconciliation scope but is time-sensitive
(uncommitted work) in a way nothing else in this report is.

---

**PASS1_VERDICT: READY_FOR_PASS2**

**PASS2_SCOPE:** Refresh the four stale living-status docs
(`README.md`, `docs/PROJECT_MAP.md`, `docs/current/WORK_STATUS.md`,
`docs/current/EXPERIMENT_INDEX.md`) plus `RESUME_HERE.md`/`NEXT_ACTIONS.md`
and `ACTIVE_EXPERIMENT_PROTECTED_PATHS.md` to reflect the true current HEAD
state (through Family-B live replication prep), add the new-policy-synthesis
roadmap section, mark the identified superseded-doc cluster, and add one
`.gitignore` clarifying comment — all pure documentation content changes,
with the three abandoned experiment stub directories and the 49 merged
branches explicitly deferred to a separately-authorized Category D/E pass.
