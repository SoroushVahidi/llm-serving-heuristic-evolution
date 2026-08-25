# Project Snapshot — 2026-08-06 (Query 2 Reconciliation)

> **SUPERSEDED FOR CURRENT STATUS.**
> See [`docs/current/RESUME_HERE.md`](RESUME_HERE.md) for authoritative current state.
> This snapshot (2026-08-06) predates MF-PSD, all NO_GOs, hierarchical routing, live re-evaluation, and Family-B replication prep.

**Status:** reconciled factual snapshot, produced by Query 2 of a four-query
project-pause sequence. This is not yet the final resume document — Query 3
will produce/refresh that. This file records what was true, and verified,
at the end of Query 2.

---

## A. Repository state

| Field | Value |
|---|---|
| Branch | `contextual-compositional-heuristics-20260731` |
| Starting SHA (Query 2) | `f967c095826900aed0eb0326d3d1f3ea60936261` |
| Authoritative remote | `origin/contextual-compositional-heuristics-20260731` (GitHub, `SoroushVahidi/llm-serving-heuristic-evolution`, private) |
| Working tree | Clean at start of Query 2; this snapshot is part of the reconciliation commit |

`main` is stale (ancestor of this branch, last commit 2026-07-17) — this branch
is the authoritative line of work, not `main`.

---

## B. Research phases (contextual composition)

| Phase | Status |
|---|---|
| CC0 | COMPLETE |
| CC1 / CC1b | COMPLETE |
| CC2 | COMPLETE (6/7 reconstructions EXACT, 1/7 documented APPROXIMATE) |
| CC3 | COMPLETE (8/8 constructs, 447 tests) |
| CC4 | COMPLETE (12 windows, 34 candidates, 408 executions, 66.7% oracle-win rate) |
| CC5 | **COMPLETE (`COMPLETE_REGIME_SPECIFIC`)** — frozen operating-envelope predictor beats best-fixed (95% CI [+0.0074,+0.0235], p<0.0001) and hard selector (95% CI [+0.0020,+0.0199], p=0.021); edge over `best_global_composition` (+0.0019 ANWG) NOT statistically distinguishable from zero (95% CI [-0.0044,+0.0083], p=0.5654) |
| CC6 | **NEXT, restricted, NOT started.** Scope: controlled temporal adaptation only inside the CC5 trusted envelope (`burst_transition`, `kv_pressure`, `long_output`, `prediction_noise`, `saturated`, `selective_admission_trap`, `underloaded`), with hysteresis and fallback. Blocked pending explicit future authorization to begin implementation — this is a standing roadmap instruction, not new in this query. |
| CC7 | BLOCKED on CC6 |
| CC8 | Not reached |

No changes were made to CC5/CC6 status, algorithms, or benchmark semantics in
this query, per this query's own scope restriction.

---

## C. External baselines

| Baseline | Implementation | Fidelity | Stress-test coverage | Comparative evaluation | Current classification | Blocker | Next action |
|---|---|---|---|---|---|---|---|
| **vLLM-LTR** | Complete | Official checkpoint, hash+architecture verified | N/A (predates canonical suite) | Complete, WildChat control only | `EVALUATION_ONLY` | None | None — revisit only if canonical-suite regime is prioritized |
| **PARS-2023** | Proxy only (`estimated_service_time_first`) | Proxy/inspired | Yes, standard set | Regular internal comparison | Already foundational as proxy | None | None planned |
| **PARS-Serve-2026** | Complete, official-code reproduction, locally trained+verified checkpoint | Trained, hash-verified, 10/10 fidelity tests | Full canonical suite (8 workloads) | Complete, independently verified | `EVALUATION_ONLY` | None | None — zero unique wins across 8 families |
| **Sarathi-Serve** | Faithful reimplementation + real Wulver A100 GPU validation (N=5) | Faithful + real-hardware validated | 7 catalog entries, 6 executable pass | Real-hardware repeated-trial vs. vLLM complete | Foundational (internal) | None | None — catalog coverage complete |
| **VTC** | Official policy reused via adapter (real, unmodified `VTCReqQueue`) | Real official algorithm, adapter-wrapped | 108-run fairness sweep, 45/45 fidelity tests | Complete, independently re-verified | `EVALUATION_ONLY` (deployment); `FOUNDATIONAL_CANDIDATE` (scientific, not registered) | Native reimplementation before foundational registration | Native, non-wrapped reimplementation |
| **DistServe** | ~~Not implemented on this branch~~ **Correction (Query 3, same day):** this was itself a stale claim, discovered while reconciling status docs — `distserve_faithful.py` already exists on this branch (implemented 2026-07-18, alongside Llumnix), registered, 35/35 tests pass. See `docs/current/WORK_STATUS.md` for the corrected row. | Faithful, untested-comparatively | None for implementation | Not evaluated | `IMPLEMENTED_UNEVALUATED` (same gap as Llumnix) | None for implementation | Comparative evaluation, same pattern as Llumnix |
| **Llumnix** | **Complete** — faithful reimplementation (`llumnix_faithful.py`, 385 lines, pin `a90824307...`, Apache-2.0) | Faithful (independent reimplementation, OSDI-badge-verified pin, not executed official code) | **Zero catalog entries** (1 candidate-identification row only) | **Not run — genuine open gap** | `UNESTABLISHED` pending evaluation | None for implementation; evaluation itself is the blocker | Run Phase F comparative sweep (`llumnix_faithful` vs. deployable policy set, multi-instance configs) — cheap, no Wulver required |
| **Apt-Serve** | **Not implemented.** Official artifact audited (`CODE_ONLY` reproducibility); Strategy C/D probe infrastructure fully prepared (2 SLURM scripts, 2 Python probes, syntax-validated) | N/A — no code yet | None | Not evaluated | Not established — Strategy C vs. D unresolved | **Wulver GSSAPI authentication failing** (see §E) | Resolve Wulver SSH auth, then submit `scripts/slurm/wulver_apt_serve_strategy_c_cpu_probe.sbatch` |

Full detail and provenance for every row: `docs/BASELINE_STATUS.md` (authoritative
cross-baseline index, corrected in this query for the Apt-Serve and Llumnix rows).

---

## D. Datasets and workload suites

| Suite | Status |
|---|---|
| WildChat control | Used by vLLM-LTR/PARS evaluations; not re-verified this query (no filesystem drift since Query 1) |
| Canonical synthetic benchmark suite | Committed (`4972dd5`), used by PARS/vLLM-LTR |
| VTC fairness suite | 6 repaired workloads × 3 seeds × 6 policies = 108 runs, 45/45 tests pass |
| Sarathi stress-test catalog | 7 entries (2 target + 3 counter + 2 literature-motivated), 6 executable pass, 1 spec-only `NOT_REPRESENTABLE` |
| Algorithm Stress-Test Library | Literature-grounded (`71d07af`), extended for Sarathi (`74c7fc8`) |
| Apt-Serve specifications | Probe scripts prepared, not yet executed — no stress specs exist yet (correctly not claimed) |
| **Llumnix missing coverage** | **Zero stress-test-catalog entries.** Most of the target/counter regime matrix (14 named regimes) is representable today or needs only two small simulator extensions (migration-bandwidth modeling, concurrent-transfer contention, ~150-300 LOC total) — see `docs/audits/llumnix_official_artifact_audit_20260806.md` §12 |

---

## E. Remote/HPC state

**Apt-Serve/Wulver — verified, not guessed:**

- No SLURM job has ever been submitted for the Apt-Serve probe (confirmed: commit
  message states this explicitly; `results/wulver_imports/` contains no Apt-Serve
  output; no job ID exists anywhere in git history or documentation).
- **Wulver SSH authentication fails.** Diagnosis performed this query (all read-only,
  non-destructive):
  - `klist` shows a currently-valid Kerberos ticket (`sv96@NJITDM.CAMPUS.NJIT.EDU`,
    valid through 18:58 today).
  - `kvno host/login02.tartan.njit.edu` **succeeds** — a valid service ticket is
    obtainable (`kvno = 41`, full realm `NJITDM.CAMPUS.NJIT.EDU`).
  - System clock is NTP-synchronized (`timedatectl`: `System clock synchronized: yes`)
    — ruling out clock skew as the cause.
  - `~/.ssh/config`'s `Host login02` alias correctly sets `User sv96`; a prior
    attempt (Query 1) had bypassed this by using the bare FQDN directly and
    authenticated as the wrong local user (`soroush`) — retried correctly as `sv96`
    this query and **still failed**.
  - `ssh -vvv` shows the client offering `gssapi-with-mic`, the server responding
    with the same "can continue" method list twice (i.e., rejecting the attempt
    without accepting it), then falling through to "No more authentication methods
    to try" (keyboard-interactive is available server-side but intentionally not
    attempted under `BatchMode=yes`, to avoid triggering an interactive prompt in a
    non-interactive diagnostic).
  - One candidate lead, **not confirmed**: the local ticket cache holds two
    service-ticket entries for `host/login02.tartan.njit.edu` — one with an
    apparently empty/referral realm suffix, one with the full realm (obtained via
    `kvno` during this diagnosis) — raising the possibility that SSH's GSSAPI
    library is presenting the wrong (referral) ticket. Not verified further, since
    doing so would require additional live authentication attempts against the real
    login node, which this task's own instructions say to avoid repeating
    unnecessarily.
  - **Required user action:** this most likely needs either IT/HPC-support
    attention (server-side GSSAPI/keytab configuration) or a manual
    `kdestroy && kinit` cycle to clear the ambiguous ticket-cache entry, tried
    interactively by the user rather than by an automated non-interactive probe.
- **Status: `REMOTE_STATE_UNVERIFIED`** for anything beyond "no job was submitted
  as of the last local evidence." A Wulver-side session could in principle have
  submitted and completed a job independent of this workstation's visibility —
  no evidence either confirms or rules this out, and this snapshot does not claim
  otherwise.

**Llumnix:** no Wulver involvement yet — evaluation blocker is purely local-CPU work
(a comparative simulator sweep), not compute access.

---

## F. Immediate next sequence

Evidence-based, adjusted from the generic four-step suggestion in this query's own
brief:

1. **Resolve Wulver SSH/Kerberos access** (likely needs interactive `kinit`/ticket-cache
   cleanup by the user, or HPC-support involvement) — this is the one blocker
   standing in front of the Apt-Serve Strategy C/D question, and is cheap to
   attempt relative to everything else queued.
2. **Run the Llumnix comparative evaluation** (Phase F in
   `docs/audits/llumnix_official_artifact_audit_20260806.md` §14) — no Wulver
   needed, no new code needed, purely local CPU; this is now the single cheapest,
   most load-bearing gap identified across both baselines audited this week.
3. **Submit the prepared Apt-Serve Strategy C CPU probe** once Wulver access works —
   resolves the standing Strategy C vs. D question; do not guess this from code
   reading (`CCD-022`).
4. **Only after (2) and (3):** decide whether Llumnix stress-test generators
   (§12/§14 of the Llumnix audit) or Apt-Serve full implementation is the better
   use of the next work session — that decision is evidence-gated on (3)'s actual
   outcome (Strategy C vs. D changes Apt-Serve's implementation scope
   substantially) and is explicitly deferred, not decided here.
5. **CC6** resumes only after a future query explicitly authorizes it and only
   after the baseline-evaluation checkpoint above — unchanged from the existing
   roadmap instruction, not newly imposed by this reconciliation.

DistServe audit/evaluation (mentioned as a possible Query 2+ item in this query's
brief) is **not** promoted ahead of the above — no new evidence this query changed
its "not prioritized" status, and inserting it would be scope creep beyond what the
evidence supports.

---

## Duplicate worktree (carried forward from Query 1, unchanged)

`.claude/worktrees/phase2b9` (repository-relative; a linked git worktree)
(branch `worktree-phase2b9`, HEAD `429e96e` — byte-identical to the already-merged
`phase2b8-rule-selector-repair` tip) still contains six dirty files
(`docs/research_status.md`, `docs/selector.md` modified;
`configs/phase2b9_selector_robustness.yaml`,
`docs/audits/phase2b9_selector_training_audit.md`,
`docs/dataset_workload_decision.md`, `docs/external_baseline_decision.md`,
`scripts/run_phase2b9_selector_robustness.py`,
`tests/test_phase2b9_selector_robustness.py` untracked), all dated 2026-06-25
19:39-19:48. Every one of these paths is already present, committed, on branch
`phase2b9-selector-robustness-and-suite-freeze` (commit `5fe977b`, ~15 minutes
later the same evening) — re-confirmed unchanged this query. **Not removed in
Query 2**, per this query's explicit scope restriction. Safe for Query 4 to
`git worktree remove` after final validation, once the user confirms.
