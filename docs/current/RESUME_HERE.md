# Resume Here

**This is the single canonical entry point for this repository.** Every other
"start here" / "resume" document in this repository now points here first.
Read this document before any other status document. It supersedes its own
prior content (the 2026-07-25 pause note below is now historical — see
`docs/current/pause_2026_07_25/` for that provenance) and the older
2026-07-23 pause documents in this directory, all of which describe earlier
project phases.

---

## A. Where are we?

| Field | Value |
|---|---|
| Repository | `llm-serving-heuristic-evolution` |
| Branch | `contextual-compositional-heuristics-20260731` |
| Authoritative SHA (at time of writing) | `e413ba1dcbe8b79f0ebc0f7511e846481548b6bb` — **verify with `git rev-parse HEAD`, do not trust this number without checking** |
| Remote | `origin/contextual-compositional-heuristics-20260731` (GitHub, `SoroushVahidi/llm-serving-heuristic-evolution`, private) |
| Expected state on resume | Working tree clean, 0 ahead / 0 behind `origin`. If not, stop and reconcile before doing anything else — do not build on top of an unexpected diff. |
| `main` branch | Stale (ancestor of this branch, last commit 2026-07-17) — not the branch to resume from |

```bash
git fetch origin
git status --short --branch
git rev-parse HEAD
git rev-list --left-right --count HEAD...@{u}
```

---

## B. What is the project?

This project studies **scheduling-policy selection for LLM inference
serving**: a GPU-calibrated discrete-event simulator in which requests
arrive with unknown output length under SLO constraints, and a policy
decides admission order and placement. Concretely:

- **Verified scheduling primitives** — 28 registered primitives
  (RANKING/ADMISSION/PLACEMENT/BATCHING/RESOURCE_GUARD,
  `src/llmserveopt/policies/primitives.py`), from which representative
  policies are reconstructed (6/7 exact, 1/7 documented approximate).
- **DSL / composition framework** — a restricted, verifiable JSON DSL
  (`src/llmserveopt/heuristics/`) for composing primitives into weighted
  sums, sparse top-k mixtures, conditional branches, and admission gates
  with declared fallback.
- **Oracle simulator datasets** — a reproducible, resumable oracle
  composition dataset (CC4: 12 workload windows, 34 verified candidates,
  408 true simulator executions).
- **Contextual composition predictor** — a regret-regression model with
  leave-one-window-out selection (CC5), finalized behind a frozen,
  deterministic operating-envelope gate.
- **Uncertainty-aware fallback** — the CC5 envelope trusts the predictor in
  7 of 12 evaluated regimes and falls back to a validation-tuned safe
  choice elsewhere.
- **External baseline validation** — faithful, pinned reimplementations of
  vLLM, vLLM-chunked-prefill, Sarathi-Serve, DistServe, TetriInfer, Llumnix,
  PARS-Serve-2026, and VTC, each checked against the official paper/artifact.
- **Stress-test library** — a target/counter-regime catalog
  (`docs/research/algorithm_stress_tests/`) for testing whether a baseline's
  claimed mechanism actually holds under adversarial conditions, not just
  average-case workloads.

---

## C. What has been completed?

**Contextual composition (CC0–CC5), all COMPLETE:**
CC0 (repo/evidence stabilization) → CC1/CC1b (composition-opportunity gap
measured) → CC2 (primitive interface, 6/7 exact reconstructions) → CC3
(DSL/verifier, 447 tests) → CC4 (oracle dataset, 66.7% composition-oracle
win rate) → **CC5 finalized `COMPLETE_REGIME_SPECIFIC`** (§D).

**External baselines:**
- **vLLM-LTR** — complete, official checkpoint hash+architecture verified, `EVALUATION_ONLY`.
- **PARS-Serve-2026** — complete, official-code reproduction with a locally trained, fidelity-verified checkpoint, `EVALUATION_ONLY`; zero unique wins across 8 canonical-suite families.
- **VTC** — official `VTCReqQueue` reused via adapter (real, unmodified code), fairness-validated 108-run sweep, `FOUNDATIONAL_CANDIDATE` (scientific classification, not registered as deployable).
- **Sarathi-Serve** — faithful reimplementation + real Wulver A100 GPU validation (N=5 repeated trials); 7-entry stress-test catalog; the real-hardware decode-protection mechanism was found provably unreproducible in-simulator under FCFS-strict admission (a structural finding, documented, gates revised accordingly).
- **Apt-Serve** — Strategy C (reuse-as-component) Wulver probe classified `STRATEGY_C_VIABLE_WITH_LIMITATIONS`. Simulator dual-tier cache architecture and external adapter specifications designed, and Phase A configuration schema and interface scaffolding completed (24 tests passing). Phase B (HybridCacheManager) is queued.
- **Llumnix** — faithful reimplementation exists and is registered (`llumnix_faithful.py`, 36 fidelity tests, 188/188 passing including cross-baseline integration); **no comparative evaluation has ever been run** (§E).
- **DistServe** — also has a faithful reimplementation and is registered (`distserve_faithful.py`, implemented the same day as Llumnix, 35 fidelity tests, all passing); `docs/BASELINE_STATUS.md`'s DistServe row was found stale in this same way during Query 3 and corrected. Same evidence gap as Llumnix: no comparative evaluation exists yet.

---

## D. What is scientifically established?

- **CC5's contextual predictor beats best fixed and the hard selector, but
  not best global composition overall.** Frozen operating-envelope system:
  ANWG 0.4044, statistically beats best fixed (paired 95% CI
  [+0.0074,+0.0235], p<0.0001) and the hard selector (paired 95% CI
  [+0.0020,+0.0199], p=0.021), 0 completion violations. Its edge over
  `best_global_composition` (+0.0019 ANWG) is **not** statistically
  distinguishable from zero (paired 95% CI [-0.0044,+0.0083], p=0.5654) —
  full-context superiority over global composition was **not** established.
  This is documented honestly, not hidden.
- **CC5 is regime-specific, not universal.** Trusted envelope:
  `burst_transition`, `kv_pressure`, `long_output`, `prediction_noise`,
  `saturated`, `selective_admission_trap`, `underloaded`. Untrusted (falls
  back to best-fixed/best-global): `azure_conversation_like`,
  `burstgpt_derived`, `long_prompt`, `mixed_slo`, `priority_conflict`.
- **Uncertainty/fallback is calibrated** — the envelope is derived
  exclusively from development-split (never held-out) evidence, and was
  evaluated once, held out, at the numbers above.
- **VTC contributes a distinct fairness niche**, not a throughput win — it
  wins/ties the Jain's-index fairness comparison in 17/18 family×seed
  combinations, with a real, bounded ANWG trade-off (0.680 vs. SCORPIO's
  0.984) in the one family designed to expose its SLO-blindness.
- **PARS-Serve-2026 and vLLM-LTR are evaluation-only** — complete,
  independently verified, but dominated by internal policies
  (`shortest_output_first`/`estimated_service_time_first`/
  `scorpio_style_slo_guard`/`regression_anwg_selector`) in every
  discriminative regime tested.
- **Sarathi-Serve's real-hardware decode-protection mechanism does not
  reproduce inside this simulator under FCFS-strict admission** — a known,
  documented, structural simulator limitation, not a tuning gap; stress-test
  gates were revised to a coarser, still-genuinely-discriminating check.
- **Apt-Serve's Strategy C vs. D question is resolved
  (`STRATEGY_C_VIABLE_WITH_LIMITATIONS`), and architecture design is complete**.
  The paper's hybrid KV/hidden-state cache design gap has been scoped into
  a structured dual-tier cache extension. Implementation of Phase A
  (configuration scaffolding) is the exact next step.
- **Llumnix's implementation is real and tested, but unevaluated** — 188/188
  tests passing is evidence of correctness, not of competitiveness against
  other policies. Treating "tests pass" as "validated" was an identified
  and corrected error this project made about its own status documentation
  (`CCD-021`).

---

## E. What is unfinished?

### LOCAL_UNFINISHED (in order)

1. **Add Llumnix stress-test catalog entries** — zero exist today; most of
   the target/counter regime matrix is representable now or needs only two
   small simulator extensions (migration-bandwidth modeling,
   concurrent-transfer contention). See
   `docs/audits/llumnix_official_artifact_audit_20260806.md` §12/§14.
2. **Generate/validate Llumnix target and counter workloads** — generators
   for the cheapest gaps (control-loop delay, migration-cost-exceeds-benefit,
   tiny-request overhead) do not exist yet.
3. **Run the Llumnix comparative evaluation** — `llumnix_faithful` vs. the
   existing deployable policy set on multi-instance configs; no new code, no
   Wulver required, purely local CPU. This is the single most load-bearing
   gap identified in the current baseline audit.
4. **Independently verify the results** — matching this project's standing
   convention for every other external baseline (re-run, re-check for
   mismatches, do not trust a single pass).
5. **Classify Llumnix** — `docs/BASELINE_STATUS.md`'s current entry is
   `UNESTABLISHED pending evaluation`; update it once (3)/(4) complete.
6. **Perform the next external-baseline checkpoint** — a short report
   summarizing whether the current baseline set (vLLM-LTR, PARS,
   VTC, Sarathi, Apt-Serve, Llumnix) is now sufficient evidence to revisit
   CC6, or whether further baseline work is needed first.
7. **Run the DistServe comparative evaluation** — not an audit: `distserve_faithful.py`
   already exists (implemented the same day as Llumnix, 35/35 tests passing);
   it has the exact same evidence gap as Llumnix (implemented, unevaluated).
   Same evaluation pattern as items 1–5 above, sequenced after Llumnix.
8. **Decide whether external evidence is sufficient to revisit CC6** — this
   decision is gated on (6), not assumed.

### WULVER_DEFERRED

**None currently queued.** The only workstream that previously lived here
— the Apt-Serve Strategy C CPU probe (authenticate, submit, inspect,
collect, decide, commit/push) — completed in full on 2026-08-06 (jobs
1163456/1163782/1164406; see
`docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md`). The
`login02.tartan.njit.edu` GSSAPI auth issue described in the prior
version of this section (see
`docs/audits/project_pause_reconciliation_query2_20260806.md` for that
diagnosis) was specific to that earlier, non-interactive audit pass and
did not recur once a real interactive Wulver session was available. The
next Apt-Serve action (adapter + dual-tier cache interface design) is a
design task, not a Wulver-execution task — it does not belong here. A
future GPU-execution step (e.g. real comparative evaluation once an
adapter/implementation exists) would repopulate this section when it is
actually queued.
8. Synchronize this local branch afterward (`git pull --ff-only`).

**This local finalization sequence intentionally defers all Wulver
reconciliation — do not assert that no Wulver job exists.** No local process
can see a job submitted independently from a direct Wulver-side session; the
correct statement is "unverified from here," not "does not exist."

---

## F. What must not be done

- Do not start CC6 implementation before the external-baseline checkpoint
  (§E, item 6) — CC6 is queued but explicitly restricted and not started.
- Do not call simulator proxies ("style"/"inspired" policies) official
  baselines — only the names in `EXTERNAL_BASELINE_REGISTRY`
  (`src/llmserveopt/policies/external_baselines_registry.py`) are faithful
  reimplementations.
- Do not treat passing tests as comparative validation (`CCD-021`) — a
  baseline with 100% passing fidelity tests may still have zero evaluation
  evidence, as Llumnix currently does.
- Do not infer Wulver job state from local files, process lists, or stale
  documentation alone (`CCD-023`) — attempt direct access first, and label
  the result `REMOTE_STATE_UNVERIFIED` rather than a false negative when it
  fails.
- Do not modify canonical workloads or benchmark definitions to favor a
  particular policy's evaluation numbers.
- Do not register a `FOUNDATIONAL_CANDIDATE` as deployable/foundational
  without an explicit decision gate — VTC, for example, remains a
  scientific classification only, not registered.

---

## G. Exact next local task

**"Llumnix stress-test coverage and first comparative evaluation"**

Do not execute it as part of reading this document — it is the next task to
pick up, not a command to run now. See §E items 1–5 and
`docs/audits/llumnix_official_artifact_audit_20260806.md` §12–§14 for full
scope.

---

## H. Exact next Wulver task

**None currently queued.** The Apt-Serve Strategy C CPU probe that
previously occupied this slot executed to completion on 2026-08-06 —
see `docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md` §9b for
the resulting `STRATEGY_C_VIABLE_WITH_LIMITATIONS` decision. The next
Apt-Serve action (adapter + dual-tier cache interface design, §10 of
that document) does not require Wulver. The next local action is the
Llumnix comparative evaluation (§G above), which also does not require
Wulver.

---

## Where to go next

- **Full navigation map:** `docs/current/PROJECT_MAP.md`
- **Per-workstream status table:** `docs/current/WORK_STATUS.md`
- **Ordered, dependency-aware action list:** `docs/current/NEXT_ACTIONS.md`
- **High-impact scientific decisions, summarized:** `docs/current/SCIENTIFIC_DECISIONS.md`
- **Full decision log:** `docs/contextual_composition_decisions.md`
- **Dated pause note (why paused, exact resumption procedure):** `docs/current/PROJECT_PAUSE_HANDOFF_20260806.md`
- **Latest reconciled factual snapshot:** `docs/current/PROJECT_SNAPSHOT_20260806.md`
- **CC-specific technical roadmap (detailed):** `docs/START_HERE_CONTEXTUAL_COMPOSITION.md` → `docs/contextual_composition_roadmap.md`
- **Cross-baseline status index:** `docs/BASELINE_STATUS.md`

## Historical provenance (superseded, do not treat as current status)

- `docs/current/pause_2026_07_25/` — the 2026-07-25 pause snapshot (dataset-expansion branch). Superseded by the contextual-composition branch and this document.
- `docs/current/PROJECT_HANDOFF_2026-07-23.md`, `project_handoff_state.json`, `PAUSE_PROVENANCE_2026-07-23.md` — the 2026-07-23 pause checkpoint. Superseded.
- `docs/current/PROJECT_STATUS.md`, `docs/current/EXPERIMENT_INDEX.md` — historical status documents for the pre-CC-branch project phase. Retained as provenance only.
