# Next Actions

Ordered, dependency-aware action map. Effort categories: `SMALL` (part of a
session), `MEDIUM` (most of a session), `LARGE` (multi-session). Nothing in
this document has been executed — it is a plan, not a log.

---

## IMMEDIATE LOCAL ACTIONS

### 1. Llumnix stress-test coverage
- **Prerequisite:** none — `llumnix_faithful.py` and its 36 fidelity tests already exist and pass.
- **Expected deliverable:** catalog entries in `configs/stress_tests/algorithm_stress_test_catalog.yaml` for the cheapest-first regimes: control-loop delay, migration-cost-exceeds-benefit, tiny-request overhead (all representable today with no simulator changes — see `docs/audits/llumnix_official_artifact_audit_20260806.md` §12).
- **Location:** Local.
- **Effort:** MEDIUM.
- **Stop condition:** if a generator requires the migration-bandwidth or concurrency-contention simulator extensions (§9 of the Llumnix audit) that don't exist yet, stop and scope that as a separate follow-up rather than half-building it.
- **Success criterion:** new catalog entries pass, following the existing Sarathi-catalog headroom-gate pattern (genuinely distinguishing, not trivially satisfied).

### 2. Llumnix comparative evaluation
- **Prerequisite:** none strictly (can run against today's deployable policy set even before (1) adds stress entries) — but doing (1) first gives richer workloads to evaluate against.
- **Expected deliverable:** `llumnix_faithful` scored against the existing deployable policy set on multi-instance configs (2, 3, 4, 8 GPUs via `multi_instance_migratory_config`), ≥3 seeds, canonical suite + new Llumnix-specific workloads.
- **Location:** Local (pure CPU simulator — no GPU, no Wulver needed).
- **Effort:** MEDIUM.
- **Stop condition:** none anticipated — this is the cheapest, most load-bearing gap identified in the current baseline audit.
- **Success criterion:** a scored comparison exists and is written up, whatever the result (win, loss, or tie against existing policies) — the point is evidence, not a predetermined outcome.

### 3. Llumnix independent verification
- **Prerequisite:** (2) complete.
- **Expected deliverable:** re-run of (2) with independent re-verification, matching this project's standing convention for every other external baseline (VTC, PARS, Sarathi all have this).
- **Location:** Local.
- **Effort:** SMALL.
- **Stop condition:** any mismatch between runs must be resolved before proceeding, not averaged away.
- **Success criterion:** zero unexplained mismatches, or an explained and documented source of nondeterminism.

### 4. Llumnix classification
- **Prerequisite:** (2) and (3) complete.
- **Expected deliverable:** update `docs/BASELINE_STATUS.md`'s Llumnix row from `UNESTABLISHED pending evaluation` to a real classification (`EVALUATION_ONLY`, `FOUNDATIONAL_CANDIDATE`, etc.) based on the actual result.
- **Location:** Local (documentation only).
- **Effort:** SMALL.
- **Stop condition:** none.
- **Success criterion:** `docs/BASELINE_STATUS.md` and `docs/audits/llumnix_official_artifact_audit_20260806.md` agree.

### 5. External-baseline checkpoint report
- **Prerequisite:** (4) complete. Apt-Serve Strategy C/D is now resolved (`STRATEGY_C_VIABLE_WITH_LIMITATIONS`, 2026-08-06 — see the Apt-Serve item below).
- **Expected deliverable:** a short report answering: is the current baseline set (vLLM-LTR, PARS, VTC, Sarathi, Apt-Serve, Llumnix) sufficient evidence to revisit CC6, or is further baseline work needed first?
- **Location:** Local.
- **Effort:** SMALL.
- **Stop condition:** none.
- **Success criterion:** an explicit go/no-go on CC6 readiness, evidence-based, not assumed.

---

## DEFERRED WULVER ACTIONS

These require a **direct Wulver login**, not this workstation. Do not attempt
any of these from a local, non-interactive session — see
`docs/current/RESUME_HERE.md` §E for why the last attempt failed and what
diagnosis has already been done.

### 1. Apt-Serve Strategy C/D probe — **DONE (2026-08-06)**
Executed on Wulver: jobs 1163456 (environment-construction bug, corrected),
1163782 (imports OK, micro-trace failed on a probe-script signature bug,
corrected), 1164406 (fully successful, reproduced structurally on a
second node). Official patched `vllm.core.scheduler.Scheduler` import
7/7, construction OK, 3/3 real `schedule()` micro-traces OK. Decision:
**`STRATEGY_C_VIABLE_WITH_LIMITATIONS`** — see
`docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md` §9b for the
five scoped caveats (technical reuse, legal redistribution, pinned-
environment requirement, remaining dual-tier-cache simulator gap,
full-system validation status).

### 2. Official-system validation where needed
- **Prerequisite:** (1) resolved in favor of Strategy C — **now satisfied**, with the limitations recorded in §9b.
- **Expected deliverable, next increment (design only, not this query):** the thin external-checkout adapter design and the minimal dual-tier cache interface specification (`docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md` §10) — implementation itself is a later, separate increment.
- **Location:** Local (design step and simulator-side implementation do not require Wulver) + Wulver only for a later real-GPU cross-check, if one turns out to be needed.
- **Effort:** LARGE overall (design increment is SMALL–MEDIUM) — genuinely new implementation work.
- **Stop condition:** do not begin full implementation before the design increment above is reviewed.
- **Success criterion:** matches this project's existing bar for a faithful baseline (fidelity tests, reference doc, registry entry).

### 3. Wulver-side commit/push reconciliation — **DONE (2026-08-06, this pass)**
Compact provenance artifacts (JSON probe reports, hashes, pip freeze,
job manifests) and the corrected probe scripts/audit doc were committed
and pushed to `origin/contextual-compositional-heuristics-20260731`; raw
per-job logs remain Wulver-local only, as intended.

---

## AFTER LLUMNIX AND APT-SERVE

### 1. DistServe comparative evaluation (re-scoped during Query 3 — not an audit)
- **Correction (2026-08-06):** Query 3's documentation-reconciliation pass
  found `docs/BASELINE_STATUS.md`'s DistServe row was **also** stale, the
  same way Apt-Serve's and Llumnix's were before Query 2 — `distserve_faithful.py`
  already exists (implemented 2026-07-18, alongside Llumnix), is registered,
  and has 35 passing fidelity tests. DistServe does **not** need a
  green-field official-artifact audit; it needs the same missing step as
  Llumnix: a comparative evaluation.
- **Prerequisite:** Llumnix (IMMEDIATE 1–5) resolved. Apt-Serve's Strategy C/D decision (WULVER 1) is now resolved (`STRATEGY_C_VIABLE_WITH_LIMITATIONS`, 2026-08-06) — DistServe is sequenced after both, not because it needs their output, but because Llumnix/Apt-Serve were judged higher-priority gaps first.
- **Expected deliverable:** `distserve_faithful` scored against the deployable policy set, using its disaggregated prefill/decode topology, following the same evaluation pattern as Llumnix's comparative sweep.
- **Location:** Local (pure CPU simulator).
- **Effort:** MEDIUM.
- **Stop condition:** none anticipated.
- **Success criterion:** a scored comparison exists; `docs/BASELINE_STATUS.md`'s DistServe row is updated from `UNESTABLISHED` to a real classification.

### 2. Decide whether DistServe implementation is necessary
- **Prerequisite:** (1) complete.
- **Expected deliverable:** an explicit go/no-go decision, added to the decision log.
- **Location:** Local.
- **Effort:** SMALL.
- **Success criterion:** a recorded decision with rationale, not silence.

### 3. Hosted API validation plan
- **Prerequisite:** none blocking — could run in parallel with the above, but sequenced here as lower priority than baseline resolution.
- **Expected deliverable:** a plan for whether/how to extend the existing Cohere/Gemini pilot work (see `docs/real_llm_cohere_gemini_comparison.md`) further.
- **Location:** Local (API calls, no Wulver).
- **Effort:** SMALL (planning only).

### 4. External-baseline sufficiency review
- **Prerequisite:** (1)–(3) above.
- **Expected deliverable:** a final review of whether the full external-baseline set is sufficient for the project's scientific claims, before returning to CC6.
- **Location:** Local.
- **Effort:** SMALL.

---

## RETURN TO CORE METHOD

### 1. Reassess the CC6 gate
- **Prerequisite:** the "AFTER LLUMNIX AND APT-SERVE" section complete, and IMMEDIATE LOCAL ACTION 5 (external-baseline checkpoint) complete.
- **Expected deliverable:** an explicit reassessment of whether CC6's entry condition (external-baseline checkpoint sufficiency) is met.
- **Location:** Local.
- **Effort:** SMALL.

### 2. Decide whether the restricted operating envelope is sufficient
- **Prerequisite:** (1).
- **Expected deliverable:** a decision on whether CC5's regime-specific envelope, as-is, is good enough to ship, or whether CC6's adaptation work is worth the additional scope.
- **Location:** Local.
- **Effort:** SMALL.

### 3. Decide: regime specialists vs. current CC5 result
- **Prerequisite:** (2).
- **Expected deliverable:** an explicit choice, recorded in the decision log, between building regime-specialist predictors or freezing CC5 as the final contextual-composition result.
- **Location:** Local.
- **Effort:** MEDIUM (analysis) before any implementation.

### 4. Only then begin CC6
- **Prerequisite:** (1)–(3), plus explicit authorization (per the standing roadmap instruction — this has never been a rubber stamp in this project's history).
- **Expected deliverable:** CC6 implementation, restricted to the 7 trusted regimes (`burst_transition`, `kv_pressure`, `long_output`, `prediction_noise`, `saturated`, `selective_admission_trap`, `underloaded`), with hysteresis and fallback.
- **Location:** Local.
- **Effort:** LARGE.
- **Stop condition:** do not enable contextual switching in unsupported regimes (`azure_conversation_like`, `burstgpt_derived`, `long_prompt`, `mixed_slo`, `priority_conflict`).
- **Success criterion:** matches CC5's own bar — paired statistical significance for any superiority claim, not point-estimate comparison alone.
