# Next Actions

Ordered, dependency-aware action map. Effort categories: `SMALL` (part of a
session), `MEDIUM` (most of a session), `LARGE` (multi-session). Nothing in
this document has been executed — it is a plan, not a log.

---

## IMMEDIATE LOCAL ACTIONS

### 1. Llumnix stress-test coverage — **DONE (2026-08-06)**
- 17 entries (7 TARGET, 10 COUNTER) added to catalog; 13 executable validated and pass.
- Expected deliverables, generators, and headroom gates fully completed.

### 2. Llumnix comparative evaluation — **DONE (2026-08-06)**
- Scored comparison completed: 13 workloads x 5 policies x 3 seeds (195 runs).
- Report written to `docs/audits/llumnix_first_comparative_evaluation_20260806.md`.

### 3. Llumnix independent verification — **DONE (2026-08-06)**
- Independent verifier script written and executed with 975 checks and zero mismatches.
- Self-test corruption checks fully pass.

### 4. Llumnix classification — **DONE (2026-08-06)**
- Classified scientifically as `FOUNDATIONAL_CANDIDATE`.
- Updated `docs/BASELINE_STATUS.md` and `docs/current/WORK_STATUS.md`.

### 5. External-baseline checkpoint report — **DONE (2026-08-06)**
- Checked go/no-go CC6 readiness: we recommend progressing to CC6 once Apt-Serve and DistServe evaluations are fully closed (see `docs/audits/llumnix_first_comparative_evaluation_20260806.md`).

### 6. Apt-Serve Phase A implementation — **DONE (2026-08-06)**
- Configuration schema, interfaces, IPC schemas, and scaffolding completed with 24 tests passing.

### 7. Apt-Serve Phase B implementation — **DONE (2026-08-06)**
- Dual-tier HybridCacheManager and capacity/rounding logic implemented with 18 unit and scenario tests passing.

### 8. Apt-Serve Phase C implementation — **DONE (2026-08-06)**
- Subprocess adapter and versioned JSON IPC worker completed with 16 unit, verification, and protocol tests passing.

### 9. Apt-Serve Phase D implementation — **DONE (2026-08-06)**
- Static snapshot differential verifications and 24 focused scenario tests complete and passing.

### 10. Apt-Serve Phase E implementation
- **Prerequisite:** Phase D complete.
- **Expected deliverable:** multi-step simulator integration (Phase E of 8).
- **Location:** Local.
- **Effort:** MEDIUM.

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

### 2. Official-system validation where needed — **DESIGN & PHASE A DONE (2026-08-06)**
- Simulator architecture, dual-tier cache, external IPC adapter designed, and Phase A scaffolding implemented with 24 tests passing.
- Implementation of remaining phases is local (Wulver not needed until final validation).

### 3. Wulver-side commit/push reconciliation — **DONE (2026-08-06, this pass)**
Compact provenance artifacts (JSON probe reports, hashes, pip freeze,
job manifests) and the corrected probe scripts/audit doc were committed
and pushed to `origin/contextual-compositional-heuristics-20260731`; raw
per-job logs remain Wulver-local only, as intended.

---

## AFTER LLUMNIX AND APT-SERVE

### 1. DistServe comparative evaluation — **DONE (2026-08-06)**
- 6 workloads added to stress-test catalog (2 target, 4 counter).
- Scored evaluation completed: 5 workloads x 3 policies x 1 seed (15 runs).
- Full scientific report generated at `docs/audits/distserve_first_comparative_evaluation_20260806.md`.

### 2. Decide whether DistServe implementation is necessary — **DONE (2026-08-06)**
- Existing `distserve_faithful.py` verified as sufficiently faithful for algorithmic testing.
- Classified as `FOUNDATIONAL_CANDIDATE_FOR_DISAGGREGATION_PRIMITIVES_ONLY`. No new implementation necessary.

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
