# External-Baseline Execution Pass 3 — 2026-08-24

**Verdict:** `EXTERNAL_BASELINE_PASS3_BLOCKED_BY_SLO_MAPPING`  
**HEAD:** `2987b7181efa2bc550d8a894c537eca8f6393eb6` (dirty preserved; no commit/push)

---

## Summary

1. Stressed-public **vLLM-style completed 60/60** (integrity OK alongside VTC 60/60).  
2. **`public_trace_stress_v1` formally frozen** (M=16, C=32).  
3. **LTR removed** from common Pext (`INPUT_REPRESENTATION_UNAVAILABLE`); WildChat prior eval reusable as separate evidence.  
4. **SOLA gate C** — dual TTFT/TPOT SLO mapping impossible without speculation; cost-model fits unpublished; **no implementation / no SOLA jobs**.  
5. Active common external set: **vLLM-style + VTC** (+ P6). Do **not** yet compute SBS/VBS(Pext) until author accepts this scope (or unlocks SOLA via a predeclared dual-SLO protocol).

---

## A–B. Preflight / Wulver

Host `al-khwarizmi`; branch unchanged; ahead 2; Wulver SSH OK; `squeue` empty; no active Pass-2 processes (jobs finished).

## C. vLLM stressed-public

COMPLETE 60/60; 0 fails; 20 windows × 3 sources; M=16 C=32; config hash `stress_v1_M16_C32`; ANWG present. Log: `logs/public_trace_stress_v1/vllm_20260824T235948Z.log`.

## D. Stress freeze

`FROZEN_PUBLIC_TRACE_STRESS_V1` in `stress_protocol.json`, stress protocol doc, `frozen_config.json`.

## E–G. LTR

Official input: prompt text. Joint/public: counts only → **`LTR_NOT_IN_COMMON_PEXT_MATRIX`**.  
WildChat prior: **`LTR_EXTERNAL_PROMPT_BEARING_EVIDENCE_REUSABLE`** (`results/vllm_ltr_first_comparative_evaluation/`, verification PASS).

## H. Final common Pext

`Pext_common` = P6 + vLLM-style + VTC.  
SOLA: Related Work / blocked. LTR: separate prompt-bearing track.

## I–N. SOLA

Primary sources: MLSys PDF + OpenReview; **no code**. Gap table in updated SOLA spec.  
Cost fits: MUST_PROFILE_LOCALLY / unpublished.  
Dual SLO: **blocked**. Memory: approximable later (moot).  
**Gate C** — no code, no tests, no profiling, no scientific SOLA jobs.

## O–S. SOLA implementation / jobs

None (blocked).

## T. Stressed-public integrity

| Method | n | sources | M,C | status |
|---|---|---|---|---|
| VTC | 60 | 20+20+20 | 16,32 | COMPLETE |
| vLLM-style | 60 | 20+20+20 | 16,32 | COMPLETE |

No scientific relative interpretation this pass.

## U–V. Updated artifacts

`run_matrix.csv`, fidelity ledger Pass-3, SOLA spec, frozen_config, stress freeze.

## W. Adequacy vs “weak repo-only schedulers”

Mitigated by: official VTC adapter; vLLM-style faithful FCFS proxy + separate native vLLM semantic validation; optional separate official LTR WildChat evidence.  
Honest gap: SOLA not faithfully comparable without dual-SLO redesign. Prefer fewer faithful baselines over a fake SOLA.

## X–Y. Blockers

Scientific: SOLA dual-SLO + portable cost model; author accept `Pext_common` without SOLA.  
Engineering: none for launched cells.

## Z. Safety

No commit/push/reset; P6 untouched; no SOLA tuning; no LTR fabrication; no Pext envelope numbers computed.

## AA. Exact next action

1. Author decision: accept `Pext_common` = P6+VTC+vLLM-style (SOLA Related Work; LTR separate), **or** authorize a **predeclared** dual-SLO overlay protocol (before any SOLA coding).  
2. If scope accepted: **then** run integrity-only merge of external columns and compute SBS/VBS(Pext) in a dedicated analysis pass.  
3. Do not implement `sola_faithful` under gate C.

## AB. Final verdict

**`EXTERNAL_BASELINE_PASS3_BLOCKED_BY_SLO_MAPPING`**
