# External-Baseline Comparison Analysis — 2026-08-24 (Pass 4)

**Scope freeze:** `Pext_common` = P6 + `official_vtc_joint_token_budget_remap` + `vllm_style_continuous_batching`.  
SOLA → Related Work. LTR → separate WildChat evidence only.

**Canonical P6 reproduction (STOP-checked):** SBS=0.314072, VBS=0.333106, headroom=0.019034 — **match**.

---

## Joint-240 Pext (240 × 8 = 1,920 cells)

| Quantity | P6 | Pext |
|---|---|---|
| SBS | 0.314072 (`kv_constrained_online`) | 0.314072 (`kv_constrained_online`) |
| VBS | 0.333106 | 0.333319 |
| Headroom | 0.019034 | 0.019247 |

**Deltas:** SBS +0; VBS +0.000214; headroom +0.000213.

**Incremental envelope gain vs P6:** VTC +0.000214; vLLM-style ≈0; both ≈ VTC alone.

### Predeclared questions

| Q | Answer |
|---|---|
| Q1 External improve SBS? | **No** |
| Q2 External improve VBS? | **Yes**, tiny (+0.000214) |
| Q3 VBS(Pext) ≫ SBS(Pext)? | **Yes** — headroom ≈0.0192 (≥0.01) |
| Q4 Multiple winners? | **Yes** — 8 policies win ≥1 scenario |
| Q5 P6 retain unique envelope? | **Yes** — all six P6 policies have positive leave-one-out VBS drop |
| Q6 External dominate any P6? | **No** strict dominance |
| Q7 Single external collapses opportunity? | **No** |
| Q8 Core portfolio claim survives? | **Yes** (qualitative) |

### Winners (Pext) / ε-unique (ε=0.01)

| Policy | Wins | ε-unique | VBS drop if removed |
|---|---:|---:|---:|
| least_laxity_first | 57 | 43 | 0.00576 |
| kv_constrained_online | 48 | 39 | 0.00709 |
| estimated_service_time_first | 45 | 28 | 0.00282 |
| weighted_fair_share | 34 | 25 | 0.00378 |
| full_prefill | 27 | 2 | 0.00011 |
| official_vtc… | 23 | 3 | 0.00021 |
| chunked_prefill_small | 3 | 2 | 0.00040 |
| vllm_style… | 3 | 0 | 0 |

### Bootstrap (n=1000, seed 20260825)

- Headroom(P6) 95% CI ≈ [0.01599, 0.02243]
- Headroom(Pext) 95% CI ≈ [0.01624, 0.02263]
- VBS(Pext)−VBS(P6) 95% CI ≈ [4.0e-5, 4.6e-4] (positive, small)
- SBS(Pext)−SBS(P6) = 0

**Interpretation (portfolio structure, not “we beat VTC”):** Strong external scheduler-level baselines do **not** replace P6 complementarity. VTC contributes a small unique envelope slice; vLLM-style does not expand the oracle envelope. SBS remains KV-constrained. Headroom stays ~0.019.

Artifacts: `experiments/external_baseline_comparison_v1/analysis/joint_*`

---

## public_trace_stress_v1 (M=16, C=32)

P6 stressed cells newly run: **360/360**. Same workload as external cells.

| | P6 | Pext |
|---|---|---|
| SBS / VBS / headroom | 0.985083 / 0.985083 / **0** | same (VTC ties P6; vLLM-style mean 0.2205) |

All six P6 policies + VTC yield **identical per-scenario ANWG** on this stress point (no ranking discrimination). vLLM-style is much worse (completion collapse under C=32 FCFS proxy).  
**Limitation:** stressed public is useful for stress realism / proxy fragility, not for P6 complementarity measurement.

---

## Family A + VTC (72 scenarios)

| Policy | Mean ANWG | Wins |
|---|---:|---:|
| WFS | 0.741 | 31 |
| ESTF | 0.720 | 22 |
| KV | 0.686 | 11 |
| VTC | 0.588 | 8 |
| LLF | 0.361 | 0 |

VTC does **not** dominate WFS (WFS better on many scenarios). Incremental envelope gain from VTC ≈ 0.00206. WFS/ESTF diversity **retained**.  
(ANWG-primary; VTC’s published fairness objective is not collapsed into ANWG alone.)

---

## Family B + vLLM-style (32 scenarios)

| Policy | Mean ANWG | Wins |
|---|---:|---:|
| full_prefill | 0.733 | 17 |
| chunked_prefill_small | 0.701 | 15 |
| vllm_style… | **0.0** | 0 |

Under Family B configs, the frozen vLLM-style proxy completes **0** requests (all dropped) — successful runs, catastrophic goodput. It does **not** eliminate Full vs Small-chunk complementarity (those two still split wins 17/15; proxy envelope gain = 0). Native-vLLM semantic validation remains separate.

---

## Family C relevance

`vllm_faithful` implements KV-block watermark admission + preemption → **mechanistically relevant**. Ran 72/72 (mean ANWG ≈ 0.609). Not required for common Pext; supporting mechanism cell only.

---

## LTR / SOLA

- **LTR:** separate WildChat evidence reusable; Spearman vs ESTF ~0.35–0.48 (behaviorally distinct); **not** in Pext_common.  
- **SOLA:** Related Work only (dual-SLO + cost-model fidelity gap).

---

## Adequacy

**`ADEQUATE_WITH_LIMITATION`**

Sufficient against “weak repo-only schedulers” via official VTC + disclosed vLLM-style proxy + native-vLLM semantics + separate LTR. Limitations: no faithful SOLA; public stress lacks P6 ranking contrast; Family B exposes proxy fragility.

---

## Core claim

**Survives.** Expanded Pext preserves nontrivial VBS−SBS headroom and multi-policy complementarity; externals do not collapse the portfolio opportunity.
