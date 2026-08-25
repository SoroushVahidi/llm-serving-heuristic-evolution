# SOLA Faithful Reimplementation Specification

**Date:** 2026-08-24 (Pass 3 primary-source reconstruction)  
**Gate:** `SOLA_NOT_FAITHFULLY_REPRODUCIBLE` (outcome C)  
**Stop code:** `SOLA_BLOCKED_BY_SLO_MAPPING`  
**Recommendation:** **DEMOTE_TO_RELATED_WORK** — do not invent a weak proxy labeled as SOLA; keep `sola_style_state_aware` unlabeled as SOLA.

---

## 1. Official primary sources (Pass 3)

| Source | URL / location |
|---|---|
| MLSys 2025 PDF | https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf |
| OpenReview | https://openreview.net/forum?id=ubIvpetAd6 |
| Author PDF mirror | https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/ce55a1f2-ff0d-45f6-8985-f4f251b2a0d4.pdf |
| Slides | https://mlsys.org/media/mlsys-2025/Slides/3231.pdf |
| Official public code | **Not found** (re-checked 2026-08-24 Pass 3) |

Authors: Ke Hong, Xiuhong Li, Lufang Chen, Qiuli Mao, Xuefei Ning, Guohao Dai, Shengen Yan, Yun Liang, Yu Wang.  
License of code: unknown (unpublished).

---

## 2. Fidelity gap table

| SOLA component | Primary-source location | Exact published rule | Required inputs | In our simulator? | Adaptation needed? | Fidelity risk |
|---|---|---|---|---|---|---|
| TTFT SLO \(T^{TTFT}\) | §1–2; Fig.1 (e.g. 500 ms); Table 1 | Hard TTFT latency cap per request | Separate TTFT budget | **No** — only unified `slo_deadline` | Would require inventing dual-SLO overlay | **CRITICAL** |
| TPOT SLO \(T^{TPOT}\) | same | Hard TPOT/ITL cap | Separate TPOT budget | **No** | Same | **CRITICAL** |
| Real-time \(t^{TTFT}_{i,r}\), \(t^{TPOT}_{i,r}\) | §4.3 State Monitor | Accumulated from iteration times | Per-request phase latencies | Partial (sim TTFT/TPOT after completion; live TPOT during decode approximate) | Map step timings | Medium |
| System ratios \(p^{TTFT/TPOT}_i=\max_r t / T\) | §4.3–4.4.1; Table 2 | Mode switch: optimize less-fulfilled SLO subject to the other | Dual SLOs + live latencies | Blocked without dual SLOs | — | **CRITICAL** |
| Order \(F_i\) | §4.4.2; Fig.6 | Prefill-priority by predicted TTFT or decode-priority by predicted TPOT | Predicted latencies from cost model | Partial | Needs \(C^p,C^d\) | High |
| Workload \(n_i,k_i\) | §4.4.3 Eqs for constrained workload | Prefill tokens until TPOT constraint; or limit prefills until TTFT constraint | Cost-model predictions | Partial | Needs cost model | High |
| Algorithm 1 | §4.1.2 | Sort wait; peak-mem check; admit until \(n_i\)/\(k_i\) | Full strategy | Structure OK | Internals blocked | High |
| Peak-memory admit | §5.1 (LightLLM-style) | Reject add if predicted peak mem exceeds capacity | Peak predictor | KV tokens only | Approximate with KV+predicted length | Medium–High |
| Cost model \(C^p,C^d\) | §5.2 Eqs (3)–(4) | Polynomial in lengths/batch; **fitted coeffs** \(a_0..d_0,a_1..c_1\) | Profiling fit | No published numeric coeffs | MUST_PROFILE_LOCALLY or invent | **CRITICAL** |
| \(l^{left}_{i,r}\) | §4.3 | Predicted remaining output from length distribution | Causal predicted length | `predicted_output_tokens` available | Use predicted, never actual | Low if dual-SLO fixed |
| Parameter defaults | Eval tables / Fig.1 | Scenario-specific SLO numbers; SplitFuse example \(n_i=256,k_i=512\) | Defaults | Not portable to our discrete-event timescales | Speculative | High |
| Tie handling | Not explicit | Unspecified beyond sort | — | — | Must invent | Medium |
| Training | None for scheduler | Offline cost-model fit only | Profiles | — | — | — |

---

## 3. Cost-model coefficient classification

| Coefficient / quantity | Class |
|---|---|
| Polynomial form of \(C^p,C^d\) | PUBLISHED_PORTABLE_DEFAULT (structure only) |
| Fit values \(a_0..d_0,a_1..c_1\) | PUBLISHED_HARDWARE_SPECIFIC / **MUST_PROFILE_LOCALLY** (no numbers in paper) |
| Peak-memory LightLLM predictor internals | UNRESOLVED without artifact |
| Mapping ServiceModel step latency → \(C^p,C^d\) | DERIVABLE_FROM_EXISTING_SIMULATOR only as a **non-paper** substitute |

Local profiling of our RTX 5060 Ti / ServiceModel would **not** recover the authors’ A100+vLLM fits; it would create a new cost model. That is allowed only as disclosed portability adaptation **after** dual-SLO mapping exists. Dual-SLO mapping does **not** exist fairly today.

---

## 4. TTFT/TPOT SLO mapping decision

Frozen workloads expose a **single** absolute `slo_deadline` used by ANWG (completion-time deadline).

SOLA’s control law **requires** independent \(T^{TTFT}\) and \(T^{TPOT}\) and live fulfillment ratios (Table 2).

Candidate mappings considered and **rejected** as unfair/speculative:

1. Split `slo_deadline−arrival` with a free parameter \(\alpha\) into TTFT/TPOT budgets — **not in paper**; would be tunable against outcomes.  
2. Paste Fig.1 constants (500 ms / 200 ms) into all scenarios — wrong timescale vs our simulator.  
3. Infer TPOT = deadline/predicted_output — invents semantics not stated for SOLA’s constrained optimization.

**Conclusion:** `SOLA_BLOCKED_BY_SLO_MAPPING`.

---

## 5. Memory-model mapping

Paper: peak-memory prediction before admit (Alg.1 line 9; §5.1).  
Simulator: `current_kv_tokens`, `max_kv_tokens`, `predicted_output_tokens`.

A KV-based proxy using **predicted** (not actual) output length is conceivable as disclosed adaptation, but is moot while dual-SLO + cost-model blocks remain.

---

## 6. Reimplementation gate

**C. `SOLA_NOT_FAITHFULLY_REPRODUCIBLE`**

Major decision rules (dual SLO mode switch; fitted cost model) remain ambiguous or unmappable without speculative invention.

**Do not implement `sola_faithful` in Pass 3.**  
**Do not** retarget `sola_style_state_aware` as SOLA.

Paper recommendation: cite SOLA in Related Work as state-aware TTFT/TPOT scheduling; compare faithfully against VTC + vLLM-style only in the common matrix.

---

## 7. Historical note

Pass 1–2: spec-only. Pass 3: primary-source gap table completed; gate C confirmed.
