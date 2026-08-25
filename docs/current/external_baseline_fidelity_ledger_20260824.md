# External-Baseline Fidelity Ledger

**Date:** 2026-08-24  
**Pass:** Execution preparation + safe launch (Pass 1)  
**Repo HEAD (dirty worktree preserved):** `2987b7181efa2bc550d8a894c537eca8f6393eb6`

This ledger records official sources and fair adaptations for mandatory
scheduler-level baselines in the common Pext simulator matrix. Architecture
advantages are excluded from the common matrix.

---

### Method: vLLM-style continuous batching (M1)

Official source:
- Paper: Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” SOSP 2023 (arXiv:2309.06180).
- Scheduler semantics also documented for early vLLM releases.

Official repository:
- https://github.com/vllm-project/vllm

Version/commit:
- Pinned reference for local faithful reimplementation: tag `v0.1.0` / commit `67d96c29fba9b72cb4c4edbc26211c208a00ebdd` (see `docs/vllm_faithful_scheduler_reference.md`).
- Native GPU semantic validation (separate evidence): local vLLM ≈0.27.x under `experiments/real_vllm_mechanism_validation_v1/` — **not** the common-matrix cell.

Official license:
- Apache-2.0

Published objective:
- Continuous batching under KV block / sequence capacity; default FCFS waiting-queue admission with paged-KV accounting.

Published inputs/features:
- Arrival order, prompt length (for block reservation), running/swapped state, KV block availability.

Published scheduling mechanism:
- FCFS among waiting requests; per-iteration schedule of running then waiting; block-manager feasibility; optional preemption under memory pressure (modeled in `vllm_faithful`).

What we will reproduce exactly:
- Simulator-side FCFS continuous-batching + block-aware admission decisions via `src/llmserveopt/policies/vllm_faithful.py`.

What cannot be reproduced exactly:
- CUDA kernels, PagedAttention performance, real GPU timing, native vLLM 0.27 scheduler_v1 internals.

Any adaptation required for our common simulator:
- Service-model flags for joint cells: keep joint `ServiceModel` kwargs but set `allow_chunked_prefill=False` and `decode_first=True` so the **pre-chunked** default scheduler is the mechanism under test (chunked-prefill is a separate Family-B / native evidence track).
- Public name in Pext: `vllm_style_continuous_batching` (implementation id `vllm_faithful`).

Why that adaptation is fair:
- Aligns the common-matrix proxy with the published **default** continuous-batching / FCFS scheduler, not with a SJF-biased heuristic (`vllm_style_token_budget`) and not with native GPU timing.

Information available to method:
- Arrival time/order, prompt tokens, predicted remaining work only where the faithful policy already uses length for block growth (not oracle future output for ranking).

Information intentionally NOT available:
- True future output length for prioritization; multi-GPU topology advantages; disaggregation.

Status:
**FAITHFUL_REIMPLEMENTATION** (common matrix) + separate **NATIVE_ONLY** semantic validation artifacts.

---

### Method: VTC (M2)

Official source:
- Sheng et al., “Fairness in Serving Large Language Models,” OSDI 2024 (arXiv:2401.00588).

Official repository:
- https://github.com/Ying1123/VTC-artifact

Version/commit:
- `192c2e2014c69c8c6c699d7113c3822e4db632e6` (cloned to `~/.cache/external_baselines/VTC`, not vendored).

Official license:
- Apache-2.0

Published objective:
- Token-cost multi-tenant fairness under continuous batching (Virtual Token Counter).

Published inputs/features:
- Tenant/client identity, input/output token service costs, waiting/running requests, memory/batch capacity gates from the serving engine.

Published scheduling mechanism:
- `VTCReqQueue`: counter lift for new clients; prioritize least-served tenant; linear (or profile) cost accounting; work-conserving admission under memory/batch gates.

What we will reproduce exactly:
- Unmodified official `VTCReqQueue` decisions via `baselines/vtc/adapter/simulator_policy.py` (`VTCFairnessPolicy`).

What cannot be reproduced exactly:
- Full S-LoRA/GPU engine, profile cost function (hardware-specific), LoRA memory (disabled matching paper `--no-lora` vanilla mode).

Any adaptation required for our common simulator:
1. Single-GPU topology only (official artifact is monolithic).
2. Tenant map = request `class_id` (joint scenarios already emit `tenant_high` / `tenant_low` [+ `_late`]).
3. **Critical units remap for joint-240:** native `GPUConfig.max_batch_tokens` is a **request-count** cap in this simulator, but official VTC treats `batch_max_tokens` as a **token** budget. Feeding the count unchanged yields near-zero admission on joint prompts. Frozen override:
   - `batch_token_budget_override = max(step_token_budget, max_prompt_tokens_in_scenario)`
   - Still executes unmodified official gate code; only the numeric capacity argument is remapped (same class of adaptation as documented “fairness-isolation VTC” / Variant C in `docs/audits/vtc_fairness_benchmark_repair_20260805.md`).
4. Cost function: `linear` only.

Why that adaptation is fair:
- Without the remap, VTC is silently crippled by a **simulator units mismatch**, not by its published algorithm. Native P6 policies are not subject to that token-budget interpretation of `max_batch_tokens`. The remap restores a feasible official gate while preserving VTC ordering/accounting.

Information available to method:
- Tenant ids, prompt lengths (placeholder token ids), predicted output length for reservation accounting (as in adapter), served counters.

Information intentionally NOT available:
- Oracle actual output length for ranking; multi-GPU pools.

Status:
**OFFICIAL_CODE_ADAPTED**

---

### Method: SOLA (M3)

Official source:
- Hong, Li, Chen, Mao, Ning, Dai, Yan, Liang, Wang. “SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling,” MLSys 2025.  
  PDF: https://proceedings.mlsys.org/paper_files/paper/2025/file/bc82dbfbfa43232be85b8d9838f49c3e-Paper-Conference.pdf  
  OpenReview: https://openreview.net/forum?id=ubIvpetAd6

Official repository:
- **None found publicly** as of 2026-08-24 (paper states a standalone Python framework integrated with vLLM; no author GitHub URL located via MLSys/OpenReview/web search).

Version/commit:
- N/A

Official license:
- Unknown (code not public)

Published objective:
- Maximize SLO attainment under TTFT and TPOT constraints via iterative state-aware scheduling (request-order + workload control).

Published inputs/features:
- Per-request TTFT/TPOT SLO state; predicted TTFT/TPOT; systemic load; waiting/running prefill vs decode composition.

Published scheduling mechanism:
- Constrained optimization switching between prioritizing prefills vs decodes based on which SLO is less fulfilled; add/retain work until the constrained SLO would be violated.

What we will reproduce exactly:
- Not yet. Spec only: `docs/current/sola_faithful_reimplementation_spec_20260824.md`.

What cannot be reproduced exactly (until code/spec complete):
- Any claim of bit-faithful SOLA; existing `sola_style_state_aware` is explicitly **not** SOLA.

Any adaptation required for our common simulator:
- Deferred until reimplementation spec is authorized for coding.

Why that adaptation is fair:
- N/A this pass.

Information available to method:
- TBD in spec (SLO deadlines, phase, predicted lengths — no oracle future length unless paper does).

Information intentionally NOT available:
- Architecture-level disaggregation, extra GPUs.

Status:
**FAITHFUL_REIMPLEMENTATION** (planned; **blocked** pending spec→code). Existing tree status for `sola_style_state_aware`: **SIMULATOR_PROXY** (out of Pext until replaced).

---

### Method: Learning-to-Rank / vLLM-LTR (M4)

Official source:
- Fu et al., “Efficient LLM Scheduling by Learning to Rank,” NeurIPS 2024 (arXiv:2408.15792).

Official repository:
- https://github.com/hao-ai-lab/vllm-ltr

Version/commit:
- `13bbf6ff3dab661791d41362551b089e5f77c91c` (see `baselines/vllm_ltr/PROVENANCE.md`).

Official license:
- Apache-2.0

Published objective:
- Approximate SJF via learned relative output-length ranking to reduce HOL blocking.

Published inputs/features:
- **Tokenized prompt text** into OPT predictor; ranking scores drive scheduler order.

Published scheduling mechanism:
- Sort waiting/running/swapped by descending `aux_model_score`; continuous batching underneath.

What we will reproduce exactly:
- Official ranking rule + offline scoring path (`baselines/vllm_ltr/adapter/`), when prompt text exists.

What cannot be reproduced exactly:
- Live aux-engine inside this simulator’s request model (no prompt text field).

Any adaptation required for our common simulator:
- Offline precomputed scores from real prompt text only.

Why that adaptation is fair:
- Matches official “score once per arrival, then rank” usage; forbids oracle lengths.

Information available to method:
- Prompt text → score (when available); feasibility state.

Information intentionally NOT available:
- Actual future output length; synthetic invented embeddings.

**Applicability classification (frozen this pass):**

| Workload | Class |
|---|---|
| Joint-240 (synthetic counts only) | **D. LTR_NOT_VALID_ON_THIS_WORKLOAD_REPRESENTATION** |
| Public BurstGPT/Azure parquet corpus (token counts; no prompt text) | **D** |
| Prior WildChat text eval | **A. OFFICIAL_MODEL_DIRECTLY_APPLICABLE** (completed historically; not Pext joint) |

Status:
**OFFICIAL_CODE_ADAPTED** where text exists; **not launchable** on joint-240 / current public corpus under class D.

---

## Architecture-level systems (Related Work only)

DistServe, Llumnix, Mooncake, FastServe, Splitwise: **not** in common Pext matrix this pass.

---

## Pass-2 updates (2026-08-24)

### VTC token-budget remap — fidelity verdict

**Verdict: A — faithful and necessary dimensional adaptation** (disclosed; retain label `official_vtc_joint_token_budget_remap`).

Evidence:
- Official `ReqQueue` / `VTCReqQueue` gate: `new_batch_total_tokens + req.input_len <= self.batch_max_tokens` (token units).
- Simulator `BasePolicy._feasible_on_gpu`: `new_batch = new_count` compared to `gpu.max_batch_tokens` (request-count units), despite `GPUConfig` comment wording.
- Joint scenarios set `max_batch_tokens = max_active_sequences` (count). Feeding that number into official VTC rejects any prompt longer than the count (empirically ANWG=0 / zero admits).
- Override `max(step_token_budget, max_prompt)` restores a feasible **token** budget aligned with the simulator’s per-step token work budget without altering VTC ordering/accounting code.

Not (C)/(D): does not change the fairness counter logic; without it the comparison is invalid against an accidentally broken admission gate.

Joint-240 VTC cells: structurally complete (240/240). Treat as **usable with disclosed remap**, not as silent “native max_batch_tokens” VTC.

### LTR applicability — Pass-2

Official ranker input: **tokenized prompt text** only (`OPTForSequenceClassification` on `input_ids`).  
Joint-240: synthetic length fields only — **no text**.  
Public corpus parquet: `prompt_tokens` / `output_tokens` only; `extra` empty — **no text**.

**Verdict: `LTR_NOT_VALID_FOR_CURRENT_WORKLOAD`**

Recommended honest paths (not ESTF-as-LTR):
- **A+C:** keep LTR on prompt-bearing corpora only (e.g. prior WildChat eval); demote from joint-240 / public_trace_stress_v1 Pext columns; retain as mechanism-specific external comparison where text exists.

### SOLA — Pass-2

**Verdict: `SOLA_FIDELITY_SPEC_INCOMPLETE_FOR_IMPLEMENTATION`**  
See updated §6 in `docs/current/sola_faithful_reimplementation_spec_20260824.md`. No `sola_faithful` code this pass.

### vLLM-style proxy — Pass-2

Joint-240: 240/240 success after metric-serialization fix (scheduling unchanged; failed first attempt archived).  
Implementation remains `vllm_faithful` FCFS continuous batching — not SJF `vllm_style_token_budget`, not native GPU vLLM.

### public_trace_stress_v1

Frozen at **M=16, C=32** after policy-blind M×C calibration (see stress protocol doc).

---

## Pass-3 updates (2026-08-24)

### public_trace_stress_v1

Status: **`FROZEN_PUBLIC_TRACE_STRESS_V1`** (M=16, C=32). VTC + vLLM-style stressed-public both **60/60** integrity-complete.

### LTR — formal common-matrix exclusion

**`LTR_NOT_IN_COMMON_PEXT_MATRIX`**  
Reason: **`INPUT_REPRESENTATION_UNAVAILABLE`** (official OPT ranker requires prompt text; joint/public parquet have counts only).

Separate evidence: **`LTR_EXTERNAL_PROMPT_BEARING_EVIDENCE_REUSABLE`**  
Path: `results/vllm_ltr_first_comparative_evaluation/` (WildChat; 300 prompts × 3 seeds × 10 policies; independent verification PASS; ANWG present). Do **not** merge into joint-240 Pext numerically.

### SOLA — gate C

**`SOLA_NOT_FAITHFULLY_REPRODUCIBLE`** / stop **`SOLA_BLOCKED_BY_SLO_MAPPING`**.  
No `sola_faithful` implementation. Prefer Related Work over a misleading proxy. Keep `sola_style_*` unlabeled as SOLA.

### Final common Pext definition (active)

`Pext_common` = P6 + `vllm_style_continuous_batching` + `official_vtc_joint_token_budget_remap`  
(SOLA and LTR not in common matrix under current frozen workloads.)

---

## Pass-4 updates (2026-08-24) — author scope freeze + Pext analysis

### Scope status: `FROZEN_EXTERNAL_BASELINE_SCOPE_PASS4`

**Pext_common** = P6 + `official_vtc_joint_token_budget_remap` + `vllm_style_continuous_batching`.

| Item | Role |
|---|---|
| official VTC | common external (token-budget remap disclosed) |
| vllm_style_continuous_batching | common external simulator proxy (NOT native vLLM) |
| native vLLM | separate engine-level semantic validation |
| vLLM-LTR WildChat | separate prompt-bearing evidence |
| SOLA | Related Work only (author demotion) |
| WAIT/Sarathi/QLM/DistServe/Llumnix/Mooncake/FastServe/Splitwise/external KV | Related Work only |

### New cells

- Family A VTC: 72/72 COMPLETE
- Family B vLLM-style: 32/32 COMPLETE (ANWG=0 throughout — proxy completes none under Family B configs; Full/Small complementarity retained)
- Family C vLLM-style: 72/72 COMPLETE (KV admission mechanistically relevant)
- public_trace_stress_v1 P6: 360/360 COMPLETE (M=16,C=32)

### Joint Pext headline

SBS unchanged (0.314072); VBS 0.333106→0.333319; headroom 0.019034→0.019247. Core portfolio claim survives. No SBS/VBS(Pext) interpretation until this pass — now complete.

### Adequacy

`ADEQUATE_WITH_LIMITATION`
