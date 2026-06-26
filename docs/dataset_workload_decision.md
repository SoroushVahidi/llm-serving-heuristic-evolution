# Dataset and Workload Decision Document

**Phase:** 2B.9  
**Date:** 2026-06-25  
**Branch:** `phase2b9-selector-robustness-and-suite-freeze`  
**Purpose:** Decisive, publication-oriented specification of which datasets and workloads to use, add, or cite for the first paper submission.

---

## A. Already Supported / Current

| Workload / Dataset | Type | Status | Notes |
|-------------------|------|--------|-------|
| Calibrated synthetic (Poisson) | Synthetic | ✅ Active | Primary workhorse; all phases 1–2B.8 |
| Calibrated synthetic (Bursty) | Synthetic | ✅ Active | Phase 2A.4 train/test bursty configs |
| Mixed-SLO synthetic | Synthetic | ✅ Active | Phase 2A.4 train + Phase 2B.7/2B.8 |
| Prefill-heavy synthetic | Synthetic | ✅ Active | Phase 2A.4 train/validation |
| High prediction noise synthetic | Synthetic | ✅ Active | Phase 2B.7/2B.8 failure-case analysis |
| KV-pressure / decode-heavy synthetic | Synthetic | ✅ Active | Phase 2B.7/2B.8 failure-case analysis |
| Overloaded + very overloaded synthetic | Synthetic | ✅ Active | Phase 2B.7/2B.8 sweeps |
| BurstGPT-derived / BurstGPT-style replay | Trace-based | ✅ Active | `data/processed/burstgpt/*.jsonl`; used in Phase 2A.4 train/val/test |

**BurstGPT availability:** 5 processed JSONL files exist in `data/processed/burstgpt/`:
- `burstgpt_scaled_moderate_10k.jsonl` — moderate load, 10k requests
- `burstgpt_scaled_high_10k.jsonl` — high load, 10k requests (test-only)
- `burstgpt_natural_10k.jsonl` — natural rate, 10k requests
- `burstgpt_moderate_noise070.jsonl` — moderate with 7% noise
- `burstgpt_moderate_exact_prediction.jsonl` — moderate with exact prediction

---

## B. Must Use Before Final Publication

These datasets are required to make publishable generalization claims about the selector and baselines.

---

### B.1 BurstGPT (Full)

**URL/Name:** BurstGPT: A Real-World Workload Dataset for LLM Serving (2024)  
Available via: `https://github.com/HKUDS/BurstGPT` or HuggingFace.

**HF token needed:** Possibly (check access; public datasets may not require token).  
**License:** Non-commercial research use (verify before submission).  
**Real timestamps:** ✅ Yes (original traces have real inter-arrival times).  
**Prompt lengths:** ✅ Yes.  
**Output lengths:** ✅ Yes.  
**Session/multi-turn:** Partial (some sessions are multi-turn).  
**Long-context support:** Limited (most prompts are short-to-medium).  
**SLO/priority labels:** ❌ Not native; synthesize from arrival time + request class.

**Preprocessing needed:**
1. Download full dataset (currently using 10k-request subset).
2. Convert to `extended_jsonl` format (arrival_time, prompt_tokens, output_tokens, request_id).
3. Assign SLO classes and priorities using rule-based synthesis (matching paper methodology).
4. Scale timestamps if needed (original rate may not match simulator parameters).

**Leakage risks:** Output lengths are known at trace load time. Must use only predicted (noisy) output lengths for scheduling; actual output only for post-hoc metric computation.

**How to use in train/val/test:**
- BurstGPT moderate → selector training (already done in Phase 2A.4)
- BurstGPT high-load → held-out test (already done in Phase 2A.4; must remain test-only)
- BurstGPT additional splits (day/night, high-variance) → add to training

**Purpose:** Arrival patterns, prompt/output lengths, request distributions matching real GPT API traffic.

---

### B.2 Azure LLM Inference Dataset 2023 / Splitwise Trace

**URL/Name:** Azure LLM Inference Dataset 2023 (Patel et al., Splitwise ISCA 2024).  
Available via: `https://github.com/Azure/AzureLLMInferenceDataset` or as supplementary data.

**HF token needed:** No (GitHub/Zenodo).  
**License:** CC-BY 4.0 (verify; may allow research use with attribution).  
**Real timestamps:** ✅ Yes.  
**Prompt lengths:** ✅ Yes.  
**Output lengths:** ✅ Yes.  
**Session/multi-turn:** Partial.  
**Long-context support:** Yes (Azure traces include long-context API calls).  
**SLO/priority labels:** ❌ Not native; synthesize.

**Preprocessing needed:**
1. Download conversation/coding/function-call subsets.
2. Convert to `extended_jsonl` format.
3. Synthesize SLO classes based on request type (e.g., coding = tight, chat = medium).
4. Normalize arrival timestamps to simulator time scale.

**Leakage risks:** Same as BurstGPT (output lengths known; use predictions for scheduling).

**How to use in train/val/test:**
- Conversation and coding subsets → augment selector training
- Function-call subset → separate held-out test workload

**Purpose:** Real enterprise API traffic patterns; supports claims about generalization to production workloads.

---

### B.3 Bailian / Qwen Trace

**URL/Name:** Bailian LLM serving trace (Qwen API, Alibaba Cloud, 2024).  
May be available via: Arxiv paper supplementary or direct request.

**HF token needed:** Unknown; check availability.  
**License:** Research-only; check terms.  
**Real timestamps:** ✅ Expected.  
**Prompt lengths:** ✅ Expected.  
**Output lengths:** ✅ Expected.  
**Long-context support:** Yes (Qwen handles long contexts).  
**SLO/priority labels:** ❌ Synthesize.

**Preprocessing needed:** Convert to `extended_jsonl`; normalize timestamps; synthesize SLO.

**Leakage risks:** Same as above.

**How to use:** Held-out test only (different distribution from Azure/BurstGPT).

**Purpose:** Validates generalization to non-Western, non-OpenAI traffic patterns.

**Note:** If not accessible, treat as optional (upgrade from must-use to strong-optional).

---

### B.4 LMSYS-Chat-1M or WildChat-1M as Prompt/Conversation Pool

**URL/Name:**
- LMSYS-Chat-1M: `lmsys/lmsys-chat-1m` on HuggingFace.
- WildChat-1M: `allenai/WildChat-1M` on HuggingFace.

**HF token needed:** Yes for LMSYS-Chat-1M (gated); WildChat may be less restricted.  
**License:** CC-BY-NC 4.0 for LMSYS; check WildChat terms.  
**Real timestamps:** ✅ Yes (conversation timestamps).  
**Prompt lengths:** ✅ Yes.  
**Output lengths:** ✅ Yes (model responses included).  
**Session/multi-turn:** ✅ Yes — both are multi-turn chat datasets.  
**Long-context support:** Moderate (some long conversations).  
**SLO/priority labels:** ❌ Not native.

**Preprocessing needed:**
1. Download via HF API (requires token for LMSYS).
2. Extract prompt_tokens and output_tokens from conversation turns.
3. Sample inter-arrival times from empirical distribution or combine with BurstGPT timestamps.
4. Synthesize SLO classes.

**Leakage risks:** Must use predicted output lengths for scheduling. Conversation data may include assistant responses (actual outputs); use only user prompts + predicted response lengths at scheduling time.

**How to use in train/val/test:**
- Use as a prompt pool for synthetic workloads with realistic length distributions.
- Do NOT use timestamps directly as arrivals (they are conversation-level, not system-level).
- Sample from the prompt/output length distributions to replace synthetic Poisson workloads.

**Purpose:** Realistic prompt and output length distributions for synthetic workload generation; replaces lognormal length distributions with empirically calibrated ones.

**Phase 2B.9 action:** Metadata-only check (see section on HF token safety). Do not download full dataset.

---

### B.5 LongBench as Long-Context Prompt Corpus

**URL/Name:** LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.  
HuggingFace: `THUDM/LongBench`.

**HF token needed:** No (public).  
**License:** Apache 2.0.  
**Real timestamps:** ❌ No (benchmark, not trace).  
**Prompt lengths:** ✅ Yes (explicitly long; 1k–60k tokens).  
**Output lengths:** ✅ Yes (task-dependent reference answers).  
**Long-context support:** ✅ Primary purpose.  
**SLO/priority labels:** ❌ Not native.

**Preprocessing needed:**
1. Download and extract prompt + reference lengths.
2. Sample arrival times from BurstGPT or synthetic Poisson.
3. Assign SLO classes (documents = loose; summary tasks = medium).
4. Run through simulator as a long-context stress workload.

**Leakage risks:** Reference output lengths are known from the benchmark; use only for evaluation, not scheduling. Selector must use predicted (shorter) output estimates.

**How to use in train/val/test:**
- Create long-context stress workloads (e.g., `longbench_stress`) as a held-out test regime.
- Verifies that selectors and baselines handle very long prefill + short decode correctly.
- Do NOT mix with standard workloads in training; keep as a separate evaluation tier.

**Purpose:** Tests KV-cache stress, chunked-prefill handling, and prefill-decode trade-offs under long-context prompts. Directly relevant to `sarathi_style`, `splitfuse_style`, and the KV-cache-aware baseline (B.3 in external_baseline_decision.md).

---

### B.6 Calibrated Synthetic Workloads Matched to Real Trace Statistics

**Description:** New synthetic workloads where arrival_rate, prompt_mean, prompt_sigma, output_mean, output_sigma are fit to empirical distributions from BurstGPT or Azure traces. This supplements real-trace replay with controllable, reproducible synthetic workloads.

**HF token needed:** No.  
**License:** N/A (synthetic).  
**Real timestamps:** ✅ Derived.  
**Purpose:** Reproducible counterpart to real-trace replay; can be shared without license restrictions.

**How to use:** Add to selector training as additional "calibrated" workloads.

---

## C. Strong Optional

| Dataset/Workload | Why | Notes |
|-----------------|-----|-------|
| Azure 2024 / DynamoLLM trace | Tests claims on more recent Azure traffic | Access TBD |
| WildChat full/gated version | Broader prompt diversity | HF access needed |
| L-Eval or LV-Eval for very long context | Tests 30k–100k token prompts | Very specialized |
| Vidur/LLMServingSim scenarios | Comparison simulation recipe validation | Not a training source |

---

## D. Cite Only / Do Not Rely On

| Dataset/Source | Reason |
|----------------|--------|
| ShareGPT variants (standard) | Widely available but not a real serving trace; use only if needed for compatibility |
| Proprietary/non-public traces | Cannot be included; cite only if published |
| Full external serving-system traces not available to us | Cannot reproduce; cite only |
| HellaSwag / MMLU / SuperGLUE | Evaluation benchmarks, not serving traces |
| Any dataset requiring terms that prohibit research distribution | Cite only; do not download |

---

## E. HF Token Safety Protocol

A HuggingFace token may exist in the environment. The following rules apply:

1. **Do NOT use the token in Phase 2B.9.** No HF downloads in this phase.
2. **Metadata-only checks** (dataset card, column names, schema) are acceptable without downloading data.
3. **Never print, expose, commit, or log the token.** If scripts use `os.environ["HF_TOKEN"]`, ensure the token is not echoed to logs.
4. **Future phases:** When downloading datasets, use `--use-auth-token` flag from a separate interactive session, not from CI or logged scripts.

---

## F. Dataset Expansion Plan for Pre-Submission

| Priority | Dataset | Action | When |
|----------|---------|--------|------|
| Critical | BurstGPT full | Download full dataset; create more train windows | Phase 2B.10 or dedicated data phase |
| Critical | Azure LLM Inference 2023 | Download; preprocess; add to selector train | Phase 2B.10 |
| High | LMSYS-Chat-1M length stats | Metadata download to calibrate synthetic | Phase 2B.10 |
| High | LongBench prompt pool | Download; create long-context stress workloads | Phase 2B.10 |
| Medium | Bailian/Qwen trace | Access check; download if available | Phase 2B.10 or later |
| Low | Azure 2024 / DynamoLLM | Access check | After Phase 2B.10 |

---

## G. Workload Taxonomy for Publication

| Category | Current coverage | Gap |
|----------|-----------------|-----|
| Synthetic calibrated (Poisson, Bursty) | ✅ Phase 1–2B.9 | Need to calibrate to real traces |
| Overloaded + tight SLO | ✅ Phase 2B.7/2B.8/2B.9 | Add more arrival-rate levels |
| High prediction noise | ✅ Phase 2B.7/2B.8/2B.9 | Add 50%, 70%, 90% noise levels |
| KV-pressure / decode-heavy | ✅ Phase 2B.7/2B.9 | Add boundary cases (mean_output ≈ 150–250) |
| Prefill-heavy | ✅ Phase 2B.7/2B.9 (fixed) | Add long-context (LongBench) |
| Bursty + mixed SLO | ✅ Phase 2A.4 + 2B.9 | Verify BurstGPT captures real burstiness |
| Real-trace BurstGPT | ✅ Phase 2A.4 | Need full dataset, not just 10k subset |
| Real-trace Azure/Splitwise | ❌ Missing | Must add (B.2) |
| Long-context stress | ❌ Missing | Must add (B.5) |
| Multi-tenant priority mix | ⚠️ Partial | Add more extreme priority ratios |
| Real-trace Bailian/Qwen | ❌ Missing | Should add (B.3) |
