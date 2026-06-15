# Planning Specifications Summary

This page summarizes the three local planning specifications produced after Phase 1.7C.
Full reports are in `results/` and are not committed (large, in-progress documents).

---

## 1. Industry-Realistic Simulator Configuration Plan

**Full report:** `results/industry_realism_spec/industry_realism_spec.md`  
**Date:** 2026-06-13  
**Status:** Design only — no experiments modified

### Purpose

Define the 5 canonical industry scenarios that will anchor Phase 2+ evaluations,
mapping each to an existing simulator config and assessing the realism gap.

### Key conclusions

- **5 scenarios selected:** Interactive Chat (A), Code Completion (B), Long-Context
  Document (C), Agentic/RAG (D), Batch/Offline (E).
- Each maps to an existing config (`mixed_slo_comparison.yaml`, `prefill_heavy_comparison.yaml`,
  `overloaded_prefill_comparison.yaml`, `burstgpt_replay_comparison.yaml`,
  `decode_heavy_comparison.yaml`).
- The BurstGPT scaled moderate trace is the best available proxy for Scenario A
  (Interactive Chat) given current simulator fidelity.
- Scenarios D and E require no new configs for baseline evaluation; extensions
  (prefix reuse, preemption) are deferred to Phase 3.

### Implementation impact

- Rename/alias the 5 configs as canonical "industry scenario" configs before Phase 2
  evaluation reports.
- Add `weighted_goodput` metric (Scenario A–E all use it as primary label).
- Do not add new simulator features until Phase 3; use existing approximations with
  explicit caveats.

---

## 2. Selector-over-Known-Scheduling-Algorithms Design

**Full report:** `results/selector_design_spec/selector_design_spec.md`  
**Date:** 2026-06-14  
**Status:** Design only — no source code modified

### Purpose

Design a supervised classifier (the "selector") that, given observable features
of the current workload window, selects the best online scheduling policy from
the 14 registered deployable policies.

### Key conclusions

- **Formulation:** Supervised multi-class classification. Window W=200 requests,
  non-overlapping. Label = name of the online-deployable policy with highest
  weighted goodput on that window.
- **Features:** 12 online-observable features including KV utilization, SLO
  tightness fraction, queue depth, inter-arrival stats, output-length prediction
  bias, and request size distribution.
- **Model v1:** Depth-8 decision tree (scikit-learn). Interpretable, auditable,
  no GPU needed for inference.
- **Train/test split:** Across experiments and regimes — the selector must not
  be trained on the same regime it is tested on.
- **Label source:** Re-simulate each window with each of 14 online policies;
  pick best. No new simulator infrastructure required.
- `oracle_srtf` is excluded from the label space (non-deployable).

### Implementation impact

- Must finalize `weighted_goodput` metric before generating labels.
- Experiment runner needs a `--window-mode` flag or post-hoc windowing from logs.
- First selector experiment uses Phase 1.7C BurstGPT data + 5 synthetic regimes.
- Target: selector beats best-single-policy baseline on held-out regimes.

---

## 3. LLM-Generated Scheduling Heuristic: DSL and Verifier Design

**Full report:** `results/llm_heuristic_dsl_spec/llm_heuristic_dsl_spec.md`  
**Date:** 2026-06-14  
**Status:** Design only — no source code, APIs called, or configs modified

### Purpose

Define the two-level JSON DSL and recursive verifier that will be used in Phase 2B
for LLM-generated scheduling heuristics.

### Key conclusions

- **Recommended representation:** Two-level JSON DSL (request score + batch score).
  Rejected: formula-only, restricted Python, Lisp-like expression trees.
- **Verifier:** JSON schema check + recursive expression-tree checker, O(n) nodes.
  No AST parsing, no sandboxing required for the DSL itself.
- **LLM reliability:** CloudRift/Cohere/Mistral reliably produce valid JSON DSL.
  JSON schema enforces grammar; most LLM errors caught before execution.
- **Repair loop:** On verifier failure, retry with error message in prompt (up to 3×).
  If still failing, fall back to FIFO as the safe default.
- **Fitness:** Weighted goodput on a fixed held-out validation window. Do not
  optimize on the same window used for selector training.
- **Prevented overfitting:** Separate train/val/test windows; generated policies
  cannot observe test-window arrival times.

### Implementation impact

- Requires `weighted_goodput` to be finalized first (shared with selector).
- DSL schema lives in `src/llmserveopt/dsl/schema.json` (to be created).
- Verifier lives in `src/llmserveopt/dsl/verifier.py`.
- Compiler (DSL → Python callable) lives in `src/llmserveopt/dsl/compiler.py`.
- Phase 2B begins after Phase 2A selector baseline is validated.

---

## 4. Baseline and API Audit

**Full report:** `results/baseline_api_audit/baseline_and_api_audit_report.md`  
**Date:** 2026-06-13  
**Status:** Read-only audit — no source modifications

### Purpose

Audit the registered baselines, provider API presence, and repository hygiene
before Phase 1.7C commits.

### Key conclusions

- 14 policies registered and functional (Phase 1.7C commit 0afb014).
- `oracle_srtf` exists in `policies/oracle.py` but is not in the registered
  baseline set — must be added as a non-deployable upper bound candidate.
- `first_fit`, `best_fit`, `earliest_feasible_gpu` exist but are not registered.
- CloudRift/Cohere/Mistral are present as env vars; no calls made in current code.
- No real API keys found in tracked files.

### Implementation impact

- Wire `oracle_srtf` as non-deployable upper bound in Phase 2A configs.
- Optionally register `first_fit`/`best_fit` as additional baselines.
- Audit confirmed the repo is safe to commit — no credential leakage.
