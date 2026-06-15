# Documentation Index

This directory contains design documents, milestone reports, and safe-claim
guidance for the LLM Serving Heuristic Evolution project.

---

## 1. Project overview

See the repository [README.md](../README.md) for a quick-start guide and
motivation summary.

Core design decisions are recorded in:
- [problem_formulation.md](problem_formulation.md) — mathematical problem statement, constraints, objectives
- [simulator_design.md](simulator_design.md) — iteration-level simulator design, timing model, step semantics
- [roadmap.md](roadmap.md) — phase-by-phase research roadmap

---

## 2. Simulator and service model

- [simulator_design.md](simulator_design.md) — deterministic discrete-event simulator
- [calibrated_service_model.md](calibrated_service_model.md) — GPU-calibrated prefill/decode timing model
- [calibration_backend_decision.md](calibration_backend_decision.md) — why HF Transformers, not vLLM, for calibration

---

## 3. Workloads and traces

- [workload_realism.md](workload_realism.md) — synthetic workload realism assessment
- [data_field_provenance.md](data_field_provenance.md) — which fields are real vs. synthetically augmented
- [real_trace_replay.md](real_trace_replay.md) — BurstGPT and ShareGPT replay pipeline

---

## 4. GPU calibration (Phase 1.7B)

- [gpu_calibration.md](gpu_calibration.md) — measurement procedure, curve fitting, MAPE
- [gpu_environment.md](gpu_environment.md) — hardware spec, CUDA version, driver
- [gpu_validation_claims.md](gpu_validation_claims.md) — safe and unsafe claims for GPU calibration results

---

## 5. Real-trace replay (Phase 1.7C)

- [milestones/phase1_7c_calibrated_real_trace.md](milestones/phase1_7c_calibrated_real_trace.md) — full Phase 1.7C results, 7 experiments, noise sensitivity, calibrated vs. synthetic comparison

---

## 6. Baselines

- [baselines.md](baselines.md) — all 14 registered policies + unregistered policies, safe/unsafe labels, provenance table

---

## 7. Industry-realism specification (local planning doc)

Full report: `results/industry_realism_spec/industry_realism_spec.md`

Five canonical industry scenarios (Interactive Chat, Code Completion, Long-Context
Document, Agentic/RAG, Batch/Offline) mapped to existing simulator configs.
See [planning_specs.md](planning_specs.md) for a summary.

---

## 8. Selector-design specification (local planning doc)

Full report: `results/selector_design_spec/selector_design_spec.md`

Design for a supervised classifier that selects the best online scheduling policy
per workload window (Phase 2A). See [planning_specs.md](planning_specs.md).

---

## 9. LLM heuristic DSL specification (local planning doc)

Full report: `results/llm_heuristic_dsl_spec/llm_heuristic_dsl_spec.md`

Design for a two-level JSON DSL for LLM-generated scheduling heuristics with a
recursive verifier (Phase 2B). See [planning_specs.md](planning_specs.md).

---

## 10. API-provider setup

- [api_provider_setup.md](api_provider_setup.md) — CloudRift, Cohere, HuggingFace, Mistral setup; credential policy

---

## 11. Milestones

| Milestone | File | Status |
|---|---|---|
| Phase 1.5 — serving-style baselines | [milestones/phase1_5_frozen.md](milestones/phase1_5_frozen.md) | COMPLETE |
| Phase 1.7A — real trace ingestion | [milestones/phase1_7a_real_traces.md](milestones/phase1_7a_real_traces.md) | COMPLETE |
| Phase 1.7B — GPU calibration | [milestones/phase1_7b_gpu_calibration.md](milestones/phase1_7b_gpu_calibration.md) | COMPLETE |
| Phase 1.7C — calibrated real-trace replay | [milestones/phase1_7c_calibrated_real_trace.md](milestones/phase1_7c_calibrated_real_trace.md) | COMPLETE |

---

## 12. Safe manuscript claims

- [result_claims.md](result_claims.md) — comprehensive safe/unsafe claim table
- [gpu_validation_claims.md](gpu_validation_claims.md) — GPU-specific claim guidance

Key safe phrasings:
- "We replay real BurstGPT arrival timestamps and token counts."
- "SLOs, priorities, and predicted output lengths are synthetically augmented."
- "The simulator uses service curves calibrated on an RTX 5060 Ti running Qwen2.5-0.5B."
- "Serving-style baselines are original implementations inspired by, not reproductions of, the cited systems."

---

## 13. Future phases

See [roadmap.md](roadmap.md). Immediate next step after Phase 1.7C:

**Phase 2A.1 — metric and oracle finalization:**
- Add/verify `weighted_goodput` metric
- Verify TTFT reporting end-to-end
- Wire `oracle_srtf` as non-deployable upper bound in experiment configs
- Optionally register `first_fit`, `best_fit`, LLF as additional baselines

**Phase 2A.2 — selector:**
- Windowed supervised classification over 14 online-deployable policies
- See `results/selector_design_spec/selector_design_spec.md`

**Phase 2B — LLM heuristic DSL:**
- Two-level JSON DSL with recursive verifier
- See `results/llm_heuristic_dsl_spec/llm_heuristic_dsl_spec.md`
