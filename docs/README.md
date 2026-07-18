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
- [research_status.md](research_status.md) — **canonical current-status doc**, updated per phase
- [experiment_tracking.md](experiment_tracking.md) — how experiment runs/results are tracked and named

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
- [dataset_workload_decision.md](dataset_workload_decision.md) — dataset/workload selection decision record
- [dataset_workload_plan.md](dataset_workload_plan.md) — earlier dataset/workload survey (superseded by the decision doc above)

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

- [baselines.md](baselines.md) — all 20 registered policies + non-deployable oracle, safe/unsafe labels, provenance table
- [external_baseline_decision.md](external_baseline_decision.md) — scope decision on which external systems (vLLM, Sarathi-Serve, DistServe, etc.) get simulator-level proxies vs. cite-only treatment
- [external_baseline_coverage_report.md](external_baseline_coverage_report.md) — earlier survey of external-baseline coverage (superseded in part by the decision doc above)
- [external_baseline_correctness_audit.md](external_baseline_correctness_audit.md) — per-policy fidelity/correctness audit (algorithm fidelity, oracle-leak checks, safe wording) for the "style" baselines
- [vllm_faithful_scheduler_reference.md](vllm_faithful_scheduler_reference.md) — pinned-commit source provenance for `vllm_faithful`, the faithful (non-proxy) vLLM v0.1.0 scheduler/KV-block reimplementation

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
per workload window (Phase 2A). See [planning_specs.md](planning_specs.md) for the
original design, or [selector.md](selector.md) for the current selector implementation
and candidate-policy set.

---

## 9. LLM heuristic DSL specification (local planning doc)

Full report: `results/llm_heuristic_dsl_spec/llm_heuristic_dsl_spec.md`

Design for a two-level JSON DSL for LLM-generated scheduling heuristics with a
recursive verifier (Phase 2B). See [planning_specs.md](planning_specs.md) for the
original design, or [llm_heuristic_dsl.md](llm_heuristic_dsl.md) for the current DSL
schema/verifier/compiler reference (implemented under `src/llmserveopt/heuristics/`).

---

## 10. API-provider setup

- [api_provider_setup.md](api_provider_setup.md) — CloudRift, Cohere, HuggingFace, Mistral setup; credential policy
- [cohere_smoke_test.md](cohere_smoke_test.md) — Cohere API connectivity/latency/TTFT smoke test (`scripts/smoke_test_cohere_api.py`)
- [cohere_api_calibration.md](cohere_api_calibration.md) — Cohere real-LLM latency/TTFT calibration pilot: dry-run/live/resume, hard caps, tmux instructions (`scripts/run_cohere_api_calibration.py`)
- [real_llm_multi_provider_plan.md](real_llm_multi_provider_plan.md) — Multi-provider rollout plan (Cohere done; Gemini/Vertex, Azure OpenAI, Fireworks dry-run/mock skeletons only); shared schema via `src/llmserveopt/real_llm/calibration_common.py`
- [real_llm_cohere_gemini_comparison.md](real_llm_cohere_gemini_comparison.md) — Cohere vs. Gemini pilot comparison: TTFT/latency/cost, RPM-wait artifact caveat, safe/unsafe claims
- [real_llm_v2_workload_proposal.md](real_llm_v2_workload_proposal.md) — Proposed (not run) v2 workload with length-targeted prompts, to fix v1's output-length-scaling gap

---

## 11. Milestones

| Milestone | File | Status |
|---|---|---|
| Phase 1.5 — serving-style baselines | [milestones/phase1_5_frozen.md](milestones/phase1_5_frozen.md) | COMPLETE |
| Phase 1.7A — real trace ingestion | [milestones/phase1_7a_real_traces.md](milestones/phase1_7a_real_traces.md) | COMPLETE |
| Phase 1.7B — GPU calibration | [milestones/phase1_7b_gpu_calibration.md](milestones/phase1_7b_gpu_calibration.md) | COMPLETE |
| Phase 1.7C — calibrated real-trace replay | [milestones/phase1_7c_calibrated_real_trace.md](milestones/phase1_7c_calibrated_real_trace.md) | COMPLETE |
| Phase 2B.2 — offline LLM generation loop | [llm_generation_loop.md](llm_generation_loop.md) | COMPLETE |
| Phase 2A.4/2B.4 — final evaluation hardening | [llm_generation_loop.md](llm_generation_loop.md) | COMPLETE |
| Phase 2B.14 — metric audit (ANWG), SCORPIO ablation | [audits/phase2b14_metric_definition_audit.md](audits/phase2b14_metric_definition_audit.md) | COMPLETE |
| Phase 2B.16 — fresh corrected-objective validation | [audits/phase2b16_fresh_corrected_objective_validation_summary.md](audits/phase2b16_fresh_corrected_objective_validation_summary.md) | COMPLETE |
| Phase 2C.1 — Azure 2023 + BurstGPT real-trace validation | [audits/phase2c1_evaluation_validity_audit.md](audits/phase2c1_evaluation_validity_audit.md) | COMPLETE |
| Phase 2C.2 — causal selector retraining | [audits/phase2c2_causal_selector_retraining.md](audits/phase2c2_causal_selector_retraining.md) | COMPLETE |
| Phase 2C.3 — external-aware analysis (negative finding) | [audits/phase2c3_labeled_dataset_and_api_calibration.md](audits/phase2c3_labeled_dataset_and_api_calibration.md) | COMPLETE |
| **PAUSE CHECKPOINT** | [audits/phase2c_project_pause_checkpoint.md](audits/phase2c_project_pause_checkpoint.md) | **2026-06-27** |

---

## 12. LLM generation loop

- [llm_generation_loop.md](llm_generation_loop.md) — offline LLM heuristic generation, Phase 2B.2 + 2B.3
- [api_provider_setup.md](api_provider_setup.md) — CloudRift, Cohere, Mistral credential policy

Phase 2B.3 adds:
- **Design targets** (7 named emphases): `slo_urgency`, `kv_pressure`, `throughput_oriented`, `prefill_heavy`, `mixed_slo`, `noisy_prediction_robust`, `balanced`
- **Candidate deduplication** by canonical SHA256
- **Multi-regime evaluation** across 4 train + 3 validation synthetic regimes
- **Search ranking** by validation `priority_weighted_slo_goodput` with overfitting detection

Safe wording: "Phase 2B.3 performs offline LLM-based heuristic search. Candidates are generated
by an LLM, verified by the DSL verifier, and evaluated deterministically in the simulator using
priority-weighted SLO goodput."

---

## 13. Safe manuscript claims

- [result_claims.md](result_claims.md) — comprehensive safe/unsafe claim table
- [gpu_validation_claims.md](gpu_validation_claims.md) — GPU-specific claim guidance

Key safe phrasings:
- "We replay real BurstGPT arrival timestamps and token counts."
- "SLOs, priorities, and predicted output lengths are synthetically augmented."
- "The simulator uses service curves calibrated on an RTX 5060 Ti running Qwen2.5-0.5B."
- "Serving-style baselines are original implementations inspired by, not reproductions of, the cited systems."

---

## 14. Historical Phase 2A.4/2B.4 results summary

This section is historical context only. For current canonical Phase 2C status and
safe wording, use:
- [audits/phase2c_project_pause_checkpoint.md](audits/phase2c_project_pause_checkpoint.md)
- [result_claims.md](result_claims.md)
- [audits/phase2c3_labeled_dataset_and_api_calibration.md](audits/phase2c3_labeled_dataset_and_api_calibration.md)

Phase 2A.4 scaled the selector to the then-current 18-policy portfolio (52 windows total). RF and DT selectors achieved +3.0 pp over best fixed on held-out test. Phase 2B.4 froze a 7-heuristic shortlist on train+val and evaluated once on 3 held-out test regimes.

Key results:
- **Selector (RF/DT)**: +3.0 pp over best fixed on selector test split (WG=0.828 vs 0.798)
- **Best LLM heuristic** (`slo_kv_balance_heuristic`): mean WG=0.9595 on final test regimes (+9.9 pp vs best fixed); 95% CI [0.00, 0.27] — exploratory
- **6/7 shortlisted heuristics** regress vs best fixed on hardest test regimes
- **oracle_srtf**: WG=0.855 on test; non-deployable and not optimal for this metric

See [result_claims.md](result_claims.md) for safe/unsafe claim guidance.

## 15. Phase 2C audit docs

- [audits/phase2c1_evaluation_validity_audit.md](audits/phase2c1_evaluation_validity_audit.md) — real-trace replay validity
- [audits/phase2c2_causal_selector_retraining.md](audits/phase2c2_causal_selector_retraining.md) — causal selector retraining results
- [audits/phase2c3_labeled_dataset_and_api_calibration.md](audits/phase2c3_labeled_dataset_and_api_calibration.md) — Phase 2C.3 negative finding, labeled dataset, Gemini dry-run
- [audits/phase2c_project_pause_checkpoint.md](audits/phase2c_project_pause_checkpoint.md) — **pause checkpoint** (resume here)

Key facts for Phase 2C:
- Best learned selector ANWG = 0.8021 (native_non_oracle_dt on 325 eval windows).
- External-style envelope = 0.8297; learned selector does **not** beat it.
- orca_style wins on 212/611 labeled windows vs scorpio but is not a good fixed choice.
- azure_2023_conv is the main failure workload (long-prompt + mixed-SLO regime).
- **No live API call was made.** Gemini calibration is dry-run only.
- Labeled dataset has 611 rows with simulator-derived ANWG labels (no API ground-truth).
- Safe-for-training labels: `label_best_native_non_oracle_policy`, all `is_*` regime flags.
- Unsafe to claim: learned selector beats external envelope; orca recovery; Gemini-validated results.

---

## 16. Real-serving validation (post-pause, outside the numbered phase sequence)

- [real_llm_latency_model_v2.md](real_llm_latency_model_v2.md) — latency model fit from Cohere/Gemini v2 length-targeted pilots
- [real_llm_simulator_integration_plan.md](real_llm_simulator_integration_plan.md) — plan for comparing simulator predictions against real-LLM latency
- [vllm_real_serving_external_baseline_pilot.md](vllm_real_serving_external_baseline_pilot.md) — first real-vLLM-server external-baseline pilot (superseded in part, see below)
- [vllm_real_serving_scaled_comparison.md](vllm_real_serving_scaled_comparison.md) — scaled real-vLLM comparison with our-method selector wired in (current)

## 17. Phase audit trail

`audits/` contains one detailed audit/summary document per research phase (failure-case
mining, metric audits, selector-training audits, etc.), referenced from
[research_status.md](research_status.md) and the phase sections above. Browse the
directory directly for the full per-phase record; not every file is individually
indexed here.

---

## 18. Future phases

See [roadmap.md](roadmap.md).
