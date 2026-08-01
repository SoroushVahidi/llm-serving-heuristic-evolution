# Documentation Index

This directory contains design documents, milestone reports, and safe-claim
guidance for the LLM Serving Heuristic Evolution project.

## 0. START HERE

For the contextual-compositional heuristic research branch, start with
**[START_HERE_CONTEXTUAL_COMPOSITION.md](START_HERE_CONTEXTUAL_COMPOSITION.md)**.
The authoritative branch-scoped roadmap is
**[contextual_composition_roadmap.md](contextual_composition_roadmap.md)** and
the decision log is
**[contextual_composition_decisions.md](contextual_composition_decisions.md)**.
The branch is intentionally paused after CC1b and before CC2 implementation;
resume from **[RESUME_CONTEXTUAL_COMPOSITION.md](RESUME_CONTEXTUAL_COMPOSITION.md)**
and the pause checkpoint
**[audits/contextual_composition_pause_checkpoint_20260731.md](audits/contextual_composition_pause_checkpoint_20260731.md)**.
The final pause-readiness report is
**[audits/contextual_composition_query7_final_pause_readiness_20260731.md](audits/contextual_composition_query7_final_pause_readiness_20260731.md)**.
These files are canonical for the
`contextual-compositional-heuristics-20260731` branch only; they do not rewrite
the historical phase record below.

For the canonical, current-state documentation set, go to
**[current/README.md](current/README.md)**, which indexes:

- [current/PROJECT_STATUS.md](current/PROJECT_STATUS.md) — authoritative current state
- [current/ARCHITECTURE.md](current/ARCHITECTURE.md) — code architecture
- [current/BASELINES.md](current/BASELINES.md) — exact policy/baseline inventory
- [current/SELECTOR_V2.md](current/SELECTOR_V2.md) — full Selector v2 research narrative
- [current/EXPERIMENTS_AND_RESULTS.md](current/EXPERIMENTS_AND_RESULTS.md) — committed vs. local-only artifacts
- [current/REPRODUCIBILITY.md](current/REPRODUCIBILITY.md) — environment, tests, GPU workflows
- [current/NEXT_STEPS.md](current/NEXT_STEPS.md) — exact next recommended action

Everything below this point is the **full legacy index** (~75 detailed
design/milestone/audit documents, current and historical alike) — useful
once you know which specific topic you need to go deeper on, but not the
place to start cold. The numbered-phase sequence indexed in §11/§14/§15
below paused at Phase 2C on 2026-06-27; §16A indexes the unnumbered
Selector v2 / external-baseline / GPU-runtime-validation track that
continued after that pause and is now the project's actual active work —
`current/PROJECT_STATUS.md` supersedes both as the status authority.

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
- [sarathi_faithful_scheduler_reference.md](sarathi_faithful_scheduler_reference.md) — pinned-commit source provenance for `sarathi_faithful`, the faithful (non-proxy) Sarathi-Serve chunked-prefill scheduler reimplementation
- [distserve_faithful_scheduler_reference.md](distserve_faithful_scheduler_reference.md) — pinned-commit source provenance for `distserve_faithful`, the faithful (non-proxy) DistServe context/decode-stage scheduler reimplementation (online scheduling only; offline parallelism/placement planning excluded)
- [tetriinfer_reference.md](tetriinfer_reference.md) — primary-source provenance and reproducibility determination for `tetriinfer_paper_reimplementation`; no official TetriInfer code/artifact exists (verified live), so this is a paper-description reimplementation, not a `_faithful` pinned-commit baseline — see section 0 for the full determination
- [llumnix_faithful_scheduler_reference.md](llumnix_faithful_scheduler_reference.md) — pinned-commit source provenance for `llumnix_faithful` (OSDI 2024 artifact repo, not the continuously-evolving `AlibabaPAI/llumnix`/`llumnix-project/llumnix`), the faithful cluster-scheduling reimplementation (dispatch, migration-pair selection, LCFS migration-candidate selection, destination admission) built on a new live cross-instance migration primitive
- [vllm_chunked_prefill_faithful_scheduler_reference.md](vllm_chunked_prefill_faithful_scheduler_reference.md) — pinned-commit source provenance for `vllm_chunked_prefill_faithful` (vLLM v0.4.2, commit `c7f2cf2b7f`), the 6th faithful baseline: chunked-prefill admission via a shared `SchedulingBudget`, distinct from `vllm_faithful`'s all-or-nothing v0.1.0 admission
- [vllm_chunked_prefill_faithful_design_audit.md](vllm_chunked_prefill_faithful_design_audit.md) — design-fidelity audit for the chunked-prefill baseline against upstream vLLM v0.4.2
- [vllm_chunked_prefill_faithful_root_cause_analysis.md](vllm_chunked_prefill_faithful_root_cause_analysis.md) — root-cause analysis reconciling this baseline against `runtime_validation_benchmark_pack.md`'s real-hardware fixtures, and its implications for Selector v2 validity (see §16A)
- [external_baseline_integration.md](external_baseline_integration.md) — unified integration audit of all six external baselines (vllm/vllm-chunked-prefill/sarathi/distserve/tetriinfer/llumnix faithful): topology comparability matrix, resource-normalization protocols (A/B/C), evaluation harness, smoke validation, invariants, and the selector-eligibility / architecture-vs-policy-selection analysis

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
- [vllm_real_serving_scaled_comparison.md](vllm_real_serving_scaled_comparison.md) — scaled real-vLLM comparison with our-method selector wired in — **superseded for the selector arm** (the selector arm was confounded by an action-space bug); see the corrected doc below for the current selector-arm result
- [vllm_real_serving_scaled_comparison_corrected.md](vllm_real_serving_scaled_comparison_corrected.md) — **current** corrected selector-arm result, superseding the selector-arm findings of the doc above

---

## 16A. Selector v2, corrected objective, and GPU runtime validation (current — outside the numbered phase sequence)

This is the project's actual active research track as of the most recent commits.
None of it is reachable from the Phase 2C sections above (§11, §14, §15) or from
`research_status.md`/`roadmap.md`, which still describe the project as paused at
the 2026-06-27 checkpoint. This section is the closest thing to a canonical index
for that work until a full documentation consolidation happens.

**Objective correction:**
- [selector_objective_audit.md](selector_objective_audit.md) — identifies and fixes a real bug in the legacy `weighted_goodput` metric (completed-request-only denominator, which can rank a policy serving 20% of arrivals above one serving 100%); introduces `arrival_normalized_weighted_goodput` (ANWG) as the corrected primary objective. `weighted_goodput` is retained (not deleted) as a distinctly-named, intentionally-scoped "conditional quality of completions" metric — see the audit for the exact distinction.

**Selector v2 / Dataset v2 (read in this order for the full current picture):**
- [selector_dataset_v2.md](selector_dataset_v2.md) — Dataset v2 design/schema and quality-gate definitions
- [selector_dataset_v2_validity_after_chunked_prefill_baseline.md](selector_dataset_v2_validity_after_chunked_prefill_baseline.md) — **historical/superseded checkpoint** (see banner in the doc itself); its "not ready" verdict was reversed by the SLO-calibration fix below
- [selector_v2_contention_frontier_search.md](selector_v2_contention_frontier_search.md) — historical intermediate step (root-caused a 300/300-tie result); superseded by the SLO-calibrated search below
- [selector_v2_slo_calibrated_frontier_search.md](selector_v2_slo_calibrated_frontier_search.md) — introduces policy-independent, per-request SLO calibration; reverses the prior "not ready" verdict (0/900 → 16.6%/910 genuinely ANWG-discriminative windows)
- [selector_v2_faithful_baseline_scope_audit.md](selector_v2_faithful_baseline_scope_audit.md) — **most authoritative current Selector v2 status doc**: `SELECTOR_SCOPE_DECISION = OPTION B` — the 8-policy historical-monolithic candidate set (`fifo`, `edf`, `scorpio_style_slo_guard`, `admission_control`, `weighted_shortest_processing`, `estimated_service_time_first`, `best_fit`, `multi_bin_batching`) is the trainable Selector v2 action space; the 3 faithful monolithic external baselines are evaluated separately, never trained on (confirmed genuinely dominated under ANWG across 1,511 window-evaluations, a faithfully-reproduced FCFS-under-overload effect, not a simulator bug)

**Simulator execution-model fix:**
- [decode_prefill_contention_execution_model.md](decode_prefill_contention_execution_model.md) — documents the opt-in `ServiceModel.enable_decode_prefill_contention` shared-budget execution path, and that the pre-existing `decode_first` flag is a dead parameter under the *default* execution model (see the note added to [simulator_design.md](simulator_design.md))

**GPU / real-hardware runtime validation:**
- [gpu_external_validity_audit.md](gpu_external_validity_audit.md) — RTX 5060 Ti real-vLLM-server audit motivating the faithful-baseline validity questions
- [wulver_gpu_validation_handoff.md](wulver_gpu_validation_handoff.md) — Wulver A100 cluster validation handoff
- [wulver_vllm_kv_pressure_results.md](wulver_vllm_kv_pressure_results.md) — A100 vLLM KV-cache-pressure validation results
- [wulver_sarathi_vllm_repeated_validation.md](wulver_sarathi_vllm_repeated_validation.md) — repeated-trial (N=5) Sarathi vs. vLLM real-hardware validation
- [runtime_validation_benchmark_pack.md](runtime_validation_benchmark_pack.md) — the committed, checksummed hardware-target benchmark pack derived from the Wulver jobs above; used as the acceptance target for `vllm_chunked_prefill_faithful` (§6)

**Not yet documented:** the most recent commit (`3406bc0`, the calibrated targeted Dataset v2 pilot implementing the Option B scope) has no corresponding `docs/*.md` yet — its only output is an experiment directory. Its prototype selector beat best-fixed on TRAIN/ID_TEST but lost on VALIDATION/OOD_TEST, so it is not yet a clean result; do not cite it as a finished win until that gap is investigated and written up.

---

## 17. Phase audit trail

`audits/` contains one detailed audit/summary document per research phase (failure-case
mining, metric audits, selector-training audits, etc.), referenced from
[research_status.md](research_status.md) and the phase sections above. Browse the
directory directly for the full per-phase record; not every file is individually
indexed here.

---

## 18. Future phases

See [roadmap.md](roadmap.md).
