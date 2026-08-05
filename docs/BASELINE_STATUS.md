# Baseline Status Index

Single cross-baseline status table for this project's external and
serving-system-inspired baselines. Generated/maintained as part of the
2026-08-04 repository organization pass
(`docs/audits/branch_and_pars_readiness_audit_20260804.md`); update this
file whenever a baseline's status changes rather than letting status drift
across multiple docs.

**Naming note:** two unrelated papers have both informally been called
"PARS" in this project's history. **PARS-2023** = Zheng et al., NeurIPS
2023, "Response Length Perception and Sequence Scheduling" (approximated
internally by `estimated_service_time_first`, never given official-code
integration; see `docs/external_baseline_coverage_report.md` §15).
**PARS-Serve-2026** = Tao et al., ISC High Performance 2026, "Ranking
Before Serving: Low-Latency LLM Serving via Pairwise Learning-to-Rank"
(official repo `SPEAR-UIC/PARS`; see `baselines/pars/`). Code identifiers
(`baselines/pars/`, `pars_semantic_reference`) are unchanged — only prose
uses the disambiguated names.

| Baseline | Paper | Venue | Year | Official repo | Pinned commit | License | Implementation status | Fidelity class | Checkpoint status | Evaluation status | Benchmark status | Foundational-library recommendation | Exact next action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Current vLLM** (as a framework) | Kwon et al. | SOSP | 2023 | `vllm-project/vllm` | Not integrated | Apache-2.0 | Not integrated as a runnable engine — this project's simulator is a discrete-event abstraction, not vLLM itself | N/A | N/A | N/A | N/A | N/A | N/A |
| **vLLM-LTR** | Fu et al., "Efficient LLM Scheduling by Learning to Rank" | NeurIPS (main conference) | 2024 | `hao-ai-lab/vllm-ltr` | `13bbf6ff3dab661791d41362551b089e5f77c91c` | Apache-2.0 | Complete | Official checkpoint execution (real, hash-verified, architecturally verified pretrained checkpoint) | Downloaded, hash-verified, architecturally verified (`LLM-ltr/OPT-Predictors`) | Complete, independently verified — WildChat control only | Not run on the canonical suite (predates it) | EVALUATION_ONLY | None — evaluation complete for the tested regime; a higher-contention regime (canonical suite) would be the natural next comparison if this baseline is revisited |
| **PARS-2023** | Zheng et al., "Response Length Perception and Sequence Scheduling" | NeurIPS | 2023 | `zhengzangw/Sequence-Scheduling` (exists, real — found during PARS-Serve-2026 research, never integrated in this repo) | Not pinned (never integrated) | Unverified in this pass | Approximated only, not integrated (`estimated_service_time_first` is "style/inspired," explicitly NOT a reproduction) | Proxy/inspired | N/A | Internal proxy evaluated regularly as `estimated_service_time_first` | Yes, part of the standard comparison set including the canonical suite | Already foundational (as the proxy, not the official model) | None planned — the official repo exists but integrating it has not been prioritized over PARS-Serve-2026 |
| **PARS-Serve-2026** | Tao et al., "Ranking Before Serving: Low-Latency LLM Serving via Pairwise Learning-to-Rank" (v1 title: "Prompt-Aware Scheduling for Low-Latency LLM Serving") | ISC High Performance | 2026 | `SPEAR-UIC/PARS` | `fd4e125b65bb73aef5eccafa79c2509434be61ec` | **None** (no upstream LICENSE file — disclosed, not hidden; see `baselines/pars/PROVENANCE.md`) | Complete | Official-code reproduction with locally trained checkpoint (no official checkpoint is released) | Trained, hash-verified (`d54be087...c33eb27`), fidelity-verified (10/10 `tests/test_pars_checkpoint_fidelity_gpu.py`, `best_val_accuracy=0.9141`) | Complete, independently verified — WildChat control + all 7 accepted canonical-suite families (8 workloads total) | Run on the full canonical suite (first baseline on this branch to be) | **EVALUATION_ONLY** | None — evaluation complete; zero unique wins across 8 families, best rank 5th/10, dominated by `shortest_output_first`/`estimated_service_time_first` and by `scorpio_style_slo_guard`/`regression_anwg_selector` in every discriminative regime; see `docs/audits/pars_first_comparative_evaluation_20260804.md` |
| **Sarathi-Serve** | Agrawal et al., "Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" | OSDI | 2024 | Not integrated (official code) | N/A | N/A | `sarathi_style` implemented, style/inspired only | Proxy/inspired, explicitly NOT an official reproduction | N/A | Part of the standard internal policy comparisons | Yes | Foundational (internal policy, always available in the standard set) | None planned |
| **DistServe** | Zhong et al. | OSDI | 2024 | Not integrated | N/A | N/A | Reference doc only (`docs/distserve_faithful_scheduler_reference.md`); no policy implemented | Not applicable | N/A | Not evaluated as an external baseline | N/A | Not applicable | Not prioritized |
| **Llumnix** | — | — | — | Not integrated | N/A | N/A | Reference doc only (`docs/llumnix_faithful_scheduler_reference.md`) | Unverified in this pass | N/A | Unverified in this pass | N/A | Unverified in this pass | Not prioritized |
| **SLAI/RAD** | — | — | — | Not integrated | N/A | N/A | Reference doc only (`docs/slai_faithful_scheduler_reference.md`) | Unverified in this pass | N/A | Unverified in this pass | N/A | Unverified in this pass | Not prioritized |
| **VTC** | Sheng et al., "Fairness in Serving Large Language Models" | OSDI | 2024 | `Ying1123/VTC-artifact` | `192c2e2014c69c8c6c699d7113c3822e4db632e6` | Apache-2.0 | Complete (fairness-validated sweep) | Official policy reused with simulator adapter (real, unmodified `VTCReqQueue` dynamically imported and executed; GPU serving-engine layer not run — see hardware blocker in `baselines/vtc/PROVENANCE.md`) | N/A (no checkpoint; VTC is a pure algorithmic scheduler) | Headroom-gated comparative sweep complete on 6 repaired fairness-extension workloads x 3 seeds x 6 policies (108 runs), independently re-verified with zero mismatches; fidelity + micro-trace + headroom tests 45/45 pass | Not run on WildChat control or the canonical suite (incompatible — no tenant semantics); dedicated fairness-extension workloads only | **EVALUATION_ONLY** | **FOUNDATIONAL_CANDIDATE** (scientific classification; not registered) — VTC wins/ties the checkpoint Jain's-index comparison in 17/18 family x seed combinations (13 outright wins), isolated to be an ordering effect (not the admission-gate confound found in the original smoke test — see `docs/audits/vtc_fairness_benchmark_repair_20260805.md`), with a real, bounded ANWG trade-off (0.680 vs. SCORPIO's 0.984) in the one family designed to expose its SLO-blindness; see `docs/audits/vtc_fairness_comparative_evaluation_20260805.md` for the full decision record and next action (native, non-wrapped reimplementation before any foundational-library registration) |
| **JITServe** | — | — | — | Not integrated | N/A | N/A | Not implemented | N/A | N/A | N/A | N/A | N/A | Not prioritized |
| **Apt-Serve** | — | — | — | Not integrated | N/A | N/A | Not implemented | N/A | N/A | N/A | N/A | N/A | Not prioritized |
| **HyGen** | — | — | — | Not integrated | N/A | N/A | Not found anywhere in this repo (no matches for "HyGen" in `docs/*.md` or `src/llmserveopt/policies/*.py`) | N/A | N/A | N/A | N/A | N/A | Not started, not researched in this pass |
| **ATHENA-Serve** | — | — | — | Not integrated | N/A | N/A | Not found anywhere in this repo | N/A | N/A | N/A | N/A | N/A | Not started, not researched in this pass |

## How to update this table

When a baseline's status changes (training finishes, an evaluation
completes, a classification is decided), update the corresponding row
here **and** the more detailed narrative doc it links to
(`docs/audits/*_baseline_*.md` / `docs/baselines.md`'s per-baseline
section) — this table is a summary index, not a replacement for the full
provenance record each baseline's own audit doc carries.

See also: [`docs/COMPUTE_POLICY.md`](COMPUTE_POLICY.md) (where to run
what), [`docs/INDEX.md`](INDEX.md) (full documentation map),
[`docs/baselines.md`](baselines.md) (per-baseline narrative detail),
[`docs/external_baseline_decision.md`](external_baseline_decision.md)
(original selection rationale for each baseline category).
