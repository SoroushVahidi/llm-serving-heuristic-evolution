# LLM 2026 Sections 5-6 Draft Audit

Date: 2026-08-24

Scope: manuscript-only pass. No scientific experiment, simulation, selector,
router, GP, vLLM, GPU, Wulver/Vulver, API, TEST, FINAL, DEV redesign, or
threshold change was launched.

## Files

- Drafted Section 5, `Constructive Falsification Beyond Selection`, in `paper/llm2026/main.tex`.
- Drafted Section 6, `Real-Serving Semantic Validation`, in `paper/llm2026/main.tex`.
- Added `paper/llm2026/figures/vllm_semantic_validation.pdf` and `.png`.
- Added selected bibliography entries to `paper/llm2026/references.bib`.
- Updated `docs/current/llm2026_number_source_of_truth_20260824.md`.
- Updated `docs/current/llm2026_claim_evidence_ledger_20260824.md`.
- Added `docs/current/llm2026_related_work_plan_20260824.md`.

## Literature-Framing Corrections

- The manuscript now frames SBS/VBS and oracle-versus-realized gain as adopted
  from algorithm-selection methodology.
- The phrase `exploitability gap` is used for the residual portfolio opportunity
  left uncaptured by the tested adaptive mechanisms, not as a newly introduced
  general concept.
- Typed GP and hyper-heuristic scheduling are positioned as established
  methodological precedents.
- Autopoiesis is positioned as an important serving-policy synthesis precedent;
  the manuscript does not claim first evolutionary LLM scheduler.

## Section 5 Claims

| Claim | Number/result | Source artifact |
|---|---:|---|
| Target-free support expansion did not move frozen DEV support enough. | 24,314 fingerprints; 156 domains; 1/104 DEV rows closer; p90 0% | `docs/current/family_a_wulver_dev_support_eval_v1_analysis_20260824.md` |
| Guarded semantic composition failed frozen gates. | 3.11% WFS-regret reduction; below no-go bar; mechanism ordering violation | `docs/current/family_a_mechanism_composite_rule_static_feasibility_v1_analysis_20260824.md` |
| Exact-parent typed GP made synthesis test interpretable. | six parent reproduction gates PASS | `docs/current/portfolio_guided_typed_gp_screen_v1_implementation_20260824.md` |
| Structural crossover did not expand the envelope. | 60 candidates/treatment; 4,320 candidate-scenario evals; crossover best MG 0.0 | `docs/current/portfolio_guided_typed_gp_screen_v1_train_analysis_20260824.md` |
| Best random grammar candidate was not freeze-ready. | MG 0.011295, 6 unique wins, concentration/regression failures | `docs/current/random_grammar_best_candidate_audit_v1_20260824.md` |

Safe wording: tested formulations failed frozen gates.

Unsafe wording avoided: all adaptive scheduling, all policy composition, or all
GP synthesis is impossible.

## Section 6 Claims

| Claim | Number/result | Source artifact |
|---|---:|---|
| Local platform was adequate for prefill/decode semantic probes. | vLLM 0.27.1, Qwen2.5-0.5B, RTX 5060 Ti | `docs/current/real_vllm_runtime_probe_v1_20260824.md` |
| Direct Family-B analogue failed despite successful execution. | 40/40 regime-runs; 300/300 requests; max waiting 7; max running 4; max KV 2.84%; 0 preemptions | `docs/current/real_vllm_prefill_decode_validation_v1_20260824.md` |
| Root cause was simulator-vLLM semantic mismatch. | FULL 6 mixed steps/0 partial; CHUNKED 16 mixed/21 partial; prefill/decode ratio 12.99 | `docs/current/real_vllm_prefill_decode_fidelity_diagnosis_v1_20260824.md` |
| Native vLLM token budget produced a reproducible tradeoff. | T512/T4096 trace separation; low-late TTFT -30.6 ms; hog E2E +16.3 ms; verdict STRONG | `docs/current/native_vllm_chunk_budget_semantics_probe_v1_20260824.md` |

Safe wording: native vLLM 0.27.1 on the tested configuration exposes a
scheduler-token-budget tradeoff.

Unsafe wording avoided: direct Family-B reproduction, vLLM generally, chunked
prefill generally, or production-scale KV claims.

## Citations Added

- Gomes and Selman, algorithm portfolios.
- AutoFolio and Hydra for configured algorithm selection / portfolios.
- Whigham for grammar-based GP.
- Branke et al. for GP/hyper-heuristic dispatching-rule evolution.
- FunSearch and Autopoiesis for program search/evolution positioning.
- Mooncake FAST 2025 for modern LLM-serving disaggregation context.

Deferred to Section 7 unless needed in prose: SOLA, FastServe, QLM,
learning-to-rank scheduling, WAIT/Nested WAIT, PARS, Llumnix details, and
QDGP/MAP-Elites scheduling.

## Compile And Page Budget

Command:

```bash
cd paper/llm2026 && tectonic --keep-logs main.tex
```

Result:

- Compile status: success.
- BibTeX status: no unresolved citation errors after removing line breaks inside
  `\cite{...}` commands.
- PDF pages: 13 one-column LNCS pages including references.
- Section 7 placeholder begins before the references, with references beginning
  immediately after the placeholder.
- Remaining budget to 15 pages: about 2 pages for Section 7 plus final
  compression/citation cleanup.

Known non-fatal issues:

- Layout warnings remain, mostly table/long-token overfull or underfull boxes.
- `amsmath` warns that it cannot redefine math accent `\vec`; this is non-fatal.

## TODOs For Next Manuscript Task

- Draft Section 7: `Implications, Limitations, Related Work, and Conclusion`.
- Compress references and earlier prose if Section 7 pushes the manuscript past
  15 LNCS pages.
- Decide whether to keep both main figures or shrink/move the vLLM semantic
  figure depending on final page pressure.
- Replace anonymous metadata only when submission/de-anonymization requirements
  permit.
