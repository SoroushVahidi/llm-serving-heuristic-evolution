# Results Index (Paper-Relevant Evidence)

Neutral index of major completed evidence used by the LLM 2026 manuscript.
Statuses below are the **recorded experiment verdicts**, not marketing claims.

For exact numbers, use
`docs/current/llm2026_number_source_of_truth_20260824.md`.
For claim-safe wording, use
`docs/current/llm2026_claim_evidence_ledger_20260824.md`.

| Experiment | Purpose | Canonical artifacts | Status / verdict |
|---|---|---|---|
| Public-trace replay | Sanity-check whether processed public windows discriminate schedulers | `experiments/public_trace_replay_v1/`; ledger / scientific audit docs | Saturated under frozen setup (60/60 windows tie at ANWG 1.0; envelope gain 0.0) |
| Unified utility matrix v2 | Controlled A/B/C six-policy complementarity matrix | `experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv` | Ready matrix; 176×6 cells; positive family headrooms |
| Joint multi-mechanism generalization v1 | Broader jointly varying 240-scenario portfolio test | `experiments/joint_multimechanism_generalization_v1/` (`decision.json`, `oracle_summary.json`, `winner_summary.json`, `coverage_summary.json`, `robustness_summary.json`) | `JOINT_GENERALIZATION_STRONG` |
| Multi-family contextual selector v1 | Pooled contextual selection across families | `experiments/multifamily_contextual_selector_v1/`; `docs/audits/multifamily_contextual_selector_v1_20260817.md` | `MULTIFAMILY_SELECTOR_NO_GO` |
| Shared cross-family features / mechanism-target studies | Feature-schema and mechanism-label rescues | audits under `docs/audits/` (shared feature / mechanism target) | `SHARED_FEATURE_SCHEMA_NO_GO` / `MECHANISM_TARGET_NO_GO` |
| Hierarchical regime router live re-eval | Detectability vs closed-loop utility | `experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json` | High macro-F1; live delta below +0.01 criterion; low VBS-gap closure |
| Family-A Wulver DEV support eval | Target-free support expansion vs frozen DEV support gate | `experiments/family_a_wulver_dev_support_eval_v1/`; analysis `docs/current/family_a_wulver_dev_support_eval_v1_analysis_20260824.md` | Support gate failed (TRAIN novelty without material DEV support movement) |
| Guarded ESTF/WFS composite static feasibility | WFS-safe ESTF-release rule search | `experiments/family_a_mechanism_composite_rule_static_feasibility_v1/` | `MECHANISM_COMPOSITE_STATIC_NO_GO` |
| Portfolio-guided typed GP screen | Equal-budget TRAIN structural synthesis screen | `experiments/portfolio_guided_typed_gp_screen_v1/`; train analysis docs | `SYNTHESIS_NO_GO` (crossover best mean MG 0.0); random candidate not freeze-ready |
| Real-vLLM prefill/decode validation | Direct Family-B analogue on native vLLM | `experiments/real_vllm_mechanism_validation_v1/prefill_decode_local_v1/`; validation analysis doc | `PREFILL_REAL_VALIDATION_NO_GO` |
| Simulator–vLLM fidelity diagnosis | Explain analogue failure | `experiments/real_vllm_mechanism_validation_v1/prefill_decode_fidelity_diagnosis_v1/` | `SIMULATOR_VLLM_SEMANTICS_MISMATCH` |
| Native vLLM token-budget probe | Hold chunking fixed; vary `max_num_batched_tokens` | `experiments/real_vllm_mechanism_validation_v1/native_vllm_chunk_budget_semantics_probe_v1/statistical_summary.json` | `NATIVE_VLLM_BUDGET_EFFECT_STRONG` |

## Manuscript pointers

- Source: `paper/llm2026/main.tex`
- PDF: `paper/llm2026/main.pdf`
- Figures: `paper/llm2026/figures/`
- Figure plot scripts: `paper/llm2026/scripts/`

## Reading order for outsiders

1. `README.md`
2. This file
3. `docs/DATA_RELEASE_POLICY.md` (what data is included vs downloaded)
4. `docs/current/llm2026_claim_evidence_ledger_20260824.md`
5. Frozen JSONs linked above
6. `paper/llm2026/main.pdf` (if the paper package is included in the release)
