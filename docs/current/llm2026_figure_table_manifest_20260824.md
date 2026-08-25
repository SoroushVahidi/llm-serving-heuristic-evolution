# LLM 2026 Figure and Table Manifest

Date: 2026-08-24

## Ranked Figures

### Figure 1: Evidence Pipeline

- Purpose: Show the coherent story, not a chronological failure list.
- Content: complementarity -> selector/shared-feature/mechanism target -> router
  -> support expansion -> guarded composition -> typed GP -> real-vLLM semantic
  validation.
- Source data: verdicts from `docs/current/llm2026_claim_evidence_ledger_20260824.md`.
- Status: needs drawing.
- Placement: Introduction or end of Section 2.

### Figure 2: Joint Workload Complementarity

- Purpose: Answer reviewer concern that complementarity is an artifact of three
  disjoint handcrafted families.
- Panels:
  1. winner distribution over six policies;
  2. oracle-gain histogram;
  3. optional pressure-axis winner map.
- Existing files:
  - `experiments/joint_multimechanism_generalization_v1/figures/winner_distribution.png`
  - `experiments/joint_multimechanism_generalization_v1/figures/oracle_gain_histogram.png`
  - `experiments/joint_multimechanism_generalization_v1/figures/winner_map_prefill_kv.png`
- Split/status: CPU-only synthetic joint workload, 240 scenarios.
- Placement: Section 4.

### Figure 3: Detectable Regimes vs Realized Router Gain

- Purpose: Show that online regime detectability is not enough.
- Source:
  - `experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json`
  - `docs/current/WORK_STATUS.md`
- Key numbers: macro-F1 0.9887; live delta 0.00616 ANWG; oracle-gap closure
  0.143.
- Status: needs drawing.
- Placement: Section 5.

### Figure 4: Constructive-Falsification Summary

- Purpose: Compress Wulver support, guarded composition, and typed GP into one
  table-like visual if space is tight.
- Source:
  - `docs/current/family_a_wulver_dev_support_eval_v1_analysis_20260824.md`
  - `docs/current/family_a_mechanism_composite_rule_static_feasibility_v1_analysis_20260824.md`
  - `docs/current/portfolio_guided_typed_gp_screen_v1_train_analysis_20260824.md`
  - `docs/current/random_grammar_best_candidate_audit_v1_20260824.md`
- Placement: Section 5 or appendix.

### Figure 5: Real-vLLM Semantic Validation

- Purpose: Show simulator abstraction -> failed direct native mapping -> native
  budget tradeoff.
- Source:
  - `docs/current/real_vllm_prefill_decode_validation_v1_20260824.md`
  - `docs/current/real_vllm_prefill_decode_fidelity_diagnosis_v1_20260824.md`
  - `docs/current/native_vllm_chunk_budget_semantics_probe_v1_20260824.md`
- Key numbers: T512 1680 steps / 165 mixed / 194 partial; T4096 1536 steps /
  55 mixed / 9 partial; low-late TTFT -30.6 ms and hog E2E +16.3 ms for
  T4096.
- Placement: Section 6.

## Tables

### Table 1: Six-Policy Mechanism Matrix

- Columns: policy, primary mechanism, online observables, strongest expected
  regime, known weakness.
- Source: `experiments/portfolio_policy_synthesis_design_v1/policy_mechanism_matrix.json`
  and policy source files.
- Placement: Section 3.

### Table 2: Workload and Evidence Suites

- Rows: public trace replay, Family A/B/C controlled suites, unified 176x6,
  joint 240, real-vLLM direct validation, native-vLLM budget probe.
- Columns: purpose, size, split/status, main metric, safe claim.
- Placement: Section 3.

### Table 3: Hypothesis / Method / Frozen Gate / Outcome

- Rows: pooled selector, shared feature, mechanism target, hierarchical router,
  Wulver support, guarded composition, typed GP structural crossover, random
  candidate audit.
- Source: claim ledger.
- Placement: Section 5.

### Table 4: Simulator-vs-vLLM Semantic Map

- Rows: step budget, chunk size, decode priority/order, waiting admission,
  queueing, TTFT, KV pressure.
- Source: `docs/current/real_vllm_prefill_decode_fidelity_diagnosis_v1_20260824.md`.
- Placement: Section 6 or appendix.

## Appendix-Only Figures/Tables

- Full Wulver support gate table.
- Typed GP candidate ledger and parent reproduction gates.
- Random candidate branch/module activation and ablation table.
- Full real-vLLM per-regime latency tables.
- Detailed old A/B/C family comparison.

## Immediate Figure Work

Priority order:

1. Convert existing joint workload plots into paper style.
2. Draw Figure 1 evidence pipeline.
3. Draw Figure 5 real-vLLM semantic validation.
4. Build Table 1 policy mechanism matrix.

