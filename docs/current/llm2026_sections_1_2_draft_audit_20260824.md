# LLM 2026 Sections 1-2 Draft Audit

Date: 2026-08-24

Scope: manuscript writing only. Science is frozen. This task edited the
manuscript scaffold and drafted only the abstract, Section 1, and Section 2.
No experiments, selector/router training, GP search, vLLM/GPU work,
Wulver/Vulver jobs, TEST/FINAL use, DEV redesign, threshold changes, push, or
destructive git operation were performed.

## Source Documents Used

- `docs/current/llm2026_full_manuscript_architecture_20260824.md`
- `docs/current/llm2026_paper_consolidation_plan_20260824.md`
- `docs/current/llm2026_claim_evidence_ledger_20260824.md`
- `docs/current/llm2026_number_source_of_truth_20260824.md`
- `docs/current/joint_multimechanism_generalization_v1_analysis_20260824.md`
- `docs/current/real_vllm_prefill_decode_validation_v1_20260824.md`
- `docs/current/real_vllm_prefill_decode_fidelity_diagnosis_v1_20260824.md`
- `docs/current/native_vllm_chunk_budget_semantics_probe_v1_20260824.md`
- `experiments/portfolio_policy_synthesis_design_v1/policy_mechanism_matrix.json`
- `src/llmserveopt/core/metrics.py`

## LaTeX / Template Status

- Updated `paper/llm2026/main.tex` to a compact two-column draft scaffold.
- No official Springer/LLM 2026 class file is present locally.
- The scaffold records a TODO to migrate to the exact Springer Nature / LLM 2026
  template before submission if the portal provides one.
- Removed appendix sections from the main scaffold because the verified venue
  page count includes figures, tables, and references and no free appendix
  allowance was found.
- Retained bibliography hooks:
  - `\bibliographystyle{plain}`
  - `\bibliography{references}`

## Drafted Content

- Title: `The Exploitability Gap in LLM-Serving Scheduler Portfolios`
- Abstract word count: 90 words
- Section 1 word count: 508 words
- Section 2 word count: 847 words
- Sections 3-7 are present only as TODO placeholders under the frozen
  seven-section architecture.

## Claims Used

| Manuscript location | Claim | Safety check |
|---|---|---|
| Abstract | The joint workload retains 0.0190 ANWG oracle headroom over the best fixed policy | Supported by `oracle_summary.json`; stated as oracle headroom, not deployable gain. |
| Abstract | Tested adaptive mechanisms fail frozen gates despite detectable regime structure | Supported by claim ledger; scoped to tested mechanisms. |
| Abstract | Native vLLM semantics do not directly match the simulator abstraction but expose a token-budget tradeoff | Supported by real-vLLM validation, fidelity diagnosis, and native budget probe. |
| Section 1 | Heterogeneous serving dimensions stress different mechanisms | Methodological framing; no universal production claim. |
| Section 1 | Exploitability gap separates oracle complementarity from realized adaptive utility | Central thesis; not an impossibility claim. |
| Section 1 | Router macro-F1 0.9887 but live gain below frozen threshold | Supported by `gate_rescoring_v1.json`; stated as one tested router result. |
| Section 1 | Structural crossover best mean MG 0.0 | Supported by typed-GP train analysis; scoped to equal TRAIN-only budget. |
| Section 2 | Public trace replay saturated under frozen setup | Supported by public trace artifacts; used only as motivation for stress workloads. |
| Section 2 | Real-vLLM setup uses Qwen2.5-0.5B, vLLM 0.27.1, RTX 5060 Ti | Supported by real-vLLM reports; results deferred to Section 6. |

## Number Source Check

| Number | Manuscript location | Source artifact | Metric definition |
|---:|---|---|---|
| 0.0190 ANWG | Abstract | `experiments/joint_multimechanism_generalization_v1/oracle_summary.json` | Oracle mean minus best fixed mean, rounded from 0.019033834. |
| 59.6% | Section 1 | `winner_summary.json` | Epsilon-0.01 unique-winner fraction. |
| 0.019034 | Section 1 | `oracle_summary.json` | Exact joint oracle headroom used in prose. |
| 0.9887 | Section 1 | `experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json` | Router macro-F1. |
| +0.00616 ANWG | Section 1 | `gate_rescoring_v1.json`; number source file | Live hierarchical-router mean ANWG delta below 0.01 threshold. |
| 0.0 | Section 1 | `portfolio_guided_typed_gp_screen_v1_train_analysis_20260824.md` | Best mean marginal gain for structural crossover treatment. |
| 60/60 | Section 2 | `docs/current/family_a_scientific_evidence_audit_20260823.md`; number source file | Public windows tying at ANWG 1.0. |
| 1.0 | Section 2 | Same | Saturated public-window ANWG. |
| 0.0 | Section 2 | Same | Public replay envelope gain. |
| 5/512 | Section 2 | Same | Active-count p99 versus capacity. |
| 0.003802 | Section 2 | Same | Max KV utilization. |
| 240 | Section 2 | `coverage_summary.json`; `scenario_manifest.csv` | Joint workload scenario count. |
| 223/240, 92.9% | Section 2 | `coverage_summary.json` | Scenarios with at least two elevated mechanism pressures. |
| 175 | Section 2 | `coverage_summary.json` | Scenarios with at least three elevated mechanism pressures. |
| vLLM 0.27.1 | Section 2 | `real_vllm_prefill_decode_validation_v1_20260824.md` | Real-serving software version. |

## Metric Definition Check

Section 2 defines ANWG from `src/llmserveopt/core/metrics.py`:

```text
sum(weight_i * 1[completed_i] * 1[completion_time_i <= deadline_i])
/
sum(weight_i over all arrivals)
```

This preserves the source-code distinction between corrected
arrival-normalized weighted goodput and the older conditional weighted-goodput
metric over completed requests only.

## Citation TODOs

No citation keys were inserted yet because `paper/llm2026/references.bib` still
contains only a placeholder. Statements that need verified citations in the next
bibliography pass:

- vLLM / continuous batching / paged attention
- chunked prefill and prefill/decode interference
- KV-cache management and serving admission control
- SLO-aware and fair scheduling
- adaptive scheduling / algorithm portfolios
- hyper-heuristics, GP, grammar-guided synthesis, and MAP-Elites/QD if retained

## Page Budget Status

- Compiled PDF page count: 3 pages.
- Abstract + Sections 1-2 plus table and TODO placeholders consume about 3
  two-column pages.
- The content is slightly over the architecture's ideal early allocation because
  the six-policy table spans both columns and later-section TODO placeholders
  occupy space.
- Practical status: acceptable for this drafting checkpoint; after Sections 3-4
  are drafted, compress Section 2 and/or move the six-policy table to a smaller
  format if the full manuscript exceeds 8 pages.

## Compile Status

Command:

```bash
cd paper/llm2026
tectonic -X compile main.tex
```

Result: build succeeded and wrote `paper/llm2026/main.pdf`.

Warnings:

- Underfull hbox warnings remain, mostly from narrow two-column prose and TODOs.
- BibTeX warning remains because `references.bib` has no real entries yet.
- No overfull table warnings remained after replacing long monospace policy names
  with compact human-readable labels.

## Unresolved TODOs

- Migrate to exact official Springer/LLM 2026 class/template if supplied by the
  submission portal.
- Add verified bibliography entries.
- Draft Sections 3-4 next:
  - `Complementarity Beyond Handcrafted Families`
  - `The Exploitability Gap`
- Build Figure 1 and the compact frozen-gate ledger table.

