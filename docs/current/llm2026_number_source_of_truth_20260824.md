# LLM 2026 Number Source of Truth

Date: 2026-08-24

Use this file as the first stop for manuscript numbers. Values are copied from
local artifacts; if a number is missing here, add it with an artifact path before
using it in the paper.

## Core Metrics

| Metric | Value | Experiment | Artifact path | Definition / notes | Split/status |
|---|---:|---|---|---|---|
| Public windows saturated | 60/60 | Public trace replay | `docs/current/family_a_scientific_evidence_audit_20260823.md` | All annotated public windows tie at ANWG 1.0 | Public descriptive |
| Public trace envelope gain | 0.0 | Public trace replay | same as above | Six-policy envelope gain | Public descriptive |
| Public active p99/capacity | 5/512 | Public trace replay | same as above | Active count p99 versus capacity | Public descriptive |
| Public max KV utilization | 0.003802 | Public trace replay | same as above | Max KV utilization | Public descriptive |
| Unified utility matrix cells | 1,056/1,056 | Unified utility matrix v2 | `experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv`; `docs/current/WORK_STATUS.md` | 176 scenarios x 6 policies | Controlled stress |
| Unified matrix unique-winner rate | 54.0% | Unified utility matrix v2 | `docs/current/WORK_STATUS.md` | Unique winner across six policies | Controlled stress |
| Family A oracle headroom | 0.021742 | Old A/B/C comparison | `experiments/joint_multimechanism_generalization_v1/old_family_comparison.json` | Oracle mean minus best fixed | Controlled stress |
| Family B oracle headroom | 0.049433 | Old A/B/C comparison | same | Oracle mean minus best fixed | Controlled stress |
| Family C oracle headroom | 0.020261 | Old A/B/C comparison | same | Oracle mean minus best fixed | Controlled stress |

## Joint Multi-Mechanism Generalization

| Metric | Value | Artifact path | Definition / notes | Split/status |
|---|---:|---|---|---|
| Scientific verdict | `JOINT_GENERALIZATION_STRONG` | `experiments/joint_multimechanism_generalization_v1/decision.json` | Frozen interpretation category | CPU-only synthetic TRAIN-compatible |
| Scenarios | 240 | `scenario_manifest.csv`; `coverage_summary.json` | Jointly sampled scenarios | Same |
| Policy cells | 1,440 | `run_integrity.json` | 240 scenarios x 6 policies, all success | Same |
| >=2 elevated pressures | 223/240 = 92.9% | `coverage_summary.json` | Mechanism-pressure indicators independent of policy outcomes | Same |
| >=3 elevated pressures | 175/240 | `coverage_summary.json` | Same | Same |
| Winner count LLF | 59 | `winner_summary.json` | Scenario argmax policy | Same |
| Winner count KV constrained | 50 | `winner_summary.json` | Scenario argmax policy | Same |
| Winner count full prefill | 46 | `winner_summary.json` | Scenario argmax policy | Same |
| Winner count ESTF | 45 | `winner_summary.json` | Scenario argmax policy | Same |
| Winner count WFS | 35 | `winner_summary.json` | Scenario argmax policy | Same |
| Winner count chunked small | 5 | `winner_summary.json` | Scenario argmax policy | Same |
| epsilon-0.01 unique-winner fraction | 59.6% | `winner_summary.json` | Unique winner counts / 240 | Same |
| Nontrivial spread fraction | 91.7% | `winner_summary.json` | Policy range >= epsilon 0.01 | Same |
| Mean policy range | 0.111658 ANWG | `winner_summary.json` | Max policy minus min policy per scenario, mean | Same |
| Best fixed policy | `kv_constrained_online` | `oracle_summary.json` | Highest mean ANWG | Same |
| Best fixed mean | 0.314072 ANWG | `oracle_summary.json` | Mean of KV constrained | Same |
| Oracle mean | 0.333106 ANWG | `oracle_summary.json` | Mean per-scenario six-policy max | Same |
| Oracle headroom | 0.019034 ANWG | `oracle_summary.json` | Oracle mean minus best fixed mean | Same |
| Bootstrap 95% CI for headroom | [0.015988, 0.022433] | `robustness_summary.json` | Scenario bootstrap, n=1000 | Same |
| Median oracle gain | 0.010622 | `oracle_summary.json` | Per-scenario oracle minus best fixed policy | Same |
| p90 oracle gain | 0.058040 | `oracle_summary.json` | Same | Same |
| Positive gain fraction | 60.4% | `oracle_summary.json` | Gain > 0 | Same |
| Gain >=0.01 fraction | 51.3% | `oracle_summary.json` | Gain >= epsilon 0.01 | Same |
| Gain from >=2 pressure scenarios | 93.1% | `mixed_mechanism_summary.json` | Share of total positive oracle gain | Same |
| Gain from >=3 pressure scenarios | 63.3% | `mixed_mechanism_summary.json` | Same | Same |
| Top-10% scenario gain share | 40.5% | `oracle_summary.json` | Concentration check | Same |

## Selector / Router / Support No-Go Chain

| Metric | Value | Experiment | Artifact path | Definition / notes | Split/status |
|---|---:|---|---|---|---|
| Pooled selector regret | 0.0463 | Multi-family selector | `docs/current/llm2026_paper_consolidation_plan_20260824.md` | Mean regret under pooled selector | Selector eval |
| Best-fixed comparator regret | 0.0233 | Multi-family selector | same | Comparator in consolidation plan | Selector eval |
| Majority comparator regret | 0.0127 | Multi-family selector | same | Comparator in consolidation plan | Selector eval |
| LOFO held-out A regret | 0.4786 | Multi-family selector | same | 6.2x worse than fixed 0.0767 | LOFO diagnostic |
| Router macro-F1 | 0.9887 | Hierarchical router | `experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json` | Stage/router diagnostic | TEST/live |
| Router live delta vs WFS | 0.00616 ANWG | Hierarchical router live re-eval | `docs/current/WORK_STATUS.md` | Below 0.01 practical bar | Live re-eval |
| Router oracle-gap closure | 0.143 | Hierarchical router live re-eval | `docs/current/WORK_STATUS.md` | Below 0.75 bar | Live re-eval |
| Wulver unique fingerprints | 24,314 | Wulver sweep | `docs/current/family_a_wulver_dev_support_eval_v1_analysis_20260824.md` | Unique target-free candidate states | TRAIN-side |
| Wulver novel TRAIN-side domains | 156 | Wulver sweep | `docs/current/family_a_selector_closure_and_pivot_v1_analysis_20260824.md` | Novel domains after support analysis | TRAIN-side |
| DEV mean NN improvement | 0.315% | Wulver DEV support eval | `docs/current/family_a_wulver_dev_support_eval_v1_analysis_20260824.md` | FEATURE_V1 primary | Frozen DEV support target |
| DEV p90 NN improvement | 0.0% | Wulver DEV support eval | same | FEATURE_V1 primary | Frozen DEV support target |
| DEV rows closer | 1/104 | Wulver DEV support eval | same | Nearest support after expansion | Frozen DEV support target |
| Top-gap overlap | 1/5 | Wulver DEV support eval | same | Predeclared top-gap features | Frozen DEV support target |

## Composition / GP

| Metric | Value | Experiment | Artifact path | Definition / notes | Split/status |
|---|---:|---|---|---|---|
| Guarded best WFS regret reduction | 3.11% | Mechanism composite static feasibility | `docs/current/family_a_mechanism_composite_rule_static_feasibility_v1_analysis_20260824.md` | Below 10% no-go threshold and 30% GO threshold | TRAIN/D1 static |
| Guarded ESTF release rate | 0.006 | Mechanism composite static feasibility | same | Best rule | TRAIN/D1 static |
| GP TRAIN scenarios | 24 | Typed GP screen | `docs/current/portfolio_guided_typed_gp_screen_v1_train_analysis_20260824.md` | 8 each A/B/C | TRAIN-only |
| GP candidate-scenario evaluations | 4,320 | Typed GP screen | same | 60 candidates x 3 treatments x 24 scenarios | TRAIN-only |
| Random grammar best mean MG | 0.011295 | Typed GP screen | same | Treatment A | TRAIN-only |
| Mutation-only best mean MG | 0.002551 | Typed GP screen | same | Treatment B | TRAIN-only |
| Structural crossover best mean MG | 0.0 | Typed GP screen | same | Treatment C | TRAIN-only |
| Random best unique wins | 6 | Random candidate audit | `docs/current/random_grammar_best_candidate_audit_v1_20260824.md` | epsilon 0.005 | TRAIN-only |
| Random best worst group regression | 0.125 | Random candidate audit | same | Freeze gate failure | TRAIN-only |
| Random best top-family MG share | 0.8915 | Random candidate audit | same | Concentration failure | TRAIN-only |

## Real-vLLM

| Metric | Value | Experiment | Artifact path | Definition / notes | Split/status |
|---|---:|---|---|---|---|
| Direct vLLM validation verdict | `PREFILL_REAL_VALIDATION_NO_GO` | Real vLLM prefill/decode | `docs/current/real_vllm_prefill_decode_validation_v1_20260824.md` | Direct simulator analogue | Local real system |
| Direct vLLM regime-runs | 40/40 | Real vLLM prefill/decode | same | Completed measured runs | Local real system |
| Direct vLLM request success | 300/300 | Real vLLM prefill/decode | same | Successful requests | Local real system |
| Max waiting / running | 7 / 4 | Real vLLM prefill/decode | same | Telemetry confirmed queueing | Local real system |
| Max KV usage | 2.84% | Real vLLM prefill/decode | same | Not high-KV-pressure evidence | Local real system |
| Fidelity diagnosis | `SIMULATOR_VLLM_SEMANTICS_MISMATCH` | Real vLLM fidelity diagnosis | `docs/current/real_vllm_prefill_decode_fidelity_diagnosis_v1_20260824.md` | Direct mapping was bundled budget/chunking intervention | Diagnostic |
| Native vLLM budget verdict | `NATIVE_VLLM_BUDGET_EFFECT_STRONG` | Native vLLM budget probe | `docs/current/native_vllm_chunk_budget_semantics_probe_v1_20260824.md` | Chunked enabled in all treatments; budget varied | Local real system |
| T512 scheduled steps | 1,680 | Native vLLM budget probe | same | Scheduler trace | Local real system |
| T512 mixed prefill/decode steps | 165 | Native vLLM budget probe | same | Scheduler trace | Local real system |
| T512 partial prefill items | 194 | Native vLLM budget probe | same | Scheduler trace | Local real system |
| T512 mean prompt tokens/prefill step | 436.1 | Native vLLM budget probe | same | Scheduler trace | Local real system |
| T4096 scheduled steps | 1,536 | Native vLLM budget probe | same | Scheduler trace | Local real system |
| T4096 mixed prefill/decode steps | 55 | Native vLLM budget probe | same | Scheduler trace | Local real system |
| T4096 partial prefill items | 9 | Native vLLM budget probe | same | Scheduler trace | Local real system |
| T4096 mean prompt tokens/prefill step | 1535.9 | Native vLLM budget probe | same | Scheduler trace | Local real system |
| Low-late T4096 late TTFT diff | -30.6 ms | Native vLLM budget probe | same | T4096 - T512; CI [-38.0, -22.8] ms; 5/5 reps | Local real system |
| Low-late T4096 hog E2E diff | +16.3 ms | Native vLLM budget probe | same | T4096 - T512; CI [9.2, 23.5] ms; 5/5 reps | Local real system |

## Manuscript Sections 3-4 Addendum

Numbers added to `paper/llm2026/main.tex` during the official-template
migration and Sections 3-4 drafting pass.

| Number | Manuscript location | Source artifact | Metric definition / exact source | Manuscript rounding |
|---|---|---|---|---|
| 176 scenarios / 1,056 policy cells | Section 3.1 | `experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv`; `docs/current/joint_multimechanism_generalization_v1_analysis_20260824.md` | Controlled A/B/C unified matrix: 176 scenarios x 6 policies | exact |
| 54.0% unique-winner rate | Section 3.1 | `docs/current/llm2026_full_manuscript_architecture_20260824.md` and unified matrix analysis | epsilon-level unique-winner fraction for controlled matrix | one decimal percent |
| Family A headroom 0.021742 | Section 3.1 | `docs/current/joint_multimechanism_generalization_v1_analysis_20260824.md` old A/B/C comparison | Best-fixed WFS vs six-policy oracle on Family A | six decimals |
| Family B headroom 0.049433 | Section 3.1 | same | Best-fixed full prefill vs oracle on Family B | six decimals |
| Family C headroom 0.020261 | Section 3.1 | same | Best-fixed KV-constrained vs oracle on Family C | six decimals |
| 60/60 public windows tied at ANWG 1.0 | Section 3.2 | `docs/current/llm2026_claim_evidence_ledger_20260824.md`; public trace replay artifacts | Frozen public trace replay saturation result | exact |
| Public trace active p99 5/512 | Section 3.2 | same | Active count p99 versus capacity | exact |
| Public trace max KV utilization 0.003802 | Section 3.2 | same | Max KV utilization in replay | six decimals |
| 240 scenarios / 1,440 successful cells | Section 3.3 | `experiments/joint_multimechanism_generalization_v1/run_integrity.json`; `docs/current/joint_multimechanism_generalization_v1_analysis_20260824.md` | 240 scenarios x 6 policies | exact |
| 223/240 = 92.9% with >=2 elevated pressures | Section 3.3 | `experiments/joint_multimechanism_generalization_v1/coverage_summary.json` | Outcome-independent mechanism-pressure audit | one decimal percent |
| 175 scenarios with >=3 elevated pressures | Section 3.3 | `coverage_summary.json` | Outcome-independent mechanism-pressure audit | exact |
| Winner counts 59/50/46/45/35/5 | Section 3.3 / Figure 1 | `experiments/joint_multimechanism_generalization_v1/winner_summary.json` | Per-scenario argmax counts for LLF, KV, full, ESTF, WFS, chunked | exact |
| epsilon-0.01 unique-winner fraction 59.6% | Sections 1 and 3.3 / Figure 1 | `winner_summary.json` | `unique_winner_fraction` = 0.5958333333 | one decimal percent |
| Nontrivial policy spread 91.7% | Section 3.3 | `winner_summary.json` | `nontrivial_policy_spread_fraction` = 0.9166666667 | one decimal percent |
| Mean policy range 0.111658 ANWG | Section 3.3 | `winner_summary.json` | `policy_range_summary.mean` | six decimals |
| Best fixed KV mean 0.314072 | Section 3.3 | `experiments/joint_multimechanism_generalization_v1/oracle_summary.json` | `best_fixed_mean_utility` | six decimals |
| Oracle mean 0.333106 | Section 3.3 | `oracle_summary.json` | `oracle_mean_utility` | six decimals |
| Oracle headroom 0.019034 | Abstract, Sections 1 and 3.3 | `oracle_summary.json` | `oracle_headroom_anwg` | abstract four decimals; body six decimals |
| Bootstrap 95% CI [0.015988, 0.022433] | Section 3.3 | `experiments/joint_multimechanism_generalization_v1/robustness_summary.json` | Bootstrap CI for oracle headroom | six decimals |
| Median gain 0.010622 / p90 0.058040 | Section 3.3 | `oracle_summary.json` | `oracle_gain_summary.median` and `.p90` | six decimals |
| Positive gain 60.4%; gain >=0.01 51.3% | Section 3.3 | `oracle_summary.json` | `positive_fraction` and `epsilon_positive_fraction` | one decimal percent |
| >=2 pressure gain share 93.1%; >=3 63.3% | Section 3.3 / Figure 1 | `experiments/joint_multimechanism_generalization_v1/mixed_mechanism_summary.json`; analysis doc | Share of total oracle gain from mixed-mechanism scenarios | one decimal percent |
| Top-10% gain share 40.5% | Section 3.3 | `oracle_summary.json` | `top10_percent_gain_share` | one decimal percent |
| Pooled selector regret 0.0463 vs best-fixed 0.0233 vs majority 0.0127 | Section 4.1 / Table 2 | `docs/audits/multifamily_contextual_selector_v1_20260817.md` | Pooled Regime B regret comparisons | four decimals |
| LOFO-A regret 0.4786 vs fixed 0.0767 | Section 4.1 | `docs/audits/multifamily_contextual_selector_v1_20260817.md` | Leave-one-family-out A comparison | four decimals |
| Router macro-F1 0.9887 / live delta 0.00616 / closure 0.143 | Sections 1 and 4.2 / Table 2 | `experiments/hierarchical_regime_router_live_reeval_v1/gate_rescoring_v1.json` | Frozen live gate rescoring | macro-F1 four decimals; delta five decimals; closure three decimals |
| Wulver 24,314 unique fingerprints / 156 domains / DEV 1/104 closer / p90 0 | Section 4.4 / Table 2 | `docs/current/family_a_wulver_dev_support_eval_v1_analysis_20260824.md` | Frozen support-evaluation no-go | exact / compact |
| Guarded 3.11% WFS-regret reduction | Section 4.4 / Table 2 | `docs/current/family_a_mechanism_composite_rule_static_feasibility_v1_analysis_20260824.md` | Best guarded static candidate; failed no-go/gate semantics | two decimal percent |
| GP 4,320 candidate-scenario evals / crossover best MG 0.0 | Sections 1 and 4.4 / Table 2 | `docs/current/portfolio_guided_typed_gp_screen_v1_train_analysis_20260824.md` | Equal-budget TRAIN screen | exact |

## Manuscript Sections 5-6 Addendum

Numbers added to `paper/llm2026/main.tex` during the Sections 5-6 drafting pass.

| Number | Manuscript location | Source artifact | Metric definition / exact source | Manuscript rounding |
|---|---|---|---|---|
| 24,314 unique behavioral fingerprints | Section 5.1 | `docs/current/family_a_wulver_dev_support_eval_v1_analysis_20260824.md` | Wulver target-free support expansion fingerprint count | exact |
| 156 novel TRAIN-side domains | Section 5.1 | `docs/current/family_a_selector_closure_and_pivot_v1_analysis_20260824.md`; `docs/current/family_a_wulver_dev_support_eval_v1_analysis_20260824.md` | Novel TRAIN-side domains discovered by target-free expansion | exact |
| 1/104 DEV rows closer | Section 5.1 | `docs/current/family_a_wulver_dev_support_eval_v1_analysis_20260824.md` | FEATURE_V1 frozen DEV nearest-neighbor support movement | exact |
| p90 DEV support improvement 0% | Section 5.1 | same | FEATURE_V1 p90 nearest-neighbor improvement | exact percent |
| Guarded WFS regret reduction 3.11% | Section 5.2 | `docs/current/family_a_mechanism_composite_rule_static_feasibility_v1_analysis_20260824.md` | Best guarded static rule versus WFS; failed no-go and mechanism ordering | two decimals |
| GP parent reproduction all six PASS | Section 5.3 | `docs/current/portfolio_guided_typed_gp_screen_v1_implementation_20260824.md`; `docs/current/portfolio_guided_typed_gp_screen_v1_train_analysis_20260824.md` | Exact-parent reproduction gate before GP screen | qualitative PASS |
| GP TRAIN scenarios 24 | Section 5.3 | `docs/current/portfolio_guided_typed_gp_screen_v1_train_analysis_20260824.md` | Frozen TRAIN-only screening subset | exact |
| GP candidates per treatment 60 | Section 5.3 | same | Equal evaluated-candidate budget for A/B/C treatments | exact |
| GP candidate-scenario evaluations 4,320 | Section 5.3 | same | 3 treatments x 60 candidates x 24 scenarios | exact |
| Random grammar best mean MG 0.011295 | Section 5.3 | same | Treatment A best marginal gain over E6 | six decimals |
| Mutation-only best mean MG 0.002551 | Section 5.3 | same | Treatment B best marginal gain over E6 | six decimals |
| Structural crossover best mean MG 0.0 | Section 5.3 | same | Treatment C best marginal gain over E6 | one decimal |
| Random grammar unique wins 6 | Section 5.3 | `docs/current/random_grammar_best_candidate_audit_v1_20260824.md` | epsilon-level TRAIN unique wins for audited best random candidate | exact |
| vLLM version 0.27.1 / model Qwen2.5-0.5B / RTX 5060 Ti | Section 6.1 | `docs/current/real_vllm_runtime_probe_v1_20260824.md`; `docs/current/real_vllm_prefill_decode_validation_v1_20260824.md` | Local runtime and direct validation setup | exact identifiers |
| Simulator Family-B full/chunked chunk sizes 65536 / 64 and shared budget 512 | Section 6.2 | `docs/current/real_vllm_prefill_decode_fidelity_diagnosis_v1_20260824.md` | Simulator semantic contract | exact |
| Direct vLLM validation 40/40 runs and 300/300 requests | Section 6.2 | `docs/current/real_vllm_prefill_decode_validation_v1_20260824.md` | Completed measured regime-runs and successful requests | exact |
| Direct vLLM max waiting/running 7/4; max KV 2.84%; preemptions 0 | Section 6.2 | same | Direct validation telemetry | exact / two decimal percent |
| Direct validation verdict `PREFILL_REAL_VALIDATION_NO_GO` | Section 6.2 | same | Frozen real-vLLM direct analogue verdict | exact |
| FULL mixed steps 6 / partial chunks 0; CHUNKED mixed steps 16 / partial chunks 21 | Section 6.3 / Figure 2 | `docs/current/real_vllm_prefill_decode_fidelity_diagnosis_v1_20260824.md` | Scheduler trace from semantic diagnosis | exact |
| Prefill 3200 latency 0.0767 s; decode ITL 0.00591 s/token; ratio 12.99 | Section 6.3 | same | Diagnostic microbenchmark | four / five decimals and two-decimal ratio |
| Diagnosis verdict `SIMULATOR_VLLM_SEMANTICS_MISMATCH` | Section 6.3 | same | Frozen semantic diagnosis | exact |
| Native budget probe 20 runs / 150 requests | Section 6.4 | `docs/current/native_vllm_chunk_budget_semantics_probe_v1_20260824.md` | Measured regime-runs and successful requests | exact |
| T512 steps/mixed/partial/mean prompt tokens 1680 / 165 / 194 / 436.1 | Section 6.4 / Figure 2 | same | Native budget scheduler trace | exact / one decimal |
| T4096 steps/mixed/partial/mean prompt tokens 1536 / 55 / 9 / 1535.9 | Section 6.4 / Figure 2 | same | Native budget scheduler trace | exact / one decimal |
| Low-late late TTFT diff -30.6 ms CI [-38.0, -22.8] | Section 6.4 / Figure 2 | same | T4096 - T512; bootstrap CI | one decimal ms |
| High-late late TTFT diff -2.5 ms CI [-7.7, 1.8] | Section 6.4 / Figure 2 | same | T4096 - T512; bootstrap CI | one decimal ms |
| Low-late hog E2E diff +16.3 ms CI [9.2, 23.5] | Section 6.4 / Figure 2 | same | T4096 - T512; bootstrap CI | one decimal ms |
| High-late hog E2E diff +23.3 ms CI [15.6, 31.8] | Section 6.4 / Figure 2 | same | T4096 - T512; bootstrap CI | one decimal ms |
| Native budget verdict `NATIVE_VLLM_BUDGET_EFFECT_STRONG` | Section 6.4 | same | Frozen native-vLLM token-budget verdict | exact |

## Manuscript Section 7 / Final Pass

Section 7 adds interpretation, limitations, related work, and conclusion. It
does not introduce new scientific result numbers beyond those already recorded
above. Final compile page count is tracked in
`docs/current/llm2026_full_manuscript_audit_20260824.md`.

## Final Submission-Preparation Addendum

Added during `llm2026_submission_readiness_audit_20260824`.

| Number | Manuscript location | Source artifact | Metric definition / exact source | Manuscript rounding |
|---|---|---|---|---|
| Abstract length 88 words | Abstract audit | `paper/llm2026/main.tex` word count script | Words between `\begin{abstract}` and `\keywords` | exact |
| Final PDF page count 15 | Submission audit | `pdfinfo paper/llm2026/main.pdf` | LNCS one-column pages including references | exact |
| BurstGPT rows read/retained/dropped 1,429,737 / 1,404,294 / 25,443 | Dataset provenance audit | `data/public_trace_corpus_v1/manifest.json` | Public trace corpus build report | exact |
| Azure 2023 conversation rows 19,366 | Dataset provenance audit | `data/public_trace_corpus_v1/manifest.json` | Rows read and retained for conversation split | exact |
| Azure 2023 code rows 8,819 | Dataset provenance audit | `data/public_trace_corpus_v1/manifest.json` | Rows read and retained for code split | exact |
| Public trace replay windows 60 base windows / 120 scenario records | Dataset provenance audit | `experiments/public_trace_replay_v1/layer2_scenario_manifest.json` | 20 windows per source, faithful and augmented views | exact |
| Public trace replay seed 20260820 / prediction noise sigma 0.3 / slack multiplier 1.0 / window size 200 | Dataset provenance audit | `experiments/public_trace_replay_v1/layer3_provenance.json` | Frozen public trace replay provenance | exact |

## Revision Pass 1 Definition Addendum

Added during `llm2026_revision_pass1_20260824`.

| Number / quantity | Manuscript location | Source artifact | Metric definition / exact source | Manuscript rounding |
|---|---|---|---|---|
| $\epsilon$-unique winner threshold 0.01 ANWG | Section 2.4; Sections 1 and 3 | `experiments/joint_multimechanism_generalization_v1/generator_spec.json`; `experiments/joint_multimechanism_generalization_v1/run_joint_multimechanism_generalization_v1.py` | A policy is epsilon-unique when its ANWG is at least 0.01 above every other policy on the scenario | exact |
| Nontrivial policy spread threshold 0.01 ANWG | Section 3.3 | same | Scenario policy range `max_policy - min_policy >= practical_epsilon` | exact |
| Elevated mechanism-pressure threshold 0.60 | Section 2.3 | `experiments/joint_multimechanism_generalization_v1/run_joint_multimechanism_generalization_v1.py` | Six outcome-independent pressure indicators normalized to [0,1]; elevated if indicator >= 0.60 | exact |
| Selector regret definition | Section 2.4; Section 4.1 | `src/llmserveopt/selector/multifamily_contextual_selector_v1.py` | Mean shortfall from per-scenario six-policy VBS: `max_policy_anwg - selected_policy_anwg` | definition |
| VBS-gap closure definition | Section 2.4; Section 4.2 | `scripts/run_hierarchical_regime_router_live_reeval_v1.py`; `src/llmserveopt/policy_separation/hierarchical_router_evaluation_v1.py` | `(adaptive_mean - SBS_mean) / (VBS_mean - SBS_mean)` | definition |
| Synthesis marginal gain definition | Section 2.4; Section 5.3 | `experiments/portfolio_guided_typed_gp_screen_v1/fitness_contract.json` | `max(candidate_reward, six_policy_envelope) - six_policy_envelope`, averaged over the pre-specified training screen | definition |
| Joint-workload CI procedure | Section 2.3; Section 3.3 | `experiments/joint_multimechanism_generalization_v1/run_joint_multimechanism_generalization_v1.py`; `robustness_summary.json` | 1,000 scenario-bootstrap resamples with replacement from the fixed 240-scenario population; percentile 95% CI | exact |
| Native-vLLM CI procedure | Section 2.3; Section 6.4 | `experiments/real_vllm_mechanism_validation_v1/run_native_vllm_chunk_budget_semantics_probe_v1.py`; `statistical_summary.json` | 4,000 bootstrap resamples over five paired repetition-level treatment differences; percentile 95% CI | exact |
