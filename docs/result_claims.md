# Result Claims — What We Can and Cannot Claim

## Safe claims (Phase 1)

These claims are directly supported by the Phase 1 simulator output:

- "In our deterministic iteration-level simulator, policy X achieves lower mean latency
  than policy Y on workload Z."
- "Under Poisson arrivals with rate R and the described GPU configuration, FIFO
  achieves P% SLO violation rate while EDF achieves Q%."
- "The Multi-Bin-style batching baseline reduces mean batch size variance by X%
  compared to FIFO in our simulator."
- "The oracle SRTF achieves the lowest mean latency among all evaluated policies on
  the small debug trace."
- "All implemented baselines are deterministic under fixed seed."

## Safe claims (Phase 2A.1 — metrics and oracle)

- "We optimize **priority-weighted SLO goodput** (internal name: `weighted_goodput`): the
  priority-weighted fraction of requests that complete before their SLO deadline.
  Formally: `Σ(priority_i × 1[completion_time_i ≤ deadline_i]) / Σ(priority_i)`."
- "The metric `priority_weighted_slo_goodput` is an alias for `weighted_goodput` and
  reports the identical value; both appear in experiment output CSVs."
- "We report `mean_ttft` and `p95_ttft` as first-class interactive-serving metrics."
- "We evaluate `oracle_srtf` only as a non-deployable hindsight upper bound using actual output lengths."
- "oracle_srtf is not a selector candidate; it is excluded from all online-policy comparison tables."
- "Any comparison involving oracle_srtf must be labeled 'oracle upper bound' or 'hindsight upper bound'."
- "`weighted_goodput` / `priority_weighted_slo_goodput` is the primary selector and evolution fitness objective."

## Safe claims (Phase 2A.3B — hardened baselines)

- "We add `least_laxity_first`, a deadline-aware LLF baseline that prioritises requests by
  laxity = deadline − now − estimated_service_time. It is an online-deployable policy that
  does not use actual output lengths."
- "We add `estimated_service_time_first`, a prompt-and-prediction-aware SJF proxy that
  approximates shortest job first via estimated prefill + decode cost (α×prompt_tokens +
  β×predicted_output_tokens). It is not a reproduction of PARS, which uses prompt-aware
  learning-to-rank."
- "The deployable baseline set now contains 18 policies; the selector chooses among all 18."
- "oracle_srtf remains excluded from the deployable set and the selector candidate set."

## Safe claims (Phase 1.5)

These claims are additionally supported once `enable_prefill_modeling=True`:

- "With explicit prefill modeling, our Sarathi-style chunked-prefill baseline reduces
  mean TTFT compared to FIFO on the prefill-heavy workload in our simulator."
- "The SplitFuse-style policy achieves higher GPU token-budget utilization than FIFO
  on the decode-heavy workload."
- "Policy X achieves a lower p95 TTFT than policy Y on the mixed-SLO workload."
- "TPOT remains stable under bursty arrivals for the SLO-slack-score policy in our simulator."
- "The Orca-style policy and vLLM-inspired token-budget policy exhibit similar throughput
  on the overloaded-prefill workload in our simulator under identical GPU configurations."

## Unsafe claims

Do NOT make the following claims without additional validation:

| Claim | Why unsafe |
|---|---|
| "Our simulator results match production vLLM latency" | Simulator omits preemption, block-level paging, real GPU throughput, networking |
| "Orca-style policy reproduces Orca OSDI 2022 results" | Our implementation is an independent approximation, not official Orca code |
| "vLLM-style policy reproduces vLLM SOSP 2023 results" | KV paging is approximated as flat token budget; no preemption or recompute |
| "Sarathi-style policy reproduces Sarathi-Serve OSDI 2024 results" | Chunked prefill is approximated at admission time, not intra-step token granularity |
| "SplitFuse-style policy reproduces DeepSpeed-FastGen results" | Token-level splitting requires intra-step control; Phase 1.5 approximates at admission |
| "Multi-Bin Batching is reproduced from [paper]" | Our implementation is an independent approximate adaptation |
| "estimated_service_time_first is PARS" | PARS uses learning-to-rank; this uses token-length estimates only |
| "Policy X is optimal" | NP-hard in general; oracle is greedy, not globally optimal |
| "Results generalize to real datacenter workloads" | Synthetic traces may not match real workload distributions |
| "GPU utilization proxy = real GPU compute utilization" | Our proxy is #active_sequences / max_active_sequences |
| "Throughput results hold under real memory constraints" | Phase 1.5 uses a simplified KV token-budget model |
| "TTFT / TPOT match production serving system measurements" | Phase 1.5 prefill cost is a token-budget approximation, not a real FLOPS model |

## Caveats to include in any publication

1. **Simulator limitations**: Phases 1 and 1.5 use a simplified iteration-level model.
   Phase 1.5 adds a token-budget prefill approximation but omits preemption, real memory
   management, and GPU bandwidth saturation at high batch sizes.

2. **Serving-style baselines are approximations**: The Orca-style, vLLM-inspired,
   Sarathi-style, and SplitFuse-style baselines implement the key scheduling insight
   of each cited system using original code.  They are **not** reproductions of the
   original systems.  Results reflect behavior in our simulator, not the original systems.

3. **Workload realism**: Synthetic workloads use log-normal / Pareto distributions.
   Real production traces (e.g., ShareGPT, Azure LLM traces) may differ significantly.

4. **GPU model**: All GPUs are identical and have no throughput degradation at high
   batch sizes.

5. **Oracle**: The oracle SRTF policy uses future information and cannot be deployed.
   It provides a lower bound on mean latency, not a tight bound.

6. **TTFT under Phase 1**: When `enable_prefill_modeling=False`, TTFT measures queuing
   delay + first decode step only.  It does not reflect real-world prefill latency.

## Safe claims (Phase 2A.4 — selector hardening)

- "We train policy selectors on 30 windows (train) and evaluate on 9 held-out windows (test), all with 18 candidate policies."
- "The Decision Tree and Random Forest selectors each achieve +3.0 pp weighted goodput over the best single fixed policy on the held-out test split."
- "Test trace (`burstgpt_scaled_high_10k.jsonl`) is absent from all train and validation configs."

## Safe claims (Phase 2B.4 — LLM heuristic final evaluation)

Safe phrasing: "We evaluate LLM-generated deterministic heuristics under a calibrated simulator and held-out workload regimes."

- "We select a frozen shortlist of 7 heuristics using only train+validation data, then evaluate once on 3 held-out test regimes."
- "`slo_kv_balance_heuristic` achieves a mean priority-weighted SLO goodput of 0.9595 across the 3 held-out test regimes, compared to 0.8602 for the best fixed baseline (`weighted_shortest_processing`)."
- "The 95% bootstrap CI for the improvement of `slo_kv_balance_heuristic` vs best fixed is [0.00, 0.27] (3 regimes, n=2000 bootstrap replicates). The CI is wide due to the small number of held-out regimes."
- "On the `test_very_overloaded` regime, `slo_kv_balance_heuristic` completed only 1240 of 2119 requests, achieving high goodput by selective request handling. Other heuristics and baselines completed all 2119 requests."
- "6 of 7 shortlisted heuristics achieve lower mean WG than the best fixed baseline on held-out test regimes, indicating the heuristics generalize well to moderate conditions but not extreme overload without selective admission."
- "oracle_srtf achieves mean WG=0.855 on test regimes. It is non-deployable (uses actual output tokens). SRTF is not optimal for priority-weighted SLO goodput; a deployable heuristic that also acts as a selective admission controller can exceed oracle_srtf on this metric."

## Unsafe claims (Phase 2A.4/2B.4)

| Claim | Why unsafe |
|---|---|
| "The LLM scheduler beats production vLLM" | All results are in the calibrated simulator, not production vLLM |
| "slo_kv_balance_heuristic significantly outperforms all baselines" | CI [0.00, 0.27] does not cross zero but is very wide (3 test regimes only) |
| "LLM-generated heuristics consistently outperform baselines" | Only 1/7 shortlisted heuristics outperforms best fixed on held-out test |
| "The selector generalizes to arbitrary workloads" | Evaluated on calibrated synthetic and BurstGPT regimes only |
| "oracle_srtf is a tight upper bound" | SRTF minimizes latency, not priority-weighted SLO goodput; it is not tight on this metric |

## Safe claims (Phase 2B.16 — fresh corrected-objective validation)

- "We validate Phase 2B.15 selectors on 174 fresh windows using entirely new seeds ([12,13,14,15] for diversity, [20,21,22] for heldout) and 21 workloads not seen during training. Selectors are frozen before any fresh window is evaluated."
- "`rf_anwg` achieves arrival-norm WG = 0.9781 on 174 fresh windows, +0.0095 over always-SCORPIO (0.9686). 95% bootstrap CI = [0.0035, 0.0155], excluding zero. Phase 2B.15 gain survives fresh evaluation."
- "`regression_anwg` (per-policy RF regressors, argmax) achieves 0.9856 on fresh windows (+0.0170 vs SCORPIO), CI [0.0127, 0.0213]. It is the strongest deployable selector under arrival-norm WG."
- "`knn_anwg` achieves 0.9818 (+0.0132 vs SCORPIO), CI [0.0076, 0.0186] on fresh validation."
- "93.1% of fresh windows (162/174) are near-ties (policy margin < 0.005). Only 12 windows are meaningful. Selector gains on fresh validation are concentrated in these 12 meaningful windows."
- "All 90 FIFO 'wins' under arrival-norm WG on fresh data are metric artifacts: FIFO 'wins' because SCORPIO CF < 1.0 while FIFO CF = 1.0, with margin < 0.001. No genuine FIFO scheduling advantage exists."
- "On 12 meaningful fresh windows: SCORPIO = 0.852, rf_anwg = 0.8647 (+0.0127), knn_anwg = 0.8656 (+0.0136), regression_anwg = 0.8650 (+0.013)."
- "Under fresh arrival-norm WG, EDF/orca_style/slo_slack_score each achieve 0.9776, outperforming always-SCORPIO (0.9686). SCORPIO ranks 5th among 20 fixed policies on fresh data."
- "always-WSP (0.9648) is statistically below always-SCORPIO on fresh validation: gap = −0.0038, CI [−0.0066, −0.0011] excludes zero. SCORPIO maintains its ANWG advantage over WSP."
- "`rf_anwg` loses to always-SCORPIO on fresh_targeted workloads (rf_anwg = 0.9521 vs SCORPIO = 0.9737, gap = −0.0216). `regression_anwg` succeeds on targeted (1.000). The RF classifier approach fails to route away from SCORPIO in regimes designed to favor non-SCORPIO policies."
- "SCORPIO satisfies CF ≥ 0.99 in only 52.3% of fresh windows. The constrained oracle (CF ≥ 0.99) achieves 0.9863, dominated by EDF/FIFO (CF = 1.0 always)."
- "dt_anwg and dt_anwg_regret are not statistically confirmed vs SCORPIO on fresh validation (CI includes zero). DT variants should not be used as production selectors."

## Safe claims (Phase 2B.15 — corrected objective selector retraining)

- "We retrain policy selectors under `arrival_normalized_wg` (corrected objective) and evaluate under 5 metric variants: `completed_request_quality`, `arrival_normalized_wg`, `cp_wg_t095_l05`, `cp_wg_t099_l05`, `cp_wg_t099_l10`."
- "RF selector trained under arrival-norm WG (`rf_anwg`) achieves arrival-norm WG = 0.9795 on 33 held-out windows, +0.0157 over always-SCORPIO (0.9638). This gain uses features only (no oracle information)."
- "Phase 2B.13 RF, RF-regret, and safe-fallback-SCORPIO selectors collapse to always-SCORPIO on 33 heldout windows under arrival-norm WG (all predict SCORPIO for every window). Phase 2B.15 selectors do not exhibit this collapse."
- "Safe-fallback with WSP as the default policy achieves oracle arrival-norm WG = 0.9848 on 33 heldout windows (+0.0210 vs always-SCORPIO). This is an oracle upper bound: it uses actual per-window rewards to decide when to deviate from WSP."
- "Under arrival-norm WG with near-tie filter ε=0.010, SCORPIO is the best policy in 81/84 meaningful windows (96%) on the full 319-window suite."
- "`scorpio_deadline_only` (laxity pre-filter only, no KV guard or credit budget) achieves arrival-norm WG gap = −0.0017 vs full SCORPIO on 7 targeted discriminative workloads. Conditional quality gap = −0.0120. Promotion recommendation: keep as ablation (CQ gap marginally exceeds 0.010 threshold)."
- "214/319 windows (67%) change best-policy label from conditional WG to arrival-norm WG. After near-tie filtering at ε=0.005, the meaningful label distribution is nearly unchanged: SCORPIO wins 82/97 windows under both metrics."
- "Always-WSP achieves arrival-norm WG = 0.9463 on 33 held-out windows, below always-SCORPIO (0.9638), confirming SCORPIO's arrival-norm WG advantage is maintained on the heldout split."

## Safe claims (Phase 2B.14 — metric audit)

- "The `weighted_goodput` metric computes `Σ(priority_i × slo_met_i) / Σ(priority_i)` over **completed requests only**. Dropped or rejected requests are excluded from both numerator and denominator."
- "We rename `weighted_goodput` as `completed_request_quality` to reflect its semantics accurately."
- "We introduce `arrival_normalized_wg = completion_fraction × completed_request_quality` as the corrected system-level goodput metric."
- "SCORPIO arrival-normalized WG = 0.8885; SCORPIO completed-request quality = 0.9846. The gap (0.096) arises because SCORPIO rejects ~10.1% of arrivals."
- "Under arrival-normalized WG, SCORPIO (0.8885) still dominates WSP (0.8540), SOF (0.8297), and all other policies."
- "Under completion-penalized WG (target=0.95, λ≥0.5), WSP becomes the best policy; SCORPIO is rank 2 or lower."
- "Under completion-penalized WG (target=0.99, λ=1.0), SCORPIO drops to rank 6 or lower; WSP remains rank 1."
- "RF and KNN selectors beat always-SCORPIO under arrival-normalized WG (+0.0059 and +0.0085 respectively)."
- "Phase 2B.10–2B.13 WG comparisons between non-SCORPIO policies (FIFO, EDF, WSP, etc.) remain valid arrival-normalized comparisons because those policies have completion fraction ≈ 0.99."
- "SCORPIO ablation performed on 7 targeted discriminative workloads; results in `results/phase2b14_metric_audit_scorpio_ablation/ablation_gap_analysis.json`."

## Unsafe claims (Phase 2B.14 corrections)

| Old claim | Corrected claim |
|---|---|
| "SCORPIO WG = 0.9846 (best system goodput)" | SCORPIO completed-request quality = 0.9846; arrival-normalized WG = 0.8885 |
| "SCORPIO dominates by 10+ pp over all baselines" | Under arrival-normalized WG, lead over WSP = 0.0345; under completion-penalized metrics, WSP wins |
| "Phase 2B.10–2B.13 WG is arrival-normalized goodput" | It is completed-only conditional quality; reinterpretation required for SCORPIO comparisons |
| "Selector training optimizes arrival-normalized goodput" | Selector training used completed-only labels; valid for non-SCORPIO policies but conflated for SCORPIO-heavy labels |

## Safe claims (Phase 2C.1–2C.3 — real-trace ingestion and causal selector retraining)

- "Phase 2C.1 validates simulator outputs against real Azure 2023 LLM Inference traces (conv and code splits) and BurstGPT traces. All claim comparisons use only simulator-measured quantities."
- "Phase 2C.2 trains a DT selector on causal feat_* features (17 features) with ANWG = reward_* × completion_* labels. Best eval ANWG = 0.8021 (dt_anwg selector, native non-oracle pool)."
- "Phase 2C.2 external-style envelope (per-window max over 7 external-style policies) = 0.8297 on eval. Gap of ~0.028 relative to dt_anwg selector. 62/325 eval windows where envelope > dt_anwg."
- "Phase 2C.3 is a negative finding: adding external-style policies to the training pool did not recover orca_style advantage. orca_style had zero full-pool training labels. external_aware_non_oracle DT is numerically identical to native_non_oracle DT."
- "Best Phase 2C.3 selector ANWG = 0.8063 (native_non_oracle_dt). Delta from Phase 2C.2 = +0.0042 (no real orca recovery; improvement is within noise)."
- "scorpio_style_slo_guard is the best fixed external-style policy overall by mean eval ANWG."
- "orca_style wins on 212/611 labeled windows by pairwise ANWG vs scorpio; it is selectively competitive but not a good always-fixed choice."
- "azure_2023_conv is the primary failure workload: windows with long prompts (>1000 tokens) and mixed tight SLO (fraction_tight_slo in [0.4, 0.7]) are where external policies most outperform the learned selector."
- "is_azure_conv_like (feature-based: is_long_prompt AND is_mixed_tight_slo) identifies 135/611 windows matching this profile, regardless of workload name."
- "The Phase 2C labeled dataset (611 rows) captures pairwise orca-vs-scorpio signal, near-tie rows (304), and failure labels. All labels are derived from simulator ANWG. No live API call was used as ground truth."
- "Gemini API calibration infrastructure is dry-run only. 24 planned calls, estimated worst-case $0.00187. No live API call has been made."
- "See docs/audits/phase2c_project_pause_checkpoint.md for full Phase 2C pause inventory."

## Unsafe claims (Phase 2C)

| Claim | Why Unsafe |
|---|---|
| "Phase 2C.3 outperforms Phase 2C.2 by +0.0042 ANWG" | External-aware DTs are numerically identical to native DTs; the delta is measurement noise |
| "Orca recovery is impossible" | Targeted synthetic training for azure_conv_like windows has not been attempted |
| "rf_anwg generalizes better than dt_anwg" | rf_anwg shows slight gains within the current eval set; CI overlap with dt_anwg |
| "Adding more external policies closes the envelope gap" | Not tested; envelope gap is a structural property of the training distribution |

## Safe claims (Phase 2C — real-vLLM scaled external-admission comparison)

Source: `experiments/real_llm/vllm_scaled_comparison_20260703T203640Z/`,
documented in `docs/vllm_real_serving_scaled_comparison.md`. Real vLLM server
(`Qwen/Qwen2.5-0.5B-Instruct`, port 8001), 3780 requests (7 policies × 540), 3
arrival regimes. No hosted API called.

- "In a real multi-regime vLLM external-admission comparison, FIFO,
  shortest-output-first, and vLLM-direct achieved the highest
  arrival-normalized weighted goodput **as executed** (0.2440 / 0.2432 /
  0.2414), ahead of the selector as executed (0.2274, rank 6/7)."
- "All six fixed baseline policies completed 540/540 requests with zero
  network/timeout failures; their ranking is a valid real-vLLM
  external-admission result."
- "The selector arm is **caveated/confounded**: it dropped 59/540 (10.9%)
  requests, all because it routed to `scorpio_style_slo_guard`, which
  load-sheds requests it judges unmeetable. The selector's conditional quality
  among served requests (0.2553) was the highest of any policy."
- "No selector-vs-baseline arrival-normalized WG difference is statistically
  distinguishable from zero (all six paired bootstrap 95% CIs include zero)."
- "Decision-divergence analysis over the identical plan found 186 requests
  whose real SLO outcome differed between the selector and at least one
  baseline (648 cells compared)."
- "This measures external admission control layered on top of vLLM, not vLLM's
  own internal scheduler."

## Unsafe claims (Phase 2C real-vLLM scaled comparison)

| Claim | Why Unsafe |
|---|---|
| "The selector is intrinsically worse than FIFO/SOF/vLLM-direct" | Its action space includes a load-shedding sub-policy (`scorpio_style_slo_guard`) whose declines the harness recorded as drops; the comparison is not like-for-like until the full action space is executed on equal admission terms or the selector is restricted to a non-shedding executable action set |
| "The selector beats any baseline on real vLLM" | All six selector-minus-baseline bootstrap CIs include zero |
| "This compares our method against vLLM's scheduler" | It does not observe vLLM's internal batching/KV scheduler |
| "The 59 dropped requests are network/harness failures" | They are intentional load-shedding by `scorpio_style_slo_guard`, correctly executed; now labeled `PolicyDeclinedAdmission` |

## How to harden claims for Phase 2+

- Validate against real vLLM / Sarathi / DeepSpeed-FastGen serving traces
- Add GPU-bandwidth-limited decode throughput model
- Add KV cache paging and preemption
- Test on publicly available workload traces (AzureLLMInferenceTrace, ShareGPT, etc.)
- Compare against production serving system baselines where licensing permits
- Use real FLOPS-based prefill cost rather than token-budget proxy
