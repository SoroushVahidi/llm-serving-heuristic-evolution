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

## How to harden claims for Phase 2+

- Validate against real vLLM / Sarathi / DeepSpeed-FastGen serving traces
- Add GPU-bandwidth-limited decode throughput model
- Add KV cache paging and preemption
- Test on publicly available workload traces (AzureLLMInferenceTrace, ShareGPT, etc.)
- Compare against production serving system baselines where licensing permits
- Use real FLOPS-based prefill cost rather than token-budget proxy
