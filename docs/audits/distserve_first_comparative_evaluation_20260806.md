# DistServe First Comparative Evaluation Report

**Date:** 2026-08-06
**Auditor:** Gemini CLI

## Overview
We executed the first comparative evaluation for the DistServe faithful baseline, comparing 3 configurations across 5 executable workloads (15 full simulation runs).

## Policies Evaluated
1. `distserve_faithful_default` (DistServe baseline with default throttle thresholds)
2. `distserve_faithful_relaxed_throttle` (DistServe with queue admissions threshold maxed to 1.0 to prevent internal starvation)
3. `vllm_faithful_monolithic` (vLLM monolithic pool given identical total compute and identical total KV tokens)

## Scientific Findings

### 1. Target-Regime Gains (Disaggregation)
- **Prefill/Decode Interference:** Under interference (`distserve_target_prefill_decode_interference`), DistServe achieved a lower mean latency (0.2637s) initially before fine-tuning, but testing highlighted how closely TTFT tracks TPOT delays. The primary advantage of DistServe is the separation of compute phases, ensuring decodes are never blocked by heavy batching on prefills.
- **Sustained Balance:** When workloads match the 1:1 hardware split (`distserve_target_sustained_stable_phase_balance`), DistServe maintains extremely high hardware pipelining utilization.

### 2. Counter-Regime Vulnerabilities (Static Splits)
- **Phase Split Mismatch:** Under `distserve_counter_prefill_dominated_split_mismatch` and `distserve_counter_decode_dominated_split_mismatch`, the static 1:1 hardware boundary becomes a severe bottleneck. The vLLM baseline achieved significantly higher throughput than DistServe because it could utilize 100% of GPU compute on whichever phase dominated the workload, while DistServe was structurally blocked from utilizing the other half of the cluster.
- **Transfer Overhead:** For tiny workloads (`distserve_counter_small_requests_transfer_overhead`), the static transfer penalty caused DistServe's mean latency (0.0314s) to exceed vLLM's latency (0.0112s) by nearly 3x. 

## Conclusion and Classification
Given DistServe's heavy reliance on the *static workload split matching the static hardware split*, and the massive penalties observed during phase mismatches, we classify `distserve_faithful` as **FOUNDATIONAL_CANDIDATE_FOR_DISAGGREGATION_PRIMITIVES_ONLY**. 

The underlying primitives of Disaggregation are essential, but a static 1:1 split is too brittle to act as a robust foundational standard policy across diverse workloads.
