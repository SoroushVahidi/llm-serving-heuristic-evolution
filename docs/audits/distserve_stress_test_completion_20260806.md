# DistServe Stress-Test Catalog Completion Report

**Date:** 2026-08-06
**Auditor:** Gemini CLI

## Summary
The DistServe evaluation gap was closed by extending the algorithm stress test catalog with 6 literature-grounded DistServe scenarios (2 TARGET, 4 COUNTER). 

## Entries Added
### TARGET
1. `distserve_target_prefill_decode_interference` (Executable, PASS): Monolithic vLLM system couples prefill and decode, causing prefill blocks to delay decode iterations. DistServe isolates them, protecting TPOT.
2. `distserve_target_sustained_stable_phase_balance` (Executable, PASS): Continuous arrivals balancing naturally across the static 1:1 hardware split.

### COUNTER
1. `distserve_counter_prefill_dominated_split_mismatch` (Executable, PASS): Massive prefills with 2-token decodes. The 1:1 split wastes 50% compute on idle decode GPUs, causing vLLM (which pools all resources) to achieve 2× the throughput of DistServe.
2. `distserve_counter_decode_dominated_split_mismatch` (Executable, PASS): Tiny prefills and massive decodes. Again, static 1:1 split starves decode and idles prefill. 
3. `distserve_counter_small_requests_transfer_overhead` (Executable, PASS): Small requests where KV transfer overhead exceeds the isolation benefit, driving up latency compared to a monolithic vLLM setup.
4. `distserve_counter_low_bandwidth_large_kv` (Spec-only, DISCLOSED): Simulator structural limit (cannot model flat-delay network latency scaling). 

## Headroom Validation
All 5 executable gates were verified rigorously. We correctly enabled `enable_decode_prefill_contention` on the simulator to ensure vLLM accurately models the actual cross-stage monolithic interference the DistServe paper is based on.
