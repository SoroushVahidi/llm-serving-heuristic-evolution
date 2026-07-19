# Sarathi vs vLLM Matched Runtime Comparison (Mistral-7B-Instruct-v0.1)

- vLLM source: `/mmfs1/scratch/ikoutis/sv96/vllm_mistral_match_1111706` (job 1111706)
- Sarathi source: `/mmfs1/scratch/ikoutis/sv96/sarathi_mistral_fp16_final_1111723`
- Real-Sarathi per-scenario throughput: not computable from stored data (see module docstring); use `sacct -j <job> --format=Elapsed` for a coarse overall figure.

## Per-scenario winner table

| Scenario | Category | TTFT winner | TPOT winner | E2E winner | Sim agrees (TTFT/E2E) |
|---|---|---|---|---|---|
| mistral_match_long_prompt_moderate_output | long_prompt | sarathi | vllm | vllm | False/True |
| mistral_match_active_decode_plus_arriving_prefill | active_decode_plus_arriving_prefill | vllm | vllm | sarathi | True/False |
| mistral_match_prefill_heavy_burst | prefill_heavy_burst | vllm | vllm | vllm | True/True |
| mistral_match_mixed_prompt_lengths | long_prompt | vllm | vllm | vllm | True/True |
| mistral_match_kv_pressure | long_prompt | vllm | vllm | sarathi | True/False |
| mistral_match_short_context_control | short_context_control | n/a | n/a | vllm | None/True |

## Raw metrics

| Scenario | Real vLLM TTFT | Real Sarathi TTFT | Real vLLM TPOT | Real Sarathi TPOT | Real vLLM E2E | Real Sarathi E2E |
|---|---:|---:|---:|---:|---:|---:|
| mistral_match_long_prompt_moderate_output | 0.5010 | 0.4535 | 0.0133 | 0.0163 | 3.8885 | 4.2853 |
| mistral_match_active_decode_plus_arriving_prefill | 0.1508 | 0.1675 | 0.0113 | 0.0163 | 1.6468 | 0.6121 |
| mistral_match_prefill_heavy_burst | 0.5746 | 0.6287 | 0.0207 | 0.0239 | 1.2151 | 1.3709 |
| mistral_match_mixed_prompt_lengths | 0.2659 | 0.2934 | 0.0143 | 0.0209 | 0.8237 | 1.1624 |
| mistral_match_kv_pressure | 1.0569 | 1.1852 | 0.0173 | 0.0175 | 13.6158 | 12.8521 |
| mistral_match_short_context_control | n/a | 0.0705 | n/a | 0.0156 | 0.0807 | 0.5624 |

## Focus-category classification

- **active_decode_plus_arriving_prefill**: n=1 scenario(s); TTFT winner(s)=['vllm']; E2E winner(s)=['sarathi']
- **prefill_heavy_burst**: n=1 scenario(s); TTFT winner(s)=['vllm']; E2E winner(s)=['vllm']
- **long_prompt**: n=3 scenario(s); TTFT winner(s)=['sarathi', 'vllm', 'vllm']; E2E winner(s)=['vllm', 'vllm', 'sarathi']
- **short_context_control**: n=1 scenario(s); TTFT winner(s)=['n/a']; E2E winner(s)=['vllm']

## Classification

- SARATHI_RUNTIME_VALIDATION = SUCCESS
- SARATHI_ADVANTAGE_REGIMES = ['long_prompt']
- SARATHI_SIMULATOR_MATCH (TTFT winner agreement) = 4/5 scenarios agree
- VLLM_SIMULATOR_MATCH (E2E winner agreement) = 4/6 scenarios agree

Caveat: 'winner' here is real-hardware TTFT/TPOT/E2E on ONE run each (no repeated trials, no variance estimate), and the two servers were configured to be comparable (matched gpu-memory-utilization/max-num-seqs/token-budget-per-step) but are not identical scheduling systems -- see docs/wulver_vllm_kv_pressure_results.md for the full caveats already established for this comparison pair.
