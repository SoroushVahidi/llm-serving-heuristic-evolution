# Repeated-Trial Sarathi vs vLLM Statistical Summary (Mistral-7B-Instruct-v0.1)

- Sarathi trials found: [0, 1, 2, 3, 4] (n=5)
- vLLM trials found: [0, 1, 2, 3, 4] (n=5)
- Matched trial indices used for paired analysis: [0, 1, 2, 3, 4]

## E2E robustness classification per scenario

| Scenario | N trials | Sarathi E2E wins | vLLM E2E wins | Mean diff (vLLM-Sarathi) | 95% CI | CI excludes 0 | Robustness |
|---|---:|---:|---:|---:|---|---|---|
| mistral_match_long_prompt_moderate_output | 5 | 0 | 5 | -0.2555 | [-0.2979, -0.2131] | True | **NOT_REPRODUCED** |
| mistral_match_active_decode_plus_arriving_prefill | 5 | 5 | 0 | 1.0172 | [0.9899, 1.0356] | True | **ROBUST** |
| mistral_match_prefill_heavy_burst | 5 | 0 | 5 | -0.1466 | [-0.1574, -0.1372] | True | **NOT_REPRODUCED** |
| mistral_match_mixed_prompt_lengths | 5 | 0 | 5 | -0.2052 | [-0.2574, -0.1608] | True | **NOT_REPRODUCED** |
| mistral_match_kv_pressure | 5 | 5 | 0 | 0.8360 | [0.7693, 0.9028] | True | **ROBUST** |

Classification rule (stated explicitly, not a significance test): ROBUST requires Sarathi winning E2E in >=80% of trials AND the bootstrap 95% CI for the mean vLLM-minus-Sarathi E2E difference excluding zero in Sarathi's favor. SUGGESTIVE requires >=60% win rate in Sarathi's favor without a CI that excludes zero. Otherwise NOT_REPRODUCED. With N<=5 trials, bootstrap CIs are wide; do not read a ROBUST label here as a formal significance claim.

## Full per-metric descriptive statistics

| Scenario | Metric | Sarathi n/mean/median/stdev | vLLM n/mean/median/stdev |
|---|---|---|---|
| mistral_match_long_prompt_moderate_output | ttft_s | 5/0.4508/0.4502/0.0023 | 5/0.4883/0.4683/0.0374 |
| mistral_match_long_prompt_moderate_output | tpot_s | 5/0.0156/0.0156/0.0003 | 5/0.0133/0.0133/0.0000 |
| mistral_match_long_prompt_moderate_output | e2e_s | 5/4.1241/4.1207/0.0774 | 5/3.8686/3.8477/0.0398 |
| mistral_match_active_decode_plus_arriving_prefill | ttft_s | 5/0.1599/0.1597/0.0020 | 5/0.1543/0.1503/0.0092 |
| mistral_match_active_decode_plus_arriving_prefill | tpot_s | 5/0.0156/0.0154/0.0005 | 5/0.0113/0.0113/0.0000 |
| mistral_match_active_decode_plus_arriving_prefill | e2e_s | 5/0.6314/0.6238/0.0274 | 5/1.6486/1.6441/0.0110 |
| mistral_match_prefill_heavy_burst | ttft_s | 5/0.6263/0.6249/0.0039 | 5/0.5678/0.5685/0.0024 |
| mistral_match_prefill_heavy_burst | tpot_s | 5/0.0235/0.0235/0.0003 | 5/0.0206/0.0206/0.0000 |
| mistral_match_prefill_heavy_burst | e2e_s | 5/1.3543/1.3556/0.0128 | 5/1.2077/1.2084/0.0031 |
| mistral_match_mixed_prompt_lengths | ttft_s | 5/0.2660/0.2606/0.0143 | 5/0.2702/0.2650/0.0316 |
| mistral_match_mixed_prompt_lengths | tpot_s | 5/0.0181/0.0176/0.0011 | 5/0.0141/0.0142/0.0003 |
| mistral_match_mixed_prompt_lengths | e2e_s | 5/1.0193/0.9935/0.0618 | 5/0.8140/0.8224/0.0201 |
| mistral_match_kv_pressure | ttft_s | 5/1.1815/1.1791/0.0087 | 5/1.0549/1.0546/0.0032 |
| mistral_match_kv_pressure | tpot_s | 5/0.0173/0.0173/0.0001 | 5/0.0172/0.0172/0.0000 |
| mistral_match_kv_pressure | e2e_s | 5/12.7097/12.7272/0.0657 | 5/13.5458/13.5271/0.0340 |
