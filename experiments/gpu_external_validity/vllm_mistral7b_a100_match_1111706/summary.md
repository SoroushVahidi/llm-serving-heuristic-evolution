# GPU External-Validity Audit

- Model: `mistralai/Mistral-7B-Instruct-v0.1`
- Server URL: `http://127.0.0.1:8030`
- vLLM HTTP version: `{"version":"0.24.0"}`
- Scenarios: 6
- Requests: 42 (42 success)
- Runtime mean latency: 3.545086163284093
- Runtime mean TTFT: 0.5098117446468677
- Simulator vLLM mean latency: 0.22577777777777777
- Median runtime/sim-vLLM latency ratio: 13.870500200522116
- Median Sarathi-sim/vLLM-sim latency ratio: 1.0327215426872767
- Scenarios with vLLM waiting >0: 5
- Max observed vLLM running sequences: 12.0
- Max observed vLLM KV-cache usage: 0.2975788288288288
- Preemption events: 0.0

## Scenario Table

| Scenario | Runtime mean latency | Runtime mean TTFT | vLLM sim mean latency | Sarathi sim mean latency | max running | max waiting | max KV | preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mistral_match_long_prompt_moderate_output | 3.8885 | 0.5010 | 0.2575 | 0.2670 | 4.0000 | 3.0000 | 0.0832 | 0.0000 |
| mistral_match_active_decode_plus_arriving_prefill | 1.6468 | 0.1508 | 0.1600 | 0.1642 | 4.0000 | 1.0000 | 0.0283 | 0.0000 |
| mistral_match_prefill_heavy_burst | 1.2151 | 0.5746 | 0.0345 | 0.0470 | 6.0000 | 5.0000 | 0.1128 | 0.0000 |
| mistral_match_mixed_prompt_lengths | 0.8237 | 0.2659 | 0.0652 | 0.0698 | 4.0000 | 3.0000 | 0.0497 | 0.0000 |
| mistral_match_kv_pressure | 13.6158 | 1.0569 | 0.7735 | 0.7956 | 12.0000 | 11.0000 | 0.2976 | 0.0000 |
| mistral_match_short_context_control | 0.0807 | n/a | 0.0640 | 0.0655 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
