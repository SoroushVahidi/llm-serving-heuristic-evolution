# GPU External-Validity Audit

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Server URL: `http://127.0.0.1:8001`
- vLLM HTTP version: `{"version":"0.24.0"}`
- Scenarios: 14
- Requests: 104 (104 success)
- Runtime mean latency: 0.41230902236462236
- Runtime mean TTFT: 0.03333469498604592
- Simulator vLLM mean latency: 0.07057745246385394
- Median runtime/sim-vLLM latency ratio: 5.927942370192554
- Median Sarathi-sim/vLLM-sim latency ratio: 1.0709786821705427
- Scenarios with vLLM waiting >0: 0
- Max observed vLLM KV-cache usage: 0.0076238504894690085

## Scenario Table

| Scenario | Runtime mean latency | Runtime mean TTFT | vLLM sim mean latency | Sarathi sim mean latency | max waiting | max KV |
|---|---:|---:|---:|---:|---:|---:|
| short_short | 0.1009 | 0.0296 | 0.0160 | 0.0170 | 0.0000 | 0.0004 |
| long_prompt_short_output | 0.1061 | 0.0343 | 0.0160 | 0.0210 | 0.0000 | 0.0035 |
| short_prompt_long_output | 0.5446 | 0.0243 | 0.0960 | 0.0973 | 0.0000 | 0.0015 |
| long_prompt_long_output | 0.5511 | 0.0344 | 0.0975 | 0.1070 | 0.0000 | 0.0045 |
| mixed_prompt_lengths | 0.1771 | 0.0180 | 0.0320 | 0.0347 | 0.0000 | 0.0037 |
| mixed_output_lengths | 0.3841 | 0.0198 | 0.0693 | 0.0713 | 0.0000 | 0.0017 |
| bursty_arrivals | 0.3857 | 0.0310 | 0.0645 | 0.0696 | 0.0000 | 0.0028 |
| high_concurrency | 0.3992 | 0.0384 | 0.0640 | 0.0665 | 0.0000 | 0.0030 |
| prefill_heavy | 0.1263 | 0.0425 | 0.0185 | 0.0310 | 0.0000 | 0.0042 |
| decode_heavy | 0.7632 | 0.0506 | 0.1280 | 0.1295 | 0.0000 | 0.0022 |
| kv_pressure_long_context | 0.7867 | 0.0453 | 0.1335 | 0.1556 | 0.0000 | 0.0076 |
| batch_turnover_mixed | 0.4208 | 0.0525 | 0.0704 | 0.0812 | 0.0000 | 0.0060 |
| burstgpt_replay_small | 0.6491 | 0.0249 | 0.1149 | 0.1192 | 0.0000 | 0.0048 |
| azure_2023_replay_small | 0.3775 | 0.0212 | 0.0674 | 0.0696 | 0.0000 | 0.0044 |
