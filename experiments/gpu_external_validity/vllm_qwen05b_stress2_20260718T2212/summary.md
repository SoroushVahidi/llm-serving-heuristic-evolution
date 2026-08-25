# GPU External-Validity Audit

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Server URL: `http://127.0.0.1:8003`
- vLLM HTTP version: `{"version":"0.24.0"}`
- Scenarios: 8
- Requests: 140 (140 success)
- Runtime mean latency: 6.7379550345995085
- Runtime mean TTFT: 5.243666902650754
- Simulator vLLM mean latency: 0.36277604166666666
- Median runtime/sim-vLLM latency ratio: 17.5112724307239
- Median Sarathi-sim/vLLM-sim latency ratio: 1.0279695794401675
- Scenarios with vLLM waiting >0: 8
- Max observed vLLM running sequences: 2.0
- Max observed vLLM KV-cache usage: 0.021448069673729364
- Preemption events: 0.0

## Scenario Table

| Scenario | Runtime mean latency | Runtime mean TTFT | vLLM sim mean latency | Sarathi sim mean latency | max running | max waiting | max KV | preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stress_high_concurrency_queue | 5.0616 | 4.3733 | 0.1284 | 0.1326 | 2.0000 | 22.0000 | 0.0025 | 0.0000 |
| stress_long_decode_kv | 6.8393 | 5.4166 | 0.5121 | 0.5152 | 2.0000 | 14.0000 | 0.0046 | 0.0000 |
| stress_long_prefill | 2.6578 | 2.1263 | 0.1035 | 0.1320 | 2.0000 | 14.0000 | 0.0160 | 0.0000 |
| stress_kv_pressure | 13.8003 | 9.8340 | 0.7735 | 0.7956 | 2.0000 | 10.0000 | 0.0214 | 0.0000 |
| stress_mixed_prefill_decode_contention | 3.4361 | 2.6478 | 0.3218 | 0.3306 | 2.0000 | 13.0000 | 0.0163 | 0.0000 |
| stress_burst_overload_recovery | 9.7045 | 8.1429 | 0.2913 | 0.3022 | 2.0000 | 19.0000 | 0.0174 | 0.0000 |
| stress_burstgpt_replay | 8.0859 | 6.2440 | 0.4706 | 0.4758 | 2.0000 | 13.0000 | 0.0136 | 0.0000 |
| stress_azure_2023_replay | 4.3181 | 3.1644 | 0.3011 | 0.3044 | 2.0000 | 11.0000 | 0.0164 | 0.0000 |
