# GPU External-Validity Audit

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Server URL: `http://127.0.0.1:8010`
- vLLM HTTP version: `{"version":"0.24.0"}`
- Scenarios: 6
- Requests: 108 (108 success)
- Runtime mean latency: 4.191505699411613
- Runtime mean TTFT: 0.6875711722364536
- Simulator vLLM mean latency: 0.35508680555555555
- Median runtime/sim-vLLM latency ratio: 14.031643520126517
- Median Sarathi-sim/vLLM-sim latency ratio: 1.030665722959096
- Scenarios with vLLM waiting >0: 4
- Max observed vLLM running sequences: 24.0
- Max observed vLLM KV-cache usage: 0.13959390862944165
- Preemption events: 0.0

## Scenario Table

| Scenario | Runtime mean latency | Runtime mean TTFT | vLLM sim mean latency | Sarathi sim mean latency | max running | max waiting | max KV | preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stress_high_concurrency_queue | 2.0797 | 1.4133 | 0.1284 | 0.1326 | 24.0000 | 0.0000 | 0.0237 | 0.0000 |
| stress_long_decode_kv | 2.6514 | 0.1611 | 0.5121 | 0.5152 | 16.0000 | 0.0000 | 0.0226 | 0.0000 |
| stress_long_prefill | 2.7654 | 0.9985 | 0.1035 | 0.1320 | 16.0000 | 10.0000 | 0.1387 | 0.0000 |
| stress_kv_pressure | 10.2254 | 0.7859 | 0.7735 | 0.7956 | 12.0000 | 6.0000 | 0.1396 | 0.0000 |
| stress_mixed_prefill_decode_contention | 3.1028 | 0.3383 | 0.3218 | 0.3306 | 16.0000 | 3.0000 | 0.0835 | 0.0000 |
| stress_burst_overload_recovery | 4.3245 | 0.4284 | 0.2913 | 0.3022 | 24.0000 | 6.0000 | 0.1264 | 0.0000 |
