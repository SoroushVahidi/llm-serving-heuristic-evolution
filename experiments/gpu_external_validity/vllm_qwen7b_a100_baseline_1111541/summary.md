# GPU External-Validity Audit

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Server URL: `http://127.0.0.1:8003`
- vLLM HTTP version: `{"version":"0.24.0"}`
- Scenarios: 6
- Requests: 108 (108 success)
- Runtime mean latency: 5.191257355420222
- Runtime mean TTFT: 2.0952132029103976
- Simulator vLLM mean latency: 0.35508680555555555
- Median runtime/sim-vLLM latency ratio: 18.729325025433567
- Median Sarathi-sim/vLLM-sim latency ratio: 1.030665722959096
- Scenarios with vLLM waiting >0: 6
- Max observed vLLM running sequences: 8.0
- Max observed vLLM KV-cache usage: 0.022173063961426154
- Preemption events: 0.0

## Scenario Table

| Scenario | Runtime mean latency | Runtime mean TTFT | vLLM sim mean latency | Sarathi sim mean latency | max running | max waiting | max KV | preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stress_high_concurrency_queue | 2.8464 | 2.2897 | 0.1284 | 0.1326 | 8.0000 | 16.0000 | 0.0019 | 0.0000 |
| stress_long_decode_kv | 3.3973 | 1.1537 | 0.5121 | 0.5152 | 8.0000 | 8.0000 | 0.0030 | 0.0000 |
| stress_long_prefill | 2.9247 | 1.4298 | 0.1035 | 0.1320 | 8.0000 | 12.0000 | 0.0170 | 0.0000 |
| stress_kv_pressure | 12.0136 | 3.3904 | 0.7735 | 0.7956 | 8.0000 | 8.0000 | 0.0222 | 0.0000 |
| stress_mixed_prefill_decode_contention | 3.5775 | 1.2484 | 0.3218 | 0.3306 | 8.0000 | 8.0000 | 0.0167 | 0.0000 |
| stress_burst_overload_recovery | 6.3881 | 3.0594 | 0.2913 | 0.3022 | 8.0000 | 15.0000 | 0.0112 | 0.0000 |
