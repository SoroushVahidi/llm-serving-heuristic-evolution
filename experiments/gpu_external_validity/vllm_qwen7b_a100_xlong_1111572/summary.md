# GPU External-Validity Audit

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Server URL: `http://127.0.0.1:8020`
- vLLM HTTP version: `{"version":"0.24.0"}`
- Scenarios: 2
- Requests: 40 (40 success)
- Runtime mean latency: 20.338828859517285
- Runtime mean TTFT: 5.477979635822825
- Simulator vLLM mean latency: None
- Median runtime/sim-vLLM latency ratio: None
- Median Sarathi-sim/vLLM-sim latency ratio: None
- Scenarios with vLLM waiting >0: 2
- Max observed vLLM running sequences: 18.0
- Max observed vLLM KV-cache usage: 0.8366727383120826
- Preemption events: 0.0

## Scenario Table

| Scenario | Runtime mean latency | Runtime mean TTFT | vLLM sim mean latency | Sarathi sim mean latency | max running | max waiting | max KV | preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stress_xlong_context_burst16 | 20.1444 | 6.5069 | nan | 0.8262 | 16.0000 | 15.0000 | 0.7473 | 0.0000 |
| stress_xlong_context_saturate | 20.5333 | 4.4491 | nan | 1.1135 | 18.0000 | 14.0000 | 0.8367 | 0.0000 |
