# Warm-up phase — NOT counted in policy metrics

One short/target=64 and one medium/target=128 request at concurrency=1,
run before any measured policy loop to absorb vLLM's one-time JIT kernel
compilation latency spike under `--enforce-eager` (observed in
`experiments/real_llm/vllm_healthcheck_*/healthcheck.md`: first request
needed ~180s, subsequent requests ~0.3s). These requests are excluded
from `requests.jsonl` and every `aggregate_by_*.csv` / policy metric.

| request_id | bucket | target | status | ttft_s | server_latency_s | wall_s |
|---|---|---|---|---|---|---|
| -1 | short | 64 | success | 0.027144722000230104 | 0.6661065319785848 | 0.6662 |
| -2 | medium | 128 | success | 0.013430904014967382 | 1.3231923970161006 | 1.3233 |
