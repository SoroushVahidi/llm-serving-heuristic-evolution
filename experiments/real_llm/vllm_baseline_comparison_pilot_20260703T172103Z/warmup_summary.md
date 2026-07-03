# Warm-up phase — NOT counted in policy metrics

One short/target=64 and one medium/target=128 request at concurrency=1,
run before any measured policy loop to absorb vLLM's one-time JIT kernel
compilation latency spike under `--enforce-eager` (observed in
`experiments/real_llm/vllm_healthcheck_*/healthcheck.md`: first request
needed ~180s, subsequent requests ~0.3s). These requests are excluded
from `requests.jsonl` and every `aggregate_by_*.csv` / policy metric.

| request_id | bucket | target | status | ttft_s | server_latency_s | wall_s |
|---|---|---|---|---|---|---|
| -1 | short | 64 | success | 0.02935018000425771 | 0.6683823650237173 | 0.6684 |
| -2 | medium | 128 | success | 0.015265927999280393 | 1.3116077469894662 | 1.3117 |
