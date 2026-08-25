# Runtime Validation Benchmark Pack

Compact, canonical real-hardware ground truth for validating simulator
baselines (in particular the future `vllm_chunked_prefill_faithful`)
without needing GPU access. See `docs/runtime_validation_benchmark_pack.md`
for the full write-up (purpose, provenance, schema, safe claims,
limitations).

## Layout

- `scenarios/` -- request-level fixtures (arrival times, prompt/output
  lengths, ordering) for the 5 primary repeated-validation scenarios. No
  full prompt text; `prompt_sha256` lets you verify regenerated text
  matches exactly.
- `hardware_targets/` -- real-hardware TTFT/TPOT/E2E statistics (mean,
  median, stdev, p50/p95), winner identity/frequency, and paired bootstrap
  95% CI, per scenario, from the N=5 repeated-trial arrays.
- `simulator_baseline_results/` -- `vllm_faithful`/`sarathi_faithful`
  simulator numbers per scenario, whether the simulator's E2E winner
  matches the robust hardware winner, and the documented reason where a
  mismatch is already understood.
- `long_context/` -- one fixture (`xlong_context_burst16.json`) that real
  vLLM completes, `vllm_faithful` cannot admit at all (2,560-token budget,
  no chunking at its v0.1.0 pin), and a future
  `vllm_chunked_prefill_faithful` must complete via chunked admission.
- `kv_progression/` -- structural (not timing) KV-pressure/preemption
  regression target across jobs 1111541/1111545/1111572.
- `manifest.json` -- source job list, positive targets, negative controls,
  and a sha256 + size for every file in this pack.

## Positive targets vs. negative controls

- **Positive targets** (`active_decode_plus_arriving_prefill`,
  `kv_pressure`): real hardware shows a ROBUST (5/5 trials, bootstrap CI
  excludes zero) Sarathi E2E advantage. A correct simulator should predict
  Sarathi wins here.
- **Negative controls** (`long_prompt_moderate_output`,
  `prefill_heavy_burst`, `mixed_prompt_lengths`): real hardware shows an
  equally robust vLLM E2E advantage. A correct simulator should predict
  vLLM wins here -- and should NOT show a false Sarathi advantage.

## Regeneration

```
python scripts/build_runtime_validation_benchmark_pack.py
```

Deterministic; reads only already-committed
`experiments/gpu_external_validity/` artifacts plus the (also already
committed, unmodified) scenario-definition source code. No GPU, no
network, no new experiments.
