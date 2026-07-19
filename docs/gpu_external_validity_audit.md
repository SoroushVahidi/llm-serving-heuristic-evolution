# GPU External-Validity Audit

This audit validates whether the monolithic faithful-baseline simulator exposes
the same qualitative regimes as a real GPU runtime before continuing large-scale
Selector Dataset v2 scenario discovery. It does not train a selector and does
not launch a broad GPU sweep.

## Environment

Audit branch and HEAD:

- branch: `selector-dataset-v2-corrected-objective`
- HEAD: `03b901addcad0edf7817d3d0b6ea88a83f565f5d`

GPU/runtime environment recorded in
`experiments/gpu_external_validity/vllm_qwen05b_20260718T_gpu_audit/environment.json`:

- GPU: 1 x NVIDIA GeForce RTX 5060 Ti
- VRAM: 16,311 MiB total
- Driver: 580.159.03
- CUDA: 13.0
- vLLM: 0.24.0
- Torch: 2.11.0
- Transformers: 5.13.0
- served model: `Qwen/Qwen2.5-0.5B-Instruct`
- model max context: 4096
- active server: OpenAI-compatible vLLM endpoint on port 8001

The live server log at
`experiments/real_llm/vllm_healthcheck_20260703T171021Z/server.log` is treated as
protected runtime output and is not part of this audit commit.

Sarathi-Serve runtime status:

- `sarathi` package: not installed in the vLLM validation environment
- `sarathi-serve` package: not installed in the vLLM validation environment
- official Sarathi runtime validation: blocked pending a separate install/runtime
  attempt

## Harness

The bounded validation harness is
`scripts/run_gpu_external_validity_audit.py`.

It runs small deterministic request matrices against an already-running vLLM
server, records streaming TTFT and end-to-end latency, polls vLLM Prometheus
metrics, and runs matched simulator traces for:

- `vllm_faithful`
- `sarathi_faithful`

Controlled scenarios:

- short prompt / short output
- long prompt / short output
- short prompt / long output
- long prompt / long output
- mixed prompt lengths
- mixed output lengths
- bursty arrivals
- high concurrency
- prefill-heavy long prompts
- decode-heavy long outputs
- bounded KV-pressure attempt with long contexts
- mixed prompt/output batch turnover

Real-trace smoke replays:

- BurstGPT local processed trace
- Azure LLM 2023 local processed trace

The real-trace runs use source-derived token lengths, deterministic synthetic
prompt text, and compressed arrival timing. They are not large trace replays.

## Results

The first bounded run executed 14 scenarios and 104 requests. All requests
completed successfully.

Summary:

- scenarios: 14
- requests: 104
- successes: 104
- runtime mean latency: 0.4123 s
- runtime mean TTFT: 0.0333 s
- simulator vLLM mean latency: 0.0706 s
- simulator Sarathi mean latency: 0.0765 s
- median runtime/simulator-vLLM latency ratio: 5.93
- median Sarathi-simulator/vLLM-simulator latency ratio: 1.07
- scenarios with observed vLLM waiting queue: 0
- maximum observed vLLM KV-cache usage: 0.0076

Representative per-scenario measurements:

| Scenario | Requests | Runtime latency | Runtime TTFT | Runtime TPOT | Max waiting | Max KV usage | vLLM sim latency | Sarathi sim latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `short_short` | 4 | 0.1009 | 0.0296 | 0.0048 | 0 | 0.0004 | 0.0160 | 0.0170 |
| `long_prompt_short_output` | 4 | 0.1061 | 0.0343 | 0.0048 | 0 | 0.0035 | 0.0160 | 0.0210 |
| `short_prompt_long_output` | 4 | 0.5446 | 0.0243 | 0.0055 | 0 | 0.0015 | 0.0960 | 0.0973 |
| `prefill_heavy` | 6 | 0.1263 | 0.0425 | 0.0056 | 0 | 0.0042 | 0.0185 | 0.0310 |
| `decode_heavy` | 6 | 0.7632 | 0.0506 | 0.0057 | 0 | 0.0022 | 0.1280 | 0.1295 |
| `kv_pressure_long_context` | 12 | 0.7867 | 0.0453 | 0.0058 | 0 | 0.0076 | 0.1335 | 0.1556 |
| `burstgpt_replay_small` | 8 | 0.6491 | 0.0249 | 0.0055 | 0 | 0.0048 | 0.1149 | 0.1192 |
| `azure_2023_replay_small` | 8 | 0.3775 | 0.0212 | 0.0054 | 0 | 0.0044 | 0.0674 | 0.0696 |

## Interpretation

The vLLM runtime is healthy and reproducible enough for controlled validation,
but the first model/configuration did not exercise the key mechanisms that
matter for Selector Dataset v2:

- no request waiting queue was observed
- KV cache usage stayed below 1%
- no preemption or block-pressure behavior was exposed
- real TTFT is measurable, while the default simulator vLLM path remains much
  cheaper than real runtime timing
- Sarathi's simulated chunked-prefill model was not externally validated because
  the runtime is not installed

The observed runtime/simulator latency ratio should not be interpreted as a
required scalar correction. The current evidence shows a qualitative mismatch in
the exercised regimes: the bounded run validates basic request replay and timing
collection, but it does not validate KV pressure, admission pressure, or
chunked-prefill advantages.

## Baseline Classification

vLLM:

- status: `NEEDS_CALIBRATION`
- reason: the real runtime was exercised successfully, but the audit did not
  reach queueing/KV/preemption regimes and the simulator underestimates TTFT and
  end-to-end latency on this hardware/model.

Sarathi:

- status: `RUNTIME_VALIDATION_BLOCKED`
- reason: official or faithful Sarathi runtime is not installed in the available
  environment; only simulator-side Sarathi behavior was measured.

## Calibration Decision

No simulator calibration is committed by this audit.

Calibration is likely needed, but the current run is insufficient to choose a
specific parameter update. Before modifying simulator defaults or adding an
opt-in calibrated mode, run a second bounded validation with:

- a dedicated vLLM server whose flags are fully controlled by the audit
- prefix caching disabled if supported by the installed vLLM version
- either a larger model or longer context lengths to raise KV usage
- lower memory/concurrency limits when safe, to create real waiting/KV pressure
- enough concurrent long-decode requests to observe batch turnover
- an isolated Sarathi installation attempt or documented incompatibility result

## Dataset Implication

Do not resume large-scale Selector Dataset v2 generation yet.

The corrected-objective Dataset v2 pilot shows selector headroom, but the GPU
audit indicates that vLLM and Sarathi faithful-baseline advantage regions are
not yet externally validated. The next dataset action should be one bounded
calibration run designed specifically to induce KV pressure and prefill/decode
contention, then either:

- resume scenario discovery with the current simulator if qualitative regimes
  match,
- add an opt-in calibrated service model and regenerate the targeted pilot, or
- treat vLLM/Sarathi as reference baselines if their strongest advantages remain
  runtime-level effects outside the simulator model.

## Second Bounded Stress Audit

The second audit used the same harness with `--phase stress`, detached under
tmux session `gpu-external-validity-2`, and checkpointed after every completed
scenario.

Initial larger-model attempt:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- revision: `main`
- port: 8002
- command: `/home/soroush/.venvs/vllm_baseline_pilot/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8002 --revision main --gpu-memory-utilization 0.38 --max-model-len 4096 --max-num-seqs 4 --max-num-batched-tokens 1024 --block-size 16 --enable-chunked-prefill --no-enable-prefix-caching --enforce-eager`
- status: startup blocked; the server remained unready after about seven
  minutes and produced no scenario checkpoints

The larger-model attempt was stopped without touching the protected existing
healthcheck server. Because the validation phase is bounded, the audit then
fell back to a cached model with deliberately tighter scheduler limits.

Successful stress run:

- output directory:
  `experiments/gpu_external_validity/vllm_qwen05b_stress2_20260718T2212`
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- revision: `main`
- port: 8003
- command: `/home/soroush/.venvs/vllm_baseline_pilot/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8003 --revision main --gpu-memory-utilization 0.25 --max-model-len 4096 --max-num-seqs 2 --max-num-batched-tokens 512 --block-size 16 --enable-chunked-prefill --no-enable-prefix-caching --enforce-eager`
- start: 2026-07-19T02:11:09Z
- end: 2026-07-19T02:13:08Z
- scenarios: 8
- requests: 140
- completed requests: 140

Summary:

- runtime mean latency: 6.7380 s
- runtime mean TTFT: 5.2437 s
- simulator vLLM mean latency: 0.3628 s
- simulator Sarathi mean latency: 0.3735 s
- median runtime/simulator-vLLM latency ratio: 17.51
- scenarios with vLLM waiting queue: 8/8
- maximum waiting queue: 22 requests
- maximum running sequences: 2
- maximum observed KV-cache usage: 0.0214
- preemption events: 0

Per-scenario stress results:

| Scenario | Requests | Runtime latency | Runtime TTFT | Runtime TPOT | Throughput | Max running | Max waiting | Max KV | Preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stress_high_concurrency_queue` | 24 | 5.0616 | 4.3733 | 0.00545 | 2.696 | 2 | 22 | 0.00247 | 0 |
| `stress_long_decode_kv` | 16 | 6.8393 | 5.4166 | 0.00541 | 1.381 | 2 | 14 | 0.00461 | 0 |
| `stress_long_prefill` | 16 | 2.6578 | 2.1263 | 0.00559 | 3.407 | 2 | 14 | 0.01599 | 0 |
| `stress_kv_pressure` | 12 | 13.8003 | 9.8340 | 0.00544 | 0.474 | 2 | 10 | 0.02145 | 0 |
| `stress_mixed_prefill_decode_contention` | 16 | 3.4361 | 2.6478 | 0.00551 | 2.429 | 2 | 13 | 0.01631 | 0 |
| `stress_burst_overload_recovery` | 24 | 9.7045 | 8.1429 | 0.00544 | 1.256 | 2 | 19 | 0.01735 | 0 |
| `stress_burstgpt_replay` | 16 | 8.0859 | 6.2440 | 0.00543 | 1.032 | 2 | 13 | 0.01365 | 0 |
| `stress_azure_2023_replay` | 16 | 4.3181 | 3.1644 | 0.00548 | 1.566 | 2 | 11 | 0.01638 | 0 |

The second audit achieved queue buildup and visible prefill/decode contention:
TTFT dominated end-to-end latency once the server admitted only two concurrent
sequences. It did not achieve meaningful KV pressure or preemption.

The generated calibration profile is:

```text
experiments/gpu_external_validity/vllm_qwen05b_stress2_20260718T2212/calibration_profile.json
```

It is an opt-in artifact only and does not change simulator defaults. Its
observed fit is:

- TTFT fit: `ttft_seconds ~= 4.57685 + 0.000735846 * prompt_tokens`
- mean TPOT: 0.005470 s/token
- warning: KV usage stayed below 20%, so KV-pressure calibration remains
  incomplete

## Sarathi Runtime Probe

The official Sarathi-Serve repository is
`https://github.com/microsoft/sarathi-serve`. Its README states that the project
has been tested with CUDA 12.3 on H100 and A100 GPUs and gives source-install
instructions using a Python environment and `pip install -e .`.

An isolated PyPI probe was attempted in `/tmp/sarathi_serve_probe`:

```bash
python -m venv /tmp/sarathi_serve_probe
/tmp/sarathi_serve_probe/bin/python -m pip install --upgrade pip
/tmp/sarathi_serve_probe/bin/python -m pip install sarathi-serve
```

Result:

```text
ERROR: Could not find a version that satisfies the requirement sarathi-serve
ERROR: No matching distribution found for sarathi-serve
```

Sarathi runtime validation remains blocked until a source install is attempted
in an isolated environment compatible with this host's CUDA 13.0 / Python stack,
or a containerized CUDA 12.3-compatible path is prepared.

## Updated Classification

vLLM:

- status: `NEEDS_CALIBRATION`
- reason: the second audit validates queue buildup and TTFT growth under
  bounded scheduler pressure, but KV pressure and preemption remain unvalidated.

Sarathi:

- status: `RUNTIME_VALIDATION_BLOCKED`
- reason: no installable runtime package is available via PyPI, and official
  source-install requirements target a different tested CUDA/hardware envelope.

Dataset action:

- do not launch large-scale Dataset v2 generation yet
- use the stress-run calibration profile only as diagnostic evidence
- run one additional targeted KV-pressure experiment after either freeing the
  protected old vLLM server or selecting a larger/cached model that can start
  reliably
