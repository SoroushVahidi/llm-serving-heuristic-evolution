# Configs

All experiment configs are YAML files. Pass to a runner script with `--config`.

**Coverage note:** this file documents the pre-Phase-2A synthetic/real-trace/GPU-calibration
config families below. `configs/phase2b*.yaml`, `configs/phase2c*.yaml`, and the
`selector/`, `heuristics/`, `api_calibration/`, `real_llm_latency/`, and `oracle/`
subdirectories are not yet individually documented here — see
[docs/research_status.md](../docs/research_status.md) for which config was used for
which phase result.

---

## Synthetic workload configs (`configs/`)

Used with `scripts/run_baseline_comparison.py`.

| Config | Description | Long-running? |
|---|---|---|
| `small_debug.yaml` | Tiny trace, 1 seed; for smoke testing (seconds) | No |
| `baseline_comparison.yaml` | Standard Poisson workload | Yes |
| `overloaded_comparison.yaml` | High arrival rate; queuing saturation | Yes |
| `overloaded_prefill_comparison.yaml` | High rate with long prompts | Yes |
| `prefill_heavy_comparison.yaml` | Long prompts, short outputs | Yes |
| `decode_heavy_comparison.yaml` | Short prompts, long outputs | Yes |
| `mixed_slo_comparison.yaml` | Mixed tight/relaxed SLO tiers | Yes |
| `stressed_comparison.yaml` | Stressed system with combined pressure | Yes |
| `burst_heavy_tail_comparison.yaml` | Heavy-tail bursty arrivals | Yes |
| `burstgpt_replay_comparison.yaml` | BurstGPT replay (synthetic service model) | Yes |
| `burstgpt_replay_scaled_load.yaml` | BurstGPT replay at scaled load | Yes |
| `default_simulator.yaml` | Default simulator parameter reference | N/A |
| `sharegpt_poisson_comparison.yaml` | ShareGPT tokens with Poisson arrivals | Yes |
| `sharegpt_bursty_comparison.yaml` | ShareGPT tokens with bursty arrivals | Yes |

---

## Real-trace configs (`configs/real_trace/`)

Used with `scripts/run_real_trace_comparison.py`. Require BurstGPT processed traces
at `data/processed/burstgpt/`. All are long-running (90–150 min each).

| Config | Service model | Trace | Load level |
|---|---|---|---|
| `burstgpt_natural_calibrated.yaml` | Calibrated | Natural BurstGPT (~318ks span) | Very low — no queuing |
| `burstgpt_scaled_moderate_calibrated.yaml` | Calibrated | Scaled to 190.7s span | Moderate |
| `burstgpt_scaled_high_calibrated.yaml` | Calibrated | Scaled to 127.2s span | High |
| `burstgpt_scaled_moderate_synthetic_service.yaml` | Synthetic | Scaled to 190.7s span | Moderate |
| `burstgpt_moderate_exact_prediction.yaml` | Calibrated | 0% prediction noise | Moderate |
| `burstgpt_moderate_noise035.yaml` | Calibrated | Natural BurstGPT prediction noise | Moderate |
| `burstgpt_moderate_noise070.yaml` | Calibrated | 70% amplified prediction noise | Moderate |

**Calibrated vs. synthetic service model:**
- `service_model.type: calibrated` — uses `results/gpu_calibration/service_curves.json`
  for RTX 5060 Ti–measured prefill/decode timing.
- `service_model.type: synthetic` — uses parametric `prefill_cost_per_token` /
  `decode_cost_per_token` values; no real GPU required.

**Prediction noise variants:**
- `exact_prediction` — `predicted_output_tokens == actual_output_tokens` for all requests.
- `noise035` — uses the original BurstGPT predicted token field (natural noise level).
- `noise070` — uses a pre-generated 70%-noise JSONL file.

---

## GPU calibration configs (`configs/gpu_calibration/`)

Used with `scripts/run_gpu_calibration.py` and `scripts/validate_simulator_calibration.py`.

| Config | Purpose |
|---|---|
| `calibration_grid.yaml` | Grid of (prompt_len, output_len) measurement points |
| `environment.yaml` | Target GPU, model, batch size settings |
| `model.yaml` | HuggingFace model ID and quantization |
| `online_validation.yaml` | Held-out validation grid for MAPE check |
| `validation_grid.yaml` | Alternate validation sweep |

---

## Trace conversion configs (`configs/traces/`)

Used with conversion scripts for pre-generating specific JSONL trace variants.

| Config | Purpose |
|---|---|
| `burstgpt_conversion.yaml` | Standard BurstGPT → JSONL conversion params |
| `burstgpt_exact_prediction.yaml` | Overrides predicted tokens = actual tokens |
| `burstgpt_noise070.yaml` | Generates 70%-noise prediction override |
| `prediction_noise_exact.yaml` | Zero-noise augmentation params |
| `prediction_noise_moderate.yaml` | Moderate noise augmentation params |
| `prediction_noise_high.yaml` | High noise augmentation params |
| `sharegpt_conversion.yaml` | ShareGPT → JSONL conversion params |

---

## Service model fields

In real-trace configs, the `service_model` block controls which service model is used:

```yaml
service_model:
  type: calibrated                          # or: synthetic
  calibration_file: results/gpu_calibration/service_curves.json
  enable_prefill_modeling: true
  max_prefill_chunk_tokens: 512
  decode_first: true
```

For `type: synthetic`:
```yaml
service_model:
  type: synthetic
  enable_prefill_modeling: true
  prefill_cost_per_token: 1.0
  max_prefill_chunk_tokens: 512
  step_token_budget: 4096
  decode_first: false
```

---

## Adding a new config

1. Copy the closest existing config.
2. Update `experiment:` name (determines the result directory prefix).
3. Update `trace_file:` and `service_model:` fields.
4. If using `type: calibrated`, ensure `results/gpu_calibration/service_curves.json` exists.
5. Run with `--config configs/your_new_config.yaml`.
6. Results land in `results/<experiment_name>/<timestamp>/`.

---

## Selector training vs. test traces

For Phase 2A (selector), use **only synthetic or BurstGPT natural-timing** traces
for selector training. The noise variants and scaled-load variants are reserved as
out-of-distribution test regimes. Do not train a selector on the same trace it is
evaluated on.
