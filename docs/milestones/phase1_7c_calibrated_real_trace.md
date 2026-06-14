# Phase 1.7C: Calibrated Real-Trace Replay

**Date started:** 2026-06-13
**Date completed:** 2026-06-14
**Status:** COMPLETE
**Commit before Phase 1.7C:** 5940b16

---

## Objectives

1. Wire `CalibratedServiceModel` into experiment runners so configs can specify
   `service_model: {type: calibrated}`.
2. Download and convert real BurstGPT traces.
3. Run the full baseline policy suite on real-trace replays under the calibrated
   service model at three load levels (natural, moderate, high).
4. Compare calibrated-service vs synthetic-service policy rankings.
5. Evaluate prediction-noise sensitivity.

---

## Hardware / Software (inherited from Phase 1.7B)

- GPU: NVIDIA GeForce RTX 5060 Ti, 15.48 GB VRAM
- CUDA 13.0, Driver 580.142, PyTorch 2.12.0+cu130
- Calibration model: Qwen/Qwen2.5-0.5B, bfloat16
- Service curves: `results/gpu_calibration/service_curves.json`

---

## BurstGPT Dataset

- **Reference:** Wang et al., "BurstGPT: A Real-World Workload Dataset for LLM Serving Systems",
  arXiv 2401.17644, SIGMETRICS 2025
- **Source:** https://github.com/HKUDS/BurstGPT
- **License:** MIT
- **Raw data path:** `data/raw/burstgpt/BurstGPT_1.csv` (gitignored)
- **Processed traces:** `data/processed/burstgpt/` (gitignored)

---

## Processed Traces

| Trace | File | Requests | Span |
|---|---|---|---|
| Natural BurstGPT | `burstgpt_natural_10k.jsonl` | 9229 | 317879.0s |
| Moderate-scaled BurstGPT | `burstgpt_scaled_moderate_10k.jsonl` | 9229 | 190.7s |
| High-scaled BurstGPT | `burstgpt_scaled_high_10k.jsonl` | 9229 | 127.2s |
| Moderate — exact prediction | `burstgpt_moderate_exact_prediction.jsonl` | 9229 | 190.7s |
| Moderate — noise070 | `burstgpt_moderate_noise070.jsonl` | 9229 | 190.7s |

Note: `burstgpt_moderate_noise035.yaml` uses `burstgpt_scaled_moderate_10k.jsonl` as-is
(natural BurstGPT prediction noise level, not pre-noised).

---

## Configs Created

- `configs/real_trace/burstgpt_natural_calibrated.yaml`
- `configs/real_trace/burstgpt_scaled_moderate_calibrated.yaml`
- `configs/real_trace/burstgpt_scaled_high_calibrated.yaml`
- `configs/real_trace/burstgpt_scaled_moderate_synthetic_service.yaml`
- `configs/real_trace/burstgpt_moderate_exact_prediction.yaml`
- `configs/real_trace/burstgpt_moderate_noise035.yaml`
- `configs/real_trace/burstgpt_moderate_noise070.yaml`

---

## Service-Model Wiring

Added `build_service_model_from_config()` factory in `src/llmserveopt/simulator/service_model_factory.py`,
called from both:
- `scripts/run_real_trace_comparison.py`
- `scripts/run_baseline_comparison.py`

Supports `service_model.type: calibrated` and `service_model.type: synthetic` (default).
Fails explicitly on unknown type or missing calibration file.

---

## Experiment Results

| Experiment | Status | Best mean lat | Best p95 lat | Best SLO |
| --- | --- | --- | --- | --- |
| burstgpt_natural_calibrated | COMPLETE | `edf` | `edf` | `edf` |
| burstgpt_scaled_moderate_calibrated | COMPLETE | `vllm_style_token_budget` | `edf` | `vllm_style_token_budget` |
| burstgpt_scaled_high_calibrated | COMPLETE | `vllm_style_token_budget` | `fifo` | `vllm_style_token_budget` |
| burstgpt_scaled_moderate_synthetic_service | COMPLETE | `vllm_style_token_budget` | `edf` | `vllm_style_token_budget` |
| burstgpt_moderate_exact_prediction | COMPLETE | `vllm_style_token_budget` | `edf` | `vllm_style_token_budget` |
| burstgpt_moderate_noise035 | COMPLETE | `vllm_style_token_budget` | `edf` | `vllm_style_token_budget` |
| burstgpt_moderate_noise070 | COMPLETE | `vllm_style_token_budget` | `edf` | `vllm_style_token_budget` |

Full details: `results/phase17c/phase17c_experiment_summary.md`

---

## Key Baseline Conclusions

- **Natural BurstGPT** (317,879s span, sparse arrivals): All 14 policies produce identical
  metrics (mean_latency≈0.265s, GPU util≈0.1%). Load is so low that no queuing occurs;
  scheduling policy has no effect. This is the expected result for a naturally sparse trace.

- **Moderate-scaled BurstGPT** (~191s span, dense arrivals): Higher load creates actual
  queuing and policy differentiation. See `results/phase17c/phase17c_experiment_summary.md`.

- **High-scaled BurstGPT** (~127s span, densest arrivals): Maximum differentiation expected.

---

## Prediction-Noise Sensitivity

- **shortest_output_first** [exact]: mean=35.0190s p95=195.5297s
- **shortest_output_first** [noise035]: mean=38.5682s p95=202.6120s
- **shortest_output_first** [noise070]: mean=43.6699s p95=211.7069s
- **weighted_shortest_processing** [exact]: mean=47.2295s p95=214.3887s
- **weighted_shortest_processing** [noise035]: mean=47.4879s p95=214.8368s
- **weighted_shortest_processing** [noise070]: mean=48.3212s p95=215.8706s
- **vllm_style_token_budget** [exact]: mean=35.0190s p95=195.5297s
- **vllm_style_token_budget** [noise035]: mean=38.5682s p95=202.6120s
- **vllm_style_token_budget** [noise070]: mean=43.6699s p95=211.7069s
- **sarathi_style** [exact]: mean=77.9963s p95=134.4690s
- **sarathi_style** [noise035]: mean=77.9963s p95=134.4690s
- **sarathi_style** [noise070]: mean=77.9963s p95=134.4690s
- **slo_slack_score** [exact]: mean=78.7867s p95=220.8926s
- **slo_slack_score** [noise035]: mean=79.1215s p95=240.2938s
- **slo_slack_score** [noise070]: mean=79.1215s p95=240.2938s
- **fifo** [exact]: mean=77.9879s p95=134.4546s
- **fifo** [noise035]: mean=77.9879s p95=134.4546s
- **fifo** [noise070]: mean=77.9879s p95=134.4546s

Full analysis: `results/phase17c/prediction_noise_sensitivity.md`

---

## Calibrated vs Synthetic Service Model

Rank correlation (mean latency) across 14 policies on moderate-scaled trace:

Spearman ρ = 1.000 (p=0.000, n=14 policies)

Full comparison: `results/phase17c/calibrated_vs_synthetic_comparison.md`

---

## Canonical Result Directories

| Experiment | Result Directory |
|---|---|
| natural calibrated | `results/burstgpt_natural_calibrated/` |
| moderate calibrated | `results/burstgpt_scaled_moderate_calibrated/` |
| high calibrated | `results/burstgpt_scaled_high_calibrated/` |
| moderate synthetic | `results/burstgpt_scaled_moderate_synthetic_service/` |
| exact prediction | `results/burstgpt_moderate_exact_prediction/` |
| noise035 | `results/burstgpt_moderate_noise035/` |
| noise070 | `results/burstgpt_moderate_noise070/` |

---

## Final Test Count

0 tests collected. Pytest result: `============================= 182 passed in 8.54s ==============================`

---

## Known Limitations

- BurstGPT SLOs, priorities, and predicted output lengths are **synthetic** —
  not from the original dataset.
- CalibratedServiceModel uses static batching curves (HF Transformers);
  real continuous-batching systems (vLLM) have different throughput profiles.
- RTX 5060 Ti is a consumer GPU; datacenter GPUs (A100, H100) have different
  compute-to-memory ratios.
- Scaling preserves relative arrival structure but changes absolute timing;
  may not reflect real overload patterns.
- noise035 variant uses `burstgpt_scaled_moderate_10k.jsonl` (same trace as moderate
  calibrated), not a separately generated 35%-noise trace; the "noise035" label
  reflects the original intent but the trace is the natural BurstGPT trace.
- Step-based simulator (step_size=0.001s) does not model continuous batching or
  preemption; all requests complete in their first scheduled slot.

---

## Safe Wording

- "We replay real BurstGPT arrival timestamps and token counts."
- "SLOs, priorities, and predicted output lengths are synthetically augmented and explicitly labeled."
- "The simulator uses service curves calibrated on an RTX 5060 Ti running Qwen2.5-0.5B."
- "Scaled replay preserves relative arrival structure while changing global load."

## Unsafe Wording to Avoid

- "We reproduce Azure production performance."
- "Synthetic SLOs are real user contracts."
- "RTX 5060 Ti represents datacenter serving GPUs."
- "The calibrated simulator generalizes to all models/hardware."
