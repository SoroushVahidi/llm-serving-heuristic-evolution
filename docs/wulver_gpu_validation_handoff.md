# Wulver GPU Validation Handoff

This handoff prepares the code path that Wulver should fetch for the next
GPU-validation phase. It does not submit jobs and does not assume authenticated
Wulver account, QOS, partition, module, or storage details.

## Branch and Commit

- branch to fetch: `wulver-gpu-validation-ready`
- base development commit before this handoff document: `e5bd2b148aef9e0b332ddf1a5f3ecede87cf1bd7`
- final pushed handoff commit: verify with `git rev-parse HEAD` after fetching
  `wulver-gpu-validation-ready`; the exact final SHA is also recorded in the
  local preparation report.

The branch is a descendant of the required development chain:

- faithful vLLM scheduler/KV baseline
- faithful Sarathi-Serve scheduler baseline
- DistServe disaggregated prefill/decode infrastructure and faithful baseline
- TetriInfer paper reimplementation
- Llumnix faithful scheduler and live migration infrastructure
- topology-aware external-baseline registry, configs, and harness
- Selector Dataset v2 schema, features, provenance, split, regret, and scenario
  search infrastructure
- corrected selector objective:
  `arrival_normalized_weighted_goodput`
- selector objective audit and historical metric compatibility
- GPU external-validity audit harness, stress phase, checkpointing, and
  calibration-profile generation

## Purpose

The next Wulver experiment should validate whether real vLLM runtime behavior on
a larger GPU/model exposes KV-pressure and preemption regimes that the local RTX
5060 Ti experiments did not reach.

This is a validation/calibration task only:

- do not train the final selector
- do not generate large Selector Dataset v2 data
- do not change historical simulator defaults
- use Slurm for long-running GPU work

## Latest Local GPU Findings

Local hardware:

- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB
- CUDA: 13.0
- vLLM: 0.24.0

First local audit:

- 14 scenarios
- 104 requests
- max vLLM waiting queue: 0
- max KV usage: less than 1%
- conclusion: too light to validate queueing, KV pressure, preemption, or
  chunked-prefill advantages

Second local stress audit:

- output directory:
  `experiments/gpu_external_validity/vllm_qwen05b_stress2_20260718T2212`
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- server flags included `--max-num-seqs 2`,
  `--max-num-batched-tokens 512`, `--enable-chunked-prefill`,
  `--no-enable-prefix-caching`
- 8 scenarios
- 140 requests
- 140 completions
- every scenario showed a nonzero waiting queue
- maximum waiting queue: 22
- maximum running sequences: 2
- maximum KV usage: 0.02145
- preemption events: 0
- runtime mean TTFT: 5.244 s
- runtime mean latency: 6.738 s
- simulator vLLM mean latency: 0.363 s
- median runtime/simulator-vLLM latency ratio: 17.51

Conclusion:

- queue buildup and TTFT growth were validated locally
- meaningful KV pressure and preemption were not validated locally
- Wulver is needed for larger model/context experiments on larger GPU memory

## Recommended Next Experiment

Run a bounded vLLM KV-pressure/preemption validation job on Wulver using one
larger GPU, preferably a full high-memory GPU rather than MIG, after verifying
the authenticated Slurm/account environment.

Recommended starting point after Wulver audit:

- model class: 7B or 8B instruction model
- example model: `Qwen/Qwen2.5-7B-Instruct`
- max model length: start at 16k if the selected node supports it
- GPU count: 1 for first Wulver validation
- vLLM settings:
  - disable prefix caching
  - enable chunked prefill
  - set explicit `max-num-seqs`
  - set explicit `max-num-batched-tokens`
  - set explicit `block-size`
  - record vLLM version and full server command
- target outcomes:
  - nonzero waiting queue
  - KV usage above 20%; ideally above 50% in at least one scenario
  - preemption events if vLLM triggers them
  - measurable TTFT, TPOT, latency, throughput, and queueing distributions

If a 7B/8B model is too large for verified Wulver constraints, use the largest
reliable model that still leaves enough KV cache for long-context pressure.

## Required Scripts and Files

- `scripts/run_gpu_external_validity_audit.py`
- `tests/test_gpu_external_validity_audit.py`
- `docs/gpu_external_validity_audit.md`
- `scripts/slurm/wulver_vllm_kv_pressure_template.sbatch`
- `experiments/gpu_external_validity/vllm_qwen05b_stress2_20260718T2212/summary.json`
- `experiments/gpu_external_validity/vllm_qwen05b_stress2_20260718T2212/scenario_results.json`
- `experiments/gpu_external_validity/vllm_qwen05b_stress2_20260718T2212/calibration_profile.json`

## Expected Outputs

Commit back to GitHub if reasonably sized:

- `environment.json`
- `summary.json`
- `summary.md`
- `scenario_summary.csv`
- `scenario_results.json`
- `calibration_profile.json`
- Slurm stdout/stderr summaries if small and non-sensitive
- a short documentation update with interpretation and next decision

Keep on Wulver storage, not GitHub:

- model weights and Hugging Face cache
- Python or Conda environments
- container images
- large raw server logs
- very large request-level traces
- multi-GB result directories
- credentials, tokens, or private cluster paths

## Reproducibility Requirements

Record:

- exact Git branch and commit
- exact Slurm job ID
- hostname and GPU node type
- GPU model and GPU memory
- CUDA/module/container environment
- vLLM version
- model ID and revision
- complete server command
- complete audit command
- output directory
- checksum or size metadata for any generated artifacts kept off GitHub

Do not submit the Wulver job until the authenticated cluster audit has verified:

- Slurm account
- QOS
- GPU partition
- GPU resource request syntax
- maximum walltime
- module/container availability
- project and scratch paths
- outbound internet/Hugging Face access policy
