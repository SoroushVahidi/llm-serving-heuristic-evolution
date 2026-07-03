# Reproducibility Metadata — vLLM Real-Server Health Check

- Generated: 2026-07-03T17:14:41Z
- Git branch: `phase2c1-real-trace-ingestion-validation`
- Git commit: `f3044d4561d012ea9b8522e63ac4a6198517773f`
- Git dirty: true (untracked additions only — this directory and its
  siblings, plus `scripts/run_vllm_external_baseline_comparison.py`; no
  tracked file was modified)

## Environment

- Isolated venv: `/home/soroush/.venvs/vllm_baseline_pilot` (separate from
  the repo's own `modal-venv`; created specifically for this vLLM work,
  nothing installed into the main environment)
- Python: 3.12.3
- pip: 26.1.2 (at install time)
- vLLM: 0.24.0 (`pip install vllm`, no version pin — latest as of
  2026-07-03)
- PyTorch: 2.11.0+cu130 (pulled in automatically as vLLM's pinned
  dependency)
- `torch.cuda.is_available()`: True
- CUDA version (torch-reported): 13.0
- GPU: NVIDIA GeForce RTX 5060 Ti, driver 580.159.03, CUDA 13.0 (from
  `nvidia-smi`)

## Install command

```
/home/soroush/.venvs/vllm_baseline_pilot/bin/pip install vllm
```

## Server launch command

See `tmux_server_command.txt` for the exact, byte-for-byte command,
including the environment-variable overrides. Summary of what those
overrides do and why:

- `CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13` and
  `PATH=<venv>/.../nvidia/cu13/bin:<venv>/bin:$PATH` — the pip-installed
  `nvidia-cuda-nvcc`/`ninja` packages provide working `nvcc`/`ninja`
  binaries, but they are not on `PATH` by default when the `vllm` script
  is invoked by absolute path rather than through an activated venv shell.
  Pointing `CUDA_HOME`/`PATH` at them let vLLM's JIT-compiled kernels find
  a real CUDA compiler, resolving the initial
  `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`
  failure.
- `VLLM_USE_FLASHINFER_SAMPLER=0` — an officially documented vLLM
  environment variable (`vllm/envs.py`) that disables the FlashInfer
  top-k/top-p sampling kernel. Even with `nvcc`/`ninja` found, FlashInfer's
  JIT-compiled sampling kernel failed with
  `CUDA compiler and CUDA toolkit headers are incompatible` — a version
  mismatch between the pip-installed `nvcc` (13.2.78) and the CCCL headers
  bundled inside the `flashinfer` package's own data directory. Rather than
  patching package versions (a deeper, less reversible change), the
  supported fallback flag was used, which vLLM explicitly logs a pointer to
  in its own warning message.
- `--enforce-eager` — disables `torch.compile`/CUDA-graph capture
  (`vllm.py:1062` warning), which also requires `nvcc` for its own
  Inductor-backend compilation. Necessary for the same underlying reason:
  no CUDA compiler toolchain is installed system-wide.

None of the above modifies vLLM's install, the isolated venv's packages,
or any file in this repository — they are process-local environment
variables and CLI flags only.

## Model

- `Qwen/Qwen2.5-0.5B-Instruct` — downloaded from HuggingFace Hub on first
  server start (954 MB), cached at
  `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct`. No
  authentication required (non-gated model).

## No live hosted API calls

Confirmed: the only network target for both the server and the health-check
client was `http://127.0.0.1:8001`. No Cohere/Gemini/OpenAI/Azure/
Fireworks/CloudRift SDK or endpoint was imported or reached.
