# Reproducibility (Canonical)

## Python environment (read this first)

The bare `pytest`/`python` resolved from `PATH` on a typical dev machine may
**not** be the right interpreter -- it can be missing `pandas`/`pyarrow` and
will silently drop test files with collection errors rather than failing
loudly. **Always invoke via `python3 -m pytest`, not bare `pytest`**, and
confirm `python3 -c "import pandas"` succeeds in whatever interpreter you use.

```bash
pip install -e ".[dev]"          # editable install, pulls pyproject.toml deps
python3 -m pytest --collect-only -q   # should report ~2,500 tests, 0 errors
```

Dependencies are declared with loose `>=` bounds in `pyproject.toml` (no
pinned lockfile as of this writing) -- `numpy`, `pandas`, `pyyaml`,
`matplotlib`, `tabulate`; `pytest`/`pytest-cov` for `[dev]`;
`transformers` for `[datasets]`.

## Running tests

```bash
python3 -m pytest                 # full non-GPU-safe suite
python3 -m pytest -m gpu          # GPU-only tests (requires a CUDA-capable GPU)
python3 -m pytest --collect-only -q   # count/list without running
```

GPU tests are properly marked (`@pytest.mark.gpu`) and skip cleanly without
CUDA -- 2 files, 8 tests, as of this writing.

## Local GPU work

- Hardware: RTX 5060 Ti, calibrated against Qwen2.5-0.5B (prefill MAPE 9%,
  decode MAPE 12%). `docs/gpu_calibration.md`, `docs/gpu_environment.md`.
- Long-running local work (real-vLLM-server pilots, calibration sweeps) is
  launched under `tmux` so it survives a disconnect -- see the tmux session
  naming convention in past pilots (`vllm-server-healthcheck`,
  `vllm-scaled-baseline-comparison`, etc.) for the pattern to follow.
- A live example: a `vllm serve` healthcheck process has been running since
  2026-07-03 in this repo's working directory on port 8001 -- see
  [PROJECT_STATUS.md](PROJECT_STATUS.md) §8 before assuming port 8001 is
  free on this machine.

## Wulver A100 cluster work

- Requires SLURM access to the specific HPC account referenced in
  `scripts/slurm/*.sbatch` (13 job scripts, one per experiment) --
  hard-coded scratch paths and account strings are expected/appropriate for
  a fixed HPC allocation, not portable to a different account without
  editing.
- Workflow: submit via `sbatch`, results land under the job's scratch
  output directory, then get pulled back and reconciled into
  `docs/wulver_*.md` + `experiments/runtime_validation_benchmark_pack/`.
  See `docs/wulver_gpu_validation_handoff.md` for the handoff protocol.
- This is `REPRODUCIBLE_WITH_EXTERNAL_RESOURCES` only -- it requires that
  specific cluster access and is not runnable generically.

## Real-LLM API pilots (Cohere, Gemini)

- `NOT_REPRODUCIBLE` for free -- these fire real, billed API requests.
- The two manual one-off launchers
  (`scripts/_run_cohere_v2_live_pilot.sh`, `scripts/_run_gemini_v2_live_pilot.sh`)
  require `LLMSERVEOPT_ALLOW_PAID_API_CALLS=1` set explicitly before they
  will do anything -- this is an intentional safety gate, not a bug. See
  each script's own header comment.
- No credentials are stored in this repository; see `.env.example` and
  `docs/api_provider_setup.md` for the credential-setup process (never
  commit `.env` or real keys).

## Sync / large-artifact transfer

- GitHub is the primary sync path for code, small committed experiment
  artifacts (`experiments/`), and documentation.
- `results/`, `logs/`, and raw `data/` are gitignored by design (see
  [EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md)) -- when a raw
  artifact from one of these needs to move between machines (e.g. off the
  Wulver cluster), use `scp`/cluster-specific transfer tooling rather than
  committing it.

## Reproducibility matrix

| Component | Classification |
|---|---|
| Core simulator experiments (synthetic configs) | `FULLY_REPRODUCIBLE` -- seeded via `numpy.random.default_rng`, same config + seed = identical results |
| Selector Dataset v2 generation | `FULLY_REPRODUCIBLE` (seeded, scripted); the calibrated targeted pilot specifically has a **confirmed** split-construction leakage bug (independently audited, not just suspected) that must be fixed before its next run's VALIDATION/ID_TEST splits can be trusted -- see [SELECTOR_V2.md](SELECTOR_V2.md) §10 |
| BurstGPT / Azure 2023 real-trace replay | `FULLY_REPRODUCIBLE` -- data already present under `data/` |
| Wulver A100 validation | `REPRODUCIBLE_WITH_EXTERNAL_RESOURCES` -- requires that specific SLURM allocation |
| Sarathi/vLLM real-runtime validation | `REPRODUCIBLE_WITH_EXTERNAL_RESOURCES` -- requires GPU hardware access (local RTX or Wulver A100) |
| Real-LLM API pilots (Cohere/Gemini) | `NOT_REPRODUCIBLE` for free -- billed API calls, explicit opt-in required |
