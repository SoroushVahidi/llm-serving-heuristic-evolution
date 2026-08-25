# Wulver Handoff

Status: 2026-07-20 local prototype cleanup.

No Wulver jobs were launched in this stage. This document describes what is
ready to scale after the local smoke path and split checks are in place.

**Scope note:** this describes only local-branch (`phase2c-final-selector-improvement`)
readiness. It predates, and does not include, the Policy Library v2/
composition/structural-synthesis work already committed on
`origin/wulver-final-integration-20260721`. See
[LOCAL_BRANCH_STATUS.md](LOCAL_BRANCH_STATUS.md) for the current handoff and
the planned synchronization step.

## Ready To Scale

- Simulator execution for deployable internal policies.
- Selector Dataset v2 row schema and full policy-vector artifacts.
- Option-B 8-policy trainable selector action space.
- Causal `feat_*` feature extraction.
- Row-range split-leakage checks for transformed real-trace windows.
- Per-policy reward-regression selector prototype training.
- Existing Wulver/SLURM templates under `scripts/slurm/`.

## Remains Local-Only For Now

- `scripts/run_local_e2e_smoke.py`: use locally to check environment and
  artifact shape before submitting larger jobs.
- `results/local_e2e_smoke/*`: local generated outputs; do not commit unless
  a tiny curated artifact is explicitly needed.
- Current stale calibrated pilot held-out claims: do not scale analysis from
  the old leaky split.

## CPU-Parallel Workloads

Good Wulver CPU-array candidates:

- clean Selector v2 pilot regeneration after split fix;
- per-window policy matrix generation;
- bootstrap result aggregation;
- selector model sweeps over CPU-friendly scikit-learn models;
- synthetic/real-trace discriminative-window search;
- LLM-generated heuristic simulator evaluation after the selector dataset is
  clean.

Recommended decomposition:

```text
job array dimension 1: workload source / temporal pool
job array dimension 2: seed or row-range block
job array dimension 3: policy subset if policy-window runtime dominates
postprocess job: concatenate vectors, verify split integrity, train selectors
```

Do not aggregate until every shard has a manifest, seed, git commit, and
nonempty policy vector.

## GPU Workloads

GPU jobs are only needed for:

- real vLLM serving validation;
- GPU service-curve calibration;
- faithful runtime validation against vLLM/Sarathi-style baselines;
- any future model-serving experiment that actually launches an LLM server.

Expected A100-class jobs:

- vLLM/Sarathi repeated validation for medium/large models;
- high-context KV-pressure serving experiments;
- calibration grids that need stable GPU timing.

CPU selector training does not require a GPU.

## Rough Resource Classes

These are planning classes, not runtime promises.

| Class | Example | Expected resource |
|---|---|---|
| Local smoke | `scripts/run_local_e2e_smoke.py --max-requests 180` | laptop/desktop CPU, seconds. |
| Clean small pilot | 250-500 retained windows x 8 policies | CPU node or modest array; measure first. |
| Full selector matrix | thousands of windows x 8-20 policies | CPU array, shard by source/seed/policy. |
| Faithful external comparison | monolithic/disagg/migratory baselines | CPU-heavy simulation; possibly separate queues. |
| Real vLLM validation | live vLLM server + clients | A100 GPU node, repeated trials. |

## Required Datasets

Currently local:

- BurstGPT raw CSV and processed variants.
- Azure LLM 2023 code/conversation raw and processed variants.

Not acquired / do not assume present:

- ShareGPT raw file.
- Azure 2024/2025 traces.
- TraceLab.
- ServeGen.
- Mooncake/Kimi traces.

Before Wulver submission, copy or stage required datasets explicitly and record
checksums. Do not make SLURM jobs download large datasets implicitly.

## Environment Setup

Baseline local setup:

```bash
python3 -m pip install -e ".[dev]"
python3 -c "import pandas, sklearn"
python3 -m pytest -q tests/test_local_e2e_smoke.py tests/test_selector_dataset_v2.py
```

For GPU validation, use the existing GPU environment docs and scripts:

- `docs/current/REPRODUCIBILITY.md`
- `docs/gpu_environment.md`
- `configs/calibration/wulver_a100_qwen25_7b_vllm024.yaml`
- `scripts/slurm/wulver_vllm_env_smoke.sbatch`

## Commands To Wrap In SLURM Later

Local preflight:

```bash
python3 scripts/run_local_e2e_smoke.py \
  --output-dir results/local_e2e_smoke/preflight \
  --max-requests 180 \
  --window-size 20 \
  --policies fifo edf scorpio_style_slo_guard weighted_shortest_processing
```

Clean calibrated pilot, after choosing scale parameters:

```bash
python3 scripts/build_selector_dataset_v2_calibrated_targeted_pilot.py \
  --output-dir results/selector_v2_clean_pilot/<run_id> \
  --target-min-retained 250 \
  --target-max-retained 500
```

Independent leakage audit:

```bash
python3 scripts/audit_selector_v2_calibrated_pilot_leakage.py \
  --pilot-dir results/selector_v2_clean_pilot/<run_id>
```

Selector prototype training, only if gates and audit pass:

```bash
python3 scripts/evaluate_selector_v2_clean_pilot.py \
  --pilot-dir results/selector_v2_clean_pilot/<run_id>
```

(`scripts/train_selector_v2_calibrated_prototype.py` is superseded by this
script -- see its own docstring -- and is retained only for reproducing
prior results computed with it.)

## What Not To Run Yet

- Do not run large Selector v2 generation from the stale leaky artifact.
- Do not compare Selector v2 against faithful external baselines until a clean
  selector pilot has passed the stricter split audit.
- Do not run real-vLLM selector superiority experiments until the simulator
  selector action space and workload plan are frozen.
- Do not claim causal/off-policy validity from simulator counterfactuals or
  non-random logged data.
- Do not use GPU nodes for ordinary scikit-learn selector training.

## Immediate Wulver-Ready Checklist

1. Run local smoke successfully.
2. Choose clean pilot target size and shard plan.
3. Stage BurstGPT/Azure processed traces with checksums.
4. Submit CPU pilot shards.
5. Concatenate policy vectors.
6. Run `audit_selector_v2_calibrated_pilot_leakage.py`.
7. Train selector only if quality gates and audit pass.
8. Decide whether GPU real-serving validation is justified by simulator
   results.
