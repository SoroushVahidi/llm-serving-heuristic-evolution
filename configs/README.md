# Config Index

Configs are intentionally not mass-moved in this cleanup pass. Many historical
scripts and audits reference exact paths. Use this file to choose the right
family before running a script.

## Current Contextual-Composition Work

- `cc1_composition_opportunity*.yaml`
- `cc4_oracle_composition_dataset.yaml`
- `cc4b_oracle_composition_expansion.yaml`

These support the CC-track experiments. Check `docs/PROJECT_MAP.md` and the
matching `docs/audits/contextual_composition_*` file before rerunning.

## Apt-Serve

- `configs/examples/apt_serve/*.yaml` - schema/config examples and validation fixtures.
- Phase G collection uses code-generated configuration in
  `scripts/run_apt_serve_phase_g.py` plus run manifests under `results/`.

The current canonical Phase G analysis is
`results/apt_serve_phase_g_analysis_20260809_190000/`.

## External Baselines

- `configs/stress_tests/algorithm_stress_test_catalog.yaml`
- `configs/stress_tests/generated/{distserve,llumnix,sarathi}/`
- `configs/workload_headroom_candidates/*.yaml`
- `configs/oracle/tiny_oracle_srtf_smoke.yaml`

Generated stress-test JSON files are tracked because they are compact
reproducibility fixtures, not disposable local results.

## Phase 2 Historical

- `phase2b*.yaml`
- `phase2c*.yaml`
- `selector/*.yaml`
- `llm_generation/*.yaml`

These are historical reproducibility configs. Retain them unless a dedicated
archive pass updates all references.

## Calibration

- `gpu_calibration/*.yaml`
- `calibration/*.yaml`
- `real_llm_latency/*.yaml`
- `api_calibration/*.yaml`

Some calibration configs require local GPU, hosted API, or Wulver-specific
environments. Check the script help and audit docs first.

## Real-System / Trace Workflows

- `real_trace/*.yaml`
- `traces/*.yaml`
- `default_simulator.yaml`
- `small_debug.yaml`
- `baseline_comparison.yaml`
- workload comparison configs such as `prefill_heavy_comparison.yaml`,
  `decode_heavy_comparison.yaml`, and `mixed_slo_comparison.yaml`.

## Legacy Marker

Legacy configs are preserved for reproducibility. A config being old does not
make it safe to delete; deletion should happen only after confirming that the
corresponding audit/result is either archived elsewhere or no longer needed.
