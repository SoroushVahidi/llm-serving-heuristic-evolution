# Repository Map

Roles of the top-level paths. "Current" means the path is used by the current
study (see [`current/README.md`](current/README.md)). "Historical" means it
belongs to earlier research lines and is kept for provenance.

## Top level

| Path | Role | Current study? |
|---|---|---|
| `README.md`, `REPRODUCIBILITY.md` | Public overview and reproducibility entry point | Current |
| `paper/` | Canonical manuscript PDF. `performance_evaluation/` holds source, figures, scripts, claim manifest and the `submission/` package. `history/` holds the superseded FGCS and withdrawn LLM 2026 manuscripts | Current (`history/` historical) |
| `release/` | Reproducibility archives (v1.1.0 current, v1.0.0 superseded); see [`../release/README.md`](../release/README.md) | Current |
| `experiments/` | Committed experiment artifacts and reports, one directory per study | Mixed; the current study's directories are listed in [`current/README.md`](current/README.md) |
| `src/llmserveopt/` | Library: `simulator/`, `policies/`, `policy_separation/` (trace replay used by the current study), plus selector, DSL, composition and workload code | Mixed |
| `scripts/` | Experiment runners, analysis and maintenance scripts; see [`../scripts/README.md`](../scripts/README.md) | Mixed |
| `tests/` | pytest suite. CI runs the CPU-only subset in `.github/workflows/ci.yml` | Mixed |
| `configs/` | Experiment and calibration configs | Mostly historical |
| `baselines/`, `external/` | External-scheduler adapters and provenance notes | Historical |
| `benchmarks/` | Workload suite definitions for the selector studies | Historical |
| `data/` | Dataset metadata. Raw traces and derived parquet windows are not committed; see [`DATA_RELEASE_POLICY.md`](DATA_RELEASE_POLICY.md) | Metadata only |
| `results/` | Local generated outputs; gitignored except a few provenance files | Local |
| `tools/` | Maintenance and cluster helpers; see [`../tools/README.md`](../tools/README.md) | Historical |
| `docs/` | Documentation; see [`README.md`](README.md) for the authority order | Mixed |
| `p2_config.yaml`, `p3_chunk_control.py`, `p5_analysis_chunk_comp.py`, `p7_runner.py`, `p8_test_runner.py` | Self-contained Family B v2 prefill-control composition falsification experiment (2026-08-17). Its audit (`docs/audits/family_b_v2_prefill_control_composition_falsification_20260817.md`) and `scripts/smoke_prefill_control_composition_v2.py` use these files from the repository root, so they stay there | Historical |

## Current study: code and artifacts

| What | Where |
|---|---|
| Simulator | `src/llmserveopt/simulator/` |
| Policies (reference: `kv_constrained_online.py`) | `src/llmserveopt/policies/` |
| Replay configuration (1 ms step, service model) | `src/llmserveopt/policy_separation/public_trace_replay_v1.py` |
| Runners | `scripts/industry_realism_action_opportunity_phase_a_v1.py`, `scripts/industry_realism_action_opportunity_phase_b_v2.py`, `scripts/fresh_latency_causal_confirmatory_v1.py` |
| Artifacts | `experiments/industry_realism_action_opportunity_phase_a_v1/`, `..._phase_b_v2/`, `experiments/fresh_production_latency_headroom_confirmatory_v1{,_robustness,_corrected}/`, `experiments/reference_reserve_sensitivity_v1/`, `experiments/real_vllm_*` |
| Manuscript checks | `paper/performance_evaluation/scripts/build_claim_manifest.py --check` |

## Documentation tiers

See [`README.md`](README.md). In short: public entry points, then the
current-study index [`current/README.md`](current/README.md), then
study-specific documents, then historical material. Most files in
`docs/current/` and all files in `docs/audits/` are dated records.
