# Scripts Index

Run scripts from the repository root. Use `python <script> --help` before
rerunning any experiment. Long-running jobs should be launched in tmux or a
cluster scheduler with wrapper metadata.

## Current Primary Runners

| Script | Purpose | Help | Input | Output | Resumable | Status |
|---|---|---|---|---|---|---|
| `scripts/run_apt_serve_phase_g.py` | Apt-Serve Phase G collection | `python scripts/run_apt_serve_phase_g.py --help` | generated Phase G workload grid / optional resume dir | `results/apt_serve_phase_g_*` | Yes | collection complete; rerun only for missing/invalid cells |
| `scripts/analyze_apt_serve_phase_g.py` | Phase G posthoc analysis | `python scripts/analyze_apt_serve_phase_g.py --help` | completed Phase G run dir | `results/apt_serve_phase_g_analysis_*` | Yes | canonical run complete at `results/apt_serve_phase_g_analysis_20260809_190000/` |
| `scripts/check_project_handoff_consistency.py` | Current documentation/status consistency check | no args required | docs | stdout | N/A | active maintenance |
| `scripts/check_contextual_composition_status.py` | Historical CC roadmap/status checks | `python scripts/check_contextual_composition_status.py --help` | CC docs | stdout | N/A | supporting/historical |
| `scripts/smoke_test.py` | Fast simulator sanity check | `python scripts/smoke_test.py --help` | none | stdout | N/A | active |

## Analysis

- `analyze_apt_serve_phase_g.py` - active Phase G analysis runner.
- `analyze_repeated_trials.py` - Sarathi/vLLM repeated-trial postprocessing.
- `analyze_selector_dataset_v2_pilot.py` - historical selector-v2 pilot analysis.
- `audit_selector_objectives.py`, `audit_selector_v2_calibrated_pilot_leakage.py` - metric/leakage audits.

## Contextual-Composition Experiments

- `run_cc1_composition_opportunity.py`
- `run_cc4_oracle_composition_dataset.py`
- `run_cc5_contextual_predictor.py`
- `run_cc5_final_operating_envelope.py`
- `run_cc5_uncertainty_regime_refinement.py`

These are research-phase runners. Check `docs/PROJECT_MAP.md` and the dated CC
audit before rerunning.

## Apt-Serve

- `run_apt_serve_headroom_check.py` - Phase F headroom check; historical after Phase G.
- `run_apt_serve_phase_g.py` - Phase G collection runner.
- `analyze_apt_serve_phase_g.py` - Phase G posthoc analysis runner.
- `scripts/apt_serve/apt_serve_scheduler_worker.py` - worker protocol implementation.
- `scripts/apt_serve/fake_scheduler_worker.py` - deterministic fake worker for tests and Phase G.

## External Baselines

- `run_pars_first_comparative_evaluation.py`
- `run_vllm_ltr_first_comparative_evaluation.py`
- `run_vtc_fairness_comparative_sweep.py`
- `run_llumnix_comparative_evaluation.py`
- `run_distserve_comparative_evaluation.py`
- `run_gpu_external_validity_audit.py`
- `run_sarathi_gpu_smoke_and_validation.py`
- `compare_sarathi_vllm_matched_runtime.py`

These scripts are provenance-bearing. Prefer reading the matching audit before
rerunning.

## Data / Calibration

- `scripts/data/*` - dataset download, conversion, validation, and real-window construction.
- `run_gpu_calibration.py`, `fit_service_curves.py`, `validate_simulator_calibration.py`
- `fit_real_llm_latency_model.py`, `fit_real_llm_latency_model_v2.py`
- `_run_cohere_v2_live_pilot.sh`, `_run_gemini_v2_live_pilot.sh` - paid live API launchers; require explicit opt-in.

## Maintenance / Status

- `build_baseline_tables.py`
- `check_cc4b_quality_gates.py`
- `check_ordering_workload_headroom.py`
- `check_project_handoff_consistency.py`
- `report_research_status.py`
- `verify_*` scripts

## Historical Phase 2

`run_phase2b*.py`, `run_phase2c*.py`, and older selector dataset builders are
retained for reproducibility. Do not delete them casually; prefer adding a
banner or audit pointer if one is misleading.

## Slurm / Wulver

- `scripts/slurm/*.sbatch` - cluster launch templates and historical job scripts.
- `scripts/wulver_probes/*` - Wulver-side probes where present.

These often include Wulver-specific paths. Treat them as cluster templates or
provenance-bearing scripts, not portable local runners unless explicitly
parameterized.

## Debug / Development

One-off helpers and plot/report scripts remain at the top level. Broad physical
moves are intentionally deferred because many docs/tests reference exact paths.
