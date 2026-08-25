# Decision criticality uncertainty (existing data only) v1

**Purpose.** Analysis-only reanalysis of the frozen terminal-ANWG decision-criticality
artifacts under `experiments/decision_criticality_terminal_anwg_v1/`.

**Non-goals.** This directory does **not** re-run the simulator, generate workloads,
retrain schedulers, or modify the original criticality experiment outputs.

**Method.** Deterministic scenario-clustered bootstrap:
- resampling unit = `canonical_scenario_id` over the 144 parent TRAIN/VAL scenarios;
- each draw carries all acquired states belonging to that scenario (multiplicity preserved);
- seed `202608251`, `B = 10_000` percentile bootstrap resamples;
- 95% CIs = empirical 2.5/97.5 percentiles;
- single-class bootstrap samples are skipped for AUROC/AUPRC (count reported).

**Primary outputs.**
- `reanalyze_uncertainty_existing_data_v1.py` — reproducible analysis script
- `summary.json` — point estimates + CIs
- `bootstrap_ci_summary.csv` — compact CI table
- `source_artifact_sha256.txt` — hashes of frozen inputs at analysis time
