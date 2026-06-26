# Experiment Tracking

This document explains the machine-readable registries used to track experiments, failure cases, and external API usage across all research phases.

## Directory layout

```
docs/templates/                         ← committed templates (gitignored results/ excluded)
docs/audits/phase2b9_*_summary.md       ← committed Phase 2B.9 result summaries (full CSVs gitignored)
    experiment_registry_template.csv
    failure_case_registry_template.csv
    api_usage_ledger_template.csv

results/                                ← gitignored; holds actual populated CSVs
    experiment_registry/
        experiment_registry.csv         ← copy of template, then filled in
    failure_cases/
        failure_case_registry.csv
    api_usage/
        api_usage_ledger.csv
    phase2b6_fair_sweep_failure_audit/  ← raw sweep outputs
        ...
```

## Registries

### Experiment Registry (`experiment_registry.csv`)

One row per experiment run. Key fields:

| Field | Description |
|---|---|
| `experiment_id` | Unique string identifier |
| `phase` | Research phase, e.g. "2B.6" |
| `config_file` | Config YAML path from repo root |
| `workload_tags` | Comma-separated workload identifiers |
| `policies` | Number of policies compared |
| `selector_wg` | Rule-based selector mean weighted goodput |
| `best_fixed_policy` | Name of best single deployable policy |
| `delta_vs_best_fixed` | `selector_wg − best_fixed_wg` |
| `run_date` | ISO date |

See `docs/templates/experiment_registry_template.csv` for full schema.

### Failure Case Registry (`failure_case_registry.csv`)

One row per selector failure — windows where the selector underperformed the best fixed policy.

| Field | Description |
|---|---|
| `failure_id` | Unique identifier |
| `experiment_id` | FK → experiment registry |
| `workload_tag` | Workload that produced the failure |
| `selector_predicted` | Policy the rule selector chose |
| `delta` | `selector_wg − best_fixed_wg` (negative = failure) |
| `failure_category` | Taxonomy label (see below) |
| `root_cause_hypothesis` | Concise explanation |
| `llm_escalated` | Whether sent to CloudRift/LLM for deeper analysis |

#### Failure taxonomy

| Category | Meaning |
|---|---|
| `wrong_rule_fired` | A rule fired when a different rule would have been better |
| `default_fallback_suboptimal` | Rule 7 (EDF default) fired but EDF was not optimal |
| `feature_not_captured` | Workload characteristic not represented in any feature |
| `admission_drop_hurts` | `admission_control` was selected but drop losses > scheduling gains |
| `n/a` | Not a genuine failure (within noise) |

See `docs/templates/failure_case_registry_template.csv` for full schema.

### API Usage Ledger (`api_usage_ledger.csv`)

One row per paid API call. This ledger ensures all external API usage is traceable, auditable, and bounded.

**Rule:** No API call appears in any test, CI job, or default script execution. All calls must be explicitly triggered (e.g., `--use-llm` flag or dedicated generation scripts). Each call must have a ledger entry.

See `docs/templates/api_usage_ledger_template.csv` for full schema.

## How to populate

1. Copy the template to the corresponding `results/` subdirectory.
2. Fill in rows after each experiment or API call.
3. Never edit templates — they must remain clean for new experiments.

## Invariants

- `results/experiment_registry/` and `results/failure_cases/` are gitignored; only templates are committed.
- The `api_usage_ledger.csv` in `results/api_usage/` is also gitignored.
- Summaries extracted from results (e.g., `docs/audits/`) are committed.

## Phase 2B.13 outputs (gitignored raw, committed summaries)

- **Raw results:** `results/phase2b13_selector_training_and_suspicion_audit/`
- **Committed summary:** `docs/audits/phase2b13_selector_training_and_suspicion_audit_summary.md`
- **Failure cases:** `docs/audits/phase2b13_failure_cases_summary.md` + `failure_cases.csv` in results
- **tmux session:** `phase2b13_selector_training`
- **Log:** `logs/phase2b13/phase2b13_selector_training.log`

Key runner outputs: `selector_comparison.csv`, `always_scorpio_comparison.csv`,
`near_tie_summary.csv`, `rf_dt_training_summary.json`, `regret_weighted_training_summary.json`.
