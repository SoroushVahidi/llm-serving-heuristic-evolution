# Policy Separation Fairness and Starvation Pilot v1 — provenance

**Status:** execution COMPLETE (Job 1182306); scientific analysis PENDING  
**Git HEAD at run:** `8b0fc6c7a88d5a596e33ae1088936f659ad1ee63`  
**Cluster scratch (authoritative raw root):**  
`/mmfs1/scratch/ikoutis/sv96/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306`  
**Cluster worktree:**  
`/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1`  
**Slurm logs:**  
`logs/policy-separation-fairness-starvation-pilot-v1.1182306.{out,err}` in that worktree  
(copies here: `slurm_job.out.txt`, `slurm_job.err.txt`)

## Scope

- 120 scenarios × 4 policies = **480** evaluations
- Policies: `fifo`, `estimated_service_time_first`, `aging_priority`, `weighted_fair_share`
- Config: `configs/policy_separation_fairness_starvation_pilot_v1.yaml`
- Runner at run time: `scripts/run_policy_separation_fairness_starvation_pilot_v1.py`
- Generator: `src/llmserveopt/policy_separation/templates_fairness_starvation.py`

## Integrity (execution)

- `final_summary.json`: `n_completed=480`, `n_failed=0`, `wrapper_exit_code=0`
- No silent task failures recorded in `run_log.txt` / scratch `run.log`

## Critical metric / provenance caveats (do not overclaim)

1. **Token lengths were synthetic**, not BurstGPT-anchored.  
   At job time the generator looked for `BurstGPT_without_fails.csv`, which was
   absent; staged shards are `BurstGPT_without_fails_{1,2,3}.csv`. The run
   therefore used `synthetic_lognormal_fallback`. Post-job code now discovers
   numbered shards; **do not back-label Job 1182306 as BurstGPT-anchored**.

2. **Column `anwg` in `per_policy_results.csv` is NOT canonical ANWG.**  
   It is the unweighted SLO-success fraction
   `(completed_without_violation) / n_loaded`.  
   Canonical `RunMetrics.arrival_normalized_weighted_goodput` was **not**
   written by this historical runner. Future corrected runs emit
   `unweighted_slo_success_rate` and `arrival_normalized_weighted_goodput`
   explicitly; this CSV is frozen historical evidence.

3. **Perfect size predictions:** `predicted_output_tokens == actual_output_tokens`.  
   Tenant identity is carried via synthetic `class_id` / `priority` interventions.

## Preserved files in this directory

| File | Role |
|---|---|
| `per_policy_results.csv` | Historical 480-row results (`anwg` = unweighted SLO success) |
| `final_summary.json` | Job-level counts / timing |
| `slurm_manifest.txt` | Host, SHA, git status, Python |
| `run_log.txt` | Runner progress (copy of gitignored `run.log`) |
| `slurm_job.out.txt` / `slurm_job.err.txt` | Slurm stream copies |
| `README.md` | This provenance note |

## Next scientific step

Analyze this frozen corpus (crossover / boundary / hypothesis checks). Do **not**
silently rewrite these rows. Any BurstGPT-path or canonical-ANWG correction
requires a **new** run id and a separate audit.
