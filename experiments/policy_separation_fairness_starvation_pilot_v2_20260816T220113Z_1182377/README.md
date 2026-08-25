# Policy Separation Fairness-vs-Size Pilot v2 — provenance

**Status:** execution COMPLETE (Job 1182377); scientific analysis COMPLETE  
**Audit:** [`docs/audits/policy_separation_fairness_starvation_pilot_v2_20260816.md`](../../docs/audits/policy_separation_fairness_starvation_pilot_v2_20260816.md)  
**Design:** [`docs/design/POLICY_SEPARATION_FAMILY_A_V2.md`](../../docs/design/POLICY_SEPARATION_FAMILY_A_V2.md)  
**Verdict:** `USEFUL_BUT_NEEDS_REFINEMENT` (include in PSD; proceed to next mechanism family)  
**Git HEAD at run:** `16ad5d3e5af2e02516dfc42cc0825fa8eb7cbf38`  
**Cluster scratch (authoritative raw root):**  
`/mmfs1/scratch/ikoutis/sv96/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377`  
**Cluster worktree:**  
`/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1`  
**Slurm logs:**  
`logs/policy-separation-fairness-starvation-pilot-v2.1182377.{out,err}` in that worktree  
(copies here: `slurm_job.out.txt`, `slurm_job.err.txt`)

## Failed predecessor (do not analyze as success)

- Job **1182373** FAILED (`KeyError: request_token`) after BurstGPT path resolved but plural headers
  (`Request tokens` / `Response tokens`) were not recognized.
- Scratch retained on cluster:
  `/mmfs1/scratch/ikoutis/sv96/policy_separation_fairness_starvation_pilot_v2_20260816T215822Z_1182373`
- Fix landed in `16ad5d3` before relaunch.

## Scope

- 72 scenarios × 4 policies = **288** evaluations
- Policies: `fifo`, `estimated_service_time_first`, `aging_priority`, `weighted_fair_share`
- Config: `configs/policy_separation_fairness_starvation_pilot_v2.yaml`
- Runner: `scripts/run_policy_separation_fairness_starvation_pilot_v2.py`
- Generator: `src/llmserveopt/policy_separation/templates_fairness_starvation_v2.py`
- Analyzer: `scripts/analyze_policy_separation_fairness_starvation_pilot_v2.py`

## Integrity (execution)

- `final_summary.json`: `n_completed=288`, `n_failed=0`, primary =
  `arrival_normalized_weighted_goodput`
- `token_length_sources_observed`: `["burstgpt_staged"]` only
- `run_manifest.json` records BurstGPT path and git HEAD
- Empty stderr; `wrapper_exit_code=0`

## Metric / provenance notes

1. **Primary metric is canonical ANWG**  
   (`arrival_normalized_weighted_goodput`). Secondary
   `unweighted_slo_success_rate` is also present. There is **no** ambiguous
   `anwg` column.

2. **Token lengths are BurstGPT-anchored** from
   `BurstGPT_without_fails_1.csv` (headers `Request tokens` /
   `Response tokens`). Tenant identity, priority, and SLO remain synthetic
   interventions.

3. **Prediction noise** is a controlled factor (`0.0` accurate control,
   `0.30` moderate unbiased lognormal).

4. **Do not mix with v1 Job 1182306 rows** for selector training without an
   explicit schema/version gate. v1 used synthetic lengths and a noncanonical
   primary alias.

## Preserved files in this directory

| File | Role |
|---|---|
| `per_policy_results.csv` | 288-row results (canonical ANWG + secondaries) |
| `scenario_features.csv` / `scenarios.jsonl` | Factor + BurstGPT provenance |
| `final_summary.json` / `run_manifest.json` / `slurm_manifest.txt` | Job metadata |
| `run.log` / `run_log.txt` | Runner progress |
| `slurm_job.out.txt` / `slurm_job.err.txt` | Slurm stream copies |
| `analysis/` | Winner maps, pairwise, seed stability, surfaces |
| `README.md` | This provenance note |

## Next scientific step

Family A v2 is usable PSD evidence with documented refinements. Next WS-P step:
**design/execute the next mechanism family** (not MAP-Elites / selector retrain
from incomplete multi-family coverage). Optional Family A seed/skew refinement
may run in parallel later.
