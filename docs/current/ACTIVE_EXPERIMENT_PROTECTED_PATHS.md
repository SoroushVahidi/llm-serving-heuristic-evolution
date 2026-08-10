# Active Experiment Protected Paths

Refreshed during the 2026-07-22 repository polish pass.

## Rule

Do not delete, move, rename, compress, rewrite, or otherwise modify an
experiment root while a SLURM workflow is active or has dependent jobs that may
still write there. Completed roots are no longer active writer paths, but they
remain scientific evidence and should still be treated as read-only provenance
unless a task explicitly creates a derived copy elsewhere.

## Current Active Writers

No currently running project jobs were observed during this polish audit.

`squeue -u sv96` did show superseded dependency-blocked jobs from earlier
SLO/SwissAI startup attempts, including old `slo_aug_*` chains and
`1127600 swissai_v2_report`. These are not expected to produce additional
scientific outputs because their dependencies are unsatisfied/superseded. They
must still not be cancelled or modified by repository-cleanup tasks.

## Completed Evidence Roots To Treat As Read-Only

| Workflow | Root | Status note |
| --- | --- | --- |
| Policy Frontier Cartography | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z` | Complete; final report exists. |
| Policy Library V2 Expanded Frontier | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z` | Complete; final report exists. |
| V2 Real-OOD 27-Policy Library Audit | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/v2_real_ood_library_20260721T222521Z` | Complete; strong V2 oracle-envelope expansion. |
| V2 Selector/Regret Benchmark | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/v2_selector_regret_benchmark_20260722T134925Z` | Complete; useful suitability signal but OOD oracle-gap caveat. |
| Module Intervention / Structural Credit | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/module_intervention_credit_20260721T224322Z` | Complete; sparse positive single-module transfer. |
| SwissAI Staging | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/swissai_trace_staging_20260722T172215Z` | Complete; novel KV/cache/reuse features. |
| SwissAI V2 Policy Sweep | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/swissai_v2_policy_sweep_20260722T184451Z` | Policy matrix complete; report repaired posthoc from causal features. Original root remains read-only. |
| TraceLab Staging | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/tracelab_staging_20260722T192050Z` | Complete; long-context/agentic novelty. |
| TraceLab V2 Policy Sweep | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/tracelab_v2_policy_sweep_20260722T214129Z` | Complete; reward saturation and zero V2 gain. |
| SLO/Deadline Augmented V2 Sweep | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/slo_deadline_augmented_v2_sweep_20260722T194529Z` | Complete corrected chain; superseded failures retained for provenance. |
| Simulator Discriminative-Power Audit | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/simulator_discriminative_audit_20260722T223236Z` | Complete; current bottleneck evidence. |

## Inspection Commands

```bash
squeue -u "$USER" -o '%i|%j|%T|%M|%D|%R'
sacct -u "$USER" --starttime 2026-07-22 --format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS -P
```
