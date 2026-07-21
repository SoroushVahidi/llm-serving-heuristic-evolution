# Active Experiment Protected Paths

Generated during Query 1 cleanup audit on 2026-07-21 and refreshed during Query 3 final validation.

## Rule

Do not delete, move, rename, compress, rewrite, or otherwise modify any path in this document while the associated SLURM workflow is active or has dependent jobs pending.

## Active Project Jobs

| Workflow | Job IDs | Current state | Protected experiment root |
| --- | --- | --- | --- |
| Policy Frontier Cartography and Adversarial Discriminative Workload Mining | 1118187, 1118188, 1118189, 1118190, 1118191, 1118192, 1118193, 1118194, 1118195, 1118196, 1118197 | `1118187` broad array running/pending; targeted array and downstream jobs pending on dependencies | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z` |
| Policy Library v2 Expanded Frontier | 1118784, 1118785, 1118786, 1118787, 1118788, 1118789 | `1118784` array running/pending; downstream jobs pending on dependencies | `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z` |

## Status Snapshot

This snapshot was taken from `squeue` and `sacct`, not inferred from missing queue entries.

- Policy Frontier broad sweep: 36 tasks completed, tasks `36-39` running, tasks `40-63` pending under the array concurrency limit.
- Policy Frontier targeted sweep: tasks `0-31` pending on dependencies.
- Policy Frontier downstream stages: combine, boundary, active mining, QD, coverage, augmentation, representation, evolutionary archive, and report jobs pending on dependencies.
- Policy Library v2 array: 15 tasks completed, tasks `13`, `15`, `17`, and `18` running, tasks `19-31` pending under the array concurrency limit.
- Policy Library v2 downstream stages: combine, complementarity, selector comparison, composition readiness, and report jobs pending on dependencies.
- Native Composition Pilot job `1120123` completed successfully and is no longer active.
- Structural Synthesis test job `1120181` completed successfully and is no longer active.

## Protected Subpaths

Treat these subtrees as protected because active jobs may still write logs, shards, manifests, or final reports there:

- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z/broad_sweep/`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z/targeted_sweep/`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z/logs/`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_frontier_cartography_20260721T154408Z/reports/`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z/shards/`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z/logs/`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z/reports/`
- `/mmfs1/project/ikoutis/sv96/llmserveopt-data/policy_library_v2_expanded_20260721T171933Z/manifests/`

## Inspection Commands

```bash
squeue -u "$USER" -o '%i|%j|%T|%M|%D|%R'
sacct -j 1118186,1118187,1118188,1118189,1118190,1118191,1118192,1118193,1118194,1118195,1118196,1118197 --format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS -P
sacct -j 1118781,1118782,1118783,1118784,1118785,1118786,1118787,1118788,1118789 --format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS -P
```
