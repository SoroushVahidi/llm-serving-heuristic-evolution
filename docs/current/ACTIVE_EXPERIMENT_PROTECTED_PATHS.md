# Active Experiment Protected Paths

Refreshed 2026-08-19 (previously refreshed 2026-07-22). Local-machine section
added; cluster section below is carried forward from the 2026-07-22 pass and
has not been independently re-verified against a fresh `squeue` this pass.

## Rule

Do not delete, move, rename, compress, rewrite, or otherwise modify an
experiment root while a SLURM workflow is active or has dependent jobs that may
still write there. Completed roots are no longer active writer paths, but they
remain scientific evidence and should still be treated as read-only provenance
unless a task explicitly creates a derived copy elsewhere.

## Current Active Writers (local machine, 2026-08-19)

None. No tmux session, and no python/pytest/opencode process with this
repository as its working directory, was found anywhere on the local machine
as of this refresh (the machine rebooted 2026-08-19 18:42:04 EDT, which
independently terminated anything that had been running before that).

## Current Active Writers (cluster, carried forward from 2026-07-22)

No currently running project jobs were observed during the 2026-07-22 polish
audit; not independently re-checked this pass.

`squeue -u sv96` did show superseded dependency-blocked jobs from earlier
SLO/SwissAI startup attempts, including old `slo_aug_*` chains and
`1127600 swissai_v2_report`. These are not expected to produce additional
scientific outputs because their dependencies are unsatisfied/superseded. They
must still not be cancelled or modified by repository-cleanup tasks.

## Frozen / Do Not Modify Casually (local, current lineage)

These are complete, frozen evidence artifacts from the MF-PSD → hierarchical
router lineage. Read-only; regenerate via their documented builder/runner
rather than hand-editing if a defect is ever found.

| Path | Why protected |
| --- | --- |
| `experiments/mf_psd_v1/` | Canonical `MF_PSD_READY` unified dataset (Step 1) |
| `experiments/unified_utility_matrix_v1/`, `experiments/unified_utility_matrix_v2/` | Frozen `UNIFIED_UTILITY_MATRIX_READY` Step-2 matrices (v1 superseded-but-preserved, v2 canonical) |
| `experiments/family_c_reconstruction_v1/` | `FAMILY_C_RECONSTRUCTION_BOUNDED` canonical reconstructed Family-C ground truth |
| `experiments/multifamily_contextual_selector_v1/`, `experiments/shared_cross_family_features_v1/`, `experiments/mechanism_choice_target_feasibility_v1/`, `experiments/cross_family_transfer_wellposedness_reassessment_v1/` | The three independent NO_GO verdicts and the reassessment that demoted cross-family transfer |
| `experiments/online_regime_signal_feasibility_v1/` | `ONLINE_REGIME_SIGNALS_READY`; includes the ~29.9MB canonical per-step telemetry CSV — the largest tracked file in the repo, intentional |
| `experiments/hierarchical_regime_router_v1_test_evaluation/`, `experiments/hierarchical_router_live_harness_v1_smoke/`, `experiments/hierarchical_regime_router_live_reeval_v1/` | Original hierarchical-router TEST result/audit, live-harness validation, and the formal live re-evaluation (`HIERARCHICAL_ROUTER_NO_GO`) |
| `configs/hierarchical_regime_router_v1_gates.json` | Frozen gate thresholds used by the live re-evaluation's formal gate-rescoring |
| `docs/design/FAMILY_B_BALANCED_REPLICATION_V1.md`, `experiments/family_b_balanced_replication_v1/frozen_scenario_selection_v1.json` | Frozen Family-B replication design and the frozen 36-scenario selection; the run *outputs* (`run_smoke_*` files) are not frozen — they are expected to be rewritten by re-running the smoke sources |
| `data/public_trace_corpus_v1/manifest.json`, `data/public_trace_corpus_v1/schema.json`, `data/public_trace_corpus_v1/distribution_stats.json`, `data/public_trace_corpus_v1/source_coverage.csv` | Public Trace Corpus v1 canonical artifacts, committed at `84fa31b`/`179a6fe` |
| All `docs/audits/**` | Immutable point-in-time audit trail; never edited after the fact |
| All `docs/design/**` once committed | Frozen preregistrations |

## Decision-Criticality / Timescale Analysis — Committed, Experiment Not Yet Run

Design, implementation, and tests are complete, tracked, and pushed (commit
`4dac220`, "feat: preregister decision-criticality train-val study") — they
are no longer local/uncommitted. The protection here is scoped narrowly: do
not launch the actual TRAIN/VAL experiment without separate explicit
authorization (no `experiments/decision_criticality_timescale_trainval_v1/`
output exists yet), and do not casually edit this frozen preregistration:

- `docs/design/DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md`
- `scripts/run_decision_criticality_timescale_trainval_v1.py`
- `src/llmserveopt/analysis/`
- `tests/test_decision_criticality_timescale_trainval_v1.py`

## Third-Party / Raw Trace Data

Raw downloaded/staged source caches are not to be committed or mirrored
elsewhere unless their license explicitly permits redistribution:

- `data/raw/burstgpt/BurstGPT_1.csv` (MIT — present locally, regeneratable, do not silently overwrite)
- `data/raw/azure/*.csv` (CC-BY-4.0 — present locally, regeneratable, do not silently overwrite)
- Mooncake data specifically: license `NOT_EXPLICITLY_SPECIFIED` — internal-only, **not redistributable**, per `docs/PROJECT_MAP.md`'s explicit warning
- `data/raw/agentperfbench/` — classified `REAL_SYSTEM_VALIDATION_SOURCE`, not ingested into the workload-input corpus; not a license-cleared redistribution target

Transient local logs (`logs/`, `run.log`, `crash.log`, `*.log`) are
gitignored generated output, not frozen science — they are not listed here
even though some are cited as evidence in audit docs (the audit doc citing
them is the frozen artifact; the log itself is not).

## Completed Evidence Roots To Treat As Read-Only

| Workflow | Root | Status note |
| --- | --- | --- |
| Policy Separation Family A Fairness/Starvation Pilot | `/mmfs1/scratch/ikoutis/sv96/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306` (+ Git copy under `experiments/…_1182306/`) | Complete execution; analysis pending; treat as read-only provenance. |
| Policy Separation Sobol Pilot v1 | `/mmfs1/scratch/ikoutis/sv96/policy_separation_sobol_pilot_20260816T183600Z_1182183` (+ Git copy under `experiments/…_1182183/`) | Complete and analyzed; read-only. |
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
