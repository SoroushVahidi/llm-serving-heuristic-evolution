# Local Artifact Cleanup Plan (not part of the primary start-here set)

`results/` is entirely gitignored (`results/*` except `.gitkeep`) -- nothing
here is committed, so nothing in this document requires a git operation.
This is a **local cleanup plan** for the machine that has these directories.

**Update (final cleanup pass):** category C below (7 directories + 5 loose
files, all repo-hygiene/audit-log output) **has been deleted** -- all 5
safety criteria (gitignored, not referenced, not canonical, reproducible in
the sense that nothing unique is lost, no active process uses them) were
independently verified before deletion, and the exact list removed is
recorded in that section. Categories A and B were left untouched
(conservative choice -- see category B for why, even though its
precondition is now confirmed met).

Method: cross-referenced all `results/` subdirectory names against literal
path citations in `docs/` and `README.md`. ~68 subdirectories total; ~25
were not cited by literal path anywhere. Classified by directory-name/content
inspection, not a full re-audit of every file inside each (that level of
depth wasn't repeated here since Query 1 already did a first pass).

## A. Research result still useful (keep locally; no doc currently cites them by path, but they represent real findings)

| Directory | Likely content |
|---|---|
| `phase2a1_metric_oracle` | Phase 2A.1 metric/oracle finalization work |
| `phase2a3b_baseline_objective` | Phase 2A.3B hardened-baseline objective work |
| `phase2b1_dsl_verifier` | Phase 2B.1 DSL verifier development |
| `phase2b6_fair_sweep_failure_audit` | Phase 2B.6 fairness sweep failure mining |
| `phase2b8_rule_selector_repair` | Phase 2B.8 rule-selector repair (has a matching `docs/audits/phase2b8_rule_selector_repair_summary.md`, which describes findings but doesn't cite this literal path) |
| `phase2b13_selector_training_after_diversity` | Very likely the local output of the orphaned branch `phase2b13-selector-training-after-diversity` (confirmed in the repo audit as the one branch with unique, un-merged commits, superseded by its sibling `phase2b13-selector-training-and-suspicion-audit`). Keep alongside that branch until a human decides both artifacts' fate together. |
| `mixed_slo_comparison`, `overloaded_comparison`, `stressed_comparison` | Synthetic-config comparison runs matching `configs/mixed_slo_comparison.yaml` / `overloaded_comparison.yaml` / `stressed_comparison.yaml` -- regeneratable from those configs via `run_baseline_comparison.py`, but the specific run outputs aren't duplicated elsewhere |
| `small_debug`, `tiny_oracle_srtf_smoke` | Smoke-test outputs matching `configs/small_debug.yaml` / `configs/oracle/tiny_oracle_srtf_smoke.yaml` |

**Recommendation**: keep. Regeneratable in principle, but low storage cost
and no compelling reason to delete before someone specifically needs the
space.

## B. Checkpoint/resume history (Phase 1.7C cluster -- see also the dedicated classification note referenced from PROJECT_STATUS.md)

`phase17c_completion_gate`, `phase17c_health_check`,
`phase17c_pending_done_check`, `phase17c_recovery`,
`phase17c_recovery_checkpoint`, `phase17c_recovery_progress`,
`phase17c_sarathi_validation_check` -- 7 directories, all small (16K-108K),
all intermediate checkpoints from what was evidently an interrupted-and-resumed
Phase 1.7C run. **`results/phase17c` (4.2M) is the canonical final result** --
it's the only one of these 8 cited by any doc
(`docs/milestones/phase1_7c_calibrated_real_trace.md`).

**`results/phase17c` completeness: CONFIRMED this pass.** Contains logs for
all 7 named real-trace experiments, a final experiment summary
(`phase17c_experiment_summary.md`), plots, prediction-noise-sensitivity
analysis, and two passing final test-suite snapshots
(`final_pytest.log`: 182 passed; `final_gpu_pytest.log`: 3 passed, 179
deselected). This is a genuinely complete, correct final result.

**Classification: `SAFE_TO_DELETE_LATER`** (precondition now met, but not
executed this pass -- kept conservative deliberately, unlike category C
below: these are actual experiment checkpoint states, not pure meta/audit
output, and the storage cost (~400K total) doesn't justify the small
residual risk of removing something with even minor historical-debugging
value). A human (or a future pass) can delete
`phase17c_completion_gate`, `phase17c_health_check`,
`phase17c_pending_done_check`, `phase17c_recovery`,
`phase17c_recovery_checkpoint`, `phase17c_recovery_progress`,
`phase17c_sarathi_validation_check` with confidence now that this
completeness check has been done.

## C. Repo-maintenance/audit output (local logs from past cleanup/audit workflows, not research results)

`repo_audit`, `repo_full_audit`, `repo_full_audit_after_final_eval`,
`repo_cleanup_polish`, `fix_side_effect_scripts`, `github_create_push`,
`baseline_coverage_audit` -- 7 directories, dated mid-to-late June 2026.
Inspected directly this pass (not just by name): these contain audit logs
(`pytest_gpu.log`, `lint_static_audit.log`, `markdown_link_check.log`,
`policy_baseline_summary.md`, etc.) and one-off summaries
(`github_readiness_summary.md`, `security_privacy_summary.md`) -- outputs of
**prior runs of a repo-hygiene/audit process**, plausibly an ancestor of
this very 5-query cleanup effort. `baseline_coverage_audit` specifically
contains only 2 files that duplicate `docs/` content verbatim
(`dataset_workload_plan.md`, `external_baseline_coverage_report.md`) --
copies, not originals.

**Classification: `DELETE_NOW` -- executed this pass.** All 5 safety
criteria independently verified per-directory before deletion: gitignored
(confirmed via `git check-ignore`), zero references anywhere in
`docs/`/`README.md`/`tests/`/`scripts/`/`configs/`/`src/` (confirmed via
repo-wide grep), not canonical (logs of past audit runs, not scientific
results), nothing unique lost (conclusions already captured in the current
`docs/` files they were auditing, or superseded by this 5-query cleanup),
and no active process uses them.

**Removed** (7 directories + 5 loose files discovered during execution with
the same `repo_audit_` naming pattern, same content type, same verification
applied):
```
results/repo_audit/                       (36K)
results/repo_full_audit/                  (516K)
results/repo_full_audit_after_final_eval/ (216K)
results/repo_cleanup_polish/              (136K)
results/fix_side_effect_scripts/          (32K)
results/github_create_push/               (68K)
results/baseline_coverage_audit/          (36K)
results/repo_audit_file_list.txt
results/repo_audit_gpu_pytest.log
results/repo_audit_pip_list.txt
results/repo_audit_pytest.log
results/repo_audit_report.md
```
Total: ~1.05M freed. `results/` is entirely gitignored, so this is a pure
local filesystem change -- `git status` is unaffected (verified).

Four other loose top-level files that happened to be nearby
(`mixed_slo_final_patch.log`, `phase15_final_patch_pytest.log`,
`phase17c_status_check_report.md`, `prefill_heavy_final_patch.log`) are
**not** audit-log output -- they look like patch/test logs tied to actual
research phases -- and were deliberately left untouched; they were not
independently verified and are out of this section's scope.

## D. Duplicated run

None identified as clear duplicates in this pass beyond the Phase 1.7C
checkpoint cluster (category B), which is a checkpoint sequence rather than
true duplication.

## E. Empty/aborted

None found in `results/` specifically (the one empty/aborted run identified
in this audit, `vllm_qwen15b_stress_20260718T2158`, lives under
`experiments/`, not `results/` -- see
[EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md)).

## F. Unknown

None -- every orphaned directory found could be classified into A-C above
by name and/or content inspection.

## Summary

```
research_keep (A):                KEEP, ~11 directories, untouched
checkpoint_history (B):           SAFE_TO_DELETE_LATER, 7 dirs (Phase 1.7C),
                                   completeness of the canonical results/phase17c
                                   confirmed this pass; not deleted (conservative)
audit_safe_delete_candidates (C): DELETE_NOW, executed -- 7 dirs + 5 loose
                                   files removed (~1.05M freed), all 5 safety
                                   criteria independently verified per item
duplicated_run (D):               0
empty_aborted (E):                0
unknown (F):                      0
```

Category C's deletion is the only filesystem change this document produced;
everything else is documentation of a plan, not an executed action.
