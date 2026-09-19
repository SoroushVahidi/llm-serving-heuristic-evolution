# Active Jobs

Last documented for Query 4: 2026-09-19 14:06:21 EDT.

The Query-4 selector health check was performed once, as required. No later
selector polling was performed while writing the final handoff.

This file records jobs that cleanup, commit, and documentation tasks must not
interrupt. It is a handoff document only; it is not a request to monitor,
restart, cancel, or modify any job.

## `sbs_cons_selector_v1`

| Field | Value |
| --- | --- |
| Status | `RUNNING_HEALTHY` at the last single check |
| Experiment | `SBS_OVERRIDE_CONSERVATIVE_SELECTOR_DEV_V1` |
| tmux session | `sbs_cons_selector_v1` |
| Source branch | `contextual-compositional-heuristics-20260731` |
| Source HEAD | `94f4621bb6610c2b426e365f659636b1a48a89f5` |
| Run root | `experiments/sbs_override_conservative_selector_dev_v1/run_v1` |
| Log | `experiments/sbs_override_conservative_selector_dev_v1/run_v1/logs/run.log` |
| Command | `python scripts/sbs_override_conservative_selector_dev_v1.py --n-jobs 8 --bootstrap-replicates 2000` |
| Current evidence | Process alive, CPU-active, `PREREGISTERED_SEARCH_DESIGN.json` exists, `outer_fold_progress/outer_0.json` and `outer_fold_progress/outer_1.json` exist, `RUN_SUMMARY.json` not yet present |

Scientific purpose:

Use joint240 development data only to run the preregistered conservative
selector search, freeze exactly one final selector, refit it on all development
scenarios, and write the one-shot fresh confirmatory protocol before any fresh
terminal outcomes are opened.

Expected final files if the run completes successfully:

- `RUN_SUMMARY.json`
- `development_outer_oof_action_predictions.csv`
- `development_outer_oof_state_decisions.csv`
- `development_inner_candidate_results.csv`
- `development_oof_result.json`
- `final_development_oof_result.json`
- `FINAL_CONFIRMATORY_SELECTOR_V1.json`
- `FINAL_CONFIRMATORY_SELECTOR_V1.joblib`
- `FINAL_CONFIRMATORY_SELECTOR_V1_DESIGN.md`
- `confirmatory_protocol_and_verdict_freeze.json`
- `reload_prediction_test.json`

Completion criteria:

- tmux session has naturally exited;
- `RUN_SUMMARY.json` exists and reports `CONFIRMATORY_EVALUATION_READY`;
- log contains `EXIT:0`;
- reload prediction test passed;
- no fresh confirmatory outcome path was accepted as a train/evaluation input.

Commands to check later:

```bash
cd /home/soroush/llm-serving-heuristic-evolution
tmux has-session -t sbs_cons_selector_v1 && echo running || echo finished
find experiments/sbs_override_conservative_selector_dev_v1/run_v1 \
  -maxdepth 2 -type f | sort
cat experiments/sbs_override_conservative_selector_dev_v1/run_v1/RUN_SUMMARY.json
tail -80 experiments/sbs_override_conservative_selector_dev_v1/run_v1/logs/run.log
```

Result-consumption plan after completion:

1. Read only development-side outputs and selector freeze artifacts.
2. Verify `RUN_SUMMARY.json`, `reload_prediction_test.json`, and the final
   selector/protocol files.
3. Commit compact development summaries and final selector/protocol metadata if
   repository convention supports them.
4. Do not run the fresh one-shot confirmatory evaluation until the final
   selector freeze is committed and explicitly authorized.

Work blocked by this job:

- final selector artifact commit;
- final development robustness comparison documentation;
- confirmatory-evaluation readiness handoff.

Confirmatory-blindness restrictions:

- Do not open fresh terminal label CSV contents.
- Do not read `Q_SBS`, `A_SBS`, terminal utilities, or positive/negative rates.
- Do not evaluate this selector against fresh labels.
- The fresh root `/mmfs1/scratch/ikoutis/sv96/sbs_override_fresh_id_confirmatory_terminal_label_v1/run_v1`
  may be referenced only for filenames, metadata, already-recorded checksums,
  structural counts, and provenance.
