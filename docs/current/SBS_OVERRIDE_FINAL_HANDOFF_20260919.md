# SBS Override Final Handoff

Created: 2026-09-19.

This is the durable pause-state document for the SBS-relative override research
line after Queries 1-4. It is intended as the first file to read when resuming
this work.

## Authority

| Field | Value |
| --- | --- |
| Repository | `/home/soroush/llm-serving-heuristic-evolution` |
| Integration branch | `contextual-compositional-heuristics-20260731` |
| Query-4 starting HEAD | `5d3c6f60757c9db9160ade8503af4e1a751bc862` |
| Remote branch | `origin/contextual-compositional-heuristics-20260731` |
| Current active experiment | `SBS_OVERRIDE_CONSERVATIVE_SELECTOR_DEV_V2` |
| Active run root | `experiments/sbs_override_conservative_selector_dev_v2/run_v2` |
| Confirmatory label access | `CONFIRMATORY_LABEL_ACCESS = NOT_ACCESSED` |

Use this document together with:

- [`ACTIVE_JOBS.md`](ACTIVE_JOBS.md)
- [`SBS_OVERRIDE_STATUS.md`](SBS_OVERRIDE_STATUS.md)
- [`WORK_STATUS.md`](WORK_STATUS.md)
- [`NEXT_ACTIONS.md`](NEXT_ACTIONS.md)
- [`SBS_OVERRIDE_QUERY3_PREP.md`](SBS_OVERRIDE_QUERY3_PREP.md)
- [`SBS_OVERRIDE_FINAL_SELECTOR_PREREGISTRATION_AUDIT_20260919.md`](SBS_OVERRIDE_FINAL_SELECTOR_PREREGISTRATION_AUDIT_20260919.md)

## Scientific Goal

The project is testing whether a conservative online selector can safely
override the native SBS action on states where SBS and the P6 portfolio disagree.
The current estimand is SBS-relative one-step causal advantage:

```text
A_SBS(s,a) = Q_SBS(s,a) - Q_SBS(s,SBS)
```

where one action is forced at state `s`, then the replay follows fixed
`kv_constrained_online` semantics with future arrivals and simulator randomness
held fixed across branches.

## Established Evidence

- The joint240 development corpus contains SBS-vs-P6 disagreement states with
  terminal SBS-relative labels and online-safe `STATE_ACTION_V1` features.
- The development V1 selector had real but uncertain signal:
  `HELD_OUT_SIGNAL_POSITIVE_BUT_UNCERTAIN`, mean OOF realized gain/state
  `+0.00019594`, CI crossing zero, 450 harmful overrides, 57 negative scenarios,
  and 3/5 positive folds.
- ExtraTrees was the strongest standalone development model in V1, and
  ACTION_DIFF features were materially useful.
- A fresh generator-holdout confirmatory corpus was frozen and terminal-labeled
  on Wulver, but its scientific outcome columns remain unopened.
- Natural external OOD support scans found zero canonical SBS-vs-P6
  disagreement for Azure 2023 code, Azure 2023 conversation, Azure 2024,
  Bailian/Qwen, and BurstGPT v2 under the frozen native replay semantics.

## Null And Negative Results

- The natural OOD result is an action-support null result, not a failed label
  run. Do not repair it by adding workload overlays and calling the result
  natural OOD confirmation.
- Earlier Family-A state-action / DAgger and closed-loop attempts did not
  justify closed-loop deployment.
- The V1 development selector was positive on mean gain but not robust enough
  to move directly to confirmatory evaluation.

## Canonical Evidence

Development corpus:

- State-action rows:
  `/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1/state_action_rows_full.csv`
- SHA-256:
  `a96af891b70bcd0895440207df6febb5d1c1545af4eaaf455367e56152950dca`
- Policy-action map:
  `/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1/state_policy_action_map_full.csv`
- SHA-256:
  `7a4118b65e4222e189aa1139ba548ade99951ae285a50b7b193fec6ef55d1324`

Development universe:

- 8,888 SBS-disagreement states
- 21,858 unique non-SBS canonical action rows
- 236 scenarios
- five existing scenario folds
- `STATE_ACTION_V1` = 95 physical-state features + 70 action-difference features

Fresh confirmatory universe:

- Raw fresh manifest: 2,918 disagreement-state rows, including 56 duplicate
  state IDs
- Canonical clean `full_support_only` universe: 2,862 unique states
- Supported fresh scenarios: 78
- Unique non-SBS actions: 6,996
- Total terminal continuations: 9,858
- Wulver label root:
  `/mmfs1/scratch/ikoutis/sv96/sbs_override_fresh_id_confirmatory_terminal_label_v1/run_v1`

Fresh labels generated does not mean fresh labels evaluated.

## Active Selector Run

Historical V1 status:

`FINAL_CONFIRMATORY_SELECTOR_V1 = HISTORICAL_NOT_FOR_CONFIRMATION`.

The V1 run completed and remains preserved as development evidence. A later
audit found ambiguous final-selector preregistration provenance: the
implementation selected the final configuration from aggregated nested-inner
candidate summaries and full-development gate refit, while the design markdown
described a final choice from outer-OOF bootstrap summaries. This is a
procedural preregistration ambiguity, not a mathematical invalidation of the V1
result.

V2 supersedes the ambiguous V1 final-selection rule. V2 keeps nested outer OOF
as evaluation-only and performs final configuration selection in a separate
full-development grouped-CV stage using development data only.

The later V2 development-only run command is:

```bash
python scripts/sbs_override_conservative_selector_dev_v2.py \
  --execute-full-selection \
  --n-jobs 8 \
  --bootstrap-replicates 2000
```

Do not run this command until the V2 design/source/test freeze is accepted.

Query 4 performed the required single V1 check only. At that check:

- tmux was alive;
- the bash wrapper and Python process were alive;
- the Python process was CPU-active;
- `PREREGISTERED_SEARCH_DESIGN.json` existed;
- `outer_fold_progress/outer_0.json` existed;
- `outer_fold_progress/outer_1.json` existed;
- `RUN_SUMMARY.json` did not exist;
- no DONE/EXIT marker existed;
- final selector, serialized model, and confirmatory protocol files did not
  exist yet;
- `logs/run.log` existed but was empty;
- no stderr/error evidence was found.

Classification: `RUNNING_HEALTHY`.

No later selector polling was performed in Query 4.

Expected successful completion files:

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

- tmux session naturally exits;
- `RUN_SUMMARY.json` reports `CONFIRMATORY_EVALUATION_READY`;
- log contains `EXIT:0`;
- reload prediction test passed;
- no fresh confirmatory outcome path was accepted as a training or evaluation
  input.

## Resume Commands

Use these when returning later:

```bash
cd /home/soroush/llm-serving-heuristic-evolution
git fetch --all --prune
git status -sb
tmux has-session -t sbs_cons_selector_v1 && echo running || echo finished
find experiments/sbs_override_conservative_selector_dev_v1/run_v1 \
  -maxdepth 2 -type f | sort
cat experiments/sbs_override_conservative_selector_dev_v1/run_v1/RUN_SUMMARY.json
tail -80 experiments/sbs_override_conservative_selector_dev_v1/run_v1/logs/run.log
python -m pytest -q tests/test_sbs_override_conservative_selector_dev_v1.py
```

If the tmux session is still running, stop there and leave the run alone.

## Result Consumption Procedure

For V2:

1. Run the full V2 development-only recomputation in a separate task.
2. Inspect development-side outputs only.
3. Verify `RUN_SUMMARY_V2.json`, `reload_prediction_test_v2.json`,
   `FINAL_CONFIRMATORY_SELECTOR_V2.json`, and
   `FINAL_CONFIRMATORY_SELECTOR_V2.joblib`.
4. Preserve compact V2 selector metadata/provenance and decide whether the
   final model binary belongs in Git or remains an external artifact.
5. Only after that, run a separate explicitly authorized one-shot fresh
   confirmation task.

Do not modify the frozen V2 selector protocol after the development run
completes.

## Confirmatory Blindness Boundary

`CONFIRMATORY_LABEL_ACCESS = NOT_ACCESSED`.

Forbidden until the separate one-shot confirmation task:

- reading fresh `Q_SBS`;
- reading fresh `A_SBS`;
- reading fresh terminal utility columns;
- counting fresh positive, negative, or zero effects;
- evaluating any selector on fresh labels;
- tuning features, thresholds, gates, or model choices from fresh outcomes.

Permitted before confirmation:

- already-recorded file paths;
- filenames and file sizes;
- already-recorded checksums;
- structural counts and provenance;
- source/scenario identity and action-support metadata.

## Branch Map

Active integration branch:

- `contextual-compositional-heuristics-20260731` at Query-4 start SHA
  `5d3c6f60757c9db9160ade8503af4e1a751bc862`

Important preserved SBS experiment branches:

- `experiment/joint240-dense-sbs-state-action-v1`:
  `11e9ac10898242a5c5fa95ac27174b9bba85339e`
- `experiment/joint240-sbs-disagreement-scan-v1`:
  `072e1a555e75ec7c732e6bf888026ef9850c6c54`
- `experiment/joint240-sbs-targeted-terminal-label-pilot-v1`:
  `e308b335776df2858941ebe2fcf00d9ce2e9d438`
- `experiment/joint240-sbs-targeted-terminal-label-full-v1`:
  `0900ee7b565ee1cbfffe86272e30e4d0db13483c`
- `experiment/joint240-sbs-advantage-learnability-v1`:
  `ddd94c0685f8c345e3d166337044c7b3caf15153`

Older phase2 and selector-v2 branches are historical context unless a future
task explicitly revives them.

## Wulver Artifacts To Retain

Retain:

- `/mmfs1/scratch/ikoutis/sv96/sbs_override_fresh_id_confirmatory_terminal_label_v1/run_v1`
- `/mmfs1/project/ikoutis/sv96/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1-src`
- Slurm/log/provenance records for jobs `1298715`, `1299504`, `1299588`,
  `1299833`, and `1299925`

Job `1299925` produced the fresh terminal labels from committed base
`ff34f6fa0303b6f21e13277553f2d5da789b01d4` plus untracked fresh scripts/design
that are now preserved locally under
`experiments/sbs_override_fresh_id_confirmatory_terminal_label_v1/provenance_freeze_1299925/`.

## Source And Generated Boundary

Track in Git:

- source code;
- tests;
- design docs;
- handoff/status docs;
- Slurm templates;
- compact manifests, checksums, and provenance metadata.

Keep external or untracked unless a later release decision says otherwise:

- large state-action CSVs;
- large terminal-label CSVs;
- raw shard outputs;
- logs;
- active run directories;
- model binaries, unless the final selector artifact is intentionally treated as
  a release artifact.

## Work Blocked

- final conservative selector artifact commit;
- one-shot fresh confirmatory evaluation;
- manuscript update with the SBS override confirmatory result;
- any closed-loop selector deployment;
- real-vLLM engine validation of this selector.

## Exact Next Action

Let `sbs_cons_selector_v1` finish naturally. When returning, run the resume
commands above once. If complete, inspect development-side outputs only and
commit the final selector/protocol metadata before any fresh outcome evaluation
is authorized.
