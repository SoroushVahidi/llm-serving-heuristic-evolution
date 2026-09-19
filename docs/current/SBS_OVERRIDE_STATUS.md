# SBS Override Status

Last updated: 2026-09-19.

This is the current navigation and provenance map for the SBS-relative override
line. Historical docs before this file are still useful, but many of them
predate the SBS-specific development and fresh-confirmatory sequence.

## Current Status

The active experiment is `SBS_OVERRIDE_CONSERVATIVE_SELECTOR_DEV_V1`. It is
running locally in tmux session `sbs_cons_selector_v1`; see
`ACTIVE_JOBS.md`.

Fresh confirmatory terminal labels exist, but their scientific outcomes remain
blind. The final selector must be frozen from development data only before the
fresh labels are joined for one-shot evaluation.

## Confirmatory Blindness Boundary

`CONFIRMATORY_LABEL_ACCESS = NOT_ACCESSED` for this cleanup pass.

Forbidden until the one-shot confirmatory evaluation task:

- reading fresh `Q_SBS`;
- reading fresh `A_SBS`;
- counting fresh positive/negative/zero effects;
- computing fresh terminal utility distributions;
- evaluating any selector on fresh labels;
- changing selector thresholds or features using fresh outcomes.

Permitted before confirmation:

- filenames;
- file sizes;
- checksums and provenance;
- already-frozen structural counts;
- source/scenario identity and action-support metadata.

The selector script added for this line refuses training/evaluation input paths
containing `sbs_override_fresh_id_confirmatory_terminal_label_v1`.

## Canonical Development Inputs

Development corpus:

`/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1/state_action_rows_full.csv`

SHA-256:

`a96af891b70bcd0895440207df6febb5d1c1545af4eaaf455367e56152950dca`

Policy/action map:

`/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1/experiments/joint240_sbs_targeted_terminal_label_full_v1/full_v1/analysis_v1/state_policy_action_map_full.csv`

SHA-256:

`7a4118b65e4222e189aa1139ba548ade99951ae285a50b7b193fec6ef55d1324`

Development universe:

- 8,888 SBS-disagreement states;
- 21,858 unique non-SBS canonical action rows;
- 236 scenarios;
- 5 scenario folds;
- `STATE_ACTION_V1` = 95 state features + 70 action-difference features.

## Fresh Confirmatory Inputs And Outputs

Fresh corpus construction root:

`experiments/sbs_override_fresh_confirmatory_corpus_v1/`

Important distinction:

- raw fresh manifest had 2,918 disagreement-state rows and 56 duplicate state
  IDs;
- clean `full_support_only` manifest is canonical for confirmation;
- actual labeled universe is 2,862 unique states across 78 supported fresh
  scenarios;
- unique non-SBS actions: 6,996;
- total terminal continuations: 9,858.

Fresh terminal-label output root on Wulver:

`/mmfs1/scratch/ikoutis/sv96/sbs_override_fresh_id_confirmatory_terminal_label_v1/run_v1`

Read-only metadata from Query 1:

- 80 DONE markers;
- 0 FAILED markers;
- 80 state-action shards;
- 80 policy-action-map shards;
- 80 summary shards;
- 80 shard provenance txt files.

Those fresh outcome shards are generated evidence and must remain outside Git.

## OOD Null Result

Natural external OOD action-support scans produced zero canonical
SBS-vs-P6 disagreement under frozen native replay semantics for:

- Azure 2023 code;
- Azure 2023 conversation;
- Azure 2024;
- Bailian/Qwen;
- BurstGPT v2.

This is a preserved action-support null result. It must not be "repaired" into
natural-OOD confirmation by adding workload overlays after the fact. Any future
stress-OOD experiment must be labeled synthetic or controlled.

## Scientific Lineage

1. Original scenario-level selector work: useful but not SBS-relative action
   causal evidence.
2. Guarded SBS fallback: showed abstention helps but did not settle safe
   override selection.
3. Decision criticality and joint240 terminal criticality: established
   state-level opportunity and timescale issues.
4. Continuation-sensitivity result: showed continuation semantics matter.
5. Family-A state-action and DAgger attempts: negative/limited closed-loop
   evidence.
6. Dense SBS state-action acquisition: branch
   `experiment/joint240-dense-sbs-state-action-v1`, commit `11e9ac1`.
7. Misalignment discovery: old state acquisition did not target the exact SBS
   disagreement support needed for override labeling.
8. SBS disagreement scan: branch
   `experiment/joint240-sbs-disagreement-scan-v1`, commit `072e1a5`.
9. Targeted terminal-label pilot: branch
   `experiment/joint240-sbs-targeted-terminal-label-pilot-v1`, commit
   `e308b33`.
10. Full targeted terminal-label corpus: branch
    `experiment/joint240-sbs-targeted-terminal-label-full-v1`, through commits
    `a1fc763`, `ff34f6f`, and `0900ee7`.
11. Held-out advantage learnability V1: branch
    `experiment/joint240-sbs-advantage-learnability-v1`, through commits
    `04f6844`, `4004c72`, and `ddd94c0`.
12. V1 result: `HELD_OUT_SIGNAL_POSITIVE_BUT_UNCERTAIN`.
13. Fresh confirmatory corpus construction and natural OOD support scans:
    untracked Wulver/local source preserved by the provenance freeze.
14. Fresh-ID confirmatory terminal labels: Wulver job `1299925`, completed.
15. Conservative selector development V1: currently running from this branch.

## Branches To Preserve

- `contextual-compositional-heuristics-20260731`: active integration branch.
- `experiment/joint240-dense-sbs-state-action-v1`: preserve source and tests.
- `experiment/joint240-sbs-disagreement-scan-v1`: preserve source and tests.
- `experiment/joint240-sbs-targeted-terminal-label-pilot-v1`: preserve pilot
  provenance.
- `experiment/joint240-sbs-targeted-terminal-label-full-v1`: preserve canonical
  development label machinery and corpus analysis.
- `experiment/joint240-sbs-advantage-learnability-v1`: preserve V1 selector
  evidence and post-run analysis.

## Source And Generated Boundary

Commit candidates:

- SBS design docs;
- run scripts and analysis code;
- tests;
- Slurm templates;
- compact manifest summaries;
- checksums;
- provenance freeze for job `1299925`;
- compact selector development summaries after the active run finishes.

Do not commit by default:

- large terminal-label CSVs;
- large state/action row CSVs;
- policy-row shard CSVs;
- raw logs;
- scratch shards;
- model binaries, unless Query 3 explicitly decides the final selector binary
  is a release artifact rather than an external artifact.

## Next Scientific Gate

After `SBS_OVERRIDE_CONSERVATIVE_SELECTOR_DEV_V1` finishes and the final
selector is committed, the next scientific action is exactly one fresh
confirmatory evaluation under the frozen protocol. Until then, fresh terminal
outcomes stay blind.
