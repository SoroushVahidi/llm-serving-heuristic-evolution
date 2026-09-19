# SBS Override Query 3 Preparation

This file lists the intended commit organization for the next query. It is a
plan, not a commit record.

## Must Recheck Before Committing

Before any commit, check whether `sbs_cons_selector_v1` has naturally
completed:

```bash
cd /home/soroush/llm-serving-heuristic-evolution
tmux has-session -t sbs_cons_selector_v1 && echo running || echo finished
find experiments/sbs_override_conservative_selector_dev_v1/run_v1 \
  -maxdepth 2 -type f | sort
```

If still running, do not commit partial final selector artifacts. It is still
acceptable to commit handoff/provenance docs and source/tests that are already
complete.

## Proposed Commit Groups

### 1. Active Job And SBS Status Docs

Files:

- `docs/current/ACTIVE_JOBS.md`
- `docs/current/SBS_OVERRIDE_STATUS.md`
- `docs/current/SBS_OVERRIDE_QUERY3_PREP.md`
- small pointer updates in `docs/current/RESUME_HERE.md`,
  `docs/current/WORK_STATUS.md`, `docs/current/NEXT_ACTIONS.md`, and
  `docs/current/ACTIVE_EXPERIMENT_PROTECTED_PATHS.md`

Validation:

- Markdown link check by sampling local links;
- no fresh confirmatory outcome files opened.

### 2. Conservative Selector Development Source

Files:

- `docs/design/SBS_OVERRIDE_CONSERVATIVE_SELECTOR_DEV_V1.md`
- `scripts/sbs_override_conservative_selector_dev_v1.py`
- `tests/test_sbs_override_conservative_selector_dev_v1.py`

Validation:

```bash
python -m py_compile scripts/sbs_override_conservative_selector_dev_v1.py \
  tests/test_sbs_override_conservative_selector_dev_v1.py
python -m pytest -q tests/test_sbs_override_conservative_selector_dev_v1.py
```

### 3. Fresh-Label Provenance Freeze

Files:

- `experiments/sbs_override_fresh_id_confirmatory_terminal_label_v1/provenance_freeze_1299925/PROVENANCE_FREEZE_1299925.md`
- `experiments/sbs_override_fresh_id_confirmatory_terminal_label_v1/provenance_freeze_1299925/local_sha256.txt`
- preserved `docs_design/`, `scripts/`, and `scripts_slurm/` files under the
  same provenance-freeze directory

Validation:

- `sha256sum -c` equivalent for the local freeze if Query 3 wants a hard gate;
- do not copy or read generated fresh label CSVs.

### 4. Fresh Corpus And OOD Support Metadata

Files to consider:

- `docs/design/SBS_OVERRIDE_FRESH_CONFIRMATORY_CORPUS_V1.md`
- `docs/design/SBS_OVERRIDE_FRESH_OOD_SUPPORT_EXPANSION_V1.md`
- `docs/design/SBS_OVERRIDE_FRESH_ID_CONFIRMATORY_TERMINAL_LABEL_V1.md`
- compact fresh corpus manifests and summary/checksum files
- compact OOD null-result summaries

Do not commit:

- large support-scan policy-row shards;
- zero-support diagnosis policy-row shards;
- fresh terminal-label shards.

### 5. Final Selector Outputs, Only If Completed

Files to consider after completion:

- `experiments/sbs_override_conservative_selector_dev_v1/run_v1/RUN_SUMMARY.json`
- `experiments/sbs_override_conservative_selector_dev_v1/run_v1/development_oof_result.json`
- `experiments/sbs_override_conservative_selector_dev_v1/run_v1/final_development_oof_result.json`
- `experiments/sbs_override_conservative_selector_dev_v1/run_v1/FINAL_CONFIRMATORY_SELECTOR_V1.json`
- `experiments/sbs_override_conservative_selector_dev_v1/run_v1/FINAL_CONFIRMATORY_SELECTOR_V1_DESIGN.md`
- `experiments/sbs_override_conservative_selector_dev_v1/run_v1/confirmatory_protocol_and_verdict_freeze.json`
- `experiments/sbs_override_conservative_selector_dev_v1/run_v1/reload_prediction_test.json`

Model binary policy:

Treat `FINAL_CONFIRMATORY_SELECTOR_V1.joblib` as an external generated artifact
unless Query 3 explicitly decides to commit it. If not committed, preserve its
path, SHA-256, size, and required regeneration command in documentation.

Do not commit:

- full OOF action-prediction CSVs unless a compact summary is insufficient;
- `run.log`;
- tmux/stdout logs.

## Files That Need Review Before Any Broad Commit

The original worktree also contains many older modified/untracked files,
including root README/license/documentation changes and large artifact bundles.
Those are not part of the SBS cleanup unless explicitly selected.

Avoid broad `git add .`. Use explicit file lists.

## Main Branch Policy

Do not update `main` in Query 3 unless explicitly authorized after review. The
safe default is to commit and push the active integration branch and preserve
the unmerged joint240 SBS experiment branches.
