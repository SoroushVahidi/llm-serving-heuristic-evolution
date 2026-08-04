# Compute Policy

Where work on this project should run, and how to keep it reproducible.
Codifies the pattern this branch's tasks have already been following in
practice (e.g. "use the local workstation if practical; use Wolverine
only if the job is genuinely large, batch/HPC-safe, reproducible, and
resource-estimated" — repeated verbatim across several task instructions
on this branch) rather than introducing new policy.

## Local workstation

**Use for:**
- Development and debugging.
- Unit tests and focused test suites (`pytest tests/test_*.py`,
  `pytest --collect-only`).
- Medium experiments: single-GPU training runs on the order of hours (the
  PARS-Serve-2026 checkpoint training — `bert-base-uncased`, 3 epochs,
  ~2 hours on this workstation's RTX 5060 Ti — is a concrete example of
  what "medium" means in practice on this branch), simulator sweeps on
  the order of tens of minutes (the vLLM-LTR comparative evaluation, 3
  seeds × 10 policies × real WildChat text, ~38 minutes measured).
- Anything that fits comfortably in a single session without tying up
  shared cluster resources for a routine task.

**This machine's actual specs, as observed this session (for future
estimation, not a guarantee — verify with `nvidia-smi`/`nproc` before
relying on these):** 20 vCPUs, 1× NVIDIA RTX 5060 Ti (16,311 MiB VRAM).

## NJIT Wolverine

**Use for:**
- Evolutionary search over large heuristic/policy spaces.
- Large benchmark sweeps (many seeds × many policies × many workloads,
  beyond what fits in a single local session in reasonable wall-clock
  time).
- Publication-scale experiments (final, citable results — not exploratory
  runs).
- Expensive simulator campaigns (thousands+ of simulator executions).
- Large hyperparameter searches (many training runs, not a single
  checkpoint).

**Before moving a job to Wolverine, per this branch's established
pattern:**
1. Confirm the job is genuinely too large for the local workstation —
   don't move to Wolverine to avoid waiting out a job that would finish
   locally in an hour or two (moving to hide an unresolved local
   performance bug is explicitly the wrong move — see
   `docs/audits/vllm_ltr_comparative_evaluation_recovery_20260804.md`'s
   own framing of this principle when the vLLM-LTR selector's real
   performance bug was fixed locally rather than papered over by moving
   the job elsewhere).
2. Make the code batch/HPC-safe (deterministic seeds, no interactive
   prompts, resumable checkpointing where the job could be preempted).
3. Confirm environment and data access are reproducible on the cluster
   (pinned versions, no machine-specific absolute paths — see the
   `PARS_OFFICIAL_CLONE_PATH`-style environment-variable-overridable
   pattern in `baselines/pars/adapter/provenance.py` for a concrete
   example of how to avoid hardcoding a workstation-specific path).
4. Document a job script and a resource estimate (expected wall-clock,
   GPU-hours, memory) before submitting — this project's own audits
   consistently do this kind of sizing before committing to a long run
   (e.g. the vLLM-LTR recovery doc's explicit before/after runtime
   estimates once the selector performance bug was understood).

**Do not use Wolverine to avoid a slow but tractable local job, and do
not use it for exploratory/debugging work.**

## Reproducibility, logging, and tmux usage

- **tmux for anything long-running.** Every long job on this branch so
  far has run in a named tmux session (`pars_training`,
  `vllm_ltr_comparison_recovery`, the CC3-CC5 sessions, etc.) with output
  piped through `tee` to a timestamped log file — this lets the session
  be monitored (`tmux capture-pane -p`) without interrupting it, and
  keeps a durable log independent of the terminal.
- **Manifests.** Every comparative-evaluation script on this branch
  writes a `run_manifest.json` recording: the exact command line, input
  file paths + SHA-256 hashes, seeds, policies compared, and a replay
  command (see `scripts/run_vllm_ltr_first_comparative_evaluation.py`,
  `scripts/run_pars_first_comparative_evaluation.py`,
  `scripts/generate_canonical_benchmark_suite.py`'s `suite_manifest.json`)
  — new evaluation scripts should follow this pattern.
- **Checkpointing.** Training scripts should save at least a "best so
  far" checkpoint (not only a final one) so a job that's killed partway
  through still leaves a usable artifact — the PARS-Serve-2026 training
  run's official script already does this (`best_model.pt` saved after
  every epoch whose validation accuracy improves).
- **Seed recording.** Every stochastic process (synthetic workload
  generation, bootstrap resampling, model training) records its seed in
  its own output manifest — never rely on an unrecorded default.
- **Resource accounting.** When a job's runtime matters for future
  planning (deciding local vs. Wolverine, or budgeting an experiment
  campaign), record actual measured wall-clock/GPU time in that work's
  audit doc, not just an a priori estimate — this branch's audits
  consistently do this (e.g. the canonical benchmark suite's measured
  73.3s generation + 5m32s full characterization pass, both recorded in
  `docs/audits/canonical_benchmark_suite_design_20260804.md`).

## Large artifacts (checkpoints, datasets, caches)

Never commit trained model checkpoints, downloaded datasets, or their
caches into this git repository — see `.gitignore`'s `results/*`,
`data/raw/*`, `data/processed/*` rules, and (for a checkpoint's own
license/provenance record) `baselines/vllm_ltr/CHECKPOINT_PROVENANCE.md`
and `baselines/pars/PROVENANCE.md` for the pattern: record hashes,
sizes, and provenance in a committed markdown file, while the actual
binary artifact lives outside git (either in a `results/`-style
gitignored local directory, or — for cloned external repositories — fully
outside this repository's directory tree entirely, as
`baselines/pars/adapter/provenance.py`'s `DEFAULT_OFFICIAL_CLONE_PATH`
does).

See also: [`docs/BASELINE_STATUS.md`](BASELINE_STATUS.md),
[`docs/INDEX.md`](INDEX.md).
