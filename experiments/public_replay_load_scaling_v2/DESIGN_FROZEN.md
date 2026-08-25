# DESIGN FROZEN -- public_replay_load_scaling_v2

Frozen: 2026-08-25, before the full-matrix SLURM array was submitted.

**v2 is an implementation-correction rerun of v1, NOT a new outcome-driven design.**
Full record: `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V2.md`. Scientific design of record
(unchanged, incorporated by reference): `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md`.

v1's raw outputs, integrity report, and analysis remain preserved, unmodified, at
`experiments/public_replay_load_scaling_v1/` and
`docs/current/public_replay_load_scaling_v1_analysis_20260825.md`. v1 was invalidated for
scientific interpretation by a simulation-horizon implementation bug (hardcoded
`SIM_MAX_STEPS=200_000`), detected via its own preregistered λ=1 reproduction gate -- a
positive reproducibility/audit outcome.

Frozen at repo HEAD `2987b7181efa2bc550d8a894c537eca8f6393eb6` (worktree dirty at freeze
time; unrelated pre-existing changes, not touched by this experiment).

## Frozen at this commit-equivalent (file hashes, sha256)

- `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V2.md`: `e7910911b22e08218b5ab1d574825c37b3cd0ef3bbd3cf66fe1776b5ddf49ef2`
- `src/llmserveopt/policy_separation/public_replay_load_scaling_v2.py`: `61bb5607155e54dd7c91db6e893ba85830bdc276559c772f9d9bd8d61e11130d`
- `scripts/run_public_replay_load_scaling_v2.py`: `d3027e0b8e30955038413597896b127f59176ed74036a0c99fb72928740002c5`
- `scripts/slurm/public_replay_load_scaling_v2.sbatch`: `7889027bd2868afb86a1b1a63dde5f6f70505ab7070549c280292f68d96c6819`
- `tests/test_public_replay_load_scaling_v2.py`: `a5c355d4e9d493eaa11d04cf1ae5a88222aa255fbbc41db20e1960a6a577d286`

All five hashes verified identical between local staging and the remote Wulver bundle
(`/mmfs1/scratch/ikoutis/sv96/llm-serving-heuristic-evolution/public_replay_load_scaling_v2/`)
before job submission.

## Exact implementation diff vs. v1 (minimal, auditable)

Only the module docstring/comments and three lines changed in
`public_replay_load_scaling_v2.py` relative to `public_replay_load_scaling_v1.py`:

```
BUILDER_VERSION = "public_replay_load_scaling_v1.0.0"   ->  "public_replay_load_scaling_v2.0.0"
WORKLOAD_ID      = "public_replay_load_scaling_v1"       ->  "public_replay_load_scaling_v2"
SIM_MAX_STEPS    = 200_000                                ->  None   # <- the actual bug fix
SIM_DRAIN_STEPS  = 50_000                                 ->  50_000 # unchanged
```

`scripts/run_public_replay_load_scaling_v2.py` differs from v1's runner only in the module
import (`prl = public_replay_load_scaling_v2`), the default output directory, and docstring
text -- no logic changes.

## Pinned dependencies (unchanged from v1, verified identical)

- VTC official clone commit: `192c2e2014c69c8c6c699d7113c3822e4db632e6` (copied server-side
  from the v1 remote bundle, not re-cloned -- guarantees byte-identical pin)
- Python executable: `/home/sv96/.conda/envs/feedback-weighted-maximization/bin/python`
  (Python 3.11.14 on compute nodes, confirmed via smoke `srun`)
- `data/`, `baselines/` copied server-side from the v1 remote bundle (unchanged corpora/code)

## Local pre-submission validation (all passed before SLURM submission)

1. `pytest tests/test_public_replay_load_scaling_v2.py`: **24/24 passed** (20 inherited from
   v1's suite re-run against the v2 module + 4 new bug-regression tests: `SIM_MAX_STEPS is
   None` guard, worst-case window `burstgpt::w0` full reproduction, all 11 v1-failing windows
   no longer truncated, no-truncation-past-200s check).
2. **λ=1 pre-full-run gate** (all 60 windows × 8 policies = 480 cells, local):
   `experiments/public_replay_load_scaling_v2/lambda1_gate_result.json` --
   **0 P6 failures / 360 checked, 0 truncated cells / 480, 0 non-success, max ANWG error from
   1.0 = 0.0 (exact)**. Gate: **PASS**.
3. **High-load smoke** (`burstgpt::w0` × λ∈{1,16,128} × {full_prefill,
   kv_constrained_online}), run both locally and via `srun` on an actual Wulver compute node
   (`n0007`, same partition/account/QOS as the full array): λ=1 exactly reproduces
   ANWG=1.0/completion=1.0/200-200 completed; `active_max` pressure increases monotonically
   with λ (2→2→5); natural termination (no artificial horizon); serialization clean; peak RSS
   ≈2GB; total runtime for 6 cells ≈9s on a compute node. Not interpreted scientifically.

## SLURM submission

- Job: **array 1195618** (parent job id `1195619` per-task), submitted 2026-08-25 ~19:17 UTC
- `--partition=general --account=ikoutis --qos=standard --mem=6G --time=00:30:00 --array=0-59%32`
- 3-minute post-submission health check: 32/60 tasks accepted and `RUNNING` immediately
  (n0007/n0009/n0010/n0017/n0023), 28 correctly `PENDING` on the `%32` array-concurrency cap,
  0 tasks in any failure state, no import/traceback errors in any `.err` log (only the
  expected/harmless `fatal: not a git repository` message from `_git_head_sha()`, identical to
  v1's logs, caught by existing `try/except`), all 32 running tasks already writing their
  `cells_window_*.jsonl` output files. Monitoring stopped at ~3 minutes per instructions; job
  was not waited on to completion in this session.
