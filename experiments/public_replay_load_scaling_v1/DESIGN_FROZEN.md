# DESIGN FROZEN -- public_replay_load_scaling_v1

Frozen: 2026-08-25, before any full-matrix cell was run and before any
scheduler outcome was inspected for this experiment.

Full design: `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md`

Frozen at repo HEAD `2987b7181efa2bc550d8a894c537eca8f6393eb6` (worktree
dirty at freeze time; unrelated pre-existing changes, not touched by this
experiment).

## Frozen at this commit-equivalent (file hashes, sha256)

- `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md`: `8c2415643f6371abef01cca96eb99331cfd7340f122c661e5b5b9d6dc6dcffa8`
- `src/llmserveopt/policy_separation/public_replay_load_scaling_v1.py`: `74d5de30af0b82246109c666eea95f262cd8d05c22fe83a88050acfe9f6584a2`
- `scripts/run_public_replay_load_scaling_v1.py`: `54739ec93f4080efb9ce0111385e23ac20ebae84f0d69214ed93afc660c741d2`

## Frozen parameters (do not change post-hoc)

- `LOAD_FACTORS = (1, 2, 4, 8, 16, 32, 64, 128)`
- `PEXT_POLICIES` = 6 native (P6) + `official_vtc_joint_token_budget_remap` +
  `vllm_style_continuous_batching` = 8 policies
- 60 canonical augmented-view windows (20 BurstGPT / 20 Azure conversation /
  20 Azure code), unchanged from `public_trace_replay_v1`
- GPU capacity held fixed at 512 / 512 / 8,000,000 for every cell
- Expected matrix size: 3,840 cells

## Local validation before any Wulver submission

- `pytest tests/test_public_replay_load_scaling_v1.py`: 20/20 passed
- Smoke (`scripts/run_public_replay_load_scaling_v1.py --smoke`): 2 windows
  x lambda in {1,16} x 2 policies, 8/8 cells succeeded, 0 failures;
  lambda=1 reproduced ANWG=1.0; lambda=16 increased `active_max` (7 -> 34 on
  window 0)
