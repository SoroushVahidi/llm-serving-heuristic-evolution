# Public Replay Load Scaling v2 -- Implementation-Correction Record

**Status:** `IMPLEMENTATION_CORRECTION_OF_V1 -- NOT A NEW SCIENTIFIC DESIGN`
**Experiment name:** `public_replay_load_scaling_v2`
**Repo HEAD at freeze time:** see `experiments/public_replay_load_scaling_v2/DESIGN_FROZEN.md`

## 0. What this document is (and is not)

**v2 is NOT a new outcome-driven scientific design.** It does not change the scientific
question, the manipulation, the metrics, the bootstrap procedure, or the verdict rules from
`docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md`, which remains the authoritative statement of
the scientific preregistration and is **incorporated verbatim by reference** for every section
below except the one named in section 2.

v2 exists because `public_replay_load_scaling_v1` was found, via its own preregistered
mandatory λ=1 reproduction gate (design §9 item 1), to contain a **simulation-harness
implementation bug** unrelated to the scientific manipulation: see
`experiments/public_replay_load_scaling_v1/integrity_report.json` and
`docs/current/public_replay_load_scaling_v1_analysis_20260825.md` for the full diagnosis. That
finding is a **positive reproducibility/audit result** -- the preregistered integrity gate
caught the bug before any scientific interpretation was attempted. v1's raw outputs, integrity
report, and analysis are preserved unmodified as part of the permanent audit trail; nothing in
v1 is deleted, silently replaced, or "repaired in place."

## 1. The v1 bug (restated for the record)

`evaluate_cell()` in `public_replay_load_scaling_v1.py` hardcoded:

```python
SIM_MAX_STEPS = 200_000
SIM_DRAIN_STEPS = 50_000
...
SimulatorConfig(..., max_steps=SIM_MAX_STEPS, drain_steps=SIM_DRAIN_STEPS)
```

Combined with the base scenario's `service_model_kwargs["step_size"] = 0.001` (inherited
unchanged from `public_trace_replay_v1` at every λ), this imposed an effective **≈200-210
simulated-second wall-clock ceiling, independent of λ**. This is a harness/engineering defect,
not a scientific manipulation: it silently truncated 256/3840 (6.67%) of v1's cells (concentrated
in 11/60 windows, worst case `burstgpt::w0` with a 34,064s real span, corrupted at **every**
preregistered λ including 128), and caused the mandatory λ=1 reproduction check to fail
materially (66/360 P6 cells, max ANWG error 0.98 vs. the required ANWG=1.0 / ≤1e-6 tolerance).

## 2. The only change from v1 (frozen)

Remove the hardcoded ceiling: `SIM_MAX_STEPS = None` (unbounded), matching the **authoritative**
`public_trace_replay_v1.evaluate_scenario_policy` exactly, which constructs
`SimulatorConfig(gpu_configs=..., service_model=...)` with **no** `max_steps`/`drain_steps`
override and therefore uses `SimulatorConfig`'s own default (`max_steps: Optional[int] = None`
-- `src/llmserveopt/simulator/simulator.py:53`). `SIM_DRAIN_STEPS` remains `50_000`, unchanged
from v1 (this is also the `SimulatorConfig` default, and the authoritative harness never
overrides it either, so this was never part of the bug).

**Nothing else changes.** In particular, unchanged from v1 and from
`docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md`:

- the 60 canonical augmented-view windows (20/20/20 BurstGPT/Azure-conv/Azure-code), unresampled
- the load-factor grid `{1, 2, 4, 8, 16, 32, 64, 128}`
- the 8-policy Pext portfolio (6 native P6 + `official_vtc_joint_token_budget_remap` +
  `vllm_style_continuous_batching`)
- GPU capacity (512 / 512 / 8,000,000) at every λ
- `step_size` and every other `service_model_kwargs` value
- the arrival-only transform (`t'_i = t_start + (t_i - t_start)/λ`, slack preserved in absolute
  seconds) -- request ordering, token lengths, `class_id`/`priority`, request counts, and
  deadline-slack convention are untouched, exactly as in v1 design §3
- the 3,840-cell matrix (60 × 8 × 8)
- ANWG and every other metric definition (§8 of the v1 design)
- the epsilon-unique-winner convention (`PRACTICAL_EPS = 0.01`)
- the bootstrap procedure and all four preregistered qualitative verdict labels (§10 of the
  v1 design)
- the integrity/sanity checks (§9 of the v1 design), now including the corrected termination
  semantics as an explicit regression target (see `tests/test_public_replay_load_scaling_v2.py`)

**Explicitly NOT done** (per the correction's own scope, and per instruction):

- no larger arbitrary `max_steps` was substituted for the removed one
- no per-load or per-window `max_steps` was derived from outcomes
- `step_size` was not changed
- the simulation horizon was not tuned heuristically
- no request arrival time was changed beyond the frozen λ transform
- no deadline, capacity, or scheduler parameter was changed
- the load-factor grid, policy set, and window set were not changed

## 3. Regression tests (new, mandatory)

`tests/test_public_replay_load_scaling_v2.py` adds tests that would have caught the v1 bug,
in addition to re-running the full v1 test suite (transform correctness, matrix completeness,
ordering/slack invariants) against the v2 module:

1. A long-span window (>200 simulated seconds at λ=1) completes with `num_completed +
   num_dropped == num_total` -- no requests silently left unprocessed.
2. `PUBLIC_TRACE::burstgpt::w0::augmented` (the worst v1 failure) reproduces the authoritative
   `ANWG=1.0`, `completion_fraction=1.0` at λ=1, within `1e-6`.
3. No cell in a `λ=1` sweep over all 11 windows that failed v1's reproduction check is
   truncated (`num_completed + num_dropped == 200` for all of them).
4. λ=1 is the identity transform (arrival times/deadlines unchanged to floating-point
   precision).
5. Request-count invariance (`n_requests_scaled == n_requests_base == 200`) across λ.
6. Ordering invariance (transformed arrival times remain non-decreasing; monotonicity of the
   transform proved algebraically and spot-checked numerically).
7. Token-length invariance (`prompt_tokens`, `actual_output_tokens`, `predicted_output_tokens`
   identical pre/post transform, at every λ).
8. Deadline-slack invariance (`slo_deadline - arrival_time` constant across λ, to floating-point
   precision).
9. Capacity invariance (`max_active_sequences`/`max_batch_tokens`/`max_kv_tokens` identical
   across λ and identical to the base replay's values).
10. Load scaling touches only `arrival_time`/`slo_deadline`; `class_id`, `priority`,
    `request_id` are unchanged objects/values.
11. `SIM_MAX_STEPS is None` (a direct assertion on the v2 module constant, so any future
    reintroduction of a hardcoded ceiling fails CI immediately).

## 4. Full-matrix execution plan

Identical structure to v1: one SLURM array task per canonical window (60 tasks), each task
evaluating all 8×8=64 (λ, policy) cells for its window, writing
`cells_window_{idx:03d}_{source}.jsonl`. Output lands in a **fresh path**,
`/mmfs1/scratch/ikoutis/sv96/llm-serving-heuristic-evolution/public_replay_load_scaling_v2/`
-- v1's remote/local outputs are never overwritten or reused.

Before submitting the full matrix:

- **λ=1 pre-full-run gate** (local, all 60 windows × 8 policies): every cell must reproduce
  the authoritative `public_trace_replay_v1` checkpoint (`ANWG`, `completion_fraction`,
  `num_completed`) within `1e-6`, with zero truncated cells. If this fails materially, the
  full matrix is **not** submitted; the failure is reported instead (same stop-gate discipline
  as v1's own design).
- **High-load smoke**: at least one long BurstGPT window, λ ∈ {1, 16, 128}, ≥2 policies --
  confirms natural termination, no artificial horizon, reasonable runtime/memory, correct
  serialization. Not interpreted scientifically.

## 5. Analysis plan (unchanged from v1, gated on integrity)

Once the full v2 matrix completes, the same mandatory post-completion gates from v1 design §9
apply (cell count, duplicates, NaNs, λ=1 reproduction, truncation, invariances). **Only if all
gates pass** does load-curve interpretation proceed, using the exact same preregistered
questions and the exact same four frozen verdict labels as v1 design §10:

- `PUBLIC_LOAD_SCALING_REVEALS_POLICY_SEPARATION`
- `PUBLIC_LOAD_SCALING_REMAINS_NONDISCRIMINATIVE`
- `PUBLIC_LOAD_SCALING_ONLY_COLLAPSE`
- `PUBLIC_LOAD_SCALING_INCONCLUSIVE`

These are not redefined after seeing v2 results, exactly as v1 design §10 already commits.

## 6. Provenance language (mandatory)

Any report or manuscript text describing this line of work must state explicitly that
`public_replay_load_scaling_v1` was invalidated for scientific interpretation by a
simulation-horizon implementation bug, detected through the preregistered λ=1 reproduction
gate, and that this is a positive reproducibility/audit outcome (the gate worked as designed),
not a concealed failure. v1's existence, its integrity report, and its `INCONCLUSIVE` verdict
are part of the permanent record and are not to be omitted from citations of this experiment
line.

## 7. Artifacts

- Design (this file): `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V2.md`
- Scientific design of record (unchanged, incorporated by reference): `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md`
- Freeze pointer: `experiments/public_replay_load_scaling_v2/DESIGN_FROZEN.md`
- Core module: `src/llmserveopt/policy_separation/public_replay_load_scaling_v2.py` (minimal diff vs. v1 -- see module docstring)
- Runner: `scripts/run_public_replay_load_scaling_v2.py`
- Tests: `tests/test_public_replay_load_scaling_v2.py`
- SLURM array script: `scripts/slurm/public_replay_load_scaling_v2.sbatch`
- v1 audit trail (preserved, unmodified): `experiments/public_replay_load_scaling_v1/`,
  `docs/current/public_replay_load_scaling_v1_analysis_20260825.md`
