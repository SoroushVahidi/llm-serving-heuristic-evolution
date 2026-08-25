# Public Replay Load Scaling v1 -- Frozen Design

**Status:** `FROZEN_BEFORE_OUTCOMES` (design frozen 2026-08-25, before any full-matrix cell was run)
**Experiment name:** `public_replay_load_scaling_v1`
**Repo HEAD at freeze time:** `2987b7181efa2bc550d8a894c537eca8f6393eb6` (worktree dirty; see `git status` at freeze time -- unrelated to this experiment, not touched)

## 1. Motivation / reviewer concern

`docs/current/public_trace_replay_v1_analysis_20260820.md` (classification
`PUBLIC_TRACE_NEAR_DEGENERACY`) established that the 60-window public-trace
replay is effectively unloaded under its frozen configuration:

- p99 active requests ~= 5 / 512 capacity (0.98%)
- max KV utilization ~= 0.0038 (0.38%)
- all six P6 policies tie exactly (ANWG = 1.0) in 60/60 windows
- oracle envelope gain over best-fixed = 0.0 in 60/60 windows

The reviewer concern this experiment answers: the public replay operates at
roughly 1% effective pressure and therefore cannot be a meaningful stress
test of scheduler differences -- the observed all-ties result may be an
artifact of insufficient load, not evidence that public workloads never
expose scheduler differences.

## 2. Relationship to the existing `public_trace_stress_v1` protocol

A prior experiment (`docs/current/external_baseline_stress_protocol_20260824.md`,
status `FROZEN_PUBLIC_TRACE_STRESS_V1`) already stress-tested the same 60
windows, but via a **confounded** two-lever transform: arrival compression
`M` (`t' = t/M`) **and** a simultaneous capacity cut `C` (from 512 down to
32), jointly calibrated via a policy-blind grid search over `M x C` to find
one point crossing a queue-positive threshold. That protocol answers "does
some feasible (M,C) combination expose scheduler separation" -- it does not
isolate arrival-rate scaling as a variable, and it does not sweep a range.

This experiment (`public_replay_load_scaling_v1`) is a **separate,
non-confounded** manipulation:

- capacity (`max_active_sequences`, `max_batch_tokens`, `max_kv_tokens`) is
  held at the base replay's original values (512 / 512 / 8,000,000) at
  **every** load factor;
- only inter-arrival timing is scaled, across a preregistered geometric
  grid, not a single calibrated point;
- the grid is fixed before any cell is run and is never adjusted based on
  outcomes.

Both experiments are valid and independently frozen. This document does not
supersede or invalidate `public_trace_stress_v1`; it isolates the load axis
that `public_trace_stress_v1` left confounded with capacity.

## 3. Primary manipulation (frozen)

For each of the 60 canonical augmented-view windows (unchanged from
`public_trace_replay_v1`; see section 4), for each request `i` with original
arrival time `t_i`:

```
t_start = min_i(t_i)              # per-window anchor, 0.0 for every canonical window
t'_i = t_start + (t_i - t_start) / lambda
```

Preserved exactly (never touched by the transform):

- request ordering (guaranteed by construction: the map `t -> t_start +
  (t - t_start)/lambda` is strictly increasing in `t` for `lambda > 0`, so
  relative arrival order is invariant)
- `prompt_tokens`, `actual_output_tokens`, `predicted_output_tokens`
- `class_id`, `priority`
- GPU capacity: `max_active_sequences=512`, `max_batch_tokens=512`,
  `max_kv_tokens=8,000,000` (identical `GPUConfig` object reused, not
  reconstructed)
- per-request deadline **slack** in absolute seconds:
  `slack_i = slo_deadline_i - t_i` is held fixed; the new deadline is
  `t'_i + slack_i`. This is not "scaling deadlines to create scheduler
  differences" -- slack (SLO tightness) is a non-arrival workload
  attribute, and holding it fixed in absolute seconds while the arrival
  anchor moves is what keeps it unchanged as a manipulation side effect.
  Not doing this (freezing the deadline's absolute clock time instead)
  would instead *loosen* every deadline as lambda grows, which would be a
  bigger and less defensible confound. This convention is identical to the
  one already frozen and used in `run_public_trace_stress_p6.py` /
  `run_public_trace_stress_external.py`.

No requests are dropped before simulation. No token lengths are scaled.

## 4. Canonical window source (audited, not resampled)

Authoritative source: `experiments/public_trace_replay_v1/`, specifically
the **augmented / controlled-annotation** evidence-class view (`AUGMENTED =
"PUBLIC_TRACE_DERIVED_WITH_CONTROLLED_ANNOTATIONS"`) produced by
`llmserveopt.policy_separation.public_trace_replay_v1.build_all_scenarios()`.

- `experiments/public_trace_replay_v1/layer2_scenario_manifest.json` (manifest)
- `experiments/public_trace_replay_v1/layer3_checkpoint.jsonl` (prior P6/faithful outcomes, used only for the lambda=1 reproduction check, never rerun or altered)
- `experiments/public_trace_replay_v1/layer3_checkpoint_integrity_report.json`
- `experiments/public_trace_replay_v1/layer3_provenance.json`
- `data/public_trace_corpus_v1/` (Layer 0/1 corpus)
- builder module: `src/llmserveopt/policy_separation/public_trace_replay_v1.py`

Frozen constants inherited unchanged (verified live against the module,
2026-08-25):

- `SOURCES = ("burstgpt", "azure_2023_conv", "azure_2023_code")`
- `WINDOW_SIZE = 200`
- `WINDOWS_PER_SOURCE = 20`  (=> 60 windows total, 20/20/20 confirmed)
- `SEED = 20260820`
- `PREDICTION_NOISE_SIGMA = 0.30`
- `SLACK_MULTIPLIER = 1.0`

The augmented view is used (not the 2-policy faithful view) because the
frozen 8-policy Pext portfolio (section 5) requires the controlled
annotations (`class_id`, `priority`, `slo_deadline`, `predicted_output_tokens`)
that only the augmented view provides. This is the same evidence-class
choice already used by `public_trace_stress_v1`.

No new trace windows were sampled. No substitution occurred.

## 5. Policy portfolio (frozen Pext, no new scheduler implemented)

Identified from `docs/current/external_baseline_fidelity_ledger_20260824.md`
(Pass-4, "Final common Pext definition (active)"):

```
Pext_common = P6 + official_vtc_joint_token_budget_remap + vllm_style_continuous_batching
```

P6 (six native policies):

1. `full_prefill`
2. `chunked_prefill_small`
3. `estimated_service_time_first` (ESTF)
4. `weighted_fair_share` (WFS)
5. `least_laxity_first` (LLF)
6. `kv_constrained_online` (KV)

External (2):

7. `official_vtc_joint_token_budget_remap` -- unmodified official VTC
   (`VTCReqQueue`, Sheng et al. OSDI 2024) via
   `baselines/vtc/adapter/simulator_policy.py`, with the disclosed
   token-budget-unit remap (Verdict A, faithful-and-necessary; fidelity
   ledger Pass-2). Frozen and validated: used in `public_trace_stress_v1`
   (72 Family-A cells, 360 P6-stress cells, 60/60 stressed-public cells).
8. `vllm_style_continuous_batching` -- `VLLMFaithfulPolicy`
   (`allow_chunked_prefill=False`, `decode_first=True`), a **simulator
   proxy** for vLLM's default FCFS continuous-batching scheduler, **not**
   native vLLM. Labeled as such everywhere in output and analysis. Native
   vLLM semantic validation is a separate, non-comparable artifact
   (`experiments/real_vllm_mechanism_validation_v1/`).

Excluded from this experiment (not implemented here, matching the ledger's
existing exclusions):

- SOLA -- `SOLA_NOT_FAITHFULLY_REPRODUCIBLE`, Related Work only.
- vLLM-LTR -- `LTR_NOT_IN_COMMON_PEXT_MATRIX`, requires prompt text the
  public-trace parquet corpus does not carry (counts only).

No new external scheduler is implemented in this experiment.

## 6. Preregistered load-factor grid (frozen)

```
lambda in {1, 2, 4, 8, 16, 32, 64, 128}
```

- `lambda = 1` reproduces the original public replay exactly (identity
  transform; see section 8 integrity check).
- Geometric spacing spans mild through severe load without a fine-grained
  search that could be tuned toward a favorable result.
- The top of the grid (128) is intentionally aggressive so the experiment
  is not stopped just short of a discriminative regime.
- The grid is not modified after seeing any scheduler outcome. If
  `lambda=128` fails for a genuine simulator-implementation-limit reason
  (as opposed to genuine overload), that will be reported explicitly, not
  silently replaced (see section 9).

## 7. Experiment matrix

```
60 windows x 8 load factors x 8 policies = 3,840 cells
```

Verified programmatically: `public_replay_load_scaling_v1.expected_cell_keys()`
returns exactly 3,840 unique keys (see
`tests/test_public_replay_load_scaling_v1.py::test_matrix_completeness_and_no_duplicates`).

Seeds: each transformed scenario reuses the base scenario's frozen seed
(`SEED=20260820`) unchanged -- no new stochastic replication is introduced.
The underlying simulator/policy evaluation for this workload is
deterministic given (scenario, policy) -- consistent with
`public_trace_replay_v1` and `public_trace_stress_v1`, neither of which used
repeated trials for this workload family. If any Pext policy is discovered
during execution to have hidden stochasticity, that will be reported before
reinterpreting results, per section L of the operating instructions; no such
stochasticity has been observed as of this freeze.

## 8. Metrics (collected per cell; ANWG/epsilon definitions not redefined)

Primary: `arrival_normalized_weighted_goodput` (ANWG), exactly as computed by
`llmserveopt.core.metrics.RunMetrics` / `metrics_to_dict` -- unchanged.

Performance: `completion_fraction`, `weighted_completion_fraction`,
`slo_violation_rate`, `mean/p95/p99 ttft`, `mean/p95/p99 latency`,
`request_throughput`, `token_throughput`, `num_completed`, `num_dropped`,
`num_total`, `sim_duration`.

System pressure (new instrumentation for this experiment, via a
non-decision-altering telemetry wrapper around the policy under test --
`_TelemetryWrapPolicy` in `public_replay_load_scaling_v1.py`): per-step
`active` count, `waiting` (queue length), `kv_util`, `active_util`;
aggregated to mean/p95/p99/max, plus `frac_steps_queue_positive`.

Portfolio-level (computed at post-completion analysis time, not per cell):
winner policy per window/lambda, epsilon-unique winner using the existing
`PRACTICAL_EPS = 0.01` convention from
`llmserveopt.analysis.public_trace_replay_v1_analysis`, per-window SBS/VBS,
VBS-SBS headroom. Epsilon and ANWG are not redefined by this experiment.

## 9. Integrity / sanity checks (mandatory, run before interpreting results)

1. `lambda=1` reproduces `public_trace_replay_v1` (ANWG=1.0, completion=1.0
   for all P6 policies on all 60 windows) within `1e-6` tolerance.
2. Request count unchanged across lambda (200/window, always).
3. Token-length distributions unchanged across lambda (identity per request).
4. Only arrival times (and dependent deadlines, via preserved slack) change.
5. Relative arrival ordering unchanged (guaranteed by the transform's
   monotonicity, verified by test).
6. No accidental deadline/SLO-tightness scaling (slack preserved exactly).
7. ANWG definition unchanged (reused verbatim from `core.metrics`).
8. No duplicate cells (`expected_cell_keys()` uniqueness check).
9. Expected matrix complete (3,840/3,840 cells present, each `status` ==
   `success` or explicitly `failed` with a captured error -- no silent
   gaps).
10. No NaN / serialization failures (every numeric field is a finite float
    or explicit `null`/`None` on failure).

If check 1 fails materially, the run stops before interpreting higher
lambda values (per operating instructions section L).

## 10. Preregistered qualitative verdicts (frozen before outcomes)

- `PUBLIC_LOAD_SCALING_REVEALS_POLICY_SEPARATION` -- higher preregistered
  loads produce reproducible nontrivial scheduler ranking diversity and
  positive VBS-SBS headroom.
- `PUBLIC_LOAD_SCALING_REMAINS_NONDISCRIMINATIVE` -- even lambda=128 yields
  little meaningful policy separation.
- `PUBLIC_LOAD_SCALING_ONLY_COLLAPSE` -- the grid transitions directly from
  trivial underload to universal overload/collapse with no useful
  discriminative region in between.
- `PUBLIC_LOAD_SCALING_INCONCLUSIVE` -- technical failures or
  trace/simulator incompatibility prevent interpretation.

These are not redefined after seeing results.

## 11. What this experiment does not do

- Does not tune lambda to find a favorable ranking (grid is fixed and
  exhaustively reported).
- Does not implement a new external scheduler.
- Does not modify `public_trace_replay_v1` or `public_trace_stress_v1`
  canonical outputs.
- Does not edit the manuscript.
- Does not redefine ANWG or the epsilon-unique-winner convention.

## 12. Artifacts

- Design (this file): `docs/design/PUBLIC_REPLAY_LOAD_SCALING_V1.md`
- Freeze pointer: `experiments/public_replay_load_scaling_v1/DESIGN_FROZEN.md`
- Core module: `src/llmserveopt/policy_separation/public_replay_load_scaling_v1.py`
- Runner: `scripts/run_public_replay_load_scaling_v1.py`
- Tests: `tests/test_public_replay_load_scaling_v1.py`
- SLURM array script: `scripts/slurm/public_replay_load_scaling_v1.sbatch`
