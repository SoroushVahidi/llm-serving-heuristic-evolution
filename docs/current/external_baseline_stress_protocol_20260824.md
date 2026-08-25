# External-Baseline Stressed Public-Trace Protocol

**Date:** 2026-08-24 (Pass 3 immutable freeze)  
**Workload name:** `public_trace_stress_v1`  
**Machine-readable:** `experiments/external_baseline_comparison_v1/stress_protocol.json`  
**Status:** `FROZEN_PUBLIC_TRACE_STRESS_V1` (immutable; do not recalibrate after seeing external utilities)

## Why Pass-1 M=32 alone was insufficient

At C=`max_active_sequences=512`, M=32 produced:

| Quantity | Value |
|---|---|
| mean max_active | ≈52 |
| active utilization | ≈52/512 ≈ 10% |
| queue-positive fraction | ≈0.072 (< 0.10 gate) |
| completion | ≈0.93 |

**Diagnosis (policy-blind):** (A)+(B) — arrival compression alone is too weak **while** active capacity remains unrealistically generous for 200-request windows (max concurrency ≪ 512). Not primarily a service-rate pathology: reducing C at fixed M immediately raises utilization and queue-positive fraction.

## Pass-2 capacity lever

Existing simulator controls only:

- `GPUConfig.max_active_sequences = C`
- `GPUConfig.max_batch_tokens = C` (count-cap field used by native `_feasible_on_gpu`)

Request contents unchanged (order, prompt/output tokens, identities).

## Predeclared selection rule (written before grid)

Grid: `M ∈ {8,16,32}`, `C ∈ {256,128,64,32}` (12 cells).  
Probe: **FIFO only**.  
Calibration subset: 4 windows × 3 sources = 12 (covers BurstGPT, Azure conv, Azure code).

A point **meets** iff all hold on the mean over the 12 windows:

1. `frac_steps_queue_positive ≥ 0.10`
2. `max_active/C ≥ 0.25` **OR** `p99_active/C ≥ 0.20`
3. `completion_fraction ≥ 0.80` (not catastrophic; predeclared, raised vs Pass-1’s 0.50)
4. all three sources present in the subset

**Pick:** among meeting points, **smallest M**, then **largest C** (lowest stress).  
**Forbidden for selection:** ANWG, winners, VTC/vLLM/P6 relative utilities.

## Calibration result

Meeting points included `(M=16,C=32)`, `(M=32,C=64)`, `(M=32,C=32)`.

**Selected:** **M=16, C=32**  
Reason: `lowest_stress_M_then_largest_C_satisfying_all_predeclared_gates`  
At selection: queue+≈0.124, util_max≈0.669, util_p99≈0.648, comp≈0.932.

Artifact: `experiments/external_baseline_comparison_v1/stress_calibration/pass2_MxC/calibration_summary.json`

## Frozen transformation

- Arrivals: `t' = t / 16` within each Layer-2 window (same indices as `public_trace_replay_v1`, seed 20260820, window size 200, 20 windows/source).
- Capacity: `max_active_sequences=32`, `max_batch_tokens=32`, `max_kv_tokens=8_000_000`.
- Overlays: augmented evidence class; `prediction_noise_sigma=0.30`; `slack_multiplier=1.0`; priorities/class as public_trace_replay_v1.
- Sources: BurstGPT, Azure 2023 conversation, Azure 2023 code (all 60 augmented windows for evaluation).

Confirmation: **no scheduler-result information used** to choose (M,C).

## Pass-3 immutable freeze

Confirmed no implementation/config bug in M×C calibration or stressed-public runners.

**Frozen point:** M=16, C=32.  
**Eval set:** 60 augmented Layer-2 windows (20× BurstGPT, Azure conv, Azure code); seed 20260820; window size 200.  
**Completed external cells under this freeze:** VTC 60/60; vLLM-style 60/60.

Do **not** change M/C after inspecting ANWG of those cells.
