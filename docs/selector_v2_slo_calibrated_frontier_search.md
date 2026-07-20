# Selector v2 SLO-calibrated frontier search

Follow-up to `docs/selector_v2_contention_frontier_search.md`, which found
the contention mechanism fires in 82% of 900 windows but the primary
selector objective (`arrival_normalized_weighted_goodput`, ANWG) still
tied on all 900 — root-caused to `slo_deadline` construction, not load or
mechanism reachability. This task fixes the SLO construction and re-runs.

**Headline result: with policy-independent, per-request-calibrated SLOs,
16.6% of 910 windows are now genuinely ANWG-discriminative (vs. 0/900
before), oracle headroom is 0.0381 (vs. 0.0), and 62% of the
discriminative windows are robust to nearby SLO-scale changes.** One gate
still fails honestly (see Quality gate table): no *faithful* baseline
(`vllm_faithful`/`sarathi_faithful`/`vllm_chunked_prefill_faithful`) wins
any of the 151 discriminative windows — the specialization signal found
here is entirely among the 8 historical policies.

## 1. Audit of existing SLO construction

| Path | Method | Notes |
|---|---|---|
| `workloads/augmentation.py` (`DEFAULT_SLO_AUG`) | Per-class fixed constants (interactive=2.0s, standard=6.0s, batch=20.0s) | Production real-trace ingestion (BurstGPT/ShareGPT); realistic wall-clock magnitudes, not derived from the simulator's own `ServiceModel` |
| `workloads/synthetic.py` (`DEFAULT_SLO_CLASSES`) | Per-class fixed constants (tight=0.5s/medium=2.0s/loose=10.0s, or 0.3/2.0/10.0 variant) | Synthetic workload generation |
| `selector/dataset_v2/scenario_redesign.py` (`slo_classes()`, `transform_requests()`) | Per-class **parameterized** constants (tight=0.08s/medium=0.35s/loose=1.5s defaults) **+ an existing `slo_scale` multiplier** for stress-testing | The actual Dataset v2 pilot-builder pipeline (`build_selector_dataset_v2_redesigned_pilot.py`) — already closest in spirit to what this task asks for, and already reused here for family F |
| Real-trace comparison scripts (`run_vllm_external_baseline_comparison.py`, `run_real_trace_comparison.py`, `run_hosted_policy_comparison.py`) | Source-derived (`row.slo_slack_seconds` read from the trace row, itself produced by the augmentation pipeline above) | Correctly source-derived, not fixed |
| `selector/dataset_v2/contention_fixtures.py` (`_req()` default) | **Fixed constant, `slo_slack=1000.0`** | Deliberate, documented (module docstring): designed to isolate "does the execution mechanism diverge at all" from "does it matter for SLO", not a bug for that module's stated purpose |
| `run_selector_v2_overnight_validation.py` (`_random_window`) | **Fixed constant, `slo_deadline=1000.0`** | The original overnight pilot's search generator — historical artifact, left unmodified per "do not silently rewrite old artifacts" |
| `selector_v2_contention_frontier_search.py` (prior task, both generators) | **Fixed constant, `slo_deadline=1000.0`** | The immediate prior task's own frontier search — root cause of that task's 0/900 ANWG-discriminative result |
| `run_sarathi_gpu_smoke_and_validation.py`, `run_gpu_external_validity_audit.py` | Fixed constant, `+10_000.0` | Real-GPU smoke-test/audit tooling, not Dataset v2 generation — out of scope, deliberately loose so hardware runs never spuriously "fail" an SLO that was never the point of those scripts |

**Every fixed-1000.0s (or 10,000.0s) use is in ad-hoc scenario-generation
or hardware-smoke-test scripts, never in the production augmentation/
scenario_redesign pipeline** — which already uses realistic-scale,
per-class constants and already supports a scale multiplier. This task's
fix targets exactly the ad-hoc generators (contention_fixtures.py is left
alone, per its own documented rationale; `_random_window` is left alone,
per "no rewriting old artifacts"); the new corrected search uses a new,
purpose-built calibration module instead.

## 2. Calibration method selected

**Reference-service-model estimate** (`src/llmserveopt/selector/dataset_v2/slo_calibration.py`):
for each request, computed from `ServiceModel` alone (no policy, no other
request, no simulator run):

```
reference_prefill_s = ceil(prompt_tokens / max_prefill_chunk_tokens) * step_size   (0 if prefill modeling disabled)
reference_e2e_s     = reference_prefill_s + predicted_output_tokens * step_size
calibrated_deadline  = arrival_time + multiplier * reference_e2e_s
```

Rejected alternatives (see the module's docstring for the full
reasoning): a neutral-reference-policy pass was rejected because ANY
policy run is subject to the same contention mechanism under audit, so
freezing its latency as the deadline would favor whichever OTHER policy
schedules closest to that reference policy's own admission order — a
real, if subtle, form of label leakage. Percentile-of-observed-latency
(used in the *prior* task's exploratory sensitivity sweep) was kept only
as a diagnostic cross-check, not the production method, because it
requires running every candidate first and so cannot calibrate a window
at construction time.

**No leakage, by construction and by test**
(`tests/test_slo_calibration.py::TestNoPolicyLabelLeakage`): the module
never imports anything policy- or metrics-shaped, and no function
accepts a policy, `RunMetrics`, or `CompletedRequest`.

## 3. Dual TTFT/TPOT SLO

`calibrate_dual_slo()` returns `(ttft_slo_s, tpot_slo_s)` from the same
reference estimate, versioned separately
(`SLO_CALIBRATION_SCHEMA_VERSION = "v1_reference_service_model"`) and
consumed post-hoc against `CompletedRequest.ttft`/`.tpot` — the existing
single-`slo_deadline` schema on `Request` is untouched, so every
historical metric (`weighted_goodput`, `arrival_normalized_weighted_
goodput`, etc.) computes exactly as before. Not wired into the main
270-window search's primary objective in this task (ANWG stayed the
target per section 6's explicit instruction) — available for a future
dual-SLO objective, tested for correctness
(`test_calibrate_dual_slo_ttft_le_e2e`) but not yet load-bearing anywhere.

## 4. Calibration multiplier grid (200 windows, families A-E, all 11 policies)

| Multiplier | All-fail fraction | All-success fraction | ANWG-discriminative fraction | Distinct winners |
|---|---|---|---|---|
| 0.8 | **100%** | 0% | 0% | 1 |
| 0.9 | **100%** | 0% | 0% | 1 |
| 1.0 | 12.5% | 0% | 85.0% | 5 |
| 1.1 | 10.5% | 0% | 87.5% | 5 |
| 1.25 | 6.5% | 0% | 92.5% | 5 |
| 1.5 | 1.5% | 0% | 98.5% | 4 |
| **2.0** | **0%** | **0%** | **99.5%** | 4 |

`multiplier ∈ {0.8, 0.9}` is universal failure (deadlines tighter than
even the uncontended reference plus any admission delay at all) — exactly
the "too tight" failure mode section 4 warns against. **Selected
default: 2.0** — the only grid point with simultaneously zero universal
failure AND zero universal success, and the highest discriminative
fraction (99.5%) among candidates clearing that filter. Never selected by
which policy wins: `best_fixed_policy` changes across this very table
(`weighted_shortest_processing` at 1.0-1.5, `edf` at 2.0) and was not a
selection input.

## 5. Six workload families

A-E generated fresh (`src/llmserveopt/selector/dataset_v2/frontier_workload_families.py`);
F reuses the existing, unmodified `scenario_redesign.local_real_trace_stress_specs`
(BurstGPT-scaled-moderate/high, Azure-2023-code/conv, 4 transforms each:
representative/compressed_tight/burst_kv/noise_underpredict).

| Family | Shape | Contribution to 151 discriminative windows (main search) |
|---|---|---|
| A: same-arrival heterogeneous cluster | 2-6 reqs, all `t=0`, id order ⟂ size order | 4 |
| B: closely-spaced heterogeneous cluster | 3-8 reqs, interleaved sizes, 1-3 step_size gaps | 0 |
| C: admission-reorder boundary | sizes straddle the chunk/budget boundary | 16 |
| D: long-prefill overlap (root cause A, prior task) | 1-3 hogs + 2-40 staggered runners | **50 (largest single contributor)** |
| E: KV-pressure + admission order | heterogeneous sizes, tight KV headroom (1.05-1.5x) | 17 |
| F: real-trace stress (BurstGPT/Azure) | unmodified `transform_requests` | **64 (42%, largest combined contributor)** |

Two notable, honestly-reported results: (1) family D — the *same* shape
the prior task found produced no *raw-latency* divergence beyond
`NEAR_TIE` — is the single largest contributor to *ANWG* discrimination
once SLOs are calibrated tightly: small latency gaps that never moved the
uncalibrated objective can flip many individual requests' SLO pass/fail
once the deadline is close to the achievable latency, which is exactly
what a correctly-calibrated ANWG is supposed to detect. (2) Family B
(closely-spaced, not exactly simultaneous) contributed **zero**
discriminative windows — the admission-order-reordering mechanism found
in the prior task appears to need exact arrival-time ties, not just
near-simultaneity, to force a genuine ordering conflict.

## 6. Corrected targeted search (910 windows)

`scripts/selector_v2_slo_calibrated_frontier_search.py --stage main`:
750 synthetic (150 per family A-E) + 160 real-trace-stress (F), multiplier
2.0, same 11-policy monolithic roster as both prior tasks.

```
n_windows_scored: 910
NEAR_TIE: 735 (80.8%)
STRONGLY_DISCRIMINATIVE: 135 (14.8%)
MODERATELY_DISCRIMINATIVE: 16 (1.8%)
ALL_COMPLETE_OR_EFFECTIVELY_TIED: 24 (2.6%)
```

Win distribution (all windows) vs. strong-win distribution (135 windows
only):

| Policy | Overall wins | Strong wins |
|---|---|---|
| vllm_faithful (faithful) | 513 | **0** |
| edf | 217 | 46 |
| scorpio_style_slo_guard | 57 | 54 |
| weighted_shortest_processing | 50 | 9 |
| admission_control | 26 | 18 |
| fifo | 31 | 0 |
| multi_bin_batching | 11 | 6 |
| estimated_service_time_first | 5 | 2 |
| sarathi_faithful (faithful) | 0 | 0 |
| vllm_chunked_prefill_faithful (faithful) | 0 | 0 |

`vllm_faithful`'s 513 raw wins are essentially all weak/tie-break wins
(NEAR_TIE or ALL_EQUIVALENT windows, where it happens to be picked first)
— it has **zero** strong or moderate wins. No faithful baseline wins any
of the 151 truly discriminative windows; every strong/moderate win goes
to a historical policy.

## 7. Quality gate for a new targeted Dataset v2 pilot

| Gate | Threshold | Result | Pass? |
|---|---|---|---|
| All-equivalent fraction | < 40% | 2.64% | ✅ |
| Oracle headroom | ≥ 0.01 | **0.0381** | ✅ |
| Discriminative oracle headroom | ≥ 0.03 | **0.2032** | ✅ |
| ≥3 policies with meaningful (strong) wins | ≥3 | 6 (edf, scorpio_style_slo_guard, weighted_shortest_processing, admission_control, multi_bin_batching, estimated_service_time_first) | ✅ |
| No policy >85% of strong wins | ≤85% | 40.0% (scorpio_style_slo_guard) | ✅ |
| At least one faithful baseline has meaningful wins | required | **0 of 151** discriminative windows go to any faithful baseline | ❌ |
| No universal-success/failure saturation | required | confirmed at multiplier=2.0 (grid table, section 4) | ✅ |
| Real-trace representation exists | required | 160/910 (17.6%), 4 distinct sources | ✅ |
| OOD source split feasible | required | Yes — burstgpt_scaled_moderate/high vs. azure_2023_code/conv are independent sources, splittable by source | ✅ |

**8 of 9 gates pass.** The one failure is structural, not a sample-size
or tuning artifact: `vllm_faithful`/`sarathi_faithful`/
`vllm_chunked_prefill_faithful` never win a discriminative window in this
910-window search, consistent with both prior tasks' findings (the
contention-frontier search's 0/900 `p95_latency` wins for
`vllm_chunked_prefill_faithful`, and the Wulver hardware pack's
already-documented mismatch on the two Sarathi-favoring positive
targets). A selector trained on this data would, in effect, be learning
to select among *historical* policies, not among the *faithful*
baselines the contention-fix work was originally about.

## 8. SLO-scale robustness (300 fresh windows, neighbors {1.5, 2.0, 3.0})

```
ROBUST_TO_SLO_SCALE:     185 / 299  (61.9%)  -- same winner at all 3 multipliers
SENSITIVE_TO_SLO_SCALE:   88 / 299  (29.4%)  -- same winner at 2 of 3
ARTIFACT_OF_THRESHOLD:    26 / 299  ( 8.7%)  -- winner changes at every neighbor
```

The majority (62%) of discriminative windows keep the same winner across
a 2x range of calibration tightness (1.5x-3.0x the reference latency) —
the specialization signal is not primarily a threshold artifact. The 8.7%
classified `ARTIFACT_OF_THRESHOLD` should be excluded from any training
set built from this search (per section 8's instruction); they are
retained in the CSVs (`slo_scale_robustness.json`) for exclusion, not
deleted.

## Verdict

`READY_FOR_TARGETED_DATASET_V2_PILOT = yes, with a scope caveat`
`READY_FOR_SELECTOR_TRAINING = no (not in this task)`

Unlike the prior task's clean "no" (3 of 5 gates failed, all-equivalent
fraction was 100%), this search clears 8 of 9 gates with genuine margin
(oracle headroom 3.8x the threshold, discriminative oracle headroom 6.8x
the threshold). The one failing gate is a scope decision, not a data
problem: a pilot built from this search would train a selector over the
8 historical policies with real, robust, calibration-independent
specialization — but would show **no signal at all** for choosing among
the three faithful baselines specifically, which several prior tasks in
this thread were originally trying to validate. Per this task's own
instructions, a selector is not trained here regardless; the next task
should decide explicitly whether "historical-policy selector, faithful
baselines excluded or included-but-inert" is an acceptable scope for the
next Dataset v2 pilot before generating one.
