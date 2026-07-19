# Wulver vLLM KV-Pressure Results: Jobs 1111541 and 1111545

Consolidated analysis of the two completed Wulver A100 vLLM validation runs
requested by `docs/wulver_gpu_validation_handoff.md`. This is a
validation/calibration report only: it does not train a selector, does not
generate Selector Dataset v2 data, and does not change historical simulator
defaults.

## 1. Jobs, hardware, model, software

| | Baseline (1111541) | Aggressive stress (1111545) |
|---|---|---|
| Slurm job | 1111541 | 1111545 |
| Node | n0003 | n0026 |
| Branch / commit | `wulver-gpu-validation-ready` @ `42635412a5dc441a514c94e84d89c8e99d3debfc` | same |
| GPU | NVIDIA A100-SXM4-80GB, 81920 MiB | same |
| Driver | 580.159.04 | same |
| vLLM | 0.24.0 | same |
| Model | `Qwen/Qwen2.5-7B-Instruct`, dtype bfloat16 | same |
| `max-model-len` | 16384 | 16384 |
| `gpu-memory-utilization` | 0.82 | 0.35 |
| `max-num-seqs` | 8 | 32 |
| `max-num-batched-tokens` | 2048 | 4096 |
| `block-size` | 16 | 16 |
| `enable-chunked-prefill` / `enable-prefix-caching` | on / off | on / off |
| Raw output dir (Wulver scratch, not in git) | `/mmfs1/scratch/ikoutis/sv96/vllm_kv_pressure_1111541` | `/mmfs1/scratch/ikoutis/sv96/vllm_kv_pressure_aggressive_1111545` |
| Canonical copy in repo | `experiments/gpu_external_validity/vllm_qwen7b_a100_baseline_1111541/` | `experiments/gpu_external_validity/vllm_qwen7b_a100_aggressive_1111545/` |
| Exact server command | see `job_manifest.txt` / `environment.json` in canonical dir | same |
| Exact audit command | `scripts/slurm/wulver_vllm_kv_pressure_verified_run1.sbatch` | `scripts/slurm/wulver_vllm_kv_pressure_aggressive_stress.sbatch` |

Both runs used `scripts/run_gpu_external_validity_audit.py --phase stress
--start-vllm-server ... --write-calibration-profile --resume` against the same
6 built-in stress scenarios, so the two runs form a controlled A/B: the
aggressive run isolates the effect of (a) removing the `max-num-seqs` cap and
(b) shrinking the KV memory budget, with everything else held fixed.

### Verification (section 1 of the task)

- Both runs: 6/6 scenarios completed, `completion_fraction = 1.0` in every row
  of `scenario_summary.csv`.
- Both runs: `num_requests == num_success == 108` (24+16+16+12+16+24 across the
  6 scenarios, sums verified against `summary.json`).
- No hidden partial failures: `requests.jsonl` has exactly 108 lines with
  `status: "success"` in both runs; no `error` fields set.
- `server.log` in both runs shows one `EngineDeadError` traceback, but it
  occurs after the final `/metrics` poll, immediately following the
  `SIGTERM`/`shutdown mode=abort` sequence in both logs — a benign artifact of
  aborting the engine on shutdown, not a mid-run failure. No other
  errors/tracebacks appear in either log.
- Git commit, GPU model, and exact vLLM server command are recorded
  identically in `job_manifest.txt` and `environment.json` for both runs and
  match each other (only the two intentionally-varied flags and port differ).

## 2. Run-level metrics

| Metric | Baseline (1111541) | Aggressive (1111545) |
|---|---:|---:|
| Scenarios | 6 | 6 |
| Requests | 108 | 108 |
| Completions | 108 | 108 |
| Completion fraction | 1.0 | 1.0 |
| Max running sequences | 8 | 24 |
| Max waiting queue | 16 | 10 |
| Max KV cache usage | 2.2173% | 13.9594% |
| Preemption events | 0 | 0 |
| Recompute events | 0 (not observed/logged) | 0 (not observed/logged) |
| Swap events | not observable (vLLM v1 engine does not expose a swap counter here; not present in server.log or metrics) | same |
| TTFT mean / p50 / p95 / p99 (s) | 2.095 / 1.877 / 5.323 / 9.407 | 0.688 / 0.575 / 1.418 / 1.673 |
| TPOT mean / p50 / p95 / p99 (s/token) | 0.01202 / 0.01141 / 0.01690 / 0.01799 | 0.01365 / 0.01313 / 0.02060 / 0.02518 |
| E2E latency mean / p50 / p95 / p99 (s) | 4.854 / 3.569 / 9.727 / 17.544 | 3.821 / 2.772 / 10.226 / 10.245 |
| Throughput range (req/s, per-scenario) | 0.679 – 6.862 | 1.170 – 11.417 |

Percentiles for TTFT/TPOT/E2E are computed by pooling all 108 per-request
records in `requests.jsonl` (mean values agree with the officially recorded
`summary.json` mean-of-scenario-means to within ~2%). vLLM/the audit harness
does not report recompute or swap counters directly; `preemption_events_delta`
is the only eviction-related counter exposed by the vLLM Prometheus metrics
this harness polls, and it was 0 in every scenario of both runs.

### Baseline vs. aggressive

- **KV utilization multiplier**: 0.139594 / 0.022173 ≈ **6.30x** higher peak
  KV usage in the aggressive run.
- **Concurrency multiplier**: 24 / 8 = **3.0x** higher max running sequences.
- **TTFT difference**: mean TTFT dropped from 2.095s to 0.688s, a **67.2%
  reduction** (3.05x lower).
- **E2E latency difference**: mean latency dropped from 5.191s to 4.192s
  (officially recorded means), a **19.2% reduction**.
- **Throughput difference**: per-scenario throughput roughly **1.7x–1.9x
  higher** at both ends of the range in the aggressive run.
- Preemption events: **0 in both runs** — the aggressive change increased KV
  pressure and concurrency substantially but did not approach the point of
  forcing eviction.

## 3. Why no preemption occurred

vLLM prints its actual KV pool size and theoretical max concurrency at
startup; both are captured verbatim in `server.log`:

| | Baseline (1111541) | Aggressive (1111545) |
|---|---:|---:|
| Available KV cache memory | 49.45 GiB | 12.12 GiB |
| GPU KV cache size (pool) | 925,824 tokens | 226,960 tokens |
| vLLM-reported max concurrency @ 16,384-token requests | 56.51x | 13.85x |
| Empirically observed peak KV usage | 2.2173% | 13.9594% |
| **Headroom ratio** (`1 / peak_kv_usage`) | **≈45.1x** | **≈7.16x** |

Even in the aggressive configuration — 6.3x smaller KV pool, seq cap raised
3x, batched-token budget doubled — peak KV demand only reached ~14% of the
available pool, i.e. there was still >7x headroom at the moment of highest
observed pressure (the `stress_kv_pressure` scenario: mean prompt 1,873 +
mean output ~768 tokens ≈ 2,641 tokens/request at up to 12 concurrent
requests). The workload's realized token footprint per request (peaking
around 2,600 tokens) stayed far below `max-model-len` (16,384), so no
combination of scenarios in this suite pushed the engine anywhere near its
eviction threshold.

**Classification: `PREEMPTION_NOT_REACHED_DUE_TO_HEADROOM`.**

This is a headroom problem, not an untested code path in the sense of "we
don't know what would happen" — the KV pool size, request footprint, and
observed usage are all directly measured, and the gap between demand and
capacity is large and consistent across two independently configured runs.
We do not claim vLLM's preemption/recompute logic was exercised or validated;
it was not.

## 4. Comparison against the simulator

The audit harness runs matched simulator traces (`vllm_faithful` and
`sarathi_faithful`) for the same 6 scenarios inline with each GPU run, stored
per-scenario in `scenario_results.json`. No separate simulator invocation was
needed; the comparison below uses those matched runs directly.

| Scenario | Runtime TTFT (base / aggr, s) | Sim TTFT (s) | Runtime E2E (base / aggr, s) | Sim E2E (s) | Runtime max running (base / aggr) |
|---|---|---:|---|---:|---|
| high_concurrency_queue | 2.290 / 1.413 | 0.0014 | 2.846 / 2.080 | 0.128 | 8 / 24 |
| long_decode_kv | 1.154 / 0.161 | 0.0011 | 3.397 / 2.651 | 0.512 | 8 / 16 |
| long_prefill | 1.430 / 0.999 | 0.0085 | 2.925 / 2.765 | 0.104 | 8 / 16 |
| kv_pressure | 3.390 / 0.786 | 0.0065 | 12.014 / 10.225 | 0.773 | 8 / 12 |
| mixed_prefill_decode | 1.248 / 0.338 | 0.0028 | 3.577 / 3.103 | 0.322 | 8 / 16 |
| burst_overload_recovery | 3.059 / 0.428 | 0.0043 | 6.388 / 4.324 | 0.291 | 8 / 24 |

Critically, the `sim_vllm_*` numbers are **identical between the baseline and
aggressive runs for every scenario** (e.g. `stress_kv_pressure` sim latency is
0.7735s in both). The simulator was not reconfigured between runs — it has no
parameter tied to `--max-num-seqs` or `--gpu-memory-utilization` in this
harness, so it cannot, by construction, respond to the exact GPU-side changes
that produced the 3x concurrency increase and 6.3x KV-utilization increase in
the real runtime.

Dimension classification:

- **A. Qualitative saturation behavior — `PARTIALLY_VALIDATED`.** The
  simulator's relative ordering of scenarios by request size (long-prefill and
  kv-pressure scenarios cost more than decode-only ones) points the same
  direction as the runtime, but the simulator shows no queue buildup or
  concurrency-driven latency growth at all — its outputs look like isolated
  per-request service times, not a loaded system.
- **B. Relative effect of removing the seq cap — `NOT_VALIDATED`.** Runtime:
  3.0x more concurrency, 67% lower TTFT. Simulator: zero change, because the
  simulator was not driven with the corresponding config change.
- **C. Relative effect of shrinking the KV budget — `NOT_VALIDATED`.**
  Runtime: 6.3x higher KV utilization. Simulator: zero change, same reason.
- **D. Queueing-induced TTFT growth — `NOT_VALIDATED`.** Runtime TTFT ranged
  0.16s–3.39s across scenarios/configs and tracked `max_vllm_waiting` (0–16).
  Simulator TTFT stayed in the 0.001s–0.009s range regardless of queue depth —
  it does not reproduce queueing delay at all for these scenario definitions.
- **E. Actual timing scale — `NOT_VALIDATED`.** Runtime/simulator latency
  ratio (median) was 18.7x in the baseline and 14.0x in the aggressive run —
  consistently large and in the same direction (simulator underestimates),
  but a full order of magnitude off, and the ratio itself shifts with
  queueing rather than being a fixed constant.

## 5. Calibration conclusion

The per-run `calibration_profile.json` prefill fits are **not** stable between
runs: intercept 1.699s (baseline) vs. 0.622s (aggressive), a 2.7x difference;
slope 3.90e-4 vs 6.46e-5 s/token, a 6x difference. This instability has a
direct cause: baseline TTFT is dominated by scheduler queueing behind
`max-num-seqs=8`, not by GPU prefill compute, so the "prefill fit" is really
fitting queueing delay in that run.

Decode throughput, by contrast, was stable: 0.01213 s/token (baseline) vs.
0.01365 s/token (aggressive), a 12.5% spread — plausible run-to-run variation
at different batch occupancy, not a config artifact.

**Decision: create an opt-in calibration profile, scoped to what was actually
stable.** `configs/calibration/wulver_a100_qwen25_7b_vllm024.yaml` records the
decode-step point estimate (~0.0129 s/token) as usable, and explicitly labels
the prefill/TTFT fit and the runtime/simulator ratio as queueing-contaminated
and not safe to reuse directly. It does not modify any historical simulator
default and is not wired into any default code path.

## 6. Scientific interpretation

Verified against the artifacts above:

- On A100 80GB with `gpu-memory-utilization=0.82`, the default-ish baseline
  config had substantial KV headroom for this workload: peak usage 2.22% of a
  925,824-token pool (~45x headroom).
- Shrinking the KV budget by ~4.1x (0.82 → 0.35 `gpu-memory-utilization`, pool
  925,824 → 226,960 tokens) increased peak observed utilization from 2.22% to
  13.96% (6.3x), but still left ~7.16x headroom — it did not force preemption.
- `max-num-seqs=8` induced queueing in the baseline (max waiting queue 16,
  4/6 and in fact 6/6 scenarios showed nonzero waiting); removing the cap
  (`max-num-seqs=32`) in the aggressive run raised max observed concurrency
  from 8 to 24 and cut mean TTFT by 67%, while max waiting queue dropped from
  16 to 10 and only 4/6 scenarios showed nonzero waiting.
- Therefore the queueing observed in the baseline run was **sequence-cap
  driven, not memory-pressure driven** — KV usage was low (2.22%) in the exact
  same run where queueing was worst, ruling out memory pressure as the cause.

## 7. Is another vLLM run needed?

**Recommendation: C — one future longer-context run is justified. Not
submitted.**

Reasoning: the binding constraint on reaching real KV pressure/preemption is
token footprint per concurrent request, not request count. vLLM's own
diagnostic in the aggressive run's `server.log` states "Maximum concurrency
for 16,384 tokens per request: 13.85x" — i.e. only ~14 concurrent
full-context (16,384-token) requests would exhaust that 226,960-token pool.
The scenarios actually run used a mean prompt+output footprint of at most
~2,641 tokens (`stress_kv_pressure`), roughly 16% of the context window, and
needed ~86 concurrent requests of that size to fill the pool — a much larger,
harder-to-generate concurrent load than the ~14 needed at near-max context.
A workload-volume increase (option B) would require pushing concurrent client
load roughly 3.6x past what was already demonstrated (24 → ~86-90) and would
likely first hit `max-num-seqs` or client-side scheduling limits rather than
memory pressure. A longer-context run needs only a ~3x increase in per-request
token length at already-demonstrated concurrency (24-32) to plausibly close
the remaining ~7.16x headroom gap, and directly exercises the `max-model-len`
budget this deployment was already configured for (16,384) rather than
requiring new infrastructure or a larger model. This is not a "GPUs are
available" recommendation — it is the more efficient of the two available
levers, quantified from data already collected in these two runs.

## 8. Preemption limitation

**`PREEMPTION_NOT_REACHED_DUE_TO_HEADROOM`** — see section 3. Do not read
these results as validation, positive or negative, of vLLM's
preemption/recompute scheduling logic. That code path was never exercised on
Wulver.

## 9. Safe claims

- On A100 80GB, this default-ish vLLM config (`gpu-memory-utilization=0.82`,
  `max-num-seqs=8`) had ~45x KV headroom for this workload; KV pressure was
  never a binding constraint.
- Shrinking the KV budget ~4.1x increased peak KV utilization 6.3x (2.22% →
  13.96%) without forcing preemption; ~7.16x headroom remained.
- The baseline's queueing (max waiting queue 16) was caused by the
  `max-num-seqs=8` cap, not by KV memory pressure, since KV usage was only
  2.22% during that same run.
- Removing the seq cap (8 → 32) increased observed concurrency 3.0x and cut
  mean TTFT 67.2%, with everything else held fixed.
- The faithful-baseline simulator, as invoked in this harness, does not
  respond to `max-num-seqs` or `gpu-memory-utilization` changes and therefore
  cannot reproduce the queueing/KV-pressure effects measured on real hardware
  in this study.
- The simulator underestimates real vLLM runtime latency by roughly an order
  of magnitude (14x–19x median) on this hardware/model/workload; the ratio is
  not constant and moves with scheduler queueing.
- Decode-step latency (~0.0129 s/token ± 12.5%) was the one runtime quantity
  that was consistent across both differently-configured runs.

## 10. Unsafe claims

- Do NOT claim vLLM preemption or KV recompute was validated, exercised, or
  ruled out as a scheduling concern — it was never triggered on this
  hardware/workload combination.
- Do NOT claim the simulator captures queueing-induced TTFT growth, the
  effect of the seq-count cap, or the effect of KV budget size — all three
  were NOT_VALIDATED in this study.
- Do NOT apply the runtime/simulator latency ratio (14x-19x) as a fixed
  correction multiplier — it varies with scheduler configuration.
- Do NOT reuse either run's `prefill_latency_fit` (from `calibration_profile.json`)
  as a hardware-intrinsic prefill cost model — both are contaminated by
  scheduler queueing and disagree with each other by 2.7x-6x.
- Do NOT generalize these findings to other models, GPUs, vLLM versions, or
  much longer context lengths — the workload used here never exceeded ~1,873
  mean prompt tokens against a 16,384-token window.

## Implication for Selector Dataset v2

`docs/selector_dataset_v2.md` states that large-scale Dataset v2 generation
should not resume until vLLM/Sarathi advantage regimes are either validated
against GPU behavior or explicitly scoped as simulator limitations. This
report provides that scoping for the vLLM side: the simulator's response to
concurrency-cap changes, KV-budget changes, and queueing-induced TTFT growth
(dimensions B, C, D in section 4) are all `NOT_VALIDATED` against real A100
hardware. Any Selector Dataset v2 scenario or feature that depends on the
simulator correctly ranking policies under KV pressure or admission-control
queueing should be treated as an unvalidated simulator assumption, not a
confirmed regime, until either (a) a longer-context Wulver run (section 7)
demonstrates real preemption behavior to compare against, or (b) the
simulator is explicitly extended to consume the same `max-num-seqs`/KV-budget
parameters used on the real server, so its output can actually respond to
those changes. The Sarathi-Serve side of this scoping is still pending a
separate runtime-validation pass (see "Sarathi install status" below) and
should be combined with this report before any resumption decision.
