# Root-cause analysis: `active_decode_plus_arriving_prefill` and `kv_pressure` positive-target mismatches

> **Update:** Finding 2/3's dead `GPUState._step_phase15` `decode_first`
> branch, described below as "deliberately not fixed here," has since been
> fixed (opt-in, backward-compatible) -- see
> `docs/decode_prefill_contention_execution_model.md`. That document's own
> "Revalidation" section reruns the exact six-scenario benchmark pack
> referenced here and finds the mismatches below unchanged, for a
> different, now-verified reason (a workload-construction fact about these
> fixtures, not a simulator-execution limitation). The investigation below
> is left as-written -- a historical record of the pre-fix state -- rather
> than edited in place.

This document investigates the two positive-target mismatches recorded in
`experiments/runtime_validation_benchmark_pack/simulator_baseline_results/
active_decode_plus_arriving_prefill.json` and `kv_pressure.json`: real
A100 hardware shows a robust (5/5 trials) Sarathi-Serve E2E advantage on
both scenarios; the `vllm_faithful`/`sarathi_faithful` simulator pair picks
vLLM as the E2E winner on both, the opposite direction. Per instruction,
this is a mechanism trace, not a tuning exercise — no constant was adjusted
to move either scenario's outcome.

## Method

Both scenarios were reconstructed from
`experiments/runtime_validation_benchmark_pack/scenarios/*.json` (exact
`prompt_tokens`/`arrival_time`/`output_tokens` per request) and run through
`llmserveopt.evaluation.run_policy.run_policy` directly (the same function
`scripts/run_gpu_external_validity_audit.py`'s `run_simulator_scenario`
calls), under three configurations:

1. **As currently run by the benchmark-pack-generating harness**
   (`run_gpu_external_validity_audit.py` lines 646-658): `vllm_faithful`
   with `ServiceModel()` (all defaults — `enable_prefill_modeling=False`),
   `sarathi_faithful` with `ServiceModel(enable_prefill_modeling=True,
   decode_first=True, step_token_budget=512, max_prefill_chunk_tokens=512)`.
2. **Controlled**: both policies run under the *same*
   `ServiceModel(enable_prefill_modeling=True, decode_first=True,
   step_token_budget=512, max_prefill_chunk_tokens=512)`.
3. **Controlled, `decode_first=False`**: both policies under the same
   service model as (2) but with `decode_first=False`, to test whether that
   flag has any effect at all.

## Finding 1 (dominant root cause): the current comparison is not an
apples-to-apples scheduler comparison — it is an apples-to-oranges
**timing-model** comparison

`vllm_faithful` is evaluated with the *default* `ServiceModel()`, which has
`enable_prefill_modeling=False`. Per `GPUState.step()`
(`src/llmserveopt/simulator/gpu.py` lines 224-226): `if service_model is
None or not service_model.enable_prefill_modeling: return
self._step_phase1(...)`. Phase 1's `_step_phase1` gives every admitted
request **zero-cost, instantaneous prefill** — `ServiceModel.
compute_prefill_tokens` returns `0` whenever `enable_prefill_modeling=False`
(`service_model.py` line 84-86), so an admitted request's very first
decode token is produced the same step it is admitted, with no prefill
delay modeled at all.

`sarathi_faithful`, in the same harness call, is evaluated with
`enable_prefill_modeling=True` and a genuine 512-token/step prefill cost.

This means the two "faithful" baselines are not being compared on their
**scheduling algorithms** at all in this harness — one of them (`vllm_
faithful`) is given a strictly more favorable timing model (zero prefill
cost) than the other, for every scenario with any nontrivial prompt length,
independent of which scheduler would actually admit/order/preempt requests
differently. This asymmetry is not documented anywhere in either policy's
own module docstring or reference doc; it is purely a property of how
`run_gpu_external_validity_audit.py`'s `run_simulator_scenario` happens to
construct each policy's `ServiceModel`.

**Empirical confirmation** (reproducing the harness's own numbers, then
controlling for this asymmetry alone):

| Scenario | Config | `vllm_faithful` TTFT / E2E | `sarathi_faithful` TTFT / E2E | Winner |
|---|---|---:|---:|---|
| `active_decode_plus_arriving_prefill` | as-harness (asymmetric) | 0.0010 / 0.1600 | 0.0053 / 0.1642 | vllm |
| `active_decode_plus_arriving_prefill` | controlled (both Phase-1.5, `decode_first=True`) | 0.0053 / 0.1642 | 0.0053 / 0.1642 | **tie (byte-identical)** |
| `kv_pressure` | as-harness (asymmetric) | 0.0065 / 0.7735 | 0.0286 / 0.7956 | vllm |
| `kv_pressure` | controlled (both Phase-1.5, `decode_first=True`) | 0.0286 / 0.7956 | 0.0286 / 0.7956 | **tie (byte-identical)** |

Once both policies are given the *same* execution-timing model, their
outputs are **byte-identical** on both positive-target scenarios — not
merely "closer," but exactly equal to 4+ decimal places. This confirms two
things at once: (a) the timing-model asymmetry, not a genuine scheduling-
algorithm difference, is what currently produces the vLLM-favoring mismatch
in the benchmark pack's own recorded numbers; and (b) at this request scale
(4-12 requests, prompt lengths well under both policies'
`max_num_batched_tokens`/`chunk_size` admission budgets), `vllm_faithful`'s
all-or-nothing admission and `sarathi_faithful`'s chunked admission never
actually diverge in which requests get admitted when — so there is currently
**no scheduling-algorithm signal being tested here at all**, only a
timing-model artifact.

## Finding 2 (latent, confirmed but not currently load-bearing): `ServiceModel.decode_first` is dead code in `GPUState._step_phase15`

`src/llmserveopt/simulator/gpu.py` lines 270-278:

```python
budget = service_model.step_token_budget
if service_model.decode_first:
    # Guarantee full decode budget before any prefill
    budget -= len(decoding)   # each decode request uses 1 token
    prefill_budget = max(0, budget)
else:
    budget -= len(decoding)
    prefill_budget = max(0, budget)
```

Both branches compute the identical expression. `decode_first=False` was
empirically confirmed (configuration 3 above) to produce **exactly the same
output** as `decode_first=True` for both scenarios — the toggle currently
has zero observable effect anywhere in the simulator. Practically, this
means `GPUState._step_phase15` gives **every** Phase-1.5 policy Sarathi's
own headline "decode is never stalled by prefill" property unconditionally,
regardless of what that policy's own scheduler actually decided about
ordering — it is baked into shared, policy-agnostic simulator
infrastructure rather than being something a scheduling algorithm can win
or lose by its own design.

This is a real, verified simulator-execution-layer limitation, but it is
**not** the dominant cause of the current benchmark-pack mismatch (Finding
1 is, since `vllm_faithful` currently doesn't even run in Phase-1.5 mode in
this harness — the `decode_first` toggle is simply never reached for it
today). It becomes directly relevant, however, once `vllm_chunked_prefill_
faithful` is evaluated fairly (i.e., with `enable_prefill_modeling=True`,
matching `sarathi_faithful`'s own evaluation config) — see Finding 3.

**Deliberately not fixed here.** `GPUState._step_phase15` is shared,
already-relied-upon simulator infrastructure (`sarathi_faithful`'s own
historical numbers depend on its current behavior). Changing it is a
simulator-semantics change outside this task's scope ("historical policies
must continue to behave exactly as before" / "do not silently change
default simulator semantics") and is not needed to implement `vllm_
chunked_prefill_faithful` faithfully at the *admission-decision* level —
only to eventually make the *execution* layer capable of showing a genuine
decode stall. Flagged as a scoped follow-up, not resolved here.

## Finding 3: even a source-faithful `vllm_chunked_prefill_faithful` cannot
currently demonstrate the real advantage regime, because of Finding 2

Per `docs/vllm_chunked_prefill_faithful_scheduler_reference.md`'s algorithm
section, the pinned v0.4.2 source's real vulnerability — a still-prefilling
sequence consuming shared per-step budget ahead of a decode-phase sequence
in the same FCFS-by-arrival `_schedule_running` pass, with no explicit
decode-priority phase the way Sarathi has — **is faithfully reproduced at
the admission-accounting level** by this baseline's Phase 1 (see that
policy's own module docstring). But whether that admission-level fact
translates into an *observably worse* E2E number depends on the execution
layer actually letting a decode-phase request receive `0` tokens in some
step because a continuing-prefill request (scheduled earlier by this
baseline's own accounting) used up the shared budget first. Given Finding
2, `GPUState._step_phase15` will **still** give that decode-phase request
its 1 token this step regardless of what `vllm_chunked_prefill_faithful`'s
own admission bookkeeping decided — so the vulnerability this baseline
faithfully models at the scheduling-decision level remains invisible at the
execution/timing level, for the same structural reason `vllm_faithful` and
`sarathi_faithful` already tie once evaluated under the same `ServiceModel`
(Finding 1's controlled-comparison result).

**Net conclusion**: closing Finding 1 (evaluating any future vLLM-family
policy under the same `ServiceModel` as `sarathi_faithful`, not the
zero-prefill-cost default) is necessary and is a prerequisite any future
benchmark-pack run of `vllm_chunked_prefill_faithful` must satisfy to be a
meaningful comparison at all. It is very likely **not sufficient** on its
own to reproduce the real 2.7x / smaller `kv_pressure` Sarathi advantage,
because of Finding 2 — the specific mechanism (a decode-phase request
actually stalling because shared budget ran out) cannot be observed by
*any* Phase-1.5 policy today, faithful admission accounting or not. This
mismatch is not resolved by this change; it is now root-caused to a
specific, named piece of shared simulator code
(`GPUState._step_phase15`'s dead `decode_first` branch), which is a
materially stronger, more actionable finding than "the simulator doesn't
model this dynamic with enough fidelity" (the pre-existing, pre-this-session
description in `docs/wulver_vllm_kv_pressure_results.md` session 3 and the
benchmark pack's own `known_mismatch_reason` fields).

## `kv_pressure` specifically

The benchmark pack's own `known_mismatch_reason` for `kv_pressure` states
it "has not been root-caused at the scheduler-algorithm level... unlike the
`active_decode_plus_arriving_prefill` mismatch" (which itself, per that
same file, was only an *observed, quantified discrepancy*, not a diagnosed
one, prior to this session — see `docs/wulver_vllm_kv_pressure_results.md`'s
own "Unsafe claims" section). The empirical results above show `kv_pressure`
exhibits **exactly the same Finding-1 mechanism** as `active_decode_plus_
arriving_prefill` (identical before/after pattern: asymmetric-config
mismatch, byte-identical controlled-config tie). No scenario-specific
mechanism beyond Finding 1/2 was found or needed to explain it. Both
positive-target mismatches share one root cause, not two independent ones.

## Addendum: the three negative controls "pass" today for the same reason
the positive targets fail -- not because the simulator captures a genuine
vLLM-favoring scheduling dynamic

Repeating the same before/after experiment for all five canonical
benchmark-pack scenarios (not just the two positive targets) surfaces a
stronger, more concerning version of Finding 1. Under the harness's current
asymmetric config, `vllm_faithful` wins E2E on **every single scenario**,
negative controls included:

| Scenario | positive_target | `vllm_faithful` E2E | `sarathi_faithful` E2E | `vllm_chunked_prefill_faithful` E2E (if scored the same asymmetric way) |
|---|---|---:|---:|---:|
| `long_prompt_moderate_output` | False | 0.2575 (win) | 0.2670 | 0.2620 |
| `active_decode_plus_arriving_prefill` | **True** | 0.1600 (win) | 0.1642 | 0.1608 |
| `prefill_heavy_burst` | False | 0.0345 (win) | 0.0470 | 0.0420 |
| `mixed_prompt_lengths` | False | 0.0652 (win) | 0.0698 | 0.0670 |
| `kv_pressure` | **True** | 0.7735 (win) | 0.7956 | 0.7904 |

`vllm_faithful` "wins" all five, both the two it should lose
(positive targets) and the three it should win (negative controls). Its
negative-control "correctness" today is not evidence the simulator captures
a genuine scheduling-algorithm reason vLLM wins those three scenarios — it
is the same Finding-1 zero-prefill-cost timing artifact, which happens to
point the same direction the negative controls want by coincidence of which
system the harness gives the unfair advantage to.

Under the FAIR, controlled config (both/all three policies given the same
`ServiceModel(enable_prefill_modeling=True, ...)`), the picture is
different again, and equally important: **all three policies tie exactly
on all five scenarios** at this request scale (4-16 requests per scenario,
well inside every policy's admission budget):

| Scenario | `vllm_faithful` | `sarathi_faithful` | `vllm_chunked_prefill_faithful` |
|---|---:|---:|---:|
| `long_prompt_moderate_output` | 0.2670 | 0.2670 | 0.2670 |
| `active_decode_plus_arriving_prefill` | 0.1642 | 0.1642 | 0.1642 |
| `prefill_heavy_burst` | 0.0470 | 0.0470 | 0.0470 |
| `mixed_prompt_lengths` | 0.0698 | 0.0698 | 0.0698 |
| `kv_pressure` | 0.7956 | 0.7956 | 0.7956 |

**Honest conclusion**: none of the five canonical, request-level
benchmark-pack scenarios currently produce ANY winner-identity
differentiation between these three policies once evaluated fairly -- not
just the two positive targets. The apparent "3/3 negative controls
correct" in the pre-existing benchmark pack numbers is not a signal this
baseline's introduction should be graded against; it is the same
mechanism, in the direction that happened not to matter. See
`tests/test_vllm_chunked_prefill_faithful_benchmark_pack.py` for the
harness that checks this directly and classifies each scenario
accordingly (`TIE_NEAR_TIE` for all five under the fair config).

## What this means for a future evaluation

- **Do not** re-evaluate `vllm_chunked_prefill_faithful` with the default
  `ServiceModel()` (zero-cost prefill) the way the current harness does for
  `vllm_faithful` — that would repeat Finding 1's asymmetry in the new
  baseline's favor and produce a misleading result in the *opposite*
  direction from today's.
- **Do** evaluate it under the same `ServiceModel(enable_prefill_modeling=
  True, ...)` used for `sarathi_faithful`, for a like-for-like comparison of
  scheduling-decision quality.
- Even so, per Finding 2/3, do not expect this alone to flip the E2E winner
  on `active_decode_plus_arriving_prefill`/`kv_pressure` to match real
  hardware — the specific mechanism both scenarios are designed to probe
  (a decode stream actually stalling because of a competing prefill) is
  structurally unobservable in the current simulator execution layer for
  *any* policy. Reproducing it for real would require revisiting
  `GPUState._step_phase15`'s `decode_first` branch — out of scope here, and
  correctly so, since it is shared infrastructure `sarathi_faithful`'s
  existing results depend on.
