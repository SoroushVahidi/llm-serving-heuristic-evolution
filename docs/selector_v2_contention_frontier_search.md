# Selector v2 contention-validation pilot: frontier-search audit

Follow-up to the overnight pilot
(`experiments/selector_v2_overnight_20260720T045536Z/`, branch
`selector-v2-contention-validation-pilot`), which stopped honestly at its
own gate boundary: 300/300 randomized windows tied
(`oracle_headroom=0.0`, `all_equivalent_fraction=1.0`) on
`arrival_normalized_weighted_goodput` (ANWG), so Phase 4 (dataset pilot)
and Phase 6/7 (selector training/eval) were correctly skipped.

This document roots out **why**, with two new mechanistic findings, a
900-window frontier sweep, an SLO-threshold sensitivity analysis, and a
structural cross-check against the Wulver hardware benchmark pack. It
does not train a selector, does not generate a large Dataset v2, and does
not add policies or tune SLOs to manufacture winners.

## Summary

1. **Root cause A (already documented, now independently reproduced and
   generalized):** the original "hog + tiny-runner-burst" fixture family
   (`contention_fixtures.py` variants A-D, and the overnight pilot's
   `_random_window` generator) is structurally self-limiting under this
   simulator's *current* execution model. Step-by-step tracing (below)
   proves `_advance_decode_protected` and `_advance_shared_contention`
   compute the **algebraically identical** "prefill gets
   `step_token_budget - n_currently_decoding`, decode gets 1 token per
   decoding request" allocation whenever admission/insertion order agrees
   with strict `(arrival_time, request_id)` order — which is true for
   every policy whose admission is itself FCFS-by-arrival (vllm_faithful,
   sarathi_faithful, vllm_chunked_prefill_faithful all are). No amount of
   parameter tuning inside that shape (more requests, tighter KV, more
   hogs, staggered trickles) breaks this equivalence — confirmed at 10x
   the original scale (450 `hog_runner_staggered` windows below).
2. **Root cause B (new):** a genuinely different, non-self-limiting
   divergence mechanism exists, reachable from a normal (non-injected)
   arrival trace: when the **admitting policy's insertion order disagrees
   with strict arrival order** (e.g. `weighted_shortest_processing`,
   `estimated_service_time_first`, `scorpio_style_slo_guard` reorder by
   something other than arrival time), `_advance_decode_protected`'s
   prefill loop (insertion order) and `_advance_shared_contention`'s
   prefill loop (strict arrival order) schedule the *same* simultaneously-
   prefilling requests in a genuinely different order. This is a
   **prefill-vs-prefill** effect, not decode-vs-prefill — it can fire with
   zero requests ever decoding. Demonstrated with a real, non-injected
   2-request scenario (`tests/test_contention_diagnostics.py::
   TestRealisticAdmissionOrderDivergence`): ~2x mean-latency divergence.
3. **Root cause C (dominant, quantified):** even where root cause B fires
   (736/900 frontier windows show nonzero `decode_stalled_steps` or
   `prefill_stalled_steps`), the `arrival_normalized_weighted_goodput`
   objective **still ties 900/900**, because every window (original pilot
   and this frontier sweep) used `slo_deadline=1000.0` seconds while
   observed latencies are ~0.02-0.1s — 4-5 orders of magnitude looser than
   necessary. The SLO-sensitivity sweep (below) shows divergence requires
   deadlines within roughly 1.1x of the window's own observed median
   latency; at 5x or looser, zero windows diverge on any SLO-based metric.
4. **Verdict:** quality gate for a new targeted Dataset v2 pilot does
   **not** clear (see gate table below). The frontier search materially
   sharpens *why* (SLO scale, not load/mechanism reachability) but does
   not change the recommendation.

## Root cause A, traced step-by-step

`variant_a_earlier_long_prefill_later_short_decode` (hog: 4000-token
prompt, arrives t=0, output=1 token; four 1-token-prompt "runners" arrive
1ms later, output=40 tokens each; `step_token_budget=514`,
`max_prefill_chunk_tokens=512`) run under both `decode_first=True`
(`vllm_faithful`) and `decode_first=False` (`vllm_chunked_prefill_faithful`)
produces **byte-identical per-step state for all 15+ traced steps**:
only 2 of the 4 runners ever bootstrap into decode (steps 2-3); the other
2 stay frozen at `prefill_remaining=1` until the hog itself finishes
prefill at step 8, in *both* execution models, because:

- At the moment those 2 runners bootstrap, no request is decoding yet, so
  `decode_protected`'s `prefill_budget = step_token_budget - 0` and
  `shared`'s "leftover after decode reqs" are the same number (0
  decoders) — insertion order and arrival order agree trivially.
- Once those 2 are decoding, `decode_protected`'s
  `prefill_budget = 514 - 2 = 512` exactly equals the hog's chunk, so the
  hog consumes it all, leaving the *other* 2 frozen runners exactly 0 —
  and `shared`'s FCFS order places the hog (arrival=0.0) before the 2
  already-decoding runners (arrival=0.001, ids 1-2) before the 2 frozen
  ones (arrival=0.001, ids 3-4, tie-broken *after* 1-2 purely because they
  happen to have higher request IDs) — the identical 512/1/1/0/0 split.

This is a live, empirical confirmation of `contention_fixtures.py`'s own
documented "fixed-point equilibrium" argument, not a new claim — but it
is now directly *observable* per-step via `contention_diagnostics.py`
rather than only inferable from equal outcome metrics.

## Root cause B: the admission-order-reordering mechanism (new)

Minimal reproduction (`tests/test_contention_diagnostics.py::
TestRealisticAdmissionOrderDivergence`): two requests, arriving
*simultaneously* at `t=0`, admitted by `WeightedShortestProcessingPolicy`
(sorts the waiting queue by predicted service length, not arrival time):
`id=0` (10,000-token prompt, "long") and `id=1` (500-token prompt,
"short").

| | decode_first=True (protected) | decode_first=False (shared) |
|---|---|---|
| Who gets served first this step | short (`id=1`) — SJF admission order | long (`id=0`) — strict arrival+id tie-break |
| `id=1` (short) latency | 0.002s | 0.022s |
| `id=0` (long) latency | 0.022s | 0.021s |
| mean_latency | 0.012 | 0.0215 |

Neither request ever decodes before this resolves (`predicted_output_tokens=1`
for both) — confirmed by `prefill_requests_stalled > 0` and
`decode_tokens_deferred == 0` throughout
(`test_mechanism_is_prefill_vs_prefill_not_decode_vs_prefill`). This is
**not** the decode-vs-prefill mechanism `contention_fixtures.py` targeted;
it is a pure prefill-vs-prefill ordering conflict, invisible to the
original fixture family and to the overnight pilot's `_random_window`
generator (whose hog always arrives *alone*, so there is never a same-
arrival-time admission-order decision to make between differently-sized
requests).

## Diagnostics added

`src/llmserveopt/simulator/contention_diagnostics.py` (new module) +
`GPUState.step_contention_diagnostics` / `Simulator._waiting_queue_history`
/ `Simulator.contention_diagnostics_summary()` (opt-in, additive,
diagnostic-only — no execution or objective code reads these fields):

- `decode_tokens_served` / `decode_tokens_deferred` — per-step decode
  progress vs. starvation.
- `prefill_tokens_served` / `prefill_requests_stalled` — per-step prefill
  progress vs. starvation (captures root cause B, which
  `decode_tokens_deferred` cannot).
- `budget_used` / `budget_total` / `budget_saturated`.
- `prefill_scheduled_while_decode_deferred` — the specific interaction
  shape the original fixtures targeted.
- `max_waiting_queue` / `mean_waiting_queue` (Simulator-level).

Proven invariant (`TestDecodeProtectedNeverDefers`): `decode_first=True`
produces `decode_tokens_deferred == 0` on *every* step, unconditionally —
decode-protected mode never throttles decode regardless of how contended
the scenario is; only `decode_first=False` can ever show a nonzero value.

## Frontier sweep (900 windows)

`scripts/selector_v2_contention_frontier_search.py`, two shapes, 450
windows each, same 11-policy roster as the overnight pilot
(`vllm_faithful`, `sarathi_faithful`, `vllm_chunked_prefill_faithful` +
8 cheap historical policies):

- `admission_reorder`: 2-4 simultaneously-arriving, differently-sized
  (2-20x size ratio) prefill-heavy requests, optionally plus a small
  decode-only runner burst — targets root cause B.
- `hog_runner_staggered`: scaled-up root-cause-A shape (up to 3 hogs, up
  to 40 runners, staggered multi-step trickle arrivals) — stress-tests
  whether root cause A's self-limiting equilibrium holds at 10x the
  original scale.

Results (`experiments/selector_v2_contention_frontier_search/
frontier_summary.json`):

| Classification | Windows | |
|---|---|---|
| SATURATED (budget_saturation_fraction ≥ 0.5) | 431 / 900 | mechanism actively exercised |
| CONTENTION_VISIBLE (stalls, not saturated) | 305 / 900 | mechanism actively exercised |
| UNDERLOADED | 164 / 900 | mechanism not exercised |
| PATHOLOGICAL_OVERLOAD | 0 / 900 | — |

**736/900 (82%) windows genuinely exercise the mechanism** per the new
diagnostics — a dramatically higher rate than the original 300-window
search suggested (which showed 0% divergence and gave no way to tell
"mechanism never fired" from "mechanism fired, no effect"). Yet:

| Objective | Classification counts (900 windows) |
|---|---|
| `arrival_normalized_weighted_goodput` (primary) | 900 `ALL_COMPLETE_OR_EFFECTIVELY_TIED` |
| `weighted_goodput` | 900 `ALL_COMPLETE_OR_EFFECTIVELY_TIED` |
| `slo_attainment` | 900 `ALL_COMPLETE_OR_EFFECTIVELY_TIED` |
| `p95_latency` | 716 tied, **184 `NEAR_TIE`** (never stronger) |
| `slo_success_throughput` / `request_throughput` | 701 tied, **199 `NEAR_TIE`** |

Every SLO-gated objective (the three that matter for selector training)
is **perfectly tied on all 900 windows** — mechanism activity is real and
common, but it never once translates into an SLO-relevant outcome
difference at `slo_deadline=1000.0`. Of the 184 `p95_latency` near-ties,
**179 (97%) come from `admission_reorder` windows** (root cause B) and
only 5 from `hog_runner_staggered` (root cause A) — confirming root cause
B, not A, is the operative latency-divergence source in this simulator.
`vllm_chunked_prefill_faithful` (the policy the contention fix was built
to validate) wins `p95_latency` in **0 of 900** windows; the wins that do
exist go to `vllm_faithful` (712), `weighted_shortest_processing` (113),
`scorpio_style_slo_guard` (65), `sarathi_faithful` (10) — i.e. even where
divergence exists, it is driven by admission-order effects among
historical/decode-protected policies, not by the shared-contention
mechanism itself.

## SLO-threshold sensitivity (task 5)

`scripts/selector_v2_contention_frontier_slo_sensitivity.py`: for each of
the 736 mechanism-active windows, recomputes an ANWG-like proxy
(`completed_within_synthetic_deadline / n_total`) under synthetic
deadlines = `k × window's own observed median latency`, for
`k ∈ {0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0}` — the same
grid applied uniformly to every window, never tuned per-window.

| k (× observed median latency) | Fraction of windows diverging |
|---|---|
| 0.5 | 37.9% |
| 0.9 – 1.0 | **40.1% – 40.6% (peak)** |
| 1.1 | 18.5% |
| 1.5 | 5.2% |
| 2.0 | 2.2% |
| 3.0 | 0.3% |
| 5.0, 10.0 | 0.0% |

Of the 444/736 windows that ever diverge at any tested `k`, 95% (421)
require `k ≤ 1.0` — i.e. the SLO deadline must be at or tighter than the
window's own median observed latency to catch any divergence at all. The
original pilot's `slo_deadline=1000.0` against ~0.02-0.1s observed
latencies corresponds to `k ≈ 10,000-50,000` — five orders of magnitude
past the point (`k=10`) where this sweep already finds zero divergence.
**This is the dominant, quantified explanation** for why 900/900 (and the
original 300/300) windows tied on every SLO-gated objective despite the
mechanism firing in 82% of windows.

## Robust specialization regions (task 6)

Grouping `admission_reorder` windows by `(n_hogs, size_disparity bucket)`
and examining `p95_latency` win distribution per bucket: `vllm_faithful`
wins 62-71% in **every** bucket tested (9/9), with
`weighted_shortest_processing`/`scorpio_style_slo_guard` a consistent
but non-dominant secondary share (10-30%) — the same ranking pattern
repeats across all parameter regions, not a clean per-region winner
swap. No `vllm_chunked_prefill_faithful`-favored or
`sarathi_faithful`-dominant region was found anywhere in either shape's
900 windows (`sarathi_faithful` wins only 10/900, always alongside — not
instead of — `vllm_faithful`/historical wins in the same high-disparity
buckets). **Honest finding, not a negative surprise given root causes
A-C: no ≥2-distinct-robust-region structure exists in this frontier.**

## Wulver ground-truth cross-check (task 8)

`docs/runtime_validation_benchmark_pack.md`'s already-documented, GPU-
validated finding: the current `vllm_faithful`/`sarathi_faithful` pair
gets **both** Sarathi-favoring positive targets
(`active_decode_plus_arriving_prefill`, `kv_pressure`) wrong (predicts
vLLM in both; real hardware robustly favors Sarathi, 5/5 trials each) and
gets all three vLLM-favoring negative controls
(`long_prompt_moderate_output`, `prefill_heavy_burst`,
`mixed_prompt_lengths`) right already. This frontier search's finding
that `vllm_chunked_prefill_faithful` (the policy meant to bring
decode/prefill contention semantics closer to real vLLM/Sarathi behavior)
never wins `p95_latency` in 900 independently-constructed windows is
**qualitatively consistent** with — not contradictory to — that
already-documented mismatch: this frontier sweep deliberately tried to
find *any* parameter region favoring the shared-contention model and did
not find one, matching the hardware pack's conclusion that the current
execution-model fix has not yet closed the Sarathi-advantage gap. No
scenario parameters here were fit to the five hardware labels (see
`contention_fixtures.py`'s own docstring and this sweep's generators: no
hardware target value is read or referenced anywhere in either).

## Quality gate for a new targeted Dataset v2 pilot

| Gate | Threshold | Result | Pass? |
|---|---|---|---|
| Informative retained windows | ≥ 50 | 736 (mechanism-active) | ✅ |
| Faithful baselines w/ meaningful specialization | ≥2 faithful, or 1 faithful + multiple historical, **on a meaningful (non-tie) objective** | 0 on ANWG/weighted_goodput/slo_attainment; on p95_latency only ever `NEAR_TIE` | ❌ |
| All-equivalent fraction (retained windows, ANWG) | < 40% | 100% (736/736 tied) | ❌ |
| Oracle headroom (ANWG) | ≥ 0.01 | 0.0 | ❌ |
| Mechanism diagnostics confirm scheduler distinction exercised | required | 736/900 (82%) — confirmed | ✅ |

**3 of 5 gates fail, including the two load-bearing ones (all-equivalent
fraction, oracle headroom).** Per the task's own instruction not to force
a match, and consistent with the overnight pilot's own honest stop:

## Verdict

`READY_FOR_NEW_TARGETED_DATASET_V2_PILOT = no`

The frontier search materially advances the original finding — it proves
the mechanism *is* reachable via realistic traces (root cause B, absent
from the original fixture/search design) far more often than the
original 300-window search suggested (82% vs 0%), and it precisely
quantifies *why* that reachability still doesn't produce a trainable
signal: the SLO-deadline scale used across every window generated so far
is 4-5 orders of magnitude looser than what the sensitivity sweep shows
is required. Any future targeted pilot should generate windows with
`slo_deadline` set to within ~1.0-1.5x of each window's *own* expected
latency scale (not a fixed global constant like `1000.0`) and should bias
window construction toward `admission_reorder`-style same-arrival-time,
differently-sized request clusters (root cause B) rather than the
original hog-plus-tiny-runner shape (root cause A, confirmed self-limiting
at 10x scale here). Even then, this search found no clean
`vllm_chunked_prefill_faithful`-favored region and no `sarathi_faithful`-
dominant region — consistent with the Wulver pack's already-documented
finding that the current execution model has not closed the real-hardware
Sarathi advantage — so expectations for what a corrected-SLO pilot would
find should stay modest.
