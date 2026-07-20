# Decode/prefill execution-contention model

This document records the fix to `GPUState._step_phase15`'s dead
`decode_first` branch, root-caused in
`docs/vllm_chunked_prefill_faithful_root_cause_analysis.md` Finding 2/3.
It exists so the design rationale for `ServiceModel.
enable_decode_prefill_contention` (and the corrected meaning of
`decode_first`) is recorded alongside the other pinned-reference /
execution-model documents in this directory.

## The bug

`src/llmserveopt/simulator/gpu.py`'s `_step_phase15` (pre-fix):

```python
budget = service_model.step_token_budget
if service_model.decode_first:
    budget -= len(decoding)
    prefill_budget = max(0, budget)
else:
    budget -= len(decoding)
    prefill_budget = max(0, budget)
```

Both branches compute the identical expression. `decode_first` was dead
code: every Phase-1.5 execution unconditionally reserved the full decode
budget before any prefill got a look at the remainder, regardless of the
flag. Empirically confirmed (root-cause doc Finding 2): `decode_first=True`
and `decode_first=False` produced byte-identical simulator output on every
scenario tested.

## Why this couldn't just be "fixed in place"

`decode_first=False` is `ServiceModel`'s own default, and is set
explicitly (redundantly, given the default) in **30+** `configs/*.yaml`
files backing the Phase 2B/2C selector-dataset generation pipeline, plus
several test files (`test_prefill_model.py`, `test_phase2c1_real_trace_
runner.py`). Every one of these runs has, for its entire history, silently
received the (buggy but internally consistent) decode-protected execution
model. Naively making `decode_first=False` mean genuine shared-budget
contention would silently change the numeric output of every one of those
runs -- exactly the "historical configurations must retain bit-identical
behavior" violation this task was scoped to avoid.

## The fix: additive opt-in execution mode

New `ServiceModel` field `enable_decode_prefill_contention: bool = False`
(also mirrored on `CalibratedServiceModel` for interface compatibility).

* **Default (`False`)** -- bit-identical to the pre-fix code. `decode_first`
  remains observably inert. Every existing config/test that does not
  explicitly opt in is completely unaffected. Internally this now runs
  through `GPUState._advance_decode_protected`, a direct, unmodified
  extraction of the original method body.
* **Opt-in (`True`)** -- `decode_first` becomes genuinely load-bearing:
  * `decode_first=True` → `_advance_decode_protected` (the same
    decode-protected formula as the historical default -- Sarathi-Serve's
    Phase 1a stall-free guarantee: decode always gets its budget first,
    unconditionally; prefill gets the remainder). Numerically identical to
    the historical default for the same requests/timing.
  * `decode_first=False` → `_advance_shared_contention` (new): decode and
    prefill requests compete for **one** combined per-step token budget,
    consumed in a single FCFS-by-arrival-time pass -- exactly vLLM
    v0.4.2's `_schedule_running` in chunked-prefill mode (see
    `docs/vllm_chunked_prefill_faithful_scheduler_reference.md`'s
    "single most important structural fact" callout: no decode-priority
    phase at all). A request later in arrival order can receive **zero**
    progress this step if an earlier-arrival request (decode or prefill)
    exhausts the budget first -- matching the pinned reference's own
    `_get_num_new_tokens(...) == 0: break` (stop scheduling the rest of
    the running queue this step, not skip-and-continue).

This is Strategy B from the task instructions ("a new opt-in mixed
prefill/decode execution mode used only by faithful modern baselines"),
not a change to `decode_first`'s default-mode meaning.

## Ordering key for the shared-contention path

`_advance_shared_contention` sorts the combined decode+prefill request set
by `(request.arrival_time, request_id)` -- the same composite key every
scheduling policy in this repo already uses
(`policies/tie_breaking.py:arrival_then_id`), re-derived locally in
`gpu.py` rather than imported from `policies/` to avoid a
simulator-depends-on-policies layering inversion (policies already depend
on simulator types, not the reverse). This is not an invented ordering: it
is the literal FCFS-by-arrival ordering both pinned references
(`sarathi_faithful`'s Phase 1a/1b, `vllm_chunked_prefill_faithful`'s
`_schedule_running`) already document their own scheduling loops as using.

## Verified divergence (micro-benchmark)

Two requests on one GPU, `step_token_budget=512`,
`max_prefill_chunk_tokens=512`: request 0 arrived at t=0 with a large
prompt (2000 tokens, 1500 remaining, still prefilling); request 1 arrived
at t=5, already decoding.

| Mode | `decode_first` | req0 `prefill_remaining` after step | req1 `tokens_decoded` after step |
|---|---|---:|---:|
| legacy (`contention=False`) | False | 989 | 2 (advanced) |
| legacy (`contention=False`) | True | 989 | 2 (advanced) |
| new (`contention=True`) | True | 989 | 2 (advanced) |
| new (`contention=True`) | **False** | 988 | **1 (stalled -- zero progress)** |

Only the new contention mode with `decode_first=False` produces the
decode-stall failure mode Sarathi-Serve's design exists to prevent, and
only there because request 0 (the still-prefilling request) arrived
*before* request 1 in this constructed example -- consuming the entire
step budget ahead of it in FCFS order. This is not automatic: if a
scenario's decode-phase request happens to have arrived *before* the
competing prefill request, arrival-order FCFS gives it priority "for
free" and no divergence from decode-protected execution is observed for
that scenario. See `docs/decode_prefill_contention_execution_model.md`'s
sibling revalidation notes in the branch's final report for which of the
six runtime-validation benchmark-pack scenarios this applies to.

## Which policies opt in

* `sarathi_faithful`: `enable_decode_prefill_contention=True`,
  `decode_first=True` -- exercises the genuine decode-protected path,
  now for a real (not accidental) reason.
* `vllm_chunked_prefill_faithful`: `enable_decode_prefill_contention=True`,
  `decode_first=False` -- exercises the genuine shared-FCFS-contention
  path, matching its own pinned v0.4.2 `_schedule_running` semantics
  (previously structurally unobservable per root-cause doc Finding 3).
* `vllm_faithful` (historical, non-chunked): left on the legacy default
  (`enable_decode_prefill_contention=False`) wherever it is evaluated
  under `enable_prefill_modeling=False` (its own historical convention,
  Phase 1, unaffected either way). Where it is evaluated under
  Phase-1.5 for a fair side-by-side (the benchmark-pack acceptance test),
  it is given `decode_first=True` alongside the same opt-in, since it has
  no chunked-admission model of its own to exercise the contention path
  meaningfully.
* Every other Phase-1.5 caller (30+ `configs/*.yaml`, `distserve_
  faithful`, `llumnix_faithful`, `tetriinfer_paper_reimplementation`,
  disaggregation tests): unchanged, on the legacy default.
