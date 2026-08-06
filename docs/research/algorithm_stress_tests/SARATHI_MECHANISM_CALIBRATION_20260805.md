# Sarathi Stress-Test Mechanism Calibration — 2026-08-05

Records why the 5 real-hardware-mirrored Sarathi catalog entries
(`configs/stress_tests/algorithm_stress_test_catalog.yaml`, section 12)
do NOT use a direct `sarathi_faithful` vs `vllm_chunked_prefill_faithful`
latency gate, despite that being the most natural first design (and the
one these entries originally shipped with, before this calibration pass).
Follows this project's own established practice
(`docs/research/algorithm_stress_tests/STRESS_TEST_VALIDATION_20260805.md`)
of diagnosing FAILs mechanistically rather than tuning parameters until a
gate passes.

## The finding

At every parameter combination tested (`step_token_budget` swept
16/32/64/128/512/4096, arrival-offset scale swept 0.05s-3.0s, plus a
deliberately adversarial inverse-arrival-order fixture), `sarathi_faithful`
(`decode_first=True`) and `vllm_chunked_prefill_faithful`
(`decode_first=False`, shared contention) produce **byte-identical**
`mean_latency` for every one of the 5 real-hardware-mirrored fixtures.
This is not a tuning failure to be fixed by trying more parameters — it
is a structural property of the two execution paths, proven below.

## Root cause (structural, not a calibration gap)

`GPUState._advance_decode_protected` (sarathi_faithful's path) reserves
`len(decoding)` tokens off the top of `step_token_budget` unconditionally,
then gives prefilling requests whatever remains, iterated in **admission
order**. `GPUState._advance_shared_contention`
(vllm_chunked_prefill_faithful's path) sorts ALL active requests
(decoding and prefilling together) by `(arrival_time, request_id)` and
consumes the shared budget in that single order.

Both `sarathi_faithful` and `vllm_chunked_prefill_faithful` admit new
requests via an **FCFS-strict** algorithm (admit from the front of
`waiting` in arrival order; stop admitting entirely on the first
non-allocatable request — see `docs/sarathi_faithful_scheduler_reference.md`
§"Per-iteration scheduling order"). Under FCFS-strict admission, any
request that is *currently decoding* was, by construction, admitted (and
therefore arrived) no later than any request that is *still trying to get
its first prefill chunk admitted*. That means: in
`_advance_shared_contention`'s arrival-time sort, every currently-decoding
request is *always* ordered at or before every currently-competing
prefilling request — identically to `_advance_decode_protected`'s
unconditional decode-first reservation. **The two execution paths are
therefore provably equivalent for any workload whose requests are admitted
FCFS-strict by both policies being compared** — which is true of every
fixture in this project's generator framework, since no generator
constructs out-of-order admission.

Verified directly via `GPUState.step_contention_diagnostics` (per-step
`decode_tokens_served`/`decode_tokens_deferred`/`prefill_tokens_served`):
for the `active_decode_plus_arriving_prefill`-mirrored fixture, both
`decode_first=True` and `decode_first=False` runs show
`decode_tokens_deferred=0` at every step, and identical
`prefill_tokens_served` at every step -- the decoding requests are never
actually contested in either mode, because they always sort first by
construction.

### Why the inverse-order fixture (long prefill arrives first) also failed to diverge

Constructing a request that arrives first with a very long prompt, then a
short-prompt request arriving shortly after (intended to quickly become a
decoding request while the long prefill is still mid-flight) also
produced identical output. Diagnosed via the same per-step trace: the
long-prefill request, being first in BOTH admission order and arrival-time
order, consumes the entire step budget every step in both execution paths
until its own prefill finishes -- the later-arriving short request cannot
even complete its own first admission/prefill chunk (let alone start
decoding) while the earlier request holds full budget priority in both
models identically. The two paths only diverge when a currently-decoding
request is ordered *after* a currently-prefilling request in the shared
sort while being unconditionally protected in the decode-first path --
which requires an admission history where the decoding request was
admitted later than the still-in-flight prefill, a combination that FCFS-
strict admission cannot produce for either policy compared here.

## Implication

This project's `enable_decode_prefill_contention` mechanism
(`docs/decode_prefill_contention_execution_model.md`) IS structurally
representable in the simulator (confirmed by the earlier
`vllm_chunked_prefill_faithful_root_cause_analysis.md` finding), but it
**cannot currently be exercised to distinguish `sarathi_faithful` from
`vllm_chunked_prefill_faithful`** by any workload built from this
project's standard FCFS-admission generator framework — this is a
genuine simulator-mechanism limitation, disclosed here rather than
hidden behind a misleadingly "FAILed" latency gate. Reproducing the real
hardware's decode-protection distinction would require either (a) a
generator or harness capable of seeding out-of-order admission directly
into `GPUState._active` (bypassing each policy's own admission logic --
not attempted here, out of scope for a stress-test catalog addition), or
(b) a different pair of policies whose admission algorithms are not both
FCFS-strict.

## What the 5 catalog entries validate instead

Diagnostic runs (see `results/stress_test_catalog/sarathi_smoke/report.json`)
show `sarathi_faithful` (and `vllm_chunked_prefill_faithful`, identically)
DOES differ substantially from `vllm_faithful` (the non-chunked v0.1.0
faithful reimplementation) on these same fixtures: `vllm_faithful` fails
to complete a meaningful fraction of long-prompt requests in 3 of 5
scenarios (`completion_fraction` 0.00-0.50, vs. 1.00 for both chunked
policies) because it has no chunking mechanism at all and cannot admit
prompts exceeding its per-step batch-token limit. This is a real,
measured, mechanism-relevant finding — it validates chunked prefill's
basic value proposition (successful admission/completion of long-prompt
workloads) even though it cannot validate the finer decode-protection
distinction real hardware isolated. The 5 catalog entries' `acceptance_gates`
were revised to test this coarser, genuinely-evaluable claim
(`sarathi_faithful.completion_fraction >= vllm_faithful.completion_fraction`),
with the original real-hardware-direction claim preserved as disclosed,
`NOT_CURRENTLY_TESTABLE_IN_SIMULATOR` context rather than deleted or
silently swapped.

## What remains true, unaffected by this finding

The real-hardware evidence itself
(`docs/wulver_sarathi_vllm_repeated_validation.md`) is untouched by this
finding — it is real A100 GPU execution of the actual official Sarathi-Serve
and vLLM 0.24.0 systems, not a simulator claim. This calibration pass
only concerns whether THIS SIMULATOR's `sarathi_faithful` vs
`vllm_chunked_prefill_faithful` comparison can reproduce that real effect
-- it currently cannot, for the structural reason above, and that
limitation is now disclosed rather than silently masked by a FAILing gate
that would misleadingly read as a claim about Sarathi's real-world
behavior.
