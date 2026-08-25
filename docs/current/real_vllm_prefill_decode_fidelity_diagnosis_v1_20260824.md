# real_vllm_prefill_decode_fidelity_diagnosis_v1

Date: 2026-08-24

Status: `SIMULATOR_VLLM_SEMANTICS_MISMATCH`

Scope: diagnostic follow-up to
`experiments/real_vllm_mechanism_validation_v1/prefill_decode_local_v1/`.
This did not run a new scientific treatment comparison, Wulver job, GPU-scale
benchmark, TEST/FINAL evaluation, selector experiment, or policy implementation.

## Question

Did the local real-vLLM run instantiate the same prefill/decode contention
mechanism represented by simulator Family B, or did the simulator-to-vLLM
mapping lack fidelity?

## Family-B Simulator Semantics

Family-B v2 compares exactly two policies:

- `full_prefill`: greedy arrival-order admission with
  `max_prefill_chunk_tokens=65536`, `decode_first=False`.
- `chunked_prefill_small`: the same greedy arrival-order admission with
  `max_prefill_chunk_tokens=64`, `decode_first=False`.

Both run under `enable_decode_prefill_contention=True`. The relevant simulator
execution path is `_advance_shared_contention`: decode and prefill consume one
combined per-step token budget in FCFS-by-arrival order. In Family-B v2 that
budget is 512 tokens. The v2 mechanism is not the dropped `decode_first`
variant. The v2 mechanism is: full prefill lets an earlier long-prompt hog
consume the step budget, while small chunks leave crumbs that can improve
late short-request TTFT at the expense of hog TTFT/completion.

Prior Family-B v2 evidence showed bidirectional practical wins: full wins all
hog-tight cells; small chunk wins nearly all late-tight cells.

## vLLM 0.27.1 Semantics

Installed source shows that vLLM V1 scheduler explicitly does not operate with
separate decoding and prefill phases. It schedules running requests first, then
waiting requests, against a token budget derived from
`max_num_batched_tokens` / `max_num_scheduled_tokens`.

Chunked prefill is enabled by allowing a request to schedule fewer tokens than
its remaining prompt length. If chunked prefill is disabled and a waiting
request needs more tokens than the remaining budget, scheduling stops rather
than chunking it.

The completed real treatments were:

- `FULL`: `--no-enable-chunked-prefill --max-num-batched-tokens 4096`
- `CHUNKED`: `--enable-chunked-prefill --max-num-batched-tokens 512`

This is not an isolated chunk-size intervention. It bundles chunking,
scheduled-token budget, waiting-admission feasibility, and vLLM's running-first
scheduler behavior.

## Simulator to vLLM Fidelity

The strongest mismatches are:

| Mechanism | Simulator Family B | vLLM local run | Consequence |
| --- | --- | --- | --- |
| Step budget | 512 for both full and small | 4096 for FULL, 512 for CHUNKED | budget is confounded with treatment |
| Small chunk | 64-token cap | actual chunks often 510-512 tokens | chunked mode is much coarser |
| Full hog behavior | hog can consume whole 512-token step | 3200-token hog fits inside one 4096-token iteration | full does not create the same convoy |
| Queue ordering | shared FCFS over decode+prefill | running requests before waiting admissions | late waiting requests may not receive crumbs |
| Decode/preempt/KV | simulator step abstraction | native vLLM batching and KV allocator | not step-commensurate |

Classification: `BUNDLED_SCHEDULING_BUDGET_INTERVENTION`, not an isolated
Family-B analogue.

## Did Real Contention Occur?

Yes. The completed run already showed queueing:

- max waiting: 7
- max running: 4
- max KV usage: 2.84%
- preemptions: 0

A tiny diagnostic scheduler trace then confirmed actual mixed engine
iterations:

| Treatment | Steps with prefill+decode | Partial prefill chunks | Max scheduled prefill tokens/step |
| --- | ---: | ---: | ---: |
| FULL | 6 | 0 | 3456 |
| CHUNKED | 16 | 21 | 512 |

Example CHUNKED step: one decode token plus 511 prefill tokens in the same
512-token scheduler iteration. Example FULL step: two decode tokens plus 3456
prefill tokens in one 4096-token iteration.

So the issue is not that no overlap happened. The issue is that the overlap
was not the simulator Family-B overlap.

## Prefill Cost vs Decode Cost

Single-request TTFT with `max_tokens=1`:

| Prompt tokens | TTFT (s) |
| ---: | ---: |
| 96 | 0.0094 |
| 512 | 0.0157 |
| 1024 | 0.0261 |
| 2048 | 0.0490 |
| 3200 | 0.0767 |

Representative decode ITL from a four-request batch:
0.00591 s/token.

Ratio:
`prefill_time(3200) / representative_decode_iteration_time = 12.99`.

The 0.5B model is not cost-free: long prefill is measurable. But the absolute
3200-token prefill is only about 77 ms, and FULL can process it in one large
4096-token iteration.

## Diagnosis

Classification: `SIMULATOR_VLLM_SEMANTICS_MISMATCH`

The local 0.5B setup induced queueing, real chunking, and prefill/decode
overlap. That is enough to diagnose native vLLM behavior. It is not enough
for high-KV-pressure claims, but high KV pressure is not the root cause of
the Family-B transfer failure.

The root cause is semantic and contractual:

1. vLLM 0.27.1 requires `max_num_batched_tokens >= max_model_len` when chunked
   prefill is disabled, forcing FULL to use a much larger budget.
2. vLLM schedules running requests before waiting requests, unlike the
   simulator's shared FCFS pass over active prefill/decode.
3. CHUNKED chunks are budget-sized, around 512 tokens here, not the 64-token
   simulator crumbs.
4. FULL does not recreate the simulator's 512-token hog monopoly because a
   3200-token prompt fits in one 4096-token iteration.

## What the NO-GO Proves

The completed `PREFILL_REAL_VALIDATION_NO_GO` is evidence against this concrete
native vLLM 0.27.1 / Qwen0.5B / RTX5060Ti configuration reproducing the
simulator Family-B winner reversal.

It is not strong evidence that Family-B transfer fails in all real systems,
because the real treatment did not faithfully instantiate the simulator
mechanism.

## Paper Implication

Use this result conservatively. It is valuable as a validation/limitation
subsection or appendix result: a simulator mechanism that looked clean did not
transfer under a direct native-vLLM mapping, and the diagnosis shows why.

Do not claim broad real-system falsification of chunked prefill. The safer
claim is that simulator abstractions of prefill/decode contention can be
invalidated by modern serving-engine scheduler semantics and token-budget
constraints.

## Cheapest Follow-Up

Do not scale to Wulver/7B yet. Scaling hardware would not fix the semantic
confound first.

Single cheapest follow-up to preregister, not run here:
`native_vllm_chunk_budget_semantics_probe_v1`.

Design: keep chunked prefill enabled for all treatments and compare
`max_num_batched_tokens=512` versus `4096` on one or two regimes with scheduler
tracing. This asks the native vLLM question directly: how does scheduled-token
budget/chunking affect late TTFT and hog completion under vLLM's actual
running-first scheduler?

