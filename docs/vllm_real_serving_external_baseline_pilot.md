# vLLM Real-Serving External-Baseline Pilot

**Status: real results.** This is the project's first comparison of its own
scheduling/admission policies against each other on a **real, running vLLM
server** — not a simulation, not a mock, not a dry-run. A tiny (24
requests/policy) smoke comparison, not final paper-scale evidence.

## What this tests, and what it does not

This tests **external admission control layered on top of vLLM's own
OpenAI-compatible HTTP API**: a client-side controller holds requests in a
queue, admits up to `concurrency_level` of them at a time, and uses one of
this project's existing scheduling policies (`src/llmserveopt/policies/`)
to choose *which* waiting request gets the next open slot, using only
information available before a response is generated (arrival time, prompt
length, declared/predicted output length, priority, deadline — never the
realized output length).

**It does not replace or observe vLLM's own internal scheduler.** Whatever
batching, KV-cache paging, and request interleaving vLLM does internally
once a request is admitted is entirely vLLM's own decision, invisible to
this harness — exactly as documented for the earlier hosted-API pilots in
`docs/real_llm_simulator_integration_plan.md`. The one thing this pilot
newly enables, that Cohere/Gemini never could, is comparing our own
admission-ordering policies against each other on a real backend — because
vLLM is a serving stack we can actually run and query, not a black box.

## Environment

- Real vLLM server, isolated venv (`/home/soroush/.venvs/vllm_baseline_pilot`,
  separate from the repo's own `modal-venv`) — see
  `experiments/real_llm/vllm_healthcheck_20260703T171021Z/` for the
  install/health-check record.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, served with `--enforce-eager`
  (no CUDA-compiler toolchain installed system-wide — see that same
  health-check directory's `reproducibility.md` for why, and the three
  environment-variable-only fixes used to get vLLM running: pointing
  `CUDA_HOME`/`PATH` at the pip-installed `nvcc`/`ninja`, and
  `VLLM_USE_FLASHINFER_SAMPLER=0` to bypass a FlashInfer JIT-kernel/CUDA
  version mismatch).
- No hosted API (Cohere/Gemini/OpenAI/Azure/Fireworks/CloudRift) was called
  anywhere in this pilot.

## Warm-up (Part B)

vLLM's first inference request under `--enforce-eager` triggers a one-time
Triton kernel JIT compile (observed in the health check: needed a 180s
client timeout on the very first request; ~0.3s afterward). Before any
measured policy run, this pilot issues one short/target=64 and one
medium/target=128 request at concurrency=1, written to
`warmup_requests.jsonl`/`warmup_summary.md` and **excluded from every
policy metric and `requests.jsonl`**.

No JIT-spike outliers were observed in the measured phase: the maximum
`server_request_latency_seconds` across all 144 measured requests was
1.436s, with p99 at 1.435s — no request exceeded 2x the p99, so no
suspected JIT artifact was silently dropped or needed flagging beyond the
warm-up itself.

## Request plan (Part C)

One fixed plan, built once (seed `20260703`) and reused unmodified for
every policy — same prompts, arrival order, priorities, deadlines, and
target output lengths in every run:

- Prompt buckets: short, medium
- Target output tokens: 64, 128
- Concurrency levels: 1, 2, 4
- Requests per cell: 2
- **Total: 2 × 2 × 3 × 2 = 24 requests per policy**, saved to
  `request_plan.jsonl` with `request_id`, `prompt_bucket`,
  `target_output_tokens` (used as `predicted_output_tokens` — the
  policy-visible estimate), `concurrency_level`, `intended_prompt_tokens`,
  `priority`, `slo_slack_seconds`, `class_id`, and `prompt_text`.
- Priority/deadline classes reused verbatim from
  `src/llmserveopt/workloads/synthetic.py`'s `DEFAULT_SLO_CLASSES`
  (tight/medium/loose, weights 0.2/0.5/0.3) — the same convention the
  simulator itself uses, not a bespoke scheme invented for this pilot.

## Policies compared (Part D)

| Policy | Wired? | Notes |
|---|---|---|
| `vllm_direct` (= `vllm_default`) | ✅ | Naive concurrency-bounded arrival-order submission, no `Action`/`ObservableState` machinery at all |
| `fifo` | ✅ | `src/llmserveopt/policies/fifo.py`, admits oldest-arrived first |
| `edf` | ✅ | Earliest-deadline-first |
| `shortest_output_first` | ✅ | SRPT-style on `predicted_output_tokens` |
| `least_laxity_first` (`llf`) | ✅ | Laxity = deadline − now − estimated service time |
| `estimated_service_time_first` (`estf`) | ✅ | SJF-style on `α·prompt_tokens + β·predicted_output_tokens` |
| `generated_heuristic` / `best_generated` | ❌ **Not wired** | See below |
| `selector` | ❌ **Not wired** | See below |

**Why the generated heuristic / selector are not wired** (investigated, not
assumed — see `scripts/run_vllm_external_baseline_comparison.py`'s
module-level comment for the full trail):

1. Most of the selector's 18 features (`src/llmserveopt/selector/
   features.py`: queue length, active count, prompt/output-length stats,
   slack stats, arrival rate/burstiness, recent SLO-violation rate) **are**
   reconstructable from this harness's own client-side bookkeeping. Only
   `kv_utilization` has no honest client-side substitute without scraping
   vLLM's `/metrics` endpoint (not implemented here).
2. More fundamentally: every serialized selector model artifact found on
   disk (`results/phase2a2_selector_dataset/`, `phase2a3_selector_eval/`,
   `phase2a4_2b4_final_eval/` — all `*.joblib`) was trained under the
   **pre-correction objective** that Phase 2B.14's metric audit found
   flawed (completed-only weighted-goodput denominator). The corrected,
   validated retraining (Phase 2B.15/16 — the one actually described as
   "best" in this project's own records) was evaluated only in-memory by
   one-off scripts and was **never persisted as a loadable model**.
   Loading a pre-correction artifact and presenting it as "our best
   selector" would misrepresent this project's own findings, so it was not
   done.

Wiring either safely would require: re-running the Phase 2B.15/16
corrected-objective training pipeline to produce a fresh, current model
artifact, then building the feature adapter described in point 1 above
(with an honest placeholder or `/metrics` scrape for `kv_utilization`),
plus tests. Not done in this task — see "Recommended next step" below.

## External admission-controller semantics (Part E)

Implemented in `run_cell_for_policy()` in
`scripts/run_vllm_external_baseline_comparison.py`:

- Requests arrive in a waiting queue (all at `arrival_time=0` within a
  cell — a burst-arrival model, matching the concurrency-sweep convention
  already used for the hosted Cohere/Gemini pilots).
- Up to `concurrency_level` requests may be in flight at once (a real
  `ThreadPoolExecutor(max_workers=concurrency_level)`; the GPU/KV/batch
  constraints in `ObservableGPUState` are set generously high so the
  **only** binding constraint is this concurrency cap — the policies'
  `_feasible_on_gpu()` check therefore reduces to "does an admission slot
  exist," matching this pilot's intent).
- Whenever a slot frees (a request completes, successfully or not), the
  selected policy's `select_action()` is re-invoked on a **fresh snapshot**
  of `ObservableGPUState`/`ObservableState` — never the harness's own live,
  mutated state object (a bug caught and fixed during development: policies
  mutate their input `ObservableGPUState` as internal bookkeeping, which
  must never leak into this harness's authoritative admission-slot
  tracking, or a slot leaks permanently once concurrency < requests
  pending).
- No policy ever sees `actual_output_tokens` — only `predicted_output_tokens`
  (`= target_output_tokens`, known before generation), exactly mirroring
  the simulator's own `ObservableRequest` contract.
- Failed/timed-out requests are recorded with `slo_violated=None` (never
  silently dropped) and count against the policy in the arrival-normalized
  objective below.
- The identical plan, SLO classes, and priorities are reused for every
  policy — no policy saw an easier or harder workload than any other.

## Objective (Part H)

`arrival_normalized_weighted_goodput = completion_fraction × conditional_WG`
where `conditional_WG = Σ(priority · met) / Σ(priority)` over **completed**
requests only, and `completion_fraction = n_completed / n_total` — so
failed/timed-out/dropped requests correctly count as zero in the
arrival-normalized numerator, matching the corrected-objective convention
established in Phase 2B.14/15/16 (`scripts/run_phase2b14_metric_audit_
scorpio_ablation.py`'s `arrival_normalized_wg`), not the older
completed-only-denominator metric that audit found misleading.

## Results (real, from the tiny 24-req/policy run)

| Policy | Completed | Failed | Arrival-norm. WG | SLO viol. (completed) | TTFT mean (s) | Server latency mean (s) | Req/s |
|---|---|---|---|---|---|---|---|
| edf | 24/24 | 0 | **0.7955** | 0.125 | 0.0157 | 0.882 | 1.70 |
| estimated_service_time_first | 24/24 | 0 | **0.7955** | 0.125 | 0.0162 | 0.930 | 1.64 |
| least_laxity_first | 24/24 | 0 | **0.7955** | 0.125 | 0.0161 | 0.879 | 1.70 |
| fifo | 24/24 | 0 | 0.7500 | 0.167 | 0.0159 | 0.937 | 1.58 |
| shortest_output_first | 24/24 | 0 | 0.7500 | 0.167 | 0.0160 | 0.929 | 1.64 |
| vllm_direct | 24/24 | 0 | 0.7500 | 0.167 | 0.0160 | 1.015 | 1.51 |

**Zero failures across all 6 policies × 24 requests = 144 total measured
requests** (plus 2 warm-up requests, also succeeded).

## Did our method beat any baselines?

The three deadline/service-time-aware policies already in this project's
registry (EDF, LLF, ESTF) **outperform** naive/length-only ordering
(FIFO, shortest-output-first, direct submission) on this tiny real-vLLM
workload: arrival-normalized WG 0.7955 vs. 0.7500, i.e. one fewer SLO
violation out of 24 (3/24 vs. 4/24). This is directionally consistent with
what the simulator itself predicts for these same policies, now confirmed
against a real serving backend for the first time — **not** a comparison
against vLLM's own scheduler (see "what this does not test" above), and
**not** a comparison including this project's generated
heuristic/selector, which are not wired (see Part D). With n=24/policy,
treat this as a smoke-scale directional signal, not a statistically
powered result.

## A real, measured caveat: output length vs. target

Unlike the hosted Cohere/Gemini v2 pilots, Qwen2.5-0.5B-Instruct largely
did **not** self-stop near its target length: `finish_reason: length` (cut
off by `max_tokens`, set to `2× target_output_tokens`) for the large
majority of requests, mean realized-length ratio 1.84× target across all
144 measured requests. `predicted_output_tokens` (what the policies see)
is still valid and honestly declared before generation — this caveat is
about interpreting `mean_output_tokens` in the aggregates, not about the
scheduling comparison's validity.

## GPU memory

| | Used | Total |
|---|---|---|
| Before pilot | 8267 MiB | 16311 MiB |
| After pilot | 8269 MiB | 16311 MiB |

Effectively unchanged (2 MiB drift, well within noise) — vLLM's KV-cache
pool is pre-allocated at server startup, so a 144-request tiny pilot does
not visibly change reported usage.

## Artifacts

`experiments/real_llm/vllm_baseline_comparison_pilot_20260703T172103Z/`:
`request_plan.jsonl`, `requests.jsonl`, `summary.json`, `summary.md`,
`aggregate_by_policy.csv`, `aggregate_by_concurrency.csv`,
`aggregate_by_target_output_tokens.csv`, `aggregate_by_prompt_bucket.csv`,
`manifest.json`, `run_config.json`, `reproducibility.md`,
`warmup_requests.jsonl`, `warmup_summary.md`, `server_status.json`,
`gpu_mem_before.txt`, `gpu_mem_after.txt`, `run.log`,
`tmux_pilot_command.txt`, `errors.jsonl` (empty — 0 failures).

## Paper-safe conclusion

**Tiny real vLLM external-admission comparison.** Real server, real
requests, real measured differentiation between admission policies already
in this project's registry. This project's generated heuristic and
selector were **not** included — they are not yet safely wirable against a
real server (see Part D) and must not be described as compared here.

## Recommended next step

1. **Re-run the Phase 2B.15/16 corrected-objective selector training to
   produce a persisted model artifact**, then build the client-observable
   feature adapter (Part D) so a future pilot can honestly include our
   selector — this is the highest-value gap, since it's the piece missing
   from "our best method vs. external baselines."
2. Alternatively, scale this same 6-policy comparison to the full
   108-request grid (matching the Cohere/Gemini v2 shape) for a tighter
   signal on the EDF/LLF/ESTF-vs-FIFO gap already observed.
3. Longer term, scrape vLLM's `/metrics` endpoint for real
   `kv_utilization`, closing the one selector feature this harness cannot
   currently reconstruct from client-side state alone.

## See also

- `experiments/real_llm/vllm_healthcheck_20260703T171021Z/` — the install/
  health-check record this pilot builds on.
- `docs/real_llm_simulator_integration_plan.md` — why hosted-API latency
  and controllable serving are different measurements, and the vLLM
  external-baseline rationale.
- `scripts/run_vllm_external_baseline_comparison.py` — the harness itself.
