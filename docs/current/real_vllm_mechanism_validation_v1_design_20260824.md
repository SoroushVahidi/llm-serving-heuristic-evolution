# real_vllm_mechanism_validation_v1 Design

Date: 2026-08-24

Status: `FEASIBILITY_BLOCKED`

Scope: feasibility and preregistration only. No scientific real-vLLM
comparison was launched. No Wulver, SLURM, GPU model run, external API,
TEST, FINAL, or DEV-driven redesign was used.

## A. Preflight

Repository: `/home/soroush/llm-serving-heuristic-evolution`

| Field | Value |
| --- | --- |
| Branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `2987b7181efa2bc550d8a894c537eca8f6393eb6` |
| Upstream | `origin/contextual-compositional-heuristics-20260731` |
| Ahead/behind | `0 2` from upstream to local, so local is ahead by 2 commits |
| Worktrees | one worktree at repo root |
| Git locks/merge/rebase/cherry-pick/bisect | none observed |

The worktree contains many existing untracked local scientific artifacts. They
were preserved.

## B. Local GPU Inventory

`nvidia-smi` reports one GPU:

| Field | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Count | 1 |
| VRAM | 16,311 MiB |
| Used at audit | 15 MiB |
| Free at audit | about 16.3 GiB |
| Driver | 580.173.02 |
| CUDA reported by driver | 13.0 |
| Active GPU process | Xorg only, 4 MiB |
| Topology | single GPU |

Host resources: 62 GiB RAM, about 58 GiB available; 654 GB disk available on
the repo filesystem; CPU is Intel i7-12700K, 20 logical CPUs.

## C. Software / vLLM Inventory

Active Python:

| Component | Value |
| --- | --- |
| Python | `/home/soroush/modal-venv/bin/python`, version 3.12.3 |
| PyTorch | 2.12.0+cu130 |
| Transformers | 5.8.1 |
| vLLM | not installed |
| Ray | not installed |

The historical vLLM environment referenced by prior docs,
`/home/soroush/.venvs/vllm_baseline_pilot`, is absent on this workstation.
The current repo scripts still assume that path by default in
`scripts/run_gpu_external_validity_audit.py`.

Prior local artifacts record successful vLLM 0.24.0 runs on this same class of
GPU with `Qwen/Qwen2.5-0.5B-Instruct`, including a stress run with queueing in
8/8 scenarios, max running sequences 2, max observed KV usage 0.0214, and
140/140 completed requests. Those prior results establish that the hardware can
support a small real-vLLM pressure test, but the current software state cannot
run a probe without restoring/installing vLLM.

## D. Model Choice

Chosen model for the first local validation, once vLLM is restored:
`Qwen/Qwen2.5-0.5B-Instruct`.

Rationale:

- It is already cached locally with a complete snapshot of about 0.93 GiB.
- It is open-weight and was already used successfully in prior local vLLM runs.
- The config advertises 32,768 max position embeddings, enough for controlled
  4k-context prefill/decode experiments.
- It fits comfortably in 16 GB VRAM, leaving room for KV/cache and server
  overhead.

Do not choose the cached 7B/8B snapshots for the local first run. Their bf16
snapshots are 14.2-15.0 GiB before vLLM overhead or KV cache, which is too
tight for a controlled long-context validation on a 16.3 GiB GPU. A 7B/xlong
KV-pressure validation is a Wulver/A100-class task.

## E. Local vs Wulver Verdict

Exact verdict: `FEASIBILITY_BLOCKED`.

Interpretation:

- Local single-GPU hardware appears sufficient for the priority-A
  prefill/decode contention validation with `Qwen/Qwen2.5-0.5B-Instruct`.
- Local software is currently blocking: no vLLM import/CLI is available, and the
  historical vLLM environment is missing.
- Wulver is not scientifically required for the first small prefill/decode
  validation.
- Wulver is required if the paper needs a 7B-class or high-KV-pressure
  validation comparable to previous A100 xlong runs.

## F. Mechanism Fidelity Map

| Simulator mechanism | vLLM implementation/control | Fidelity | Notes |
| --- | --- | --- | --- |
| ESTF | client-side external admission by estimated prompt+output service | APPROXIMATE | Faithful to repo policy logic, not vLLM internal scheduler replacement |
| WFS | possible client-side admission by class deficit/priority/service | APPROXIMATE | Current real-vLLM harness does not wire WFS yet |
| LLF | client-side external admission; already wired in real-vLLM harness | APPROXIMATE | Uses online deadline and service estimate |
| chunked/full prefill | native vLLM flags `--enable-chunked-prefill` / `--disable-chunked-prefill` plus token/sequence limits | EXACT | Strongest real validation target |
| KV-aware scheduling | induce/measure KV pressure with max model len, memory utilization, max seqs; no faithful KV scheduler hook | APPROXIMATE | High KV pressure likely needs Wulver/A100 |
| queue wait | vLLM Prometheus waiting metric plus client timestamps | EXACT | Prior local stress observed waiting |
| TTFT | streaming first-token timestamp | EXACT | Existing harness records it |
| SLO/goodput | harness-level deadlines and priorities | APPROXIMATE | Valid as controlled workload metric, not vLLM-native SLO |
| KV pressure | vLLM Prometheus KV-cache usage and preemption metrics | EXACT for measurement | Local induction not yet proven beyond low pressure |

## G. Real Policies To Test

Primary real-system comparison:

1. `real_vllm_full_prefill`: vLLM server with chunked prefill disabled, prefix
   caching disabled, fixed max model length, max sequences, and block size.
2. `real_vllm_chunked_prefill`: vLLM server with chunked prefill enabled and
   fixed `max_num_batched_tokens=512`, all other controls matched.

Secondary optional comparison, only after harness review:

- `estimated_service_time_first` vs `weighted_fair_share` as client-side
  external admission controllers. This is lower fidelity than prefill/chunking
  because it does not replace vLLM's internal scheduler.

Excluded from the first local run:

- KV-pressure/urgency validation. Prior local 0.5B stress reached only 2.1%
  KV usage. Prior A100 7B xlong reached 83.7% KV usage; that class of test is
  Wulver-scale.

## H. Workload Design

Use four compact prefill/decode regimes:

| Regime | Geometry | Expected qualitative winner |
| --- | --- | --- |
| `pd_real_hog_ttft_low_late` | long-prompt hog convoy, modest late short tenants, hog-tight TTFT/SLO | full prefill |
| `pd_real_late_ttft_low_late` | same geometry, late tenants tight | chunked prefill |
| `pd_real_hog_ttft_high_late` | higher late pressure, hog-tight | full prefill, possibly smaller margin |
| `pd_real_late_ttft_high_late` | higher late pressure, late-tight | chunked prefill |

Prompts are deterministic synthetic text targeting long hog prompts
approximately 4k-8k tokens and late prompts approximately 64-256 tokens.
Outputs are controlled at 64-128 tokens. Client concurrency should be high
enough to build a server waiting queue while vLLM `max_num_seqs` is kept small
for local pressure.

## I. Metrics

Primary:

- TTFT by class
- E2E latency by class
- arrival-normalized weighted goodput when SLO labels are valid
- SLO attainment by class

Secondary:

- throughput
- vLLM waiting queue
- running sequence count
- KV-cache utilization
- preemption count
- prompt/decode token statistics where exposed by vLLM metrics

## J. Repetition / Statistics Plan

- Warm up each server mode with at least one short and one medium request.
- Run 5 measured repetitions per treatment/regime.
- Randomize or alternate treatment order with a fixed seed.
- Analyze qualitative sign stability, not exact numeric simulator agreement.
- Use paired per-regime comparisons where treatment order and workloads are
  matched.

Prefill/decode corroboration criterion: at least 3/4 regimes have the predicted
sign, with a material TTFT difference of at least 5% or ANWG/SLO difference of
at least 0.02 in at least two regimes, and no systematic request failures.

Overall preregistered outcome labels:

- `REAL_MECHANISM_VALIDATION_STRONG`: at least two major mechanism families show
  stable qualitative corroboration.
- `REAL_MECHANISM_VALIDATION_PARTIAL`: one family is corroborated, or additional
  effects are noisy/incomplete.
- `REAL_MECHANISM_VALIDATION_NO_GO`: mechanisms do not reproduce or fidelity is
  inadequate.

With the current local plan, the maximum likely outcome is `PARTIAL` unless the
fairness/completion external-admission comparison is added and passes a separate
fidelity review.

## K. Tiny Feasibility Result

No live tiny probe was run.

Reason: `python -m vllm` fails with `No module named vllm`, `command -v vllm`
finds no executable, and the historical vLLM virtualenv is absent. Running a
probe would require a software/environment change first.

## L. VRAM / KV Headroom

For `Qwen/Qwen2.5-0.5B-Instruct`, the local 16 GB GPU has sufficient headroom
for the proposed 4k-context prefill/decode validation. Prior local stress with
`gpu_memory_utilization=0.25`, `max_model_len=4096`, `max_num_seqs=2`,
`max_num_batched_tokens=512`, and chunked prefill enabled completed 140/140
requests and created waiting queues in every scenario.

For 7B/8B bf16 cached models, local headroom is inadequate for a meaningful
long-context/KV-pressure design because weights alone occupy roughly 14-15 GiB.

## M. Expected Run Cost

Once vLLM is restored, a non-scientific health probe should be under a few
minutes plus first-request warmup. The preregistered small A-family validation
should be local CPU/GPU scale, plausibly tens of minutes depending on server
startup and repetition count. A 7B/KV-pressure validation should be treated as
Wulver/A100 work.

## N. Readiness

Current readiness: not ready to launch.

Blocking item: restore or create a controlled local vLLM environment. After
that, run only a tiny non-scientific health probe to verify model load,
concurrency, TTFT instrumentation, and vLLM metrics before launching the
scientific validation.

## O. Exact Next Task

Restore the local vLLM runtime environment, preferably matching the prior
documented vLLM 0.24.0 setup, then run a tiny non-scientific health probe with
`Qwen/Qwen2.5-0.5B-Instruct`. Do not run the scientific comparison until that
probe passes and the run command is preregistered.

## P. Safety Confirmation

No Wulver job, SLURM job, GPU model run, external API call, TEST/FINAL access,
scientific experiment, or heavy simulation was launched. Existing frozen result
artifacts were not modified. Git state was not reset, cleaned, stashed, rebased,
merged, pushed, or otherwise destructively changed.
