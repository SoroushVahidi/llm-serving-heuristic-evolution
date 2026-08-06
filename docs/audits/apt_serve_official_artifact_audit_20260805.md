# Apt-Serve Official Artifact Audit — 2026-08-05

Audit-only pass, per this task's explicit instructions: does not implement
Apt-Serve, does not touch CC5/CC6, the canonical benchmark suite, or
VTC/PARS/vLLM-LTR/Sarathi results. This document is the complete
scientific, engineering, and environment audit preceding any
integration decision, following the pattern established by
`docs/audits/sarathi_official_artifact_audit_20260805.md`.

## 0. Repository state at start (task step 1)

- Branch: `contextual-compositional-heuristics-20260731`
- Starting SHA: `74c7fc8674fced9ed38583362606b7d90fcc956e`
- Upstream: 0 ahead / 0 behind, working tree clean
- Status checker + resume-readiness: both passed
- No active job writing repository files (checked `ps aux`; only
  unrelated system/unattended-upgrade and an unrelated local web app
  process running)
- Confirmed baseline state, from `docs/BASELINE_STATUS.md`:
  - vLLM-LTR: `EVALUATION_ONLY`
  - PARS-Serve-2026: `EVALUATION_ONLY`
  - VTC: `FOUNDATIONAL_CANDIDATE` (scientific classification, not registered)
  - Sarathi-Serve: stress-test catalog coverage complete (7 entries,
    `docs/audits/sarathi_stress_test_catalog_completion_20260805.md`)
  - Apt-Serve: confirmed **zero existing code** anywhere in this
    repository (`grep -rin "apt.serve\|apt_serve"` across `docs/`,
    `src/`, `configs/`, `tests/`, `baselines/`, `scripts/` matches only
    prose mentions in `docs/BASELINE_STATUS.md` ["Not implemented"],
    `docs/external_baseline_decision.md` [rationale for NOT including
    it as a first-paper baseline: "High" difficulty, "No" as selector
    candidate], `docs/external_baseline_coverage_report.md` ["None"],
    and `docs/research/algorithm_stress_tests/ALGORITHM_INVENTORY_20260805.md`
    ["zero existing code... not locatable with confidence in the
    literature pass"]). **This is a genuine cold start**, confirmed by
    repository evidence, not assumed.

## 1. Official Apt-Serve sources (task step 2)

| Field | Value |
|---|---|
| Title | "Apt-Serve: Adaptive Request Scheduling on Hybrid Cache for Scalable LLM Inference Serving" |
| Authors | Shihong Gao, Xin Zhang, Yanyan Shen, Lei Chen |
| Affiliations | HKUST, HKUST (Guangzhou), Shanghai Jiao Tong University |
| Venue | Proceedings of the ACM on Management of Data (PACMMOD), Vol. 3, No. 3 — the SIGMOD 2025 journal-track venue ("SIGMOD 25" per the repo's own README) |
| DOI | 10.1145/3725394 |
| arXiv | [2504.07494](https://arxiv.org/abs/2504.07494) (submitted 2025-04-10, v1; PDF confirmed 28 pages, matching the published article length) |
| Official repository | `github.com/eddiegaoo/Apt-Serve` — verified author-owned (matches paper's own footnote "Publicly available at: https://github.com/eddiegaoo/Apt-Serve", page 130:15) |
| Repository created / last push | 2025-03-31 / 2025-06-09 (verified via `gh api`) — no activity in ~14 months as of this audit |
| Tags / releases | None (`gh api .../tags`, `.../releases` both empty arrays) |
| License | **None** — no LICENSE file in the repo, `license.spdx_id: null` via GitHub API. Same disclosure category as PARS-Serve-2026 (`baselines/pars/PROVENANCE.md`'s precedent: disclosed, not hidden) |
| Repository history | Not a normal incremental development history — a one-time bulk upload (2025-03-31, "Add files via upload"/"Create __init__.py" commits) plus one README edit (2025-04-12) plus **exactly one** later commit (2025-06-09, "Update insert_designs.sh") |
| Paper-era vs. current commit divergence | **None that matters.** The only post-initial-upload commit (`c953217988`, 2025-06-09) touches exactly one file: `additional_designs/insert_designs.sh` (the file-copy installer script) — confirmed via `gh api .../commits/c953217988` file list. The actual algorithm files (`aptserve_scheduler.py` etc.) are unchanged since the initial upload. This is a **single-shot artifact release**, structurally different from Sarathi-Serve's continuously-evolving repo — no `ceaa0660`-vs-`96f9911`-style drift question applies here. |
| Datasets | ShareGPT, HumanEval (164 problems), LongBench — all standard, publicly available; sampling scripts provided (`sample_requests_from_datasets/sample_from_{sharegpt,humaneval,longbench}.py`). Ultra-long-context generalization study (§6.7 of the paper) additionally uses WikiText, Arxiv, BookCorpus. |
| Checkpoints/models | None released or required — uses standard HuggingFace `facebook/opt-13b`/`opt-30b`/`opt-66b` (downloaded from HF Hub, not authored by the paper) for main experiments; `LLaMA3-8B-Instruct262K`, `Yi-6B-200K` for the ultra-long-context generalization study (§6.7) |
| Benchmark scripts | `gen_client_requests.py` (client-side request-rate/CV-controlled load generator), `backend_request_func_SLO.py` (SLO-aware backend request function) |
| Expected models/hardware (paper) | OPT-13B/30B/66B on 1/2/4× A100 (40GB each), NVLink, FP16, tensor parallelism for multi-GPU (Table 2, page 130:17) |

No competing "official" repository exists; `eddiegaoo/Apt-Serve` is the
sole author-linked artifact, directly cited by the paper's own footnote.

## 2. Reproducibility audit (task step 3)

| Dependency | Official requirement | Evidence |
|---|---|---|
| Base system | vLLM **0.5.0.post1** exactly (not "≥" or "compatible with") | README: "Install the backbone system (vLLM 0.5.0. post1) first" |
| Python | ≥3.8 (inherited from vLLM 0.5.0.post1's own `requires_python`) | PyPI metadata for `vllm==0.5.0.post1` |
| PyTorch | **`torch==2.3.0`** exact pin | PyPI `requires_dist` for `vllm==0.5.0.post1` |
| Attention/kernels | `xformers==0.0.26.post1`, `vllm-flash-attn==2.5.9` (both exact pins), plus a **custom raw CUDA kernel** (`additional_designs/mixed_cache_kernels/mixed_cache.cu`) built via `torch.utils.cpp_extension.CUDAExtension`/`BuildExtension`, no explicit `TORCH_CUDA_ARCH_LIST` set (relies on auto-detection) | PyPI metadata; repo file inspection |
| Compiler/build | `cmake`, `ninja`, `setuptools≥49.4.0` (vLLM's own build-system requirements, confirmed triggered live — see §10) | Live probe (§10) |
| Installation method | **Not pip-installable as a self-contained package.** `insert_designs.sh` literally `cp`'s 13 Python files over the corresponding files inside an already-`pip install`'d vLLM package directory (e.g. `cp ./core/aptserve_scheduler.py "$vllm_dir/core/scheduler.py"`), then a separate CUDA extension build step. No Docker image, no conda `environment.yml`, no `requirements.txt` in the repo itself. | `insert_designs.sh` full contents (read directly) |
| Supported models | **OPT architecture only.** `additional_designs/model_executor/models/` contains exactly one file, `aptserve_opt.py` — no Llama/Mistral/etc. support added (the ultra-long-context study's LLaMA3/Yi models are evaluated only via vanilla vLLM's own pre-existing model support, comparing against Apt-Serve on OPT-family models elsewhere; confirmed by file listing, not inferred from the paper text alone) |
| Multi-GPU | Yes, tensor-parallel via NCCL (paper Table 2: 2×A100 for 30B, 4×A100 for 66B) — inherited from vLLM's own TP implementation, not a new mechanism |
| Expected GPU memory | 40GB per A100 (paper's exact test hardware); no explicit minimum stated, but the 13B/26GB-weights configuration is the smallest tested |
| Datasets availability | Public (ShareGPT, HumanEval, LongBench all standard/downloadable); sampling scripts provided and runnable independent of the GPU/CUDA stack |
| Expected runtime | Not explicitly stated for full reproduction; Table 6 (page 130:22) gives scheduling-algorithm-only overhead (0.3-10.8ms for 50-1600 candidate requests) — a small fraction of total experiment runtime, which is dominated by actual GPU inference, not documented in wall-clock terms in the paper |
| Figure/table reproducibility from public materials | The 3 main datasets, sampling scripts, and full source are public; the exact request traces used in the paper's own runs are NOT published as fixed artifacts (traces are freshly sampled via Poisson-distributed arrivals seeded by the provided scripts, not a frozen `.jsonl`) — meaning results are reproducible *in distribution*, not bit-for-bit from a shipped trace file |

**Classification: CODE_ONLY.**

Justification: source code for the core algorithmic contribution
(scheduler, hybrid-cache assigner, custom CUDA kernel) is public and,
per the greedy-scheduling appendix and paper text, matches the
described algorithm closely (verified directly — see §4). Datasets and
sampling scripts are public. However: (a) no Docker/conda environment
manifest is provided — the install path is a manual "clone vLLM,
`pip install`, then overwrite 13 files, then compile a CUDA extension"
sequence with zero automation beyond a single `.sh` script and no
pinned transitive-dependency lockfile beyond what vLLM 0.5.0.post1
itself declares; (b) no exact request traces are shipped, only
generators — full bit-for-bit figure reproduction is not possible from
public materials alone, only distributional reproduction; (c) no
released checkpoints are needed (uses stock HF models), which is a
point in favor, but does not offset (a)/(b). This falls short of
MOSTLY_REPRODUCIBLE (which would require closer to a working
environment recipe) but is well above INSUFFICIENT_PUBLIC_ARTIFACT
(the actual mechanism-critical code, including the CUDA kernel and the
formal approximation-guarantee proof, is fully public and internally
consistent).

## 3. Understanding Apt-Serve precisely (task step 4)

All of the following is grounded directly in the paper (arXiv:2504.07494,
28pp, read in full), the greedy-scheduling appendix PDF
(`greedy_scheduling_appendix.pdf`, read in full, 3pp), and the actual
`additional_designs/core/aptserve_scheduler.py` source (2,220 lines,
read in full via `gh api`) — not inferred from filenames.

### Hybrid cache — what it stores

Two cache types share one unified, block-wise memory pool
(paper §4.3, Figure 6):

- **KV cache** (standard): stores key/value vectors `k_i^ℓ`, `v_i^ℓ` per
  token per Transformer layer — the conventional mechanism.
- **Hidden cache** (novel): stores the **input hidden state vector**
  `x_i^ℓ` per token per layer instead — from which `k_i^ℓ`/`v_i^ℓ` can be
  recomputed on demand via the same linear projections already used in
  the self-attention module (Eq. 1 in the paper). Costs exactly **half**
  the memory of KV cache per token (one vector `x_i^ℓ` vs. two vectors
  `k_i^ℓ`+`v_i^ℓ`, same dimension), at the cost of an extra `O(n)`
  on-the-fly linear transformation to reconstitute K/V when a
  hidden-cached request needs to attend (vs. `O(1)` direct read for
  KV cache) — Figure 3 in the paper illustrates this exactly.

### Cache object lifetime / admission / eviction

Cache type is **not fixed at admission** — it is re-decided every
scheduling iteration as part of the greedy batch-selection process
(§4.2-4.3, §5 of the paper). A request can be KV-cached in one iteration
and hidden-cached in the next (or vice versa) if the greedy algorithm's
marginal-value calculation favors a switch; a cache-type switch requires
discarding the existing cache and **re-running a prefill iteration** to
recompute the cache in the new type (paper, end of §5: "a request may
need to switch cache types according to the scheduling result of a
particular iteration. In such cases, Apt-Serve discards the existing
cache and schedules a prefill iteration to recompute the cache in the
required type"). Confirmed independently in the source
(`aptserve_scheduler.py`'s `greedy_selection_decode`: "For KV --> hidden
or hidden --> KV, the implementation now is preempt them first any
way"). Eviction is implicit: the greedy knapsack packing (§below) simply
does not select lower-marginal-value candidates once the memory budget
`M^e` is exhausted; there is no separate LRU/LFU eviction policy.

### Request value calculation ("quantification model," paper §4.2, Eq. 5-9)

For request `i` at iteration `e`:

```
g_i^e = p_i^e − β_i^e·(|W^e|+|R^e|)·t_i^e         (Eq. 5)
t_i^e = ρ·m_i^e                                    (Eq. 6)
```

where `p_i^e` is the request's **pending time so far** (elapsed time
since arrival, if never yet prefilled, or since its last received
token — a purely *observed*, not *predicted*, quantity — Apt-Serve
never estimates a request's *future* remaining length, unlike
ESTF/vLLM-LTR/PARS in this project), `β_i^e ∈ {0,1}` is whether hidden
cache is used, `m_i^e` is the request's current max memory requirement,
`ρ` is an empirically-calibrated linear coefficient (paper: "a marginal
preprocessing cost of approximately 30 seconds in practice" to
calibrate), and `(|W^e|+|R^e|)` (waiting-queue-size plus running-queue-size)
is a **system-contention scaling factor** that makes the hidden-cache
"extra latency" penalty grow with overall load — under low contention,
hidden cache's throughput cost barely matters; under high contention, it
matters more, since `t_i^e`'s effect on *every other* pending request
(not just `i`) is what's being penalized.

### Batch-composition optimization (Definition 1, paper §5, page 130:14)

Formal 0-1 integer program:

```
max  Σ g_i^e·α_i^e
s.t. Σ (1 − β_i^e/2)·m_i^e·α_i^e ≤ M^e     (memory constraint)
     α_i^e, β_i^e ∈ {0,1}
```

Proven **NP-hard** by direct reduction from 0-1 knapsack (the case
`β_i^e=0 ∀i`, no hidden cache allowed, is exactly 0-1 knapsack — paper
page 130:14, explicit).

### Greedy approximation + proven guarantee

`greedy_scheduling_appendix.pdf` (fetched and read directly, not
summarized secondhand): Algorithm 1 sorts all `(request, cache-type)`
candidate marginal-gain-per-memory ratios `θ_i` in decreasing order,
greedily admits until the memory budget is exhausted, with a
"double-check" tweak (compare against the single best-would-have-been-admitted
candidate `V'`, take whichever of `V`/`V'` is larger) specifically to
bound the worst case. **Theorem 1** (proven in the appendix via an
LP-relaxation argument): `Φ_OPT / Φ_ALG ≤ 2` — **the greedy algorithm is
formally a 2-approximation**, `O(n log n)` complexity (dominated by the
initial sort). This is a genuine, proven worst-case bound on the
algorithm's own suboptimality, not a hypothesis — classified
`PROVEN_WORST_CASE` in §4 below.

### Scheduling rule / tie-breaking / admission

Two-stage per-iteration decision (`_dynamic_priority` in the source,
matching paper §5's "Deciding the Iteration Type"):
1. Compute `waiting_urgency` (total elapsed pending time summed over the
   waiting queue) and `running_urgency` (same, summed over the running/
   decoding queue). If `running_urgency > waiting_urgency` (or nothing
   schedulable from the waiting side), do a **decode** iteration this
   step; otherwise do a **prefill** iteration.
2. Within whichever queue is chosen, run the greedy knapsack
   (`greedy_selection_prefill`/`greedy_selection_decode`) over that
   queue's marginal-value-sorted candidates.

**SLO-aware fallback / tie-breaking for violated requests**: a request
whose pending time already exceeds its `ttft_slo`/`tbt_slo` gets its
scheduling value `g_i^e` substituted with a **near-zero constant**
(`dummy_net_profit`) — explicitly deprioritizing (not prioritizing)
already-violated requests, favoring using freed memory to help
requests that can still make their SLO rather than continuing to spend
memory on already-lost causes. This is a genuine, disclosed policy
choice with a real cost: paper §6.6 explicitly reports "a small fraction
of requests (10%) experience starvation" as a *consequence* of this
exact mechanism, with a proposed partial mitigation (a decaying-factor
variant "Apt-Serve's Scheduling*", evaluated in Figure 10) that the
paper presents as a supplementary result, not the main configuration.

### Interaction with continuous batching / recomputation / prefix reuse

Standard vLLM iteration-level batching is retained unmodified as the
outer loop (paper §2.2, explicitly: "Our proposed Apt-Serve framework
adopts the standard iteration-level batching... but introduces an
innovative adaptive scheduling mechanism"). Recomputation occurs (a) on
ordinary preemption (same as vanilla vLLM — evicted requests restart
from their prompt) and (b) specifically on a cache-**type** switch
(discard + re-prefill, as above) — a second, Apt-Serve-specific
recomputation trigger beyond vanilla vLLM's. No cross-request prefix
caching/reuse is modeled at all (Apt-Serve's "reuse" is intra-request
reuse-across-iterations of a single request's own KV/hidden state, not
inter-request shared-prefix caching — different concept from, e.g.,
vLLM's own `AlignedServe`/prefix-caching line of work cited in Related
Work).

### Hardware-specific assumptions

The custom `mixed_cache_ops` CUDA kernel ("fuses reshaping with
read/write operations" for fragmented block-wise hidden-cache access,
paper §6.1) is architecture-generic C++/CUDA (`mixed_cache.cu`, no
explicit arch pragmas found), but is only ever validated by the authors
on A100 (`sm_80`); the exact compute-capability portability is untested
by the authors and unverified by this audit.

### Exact source files (task step 4's "do not infer from filenames alone")

Read in full: `additional_designs/core/aptserve_scheduler.py` (2,220
lines — the entire scheduling/value/greedy-selection logic). Role
confirmed from direct usage within that file (not filename alone) for:
`additional_designs/core/aptserve_block_manager.py` (referenced via
`self.block_manager.num_total_gpu_blocks`/`.get_num_free_gpu_blocks()`
calls inside the scheduler — hybrid-cache memory bookkeeping),
`additional_designs/core/aptserve_interfaces.py` (block-manager
interface contract). Role confirmed from `insert_designs.sh`'s explicit
file-replacement mapping (not filename alone) for the remaining 10
files: `aptserve_block.py`/`aptserve_sequence.py` (data structures,
replace vLLM's `block.py`/`sequence.py`), `aptserve_llm_engine.py`
(engine, replaces `engine/llm_engine.py`), `aptserve_cache_engine.py`/
`aptserve_model_runner.py`/`aptserve_worker.py` (worker-side execution,
replace `worker/{cache_engine,model_runner,worker}.py`),
`aptserve_linear.py`/`aptserve_opt.py` (model-executor layer, replace
`model_executor/{layers/linear,models/opt}.py`), and the attention
stack (`aptserve_layer.py`/`aptserve_abstract.py`/`aptserve_flash_attn.py`,
replace `attention/{layer,backends/abstract,backends/flash_attn}.py`).
These 10 support files were not read line-by-line in this pass (a
disclosed scope limit, not a guess) — their role as "the mechanical
plumbing that makes the scheduler's `α_i^e`/`β_i^e` decisions physically
happen in GPU memory" is established from the scheduler's own calls
into them and the installer script's own explicit mapping, not filename
pattern-matching.

## 4. Scientific claim and limitation audit (task step 5)

**Primary objective**: maximize *effective throughput* — "the highest
sustainable online request rate that meets specified SLOs attainment
criteria (e.g., serving at least 70% of requests within target SLOs)"
(paper §1, page 130:2, explicit definition).

**Strongest contribution**: the hybrid KV/hidden-cache scheme
(orthogonal to prior chunked-prefill/prefill-decode-coalescing work —
paper explicitly states this re: Sarathi-Serve, §6.2: "orthogonal... and
can be combined together") plus a *proven* 2-approximation greedy
scheduler for the resulting NP-hard batch-composition problem.

**Assumptions required for benefit** (evidence-cited):

- Genuine KV-cache memory pressure / batch-size-limited regime. Motivating
  experiment: Figure 2a (page 130:7) shows SLO attainment collapses
  specifically once request rate pushes the system to operate "at the
  batch size limit" for a large fraction of serving time; the paper's
  own two identified bottlenecks (§3.1 KV-cache memory, §3.2 FCFS
  rigidity) are both LOAD-dependent, not present at low request rates.
- Sufficient reuse/cache lifetime for the hidden-cache half-memory saving
  to matter enough to offset its extra compute cost. Directly evidenced
  by §6.3's own comparative discussion (page 130:19): on HumanEval
  (short outputs → short per-request cache lifetime), "Sarathi-Serve and
  FastGen perform better... likely due to the short per-request cache
  lifetime... However, when per-request cache lifetime increases (e.g.,
  on ShareGPT), this [chunked-prefill-only] optimization alone is
  inadequate."
- Heterogeneous/bursty arrival and length distributions for the adaptive
  (vs. FCFS) scheduling half of the contribution to matter. §3.2's own
  motivating experiment (Figure 4a-4c) shows FCFS underperforming random
  scheduling specifically because "online requests often vary in
  sequence length... [FCFS] limits the system's flexibility."

**Documented limitations, evidence-cited and figure/page-precise:**

| Limitation | Evidence | Classification |
|---|---|---|
| Greedy scheduler's own worst-case suboptimality bound | Appendix Theorem 1: `Φ_OPT/Φ_ALG ≤ 2` | **PROVEN_WORST_CASE** |
| ~10% request starvation from the SLO-aware fallback (already-violated requests get deprioritized, not helped) | Paper §6.6, Figure 10, page 130:21 (empirically measured, with an author-proposed partial mitigation "Scheduling*") | **DOCUMENTED_LIMITATION** |
| Diminishing relative advantage on short-output/low-cache-lifetime workloads (HumanEval) vs. long-output ones (ShareGPT/LongBench) | Paper §6.3, page 130:19 (direct comparative discussion, cross-referencing Figure 8b) | **DOCUMENTED_LIMITATION** |
| Prefill/decode interference remains severe in ultra-long-context regimes (both Apt-Serve and vLLM struggle to exceed 60% TBT SLO attainment on BookCorpus/Yi-6B-200K at 0.5 req/s) | Paper §6.7, Figure 12, page 130:23 — authors explicitly note "Integrating disaggregated distributed architectures... could help address this" as future work, i.e., an acknowledged gap Apt-Serve does not itself solve | **DOCUMENTED_LIMITATION** |
| Cache-type-switch requires a full re-prefill (discard + recompute), an extra cost not present in KV-only systems | Paper, end of §5 (page 130:15) | **DOCUMENTED_LIMITATION** (structural, not separately quantified in the paper's own ablations) |
| Coefficient `ρ` (linear cost model for hidden-cache overhead) requires an offline calibration pass ("approximately 30 seconds") before serving begins | Paper §4.2, page 130:11 | **DOCUMENTED_LIMITATION** (a real deployment precondition, not evaluated for sensitivity to mis-calibration in the paper) |
| Only OPT architecture natively supported by the released code; main results (Figure 8, the paper's central claim) are entirely OPT-13B/30B/66B | Repo file inspection (`aptserve_opt.py` is the only model file) + paper Table 2 | **DOCUMENTED_LIMITATION** (external to the algorithm's own claims, but a real reproducibility/generalization caveat) |

**Paper-motivating stress cases (used by the authors themselves,
figure-cited):** Figure 1 (TTFT SLO attainment collapse vs. request
rate, vanilla vLLM, motivating the whole paper); Figure 2a/2b (batch-size-limit
time fraction vs. SLO attainment, isolating the KV-cache-memory
mechanism); Figure 4a-4c (FCFS vs. random scheduling, isolating the
batch-composition mechanism); Figure 9 (SLO attainment vs. arrival
burstiness/CV, explicitly a robustness/counter-style stress sweep the
authors ran themselves); Table 4 (hybrid-cache ablation across request
rate × burstiness); Table 5 (adaptive-scheduling ablation, same design).
All classified **PAPER_MOTIVATING_STRESS_CASE**.

**Hypothesized adversarial regimes (not directly tested by the authors,
reasoned from the mechanism's own structure):**

- *Uniform, low-variance, low-burstiness requests at low request rate*:
  neither the hybrid-cache mechanism (no memory pressure to relieve) nor
  the adaptive scheduler (FCFS ≈ optimal when arrival order already
  correlates with a good schedule) should offer much advantage —
  directly implied by, but not separately isolated as its own
  experiment in, the paper's own ablation trend ("becomes more prominent
  with higher request rate, burstier request load and longer requests,"
  §6.5) — the logical converse is not itself run as a named experiment.
  **HYPOTHESIZED_ADVERSARIAL_REGIME**.
- *Systematically mis-calibrated `ρ`* (the hidden-cache linear-overhead
  coefficient): if `ρ` is stale or measured on different hardware/model
  than deployed, `t_i^e` (Eq. 6) becomes wrong, which could cause the
  greedy algorithm to over- or under-favor hidden-cache conversion
  relative to the TRUE cost — not evaluated for sensitivity anywhere in
  the paper. **HYPOTHESIZED_ADVERSARIAL_REGIME**.
- *Adversarial popularity/value distribution designed to trigger the
  double-check "extreme case"* the appendix's Algorithm 1 lines 17-25
  exists specifically to bound (a single very-high-value, very-large
  candidate that the naive greedy sort would skip past in favor of many
  small ones) — the appendix proves the 2-approximation bound HOLDS even
  here, so this is really a test of whether an implementation correctly
  includes the double-check step, not a claim Apt-Serve fails here.
  **HYPOTHESIZED_ADVERSARIAL_REGIME** (a correctness/implementation-fidelity
  probe, not an expected-failure regime).

**Scenarios where ordinary KV-only caching should match or outperform
Apt-Serve**: directly evidenced by Table 4's own numbers (page 130:20) —
at CV=1 (low burstiness) and the lower of the two tested request rates
per dataset, "KV Cache" and "Hybrid Cache" SLO attainment are within
0.6-1.6 percentage points of each other in several rows (e.g. ShareGPT,
rate=6, CV=1: 65.7% vs. 67.3%; LongBench, rate=3, CV=1: 70.0% vs. 77.6%
— still a gap, but far smaller than the double-digit gaps seen at higher
CV/rate) — the paper's own ablation data, not an external inference,
shows the hybrid-cache advantage shrinking toward (though not fully
reaching) parity as contention decreases.

## 5. Existing stress-test coverage (task step 6)

Checked `configs/stress_tests/algorithm_stress_test_catalog.yaml`,
`docs/research/algorithm_stress_tests/STRESS_TEST_CATALOG.md`,
`docs/research/algorithm_stress_tests/COVERAGE_MATRIX.md`, and the
latest stress-test audit
(`docs/audits/sarathi_stress_test_catalog_completion_20260805.md`):
zero matches for "apt.serve"/"aptserve" anywhere (`grep -in` returned no
lines, exit code 1). **Classification: MISSING** (not PARTIAL — there is
no target entry, no counter entry, no generator, no stub, no source
citation anywhere in the stress-test library; ALGORITHM_INVENTORY's own
entry 16 explicitly documents this as unidentified/unlocated in the
prior pass, now resolved by this audit's own primary-source research
above). No documentation-only correction is needed or made in this
audit-only pass — see §12 for the specification this audit prepares for
a future implementation task.

## 6. Simulator-compatibility analysis (task step 7)

Cross-checked directly against this project's actual simulator code
(`src/llmserveopt/core/types.py`'s `GPUConfig`,
`src/llmserveopt/simulator/kv_block_manager.py`), not assumed:

| Mechanism | Category | Basis |
|---|---|---|
| Request pending-time-based value (`p_i^e`) | **FULLY_REPRESENTABLE** | `arrival_time`, current sim time already drive `aging_priority`/`scorpio_style_slo_guard`'s own urgency terms |
| TTFT/SLO objectives (`ttft_slo`/`tbt_slo` thresholds, violation-triggered deprioritization) | **FULLY_REPRESENTABLE** | `Request.slo_deadline`, `CompletedRequest.ttft`/`.tpot`/`.slo_violated` already exist project-wide |
| Batch-composition optimization (greedy, memory-budget-constrained, marginal-value-sorted admission) | **FULLY_REPRESENTABLE as a scheduling POLICY** | This is exactly the shape of an admission-ordering `BasePolicy` (same pattern as `scorpio_style_slo_guard`'s credit-budget admission throttling) — the greedy-knapsack RANKING logic itself needs no new simulator infrastructure |
| Single-tier KV memory constraint (`M^e`, `m_i^e`) | **FULLY_REPRESENTABLE** | `GPUConfig.max_kv_tokens`, `KVBlockSpaceManager` already model exactly this |
| **Hybrid dual-tier cache (KV vs. hidden, per-request runtime-switchable type, half-memory-cost tier)** | **NOT_REPRESENTABLE** | `GPUConfig`/`KVBlockSpaceManager` have exactly ONE memory tier, no notion of a second, cheaper cache type at all, no `β_i^e`-style per-request type-assignment field anywhere in `Request`/`ObservableRequest`. This is the paper's OWN headline contribution, and it is the single largest representability gap. |
| Cache-type-switch cost (discard + re-prefill when converting KV↔hidden) | **PARTIALLY_REPRESENTABLE** | The simulator's existing generic preemption/recompute primitive (`Action.preempt`, `GPUState.evict()`, reused by `vllm_faithful`/`sarathi_faithful`) could approximate the "discard and restart" mechanic, but WITHOUT a hidden-cache tier to switch FROM/TO, there is nothing to trigger this specific transition — the primitive exists, the triggering condition does not |
| Hidden-cache read overhead (`O(n)` linear-transform cost vs. `O(1)` direct KV read, the `t_i^e`/`ρ·m_i^e` term) | **NOT_REPRESENTABLE** | `ServiceModel` has no per-request, per-cache-type decode-cost multiplier; all decode steps cost the same regardless of any cache-type concept that does not exist |
| Eviction (implicit, via the greedy cutoff not selecting a candidate) | **PARTIALLY_REPRESENTABLE** | Achievable AS a consequence of implementing the batch-composition policy above (any candidate not selected is implicitly "evicted" for that step) — no new eviction-specific mechanism needed beyond what admission-ordering already provides, but only meaningfully testable once even a *simulated* second memory tier exists to make the eviction decision non-trivial |
| Per-layer / per-token memory behavior (the paper's `x_i^ℓ`/`k_i^ℓ`/`v_i^ℓ` accounting is per-Transformer-layer) | **NOT_REPRESENTABLE** | This simulator's KV accounting (`allocated_kv_capacity_for`) is per-request-token-count only, with no layer dimension at all — a limitation shared by every other baseline in this project (not Apt-Serve-specific) |
| Real GPU kernel effects (the custom `mixed_cache_ops` fused reshape/read/write kernel) | **NOT_REPRESENTABLE** | By design/scope, this simulator never models hardware kernel-level costs for any baseline |

**Required extension for anything beyond the admission-ordering half of
the algorithm**: a genuine second memory-tier dimension in
`GPUConfig`/`KVBlockSpaceManager` (a `hidden_cache_tokens`-style capacity
alongside `max_kv_tokens`, plus a per-request cache-type field feeding
into the memory-accounting arithmetic and a decode-cost multiplier for
hidden-cache-backed requests in `ServiceModel`). This is a **structural
simulator extension**, comparable in scope to the dedicated `hidden
cache tier` concept the paper itself treats as its primary novelty —
not a small addition. **Not implemented in this audit-only pass**, per
explicit instruction.

**Risk of semantic distortion if skipped**: implementing ONLY the
batch-composition/admission-ordering half (fully representable today)
while omitting the hybrid-cache tier would produce a policy that reuses
Apt-Serve's *value function and greedy selection rule* but can never
exercise the *reason those choices differ from a KV-only greedy
scheduler* — i.e., it would silently degrade to "yet another
urgency-weighted greedy admission policy," indistinguishable in
mechanism from this project's own `scorpio_style_slo_guard` in every
scenario that doesn't involve genuine hidden-cache memory pressure. Any
future implementation must disclose this explicitly if the memory-tier
extension is deferred, exactly as `sarathi_faithful.py`'s own docstring
discloses its own scope limits.

## 7. Integration strategy (task step 8)

**Chosen strategy: D — FAITHFUL_INDEPENDENT_REIMPLEMENTATION_PLUS_OFFICIAL_VALIDATION**,
with an honest, evidence-based accounting of why higher-fidelity options
were considered and rejected for now (not dismissed without checking,
per this task's explicit instruction).

**Is the scheduler separable from the engine? More than Sarathi-Serve's,
less than VTC's.** `aptserve_scheduler.py`'s core algorithm
(`greedy_selection_prefill`/`_decode`, `_get_running_urgency`,
`_process_running_queue`/`_process_waiting_queue`, `_dynamic_priority`)
operates on abstract `SequenceGroup`/`Sequence` objects and plain
integers (`n_blocks`, `n_tokens`) — structurally, this is closer to
VTC's pure-Python `VTCReqQueue` than to something requiring real GPU
tensors at decision-time. However, unlike VTC's self-contained
`vtc_req_queue.py`/`req_queue.py` pair, Apt-Serve's scheduler imports
directly from `vllm.core.interfaces`, `vllm.core.policy`, `vllm.sequence`,
`vllm.lora.request`, `vllm.config` — real vLLM package modules — and
depends on a SIBLING replacement file
(`additional_designs/core/aptserve_block_manager.py`, not read in full
this pass) for `self.block_manager.num_total_gpu_blocks`/
`.get_num_free_gpu_blocks()`. A working dynamic-import adapter would need
AT MINIMUM: `aptserve_scheduler.py` + `aptserve_block_manager.py` +
`aptserve_interfaces.py` + enough of `vllm.core`/`vllm.sequence`/
`vllm.config`/`vllm.lora` to satisfy imports (either a real, if
CPU-only, vLLM 0.5.0.post1 install, or a VTC-style synthetic stub
package) — a materially larger adapter surface than VTC's 2-file
precedent.

**Is direct reuse (Strategy A/B) possible?** Not established as
infeasible by assumption — genuinely tested this pass (§10): attempting
to even *resolve* `vllm==0.5.0.post1`'s dependencies on this exact host
(Python 3.12, no prebuilt wheel available for this platform/version
combination) triggers a from-source build path requiring `torch==2.3.0`
as a build-time dependency before any CUDA compilation is even
attempted. Torch 2.3.0 has no published wheel for any CUDA version
newer than what predates Blackwell entirely (confirmed: only
`cu118`/`cu121`-family wheels exist for torch 2.3.0, and Blackwell
`sm_120` support did not exist in any CUDA toolkit torch 2.3.0 was ever
built against). This is the same categorical hardware-generation
blocker already established for VTC (CUDA 11.8) and Sarathi-Serve (CUDA
12.3) on this exact workstation, now confirmed for Apt-Serve's own
(older still) pin. **Strategy A (direct official execution) is not
possible on local hardware**, and B (official code with a thin adapter,
i.e. actually running the modified vLLM engine end-to-end) inherits the
identical blocker, since it requires the same real GPU execution.

**Is a dynamic-import strategy for JUST the scheduler (Strategy C, VTC-style)
possible?** Plausible but **not confirmed** this pass — the probe that
would confirm it (does `vllm.core.scheduler`/`vllm.sequence`/
`vllm.config` import cleanly on CPU-only Python without triggering a
CUDA-touching code path) was not completed, because the PRECONDITION for
even attempting that probe (a working `pip install vllm==0.5.0.post1`)
already failed for the more basic reason above (from-source build
required, `torch==2.3.0` unfetchable-for-purpose here). A future attempt
could try installing `vllm` from a *different*, CPU-compatible
torch/CUDA combination the same way VTC's adapter used a synthetic
package hierarchy — genuinely worth trying before defaulting to D, but
that is real implementation work (Phase A of §13's roadmap), not
something to resolve by further audit-only probing.

**What may be reused unchanged (if Strategy C is later proven feasible)**:
the mathematical CORE — `θ_i` marginal-gain formulas (Eq. 5-9 from the
paper, cross-checked against the actual code), the greedy sort-and-pack
loop (Algorithm 1), and the SLO-fallback near-zero-substitution rule —
all operate on plain Python numbers and would translate directly into
either (a) a dynamically-imported real `aptserve_scheduler.Scheduler`
instance (if Strategy C's import-chain probe succeeds) or (b) a faithful
from-scratch reimplementation reading the same formulas (Strategy D,
the fallback if C fails, exactly analogous to how `sarathi_faithful.py`
was built by reading `SarathiScheduler`'s pinned source directly rather
than importing it).

**What adapter code would be required either way**: a genuine
second-memory-tier extension to this simulator's `GPUConfig`/
`KVBlockSpaceManager`/`ServiceModel` (§6 above) — needed regardless of
whether the scheduling LOGIC comes from a dynamic import or a faithful
reimplementation, since without it there is no hybrid-cache mechanism to
schedule over in the first place. This is the dominant engineering cost
in either integration path, not the scheduler-adapter question.

**Should official-system and simulator tracks be separated?** Yes,
exactly as this project's established pattern for Sarathi-Serve:
(1) an in-repo simulator baseline (once the memory-tier extension
exists) validated against (2) real official-code GPU execution,
performed EXTERNALLY on Wulver given the same CUDA-generation blocker
rules out local real-hardware validation, mirroring the Sarathi
dual-track precedent exactly.

**Why D is the correct preliminary choice, not a shortcut**: given (a)
direct execution is proven infeasible locally, (b) Strategy C's
feasibility is genuinely unresolved (not ruled out, but not free either
— the adapter surface is 3-5x VTC's), and (c) the memory-tier simulator
extension is required in EITHER case and is the larger cost regardless
of scheduler-adapter choice, committing scarce implementation effort to
resolving the Strategy-C import question before the memory-tier
extension exists would not by itself unlock a runnable baseline. D
(reimplement the scheduler faithfully from the paper's formulas + the
verified source, build the memory-tier extension, THEN separately
pursue real Wulver GPU validation of the official system) delivers a
working, testable in-repo baseline sooner, while leaving the door open
to substitute a dynamically-imported real scheduler later if Phase A of
the roadmap confirms Strategy C's feasibility — never described as "the
official baseline" unless and until it demonstrably is one, per this
task's explicit instruction.

## 8. Local workstation vs. Wulver (task step 9)

| Activity | Feasibility | Basis |
|---|---|---|
| Source inspection, dependency-metadata reading | **LOCAL_CPU** | Done extensively this pass, zero blockers |
| `pip install vllm==0.5.0.post1` (base system) | **BLOCKED even for LOCAL_CPU** | No prebuilt wheel for this Python/platform; from-source build requires torch==2.3.0 fetch + a full native build toolchain invocation — not attempted to completion (§10) |
| Unit tests of the scheduler's pure-Python logic (if extracted/reimplemented) | **LOCAL_CPU** | The value-function/greedy-selection math (Eq. 5-9, Algorithm 1) is plain arithmetic — fully testable without any GPU, exactly like `sarathi_faithful.py`'s own test suite |
| Smoke model execution / small official-system benchmark | **WULVER_SINGLE_GPU** (not LOCAL_GPU) | Requires: a CUDA 11.8/12.1-era toolchain matching torch 2.3.0 (this workstation has neither `nvcc` nor a compatible driver-level CUDA version — confirmed CUDA 13.0/Blackwell `sm_120`, same blocker class as VTC/Sarathi), PLUS the custom `mixed_cache_ops` kernel compiled against that same toolchain |
| Hybrid-cache behavior (the paper's actual novel mechanism, requiring the custom CUDA kernel) | **WULVER_SINGLE_GPU minimum** | Needs a real A100 (or compute-capability-compatible) GPU; the paper's own smallest config (OPT-13B, 1×A100, 40GB) is a reasonable minimum target |
| Multi-GPU (30B/66B) full paper-scale experiments | **WULVER_MULTI_GPU** | Table 2: 2×A100 (30B), 4×A100 (66B), NCCL tensor parallelism |
| Full paper reproduction (all figures, all 3 datasets, all 3 model sizes) | **WULVER_MULTI_GPU** + non-trivial wall-clock budget | Not estimated precisely by the paper itself; Table 6's scheduling-only timing (10.8ms/1.6K requests) implies the algorithm itself is cheap, but full GPU inference across 3 datasets × 3 model sizes × multiple request rates × 4 systems (vLLM/Sarathi-Serve/DeepSpeed-FastGen/Apt-Serve) is a substantial multi-GPU campaign, comparable in scale to this project's own Sarathi-vs-vLLM Wulver campaign but larger (4 systems, not 2) |
| Mechanism-level (in-simulator) evaluation, once the memory-tier extension exists | **LOCAL_CPU** | Same as every other in-repo policy in this project — no GPU needed for the simulator itself |

**No Wulver job submitted in this audit** — none of the above required
one to reach a defensible conclusion; §10's local probe already
established the categorical blocker without needing remote compute.

## 9. Optional isolated clone/build probe (task step 10)

**Attempted, partially completed, deliberately stopped before a full
build.** Rather than a literal `git clone` to `~/.cache/external_baselines/`
(the repository's small size and GitHub API accessibility made direct
`gh api` content reads equivalent to a clone for this audit's read-only
inspection needs — every source file examined in §3 was fetched this
way, pinned to whatever `main` currently is, which per §1 is
functionally identical to the paper-era code), the actual environment
probe performed was:

```bash
python3 -m venv /tmp/apt_serve_probe
/tmp/apt_serve_probe/bin/pip install --dry-run "vllm==0.5.0.post1"
```

**Result**: pip could not resolve a prebuilt wheel for this
Python 3.12/platform combination and began an isolated-build-environment
install of vLLM's own build-time dependencies (`cmake>=3.21 ninja
packaging setuptools>=49.4.0 torch==2.3.0 wheel`) — i.e., it started
attempting to fetch `torch==2.3.0` (a multi-hundred-MB-to-multi-GB
package) purely as a BUILD dependency, before any actual CUDA
compilation would even begin. This process was **deliberately killed**
after confirming this behavior (rather than let it complete a large,
slow download for an audit-only probe with an already-known
hardware-incompatible destination), and the temporary venv removed.
This is not a build FAILURE finding — it is a **cost/necessity**
finding: even the dependency-resolution step alone is expensive on this
host for this exact package, independent of the GPU-compatibility
question already established in §7-8. No CUDA compilation was attempted
(would fail for the same categorical reason as VTC/Sarathi, per §7 — not
re-derived by an actual failed build here, but not asserted without
basis either: the torch-2.3.0-CUDA-wheel-availability check in §7 is
direct evidence, not a guess).

**Recorded**: clone path — N/A (GitHub API reads used instead, all
commands shown inline in §1-4); commit inspected — `main` tip,
`c953217988` (2025-06-09); commands — as shown above; success/failure —
dependency resolution not completed, stopped by design; blockers — no
platform-matching prebuilt wheel, from-source build path requires a
large `torch==2.3.0` fetch as a build dependency alone, separate from
and prior to the already-established CUDA/Blackwell incompatibility;
missing assets — none (all needed source/paper/appendix material was
successfully obtained via `gh api`/arXiv).

## 10. Apt-Serve evaluation plan (task step 11) — design only, not run

**Track 1 — Official real-system evaluation** (requires Wulver, per §8):
- Policies: vLLM (baseline), Sarathi-Serve (already faithfully modeled
  in this repo — real comparison point exists), DeepSpeed-FastGen (not
  currently in this repo), Apt-Serve (official code)
- Models: OPT-13B minimum (matches paper's smallest config and this
  project's existing `mistralai/Mistral-7B-Instruct-v0.1` Wulver
  precedent scale); OPT-30B/66B only if the 13B result is promising
  enough to justify the larger multi-GPU budget
- Workloads: ShareGPT (primary — largest documented Apt-Serve advantage,
  §6.3), HumanEval (secondary — smallest documented advantage, a
  natural counter-regime probe), LongBench (tertiary)
- Cache capacities: match paper's own GPU-memory-utilization convention
  (consistent cache-storage budget across all compared systems, per the
  paper's own "fair comparison" methodology, §6.2)
- Reuse levels / SLO settings: reuse Table 3's exact SLOs (TTFT/P99 TBT
  per dataset/model-size) rather than inventing new ones
- Metrics: SLO attainment rate (%) at varying request rates (the paper's
  own primary metric, Figure 8-style curves), plus this project's
  standard TTFT/TPOT/throughput/ANWG/completion-fraction battery for
  comparability with existing baselines
- Multiple seeds: N=5 repeated trials per scenario, matching the
  Sarathi-vs-vLLM Wulver precedent's own statistical methodology
  (paired bootstrap, ROBUST/SUGGESTIVE/NOT_REPRODUCED classification)
- Raw output requirements: full per-request JSONL + summary JSON/CSV,
  committed selectively (small structured artifacts only), matching this
  project's established `results/`/`experiments/` conventions
- Independent verification: a second, independently-run trial set before
  any claim is finalized, matching this project's own repeated-trial
  precedent

**Track 2 — Simulator/mechanism-level evaluation** (requires the
memory-tier extension from §6, not yet built):
- Compare a faithful `apt_serve_faithful`-style policy (once it exists)
  against `sarathi_faithful`, `vllm_chunked_prefill_faithful`,
  `vllm_faithful`, `scorpio_style_slo_guard` — the last is the closest
  existing internal analog (urgency+admission-credit-based), making it
  the most informative mechanism-level comparison point
- Metrics: this simulator's standard battery plus, if the memory-tier
  extension is built, cache-type distribution (fraction of requests
  KV-cached vs. hidden-cached over time) and cache-type-switch count as
  Apt-Serve-specific diagnostics

**Track 3 — Stress-test evaluation** (this project's Algorithm
Stress-Test Library, once a generator/policy exists): see §12 below.

**Do not run any of this now** — design only, per explicit instruction.

## 11. Target and counter workloads (task step 12) — specification only

| Workload | Role | Evidence class | Simulator compatibility | Real-system requirement |
|---|---|---|---|---|
| High-request-rate, long-output, ShareGPT-shaped (mirrors paper Figure 8a's own strongest result) | TARGET | PAPER_MOTIVATING_STRESS_CASE | Admission-ordering half only (memory-tier extension needed for full fidelity) | Wulver, for real validation |
| Bursty arrivals (high CV) at fixed mean rate, mirroring Figure 9/Table 4/5's own robustness sweep | TARGET | PAPER_MOTIVATING_STRESS_CASE | Same as above | Wulver |
| Mixed KV/hidden-reuse-opportunity workload (heterogeneous prompt/output lengths forcing genuine cache-type-mix decisions) | TARGET | PAPER_MOTIVATING_STRESS_CASE (generalized from §6.3's ShareGPT/LongBench discussion) | **NOT executable without the memory-tier extension** — this is precisely the mechanism that extension is for | Wulver |
| Cache-constrained batch selection (tight memory budget forcing genuine greedy-knapsack tradeoffs) | TARGET | DOCUMENTED_LIMITATION (directly, Figure 2a/2b) | Admission-ordering half only | Wulver |
| TTFT-sensitive heterogeneous request mix (short + long prompts contending for prefill priority) | TARGET | PAPER_MOTIVATING_STRESS_CASE (Figure 4a-4c) | Fully representable today (no memory-tier dependency — pure scheduling-order test) | None (executable in-simulator now, once a policy exists) |
| Low request rate, low burstiness, homogeneous lengths (HumanEval-shaped) | COUNTER | DOCUMENTED_LIMITATION (§6.3's own comparative discussion) | Admission-ordering half only | Wulver |
| Cache thrashing (workload engineered to force repeated KV↔hidden switches, each costing a re-prefill) | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME (reasoned from §5's discard-and-recompute mechanic, not directly tested by the authors) | **NOT executable without the memory-tier extension** | Wulver |
| Reuse outside the cache horizon (requests whose reusable state has already been evicted/discarded by the time it would help) | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME | **NOT executable without the memory-tier extension** | Wulver |
| Uniform, short-lifetime requests (adds little — hidden cache's benefit requires enough lifetime to amortize) | COUNTER | DOCUMENTED_LIMITATION (§6.3 HumanEval discussion) | Admission-ordering half only | Wulver |
| Oversized cache entries under small capacity (single very-large request dominating the memory budget, testing the double-check "extreme case" the appendix's Algorithm 1 specifically guards against) | COUNTER | PROVEN_WORST_CASE-adjacent (tests whether an implementation correctly reproduces the appendix's own worst-case-bounding step) | Admission-ordering half only | None — this is testable in-simulator once ANY faithful reimplementation of Algorithm 1 exists, independent of the memory-tier extension |
| Adversarial popularity distribution (many low-value, cheap candidates crowding out one high-value, expensive one) | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME | Admission-ordering half only | None |
| Incorrect value estimates (systematically stale/wrong `ρ` coefficient) | COUNTER | HYPOTHESIZED_ADVERSARIAL_REGIME (§4's own explicit gap — not evaluated for sensitivity in the paper) | Admission-ordering half only, IF the memory-tier extension also exists (this specifically probes the `t_i^e` term, which requires the hidden-cache concept to be meaningful at all) | Wulver, for a real-world-calibration-error scenario |
| SLO/value conflict (the ~10% starvation regime, Figure 10) | COUNTER | DOCUMENTED_LIMITATION (directly, Figure 10) | Fully representable today (pure scheduling-priority test, no memory-tier dependency) | None — executable in-simulator now |

**Not generated in this audit-only pass** — several rows above ARE
already executable with today's simulator (the pure scheduling-order/
SLO-fallback ones), but generating even a tiny smoke workload for them
would require a policy implementation that does not yet exist; per this
task's own scope ("do not generate these workloads yet unless an
existing generator already supports them without modification") none of
this project's existing generators produce Apt-Serve-shaped requests, so
none are generated here.

## 12. Implementation roadmap (task step 13) — estimate only, not built

**PHASE A — official artifact setup** (~2-4 days if pursuing Strategy-C
feasibility; skippable in favor of going straight to reimplementation):
resolve the from-source vLLM 0.5.0.post1 build question properly (either
find a working prebuilt-wheel combination, or accept the from-source
cost once, in an environment where it's actually useful — likely Wulver,
not local); attempt the `vllm.core.scheduler`/`vllm.sequence` CPU-only
import probe this audit could not complete; STOP CONDITION: if the
import chain requires real CUDA initialization at module-import time
(not just at `Worker`/`LLMEngine` construction), Strategy C is
infeasible and Phase A concludes "proceed directly to D."

**PHASE B — fidelity proof** (~1-2 days): whichever path A concludes,
produce a `docs/apt_serve_faithful_scheduler_reference.md`-style
document (mirroring `sarathi_faithful_scheduler_reference.md`'s own
precedent) with exact source citations for every formula (already
substantially drafted by this audit's §3-4, reusable directly).

**PHASE C — simulator representation** (~3-5 days, the dominant cost):
build the `GPUConfig`/`KVBlockSpaceManager`/`ServiceModel` memory-tier
extension identified in §6 — a genuinely new, shared piece of
infrastructure (not a policy-only change), requiring its own dedicated
test suite before any Apt-Serve-specific policy is layered on top;
implement `apt_serve_faithful.py` (or a Strategy-C adapter) reusing that
extension; expected LOC: 400-700 for the memory-tier extension itself
(comparable in scope to the original `KVBlockSpaceManager`), 300-500 for
the scheduling policy (comparable to `sarathi_faithful.py`'s own size).

**PHASE D — stress-test generation** (~1-2 days, once C exists): add the
§11 target/counter entries to the Algorithm Stress-Test Library catalog,
following the exact provenance/evidence-class discipline established for
Sarathi-Serve's 7-entry addition — expected 8-10 entries given §11's
table.

**PHASE E — official-system evaluation** (Wulver, timeline dependent on
queue/account access — not estimable from this audit alone; scope
comparable to or larger than the Sarathi-vs-vLLM Wulver campaign given 4
systems being compared instead of 2).

**PHASE F — final foundational/evaluation-only decision**: not
reachable without E's real-hardware evidence, per this project's own
established practice (Sarathi/VTC/PARS/vLLM-LTR precedent — no baseline
in this project has been registered foundational without either
real-hardware validation or a fully-verified official-code reproduction
first).

**Expected LOC total**: ~1,200-2,000 across B-D. **Expected GPU
requirement**: none for B-D, Wulver A100 (minimum) for A (if pursued)
and E. **Main engineering risk**: the memory-tier simulator extension
(Phase C) touches shared infrastructure (`GPUConfig`, used by every
policy/baseline in this project) — must be strictly additive/opt-in,
exactly matching the precedent set by `enable_prefill_modeling`/
`enable_decode_prefill_contention`'s own careful backward-compatible
rollout (see `docs/decode_prefill_contention_execution_model.md`), not
a breaking change. **Main scientific risk**: without Phase A/C actually
existing, no claim about Apt-Serve's real advantage can be tested at
all in this project yet — this audit establishes WHAT to build and WHY,
not that it will reproduce the paper's 8.8× headline number, which
remains unverified by this project until E.

## 13. `docs/BASELINE_STATUS.md` — not updated

The existing Apt-Serve row ("Not implemented... Not prioritized" across
every column) is **not stale or incorrect** — it accurately reflects
that Apt-Serve remains unimplemented, which this audit does not change
(no implementation was performed, per explicit task scope). No update
made this pass; a future implementation task should update it once
Phase B-D of §12 produces real artifacts.

## 14. Validation (task step 15)

Only new documentation files were added this pass (this audit doc); no
`src/`, `scripts/`, `configs/`, or `tests/` files were modified.
Per task step 15's own instruction, the compileall/status-checker/
resume-readiness/collect-only battery is run below for completeness
(cheap, and confirms zero incidental disruption), though strictly
unnecessary given no Python-adjacent files changed. No Apt-Serve-adjacent
tests exist yet (confirmed §5) — none run. No full benchmark, no full
non-live suite run (neither needed nor practical for a documentation-only
change).
