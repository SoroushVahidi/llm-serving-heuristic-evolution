# Pinned TetriInfer Reference — Primary-Source Provenance and Reproducibility Determination

This document records the exact primary source used for the
`tetriinfer_paper_reimplementation` policy, the reproducibility
determination that governs its implementation label, and the boundary
between (A) behavior directly specified by the primary source, (B)
behavior this project must adapt or invent because the primary source
does not specify it precisely enough, (C) existing infrastructure reused
unchanged, (D) new shared infrastructure added for this baseline, and (E)
differences that cannot be faithfully represented.

**Read this document before touching `src/llmserveopt/policies/tetriinfer_paper_reimplementation.py`.**

## 0. Reproducibility determination (read this first)

Unlike `vllm_faithful`, `sarathi_faithful`, and `distserve_faithful` — each
pinned to a specific commit of a real, author-maintained code repository
that could be read line-by-line — **no official TetriInfer code repository
or artifact exists** (confirmed below). The only primary source is the
arXiv preprint itself, which describes the scheduling algorithm in prose
and figures, not pseudocode or source code.

**Verification performed** (2026-07-18, live queries, not from memory):
- `gh api users/lastweek/repos` (Yizhou Shan, the corresponding/most
  GitHub-active author) — no TetriInfer repository among ~140 public repos.
- `gh search code "TetriInfer"` across all public GitHub — every hit is a
  third-party citation, course note, or survey reference; zero hits are an
  implementation of the system itself.
- Web search for "TetriInfer github", "TetriInfer official implementation",
  and author-name + "TetriInfer" + "github" — no repository found.
- Semantic Scholar API (`paperId 21e53e51...`) — `publicationVenue: arXiv.org`,
  `year: 2024`, DBLP id `journals/corr/abs-2401-11181` (DBLP's "CoRR"
  bucket for arXiv-only preprints, not a conference/journal proceeding
  record) — confirms no peer-reviewed venue publication as of this audit.
- arXiv submission history: exactly one version (`v1`, 2024-01-20); no
  later revision exists that might have added a code/artifact link.

**Conclusion:** TetriInfer has **no official repository, no
author-maintained implementation, and no peer-reviewed venue** as of this
audit. It is **not eligible for the `_faithful` label** as this project
uses that term (verified against real, pinned scheduler source code). The
paper's prose description is nonetheless considerably more precise than a
generic "inspired by" summary for several components (exact thresholds,
an explicit three-step routing algorithm, explicit reservation-policy
definitions) — precise enough to reimplement those specific decisions
faithfully to the *paper's stated description*, while other components
(the ML length predictor, and the exact "least interference" scoring
function) are not specified precisely enough to reproduce bit-for-bit and
require this project's own, disclosed operationalization.

**Chosen label: `tetriinfer_paper_reimplementation`** — not `_faithful`.
Every docstring, doc section, and safe/unsafe claim in this baseline uses
this label deliberately; see §E for the itemized list of exactly which
decisions are paper-specified vs. this-project-invented.

## 1. Paper

- **Title:** "Inference without Interference: Disaggregate LLM Inference
  for Mixed Downstream Workloads"
- **Authors:** Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang
  Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui
  Sun, Yizhou Shan
- **Affiliations:** University of Chinese Academy of Sciences / ICT, CAS
  (Hu, Huang, Wang, Wang, Bao, Sun); Huawei Cloud (Xu, Chen, Xu, Chen,
  Feng, Shan). First author's work "done while intern at Huawei Cloud."
- **Venue:** arXiv preprint only (cs.DC). **No peer-reviewed conference or
  journal publication found** as of this audit (see verification above).
  Do not describe this as an "OSDI"/"SOSP"/etc. paper in any downstream
  doc — unlike `vllm_faithful`'s SOSP 2023, `sarathi_faithful`'s OSDI
  2024, and `distserve_faithful`'s OSDI 2024 pins.
- **arXiv:** [2401.11181](https://arxiv.org/abs/2401.11181), version v1
  (only version), submitted 2024-01-20.
- **License:** CC BY 4.0 (permits reuse/adaptation with attribution).
- **PDF:** https://arxiv.org/pdf/2401.11181

## 2. Official repository / artifact

**None exists.** See the verification steps in §0. This section exists
(rather than being omitted) specifically so a future reader does not
re-search for one under the mistaken assumption that this project simply
failed to find it — the absence was verified, not assumed.

## 3. What the paper specifies precisely enough to reimplement (§A) vs. what it does not (§E)

### A. Behavior specified precisely enough to reimplement

**System architecture (§3.1–3.2 of the paper):**
- A centralized control plane with two components: a **global scheduler**
  (assigns each new request to a prefill instance; maintains a request
  status table) and a **cluster monitor** (collects load stats from every
  instance every ~100ms, broadcasts them to prefill instances, and
  manages instance add/remove/flip).
- **Global scheduler policy:** "choose a prefill instance with the least
  load." No tie-breaking rule is specified in prose (see §E).
- Prefill and decode instances are **virtual roles**, not fixed hardware —
  an instance can "flip" between roles (paper §3.5); the paper's own flip
  trigger is "idle for a minute." This is a secondary, evaluation-specific
  policy — see §E for why it is excluded from this implementation.

**Prefill instance / local prefill scheduler (§3.3.1):**
- Maintains a raw-request queue and a scheduled queue.
- Three explicit, named policies: **FCFS**, **SJF** (shortest-job-first),
  **LJF** (longest-job-first) — "we can accurately estimate a request's
  prefill time based on the number of tokens in its prompt," so SJF/LJF
  sort by prompt token count. All three are **non-preemptive**
  ("We only explore non-preemptive policies").
- `PrefillSchedBatch`: a batch-size knob for how many *sorted* requests are
  considered per scheduling round (evaluation default 16, tested up to
  128) — analogous to this project's own `context_max_batch_size`.

**Chunked prefill (§3.3.3):**
- Fixed-size chunks; **`ChunkSize` is explicitly stated to be
  hardware/model-dependent**, not universal: "The accelerator and the LLM
  model architecture determine the ChunkSize... in our test environment,
  the value is 512 tokens for OPT-13B." This is evaluation-environment
  evidence for *a* value, not a universal paper default the way Sarathi's
  512 was (Sarathi's evaluation scripts used 512 as their own shipped
  configuration default across the board; TetriInfer's 512 is explicitly
  presented as *derived from* a specific accelerator+model pairing this
  project has no equivalent of). Treated as a documented, disclosed
  starting default here, not an assumed universal constant (see §4/task
  step 8).
- Explicitly contrasted with Sarathi: TetriInfer's chunks are
  **prefill-only** (never mixed with decode tokens in the same chunk),
  unlike Sarathi-Serve's stall-free prefill-decode-mixed chunks.

**Length predictor, architecture and training procedure (§3.3.2):**
- A separate, small model (OPT-125M) is fine-tuned as a **sequence
  classifier** to predict which fixed-size **bucket** (granularity `g`)
  the target model's output length will fall into, given the prompt.
  Buckets: `[0, g)` → label 0, `[g, 2g)` → label 1, etc. Trained on 75K
  ShareGPT prompts, using the *target* model's own generated responses as
  ground truth to derive bucket labels.
- Granularity is a tunable tradeoff (paper tests 100/200/400); **granularity
  200 achieves 74.9% accuracy** in the paper's own setup.
- "It's easy to calculate resource usage's upper and lower bound" from a
  predicted bucket — i.e., predicting a bucket directly yields a
  `(lower_bound, upper_bound)` token-count range, not a point estimate.

**Inter-instance decode routing / dispatcher (§3.3.4) — the most precisely
specified algorithm in the paper:**
1. Categorize all decode instances into set **α** (enough resources to
   run this request's decode phase, given its predicted length range and
   each instance's broadcast load) and set **β** (not enough).
2. **Power-of-two choice:** randomly select **two** instances from α.
3. From those two, pick the one that "would encounter the least
   interference" — operationalized by the paper's own stated goal:
   **minimize the (heavy-decode : light-decode) ratio** on the receiving
   instance, i.e. prefer sending a request to whichever of the two
   candidates would end up with a more balanced heavy/light mix (paper's
   own evaluation objective, §5.2.3: "establish the lowest average ratio
   of heavy decode:light decode... spread heavy decode requests evenly").
- Runs **per-request, decentralized, at the prefill instance** (not at the
  global scheduler) once a request's first chunk is prefilled.
- Alternative designs (predictor at each decode instance; predictor at the
  global scheduler) are explicitly discussed and rejected by the paper
  authors themselves — useful negative evidence that the implemented
  design (predictor at the prefill instance, feeding the dispatcher) is
  the one to reproduce.

**Local decode scheduler / capacity management (§3.3.5 area / Fig. 18 discussion):**
- Baseline: vLLM's own greedy policy (admit as long as spare memory
  exists this iteration; oblivious to future working-set growth).
- **Reserve-static:** admit a new request only if its predicted memory
  usage is smaller than *currently* available accelerator memory.
- **Reserve-dynamic:** admit a new request only if there will still be
  spare memory at the future point in time when the *shortest remaining
  job currently in the batch* finishes (a proactive, future-aware
  admission check using each in-batch request's predicted remaining
  length).
- Both explicitly aim to **avoid ever triggering swap/thrashing** — this
  is the opposite of DistServe's model, which uses swap as an active
  capacity-management mechanism. TetriInfer's decode-side story is
  admission-time avoidance, not runtime eviction.
- **Resource estimates use the *lower end* of the predicted bucket range**
  (paper §5.2.3: "Our policies estimate resource usage using the
  predicted length range's lower end") — a specific, quotable, easy-to-get-
  wrong detail: NOT the midpoint, NOT the upper bound.

**KV transfer:** request-level (whole-request), not chunk-level — "we only
implement request-level transfer for simplicity" (§3.4 area). This maps
directly onto this project's existing bridge-queue/`transfer_ready_time`
mechanism (one handoff per request, once fully prefilled) with **no new
shared infrastructure required** for the transfer mechanism itself.

**Heavy/light classification (§5, evaluation setup — used consistently
across the interference study and the dispatcher's own stated objective):**
- Heavy prefill: prompt tokens > 512. Light prefill: ≤ 512.
- Heavy decode: generated tokens > 128 (paper: "ShareGPT answers' median
  length is 128"). Light decode: ≤ 128.

### E. What the paper does NOT specify precisely enough — disclosed adaptations

These are the places this implementation must make its own documented
choice rather than reproduce the paper, because the paper does not give
enough detail (or the detail is fundamentally non-reproducible, e.g. an ML
model's learned weights):

1. **The length predictor itself is an ML model** (fine-tuned OPT-125M),
   not an algorithm. Its 74.9%-accuracy figure is an empirical result tied
   to the authors' specific model/dataset/training run — it cannot be
   reproduced, only *approximated behaviorally*. This project implements
   a **pluggable prediction abstraction** (exact / bucketed / configurable
   Gaussian-noise-then-bucketed modes) built on this project's own
   existing `predicted_output_tokens` field, explicitly NOT a trained
   classifier — see §D and the module docstring of
   `length_prediction.py`. The bucket-boundary MATH (labeling a
   `[0,g), [g,2g), ...` range) is paper-specified and reproduced exactly;
   the classifier that assigns a request to a bucket is not.
2. **"Least interference" is a stated objective, not a closed-form
   scoring function.** The paper's own evaluation section states the goal
   as minimizing the heavy:light decode ratio, so this project uses that
   stated goal directly as the tie-break scoring function between the two
   power-of-two candidates — a direct, disclosed operationalization of the
   paper's own words, not an invention of unrelated criteria.
3. **No tie-breaking rule is given** for the global scheduler's
   least-loaded prefill-instance choice, nor for the dispatcher's
   power-of-two random sampling. This project uses this codebase's
   existing `arrival_then_id`-style deterministic tie-breaking
   (`gpu_id` ascending) for the former, and a seeded PRNG (explicit,
   constructor-configurable seed) for the latter, so runs remain
   deterministic and reproducible — consistent with every other policy in
   this repository.
4. **`ChunkSize=512` and `PrefillSchedBatch` are the paper's own
   environment-specific measurements**, not universal defaults the way
   Sarathi's 512 was — exposed as explicit, disclosed config parameters
   (see §4/task step 8), never silently hardcoded as "the paper's value."
5. **Instance flip** (idle-for-a-minute role switching) is an evaluation-
   scale elasticity mechanism, not a scheduling-decision algorithm this
   project's fixed-topology simulator can meaningfully exercise (this
   simulator does not model instance provisioning/deprovisioning at all,
   for any baseline) — excluded from this implementation, same "core
   algorithm, not every ablation" reasoning already applied to
   `vllm_faithful`/`sarathi_faithful`/`distserve_faithful`'s own excluded
   ablations.
6. **Network stack / KV-transfer bandwidth modeling** (IB Verbs, C++
   shared-memory bridge, mock bandwidth emulation) is real-system plumbing
   with no algorithmic scheduling content — out of scope, same reasoning
   as DistServe's excluded bandwidth-aware transfer-cost model.
7. **The global scheduler's "least load" metric** is not defined exactly
   (queue depth? active-sequence count? KV occupancy?) — this project
   uses this codebase's existing `ObservableGPUState` occupancy signal
   (active sequence count, consistent with how "load" is already used
   throughout this codebase's other policies) rather than inventing a new
   metric.

## 4. New shared infrastructure required

TetriInfer is **the first baseline in this project requiring genuine
multi-instance decode-side routing** — every prior disaggregated baseline
(`distserve_faithful`) enforces exactly one decode-role GPU. This requires:

- Support for **multiple `role="decode"` GPUs** simultaneously (the
  existing `GPUConfig.role`/bridge-queue/`Action.admit` infrastructure
  already supports this — `distserve_faithful`'s single-decode-worker
  requirement was a *policy-level* validation choice, not a simulator
  limitation, so no simulator-level change is needed here).
- A **length-prediction abstraction** (new:
  `src/llmserveopt/policies/tetriinfer_length_prediction.py` — see §E.1).
- A **seeded, deterministic power-of-two routing helper** (new:
  `src/llmserveopt/policies/tetriinfer_routing.py`; policy-internal, not
  simulator infrastructure, since routing is a scheduling *decision*, not
  a mechanism).

Reused unchanged: `GPUConfig.role`, the bridge queue
(`Simulator._migrating`/`_migrating_map`), `RequestPhase.MIGRATING`,
`transfer_ready_time`, `ServiceModel.enable_disaggregation`/
`migration_transfer_delay`, `KVBlockSpaceManager` (for local decode-side
KV accounting), `ObservableRequest.predicted_output_tokens` (the existing
field this project already threads through for non-oracle policies). No
DistServe-specific infrastructure (swap, `Action.swap`) is reused, since
TetriInfer's decode-side story is admission-time avoidance, not runtime
eviction — see §A.

### Implementation note: admission GATE vs. actual block allocation are separate

The paper describes reserve-static/reserve-dynamic purely as admission
conditions ("a request is scheduled only if its predicted memory usage is
smaller than the available accelerator memory") and does not specify the
underlying block-allocation mechanism. An early draft of this policy
allocated the FULL predicted footprint upfront at admission time, which
produced two real bugs during implementation (caught via targeted tests
and a randomized multi-config stress harness, not by inspection):

1. It crashed (`KVBlockManagerError`) whenever reserve-dynamic's admission
   decision relied on a *future* projection (blocks not yet actually free)
   — the code then immediately tried to physically allocate the full
   projected amount, which does not exist yet.
2. It double-counted a decode-side request's footprint inconsistently
   between the dispatcher's routing-eligibility check and the local
   scheduler's own admission check when the two used different token
   quantities (predicted output alone vs. prompt + predicted output).

The corrected design (`_decode_admission_check` in
`tetriinfer_paper_reimplementation.py`) treats reserve-static/
reserve-dynamic as **admission gates only**: they decide whether to admit
a candidate at all, using the full predicted sequence footprint
(prompt + predicted output + 1) for the *decision*, but the actual blocks
reserved at admission are always just `prompt_tokens + 1` (the transferred
prompt KV plus one decode token) — growing incrementally by one token per
step afterward via `_grow_active_decode_requests`, identical to
`vllm_faithful`/`distserve_faithful`'s own `append_slot` mechanism. This
matches how vLLM's real paged-attention actually works (TetriInfer is
explicitly built on top of vLLM per the paper's Implementation section)
and is this project's own disclosed adaptation where the paper is silent
on the underlying mechanism. The admission gate additionally always
requires that the small incremental amount can be allocated immediately
(never just the future-projected amount), so an admission decision never
attempts to allocate blocks that do not yet exist.

## 5. Do not use secondary summaries as a substitute for this document

Blog posts, survey papers, and course notes describing TetriInfer
(several were found during this audit, e.g. citing it alongside Splitwise
and DistServe) were used only to *locate* the primary source faster, never
as a substitute for reading the arXiv HTML/PDF directly. Every claim in
§3/§A/§E above is anchored to a direct quote or figure/section reference
from `https://arxiv.org/html/2401.11181v1`, fetched live during this audit
(2026-07-18), not recalled from training data.
