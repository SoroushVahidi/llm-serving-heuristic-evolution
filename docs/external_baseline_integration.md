# External Baseline Integration: Matrix, Resource-Normalization Protocols, and Selector-Eligibility Analysis

This document covers the unified integration and validation phase for the
five faithful/paper-reimplementation external baselines built in prior
PRs: `vllm_faithful`, `sarathi_faithful`, `distserve_faithful`,
`tetriinfer_paper_reimplementation`, `llumnix_faithful`. It does **not**
introduce a sixth baseline or retrain the selector — it audits the five
together, builds the minimum infrastructure to evaluate them fairly, and
makes an explicit, justified recommendation for how (and whether) they
belong in a future selector-training phase.

**Read this document before running any cross-baseline comparison, and
before deciding to register any of these five as a selector candidate.**

## 0. What this phase deliberately does NOT do

- Does not add any of the five to `registry.py`'s `BASELINE_NAMES` /
  `SELECTOR_CANDIDATE_NAMES` (still 20/20, unchanged — see
  `tests/test_external_baseline_integration.py::test_historical_baseline_count_unchanged`).
  A separate registry (`policies/external_baselines_registry.py`) exists
  specifically so this new metadata never touches the historical one.
- Does not train or retrain the selector.
- Does not run large-scale/manuscript-scale experiments — only smoke-scale
  validation (§8) to catch crashes, topology violations, and accounting
  bugs before any such experiment is designed.
- Does not force all five behind one abstraction that changes any
  baseline's own algorithm — each policy's `select_action` runs completely
  unmodified through the harness; only its *return value* is inspected.

## 1. Integration matrix

| | `vllm_faithful` | `sarathi_faithful` | `distserve_faithful` | `tetriinfer_paper_reimplementation` | `llumnix_faithful` |
|---|---|---|---|---|---|
| **Fidelity** | faithful (pinned commit) | faithful (pinned commit) | faithful (pinned commit) | **paper reimplementation** (no official code exists) | faithful (pinned OSDI'24 artifact commit) |
| **Pinned source** | vLLM `67d96c29` (v0.1.0) | sarathi-serve `ceaa0660` | DistServe `0ec355c8` | arXiv:2401.11181 (no repo) | `alibaba/llm-scheduling-artifact` `a908243` |
| **Topology class** | monolithic | monolithic | disaggregated prefill/decode | disaggregated prefill/decode | multi-instance migratory |
| **Required GPU roles** | `role=None` only | `role=None` only | `role="prefill"` + `role="decode"` | `role="prefill"` + `role="decode"` | `role=None` only |
| **Min GPU count** | 1 (any N, shared queue) | 1 (any N, shared queue) | **exactly 2** (1+1, hard) | 2 minimum (1 prefill + ≥1 decode) | 1 (code minimum; migration needs ≥2 to do anything) |
| **KV block model** | yes (own `KVBlockSpaceManager`) | yes (reuses vLLM's) | yes (reuses vLLM's) | yes (reuses vLLM's) | yes (via composed `vllm_faithful`) |
| **Disaggregation** | no | no | yes (core mechanism) | yes (core mechanism) | no |
| **Cross-instance migration** | no | no | no (swap ≠ migration — same instance) | no | **yes (core mechanism)** |
| **Preemption mode** | recompute | recompute (reuses vLLM's) | swap | admission-avoidance (reserve-static/dynamic; never evicts) | recompute (via composed `vllm_faithful`) |
| **Chunked-prefill scheduling** | no | yes (core mechanism) | no | yes (FCFS/SJF/LJF) | no (inherited from composed `vllm_faithful`) |
| **Length prediction** | no | no | no | yes (own deterministic, non-ML abstraction) | no |
| **Supported workload assumptions** | any `Request` trace; no session/priority-type semantics | same | same | same; heavy/light thresholds are paper-sourced, not enforced | same; no session/multi-turn concept (dispatch stickiness degenerates to round-robin) |
| **Configurable parameters** | `block_size`, `max_num_batched_tokens`, `max_num_seqs`, `watermark` | `chunk_size`, `block_size`, `watermark` | `context_max_batch_size`, `context_max_tokens_per_batch`, `decode_max_batch_size`, `decode_max_tokens_per_batch`, `block_size` | `prefill_local_policy`, `context_max_batch_size`, `context_max_tokens_per_batch`, `decode_local_policy`, `decode_max_batch_size`, `predictor_granularity/mode/noise_std_tokens/seed`, `routing_seed`, `block_size` | `need_migrate_frequency`, `migrate_out_threshold`, `priority_exempt_threshold`, `block_size`, `max_num_batched_tokens`, `max_num_seqs`, `watermark` |
| **Registration status** | not registered (deployable or selector) | not registered | not registered | not registered | not registered |

(Full per-field detail, including exact enum values, lives in
`policies/external_baselines_registry.py`'s `ExternalBaselineSpec`
instances — this table is a human-readable summary of that machine-readable
source of truth; if the two ever disagree, the registry module is
authoritative.)

### Which baselines can be compared under identical physical topology?

**Not all pairs, and "can be co-located" is not the same as "should be
compared."**

- **`vllm_faithful` ↔ `sarathi_faithful`**: YES, directly. Both assume the
  identical resource-sharing model (N `role=None` GPUs, ONE shared global
  admission queue visible to the policy). Same GPU count ⇒ same physical
  topology ⇒ a genuinely apples-to-apples scheduling-policy comparison.
- **`distserve_faithful` ↔ `tetriinfer_paper_reimplementation`**: PARTIALLY.
  Both are disaggregated prefill/decode, so they CAN be co-located at
  `distserve_faithful`'s forced 1-prefill+1-decode minimum. But
  `tetriinfer_paper_reimplementation`'s actual point of interest (power-of-two
  routing across *multiple* decode workers) has **no `distserve_faithful`
  analogue at all** — `distserve_faithful` cannot structurally scale past
  1 decode worker (hard `ValueError` otherwise). A comparison at N>2 decode
  GPUs is not "the same topology" for both; it silently drops
  `distserve_faithful` from the comparison or forces it into an
  under-resourced corner it was never designed for.
- **`llumnix_faithful` ↔ `vllm_faithful`/`sarathi_faithful` at the same GPU
  count**: **NO, not directly**, despite all three using only `role=None`
  GPUs. `vllm_faithful`/`sarathi_faithful` at N GPUs model ONE shared-queue
  pool (any GPU may serve any request, policy has global visibility every
  step); `llumnix_faithful` at the same N GPUsmodels N **independent**
  admission queues connected only by an explicit, delayed, capacity-gated
  migration primitive. These are different points in the design space by
  construction — comparing them at "the same GPU count" answers "does a
  shared-queue pool or an independent-instances-plus-migration pool serve
  this workload better," which is a real and interesting question, but it
  is an **architecture** question, not a scheduling-**policy** question,
  and must be reported as such (see §10).
- **`llumnix_faithful` ↔ `distserve_faithful`/`tetriinfer_paper_reimplementation`**:
  NO — disjoint role requirements (`role=None` vs. `role="prefill"`/`"decode"`)
  make them structurally impossible to co-locate on identical GPU
  configs at all (not even a resource-normalization question — the harness's
  own `validate_topology` rejects any attempt).

The task's own warning bears repeating here: **do not pretend a two-stage
disaggregated system and a single-instance scheduler are directly
comparable under identical resource accounting without an explicit
protocol.** §4 defines exactly what "identical resource accounting" can
and cannot mean across these three topology classes.

## 2. Topology-aware registry design

`src/llmserveopt/policies/external_baselines_registry.py` adds
`ExternalBaselineSpec` (name, fidelity_class, topology_class,
pinned_source, reference_doc, factory, min_gpu_count, required_roles,
min_role_counts, six mechanism-requirement booleans/enums,
selector_eligible, historical, notes) and `EXTERNAL_BASELINE_REGISTRY:
dict[str, ExternalBaselineSpec]`. It is intentionally **not** wired into
`registry.py` in any direction — no import either way, no shared
`BASELINE_NAMES` list, no shared `make_policy` function. This is the
simplest design that satisfies the hard requirement ("existing historical
policy lists and selector candidate lists must remain unchanged by
default") without needing conditional logic anywhere in the historical
registry to "hide" the new entries.

## 3. Explicit external-baseline evaluation path

`make_external_baseline(name, **kwargs)` instantiates any of the five by
name (mirrors `registry.py`'s `make_policy`, but reads
`EXTERNAL_BASELINE_REGISTRY`, never `BASELINE_NAMES`).
`external_baselines_by_topology(topology_class)` groups them by topology
class. Verified classification (§1's table), confirmed against the actual
implementations (not assumed) via
`tests/test_external_baseline_integration.py::test_validate_topology_accepts_each_baselines_own_native_config`:

- **A. Monolithic**: `vllm_faithful`, `sarathi_faithful` — confirmed.
- **B. Disaggregated prefill/decode**: `distserve_faithful`,
  `tetriinfer_paper_reimplementation` — confirmed, with the min-role-count
  asymmetry noted above (`distserve_faithful` exactly 1+1;
  `tetriinfer_paper_reimplementation` ≥1 prefill + ≥1 decode).
  `min_role_counts` in the registry captures this precisely (both store
  `(1, 1)` as the *minimum*, but `distserve_faithful` additionally has a
  hard-coded exact-match check the harness's `validate_topology` enforces
  separately).
- **C. Multi-instance/migratory**: `llumnix_faithful` — confirmed, and
  currently the sole member of its topology class.

## 4. Resource-normalization protocols

Three protocols were designed (`src/llmserveopt/evaluation/external_baseline_configs.py`):

### Protocol A — Equal total GPU count (`matched_gpu_count_configs`)

Every baseline receives exactly `n_gpus` total GPUs and the same aggregate
KV-token budget, split per each baseline's own role structure (e.g. at
`n_gpus=4`: `tetriinfer_paper_reimplementation` gets 1 prefill + 3 decode;
`llumnix_faithful` gets 4 independent instances; `distserve_faithful` is
simply **excluded** unless `n_gpus==2`, since no other split satisfies its
hard 1+1 constraint).

- **Strength**: simplest to explain, matches a naive "same hardware
  budget" framing.
- **Weakness**: conflates architecture effects with policy effects (see
  §1's `llumnix_faithful` vs. `vllm_faithful` discussion) and silently
  favors whichever architecture happens to use its GPUs more efficiently
  for reasons having nothing to do with request scheduling quality (e.g.
  disaggregation's inherent prefill/decode interference removal, which is
  real and valuable, but is an **architecture** benefit being
  misattributed to a **scheduling policy** comparison if not labeled).

### Protocol B — Equal aggregate modeled compute/KV capacity

Hold the *sum* of `max_kv_tokens` (and, if desired, `max_active_sequences`)
constant across topologies regardless of how many physical GPUs that sum
is split across, e.g. "2000 total KV tokens" as either one monolithic GPU,
or split 1000/1000 across one prefill + one decode GPU, or split
500/500/500/500 across four independent Llumnix instances.

- **Strength**: more directly isolates "does this scheduling/placement
  *decision-making* extract more value from the same raw resource pool,"
  closer to what a scheduling-policy comparison should mean.
- **Weakness**: the normalization choice itself is not unique or
  paper-sourced — equal total KV tokens, equal total sequence slots, and
  equal total batch-token budget are three different (and not mutually
  consistent) ways to define "equal aggregate capacity," and none of the
  five pinned references specifies how a real deployment would trade
  these off when moving from one topology to another. `prefill_kv_fraction`
  (disaggregated configs) and per-instance KV splits (migratory configs)
  are this project's own disclosed, non-paper-sourced choices (see
  `external_baseline_configs.py`'s module docstring) — defensible for
  smoke validation, not yet validated as *the* right normalization for a
  manuscript claim.

### Protocol C — Architecture-native comparison (`native_config_for`)

Each baseline runs in its own intended topology at a given total KV
budget (`vllm_faithful`/`sarathi_faithful`: 1 GPU; `distserve_faithful`:
1+1; `tetriinfer_paper_reimplementation`: 1 prefill + 2 decode by default;
`llumnix_faithful`: 3 independent instances by default), with the EXACT
resource consumption (GPU count, role split, per-role KV totals) recorded
and reported alongside every result via `TopologyDescription` — never
hidden, never implied to be "the same" as another baseline's.

- **Strength**: the only protocol that does not smuggle in an unstated
  equivalence assumption between architecturally different systems. A
  claim of the form "under its own intended deployment, system X achieves
  Y" is always defensible; a claim of the form "system X beats system Y"
  under Protocol C requires the reader to also weigh the (explicitly
  reported) resource difference, which is honest rather than hidden.
- **Weakness**: does not produce a single head-to-head number; requires
  the reader/downstream analysis to reason about resource trade-offs
  explicitly rather than getting a free "controlled experiment" framing.

### Recommendation

**Protocol C (architecture-native, explicit resource reporting) is the
primary, scientifically defensible protocol for any manuscript claim
comparing across topology classes.** Protocol A is retained as a cheap
sanity/smoke-scale check (already used in §8) but must never be the basis
of a cross-architecture claim without the Protocol-C-style explicit
resource disclosure alongside it. Protocol B is documented and implemented
as a building block (`disaggregated_config`/`multi_instance_migratory_config`'s
`total_kv_tokens` parameter already supports it) but is **not** recommended
as a primary manuscript protocol yet — its own normalization choice needs
its own methodological justification, which is out of scope for this
integration phase. Within a single topology class (`vllm_faithful` vs.
`sarathi_faithful`; or `distserve_faithful` vs.
`tetriinfer_paper_reimplementation` constrained to 1+1), Protocol A/B/C
coincide (same topology ⇒ no normalization ambiguity), so this
recommendation only bites for cross-topology-class claims.

## 5. Evaluation configs

`external_baseline_configs.py` provides `monolithic_config`,
`disaggregated_config`, `multi_instance_migratory_config` (each returns
`(gpu_configs, service_model, TopologyDescription)`), plus
`native_config_for(name, ...)` (Protocol C) and
`matched_gpu_count_configs(n_gpus, ...)` (Protocol A). Every default
(`DEFAULT_TOTAL_KV_TOKENS=20_000`, `DEFAULT_PREFILL_KV_FRACTION=0.5`,
`DEFAULT_TRANSFER_DELAY`/`DEFAULT_MIGRATION_DELAY=0.005`) is disclosed in
the module docstring as this project's own choice, not sourced from any
pinned reference.

## 6. Evaluation harness

`external_baseline_harness.py`'s `run_external_baseline(name, requests,
gpu_configs, service_model, ...)`:
1. Resolves the baseline's `ExternalBaselineSpec`.
2. Calls `validate_topology` — raises `TopologyValidationError` with a
   specific, actionable message (never a generic "run failed") on any
   structurally incompatible config, BEFORE the simulator ever runs.
3. Wraps `policy.select_action` in a counting closure (admit/preempt/
   swap/migrate event counts) — the policy's own logic is never touched.
4. Runs via the existing `evaluation.run_policy.run_policy` (unchanged,
   reused as-is).
5. Returns `ExternalBaselineRunResult`: baseline provenance (name,
   fidelity class, topology class, pinned source, reference doc), every
   constructor parameter used, the seed, `TopologyDescription`
   (reconstructed from the ACTUAL `gpu_configs` used, not a caller-supplied
   claim), the full existing `RunMetrics`, the four event counters, and a
   `notes` list flagging metrics that are structurally always-zero (e.g.
   preempt/swap counts for `tetriinfer_paper_reimplementation`'s
   admission-avoidance design — a genuine zero, documented as such, not a
   silently-missing metric) or not tracked by this pass (KV utilization
   over time — see §7).
6. `results_to_json` serializes a list of results for downstream analysis.

## 7. Common metrics

Reused directly from the existing `RunMetrics` (no schema change):
`num_total`/`num_completed`/`num_dropped`/`completion_fraction`,
`request_throughput`/`token_throughput`, `weighted_goodput`,
`mean/p95/p99_ttft` and `mean/p95_tpot` (no median/p50 for TTFT/TPOT in
the existing schema — noted as a gap, not silently computed as a fake
value; `median_latency` DOES exist for end-to-end latency),
`mean/median/p95/p99/max_latency`,
`slo_violation_rate`. Arrival-normalized weighted goodput is NOT part of
`RunMetrics` currently (`weighted_goodput`'s existing definition is the
priority-weighted SLO-goodput used throughout this project, per
`docs/research_status.md`/Phase 2B.14's own metric audit — see that memory
record) — computing an arrival-normalized variant for these five
baselines is deferred to the dataset-construction phase, not fabricated
here.

**Harness-added, per-run** (never silently defaulted to 0 for a baseline
where the event genuinely cannot occur vs. simply wasn't observed — see
each spec's `notes`): `num_admit_events`, `num_preempt_events`,
`num_swap_events`, `num_migrate_events`.

**Explicitly NOT collected in this pass** (documented, not silently
zero-filled): per-GPU KV utilization over time (would require sampling
`ObservableGPUState.current_kv_tokens`/`max_kv_tokens` every step and
aggregating — a real, additive harness extension for the dataset-construction
phase, out of scope here), per-role (prefill vs. decode) utilization
breakdown for disaggregated baselines (currently only the pooled
`mean_gpu_utilization`/`mean_active_batch_size` from `RunMetrics` is
available, averaged across both roles).

## 8. Smoke-scale cross-baseline validation

Ran all 5 baselines × 8 scenario families × 2 seeds = 80 combinations
(`tests/test_external_baseline_integration.py::test_smoke_cross_baseline_scenario`,
parametrized): short-prompt/short-output, long-prompt/short-output,
short-prompt/long-output, long-prompt/long-output, bursty arrivals, high
KV pressure, mixed lengths, SLO-sensitive. **80/80 clean**: zero crashes,
zero simulator warnings (no admission-capacity violations), and
`num_completed + num_dropped == num_total` for every combination. This is
explicitly smoke-scale validation only (small traces, `total_kv_tokens=8000`)
— not a manuscript conclusion about any baseline's relative quality.

## 9. Invariants

Verified for all 5 baselines (`tests/test_external_baseline_integration.py`):
no request completes twice; no admitted request silently disappears
(`num_completed + num_dropped == num_total` under a recording wrapper that
also checks no request is ever visible in `waiting_queue` past its
arrival time); KV capacity never exceeded (no admission-rejection
warnings under moderate load); GPU role constraints match each baseline's
own declared `required_roles` exactly; deterministic reproduction (same
seed twice ⇒ identical `num_completed`/`mean_latency`/event counts);
migration (`llumnix_faithful`) and disaggregation (`distserve_faithful`,
`tetriinfer_paper_reimplementation`) both conserve total request count
end to end.

**Differential test**: `llumnix_faithful` configured with exactly 1
instance must degenerate EXACTLY to `vllm_faithful` alone on an identical
trace/GPU config — migration is structurally impossible with only one
instance (no destination exists), and `llumnix_faithful`'s own local
scheduler *is* `vllm_faithful`'s per-GPU worker (composed, not
reimplemented). Verified bit-for-bit
(`num_completed`, `mean_latency`, `mean_ttft`, `num_dropped` all
identical) — a strong, free correctness check that would fail loudly if
either policy's admission logic ever diverged unexpectedly.

## 10. Selector eligibility

| Baseline | Classification |
|---|---|
| `vllm_faithful` | ELIGIBLE_AFTER_ADDITIONAL_VALIDATION |
| `sarathi_faithful` | ELIGIBLE_AFTER_ADDITIONAL_VALIDATION |
| `distserve_faithful` | NOT_SELECTOR_COMPATIBLE (with the current selector) |
| `tetriinfer_paper_reimplementation` | NOT_SELECTOR_COMPATIBLE (with the current selector) |
| `llumnix_faithful` | REFERENCE_ONLY |

**`vllm_faithful`/`sarathi_faithful`**: topology-compatible with the
existing selector's candidate pool (monolithic, `role=None`, same
resource-sharing semantics as every one of the current 20 baselines).
Not marked ELIGIBLE_NOW because neither has ever been run through the
selector's actual feature-extraction/labeling/held-out-CI pipeline (Phase
2A.4–2B.16's own onboarding process, per `docs/research_status.md`) — that
is real, separate validation work, not a formality. Also note for the
manuscript: this project already has `vllm_style_token_budget`/
`sarathi_style` ("inspired-by" heuristics) in the current 20-baseline
portfolio; adding the faithful reimplementations alongside them is not a
replacement, and both should be described as distinct fidelity tiers in
any future write-up (already the case in `docs/baselines.md`).

**`distserve_faithful`/`tetriinfer_paper_reimplementation`**: structurally
incompatible with the current selector, which assumes one homogeneous
`role=None` pool — its feature extraction has no concept of
`role="prefill"`/`role="decode"` GPUs at all. Making these selector-
eligible would require either a new disaggregation-aware selector variant
or nontrivial feature-engineering work; neither exists yet. Additionally,
`tetriinfer_paper_reimplementation`'s lower source-confidence (paper
reimplementation, no pinned code) is a second, independent reason for
caution even once topology support exists.

**`llumnix_faithful`**: also topology-incompatible with the current
selector (multi-instance, no shared queue). More fundamentally, a
selector needs **at least two comparable candidates within the same
topology class** to be a meaningful "selection" problem at all —
`llumnix_faithful` is currently the sole occupant of the
multi-instance-migratory class, so there is nothing to select *among* yet.
It remains valuable as a fixed comparison point (e.g., "does migration
help at all, holding everything else constant") until a second baseline
in this class exists.

### Is this a policy-selection problem or an architecture-selection problem?

**Both — and they must not be conflated.** The existing selector (all 20
current baselines) answers a well-posed, single-timescale question: given
a FIXED hardware deployment (a set of `role=None` GPUs already
provisioned), which per-step admission/scheduling heuristic should run?
That is a genuine policy-selection problem: every candidate can be
swapped in or out with zero infrastructure change, at the same decision
granularity the selector already operates at (per-step).

Choosing between "monolithic vLLM/Sarathi," "disaggregated DistServe/
TetriInfer," and "migratory Llumnix" is a different kind of decision: it
requires provisioning GPUs with specific roles (or a migration control
plane) *in advance*, at a timescale far slower than per-step scheduling,
and typically made by a human operator or a separate capacity-planning
process — none of these five baselines models the cost of *changing*
topology at runtime. Treating this as just another entry in the same
per-step selector's action space would silently claim the selector can
also reconfigure physical hardware every step, which none of the
underlying simulators or pinned references support or even discuss.

This is why §1 repeatedly flags cross-topology-class comparisons as
requiring explicit resource disclosure (§4) rather than a bare "policy A
beat policy B" framing: the same evidence that makes cross-topology
comparison *possible* (Protocols A/B/C) does not make it a *policy*-level
finding — it is, at best, evidence for an *architecture*-level decision.

### Recommended selector-training design

**Option B now (separate selectors per topology class), Option C as the
long-term target once each per-topology selector is independently
validated and enough cross-topology workload-characterization data
exists.**

Justification:
- Option A (one selector over all candidates sharing a fixed topology) is
  already what exists today for the monolithic class (20 baselines); it
  simply cannot include `distserve_faithful`/`tetriinfer_paper_reimplementation`/
  `llumnix_faithful` without either breaking the "fixed topology" premise
  or requiring one of those three to be evaluated a in monolithic-only
  configuration it was never designed for.
- Option B is the direct, incremental extension: once `vllm_faithful`/
  `sarathi_faithful` complete their own selector-onboarding validation
  (making them ELIGIBLE_NOW within the *existing* monolithic selector),
  a **second, separate** selector could eventually be trained over the
  disaggregated-topology class (`distserve_faithful` vs.
  `tetriinfer_paper_reimplementation`, once topology-aware feature
  extraction exists) — these two ARE genuinely comparable under a
  matched 1+1 topology (§1), making a real 2-candidate selection problem.
  `llumnix_faithful`'s topology class has no second candidate yet, so no
  selector is trainable there regardless of design — it stays
  REFERENCE_ONLY until that changes.
- Option C (hierarchical: Level 1 architecture choice, Level 2 policy
  choice within it) is the scientifically ideal EVENTUAL structure once
  Option B's per-topology selectors each exist and are validated
  independently — Level 2 would literally be "run the appropriate
  per-topology selector from Option B." Attempting Option C now would
  require training a Level-1 architecture-selector before any Level-2
  selector it would dispatch to (monolithic's already exists; the
  disaggregated one does not yet) even exists — premature.
- Option D was considered (e.g., a single flat selector over all
  candidates with topology as an additional input feature) and rejected:
  it would reintroduce exactly the conflation §10's architecture-vs-policy
  analysis warns against, since the selector would then implicitly be
  "choosing hardware topology" through an ordinary supervised-learning
  objective never designed to represent provisioning cost, migration
  control-plane overhead, or the vastly different decision timescales
  involved.

## 11. Safe / unsafe cross-baseline claims

**Safe:**
- "Under Protocol C (each system's own intended topology, resources
  explicitly reported), system X achieves metric Y."
- "`vllm_faithful` and `sarathi_faithful` are directly comparable at equal
  GPU count (same resource-sharing model)."
- "`llumnix_faithful` with exactly 1 instance is behaviorally identical to
  `vllm_faithful` alone" (verified, §9).
- "These five baselines span three structurally distinct topology classes
  and are not all simultaneously selector-eligible under the current
  (monolithic-only) selector architecture."

**Unsafe:**
- Any bare "system X beats system Y" claim spanning topology classes
  without the Protocol-C-style explicit resource disclosure alongside it.
- Treating `distserve_faithful` vs. `tetriinfer_paper_reimplementation` as
  fully comparable at any decode-worker count other than `distserve_faithful`'s
  forced 1+1 minimum.
- Implying any of the five is currently selector-eligible, deployable, or
  part of the historical 20-baseline portfolio.
- Implying a single per-step selector could legitimately choose among all
  five as if a hardware-topology change were as cheap and reversible as a
  scheduling-policy swap (see §10's policy-vs-architecture analysis).
