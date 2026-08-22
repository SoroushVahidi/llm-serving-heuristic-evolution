# Public Trace Replay Scenarios v1 — Design and Preregistration (Layer 2/3)

Date: 2026-08-20

## 0. Scope

**DATASET-CONSTRUCTION PREREGISTRATION.** Turns the already-complete Public Trace Corpus v1
(Layer 0/1: `data/public_trace_corpus_v1/`, commits `84fa31b`/`179a6fe`) into canonical replay
scenarios (Layer 2) and same-scenario multi-policy outcomes (Layer 3), reusing the existing
simulator, the existing frozen six-policy portfolio (`CANONICAL_ANCHOR_IDS` in
`unified_utility_matrix.py`), and the existing `run_cell`-style evaluation pattern. No new
synthetic workload family is created. No selector/router/composition experiment is implied. Does
not modify MF-PSD v1, Unified Utility Matrix v2, or any frozen historical evidence. Layer 5
(decision-critical labels) is explicitly out of scope — it depends on the separate
decision-criticality/timescale methodology.

Window/scenario selection in this document is fixed **before** any multi-policy replay is run;
nothing here is chosen or adjusted based on an observed disagreement/separation result.

---

## 1. Central Methodological Problem: Missing-Field Semantic Audit

The six frozen policies were designed against `core.types.Request`, which requires
`slo_deadline`, `priority`, `class_id`, and `predicted_output_tokens` — none of which
`public_trace_corpus_v1`'s schema provides natively (`schema.json`'s own invariant: "No field is
fabricated; unavailable fields are null"). Verified `session_id` (the one field that could have
supplied a real tenant/class grouping) is **0% populated** in all three sources (BurstGPT: 0/1,404,294;
Azure conv: 0/19,366; Azure code: 0/8,819) — there is no native class signal to fall back on.

Field-by-field audit, from direct inspection of each policy's `select_action`/`_sort_key`/`_score`
(not assumed):

| Policy | `priority` | `slo_deadline` | `class_id` | `predicted_output_tokens` | Trace-faithful without synthesis? |
|---|---|---|---|---|---|
| `full_prefill` (`GreedyArrivalPrefillControlPolicy`) | not read | not read | not read | not read (admission key is `(arrival_time, request_id)` only) | **YES** |
| `chunked_prefill_small` (same policy class) | not read | not read | not read | not read | **YES** |
| `estimated_service_time_first` | tie-break only (3rd key) | tie-break only (2nd key) | not read | **primary** (`est = α·prompt + β·predicted_output`, 1st sort key) | No — needs `predicted_output_tokens` as a primary driver |
| `least_laxity_first` | tie-break only | **primary** (`laxity = deadline − now − service_est`, 1st sort key) | not read | needed (feeds `service_est`) | No — needs `slo_deadline` as a primary driver |
| `kv_constrained_online` | **real driver** (`kv_cost / priority` in the ranking score) | **real driver** (urgency gate: `laxity ≤ urgent_laxity_seconds`) | not read | needed (feeds `kv_cost`) | No — needs both `slo_deadline` and `priority` as real (non-tie-break) drivers |
| `weighted_fair_share` | **real driver** (multiplicative in the fairness score) | not read | **primary and defining** (the entire mechanism is fairness *across classes*; docstring: "the simulator has no tenant ID... models class-level fairness") | needed (feeds `est_steps`) | No — cannot express its mechanism at all without `class_id` |

**Conclusion**: exactly 2 of 6 policies (`full_prefill`, `chunked_prefill_small`) are genuinely
trace-faithful. The other 4 require synthesized fields as *primary or defining* mechanism inputs,
not incidental tie-breaks — a single common trace-faithful overlay cannot fairly exercise all six.
This finding is based on source-code semantics alone, decided before any replay is run.

Additionally: `full_prefill`/`chunked_prefill_small` only diverge from each other when
`ServiceModel.enable_prefill_modeling=True` (confirmed by `unified_utility_matrix.py`'s own
`DEGENERATE_MECHANISM_POLICIES` set and `DEGENERATE_REASON` string — this is a documented,
pre-existing project finding, not new). This is a required **execution config** choice for the
scenario (§5), not a per-request annotation, so it does not compromise trace-faithfulness.

---

## 2. Design Decision: Two Evidence-Class Views

Per §1's finding, this design uses **two clearly separated views**, sharing the same underlying
real request core (arrival times, prompt/output token counts, source lineage) but differing in
which fields are overlaid and which policies are run:

- **`PUBLIC_TRACE_FAITHFUL`**: only `arrival_time`, `prompt_tokens`, `actual_output_tokens` (from
  the trace's own `output_tokens`), and `request_id` are used. `predicted_output_tokens` is set
  equal to `actual_output_tokens` (zero-noise; the only substitute that introduces no synthetic
  distortion — the four field-needing policies are *not* run in this view, so this default is
  inert). `priority=1.0` uniformly, `class_id="default"` uniformly (both inert, since only the two
  field-free policies run here). **Only `full_prefill` and `chunked_prefill_small` are evaluated.**
- **`PUBLIC_TRACE_DERIVED_WITH_CONTROLLED_ANNOTATIONS`**: adds the deterministic overlays defined
  in §3 for `predicted_output_tokens`, `slo_deadline`, `priority`, `class_id`. **All six canonical
  policies are evaluated.**

Every scenario record carries a `scenario_evidence_class` field with exactly these two values.
The two views are never pooled into one statistic without this label attached. This mirrors the
project's own existing `provenance_states` pattern (`schema.json`: `NATIVE` /
`DETERMINISTIC_DERIVED` / `SOURCE_SPECIFIC` / `UNAVAILABLE`), extended with one new state:
`EXPERIMENTAL_CONTROLLED_ANNOTATION` for fields that are deterministic but not trace-observed.

This is not a symmetric six-vs-six comparison choice — it is forced by §1's semantics. Choosing
otherwise (e.g., only ever running the augmented view) would silently misrepresent 4 of the 6
policies' inputs as "real workload" when they are not.

---

## 3. Controlled Annotation Rules (frozen before any replay)

### `predicted_output_tokens` (augmented view only)
Reuses the project's existing methodology verbatim:
`apply_prediction_noise(rng, actual_output_tokens, sigma)` from
`templates_fairness_starvation_v2.py` (lognormal multiplicative noise, `sigma=0` ⇒ exact,
clipped to `[1, 4096]`). **`sigma = 0.30`** — the same value already used pervasively as the
project's standard "moderate noise" setting across existing Family A/B scenario IDs
(`...noise0.30...`), reused rather than invented. `actual_output_tokens` (simulator ground truth,
never policy-visible) is the trace's real `output_tokens` value, unmodified.

### `slo_deadline` (augmented view only)
Reuses the existing formula *structure* (`templates_fairness_starvation_v2.py`:
`arrival_time + predicted_output_tokens·STEP_SIZE + slack`, `STEP_SIZE = 0.001`), but the
existing project's fixed absolute `slack` constants (0.05–2.0s) were calibrated for a
millisecond-arrival-rate microbenchmark and are not meaningful at public-trace timescales (mean
interarrival: BurstGPT ≈3.75s, Azure conv ≈0.18s, Azure code ≈0.39s, from `manifest.json`'s
`time_range_seconds`/`n_records`). Using a **workload-relative** slack instead avoids importing a
scale mismatch while keeping the same deterministic, non-tuned spirit:

```
service_est   = α·prompt_tokens + β·predicted_output_tokens   (α, β = existing scoring.py DEFAULT_ALPHA/DEFAULT_BETA)
slo_deadline  = arrival_time + service_est + SLACK_MULTIPLIER · service_est
             = arrival_time + service_est · (1 + SLACK_MULTIPLIER)
```

**`SLACK_MULTIPLIER = 1.0`** (deadline gives exactly one extra service-time-width of headroom) —
a single fixed value, chosen for being the simplest symmetric choice (double the minimum feasible
completion time), not swept or tuned. No multi-tier SLO sweep is preregistered for v1; if a future
task needs one, it must be preregistered separately, before seeing any result.

### `priority` (augmented view only)
**Uniform `priority = 1.0`** for every request. Rationale: the existing project's own precedent
(`templates_fairness_starvation_v2.py`) only assigns a non-uniform `priority` when it is *itself*
the controlled experimental variable (`priority = tenant_weight_skew`); there is no such preregistered
contrast variable for public traces, and inventing an arbitrary priority hierarchy not evidenced by
the trace would be exactly the "manufactured comparability" this design must avoid. `priority`'s
only role here is to remain a legitimate input `kv_constrained_online`/`weighted_fair_share` can
read without being a source of fabricated differentiation.

### `class_id` (augmented view only)
**`class_id = source_dataset`** (i.e., `"burstgpt"` / `"azure_2023_conv"` / `"azure_2023_code"`).
This is the only real categorical grouping the corpus has (confirmed no `session_id`/tenant
signal exists in any source, §1) and is deterministic, outcome-independent, and non-arbitrary: it
reflects a genuine difference in workload character already documented in Layer 1
(`distribution_stats.json`; e.g. Azure-code has long prompts/short outputs, Azure-conv the
reverse). Within a single-source scenario window this makes `class_id` constant across the
window — `weighted_fair_share`'s fairness mechanism is then honestly inert for that scenario (no
class contrast to exploit), which is itself a reportable finding, not a defect to engineer around
(§8 of the authorizing task: "a lower disagreement rate ... may itself be an important finding").

### Provenance labeling
Every derived field carries `field_provenance = EXPERIMENTAL_CONTROLLED_ANNOTATION` in the
scenario manifest (extending, not duplicating, the Layer-1 `provenance_states` vocabulary).
`PUBLIC_TRACE_FAITHFUL` scenarios never carry this state — every field they expose is `NATIVE` or
the existing `DETERMINISTIC_DERIVED` (for `actual_output_tokens = predicted_output_tokens`, an
identity mapping, not a distortion).

---

## 4. Window Extraction Rule (frozen before any replay)

- **Input**: `data/public_trace_corpus_v1/{source}/records.parquet`, already monotonically
  ordered by `relative_arrival_time` (Layer-1 invariant, not re-verified assumption — checked in
  §6 tests).
- **Window size**: **200 requests per scenario window**, chosen for being the same order of
  magnitude as MF-PSD's own scenario request counts and computationally tractable at full-corpus
  replay scale; fixed before any replay, not tuned on separation results.
- **Non-overlapping, sequential, deterministic**: windows are contiguous, non-overlapping slices
  in arrival-time order — no shuffling (preserves real temporal/burst structure), no
  overlap-induced leakage between windows.
- **Downsampling rule (source balance)**: each source is downsampled to a **fixed 20 windows**,
  selected by deterministic even-spacing across the source's full available window count (not the
  first 20, to avoid over-representing only the trace's opening period) —
  `stride = floor(n_available_windows / 20)`, windows at indices `0, stride, 2·stride, ...`
  (19 further windows), giving representative temporal coverage. This yields **20 windows × 3
  sources = 60 base windows**, each instantiated in both evidence-class views ⇒ **120 canonical
  scenario records**.
- **Arrival-time rebasing**: each window's `arrival_time` values are rebased so the window's first
  request has `arrival_time = 0.0` (required for the simulator; preserves all *relative*
  inter-arrival structure within the window untouched).
- **No cross-window overlap and no test/train split**: this is a replay/evaluation corpus, not a
  fitted-model dataset — no model is trained on it, so no split is required. (If a future task
  needs one, e.g. to withhold windows from any later fitting, it must define that split explicitly
  and separately — none is preregistered here.)
- **GPU config**: one GPU per scenario, `max_active_sequences=512`, `max_batch_tokens=512`,
  `max_kv_tokens=8_000_000` — the same canonical single-GPU config already used by
  `templates_prefill_decode_v2.py`/`templates_fairness_starvation_v2.py`, reused rather than
  invented.
- **`ServiceModel` config**: `step_size=0.001`, `enable_prefill_modeling=True`,
  `prefill_cost_per_token=1.0`, `step_token_budget=512`, `enable_decode_prefill_contention=True`
  — identical to the existing Family-B v2 config (§1: required for `full_prefill`/
  `chunked_prefill_small` to diverge meaningfully); `decode_first` is overridden per-policy by the
  runner exactly as the existing `_apply_prefill_chunk_override`/`_PREFILL_CHUNK_BY_POLICY`
  pattern already does.
- **Scenario ID**: `PUBLIC_TRACE::<source>::w<window_index>::<evidence_class_short>`
  (e.g. `PUBLIC_TRACE::burstgpt::w0::faithful`, `PUBLIC_TRACE::burstgpt::w0::augmented`) — globally
  unique, deterministic, reconstructible from `(source, window_index, evidence_class)` alone.
- **Seed**: `seed = 20260820` (a single fixed run-date-derived seed, used only by
  `apply_prediction_noise`'s RNG for the augmented view — the faithful view is fully
  deterministic with no RNG at all).

---

## 5. Policy Applicability (frozen)

| View | Policies evaluated | Expected Layer-3 cells |
|---|---|---|
| `PUBLIC_TRACE_FAITHFUL` | `full_prefill`, `chunked_prefill_small` | 60 scenarios × 2 = 120 |
| `PUBLIC_TRACE_DERIVED_WITH_CONTROLLED_ANNOTATIONS` | all 6 canonical anchors | 60 scenarios × 6 = 360 |
| **Total** | — | **480 scenario-policy cells** |

No policy is silently dropped from a view; the applicability table above is the complete,
explicit record (per the authorizing task's explicit instruction).

---

## 6. Layer-4 Trajectory Schema

Reuses the existing per-step logging shape already used by
`LiveHierarchicalRouterPolicy._log_step` / the decision-criticality diagnostic's trajectory
frames — no new logging system. Per step: `canonical_scenario_id`, `policy_id`,
`scenario_evidence_class`, `step`, `time`, `queue_length`, `active_request_count`,
`mean_kv_utilization`, `admitted_request_ids`, `resulting_active_count`. Raw/replay evidence only
— no counterfactual or decision-critical fields (those are Layer 5, out of scope here).

---

## 7. Success/Failure Integrity Gates

Before any result is trusted:
1. Exactly 60 base windows (20/source), 120 scenario records, 480 scenario-policy cells expected
   — asserted before results are written, matching the decision-criticality runner's own
   fail-fast pattern.
2. Every `canonical_scenario_id` unique.
3. `relative_arrival_time` monotonic non-decreasing within every source (re-verified, not
   assumed).
4. No `PUBLIC_TRACE_FAITHFUL` record carries `field_provenance = EXPERIMENTAL_CONTROLLED_ANNOTATION`.
5. Every derived field in the augmented view traces to exactly one rule in §3 — no ad hoc value.
6. No scenario/window selection depends on any replay outcome (verified by construction: windows
   are fixed by index before `run_cell` is ever called).
7. No frozen historical artifact (MF-PSD, Unified Utility Matrix v2, Public Trace Corpus v1
   Layer 0/1 files) is modified by this build (checksum-verified, mirroring
   `unified_utility_matrix.py`'s own "0 mutation of frozen sources" convention).
