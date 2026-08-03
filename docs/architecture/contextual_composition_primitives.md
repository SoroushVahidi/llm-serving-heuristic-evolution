# CC2 Canonical Scheduling Primitive Interface

Status: CC2 implementation, canonical branch
`contextual-compositional-heuristics-20260731`. Canonical issue:
[#2](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/2).

This document describes the primitive interface implemented for CC2:
`src/llmserveopt/policies/primitives.py` (registry and atomic/derived
primitives) and `src/llmserveopt/policies/primitive_reconstructions.py`
(representative-policy reconstructions built only from those primitives).
Equivalence evidence lives in `tests/test_primitive_interface.py` and
`tests/test_primitive_reconstructed_policies.py`.

This phase does **not** extend the JSON DSL
(`src/llmserveopt/heuristics/`). CC3 owns that; see "How CC3 Will Later
Expose These Primitives" below.

## 1. Primitive Taxonomy

Five families, deliberately kept separate rather than collapsed into one
scalar score (per the CC2 roadmap requirement):

| Family | Purpose | Registry entries |
| --- | --- | --- |
| `RANKING` | Per-request ordering features, composed into deterministic lexicographic sort keys | `deadline_urgency`, `laxity`, `prompt_length`, `predicted_output_length`, `estimated_service_time`, `priority`, `queue_age`, `fairness_starvation_bonus`, `laxity_urgency`, `weighted_shortest_processing_score`, `request_id_tiebreak` |
| `ADMISSION` | Gates and continuous risk scores deciding scheduling-eligibility | `laxity_gate`, `ttft_slack_gate`, `admission_risk` |
| `PLACEMENT` | GPU-selection keys and placement engines | `projected_gpu_load`, `kv_pressure` (per-GPU), `tightest_kv_fit`, `least_loaded`, `round_robin_placement`, `greedy_key_placement` |
| `BATCHING` | Token-budget and admission-credit parameters | `token_budget_remaining`, `admission_credit_budget` |
| `RESOURCE_GUARD` | Feasibility and system-pressure guards | `feasible_on_gpu`, `system_kv_pressure`, `decode_pressure`, `prefill_pressure`, `queue_pressure`, `system_overload_guard` |

28 primitives are registered in total, covering every family named in the
roadmap and every primitive explicitly required: deadline urgency, laxity,
prompt length, predicted output length, estimated service time, priority,
queue age, KV pressure, projected GPU load, admission risk, prefill
pressure, and fairness/starvation prevention.

### Two related but distinct "deadline" primitives

The roadmap names "deadline urgency" as one required primitive. The
existing codebase, however, has two genuinely different formulas that
both go by "urgency" depending on which original policy you look at:

* **`deadline_urgency`** = raw `slo_deadline` (ascending: earlier
  deadline first). This is the literal EDF definition and is what EDF's
  `select_action` actually sorts by -- it ignores estimated service time
  entirely.
* **`laxity`** = `slo_deadline - now - step_size*(alpha*prompt +
  beta*output)` (ascending: smaller/more-negative slack first). This is
  what `admission_control` and `scorpio_style_slo_guard` actually use,
  and additionally what `laxity_urgency = 1/max(laxity, eps)` (derived)
  is built from for SCORPIO's composite score.

Conflating these two would have made an exact EDF reconstruction
impossible (EDF's ordering only matches `deadline_urgency`, not
`laxity`, whenever predicted service times differ across requests). They
are kept as two separate registry entries for this reason; the
architectural decision is recorded here rather than silently baked into
one name.

### Derived (composite) primitives

`fairness_starvation_bonus`, `laxity_urgency`, `weighted_shortest_processing_score`,
and `admission_risk` are explicitly marked `derived_from=(...)` in their
`PrimitiveSpec` -- they are documented compositions of atomic primitives
within the *same* family (mostly `RANKING`), not violations of the
"don't force everything into one scalar" rule, which applies *across*
families (ranking vs. admission vs. placement vs. batching), not within
one.

## 2. Interfaces

```python
class PrimitiveFamily(str, Enum):
    RANKING = "ranking"
    ADMISSION = "admission"
    PLACEMENT = "placement"
    BATCHING = "batching"
    RESOURCE_GUARD = "resource_guard"

@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    family: PrimitiveFamily
    input_type: str
    output_type: str
    param_bounds: Mapping[str, ParamBound]
    doc: str
    compatible_families: frozenset[PrimitiveFamily]
    deterministic: bool = True   # enforced at registration
    causal: bool = True          # enforced at registration
    derived_from: Tuple[str, ...] = ()
```

`register_primitive` raises `PrimitiveError` for a duplicate name, a
non-causal primitive, or a non-deterministic one -- there is no way to
register an oracle/future-information primitive through this API.
`get_primitive_spec` raises `PrimitiveError` for unknown names. Every
parameter is validated through `ParamBound.validate`, which raises
`PrimitiveError` on NaN, non-numeric, or out-of-bounds values, and on any
parameter name the primitive does not declare.

Four small wrapper classes carry the actual computation, one per
family-shape (kept separate so a RANKING value can never accidentally be
passed where a PLACEMENT key or ADMISSION predicate is expected):

* `RankingPrimitive.value(req, state, **params) -> float` and
  `.sort_key_component(...)` (direction-normalized for tuple sorting).
* `AdmissionGate.passes(req, state, **params) -> bool`.
* `PlacementKeyPrimitive.key(gpu, req, **params) -> tuple`.
* `AdmissionCreditBudget` -- the one explicitly *stateful* primitive
  (refill/consume across steps), used for SCORPIO-style rate limiting.
  Every other primitive in the registry is a pure function of its
  inputs.

Two placement **engines** (not per-GPU keys) compose ranked requests with
GPU capacity constraints into an `Action`:

* `place_round_robin(state, ranked, *, advance_index_on_failure, max_admits)`
  -- cycles a rotating start index across `state.gpu_states`, matching
  every classical policy's `gpu_idx` loop. `advance_index_on_failure`
  exists because the codebase actually has two different conventions:
  `fifo`/`edf`/`weighted_shortest_processing`/`estimated_service_time_first`
  never advance the index when no GPU is feasible; `admission_control`
  and `scorpio_style_slo_guard` do.
* `place_greedy_key(state, ranked, key_primitive, *, admit_filter, max_admits)`
  -- no rotation; admits each ranked request to the single feasible GPU
  with the smallest key (best_fit, least_loaded, pressure-based
  placement).

`build_ranking_key(components, state)` composes an ordered list of
`(RankingPrimitive, params)` pairs into one Python tuple-sort key --
exactly the pattern every original policy's `_sort_key`/`sort_key`
already used ad hoc. `rank_requests(state, components)` is the
one-line convenience wrapper (`sorted(state.waiting_queue, key=...)`).

## 3. Causal Inputs

Every primitive takes only `ObservableRequest` / `ObservableGPUState` /
`ObservableState` fields (`core/types.py`) -- the same surface every
existing deployable policy uses. None of the following ever appears in a
primitive: `Request.actual_output_tokens`, any future arrival, or any
post-hoc metric (TTFT/TPOT/actual completion time). `feasible_on_gpu`
additionally raises `PrimitiveError` if a GPU's own capacity bounds are
non-positive, an explicit unsupported-state error rather than silent
wraparound.

## 4. Parameter Semantics

All time-scale parameters (`alpha`, `beta`, `step_size`) reuse
`scoring.DEFAULT_ALPHA`/`DEFAULT_BETA` (0.5 / 1.0) and the simulator's
default `step_size` (0.001s/decode-step) as defaults, matching every
original policy's defaults exactly. Threshold parameters
(`laxity_threshold`, `ttft_slack_threshold`) default to values that
disable filtering (`inf` / `0.0`), matching `AdmissionControlPolicy`'s
and `ScorpioStyleSloGuardPolicy`'s own defaults. `ParamBound` allows an
`inf` maximum specifically so `laxity_threshold`'s "disabled" default is
representable without a special case in the validator.

## 5. Composition Boundaries

* A `RankingPrimitive`'s value may be read by an `ADMISSION` gate (e.g.
  `laxity_gate` reuses the `laxity` formula) or folded into another
  `RANKING` composite (`laxity_urgency`, `weighted_shortest_processing_score`) --
  this is recorded via `compatible_families` and `derived_from`.
* A `PlacementKeyPrimitive` is never used as a sort key for the ranking
  stage, and a ranking value is never used directly as a GPU key --
  placement always goes through `PlacementKeyPrimitive.key` or a
  placement engine.
* `RESOURCE_GUARD` scalars (`system_kv_pressure`, `decode_pressure`,
  `prefill_pressure`, `queue_pressure`) feed `system_overload_guard`
  (a `RESOURCE_GUARD`/`ADMISSION`/`BATCHING`-compatible boolean) and nowhere
  else automatically -- a reconstructed policy must explicitly wire guard
  output into its own admission/batching logic, exactly as
  `ScorpioStyleSloGuardPolicy._guard_active` originally did.
* `BATCHING`'s `AdmissionCreditBudget` only ever bounds `max_admits`
  passed to a placement engine; it never influences ranking or the
  admission gate's boolean pass/fail decision.

## 6. Representative Policy Mappings

| Original | Reconstruction | Ranking components | Admission | Placement | Equivalence |
| --- | --- | --- | --- | --- | --- |
| `fifo` | `PrimitiveFIFOPolicy` | `queue_age` (desc), `request_id_tiebreak` | none | `round_robin_placement(advance_index_on_failure=False)` | **EXACT** |
| `edf` | `PrimitiveEDFPolicy` | `deadline_urgency`, `queue_age` (desc), `request_id_tiebreak` | none | `round_robin_placement(advance_index_on_failure=False)` | **EXACT** |
| `weighted_shortest_processing` | `PrimitiveWeightedShortestProcessingPolicy` | `weighted_shortest_processing_score`, `queue_age` (desc), `request_id_tiebreak` | none | `round_robin_placement(advance_index_on_failure=False)` | **EXACT** |
| `estimated_service_time_first` | `PrimitiveEstimatedServiceTimeFirstPolicy` | `estimated_service_time`, `deadline_urgency`, `priority`, `request_id_tiebreak` | none | `round_robin_placement(advance_index_on_failure=False)` | **EXACT** |
| `best_fit` (placement-oriented) | `PrimitiveBestFitPolicy` | `queue_age` (desc), `request_id_tiebreak` | none | `place_greedy_key(TIGHTEST_KV_FIT)` | **EXACT** |
| `admission_control` (admission-oriented) | `PrimitiveAdmissionControlPolicy` | `laxity`, `priority`, `estimated_service_time`, `deadline_urgency`, `request_id_tiebreak` | `laxity_gate` | `round_robin_placement(advance_index_on_failure=True)` | **EXACT** |
| `scorpio_style_slo_guard` | `PrimitiveScorpioStyleSloGuardPolicy` | composite (`laxity_urgency` + priority + `queue_age` - decode penalty) | `laxity_gate` + `ttft_slack_gate` | `round_robin_placement(advance_index_on_failure=True)` | **APPROXIMATE** (see below) |

`queue_age` (descending) as a FIFO/tie-break substitute for raw
`arrival_time` (ascending) is exact, not approximate: within one
`select_action` call `state.time` ("now") is identical for every request
in the queue, so `age_i - age_j = arrival_j - arrival_i` exactly --
ordering by age descending and by arrival ascending are the same total
order, term for term, with no floating-point risk beyond what raw
arrival-time comparison already has.

### Why `scorpio_style_slo_guard` is APPROXIMATE, not EXACT

The reconstruction reuses the *exact same thresholds and formulas* as
the original (same `laxity_gate`/`ttft_slack_gate` cutoffs, same
`system_overload_guard` OR-of-thresholds, same composite score
`urgency + priority_weight*priority + age_bonus*age - penalty`, same
`AdmissionCreditBudget` refill/consume/cap arithmetic). The only
structural difference is that the original computes each request's
composite score and the long-decode filter inline in a single pass,
while the reconstruction calls out to primitive functions
(`LAXITY.value`, `QUEUE_AGE.value`, ...) that recompute intermediate
quantities (e.g. `laxity` is evaluated once for the gate and again
inside the composite score). This can only ever produce identical
floating-point results for these formulas (no reordering of summation,
no reassociation), and 0 mismatches were observed across all equivalence
fixtures (synthetic single/multi-GPU states, 60-trial randomized fuzz,
3-seed simulator-trace runs, and a dedicated capacity-constrained
overload-branch run -- see the test list below). It is nonetheless
labeled **APPROXIMATE** rather than **EXACT** per the CC2 exit-gate
instruction to document every non-exact-equivalence justification
explicitly, since composing primitive calls is a structurally different
code path from the original's single inline computation and no formal
floating-point-identity proof is claimed, only empirical equivalence
across the tested envelope.

This mirrors the existing `capabilities.DSL_MAPPING_STATUS` convention
(`fifo`/`edf`/`weighted_shortest_processing`/`estimated_service_time_first`
already `EXACT` there; `admission_control`/`scorpio_style_slo_guard`
already `APPROXIMATE` there) -- CC2's primitive-level equivalence
verdicts are consistent with the DSL-level verdicts already on record.

## 7. Equivalence Test Evidence

* `tests/test_primitive_interface.py` (42 tests) -- registry contracts
  (unique names, all 5 families present, every required primitive name
  present, non-empty docstrings, causal+deterministic enforcement),
  parameter-bound validation and unsupported-parameter/NaN/out-of-range
  errors, `feasible_on_gpu`'s non-positive-capacity guard, and
  per-primitive unit checks (ordering direction, formula equivalence to
  the pre-existing `scoring.py`/`policy_library_v2_helpers.py`/
  `feasibility.py`/`composition.py` helpers it reuses).
* `tests/test_primitive_reconstructed_policies.py` (65 tests) -- for
  every representative-policy pair: single-GPU synthetic state, a
  feasibility-constrained 3-GPU synthetic state, an empty-queue no-op
  case, a 60-trial randomized-fuzz case restricted to physically valid
  states (see "Known Gaps" below), a 3-seed full simulator-trace run
  compared on `num_completed`, `num_dropped`, `num_total`,
  `completion_fraction`, `arrival_normalized_weighted_goodput`, and
  `weighted_goodput`, a deterministic-replay check (same seed run twice
  through the reconstruction alone), and one dedicated
  capacity-constrained run confirming the SCORPIO overload-guard branch
  is actually exercised (not just the common-case path) with matching
  ANWG.

All 107 new tests pass; tolerance for the metric comparisons is
`abs=1e-9` (effectively exact, allowing only IEEE-754 accumulation
noise across the two independent metric computations).

## 8. Known Gaps

* **Physical-state precondition.** `queue_age`'s `max(0.0, now -
  arrival_time)` clamp, and the arrival-ordering equivalences used
  throughout this document, both assume `arrival_time <= state.time`
  for every waiting request and that `request_id` is non-decreasing with
  `arrival_time`. Both hold for every real simulator state and every
  trace generator in this repository, but an adversarially constructed
  `ObservableState` that violates them (a request "waiting" before it
  arrives, or IDs assigned out of arrival order) can make the
  reconstruction disagree with the original on tie-break order. This was
  discovered during CC2 fuzz-testing itself (an early, invalid fuzz
  harness produced spurious mismatches) and is now an explicit,
  documented precondition rather than a silent assumption.
* **SCORPIO reconstruction is APPROXIMATE by policy, not by observed
  failure** -- see the dedicated explanation above. If a future
  regression surfaces an actual floating-point divergence, it should be
  logged here with a reproducing fixture rather than silently patched.
* **No PLACEMENT-family reconstruction of `least_loaded`.** `LEAST_LOADED`
  is implemented and unit-tested in `test_primitive_interface.py` but no
  representative policy reconstruction wraps it (the task only requires
  one placement-oriented policy; `best_fit` was chosen because its
  ranking stage is trivial, isolating the placement-key equivalence
  claim more cleanly than `least_loaded`, which shares the same
  arrival-order ranking).
* **`PreemptionRule`/`CacheReuseRule`-equivalent primitives are still
  absent**, consistent with `composition.py`'s `default_module_specs()`
  marking those `ModuleKind`s unsupported. CC2 does not add preemption or
  cache-reuse primitives; this is unchanged scope, not a new gap.
* **No stateful `RANKING`/`PLACEMENT` primitive exists analogous to
  `AdmissionCreditBudget`.** Every ranking and placement primitive is a
  pure function; if a future contextual-composition experiment needs
  hysteresis or a rolling window inside ranking/placement itself (rather
  than at the `AdmissionCreditBudget` layer), that is new CC2+ scope, not
  covered here.

## 9. How CC3 Will Later Expose These Primitives Through The DSL

CC2 deliberately does not touch `src/llmserveopt/heuristics/` (the JSON
DSL, `dsl_schema.py`/`expressions.py`/`verifier.py`/`compiler.py`). The
primitive registry in this document is designed so CC3 can add DSL
support incrementally without redesigning it:

* Each `PrimitiveSpec.name` is a natural candidate for a new DSL
  `var` or a new named function callable from a `weighted_sum`/
  `if_then_else` expression -- `param_bounds` maps directly onto the
  DSL verifier's existing bounded-coefficient/bounded-parameter checks
  (`ALLOWED_OPS`, `DEFAULT_LIMITS` in `dsl_schema.py`).
  `compatible_families` gives the CC3 verifier a ready-made table for
  rejecting a composition that mixes a `PLACEMENT` key into a
  `RANKING` expression, or an `ADMISSION` gate into a `BATCHING`
  parameter, without re-deriving those rules from scratch.
* The two placement engines (`place_round_robin`, `place_greedy_key`)
  are natural candidates for a new DSL `PlacementRule` construct (already
  stubbed, unsupported, in `composition.default_module_specs()`); CC3
  would need to add a verifier rule ensuring a compiled placement engine
  choice and its key primitive are declared together.
* `AdmissionCreditBudget` is the only stateful primitive; CC3's verifier
  will need one new rule class for "stateful primitives must declare
  their refill/consume/reset contract" that does not yet exist for any
  purely-expression-based DSL rule today.
* No change to this document's registry is required to add DSL
  exposure -- CC3 is expected to add a separate compiler-facing adapter
  module (e.g. `heuristics/primitive_bridge.py`) that imports
  `primitives.py` read-only, mirroring how `composition.py` already
  imports `policy_library_v2_helpers.py` without modifying it.
