# Contextual Composition CC2 Primitive Interface Report - 2026-08-02

Branch: `contextual-compositional-heuristics-20260731`.
Starting SHA: `4d806c8b1be0c4c9e202bbc7a20b3455c9c510b8`.
GitHub issue: [#2](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/2)
(closed on completion); [#3](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/3)
(now active).

## Scope

CC2 goal: define and implement the canonical primitive interface for
reusable scheduling behavior, then prove that representative existing
policies can be reproduced through those primitives. Do not extend the
JSON DSL. Do not begin CC3.

Both constraints were honored: `src/llmserveopt/heuristics/` (the DSL,
schema, expressions, verifier, compiler) was not modified, and no DSL
constructs, contextual predictors, or CC3 work were started.

## 1. Resume State Verification

Verified before implementation:

- Branch: `contextual-compositional-heuristics-20260731` (correct).
- Working tree: clean at start.
- `git fetch --all --prune`: no new remote history; local HEAD
  `4d806c8b1be0c4c9e202bbc7a20b3455c9c510b8` matched
  `origin/contextual-compositional-heuristics-20260731` exactly (0 ahead,
  0 behind).
- `python scripts/check_contextual_composition_status.py --resume-readiness`
  passed, confirming CC2 was the single `NEXT` phase and issue #2 was the
  active issue.

No newer commits existed to inspect; no reset or overwrite was needed.

## 2. Architecture Implemented

Two new modules under `src/llmserveopt/policies/`:

- **`primitives.py`** (canonical registry): five families
  (`RANKING`, `ADMISSION`, `PLACEMENT`, `BATCHING`, `RESOURCE_GUARD`),
  28 registered primitives, `PrimitiveSpec`/`ParamBound` typed metadata,
  `register_primitive`/`get_primitive_spec`/`list_primitives` registry
  API, and four small wrapper classes
  (`RankingPrimitive`, `AdmissionGate`, `PlacementKeyPrimitive`,
  `AdmissionCreditBudget`) plus two placement engines
  (`place_round_robin`, `place_greedy_key`) and one ranking-key builder
  (`build_ranking_key`/`rank_requests`).
- **`primitive_reconstructions.py`**: seven representative-policy
  reconstructions built only from the primitives above.

Full design rationale, taxonomy, causal-input notes, parameter semantics,
composition boundaries, and the CC3 DSL-exposure plan are in
`docs/architecture/contextual_composition_primitives.md` (not duplicated
here).

Existing infrastructure was extended, not duplicated: primitives reuse
`scoring.py` (`predicted_service_proxy`, `kv_fill_ratio`, `remaining_kv`),
`policy_library_v2_helpers.py` (`gpu_pressure`), and follow the same
per-request-feature/rank-aggregation pattern already established by
`composition.py`'s `rank_with_named_expert` -- but decompose it one level
further, into typed atomic/derived primitives per family, per the CC2
requirement to keep ranking/admission/placement/batching semantics
separate rather than one scalar score.

## 3. Files Changed

New:

- `src/llmserveopt/policies/primitives.py`
- `src/llmserveopt/policies/primitive_reconstructions.py`
- `tests/test_primitive_interface.py`
- `tests/test_primitive_reconstructed_policies.py`
- `docs/architecture/contextual_composition_primitives.md`
- `docs/audits/contextual_composition_cc2_primitive_interface_report_20260802.md` (this file)

Modified (navigation/status only, no code semantics changed elsewhere):

- `docs/contextual_composition_roadmap.md` (CC2 → COMPLETE, CC3 → NEXT,
  YAML marker, evidence links, CC2 phase narrative)
- `docs/START_HERE_CONTEXTUAL_COMPOSITION.md` (current phase, read-order
  list, "what not to do yet")
- `docs/RESUME_CONTEXTUAL_COMPOSITION.md` (current phase, exact next
  task, verify-state pytest command, GitHub issue pointer)
- `docs/CONTEXTUAL_COMPOSITION_BRANCH.md` (status, query sequence,
  guardrail, next action)
- `docs/contextual_composition_decisions.md` (new CCD-012 entry)
- `scripts/check_contextual_composition_status.py` (marker/status-table/
  required-strings updated for the CC2→CC3 transition)
- `tests/test_contextual_composition_status_checker.py` (matching
  updates to the checker's own test expectations)

## 4. Representative Policies Mapped

| Original | Reconstruction | Orientation | Equivalence |
| --- | --- | --- | --- |
| `fifo` | `PrimitiveFIFOPolicy` | ranking | **EXACT** |
| `edf` | `PrimitiveEDFPolicy` | ranking | **EXACT** |
| `weighted_shortest_processing` | `PrimitiveWeightedShortestProcessingPolicy` | ranking | **EXACT** |
| `estimated_service_time_first` | `PrimitiveEstimatedServiceTimeFirstPolicy` | ranking | **EXACT** |
| `best_fit` | `PrimitiveBestFitPolicy` | **placement** (required) | **EXACT** |
| `admission_control` | `PrimitiveAdmissionControlPolicy` | **admission** (required) | **EXACT** |
| `scorpio_style_slo_guard` | `PrimitiveScorpioStyleSloGuardPolicy` | admission (bonus) | **APPROXIMATE** (documented) |

This covers all six required representative policies (fifo, edf,
weighted_shortest_processing, estimated_service_time_first, one
placement-oriented policy, one admission-oriented policy) plus one bonus
reconstruction (`scorpio_style_slo_guard`, explicitly named as an
acceptable alternative admission-oriented policy in the task).

### Exact vs. approximate, and why

Six of seven are **EXACT**: verified bit-for-bit identical
`Action.admit` mappings across every test fixture (see below), with no
tolerance needed. The `PrimitiveEDFPolicy`/`PrimitiveFIFOPolicy` cases
rely on one documented but provably-exact substitution: sorting by
`queue_age` descending is mathematically identical to sorting by
`arrival_time` ascending within one `select_action` call, because
`state.time` ("now") is the same for every waiting request in that call
(`age_i - age_j = arrival_j - arrival_i` exactly).

`PrimitiveScorpioStyleSloGuardPolicy` is **APPROXIMATE**: it reuses the
exact same formulas, thresholds, and `AdmissionCreditBudget`
refill/consume/cap arithmetic as the original, but composes them through
primitive-function calls rather than the original's single inline
per-request computation. Zero mismatches were observed across every
tested fixture (see Section 5), but this is reported as an empirical
equivalence claim over the tested envelope, not a formal
floating-point-identity guarantee, per the CC2 instruction to document
every non-exact case explicitly. This mirrors the pre-existing
`capabilities.DSL_MAPPING_STATUS` convention, where
`scorpio_style_slo_guard` is already `APPROXIMATE` at the DSL level for
the same reason (composing a formula from smaller pieces rather than one
inline pass).

No failed mappings: all seven attempted reconstructions produced usable
equivalence evidence (six exact, one approximate-with-evidence). None
were abandoned or left without a resolution.

## 5. Equivalence Test Evidence

Two new test files, 107 tests total, all passing:

- **`tests/test_primitive_interface.py`** (42 tests): registry contracts
  (unique names; all 5 families represented; every roadmap-required
  primitive name present; non-empty docstrings; causal+deterministic
  enforcement at registration), parameter-bound validation (out-of-range,
  NaN, unknown-parameter rejection), `feasible_on_gpu`'s
  non-positive-capacity guard, and per-primitive unit tests confirming
  each primitive's formula matches the pre-existing helper it wraps
  (`scoring.py`, `policy_library_v2_helpers.py`, `feasibility.py`,
  `composition.py`).
- **`tests/test_primitive_reconstructed_policies.py`** (65 tests), for
  every one of the seven pairs above:
  - single-GPU synthetic-state admitted-ID/ordering match;
  - a feasibility-constrained 3-GPU synthetic-state match (exercises
    `Action.admit` GPU-assignment equivalence, not just ordering);
  - an empty-waiting-queue no-op case;
  - a 60-trial randomized-fuzz case per policy (420 total fuzz
    comparisons), restricted to physically valid states (see "Known
    Gaps" below);
  - a 3-seed full simulator-trace run (`run_policy`, 50-request Poisson
    arrivals, 2 GPUs) compared on `num_completed`, `num_dropped`,
    `num_total`, `completion_fraction`, `arrival_normalized_weighted_goodput`,
    and `weighted_goodput` at `abs=1e-9` tolerance;
  - a deterministic-replay check (same reconstruction, same seed, run
    twice, metrics compared at `abs=1e-9`);
  - one dedicated capacity-constrained fixture confirming the SCORPIO
    overload-guard branch (KV/decode/queue thresholds, long-decode
    filtering, credit-budget throttling) is actually exercised, not just
    the common-case path, with matching ANWG.

All 107 tests pass. Tolerance for every metric comparison is `abs=1e-9`
(allows only IEEE-754 accumulation noise between two independently
computed metric objects; not a loosened equivalence bar).

### A precondition discovered during testing (see also "Known Gaps")

An early, invalid version of the randomized-fuzz harness allowed
requests with `arrival_time > state.time` and non-arrival-monotonic
`request_id` values -- states that cannot occur in any real simulator
run but that the harness could still construct directly. This produced
spurious FIFO/EDF/WSP mismatches that were traced to the harness, not the
primitive interface: `fifo.py`'s own docstring already states the
simulator guarantees `waiting_queue` arrives arrival-ordered, and every
trace generator in this repository assigns `request_id` non-decreasing
with `arrival_time`. The fuzz harness was corrected to only construct
physically valid states (arrival-monotonic IDs, `now >= max(arrival_time)`),
after which all 3500 ad hoc trial comparisons (and the 420 in the
committed test suite) matched exactly. This precondition is now
documented explicitly in `primitives.py`'s `queue_age` docstring and in
the architecture doc's "Known Gaps" section, rather than left as a silent
assumption.

## 6. Existing Test Suite Impact

- `tests/test_policy_composition.py`,
  `tests/test_score_and_reciprocal_rank_composition.py`,
  `tests/test_policy_genome_coverage.py`,
  `tests/test_estimated_service_time_first_policy.py`,
  `tests/test_admission_control_policy.py`,
  `tests/test_scorpio_style_slo_guard_policy.py`,
  `tests/test_policy_feasibility.py`: 279 tests, all pass unchanged (no
  original policy or composition code was modified).
- Full repository suite: `python -m pytest -q` -- 3054 tests collected
  (`pytest --collect-only -q`); full run: 3047 passed, 5 skipped, 2
  failed. One failure is the expected `check_contextual_composition_status.py
  --resume-readiness` "working tree is not clean" result, which resolves
  once this query's changes are committed (see Section 7). The other,
  `test_decode_prefill_contention_execution.py::TestLegacyModeUnchanged::test_existing_yaml_configs_do_not_set_new_field`,
  is a **pre-existing failure unrelated to CC2**: `git log` confirms both
  `configs/cc1b_composition_discriminative.yaml` (last touched in commit
  `db4dcaa`, "research: resolve CC1 composition discriminativeness") and
  the failing test (last touched in `eb2f7db`, predating this branch's
  CC1b work) were unmodified in this session, and both predate this
  query's starting SHA `4d806c8b1be0c4c9e202bbc7a20b3455c9c510b8`. Not
  fixed here, as it is out of CC2 scope; noted for future cleanup.
- `python -m compileall -q src/` and the two new/changed test files:
  clean.
- YAML marker in `docs/contextual_composition_roadmap.md`: parses with
  `yaml.safe_load` to the expected dict.
- Internal markdown links across the six touched/created
  contextual-composition docs: checked programmatically; one pre-existing
  `../architecture/...` relative-path bug (an extra `../`, since
  `contextual_composition_roadmap.md` lives directly under `docs/`, not
  one level deeper) was found and fixed during this pass.

## 7. Status And Resume-Readiness Checkers

`scripts/check_contextual_composition_status.py` and
`tests/test_contextual_composition_status_checker.py` were both updated
in lockstep with the doc changes (marker fields, status-table expected
phases, required-string checks against the current-state docs), while
leaving all checks against frozen historical documents (the Query 5/6/7
pause-era reports and their recorded ANWG numbers) untouched, since those
remain true historical records.

```bash
python scripts/check_contextual_composition_status.py
# -> contextual composition status check passed
python scripts/check_contextual_composition_status.py --resume-readiness
# -> passes once this query's changes are committed (fails on an
#    uncommitted working tree by design, which is the correct behavior)
```

## 8. Unresolved Gaps

See `docs/architecture/contextual_composition_primitives.md` Section 8
("Known Gaps") for full detail. Summary:

1. Ranking/placement primitives assume physically valid causal states
   (arrival-monotonic request IDs, `arrival_time <= state.time`); this
   holds for every real simulator state but is not independently
   re-validated by a runtime assertion inside the primitives themselves.
2. `PrimitiveScorpioStyleSloGuardPolicy` is labeled APPROXIMATE by policy
   (composed-call structure differs from the original's single inline
   pass) rather than by any observed failure.
3. `LEAST_LOADED` placement key is implemented and unit-tested but has no
   dedicated representative-policy reconstruction (only one
   placement-oriented reconstruction was required; `best_fit` was chosen
   because its ranking stage is trivial, isolating the placement-key
   equivalence claim).
4. No preemption or cache-reuse primitives were added, matching
   `composition.py`'s existing `ModuleKind.PREEMPTION`/`UNSUPPORTED`
   scope boundary -- unchanged, not a new gap.
5. `AdmissionCreditBudget` remains the only stateful primitive; no
   stateful ranking/placement primitive exists yet (not needed for any
   of the seven reconstructions).

None of these block the CC2 exit gate, which requires representative
policies to be reproducible exactly or approximately with evidence -- met
for all seven attempted reconstructions.

## 9. CC2 Verdict

**COMPLETE.** The equivalence gate passed: 6/7 representative policies
EXACT, 1/7 documented APPROXIMATE with 0 observed mismatches across all
tested fixtures. The DSL was not touched. 107 new tests added; the full
pre-existing test suite (3054 tests collected) is unaffected.

## 10. Query 9 Recommendation

Begin CC3 (compositional DSL and verifier). Concretely:

1. Extend `src/llmserveopt/heuristics/dsl_schema.py`'s `ALLOWED_OPS`/
   variable tables (or an adjacent adapter module, e.g. a new
   `heuristics/primitive_bridge.py` that imports `primitives.py`
   read-only) to expose named references to the CC2 primitive registry,
   using each `PrimitiveSpec.param_bounds` as the verifier's bound source
   and `compatible_families` as the family-mixing rule source (see the
   architecture doc's Section 9 for the specific integration plan).
2. Add the still-unsupported `PlacementRule` DSL construct backed by
   `place_round_robin`/`place_greedy_key`, plus one new verifier rule
   class for the `AdmissionCreditBudget` stateful-primitive contract
   (refill/consume/reset), since no purely-expression-based DSL rule
   covers stateful primitives today.
3. Do not begin CC4 (offline oracle composition dataset) until CC3's own
   compile/verify/equivalence/property/adversarial-test gate passes.
4. Continue tracking work in GitHub issue #3; issue #2 is closed.
