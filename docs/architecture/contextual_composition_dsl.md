# CC3: Compositional DSL And Verifier

Status: COMPLETE (see `docs/audits/contextual_composition_cc3_dsl_verifier_report_20260803.md`
for the full implementation report, test evidence, and exit-gate verdict).

This document describes the CC3 extension to the JSON heuristic DSL
(`src/llmserveopt/heuristics/`), which lets a verified heuristic program
reference and compose the CC2 canonical primitive registry
(`src/llmserveopt/policies/primitives.py`, see
`docs/architecture/contextual_composition_primitives.md`). CC3 adds no new
Python execution paths (no `eval`/`exec`/dynamic imports/live LLM calls) and
changes no existing DSL document's meaning -- every addition below is
optional and additive.

## 1. Grammar And Schema

### 1.1 New leaf node kinds

Alongside the existing `{"const": v}` / `{"var": name}` leaf nodes, CC3 adds
three more, recognized anywhere an expression is expected:

```json
{"primitive": "<name>", "params": {...}}
{"primitive_gate": "<name>", "params": {...}}
{"param": "<name>"}
```

* `{"primitive": ...}` references a RANKING-family, ADMISSION-value, or
  system-level (`system_kv_pressure`, `decode_pressure`, `prefill_pressure`,
  `queue_pressure`) primitive; evaluates to a float.
* `{"primitive_gate": ...}` references an `AdmissionGate` primitive
  (`laxity_gate`, `ttft_slack_gate`); evaluates to `1.0` (pass) or `0.0`
  (reject).
* `{"param": name}` references a declared top-level `"parameters"` entry
  (see 1.4); evaluates to that parameter's currently-resolved value.

`params` is always optional; omitted parameters use the primitive's own
registry defaults (`primitives.ParamBound.default`).

### 1.2 New ops

```json
{"op": "topk_mixture", "k": K, "terms": [[e1, w1], ...]}
{"op": "bool_and", "args": [e1, e2, ...]}
{"op": "bool_or",  "args": [e1, e2, ...]}
{"op": "bool_not", "args": [e1]}
```

`topk_mixture` sums only the `K` terms with the largest `|weight|` (ties
broken by ascending term index -- deterministic), each contributing
`evaluate(e) * w`. `bool_and`/`bool_or`/`bool_not` treat operands as boolean
via `>0.0`, matching `if_then_else`'s existing convention.

`weighted_sum` (pre-existing) gained one new rule when used as a primitive
mixture: if two or more of its terms are bare `{"primitive": name}` roots,
their `compatible_families` (from the CC2 registry) must share a nonempty
intersection, or verification fails with `MIXTURE_FAMILY_INCOMPATIBLE`.
**Normalization rule (documented, not automatic):** `weighted_sum` and
`topk_mixture` weights are literal, deterministic coefficients -- there is no
implicit renormalization (e.g. to sum to 1). This is itself the "deterministic
normalization rule": the same document always produces the same weighting,
with no hidden per-call rescaling.

### 1.3 New top-level fields

All optional; a document with none of these behaves exactly as it did before
CC3.

```json
{
  "fallback": {"policy": "fifo_like"},
  "on_no_admits": "safe_fallback",
  "placement": {"keys": [{"name": "projected_gpu_load", "params": {}}, ...]},
  "admission_budget": {"primitive": "admission_credit_budget", "params": {...}},
  "parameters": [
    {"name": "kv_weight", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5}
  ]
}
```

* **`fallback.policy`** must be one of `ALLOWED_FALLBACK_POLICIES =
  {"fifo_like", "edf_like"}` (`dsl_schema.py`) -- a reference to an
  already-verified canonical safe policy, never arbitrary recursively
  verified DSL. Absent `"fallback"` is treated as an **inherited default of
  `fifo_like`**, recorded (not silent) on `CompiledHeuristic.fallback_name`.
* **`on_no_admits`** (`"safe_fallback"` | `"admit_best_effort"`) is
  **required** whenever an `admission_condition` references a
  `{"primitive_gate": ...}` (the new CC3 "admission gate" construct).
  Pre-CC3 `admission_condition` expressions (plain var/op composition, no
  `primitive_gate`) remain legacy and do **not** require it -- this is what
  keeps every existing genome-generated and hand-authored heuristic backward
  compatible.
* **`placement.keys`** is an ordered list of PLACEMENT-family primitive
  references; a candidate GPU's composite key concatenates each declared
  key's tuple in order, with `gpu_id` **always** appended last as a
  structural, non-configurable final tie-break (never user-overridable).
  Absent `"placement"` preserves the exact pre-CC3 first-feasible-GPU
  behavior.
* **`admission_budget`** declares the one stateful primitive
  (`admission_credit_budget`) at a single, unambiguous site.
  `{"primitive": ...}` referencing a stateful primitive **anywhere else**
  (inline in an expression) is rejected with `STATEFUL_PRIMITIVE_MISPLACED`.
* **`parameters`** declares externally-supplied bounded scalars (CC3 scope:
  `"type": "float"` only). Every entry needs `name`/`type`/`min`/`max`/
  `default`; `min <= default <= max`. Referenced via `{"param": name}`;
  unknown names are rejected (`PARAM_UNDECLARED`). At compile time,
  `compile_heuristic(doc, param_overrides={...})` may override declared
  defaults -- **unknown override keys are always rejected**
  (`CompilationError`), so no undeclared parameter is ever accepted. CC3
  supports declaration + default/override resolution only; a learned
  contextual predictor that sets these at runtime is explicitly CC5+ scope.

## 2. Verifier Invariants

New error codes (full list and one-line meaning in
`heuristics/verifier.py`'s module docstring): `RESERVED_VAR_NAME`,
`PRIMITIVE_UNKNOWN`, `PRIMITIVE_WRONG_SHAPE`,
`STATEFUL_PRIMITIVE_MISPLACED`, `PRIMITIVE_PARAM_INVALID`,
`PRIMITIVE_BUDGET_EXCEEDED`, `MIXTURE_EMPTY`, `MIXTURE_FAMILY_INCOMPATIBLE`,
`TOPK_INVALID_K`, `PARAM_UNDECLARED`, `PARAM_SCHEMA_INVALID`,
`PARAM_DUPLICATE_NAME`, `FALLBACK_INVALID`, `ON_NO_ADMITS_MISSING`,
`ON_NO_ADMITS_INVALID`, `PLACEMENT_EMPTY`, `PLACEMENT_TOO_MANY_KEYS`,
`PLACEMENT_KEY_UNKNOWN`, `ADMISSION_BUDGET_INVALID`.

Verification order: (1) schema/type checks, (2) primitive/gate/param leaf
resolution + family/bounds checks, (3) expression depth/node/terms/cost
budget checks (existing `max_expression_depth`/`max_nodes`/`max_terms` plus
new `max_active_primitives`/`max_placement_keys`/`max_parameters`), (4)
mixture family-compatibility checks, (5) regime/admission/placement/
parameter/fallback block-level checks, (6) a finite-evaluation dry run
against a dummy context (primitive/param leaves are lowered to dummy-valued
vars first, so a well-formed program with valid primitives/params always
dry-runs cleanly).

Literal `{"var": name}` references to either reserved prefix
(`__prim__::`, `__param__::` -- see `dsl_schema.PRIMITIVE_VAR_PREFIX`/
`PARAM_VAR_PREFIX`) are rejected (`RESERVED_VAR_NAME`); these names only
ever exist in the compiler's internal lowered form.

## 3. Compiler Behavior

Verification runs against the as-authored document (error locations use
`"primitive"`/`"primitive_gate"`/`"param"`, matching what the author wrote).
After verification passes, `compile_heuristic()` performs a **lowering
pass** on a deep copy: `{"primitive": n, "params": p}` →
`{"var": "__prim__::<canonical key>"}` (same for `primitive_gate`),
`{"param": n}` → `{"var": "__param__::n"}`. `heuristics/expressions.py`
itself never changed to know about primitives -- it only ever evaluates
ordinary `var`/`const`/`op` nodes.

`HeuristicPolicy` (`policy.py`) collects the heuristic's distinct
`(kind, name, bound_params)` primitive references once at construction
(`heuristics/primitive_bridge.py:iter_expression_blocks` +
`collect_primitive_refs`, restricted to genuine expression-tree fields --
**not** `admission_budget`/`placement`/`parameters`, which reuse the
`{"primitive": ...}` JSON shape for non-expression declarations). Each
scheduling step, it resolves them against the real
`ObservableRequest`/`ObservableState` via `primitive_bridge.
build_runtime_context()` (merged into `req_vars`) and
`build_system_context()` (merged into `sys_vars`, for the `system_value`-kind
primitives that regime `condition` expressions can actually see -- `condition`
is only ever evaluated against `sys_vars`/`batch_vars`, never `req_vars`,
which was already true before CC3).

### Fallback semantics

Any expression-evaluation failure inside `score_request`/`score_batch`/
`check_admission` now **explicitly delegates to the compiled fallback's same
method** (previously: silently returned `0.0`/`True`). The two canonical
fallback policies (`fifo_like`, `edf_like`) are their own terminal case
(`fallback=None`) to avoid infinite recursion; both are simple enough
(negated waiting time / raw deadline urgency) that they never fail
themselves in practice.

### Instrumentation

`score_request(..., trace=my_dict)` populates `active_regime`,
`dsl_version`, `compiler_version`, `fallback_activated`, and
`active_primitives` (all distinct canonical primitive-ref keys the document
declares) when a `trace` dict is passed -- zero-cost when `trace=None`
(the default). `HeuristicPolicy.last_trace` exposes the most recent step's
trace when the heuristic declares any primitive references.

## 4. Examples

Seven runnable examples in `heuristics/primitive_composition_examples.py`
(also saved as JSON in `configs/heuristics/examples/`), one per required
construct:

| Construct | Example |
|---|---|
| Named primitive reference | `edf_primitive` |
| Weighted sum | `weighted_deadline_length_ranking` |
| Sparse top-k mixture | `sparse_topk_ranking_mixture` |
| Conditional branch | `conditional_kv_pressure_branch` |
| Admission gate + fallback | `admission_gate_with_fallback` |
| Placement-score composition | `placement_score_composition` |
| Bounded external parameter | `bounded_external_parameter_example` |

Deliberately-invalid fixtures for each major verifier rule are inline in
`tests/test_contextual_composition_cc3_dsl.py` rather than separate JSON
files (a documented scope decision -- see the CC3 audit report).

## 5. Backward Compatibility

All four pre-CC3 canonical examples (`fifo_like`, `edf_like`,
`slo_kv_balanced`, `throughput_oriented`) and every `SchedulerGenomeV1`
-derived heuristic (`policies/genome.py`) continue to verify, compile, and
score identically -- pinned by regression tests. No migration is required;
`DSL_SCHEMA_VERSION` was bumped to `2` for documentation only.

## 6. Limitations

* Regime `condition` expressions can only reference **system-level**
  primitives (`system_kv_pressure`/`decode_pressure`/`prefill_pressure`/
  `queue_pressure`); a per-request (`req`-dependent) primitive referenced
  inside a `condition` will raise `ExpressionError` at evaluation time and
  silently fall through to the next regime (matching the pre-existing,
  unchanged `req.*`-in-`condition` limitation).
* `admission_budget`'s global per-step cap is applied on the
  normal/`admit_best_effort` admission paths only; when `on_no_admits`
  delegates the whole step to the fallback policy, that fallback's own
  (unconstrained) admission behavior applies for that step.
* `placement.keys` composes PLACEMENT primitives lexicographically
  (concatenated tuples); it does not support a weighted/normalized blend of
  placement keys (those are tuples, not floats, so arithmetic weighting
  isn't well-defined without a separate design decision).
* `"parameters"` type is numeric-only (`"float"`); no categorical/boolean
  parameter types are defined yet.

## 7. CC4 Interface Expectations

CC4 (offline oracle composition dataset generation) is expected to:

* generate heuristics using `compile_heuristic()`/`verify_heuristic()`
  exactly as authored here -- no new DSL surface should be required for
  dataset generation itself;
* use `dsl_schema.heuristic_hash()` (stable, sort-keys, no-NaN canonical
  JSON → SHA-256) to deduplicate/identify generated programs;
* treat `CompiledHeuristic.primitive_refs`/`placement_keys`/
  `admission_budget_spec`/`param_declarations` as the causal-input surface
  to sample/mutate over, rather than re-deriving primitive metadata;
* continue treating oracle/diagnostic fields as strictly separate from the
  causal DSL surface documented here (roadmap invariant 5).
