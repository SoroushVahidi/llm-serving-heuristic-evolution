# Unified Policy-Utility Matrix — Step 2 Preregistration (v1)

Date: 2026-08-17

## 0. Scope

This is the revised roadmap's **Step 2: Unified Baseline Evaluation**
([`../audits/reassessment_composition_hypothesis_20260817.md`](../audits/reassessment_composition_hypothesis_20260817.md)
§O). It evaluates the six canonical anchor policies across all MF-PSD v1
scenarios to build the dense policy-utility matrix. **Data generation only.**
No selector is trained, no pairwise-regret model is fit, no mechanism
attribution is performed, no composition/synthesis work is started. Frozen
MF-PSD v1 (`experiments/mf_psd_v1/`) and all three frozen family source runs
are read-only inputs; none are modified.

## 1. Canonical Policy Set (frozen before any scoring)

The reassessment doc's own roadmap (§O step 2) names the six anchors
verbatim: `estf`, `wfs`, `full_prefill`, `chunked_prefill_small`,
`least_laxity`, `kv_constrained`. This is identical, term-for-term, to the
MF-PSD v1 audit's own canonical-anchor table (§D). There is no ambiguity to
resolve and no post-hoc selection was performed.

| Shorthand | `canonical_policy_id` | Native family |
|---|---|---|
| `estf` | `estimated_service_time_first` | A |
| `wfs` | `weighted_fair_share` | A |
| `full_prefill` | `full_prefill` | B |
| `chunked_prefill_small` | `chunked_prefill_small` | B |
| `least_laxity` | `least_laxity_first` | C |
| `kv_constrained` | `kv_constrained_online` | C |

**Excluded (2 of the 8 MF-PSD policy identities): `fifo`, `aging_priority`.**
Both are Family-A-only extra baselines the original Family A v2 pilot
evaluated as additional diagnostic comparators; neither was ever run in
Family B or C, and neither is named as an anchor by the reassessment
roadmap. MF-PSD v1 §D documents the same exclusion. This build takes no
position on whether they should be cross-family-evaluated in a future step
(MF-PSD v1 audit §Q item 4) — out of scope here.

## 2. Cross-Family Executability Audit

12 non-native (policy, family) combinations require new evaluation (6
anchors × 2 non-native families each). Two independent, code-verified
findings materially change which combinations are scientifically valid.

### 2.1 Finding A — `full_prefill`/`chunked_prefill_small` are a ServiceModel
config toggle, not a decision-function difference

Reading `src/llmserveopt/policies/prefill_control_variants.py`: both
`full_prefill` and `chunked_prefill_small` instantiate the **same**
`GreedyArrivalPrefillControlPolicy` (arrival-ordered greedy admission,
identical to a FIFO-like `select_action`). Their only difference is the
paired `service_model_kwargs` override (`max_prefill_chunk_tokens`,
`decode_first`), which only has an observable effect on simulated behavior
when `ServiceModel.enable_prefill_modeling=True` (`service_model.py` line
73, default `False`; confirmed by `GPUState.step()` in `simulator/gpu.py`
line 246: `if service_model is None or not service_model.enable_prefill_modeling:` skips the phase-split path entirely).

Checked directly: Family A's templates
(`templates_fairness_starvation_v2.py`) and Family C's templates
(`templates_kv_pressure_v2.py`, and explicitly documented in
`templates_kv_pressure.py`: *"`enable_prefill_modeling` is left at its
default (False, instant prefill)"*) never set `enable_prefill_modeling` in
their generated `service_model_kwargs`. The runner's own merge order
(`merged = dict(scenario.service_model_kwargs); merged.update(variant_kwargs)`,
`run_policy_separation_prefill_decode_pilot_v2.py` line 270) only ever
overrides `max_prefill_chunk_tokens`/`decode_first`, never
`enable_prefill_modeling`.

**Consequence:** on any Family A or Family C scenario, `full_prefill` and
`chunked_prefill_small` are computed by the identical policy object under a
`ServiceModel` where the chunk-budget override has no observable effect —
the two named policies collapse to one behavior (plain greedy-arrival
admission, effectively a FIFO twin) and are **indistinguishable from each
other** outside Family B. This is real, mechanistic, and expected, not a
bug — it is exactly the "does a policy degenerate into another on a family"
check this preregistration is required to make explicit.

**Classification:** `full_prefill` × {A, C} and `chunked_prefill_small` ×
{A, C} → **`VALID_WITH_DOCUMENTED_NEUTRAL_DEFAULT`**. They execute cleanly,
introduce no leakage, and are launched (per §8's explicit allowance for this
classification), but every such row is tagged
`degenerate_mechanism=True, degenerate_reason="enable_prefill_modeling=False on native scenario ServiceModel; chunk-budget override has no observable effect"`
so no downstream consumer mistakes these cells for a genuine chunking-mechanism
signal.

### 2.2 Finding B — Family C / KV v2 scenario reconstruction is not
byte-exact (confirmed, blocks that direction)

`least_laxity_first`, `kv_constrained_online`, `estimated_service_time_first`,
and `weighted_fair_share` are genuine `ObservableState`-driven policies (no
missing-field dependency: `slo_deadline`, `priority`, `class_id`,
`predicted_output_tokens`, and every `ObservableGPUState`/`GPUConfig` field
including `max_kv_tokens` are universal across all three families' schemas).
Numerically they execute on any family without error. But evaluating *new*
anchors on a family's scenarios requires **regenerating that family's
`requests`/`gpu_configs` from `(template function, params, seed)`**, per
`schema.py`'s documented contract — MF-PSD's frozen artifacts do not carry
full per-request traces.

Before writing any harness code, reconstruction fidelity was verified
directly (read-only, against each family's own frozen `per_policy_results.csv`,
using the exact same `build_scenarios_from_config`/`build_scenarios`
functions each pilot's own runner script uses):

| Family | Scenarios regenerated | Cells compared (native policies) | Mismatches (`>1e-9`) | Max `|Δ ANWG|` |
|---|---|---|---|---|
| A v2 (fairness/starvation) | 72/72 | 144 | **0** | 0 |
| B v2 (prefill/decode) | 32/32 | 64 | **0** | 0 |
| C / KV v2 | 72/72 | 144 | **99** | **0.25** |

Family A and Family B reconstruct byte-exact against their own frozen
native cells — regenerating their scenarios for cross-family evaluation is
sound. **Family C does not** — this independently reproduces, at
essentially the same magnitude, the pre-existing finding in
[`kv_v2_reproducibility_forensic_20260817.md`](../audits/kv_v2_reproducibility_forensic_20260817.md)
(`REPRODUCIBILITY_GAP_BOUNDED`, root cause not identified, traced to
BurstGPT dataset sampling-pool sensitivity). A new anchor evaluated on a
*regenerated* Family-C scenario would not be evaluated on the same
underlying request trace as that scenario's frozen native cells — breaking
the matrix's core "same scenario, different policy" comparability
guarantee.

**Classification:** `estf`, `wfs`, `full_prefill`, `chunked_prefill_small`
× Family C (all 4 anchors × 72 scenarios = 288 cells) →
**`UNSUPPORTED` (scenario-reconstruction fidelity failure, not a policy
defect)**. **Not launched in this build.** This is a pre-existing,
separately-tracked, unresolved problem (see the forensic audit) — this task
does not attempt to fix it, per its own explicit scope (data generation
only) and the standing rule against forcing a dense matrix from
scientifically meaningless combinations.

### 2.3 Remaining 8 combinations — VALID

`estf`, `wfs` × Family B (32 scenarios each) and `least_laxity`,
`kv_constrained` × Family A (72 scenarios each): genuine `ObservableState`
decision functions, byte-exact scenario reconstruction (Family A, B both
verified above), no missing fields, no leakage risk (see §6). **VALID.**

### 2.4 Net launchable cell count

| Direction | Cells | Classification |
|---|---|---|
| New anchors on Family A (72 scenarios × 4 anchors) | 288 | 144 VALID (`least_laxity`, `kv_constrained`) + 144 VALID_WITH_DOCUMENTED_NEUTRAL_DEFAULT (`full_prefill`, `chunked_prefill_small`) |
| New anchors on Family B (32 scenarios × 4 anchors) | 128 | 128 VALID |
| New anchors on Family C (72 scenarios × 4 anchors) | 288 | **288 UNSUPPORTED — not launched** |
| **Total planned (roadmap estimate)** | **704** | |
| **Total launched this build** | **416** | |

This build produces a **partial** dense matrix: full 6-anchor coverage for
Family A's 72 and Family B's 32 scenarios (104 of 176), Family C's 72
scenarios remain at native 2/6 anchor coverage, unchanged from MF-PSD v1.
This is a direct, evidence-based consequence of §2.2, not a scope-narrowing
choice made for convenience.

## 3. Frozen Design

- **Policy set:** the 6 anchors in §1, `canonical_policy_id` values
  identical to MF-PSD v1.
- **Scenario set:** all 176 MF-PSD v1 canonical scenarios (context/identity
  only); **new evaluations run only for the 416 cells in §2.4**.
- **Expected new-cell count:** 416 (288 Family A + 128 Family B), 0 skipped
  silently — every one of the 288 Family-C cells is recorded as an explicit
  `status="unsupported_scenario_reconstruction"` placeholder row, not
  omitted.
- **Scenario regeneration:** `build_scenarios_from_config` /
  `build_scenarios` (imported from the existing frozen runner scripts,
  unmodified) called with the exact same `configs/*.yaml` used by each
  pilot's original launch (`configs/policy_separation_fairness_starvation_pilot_v2.yaml`,
  `configs/policy_separation_prefill_decode_pilot_v2.yaml`), same grid
  order, same seeds. `LLM_SERVEOPT_BURSTGPT_CSV` pointed at the repo's
  staged `.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv` (same
  file used for the §2.2 verification, which reproduced Family A/B
  byte-exact).
- **Policy construction:** `estimated_service_time_first`,
  `weighted_fair_share`, `least_laxity_first`, `kv_constrained_online` via
  their native constructors (default hyperparameters, matching the frozen
  source pilots — no retuning). `full_prefill`/`chunked_prefill_small` via
  `make_prefill_decode_variants_v2(chunk_small=64)` (matching Family B v2's
  own `chunk_budgets`).
- **ServiceModel per cell:** the target scenario's own generated
  `service_model_kwargs`, with the prefill-variant's `variant_kwargs`
  merged on top **only** for `full_prefill`/`chunked_prefill_small` cells
  (identical merge order to the native Family B v2 runner). No other cell
  overrides any ServiceModel field — every non-native cell runs under its
  target scenario's own **native, already-calibrated** execution
  environment. Grafting a foreign family's execution semantics
  (e.g. forcing `enable_prefill_modeling=True` onto a Family A scenario) was
  considered and rejected: it would evaluate a scenario under conditions
  its SLOs/service times were never calibrated against, confounding the
  cross-family comparison rather than isolating the policy's mechanism.
- **Feature visibility:** policies receive only `ObservableState` /
  `ObservableRequest` — no `mechanism_family`, `scenario_id`,
  `canonical_scenario_id`, seed, source split, oracle score, or other-policy
  utility. Identical guarantee to every frozen source pilot (all policies
  are the exact frozen classes from `src/llmserveopt/policies/`, unmodified).
- **Primary metric:** `arrival_normalized_weighted_goodput`, from
  `RunMetrics` (`src/llmserveopt/core/metrics.py`) — same shared,
  code-verified computation as MF-PSD v1 §F.
- **Secondary metrics:** `completion_fraction` (from the same shared
  `RunMetrics`) and `unweighted_slo_success_rate`
  (`(len(completed) - n_violated) / max(1, n_total_requests)`, same formula
  verified byte-identical across all three families' runners in MF-PSD v1
  §F).
- **Success/failure semantics:** `status="success"` on a clean run;
  `status="failed"` with `error`/`traceback` recorded and primary metric
  `NaN` on any exception; `status="unsupported_scenario_reconstruction"` for
  the 288 explicitly-blocked Family-C cells (never attempted).
- **Determinism:** every cell's inputs (scenario regeneration params, seed,
  policy constructor args, ServiceModel kwargs) are fixed by config; no
  randomness outside the frozen `seed` already used by the scenario's own
  request generation and `sim.run(..., seed=scenario.seed)`.
- **Matrix completeness criterion:** every one of the 416 launched cells
  has an explicit row (`success` or `failed`, never silently missing); the
  288 blocked cells have an explicit `unsupported_scenario_reconstruction`
  row so the matrix's shape stays fully accounted-for even though it is not
  fully dense.
- **Comparability criterion:** a cell is comparable to another cell on the
  same scenario only if both ran on byte-identical `requests`/`gpu_configs`
  — verified for Family A/B by §2.2's reproducibility check; **not claimed**
  for Family C (hence §2.2's exclusion).
- **Stop conditions:** any of — regenerated Family A/B scenario count
  mismatches MF-PSD's recorded 72/32; any leakage-guard assertion fails;
  any frozen source or MF-PSD v1 artifact is modified (checksum drift);
  failure rate on the smoke subset exceeds 0 unexplained failures.

## 4. Preserving Native Cells

The existing 496 MF-PSD v1 rows are read, never recomputed. The Step-2
long-form table carries a `cell_source` column with values
`SOURCE_NATIVE` (496 rows, copied read-only from MF-PSD v1, same values,
same provenance) or `STEP2_CROSS_FAMILY_EVALUATION` (up to 416 newly
computed rows + 288 explicit unsupported placeholders). No native value is
ever overwritten by a new evaluation, including on scenarios where a smoke
check independently re-evaluates a native policy (§7) — that comparison is
recorded separately and never replaces the frozen value.

## 5. Output Location

`experiments/unified_utility_matrix_v1/` — separate from
`experiments/mf_psd_v1/`, which is not modified. Non-timestamped path for
the same reason MF-PSD v1 used one (§Q rationale there): deterministic,
byte-reproducible transform/evaluation of frozen+regenerated inputs, not a
one-off stochastic run needing a timestamp to avoid collisions.
