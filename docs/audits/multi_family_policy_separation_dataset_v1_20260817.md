# Multi-Family Policy Separation Dataset (MF-PSD) v1 — Build and Audit

Date: 2026-08-17

## 0. Scope

This document audits the construction of the canonical **Multi-Family Policy
Separation Dataset (MF-PSD) v1**, the first task of the revised roadmap in
[`reassessment_composition_hypothesis_20260817.md`](reassessment_composition_hypothesis_20260817.md)
(`COMPOSITION_DEMOTED`, revised roadmap Step 1: *"Data Unification"*).

**This task is DATA UNIFICATION ONLY.** No selector was trained, no selector
hyperparameters were tuned, no pairwise-regret learning happened, no
mechanism attribution happened, no composition/synthesis experiment was
launched, and no frozen historical experiment or verdict was modified.

## A. Source Runs and Frozen Provenance

Per the higher-level reassessment (section J), three structurally distinct
mechanism families reached their respective composition-readiness gates and
are the intended MF-PSD sources:

| Family | Mechanism | Source run directory | Launch git SHA | Launch branch | Audit | Family verdict |
|---|---|---|---|---|---|---|
| **A v2** | Fairness/starvation (ranking) | `experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/` | `16ad5d3e5af2e02516dfc42cc0825fa8eb7cbf38` | `policy-separation-v1-wulver-20260809` | [`policy_separation_fairness_starvation_pilot_v2_20260816.md`](policy_separation_fairness_starvation_pilot_v2_20260816.md) | `USEFUL_BUT_NEEDS_REFINEMENT` |
| **B v2** | Prefill/decode TTFT contention (chunking) | `experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/` | `ecc0422286886c83d263e87655ed1123e62d2565` | `contextual-compositional-heuristics-20260731` | [`policy_separation_prefill_decode_pilot_v2_20260817.md`](policy_separation_prefill_decode_pilot_v2_20260817.md) | `FAMILY_B_COMPOSITION_READY` |
| **C / KV v2** | KV-pressure admission control (memory) | `experiments/kv_pressure_pilot_v2_20260817T165053Z/` | `6be526ebffe4c3eba6428eab27f9adae1835d320` | `contextual-compositional-heuristics-20260731` | [`family_c_kv_pressure_pairwise_separation_v2_20260817.md`](family_c_kv_pressure_pairwise_separation_v2_20260817.md) | `KV_FAMILY_COMPOSITION_READY` |

**Verification method, not inference from names.** For A v2 and B v2, the
launch SHA/branch was read directly from `run_manifest.json` /
`git_state.txt` inside the frozen run directory. For KV v2 (which predates
the provenance guard added in commit `c757d00`, and has no
`run_manifest.json` of its own — only `final_summary.json`, `run.log`,
`per_policy_results.csv`), the launch commit was cross-checked two
independent ways: (1) `git log --oneline -- configs/kv_pressure_pilot_v2.yaml
docs/design/POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md` shows the only
matching commit is `6be526e` ("feat: Family C v2 KV-pressure reserve
refinement — KV_FAMILY_COMPOSITION_READY"), and (2) that same SHA is
independently named as the KV v2 launch commit in
[`kv_v2_reproducibility_forensic_20260817.md`](kv_v2_reproducibility_forensic_20260817.md).
That forensic audit's caveat (the current environment cannot reproduce the
historical KV v2 CSV bit-for-bit even via the original unmodified runner)
applies to *re-running* KV v2, not to reading its already-frozen,
already-committed `per_policy_results.csv` — which this task treats as
immutable evidence, unchanged, per its own SHA-256 checksum recorded in
`mf_psd_provenance_v1.json`.

**Family A v2's literal verdict is `USEFUL_BUT_NEEDS_REFINEMENT`, not
`*_COMPOSITION_READY`.** It is included per the reassessment document's
explicit instruction (section J/K): *"Family A v2 + Family B v2 + KV v2
provide high-quality, verified, low-tie-rate scenario boundaries"* and
*"three completely distinct structural families that have successfully
passed rigorous policy-separation audits with confirmed bidirectional
niches and low tie rates."* This report does not alter or upgrade Family
A v2's own frozen verdict; it only notes the reassessment's own basis for
including it here.

All three source result CSVs are git-tracked, unmodified originals — see
§I for the exact SHA-256 checksums recorded and §S below confirming zero
diff after this build.

## B. Source Dataset Sizes

| Family | `per_policy_results.csv` rows | Distinct scenarios | Policies evaluated | Dense/sparse (within family) |
|---|---|---|---|---|
| A v2 | 288 | 72 | 4 (`estimated_service_time_first`, `weighted_fair_share`, `fifo`, `aging_priority`) | Dense: 72 × 4 = 288 |
| B v2 | 64 | 32 | 2 (`full_prefill`, `chunked_prefill_small`) | Dense: 32 × 2 = 64 |
| C/KV v2 | 144 | 72 | 2 (`kv_constrained_online`, `least_laxity_first`) | Dense: 72 × 2 = 144 |
| **Total** | **496** | **176** | **8 distinct policy names** | — |

All three sources report `status == "success"` on every row (0 failures in
every source's own `final_summary.json`); the builder's failure-handling
path (NaN-with-explicit-status for non-success rows) is implemented and
unit-tested but not exercised by any row in v1 — see §L.

## C. Canonical Schema

Two tables, plus schema/provenance manifests, all produced by
`src/llmserveopt/policy_separation/mf_psd.py` (`build_mf_psd`), invoked via
`scripts/build_mf_psd_v1.py`. Output directory:
`experiments/mf_psd_v1/` (see §Q for why this path is stable/non-timestamped
rather than following the usual `experiments/<name>_<timestamp>/`
convention).

### C.1 Long-form table (`mf_psd_long_v1.csv`) — one row per (source scenario, evaluated policy)

| Column | Type | Category | Meaning |
|---|---|---|---|
| `mf_psd_row_id` | str | identity | `{canonical_scenario_id}::{source_policy_name}`, globally unique |
| `canonical_scenario_id` | str | identity | `{mechanism_family}::{source_scenario_id}`, globally unique per scenario |
| `source_scenario_id` | str | identity | Raw `scenario_id` from the source CSV |
| `mechanism_family` | str | audit-only | One of `FAMILY_A_FAIRNESS_STARVATION_V2` / `FAMILY_B_PREFILL_DECODE_V2` / `FAMILY_C_KV_PRESSURE_V2` |
| `source_run_id` | str | provenance | Source experiment directory name |
| `source_result_path` | str | provenance | Repo-relative path to the frozen `per_policy_results.csv` |
| `source_row_index` | int | provenance | 0-based row index in that source CSV (exact row traceability) |
| `group_key` | str | audit-only | Seed-stripped scenario-config identity, for leakage-safe grouping (§H) |
| `seed` | str | audit-only | Raw seed value |
| `source_split_raw` | str | audit-only | `NOT_DESIGNATED` (A/B, no source-native split) or `held_out_eval_seed`/`calibration_seed` (C, from the source `held_out` column) |
| `source_policy_name` | str | identity | Raw policy name from the source CSV |
| `canonical_policy_id` | str | identity | Canonicalized policy name (identity mapping in v1 — see §E) |
| `is_canonical_anchor` | bool | audit-only | True for the 6 anchor policies (§E), False for `fifo`/`aging_priority` |
| `status` | str | audit-only | `success`/`failure`/`unknown`, passthrough |
| `primary_utility_anwg` | float | **utility** | `arrival_normalized_weighted_goodput`, NaN iff `status != success` |
| `secondary_completion_fraction` | float | **utility** | Shared canonical secondary metric (verified identical formula, §F) |
| `secondary_unweighted_slo_success_rate` | float | **utility** | Shared canonical secondary metric (verified identical formula, §F) |
| `source_row_json` | str (JSON) | provenance | Full original `per_policy_results.csv` row, verbatim, for complete fidelity |
| `source_scenario_features_json` | str (JSON) | provenance | Full original scenario-feature row (from `scenario_features.csv` for A/B, or derived from `per_policy_results.csv` itself for C — §D) |
| `builder_version` | str | provenance | `mf_psd_v1.0.0` |

### C.2 Scenario-level context table (`mf_psd_scenarios_v1.csv`) — one row per canonical scenario

Identity/audit columns: `canonical_scenario_id`, `source_scenario_id`,
`mechanism_family`, `source_run_id`, `group_key`, `seed`, `source_split_raw`,
`n_policies_evaluated`, `policies_evaluated_json`, `builder_version`.

Learnable feature columns: every column in `LEARNABLE_FEATURE_ALLOWLIST`
(§F), family-prefixed (`feat_A__*`, `feat_B__*`, `feat_C__*`), one row per
scenario, **explicit missingness** (empty string) for the two families a
given scenario does not belong to.

### C.3 Manifests

- `mf_psd_schema_v1.json` — machine-readable schema: column lists, the
  learnable allowlist, the forbidden/audit-only denylist.
- `mf_psd_provenance_v1.json` — per-source SHA-256 of every file read,
  launch git SHA/branch, audit/design doc pointers, family verdicts, plus
  SHA-256 of every output file this build produced.
- `mf_psd_build_manifest_v1.json` — row/scenario counts and the full
  validation report (§K/§L) from the build that produced the committed
  artifacts.

## D. Canonical Policy Inventory

The revised roadmap (reassessment doc, §O step 2) names six anchor
policies: `estf`, `wfs`, `full_prefill`, `chunked_prefill_small`,
`least_laxity`, `kv_constrained` — matched here to their exact canonical
identifiers:

| Roadmap shorthand | `canonical_policy_id` | Family |
|---|---|---|
| `estf` | `estimated_service_time_first` | A |
| `wfs` | `weighted_fair_share` | A |
| `full_prefill` | `full_prefill` | B |
| `chunked_prefill_small` | `chunked_prefill_small` | B |
| `least_laxity` | `least_laxity_first` | C |
| `kv_constrained` | `kv_constrained_online` | C |

Two **extra, non-anchor** policies exist only in Family A's frozen source
(`fifo`, `aging_priority` — additional baselines the original Family A v2
pilot evaluated but which the reassessment's composition-evidence table
never named as one of the three families' complementary-pair anchors).
These are real, non-fabricated observations and are preserved in the
long-form table with `is_canonical_anchor = False`; they are excluded from
`CANONICAL_ANCHOR_POLICIES` and were never run in Families B or C.

`canonical_policy_id` is currently an identity mapping (`_canonical_policy_id`
in `mf_psd.py`) — every one of the 8 distinct policy names observed is
already unique and unambiguous across all three sources, so no renaming was
needed. The function exists (rather than a bare passthrough) so a future
source with non-canonical naming has one documented place to add a real
mapping.

## E. Learnable Feature Inventory

`LEARNABLE_FEATURE_ALLOWLIST` (34 columns total, all family-prefixed,
defined in `mf_psd.py`) is the scenario-level table's **default selector
input set**:

- **Family A (9 columns, `feat_A__*`):** `target_utilization`,
  `tenant_weight_skew`, `favored_tenant_size`, `other_tenant_size`,
  `prediction_noise_sigma`, `token_length_source`,
  `size_priority_alignment`, `max_active_sequences`,
  `stress_control_relationship`.
- **Family B (20 columns, `feat_B__*`):** `hog_count`, `late_pressure`,
  `slo_emphasis`, `n_total_jobs`, `n_hog`, `n_late`, `step_token_budget`,
  `max_active_sequences`, `hog_prompt_median`, `late_prompt_median`,
  `output_median`, `late_start_s`, `slack_hog_s`, `slack_late_s`,
  `tbt_slo_s`, `arrival_shape`, `output_intervention`, `token_sources`,
  `mean_e2e_slack_hog`, `mean_e2e_slack_late`, `stress_control_relationship`.
- **Family C (3 columns, `feat_C__*`):** `bulk_pressure`,
  `urgent_arrival_phase`, `urgent_tightness`.

**Every column is family-prefixed on purpose, even where two families use
an identically-named source column** (`max_active_sequences` appears in
both A and B's `scenario_features.csv`; `stress_control_relationship` is
genuinely defined once, in the shared `PolicySeparationScenario` dataclass
in `src/llmserveopt/policy_separation/schema.py`, and used by both A and
B). Rather than silently merging these into one shared column, v1 keeps
them separate (`feat_A__max_active_sequences` vs
`feat_B__max_active_sequences`) — see §P for why this is deliberately
conservative rather than a proven inequivalence.

`mean_e2e_slack_hog`/`mean_e2e_slack_late` (Family B) deserve a specific
provenance note: they are **scenario-level aggregates of genuinely
per-request-observable quantities** (`slo_deadline - arrival_time` for each
request, both known at that request's own arrival — verified by reading
`scripts/run_policy_separation_prefill_decode_pilot_v2.py` lines ~430-448),
not simulation *outcomes*. They are legitimate context, but like every
scenario-level feature in this dataset they are a whole-trajectory
aggregate, not a strictly step-wise-online quantity — see §P.

## F. Audit-Only / Forbidden Feature Inventory

`FORBIDDEN_AUDIT_ONLY_FIELDS` (machine-readable in `mf_psd_schema_v1.json`)
explicitly denies every long-form identity/outcome column from ever
becoming a default learnable input, including:
`mf_psd_row_id`, `canonical_scenario_id`, `source_scenario_id`,
**`mechanism_family`**, `source_run_id`, `source_result_path`,
`source_row_index`, `group_key`, `seed`, `source_split_raw`,
`source_policy_name`, `canonical_policy_id`, `is_canonical_anchor`,
`status`, `primary_utility_anwg`, `secondary_completion_fraction`,
`secondary_unweighted_slo_success_rate`, `source_row_json`,
`source_scenario_features_json`, `builder_version`, `n_policies_evaluated`,
`policies_evaluated_json`.

`mechanism_family` is retained as **audit metadata** (required for
leave-one-family-out evaluation, §H) but is explicitly on the forbidden
list for default learnable inputs, per the task's anti-leakage
requirement. `LEARNABLE_FEATURE_ALLOWLIST` and `FORBIDDEN_AUDIT_ONLY_FIELDS`
are asserted disjoint by `test_learnable_allowlist_disjoint_from_forbidden_fields`.

**Canonical shared secondary metrics.** `primary_utility_anwg`,
`secondary_completion_fraction`, and `secondary_unweighted_slo_success_rate`
are promoted to shared top-level columns (rather than buried in the
per-family JSON blob) because they are computed by verifiably identical
code, not merely similarly-named columns:

- `arrival_normalized_weighted_goodput` and `completion_fraction` both come
  from the single shared `compute_metrics()` in `src/llmserveopt/core/metrics.py`
  (`RunMetrics.arrival_normalized_weighted_goodput`,
  `RunMetrics.completion_fraction`), used unmodified by all three families'
  runners.
- `unweighted_slo_success_rate` is computed inline in each of the three
  runner scripts (`run_policy_separation_fairness_starvation_pilot_v2.py`,
  `run_policy_separation_prefill_decode_pilot_v2.py`,
  `run_policy_separation_kv_pressure_pilot_v1.py`), but the formula was
  read and confirmed byte-identical in all three:
  `(len(completed) - n_violated) / max(1, n_total_requests)`.

Every other family-specific metric (`jains_fairness_index`, `mean_ttft`,
TTFT/TPOT percentiles, `hog_*`/`late_*` breakdowns, `peak_kv_utilization`,
`n_reserve_deferrals`, etc.) is preserved verbatim inside
`source_row_json` but was **not** promoted to a shared column — per the
task's explicit anti-conflation requirement, superficially similar names
(e.g. `mean_ttft` appears in both A and B) were not assumed to share
semantics without the same level of code-path verification given to the
three metrics above.

## G. Provenance Strategy

Every long-form row carries: `source_run_id`, `source_result_path`,
`source_row_index` (exact byte-traceable row), `source_row_json` (verbatim
original row), and `source_scenario_features_json` (verbatim original
scenario-feature row). `mf_psd_provenance_v1.json` records, per source: the
repo-relative path and SHA-256 of every file actually read, the launch git
SHA/branch, the audit/design doc, the family verdict, and free-text notes
on how each field was verified (see §A). It also records the build's own
git HEAD SHA, a UTC build timestamp, and SHA-256 of every output file. The
builder's own version string (`mf_psd_v1.0.0`) is embedded in every row and
in the schema/provenance manifests.

## H. Long-Form Dataset Dimensions

**496 rows × 19 columns.** Per-family: A=288, B=64, C=144 (see §B).

## I. Scenario-Table Dimensions

**176 rows × 44 columns** (10 identity/audit columns + 34 learnable feature
columns). Per-family: A=72, B=32, C=72.

## J. Row/Scenario Conservation Results

From `mf_psd_build_manifest_v1.json` (embedded validation report,
reproduced here):

```json
"long_form_validation": {
  "expected_total_rows": 496, "actual_total_rows": 496,
  "per_family_expected_rows": {"FAMILY_A_FAIRNESS_STARVATION_V2": 288,
    "FAMILY_B_PREFILL_DECODE_V2": 64, "FAMILY_C_KV_PRESSURE_V2": 144},
  "per_family_actual_rows": {"FAMILY_A_FAIRNESS_STARVATION_V2": 288,
    "FAMILY_B_PREFILL_DECODE_V2": 64, "FAMILY_C_KV_PRESSURE_V2": 144},
  "duplicate_row_ids": 0, "duplicate_scenario_policy_cells": 0,
  "non_finite_anwg_on_success_rows": 0, "untraceable_rows": 0
},
"scenario_table_validation": {
  "expected_scenarios_per_family": {"FAMILY_A_FAIRNESS_STARVATION_V2": 72,
    "FAMILY_B_PREFILL_DECODE_V2": 32, "FAMILY_C_KV_PRESSURE_V2": 72},
  "actual_scenarios_per_family": {"FAMILY_A_FAIRNESS_STARVATION_V2": 72,
    "FAMILY_B_PREFILL_DECODE_V2": 32, "FAMILY_C_KV_PRESSURE_V2": 72},
  "duplicate_scenario_ids": 0,
  "scenario_ids_only_in_long": [], "scenario_ids_only_in_scenario_table": [],
  "scenarios_with_non_invariant_features": 0
}
```

Every source row is accounted for; zero rows silently dropped; exact
scenario-count match against each source's own distinct-`scenario_id` count.

## K. Duplicates / NaN / Integrity Checks

- **Duplicate `mf_psd_row_id`:** 0.
- **Duplicate `(canonical_scenario_id, canonical_policy_id)` cells:** 0.
- **Duplicate `canonical_scenario_id` in the scenario table:** 0.
- **Non-finite ANWG on any `status == success` row:** 0 (all 496 rows are
  `success` in this build — see §B).
- **Scenario-level feature non-invariance across policy rows for the same
  scenario:** 0 scenarios (independently re-checked directly against raw
  source rows, not merely assumed from construction — see
  `validate_scenario_table` / `test_scenario_features_invariant_across_policy_rows`).
- **Forbidden/learnable field overlap:** 0 (verified both at build time and
  by a dedicated test).

**Utility distributions by family/policy** (mean ± population-std ANWG,
n=scenario count):

| Family | Policy | n | mean | std | min | max |
|---|---|---|---|---|---|---|
| A | `weighted_fair_share` | 72 | 0.7406 | 0.1776 | 0.4139 | 0.9864 |
| A | `estimated_service_time_first` | 72 | 0.7204 | 0.2283 | 0.2576 | 0.9864 |
| A | `aging_priority` | 72 | 0.5877 | 0.2401 | 0.1750 | 1.0000 |
| A | `fifo` | 72 | 0.2837 | 0.1669 | 0.0621 | 0.7250 |
| B | `full_prefill` | 32 | 0.7328 | 0.1719 | 0.3750 | 1.0000 |
| B | `chunked_prefill_small` | 32 | 0.7007 | 0.2358 | 0.3333 | 1.0000 |
| C | `kv_constrained_online` | 72 | 0.8675 | 0.1129 | 0.6176 | 1.0000 |
| C | `least_laxity_first` | 72 | 0.7597 | 0.1908 | 0.3529 | 1.0000 |

**Winner/tie structure by family**, independently recomputed from the
unified long-form table at a practical margin ε=0.01 (winner = the anchor
with the highest ANWG on that scenario among the policies that family
actually evaluated; a tie is any scenario where more than one policy is
within ε of the best):

| Family | Winner counts | Ties |
|---|---|---|
| A (4-way among `estf`/`wfs`/`fifo`/`aging`) | `wfs`=27, `estf`=23, `aging_priority`=9 | 13/72 (18.1%) |
| B (`full_prefill` vs `chunked_prefill_small`) | `full_prefill`=16, `chunked_prefill_small`=15 | 1/32 (3.1%) |
| C (`kv_constrained_online` vs `least_laxity_first`) | `kv_constrained_online`=45, `least_laxity_first`=5 | 22/72 (30.6%) |

These are broadly consistent with, but not necessarily numerically
identical to, the figures reported in each family's own audit (e.g. Family
B v2's audit reports "16/15 practical wins... near-tie 3.1%" — matches
exactly; Family C v2's audit reports "29-vs-4/48" bidirectional wins on a
specific held-out-seed subset and "tie rate 31.2%" — close to, but not
identical to, the 45-vs-5/72 and 30.6% computed here over **all** 72
scenarios including calibration seeds, since the original audit's gate
computation used a different scenario subset/methodology than this
independent full-population recomputation). This is expected and is not a
discrepancy in the underlying frozen data — see §P.

## L. Feature Missingness Summary

By construction, every `feat_<letter>__*` column is populated (non-empty)
for its own family's scenarios and explicitly empty (`""`) for the other
two families' scenarios — verified by
`test_scenario_table_feature_missingness_is_family_scoped` across all 176
scenarios × 34 feature columns. There is no *unintentional* missingness
within a family: every one of the three sources' `scenario_features.csv` (A,
B) or embedded per-row fields (C) is fully populated for every scenario in
this build.

## M. Six-Policy Coverage: Dense or Sparse

**Sparse, and honestly preserved as such — not assumed dense.** No
scenario in the frozen evidence has more than 2 of the 6 canonical anchors
evaluated on it (each family only ran its own 2 anchors on its own
scenarios); the theoretical dense 6×176 matrix would have 1,056 cells, and
only 336 of those (176 scenarios × 2 anchors each, plus A's 2 extra
non-anchor policies = 496 total long-form rows, 336 of which are
canonical-anchor rows) are actually populated. `test_six_policy_matrix_is_sparse_not_dense`
enforces `max_anchors_evaluated_per_scenario == 2` as a standing regression
guard against a future rebuild silently fabricating cross-family
evaluations.

**What Step 2 (not run here) would require to build the full dense matrix:**
running all 6 anchor policies on scenarios generated for the *other* two
families — concretely: `estimated_service_time_first` and
`weighted_fair_share` on all 32 Family-B scenarios and all 72 Family-C
scenarios; `full_prefill` and `chunked_prefill_small` on all 72 Family-A and
72 Family-C scenarios; `kv_constrained_online` and `least_laxity_first` on
all 72 Family-A and 32 Family-B scenarios. That is 4 × 176 = 704 new
policy-scenario evaluations (176 scenarios × the 4 anchors not native to
that scenario's own family), on top of the 336 already-evaluated
canonical-anchor cells, to reach a fully dense 6 × 176 matrix. This MF-PSD
build performs none of those runs.

## N. Split/Group/Holdout-Family Feasibility

**Group structure (audit metadata, not learnable):**

| Family | Scenarios | Unique seeds | Unique `group_key`s (seed-stripped config identity) | Scenarios per group |
|---|---|---|---|---|
| A | 72 | 2 (`20260816`, `20260817`) | 36 | 2 |
| B | 32 | 4 (`20260820`–`20260823`) | 8 | 4 |
| C | 72 | 6 (`20260910`–`20260915`, 2 held out) | 12 | 6 |

1. **Within-family train/validation/test:** feasible using `group_key`
   (36/8/12 distinct config groups per family respectively) — group-aware
   assignment (e.g. the existing `assign_group_aware_split` pattern in
   `src/llmserveopt/selector/dataset_v2/splits.py`) would prevent the same
   underlying scenario config's different-seed variants from being split
   across train and test. Family A and B have **no source-native split
   column** (`source_split_raw = "NOT_DESIGNATED"`); any split must be
   newly assigned by a future step, not inherited.
2. **Seed-grouped evaluation:** directly supported — `seed` and `group_key`
   are both present and audit-only (never learnable). Family C already
   carries a source-native `held_out` designation
   (`source_split_raw = held_out_eval_seed` for seeds `20260914`/`20260915`,
   12 of 72 rows; `calibration_seed` for the other 60) — a genuine
   pre-registered held-out split, unlike A/B.
3. **Leave-one-mechanism-family-out (LOMFO) evaluation:** directly
   supported — `mechanism_family` is a first-class per-row field on both
   tables, retained as audit metadata and denylisted from learnable inputs
   specifically so a future selector cannot learn to key off it as a
   feature while a LOMFO evaluation still holds an entire family out by
   this field.

**Leakage risks identified (not resolved here, for a future split-design
step to address deliberately):**

- **Repeated seeds across families are not a collision risk** — `group_key`
  is family-prefixed, so no group key can span two families.
- **Family A's small group multiplicity (2 seeds/group)** limits how much
  a within-family split can rely on "average over group" grouping;
  effectively 36 independent configuration points. A held-out-family
  evaluation is more informative here than a within-family split.
- **`stress_control_relationship` correlates with `pair_id`-style
  scenario construction** (per `schema.py`, paired "stress" vs "control"
  scenarios share `changed_parameters` by construction in the underlying
  `PolicySeparationScenario` objects, per §E). This MF-PSD build does not
  carry the original `pair_id` field (it was not present in the frozen
  result CSVs — only in the JSONL/manifest scenario objects, which are not
  part of the immutable *result* artifacts this build reads). **If a
  future step wants to enforce "never split matched stress/control pairs
  across train/test," it will need to either regenerate that pairing from
  each source's original `scenarios.jsonl` (present for A and B, absent for
  C) or accept `group_key` as a coarser, config-level proxy** — flagged
  explicitly as a known limitation (§P), not resolved.
- **Family C's `held_out_eval_seed` design was calibrated on the *v2 pilot's
  own* mechanism-timing gates**, not designed with a future selector's
  train/test split in mind — reusing it as the selector's held-out split is
  reasonable (it is a genuine pre-registered seed holdout) but was chosen
  for a different original purpose, worth reconfirming before relying on it
  for selector evaluation specifically.

## O. Tests / Checkers and Exact Results

- **Focused unit tests:** `tests/test_mf_psd_v1.py`, 31 tests, all passing:
  ```
  31 passed in 0.38s
  ```
  Covers (per task §8): exact source-row/scenario conservation, no
  duplicate canonical IDs, no duplicate scenario/policy cells, finite ANWG
  on success rows, canonical primary-metric identity, source-family
  traceability, deterministic rebuild (both the row-builder functions and
  the full `build_mf_psd` pipeline into two separate tmp dirs, byte-for-byte
  file comparison), stable/sorted ordering, learnable-feature allowlist
  well-formedness, forbidden-feature exclusion, `mechanism_family`
  explicitly excluded from learnable inputs, scenario-feature invariance
  across policy rows, checksums/provenance populated, **source artifacts
  not mutated by the build** (byte comparison of all 5 source files before
  and after `build_mf_psd`), sparse (not fabricated-dense) six-anchor
  coverage, and explicit `MFPSDValidationError` raised on injected
  duplicate-ID corruption (both tables).
- **Regression check on the pre-existing policy-separation test suite:**
  `pytest tests/ -k "policy_separation or mf_psd"` → **186 passed** (155
  pre-existing + 31 new), 0 failures — confirms this purely-additive change
  did not disturb any existing policy-separation code path (the new module
  imports nothing from and is imported by nothing in the existing
  `templates_*`/`run_*`/`analyze_*` modules).
- **Project handoff consistency checker:**
  `python scripts/check_project_handoff_consistency.py` → `project handoff
  consistency check passed`.

## P. Known Limitations

1. **Family A v2's own verdict is `USEFUL_BUT_NEEDS_REFINEMENT`, not
   `_COMPOSITION_READY`** — included per the reassessment doc's explicit
   direction (§A), but any downstream use should keep that distinction
   visible rather than treating all three families as equally "ready."
2. **Six-policy matrix is sparse (§M)** — no scenario has cross-family
   anchor coverage; this is not a defect of the build, but a real property
   of the frozen evidence that Step 2 must address by running new
   evaluations, not by any transformation of this dataset.
3. **`max_active_sequences` and `stress_control_relationship` are kept
   family-prefixed even though they may be genuinely identical-semantic
   fields between A and B** (the latter is provably shared code,
   `schema.py`'s `PolicySeparationScenario.stress_control_relationship`).
   v1 does not merge them, to avoid asserting an equivalence this task was
   not scoped to verify exhaustively (e.g. whether `max_active_sequences`
   was driven by the same `SimulatorConfig` wiring in both runners). A
   future harmonization pass could examine promoting shared-schema fields
   like `stress_control_relationship` to one canonical column with
   per-family value validation.
4. **Scenario-level features are whole-trajectory aggregates**, not
   strictly step-wise-online quantities (§E) — appropriate for the
   scenario-level top-1 selection paradigm already used by every
   composition-falsification pilot in this project (ESTF/WFS, PrefillControl,
   KV), but a future step-level online policy would need genuinely
   per-step-observable state, not scenario averages.
5. **No `pair_id`/matched-stress-control grouping carried into MF-PSD**
   (§N) — only `group_key` (seed-stripped scenario-config identity) is
   available as a grouping proxy; the original `stress`/`control` pairing
   metadata exists in Family A/B's `scenarios.jsonl` but not in the frozen
   result CSVs this build reads.
6. **Independently recomputed winner/tie figures do not numerically match
   each source family's own audit figures exactly** (§K) — expected, since
   those audits used family-specific subset/gate methodology (e.g.
   held-out-seed-only subsets, specific matched-cell comparisons) rather
   than a full-population recount; the underlying ANWG values themselves
   are unchanged (verified via checksum, §S).
7. **KV v2's historical reproducibility gap** (documented in
   `kv_v2_reproducibility_forensic_20260817.md`) is a property of
   *re-running* the KV v2 pilot, not of reading its already-frozen,
   checksummed CSV — this MF-PSD build reads that frozen CSV as immutable
   evidence and does not attempt to reproduce it.

## Q. Exact Requirements for Step 2

Per the reassessment roadmap (§O step 2: *"Unified Baseline Evaluation ...
GO: Completed utility matrix with no missing cells"*), Step 2 must:

1. Run the 4 non-native canonical anchors on each family's own scenarios
   (§M: 704 new evaluations) using the exact frozen scenario generation
   parameters already recorded in `source_scenario_features_json` /
   `params` (regenerable byte-for-byte from `(template function, params,
   seed)` per `schema.py`'s own documented contract) — **not** by
   perturbing or re-deriving scenarios.
2. Extend `mf_psd_long_v1.csv` (or build a v2 dataset that supersedes it)
   with these new rows using the *same* canonical schema (long-form
   columns, `is_canonical_anchor = True` for all of them by construction),
   preserving the existing 496 rows unchanged.
3. Re-run the same validation suite (row/scenario conservation now checked
   against the new expanded row count, e.g. 176 × 6 = 1,056 rows if fully
   dense) plus a new check that every `(canonical_scenario_id,
   canonical_anchor_policy)` cell is populated exactly once.
4. Decide, and document explicitly, whether the 2 non-anchor Family-A
   policies (`fifo`, `aging_priority`) are cross-family-evaluated too, or
   remain Family-A-only diagnostic baselines (this MF-PSD build takes no
   position on that — it only preserves them as-is).

This build's non-timestamped `experiments/mf_psd_v1/` path (§Q rationale in
§Explanation below) is intended to remain the canonical v1 location; Step 2
should produce a clearly versioned successor (e.g.
`experiments/mf_psd_v2/` or an explicit `_dense` suffix) rather than
overwriting v1 in place, so v1 remains available as a frozen sparse
baseline for comparison.

**Rationale for the non-timestamped output path.** Unlike the source pilots
(genuine simulation runs with real wall-clock/RNG execution), the MF-PSD
build is a deterministic, byte-for-byte reproducible transform of
already-frozen inputs (verified in §O). A stable path
(`experiments/mf_psd_v1/`) avoids proliferating near-duplicate timestamped
directories on every rebuild during iteration, and the artifact is versioned
by its own content (`mf_psd_provenance_v1.json` SHA-256 of every output
file) rather than by directory name.

## R. Final Dataset-Readiness Verdict

**`MF_PSD_READY`**

Justification against every criterion in the task's readiness gate:

- All three intended frozen families are incorporated (§A, §B). ✓
- Provenance is complete enough to trace every row back to its exact
  source file, row index, and launch git SHA (§C.1, §G). ✓
- Source rows/scenarios are conserved exactly, with zero silent drops
  (§J). ✓
- No unexplained duplicates exist (§K: 0 across all four uniqueness
  checks). ✓
- Learnable features are explicitly separated from audit-only/leaking
  metadata via a machine-readable allowlist/denylist, verified disjoint by
  a dedicated test (§E, §F, §O). ✓
- Output is deterministic — verified both at the row-builder level and the
  full `build_mf_psd` pipeline level, byte-for-byte (§O). ✓
- All focused validation tests pass (31/31) and the pre-existing
  policy-separation regression suite is unaffected (186/186) (§O). ✓
- No unresolved schema ambiguity would invalidate subsequent selector
  evaluation — the six-policy sparsity (§M) and the group/split limitations
  (§N) are real properties of the evidence, explicitly documented rather
  than hidden, and do not block Step 2 from being correctly scoped (§Q). ✓

## S. Frozen-Evidence Integrity Confirmation

All five source files read by this build (three `per_policy_results.csv`,
two `scenario_features.csv`) were byte-compared before and after the build
(`test_source_artifacts_not_mutated_by_build`) — identical. `git status`
after the full build-and-test cycle shows the three source run directories
(`experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/`,
`experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/`,
`experiments/kv_pressure_pilot_v2_20260817T165053Z/`) with **zero diff**.
