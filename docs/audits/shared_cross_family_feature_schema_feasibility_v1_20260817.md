# Shared Cross-Family Feature-Schema Feasibility / Redesign — v1 Audit

Date: 2026-08-17

## 0. Scope

This is a **FEATURE-SCHEMA INVESTIGATION / REDESIGN task only**, launched
from the exact next-scientific-action note at the end of
[`multifamily_contextual_selector_v1_20260817.md`](multifamily_contextual_selector_v1_20260817.md)
(`MULTIFAMILY_SELECTOR_NO_GO`): *"a feature-schema redesign task —
investigating whether a genuinely shared, cross-family feature
representation ... could let a selector demonstrate real mechanism-level
transfer without relying on family identification."*

This task did **not**: retrain the full selector suite as the main
experiment, tune ML hyperparameters, start mechanism attribution, start
composition/synthesis, build new workload families, modify any frozen
utility outcome, or alter policy/simulator semantics. The only modeling run
anywhere in this task is one unoptimized `RandomForestClassifier` used
purely as a family-identifiability diagnostic (§H), exactly as authorized.

## A. Initial Git State

Branch `contextual-compositional-heuristics-20260731`, clean working tree,
up to date with `origin/contextual-compositional-heuristics-20260731`. Tip
commit `6c31443` ("feat: run Step-3 multi-family contextual selector
experiment -- MULTIFAMILY_SELECTOR_NO_GO"). No uncommitted changes existed
before this task began.

## B. Current 33-Feature Schema Audit

`LEARNABLE_FEATURE_ALLOWLIST` in `src/llmserveopt/policy_separation/mf_psd.py`
(34 columns per the schema constant name's own docstring header count
including the family prefix accounting — 33 distinct source-column names,
9+20+3 confirmed below) is family-prefixed and structurally family-scoped
missing. Full per-column audit (source: `mf_psd.py`
`FAMILY_A/B/C_LEARNABLE_SOURCE_COLUMNS`, cross-checked against
`experiments/policy_separation_*/scenario_features.csv` headers and
`experiments/kv_pressure_pilot_v2_20260817T165053Z/per_policy_results.csv`):

| Feature | Family | Units | Online-observable | Missing outside family | Cross-family analog | Identifies family | Useful for discrimination |
|---|---|---|---|---|---|---|---|
| `feat_A__target_utilization` | A | dimensionless ratio | Yes (sweep design param) | Yes (100% missing on B/C) | No | Directly (missingness) | Yes, within A |
| `feat_A__tenant_weight_skew` | A | dimensionless | Yes | Yes | No | Directly | Yes, within A |
| `feat_A__favored_tenant_size` | A | categorical | Yes | Yes | No | Directly | Yes, within A |
| `feat_A__other_tenant_size` | A | categorical | Yes | Yes | No | Directly | Yes, within A |
| `feat_A__prediction_noise_sigma` | A | dimensionless | Yes | Yes | No | Directly | Yes, within A |
| `feat_A__token_length_source` | A | categorical | Yes | Yes | No | No | Weak (near-constant) |
| `feat_A__size_priority_alignment` | A | categorical | Yes | Yes | No | No | Yes (encodes hypothesis condition) |
| `feat_A__max_active_sequences` | A | count | Yes | Yes | **Same name in B, unverified equivalence (§P item 3, MF-PSD audit)** | Directly (name+missingness) | Weak (near-constant within A) |
| `feat_A__stress_control_relationship` | A | categorical | Yes | Yes | **Same field, shared code (`schema.py`), also used by B** | Directly (missingness) | Yes |
| `feat_B__hog_count` | B | categorical | Yes | Yes | No | Directly | Yes, within B |
| `feat_B__late_pressure` | B | categorical | Yes | Yes | No | Directly | Yes, within B |
| `feat_B__slo_emphasis` | B | categorical | Yes | Yes | No | Directly | Yes, within B |
| `feat_B__n_total_jobs` | B | count | Yes | Yes | No (A has no equivalent job-count sweep var) | Directly | Weak |
| `feat_B__n_hog` | B | count | Yes | Yes | No | Directly | Yes, within B |
| `feat_B__n_late` | B | count | Yes | Yes | No | Directly | Yes, within B |
| `feat_B__step_token_budget` | B | tokens/step | Yes | Yes | No | Directly | Weak (near-constant) |
| `feat_B__max_active_sequences` | B | count | Yes | Yes | **Same name in A, unverified equivalence** | Directly | Weak |
| `feat_B__hog_prompt_median` | B | tokens | Yes | Yes | No | Directly | Yes |
| `feat_B__late_prompt_median` | B | tokens | Yes | Yes | No | Directly | Yes |
| `feat_B__output_median` | B | tokens | Yes | Yes | No | Directly | Weak |
| `feat_B__late_start_s` | B | seconds | Yes | Yes | No | Directly | Yes |
| `feat_B__slack_hog_s` | B | seconds | Yes | Yes | No | Directly | Yes |
| `feat_B__slack_late_s` | B | seconds | Yes | Yes | No | Directly | Yes |
| `feat_B__tbt_slo_s` | B | seconds | Yes | Yes | No | Directly | Weak (near-constant) |
| `feat_B__arrival_shape` | B | categorical | Yes | Yes | No | Directly | Weak (near-constant) |
| `feat_B__output_intervention` | B | categorical | Yes | Yes | No | Directly | Weak (near-constant) |
| `feat_B__token_sources` | B | JSON blob (string) | Yes | Yes | No | Directly | Weak |
| `feat_B__mean_e2e_slack_hog` | B | seconds | Yes (per-request, arrival-known) | Yes | No | Directly | Yes |
| `feat_B__mean_e2e_slack_late` | B | seconds | Yes | Yes | No | Directly | Yes |
| `feat_B__stress_control_relationship` | B | categorical | Yes | Yes | Same shared field as A | Directly | Yes |
| `feat_C__bulk_pressure` | C | categorical | Yes | Yes | No | Directly | Yes, within C |
| `feat_C__urgent_arrival_phase` | C | categorical | Yes | Yes | No | Directly | Yes, within C |
| `feat_C__urgent_tightness` | C | categorical | Yes | Yes | No | Directly | Yes, within C |

**33 distinct columns total (9 + 20 + 3 + 1 doubled name), all
family-prefixed, all with 100%-family-scoped explicit missingness by
construction** (`mf_psd.py` `build_scenario_table_rows`: every non-native
family's columns are set to `""`). This is confirmed by the already-run
Step-3 diagnostic (`multifamily_contextual_selector_v1_20260817.md` §D):
**100.0% family-classification accuracy from the `<col>__missing`
indicators alone.** This audit does not re-run that diagnostic (it is
already frozen evidence); §H below instead runs the *equivalent* diagnostic
against the new SHARED_CORE_V1 schema, which has zero missingness by
construction.

**Family-identifying structural artifacts (this task's §C deliverable):**
The `feat_A__/feat_B__/feat_C__` prefix itself and the associated per-family
`""`-vs-populated missingness pattern are the entire mechanism by which
family becomes trivially predictable — no individual feature *value* need
be examined for a classifier keyed on missingness indicators to reach 100%
accuracy. Two column-name collisions (`max_active_sequences`,
`stress_control_relationship`) are flagged in the original MF-PSD audit as
possibly-equivalent-but-unverified; this task treats that as an open
question and does not assume equivalence — instead §D/§E below build new,
independently-defined replacements from the raw `Request`/`GPUConfig`
objects, which sidesteps the question entirely.

## C. Family-Identifying Structural Artifacts

See §B. Concretely: `test_scenario_table_feature_missingness_is_family_scoped`
(existing MF-PSD test) already asserts, as a *design property*, that every
`feat_<X>__*` column is non-empty iff the scenario belongs to family X. That
design property is precisely what a missingness-aware model (any tree-based
model, and even a linear model given `__missing` indicator columns per the
selector harness's own missing-value handling, §C of the Step-3 audit) can
exploit for perfect family recovery without learning anything about
mechanism.

## D. Candidate Shared Semantic Features — Discovery

The decisive fact that makes a genuinely shared schema possible at all:
**every one of the three families' scenario-generation templates already
builds its `PolicySeparationScenario.requests` /  `.gpu_configs` from the
exact same two shared, frozen dataclasses**, `Request` and `GPUConfig`
(`src/llmserveopt/core/types.py`), not from family-specific request types.
Verified by direct inspection of `templates_fairness_starvation_v2.py`
(`case_fairness_vs_size_v2`), `templates_prefill_decode_v2.py`
(`case_prefill_decode_ttft_contention`), and
`templates_kv_pressure_v2.py`/the Family C Reconstruction v1 artifact — all
three call the shared `req()` builder (`policy_separation/builders.py`) or
construct `Request(...)` directly with the same 8 fields:
`request_id, arrival_time, prompt_tokens, predicted_output_tokens,
actual_output_tokens, slo_deadline, priority, class_id`; and the same
`GPUConfig` with `max_active_sequences, max_batch_tokens, max_kv_tokens`
(+ unused-here disaggregation/hybrid-cache fields). `arrival_time` and
`slo_deadline` are documented to share one simulator time unit
(`STEP_SIZE = 0.001` s, confirmed identical in
`templates_fairness_starvation.py` and `templates_prefill_decode.py`; Family
C's reconstructed `arrival_time` values are the same order of magnitude and
come from the same `ServiceModel` time convention).

`actual_output_tokens` is excluded from every candidate feature: the
`Request` dataclass's own field comment states it is "hidden from online
policies," so using it in a *context* feature (even a whole-scenario
aggregate) would be observing information no real online policy has.

Category-by-category assessment against the task's candidate list:

- **LOAD/QUEUE** — available: `n_requests`, arrival-time-derived window/rate
  proxies, all computable identically from `Request.arrival_time`.
- **REQUEST SIZE** — available: `prompt_tokens`, `predicted_output_tokens`
  (never `actual_output_tokens`) admit mean/CV/combined-total statistics.
- **URGENCY/SLO** — available: `slo_deadline - arrival_time` ("slack") is
  defined identically for every request in every family.
- **PRIORITY/FAIRNESS** — available: `priority` (float) and `class_id`
  (categorical) are populated identically; CV of priority and count of
  distinct `class_id` are genuine cross-family fairness/heterogeneity
  proxies.
- **RESOURCE PRESSURE** — available: `GPUConfig.max_active_sequences` /
  `max_kv_tokens` are populated for every scenario in every family (every
  template builds exactly 1 `GPUConfig`, confirmed by direct inspection —
  no family builds a disaggregated multi-GPU scenario). Derived
  token-footprint-vs-KV and concurrency-vs-capacity ratios are legitimate,
  unit-consistent pressure proxies.
- **CONTENTION/EXECUTION MIX (prefill/decode split)** — **excluded from
  SHARED_CORE_V1.** Only Family B's `service_model_kwargs` models
  `enable_decode_prefill_contention`/chunking as a distinct mechanism; A and
  C's runners never enable it. There is no online-observable per-request
  field that would let a step-wise prefill/decode mix be computed
  identically for A/C scenarios (no chunking state exists to observe there)
  — forcing an "analog" here would fabricate a feature which is genuinely
  undefined for two of the three families, which §2 of the task explicitly
  forbids. This is a real (not artificial-schema) family difference,
  documented, not hidden.

## E. Final SHARED_CORE_V1 Feature List

17 features (`SHARED_CORE_V1_FEATURES` in
`src/llmserveopt/policy_separation/shared_context_features_v1.py`), no
family prefixes, no per-scenario missingness (every value populated for
every one of the 176 scenarios — verified, §K):

| # | Feature | Category |
|---|---|---|
| 1 | `n_requests` | Load/queue |
| 2 | `window_span_s` | Load/queue |
| 3 | `offered_rate_rps` | Load/queue |
| 4 | `mean_prompt_tokens` | Request size |
| 5 | `cv_prompt_tokens` | Request size |
| 6 | `mean_predicted_output_tokens` | Request size |
| 7 | `cv_predicted_output_tokens` | Request size |
| 8 | `mean_predicted_total_tokens` | Request size (service-time proxy) |
| 9 | `mean_slack_s` | Urgency/SLO |
| 10 | `min_slack_s` | Urgency/SLO |
| 11 | `frac_tight_slack` | Urgency/SLO |
| 12 | `priority_cv` | Priority/fairness |
| 13 | `n_distinct_request_classes` | Priority/fairness |
| 14 | `max_active_sequences` | Resource pressure |
| 15 | `max_kv_tokens` | Resource pressure |
| 16 | `token_footprint_per_kv` | Resource pressure |
| 17 | `concurrency_pressure` | Resource pressure |

No `SHARED_PLUS_MASKED_V1` extension was built — no candidate feature was
found that is genuinely shared between exactly two (not three) families with
a scientifically defensible semantic (the prefill/decode-mix category, the
only real candidate, is excluded entirely per §D, not partially masked).

## F. Feature Formulas / Units / Source

All formulas implemented in `compute_shared_context_features()`
(`shared_context_features_v1.py`), a pure function of
`(requests: Sequence[Request], gpu_configs: Sequence[GPUConfig])`:

| Feature | Formula | Unit |
|---|---|---|
| `n_requests` | `len(requests)` | count |
| `window_span_s` | `max(arrival_time) − min(arrival_time)` | seconds |
| `offered_rate_rps` | `(n−1) / window_span_s` (0 if span=0) | requests/s |
| `mean_prompt_tokens` | mean of `prompt_tokens` | tokens |
| `cv_prompt_tokens` | population std / mean of `prompt_tokens` | dimensionless |
| `mean_predicted_output_tokens` | mean of `predicted_output_tokens` | tokens |
| `cv_predicted_output_tokens` | population std / mean of `predicted_output_tokens` | dimensionless |
| `mean_predicted_total_tokens` | mean of `(prompt_tokens + predicted_output_tokens)` | tokens |
| `mean_slack_s` | mean of `(slo_deadline − arrival_time)` | seconds |
| `min_slack_s` | min of `(slo_deadline − arrival_time)` | seconds |
| `frac_tight_slack` | fraction of requests with slack `< 0.5 × median(slack)` | fraction [0,1] |
| `priority_cv` | population std / mean of `priority` | dimensionless |
| `n_distinct_request_classes` | `len({class_id})` | count |
| `max_active_sequences` | mean over `gpu_configs[i].max_active_sequences` | sequences |
| `max_kv_tokens` | mean over `gpu_configs[i].max_kv_tokens` | tokens |
| `token_footprint_per_kv` | `mean_predicted_total_tokens × n_requests / max_kv_tokens` | dimensionless ratio |
| `concurrency_pressure` | `n_requests / max_active_sequences` | dimensionless ratio |

Aggregation window: the entire scenario's request set (whole-trajectory
aggregate, matching the existing MF-PSD scenario-level paradigm — MF-PSD
audit §P item 4 already documents this same limitation for the 33-column
schema; this is not a new limitation introduced here). No normalization is
applied beyond the ratios/CVs that are dimensionless by construction — raw
means/counts/seconds are left in native units so real cross-family scale
differences remain visible for the overlap analysis (§I), not concealed by
per-feature z-scoring at build time.

**No post-outcome information anywhere**: `actual_output_tokens` is never
read (§D); no ANWG/utility/regret/policy-outcome column is read by
`compute_shared_context_features` or written to the output table.

## G. Cross-Family Computability

| Source | Requests/GPU available? | Method | Verification |
|---|---|---|---|
| Family A | `AVAILABLE_AFTER_DETERMINISTIC_REPLAY` | `mf_psd_long_v1.csv`'s recorded `(seed, params)` fed to the original `case_fairness_vs_size_v2` template function | **Verified for all 72/72 scenarios**: replayed `scenario_id` exactly matches the recorded `source_scenario_id` (a strong check, since the ID string encodes `target_utilization`/`tenant_weight_skew`/`favored_tenant_size`/`prediction_noise_sigma`/`seed` to fixed precision) |
| Family B | `AVAILABLE_AFTER_DETERMINISTIC_REPLAY` | Same pattern via `case_prefill_decode_ttft_contention` | **Verified for all 32/32 scenarios**, same exact-ID-match check, plus spot-checked `hog_prompt_median` recomputed-vs-recorded exact match |
| Family C | `AVAILABLE_ALL` | `experiments/family_c_reconstruction_v1/family_c_reconstruction_v1_scenarios.jsonl` already stores full `requests`/`gpu_configs` payloads verbatim (built by the frozen, already-audited Family C Reconstruction v1 task) — direct deserialization, no replay | **72/72 scenarios present**, 1:1 `scenario_id` match against MF-PSD's Family C rows |

No policy was re-run and no workload was regenerated to obtain this: A/B
replay calls only the pure scenario-*construction* template functions (not
any simulator/policy execution), reading a local staged BurstGPT CSV
(`.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv`, resolved via
`LLM_SERVEOPT_BURSTGPT_CSV`) rather than the original cluster path recorded
in provenance — this is the same dataset already used to build the frozen
Family C Reconstruction v1 artifact, and every one of the 104 replays
independently reproduced its own scenario's exact original ID string, which
is strong evidence the local CSV's first 20,000 rows match what was staged
on the original cluster run. All 176/176 scenarios are `AVAILABLE_ALL` or
verified `AVAILABLE_AFTER_DETERMINISTIC_REPLAY`; zero scenarios are
`NOT_AVAILABLE` or `SEMANTICALLY_INCOMPATIBLE`.

## H. Family-Predictability Diagnostic (SHARED_CORE_V1)

Simple, unoptimized `RandomForestClassifier(n_estimators=100, max_depth=6)`,
5-fold **group-aware** CV (grouped by MF-PSD's own `group_key`, so
same-config-different-seed scenarios never split across train/test):

| Metric | Value |
|---|---|
| Mean CV accuracy | **100.0%** (5/5 folds, every fold 100%) |
| Majority baseline | 40.9% (Family A/C tie at 72/176) |
| Confusion matrix | perfectly block-diagonal (72/32/72, zero off-diagonal) |

**This is the central finding of this audit.** Even with zero structural
missingness, zero family-prefixed columns, and zero mechanism_family/ID
input, family remains perfectly predictable from SHARED_CORE_V1 alone. Per
§6's required distinction:

- **(A) identifiable due to artificial schema structure**: ruled out by
  construction — there is no missingness pattern left to exploit, and every
  feature is defined once, not per-family.
- **(B) identifiable because workloads truly occupy different regions of
  shared feature space**: **confirmed as the actual cause** — §I shows
  per-feature effect sizes between families are extreme (Cohen's d in the
  tens-to-thousands for most features) and range-overlap is ≈0 for the
  large majority of features. Two features (`max_active_sequences`,
  `max_kv_tokens`) are literally **constant within each family** (each
  family's pilot fixed one GPU config for its entire sweep — e.g. Family C
  always used `max_kv_tokens=6000`, Family B always `8,000,000`, Family A's
  `generous_gpu` default `200,000`), which is itself a genuine property of
  how each pilot's experiment was designed, not an artifact of this
  schema's construction — but it does mean those two features function
  almost like a second family fingerprint, worth flagging explicitly (§P).

## I. Family Distribution Overlap

Per-feature pairwise Cohen's d and range-overlap fraction (`experiments/shared_cross_family_features_v1/shared_core_v1_diagnostics.json`, `distribution_overlap`). Representative rows:

| Feature | A vs B \|d\| | A vs B overlap | A vs C \|d\| | A vs C overlap | B vs C \|d\| | B vs C overlap |
|---|---|---|---|---|---|---|
| `mean_prompt_tokens` | 2.85 | 0.00 | 12.15 | 0.00 | 1.61 | 0.10 |
| `mean_slack_s` | 14.62 | 0.00 | 7.88 | 0.00 | 1.37 | 0.31 |
| `frac_tight_slack` | 1.78 | 0.00 | 4.74 | 0.00 | 0.97 | 0.59 |
| `offered_rate_rps` | 7.49 | 0.00 | 6.34 | 0.00 | 6.91 | 0.00 |
| `cv_predicted_output_tokens` | 8.72 | 0.00 | 0.95 | 0.47 | 3.59 | 0.00 |
| `max_active_sequences` | n/a (constant) | 0.00 | n/a (constant) | 0.00 | n/a (constant) | 0.00 |
| `n_distinct_request_classes` | n/a (both always 2) | **1.00** | n/a | **1.00** | n/a | **1.00** |

**Only `n_distinct_request_classes` overlaps fully (every family always
uses exactly 2 request classes — favored/other, hog/late, bulk/urgent).
Every other feature has range-overlap ≤ 0.59, and the large majority are
≈0.00** — the three families occupy almost entirely disjoint regions of
17-dimensional SHARED_CORE_V1 space, not merely different means with
overlapping tails. This is a property of how the three pilots' sweep
*ranges* were independently designed (each family's authors chose
parameter ranges to isolate their own mechanism), not a defect of the
feature formulas.

## J. Cross-Family Nearest-Neighbor / Utility Consistency

For every scenario, its nearest neighbor (standardized Euclidean distance in
SHARED_CORE_V1 space) among scenarios of a *different* family was found, and
the two scenarios' full 6-policy ANWG vectors (from the frozen, dense
`experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv`
— no policy re-run) were compared:

| | Nearest-neighbor pairs | Random cross-family pairs (baseline) |
|---|---|---|
| Mean Spearman corr. of 6-policy utility vector | **−0.038** | 0.197 |
| Top-1 policy agreement | **16.5%** | 27.8% |

**The nearest-neighbor pairs are not more utility-consistent than random
cross-family pairs — they are numerically worse on both metrics.** This
directly answers §8's key question: *no*, scenarios from different families
with similar SHARED_CORE_V1 vectors do **not** exhibit similar
policy-preference structure. Inspecting individual nearest-neighbor pairs
(e.g. Family A `fs2.util1.1000...s20260816` ↔ its Family-C nearest
neighbor) shows standardized distances of ≈5.3 in 17-dimensional z-space —
even the "nearest" cross-family neighbor is far away in absolute terms,
consistent with §I's near-zero overlap: there is no genuinely close
cross-family match to be found for most scenarios, so "nearest neighbor"
here is best understood as "least extremely far," not "similar."

Lightweight per-feature correlation with the 6-policy oracle max ANWG
(Spearman, `feature_vs_oracle_correlation` in the diagnostics file) shows
real but moderate signal for several features (`n_requests` ρ=−0.40,
`frac_tight_slack` ρ=−0.30, `min_slack_s` ρ=+0.29, `mean_predicted_output_tokens`
ρ=−0.28) — SHARED_CORE_V1 is not *uninformative* about utility level, it
simply does not carry enough shared cross-family structure to make
similar-context scenarios have similar policy *rankings*.

## K. Six-Policy Target-Semantics Audit

Using the same frozen dense unified utility matrix (no re-run), clustering
each family's 4 non-native policy columns by bit-exact-identical ANWG
values within that family:

| Family | Identical-value clusters among non-native policies |
|---|---|
| A | `{full_prefill, chunked_prefill_small}` (2, identical); `kv_constrained_online` (distinct); `least_laxity_first` (distinct) |
| B | `{estimated_service_time_first, kv_constrained_online, least_laxity_first, weighted_fair_share}` (**all 4, identical**) |
| C | `{full_prefill, chunked_prefill_small}` (2, identical); `estimated_service_time_first` (distinct); `weighted_fair_share` (distinct) |

Per-policy classification:

| Policy | Native family | Classification | Basis |
|---|---|---|---|
| `full_prefill` | B | **DEGENERATE_OUTSIDE_NATIVE_FAMILY** | Bit-identical to `chunked_prefill_small` on both A and C — the chunking mechanism that distinguishes them is only modeled in B's `service_model_kwargs` |
| `chunked_prefill_small` | B | **DEGENERATE_OUTSIDE_NATIVE_FAMILY** | Mirror of above |
| `estimated_service_time_first` | A | MECHANISM_SPECIFIC_BUT_EXECUTABLE | Distinct, non-degenerate on C; but part of the 4-way identical collapse on B |
| `weighted_fair_share` | A | MECHANISM_SPECIFIC_BUT_EXECUTABLE | Same pattern as ESTF |
| `kv_constrained_online` | C | MECHANISM_SPECIFIC_BUT_EXECUTABLE | Distinct, non-degenerate on A; collapses on B |
| `least_laxity_first` | C | MECHANISM_SPECIFIC_BUT_EXECUTABLE | Same pattern as `kv_constrained_online` |

**No policy is `GLOBAL_MEANINGFUL`** (uniformly distinct and non-degenerate
across all three families independently). On any Family-B scenario the
"six-policy choice" is in reality a **binary** choice (chunk or don't —
the other 4 labels are one indistinguishable action); on any Family-A or
Family-C scenario it is effectively a **5-way** choice (4 distinct actions
+ 1 collapsed prefill-pair action). A single universal 6-class classifier
target is measuring a *different number of real degrees of freedom*
depending on which family a scenario belongs to — this is a structural
problem with the target, independent of any feature representation.

## L. Alternative Target Assessment

Given §K's finding, of the five candidate reformulations named in the task:

- **(A) six-policy top-1 identity** — current target; shown incoherent (§K).
- **(B) policy family / mechanism choice** (fairness-ranking vs.
  chunk-control vs. KV-reserve, i.e. a 3-way "which mechanism dial to turn"
  target) — directly motivated by §K: the three *native* pairs
  (ESTF/WFS, full_prefill/chunked, kv_constrained/least_laxity) are exactly
  the three mechanisms that are non-degenerate on their own family. This
  collapses the semantically-broken 6-way problem into a 3-way problem that
  respects the observed degeneracy structure. **Most directly motivated by
  this audit's findings.**
- **(C) pairwise regret against a reference policy** — already tried at the
  6-policy level in Step-3 (`pairwise`, the single worst model, Step-3 audit
  §G) — no positive evidence for this direction from data already in hand.
- **(D) normalized utility residuals** — a regression reformulation;
  doesn't resolve the degeneracy problem (a model still has to predict a
  near-constant target on 4/6 columns for Family-B rows), only changes the
  loss function.
- **(E) mechanism action recommendation** ("favor short-job ranking," "use
  chunking," "use KV reserve") — essentially equivalent to (B) once the
  three mechanisms are named explicitly; the more interpretable framing of
  the same underlying reformulation.

**(B)/(E) (a 3-way mechanism-choice target) is the most defensible
direction for a future experiment** — it is the only reformulation directly
supported by evidence gathered in this task rather than by
un-tested a priori intuition (C/D were not newly evidenced here).

## M. Shared-Schema Implementation Status

**Implemented** (additive only, `mf_psd_v1/` untouched):

- `src/llmserveopt/policy_separation/shared_context_features_v1.py` —
  `compute_shared_context_features()`, pure function, `SHARED_CORE_V1_FEATURES`.
- `scripts/build_shared_cross_family_features_v1.py` — deterministic
  builder (replay for A/B, direct load for C) → 176-row table.
- `scripts/analyze_shared_cross_family_features_v1.py` — the §H/§I/§J/feature-
  correlation diagnostics (no selector training).
- `experiments/shared_cross_family_features_v1/`:
  `shared_core_v1_scenarios.csv` (176 rows × 17 learnable + 4 identity
  columns), `shared_core_v1_schema.json` (machine-readable allowlist/denylist),
  `shared_core_v1_provenance.json` (source SHA-256s, git HEAD, replay
  confirmation), `shared_core_v1_diagnostics.json` (§H–§J raw output).

Build verified **deterministic**: rebuilding into a separate directory
produced a byte-identical `shared_core_v1_scenarios.csv` and
`shared_core_v1_schema.json`. Source artifacts read
(`experiments/mf_psd_v1/mf_psd_long_v1.csv`,
`experiments/family_c_reconstruction_v1/family_c_reconstruction_v1_scenarios.jsonl`)
verified byte-unmodified after the build (checksums match the recorded MF-PSD
provenance).

## N. Tests

`tests/test_shared_cross_family_features_v1.py`, **12/12 passing**:
feature-name hygiene (no family prefix / family / scenario / utility / policy
substrings in any learnable name), exact-allowlist-membership of computed
output, hand-computed formula correctness on a synthetic 3-request scenario,
`frac_tight_slack`/`priority_cv` range sanity, **invariance to
`actual_output_tokens`** (explicit anti-leakage regression test), rejection
of empty request/GPU-config input, plus 6 build-artifact tests (skipped if
the artifact isn't present locally — requires a staged BurstGPT CSV): exact
176-row/scenario-ID match against MF-PSD, zero missing values, 100% replay
verification, schema denylist correctness, per-family row counts, and
MF-PSD source non-mutation.

## O. Feasibility Verdict

**`SHARED_FEATURE_SCHEMA_NO_GO`**

Justification against the frozen decision logic (§11): this is **not** a
case where "no genuinely comparable cross-family feature representation
exists" in the schema-definition sense — SHARED_CORE_V1 *is* a compact,
semantically consistent, replay-verified, zero-missingness, unit-consistent
17-feature representation, and that half of the investigation succeeded.
The NO_GO is triggered by the disjunctive second clause — **target/policy
semantics make universal transfer fundamentally ill-posed** (§K: no policy
is globally meaningful; 2/6 are always-degenerate outside their native
family and the other 4/6 collapse to one action specifically on Family B) —
**compounded by** an independent feature-geometry failure that on its own
would already have blocked a `READY` verdict: family remains 100%
classifiable from SHARED_CORE_V1 alone (§H) because the three families'
workloads occupy almost entirely disjoint regions of the shared feature
space (§I: range-overlap ≈0 on 15/17 features), and similarity in that space
does not predict similar policy preference (§J: nearest-neighbor
cross-family pairs are *less* utility-consistent than random pairs).

Either failure mode alone would already prevent `SHARED_FEATURE_SCHEMA_READY`;
together they leave no ambiguity. The feature-geometry failure, taken in
isolation, most resembles `SHARED_FEATURE_SCHEMA_NEEDS_MORE_DATA` (semantics
are good; overlap is the problem) — but per the task's own gate logic, the
independently-confirmed target-semantics failure (§K) is sufficient by
itself to force `NO_GO` regardless of the feature side, so `NO_GO` is the
correct single verdict, not `NEEDS_MORE_DATA`.

## P. Scientific Interpretation

1. **Was the feature-schema hypothesis (family-prefixed missingness is what
   breaks transfer) correct?** Only partially. Removing all family-prefixed
   structure and all missingness did **not** remove family-predictability
   (§H: still 100%) — the three pilots' independently-chosen sweep ranges
   put each family in its own region of any reasonable shared feature space,
   not just this specific 33-column encoding. The original schema's
   structural missingness made family-identification *free*; SHARED_CORE_V1
   shows that even without that shortcut, the underlying workload designs
   are separable on their merits.
2. **Is that separability itself evidence against transfer, or just against
   *this* selector formulation?** Genuinely ambiguous from feature geometry
   alone — but §J adds a second, independent line of evidence: even where
   two scenarios from different families *do* end up closest in shared
   space, their policy preferences are not more aligned than chance. This
   is stronger evidence than distributional separation alone.
3. **Is the six-policy target itself part of the problem?** Yes,
   demonstrably (§K) — largely independent of any feature question. Two of
   the six policies are provably indistinguishable (bit-identical ANWG)
   whenever evaluated outside Family B, and four of the six collapse
   together specifically on Family B. A selector cannot be asked to make a
   semantically meaningful 6-way choice when the ground truth itself only
   has 2–5 truly distinct options depending on which family the scenario
   happens to belong to.
4. **What would a future attempt need to change?** Both the feature-overlap
   problem (§I/§O) and the target-formulation problem (§K/§L) are real and
   independent; a future experiment addressing target formulation alone
   (§L's mechanism-choice reformulation) would still face the feature
   -overlap problem, and vice versa. Neither is fixable by re-tuning a
   selector — both require redesigning what is being asked, not how hard a
   model tries to answer it.
5. **`max_active_sequences`/`max_kv_tokens` are a subtle caveat** (§H): these
   two "resource pressure" features are literal per-family constants in the
   current frozen evidence (each pilot fixed its own GPU config for its
   entire sweep), not because the feature formula is wrong, but because no
   pilot varied its own hardware config. A future data-collection effort
   that deliberately varied GPU config *within* each family's sweep could
   make these two features genuinely informative rather than incidental
   family fingerprints — one concrete way §I's "needs more data" framing
   could partially apply to a future redesign, even though it does not
   change today's overall NO_GO (§O).

## Q. Files Changed

**New (additive only):**
- `src/llmserveopt/policy_separation/shared_context_features_v1.py`
- `scripts/build_shared_cross_family_features_v1.py`
- `scripts/analyze_shared_cross_family_features_v1.py`
- `tests/test_shared_cross_family_features_v1.py`
- `experiments/shared_cross_family_features_v1/` (4 files: scenarios CSV,
  schema JSON, provenance JSON, diagnostics JSON)
- `docs/audits/shared_cross_family_feature_schema_feasibility_v1_20260817.md` (this document)

**Confirmed unmodified (checksum-verified against recorded provenance):**
`experiments/mf_psd_v1/*` (all 3 output files' SHA-256 match
`mf_psd_provenance_v1.json`), `experiments/unified_utility_matrix_v2/*`,
`experiments/multifamily_contextual_selector_v1/*`, all three families'
frozen source run directories, every prior audit document.

## R. Commit / Push State

Committed on `contextual-compositional-heuristics-20260731` and pushed to
`origin`. No force push. See the corresponding commit for the exact SHA.

## S. Exact Single Next Scientific Action

**Not composition/synthesis, not mechanism attribution, and not a rerun of
the existing 6-policy multi-family selector.** The most directly motivated
next step, if pursued, is a **new, separately authorized experiment**
testing the §L(B)/(E) 3-way mechanism-choice reformulation (fairness-ranking
vs. chunk-control vs. KV-reserve) on top of SHARED_CORE_V1, with the
explicit awareness (from §O) that the feature-overlap problem is
independent and may still block a `GO` even if the target reformulation
alone helps. Not started here.
