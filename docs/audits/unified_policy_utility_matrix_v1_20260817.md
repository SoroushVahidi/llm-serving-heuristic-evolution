# Unified Policy-Utility Matrix (Step 2) — Build and Audit v1

Date: 2026-08-17

## 0. Scope

This audits the revised roadmap's **Step 2: Unified Baseline Evaluation**
([`reassessment_composition_hypothesis_20260817.md`](reassessment_composition_hypothesis_20260817.md)
§O), executed per the frozen preregistration
[`../design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md`](../design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md).
**Data generation and audit only.** No selector was trained, no
pairwise-regret model was fit, no mechanism attribution was performed, no
composition/synthesis work was started. MF-PSD v1
(`experiments/mf_psd_v1/`) and all three frozen family source runs remain
untouched (checksum-verified, §I).

## A. Canonical Policy Set

The six anchors named verbatim by the reassessment roadmap (§O step 2) —
`estf`, `wfs`, `full_prefill`, `chunked_prefill_small`, `least_laxity`,
`kv_constrained` — are identical to MF-PSD v1's own canonical-anchor table.
No ambiguity, no post-hoc selection. Excluded: `fifo`, `aging_priority`
(Family-A-only extra diagnostic baselines, never run in B/C, not named as
roadmap anchors — same exclusion MF-PSD v1 already documents).

## B. Cross-Family Executability Audit — Two Pre-Launch Findings

Full detail in the design doc §2. Summary:

**Finding 1 — `full_prefill`/`chunked_prefill_small` are a ServiceModel
config toggle, not a decision-function difference.** Both instantiate the
identical `GreedyArrivalPrefillControlPolicy`; their sole differentiator
(`max_prefill_chunk_tokens`) only has an observable effect when
`ServiceModel.enable_prefill_modeling=True`, which is Family B's own native
scenario config and is never set by Family A's or Family C's generators
(default `False`). Classified `VALID_WITH_DOCUMENTED_NEUTRAL_DEFAULT` on
non-native families; every such row is tagged `degenerate_mechanism=True`.

**Finding 2 — Family C / KV v2 scenario reconstruction is not byte-exact.**
Verified directly (read-only, against each family's own frozen native
cells, using each pilot's own unmodified `build_scenarios_from_config`):

| Family | Scenarios regenerated | Native cells compared | Mismatches | Max `|Δ ANWG|` |
|---|---|---|---|---|
| A v2 | 72/72 | 144 | **0** | 0 |
| B v2 | 32/32 | 64 | **0** | 0 |
| C / KV v2 | 72/72 | 144 | **99** | **0.25** |

Family C's regeneration gap independently reproduces
[`kv_v2_reproducibility_forensic_20260817.md`](kv_v2_reproducibility_forensic_20260817.md)'s
finding at the same magnitude. Because a new anchor evaluated on a
regenerated Family-C scenario would not share the same underlying request
trace as that scenario's frozen native cells, **all 4 non-native anchors ×
72 Family-C scenarios (288 cells) are classified `UNSUPPORTED` and were not
evaluated** — recorded as explicit placeholder rows, never silently
dropped.

**Net launch scope:** 416 real cells (288 on Family A + 128 on Family B),
288 explicit unsupported placeholders — not the roadmap's naive 704.

## C. Launch and Execution

- **Preregistration:** [`UNIFIED_UTILITY_MATRIX_STEP2_V1.md`](../design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md),
  committed and pushed before launch (`bda0755`).
- **Harness:** `src/llmserveopt/policy_separation/unified_utility_matrix.py`
  (reuses each family's own frozen runner script's scenario-construction
  function, imported by path, unmodified — guarantees scenario
  reconstruction is identical to what §B's reproducibility check already
  validated). CLI: `scripts/build_unified_utility_matrix_v1.py`
  (resume-safe: skips any `uum_row_id` already present in the output CSV).
- **Pre-launch tests:** `tests/test_unified_utility_matrix_v1.py`, 20 tests,
  all passing — canonical policy-set identity, expected task count (12
  family/policy pairs → 704 theoretical cells), no duplicate cells, native
  exclusion, blocked-family exclusion, policy construction (all 6 anchors),
  deterministic scenario reconstruction (both families, twice), anti-leakage
  (spied `select_action` calls carry no `scenario_id`/`mechanism_family`/
  `canonical_scenario_id` attributes), canonical metric identity, provenance
  fields, no mutation of MF-PSD v1 or frozen source artifacts (checksum
  before/after).
- **Regression check:** `pytest tests/ -k "policy_separation or mf_psd or
  unified_utility_matrix"` → 223/223 relevant tests pass (203 pre-existing +
  20 new); the run also surfaced 3 unrelated failures caused purely by
  `LLM_SERVEOPT_BURSTGPT_CSV` being set in the invoking shell (a
  pre-existing test/environment interaction in
  `test_policy_separation_fairness_starvation*.py`, confirmed to pass
  cleanly without that env var — not caused by this change, not modified by
  this change).
- **Smoke test:** 1 scenario per family × its 4 non-native anchors (8 real
  cells) — all succeeded; `full_prefill`/`chunked_prefill_small` produced
  byte-identical ANWG on the Family-A smoke scenario as predicted by
  Finding 1.
- **Launch gate:** passed — policy set resolved, all combinations classified
  VALID/VALID_WITH_DOCUMENTED_NEUTRAL_DEFAULT/UNSUPPORTED (none forced),
  design frozen, tests + smoke pass, no leakage, no frozen-artifact
  mutation.
- **tmux session:** `uum_v1_build`, command
  `python scripts/build_unified_utility_matrix_v1.py --out-dir experiments/unified_utility_matrix_v1 --workers 8`,
  launch SHA `bda0755`. Ran to natural completion in 28.8s (well inside the
  standard health-check window) — **416/416 real cells succeeded, 0
  failures**, 288 unsupported placeholders written, 704 total rows. No
  process left running.

## D. Run Integrity

- Total rows: 704 (416 `success` + 288 `unsupported_scenario_reconstruction`).
- Duplicate `uum_row_id`: **0**.
- Failures: **0** (of 416 attempted real cells).
- Non-finite (`NaN`/`Inf`) `primary_utility_anwg` on any `success` row: **0**.
- Out-of-`[0,1]` `primary_utility_anwg` on any `success` row: **0**.
- Per-(family, policy) row counts match the frozen task list exactly (72 ×
  4 for Family A, 32 × 4 for Family B, 72 × 4 unsupported for Family C —
  verified programmatically, §C).
- `mf_psd_provenance_v1.json`/`mf_psd_scenarios_v1.csv`/`mf_psd_long_v1.csv`
  SHA-256 in the build manifest match MF-PSD v1's own recorded output
  checksums exactly — confirms zero mutation.
- `git status --short` on all three frozen source run directories and
  `experiments/mf_psd_v1/`: **empty** (zero diff) both before and after the
  build.

## E. Dense Matrix Dimensions

Combining MF-PSD v1's 496 native rows with this build's 416 new valid rows
(1,056 theoretical dense cells = 176 scenarios × 6 anchors):

| Family | Scenarios | Anchors populated | Cells populated / possible |
|---|---|---|---|
| A (fairness/starvation) | 72 | **6/6 — fully dense** | 432/432 |
| B (prefill/decode) | 32 | **6/6 — fully dense** | 192/192 |
| C (KV pressure) | 72 | 2/6 — unchanged from MF-PSD v1 | 144/432 |
| **Total** | **176** | — | **768/1,056 (72.7%)** |

Wide-form matrix: `experiments/unified_utility_matrix_v1/unified_utility_matrix_wide_v1.csv`
(one row per scenario, one `anwg__<policy>` + `source__<policy>` column per
anchor, `n_anchors_populated` for quick filtering). Long-form:
`unified_utility_matrix_long_v1.csv` (704 rows, `cell_source` distinguishes
`STEP2_CROSS_FAMILY_EVALUATION` from nothing — no native rows are
duplicated into this file; join against MF-PSD v1's `mf_psd_long_v1.csv` by
`canonical_scenario_id`/`canonical_policy_id` for the full 912-row union of
populated cells... note: 496 native + 416 new = 912 total *populated
policy-family pairs counted without dedup*, but §J below counts unique
(scenario, anchor) cells, which is 768 — the 144 difference is MF-PSD's own
`fifo`/`aging_priority` extra rows, not part of the 6-anchor matrix).

## F. Winner / Tie / Oracle Analysis (dense-only: Family A ∪ Family B, n=104)

Computed directly from the unified long-form data (496 native + 416 new),
restricted to the 104 scenarios with all 6 anchors populated (Family C
excluded — still only 2/6, not meaningfully comparable across 6 policies).
Practical margin ε=0.01, matching MF-PSD v1's own convention.

**Winner counts** (a scenario can have >1 winner if tied within ε):

| Anchor | Wins | % of 104 |
|---|---|---|
| `weighted_fair_share` | 56 | 53.8% |
| `estimated_service_time_first` | 56 | 53.8% |
| `kv_constrained_online` | 33 | 31.7% |
| `full_prefill` | 17 | 16.3% |
| `least_laxity_first` | 17 | 16.3% |
| `chunked_prefill_small` | 16 | 15.4% |

Unique-winner scenarios: 67/104 (64.4%). Tie scenarios (>1 anchor within ε
of best): 37/104 (35.6%).

**Mean ANWG by anchor** (n=104): `weighted_fair_share` 0.7382,
`estimated_service_time_first` 0.7242, `kv_constrained_online` 0.7001,
`least_laxity_first` 0.4756, `full_prefill` 0.4219, `chunked_prefill_small`
0.4120.

**Oracle vs. best global fixed policy:** best fixed = `weighted_fair_share`
(mean 0.7382); oracle (per-scenario best-of-6) mean = 0.7684; **gain =
0.0303**. Per family: Family A gain = 0.0217 (best fixed
`weighted_fair_share`, 0.7406); Family B gain = **0.0494** (best fixed
`estimated_service_time_first`, 0.7328 — notably not either of Family B's
own native anchors, see §G).

**Pairwise practical-win matrix** (row beats col count, ε=0.01, n=104):

```
        estf    wfs   full  chunk    llf    kvc
estf:      -     26     72     88     53     42
 wfs:     29      -     72     88     68     52
full:      0      0      -     16     29      1
chunk:    15     15     15      -     44     16
 llf:     15      2     37     53      -      5
 kvc:     22     13     71     87     66      -
```

## G. Cross-Family Degeneracy Analysis (§B/§F's findings, quantified in full)

This is the single most important result of this build, beyond simply
filling in cells.

**G1 (pre-registered, confirmed exactly).** `full_prefill` and
`chunked_prefill_small` are byte-identical on all 72 Family-A cells (exact
match, `|Δ|<1e-9`), plus 1 incidental exact match on a Family-B scenario —
73/104 total, matching the prediction in §B Finding 1 precisely (72 forced
+ 1 coincidental).

**G2 (discovered post-hoc, not pre-registered — an honest exploratory
finding, not used to alter any scored cell).** On **all 32 of 32** Family-B
scenarios, the four non-native anchors (`estimated_service_time_first`,
`weighted_fair_share`, `least_laxity_first`, `kv_constrained_online`) are
**exactly equal to each other, and exactly equal to the native
`full_prefill`** (32/32 for every one of these five-way comparisons).
`chunked_prefill_small` is the only anchor of the six that differs from
this common value (73/104 exact-tie count above already double-counts the 1
incidental full/chunked match).

**Root cause, verified directly:** Family B's scenario generator
(`templates_prefill_decode_v2.py`) never sets `max_prefill_chunk_tokens` in
its base `service_model_kwargs` (only `enable_prefill_modeling`,
`enable_decode_prefill_contention`, `step_size`, `prefill_cost_per_token`,
`step_token_budget`, `decode_first`). Per the design doc's frozen rule
(§3: "no other cell overrides any ServiceModel field"), the four
non-prefill-variant anchors run with `ServiceModel`'s own dataclass default
`max_prefill_chunk_tokens=512` — distinct from both `full_prefill`'s
override (65536, effectively unlimited) and `chunked_prefill_small`'s
(64), but in practice indistinguishable from `full_prefill` because every
new-cell Family-B evaluation has `completion_fraction=1.0` (verified: min =
max = mean = 1.0 across all 128 new Family-B rows) — **the system is never
capacity-constrained enough, under a 512-token chunk budget and 512
max-active-sequences, for admission ORDER to matter at all.** Only
`chunked_prefill_small`'s much tighter 64-token chunk budget actually binds
and changes outcomes.

**Interpretation.** Family B's differentiating mechanism (chunk-budget
constraint under prefill/decode contention) lives on an axis that pure
admission-ranking policies (`estf`/`wfs`/`llf`/`kvc`) never touch. Cross-family
evaluation genuinely executed these policies (no leakage, no crash, no
silently-swapped identity — confirmed by the anti-leakage test) but
revealed that **on Family B specifically, 5 of the 6 canonical anchors are
behaviorally indistinguishable from a single unconstrained-admission
baseline**, and the entire family's cross-family contextual-selection
opportunity reduces to one binary contrast:
{`chunked_prefill_small`} vs. {everything else}. This is exactly the kind
of "does the expanded policy library create a meaningful contextual
problem" finding this task requires surfacing rather than hiding (§12).

**No catastrophic failures, no NaN/Inf, no completion-rate collapse**
anywhere in the 416 new cells (§D). The degeneracy in G1/G2 is a genuine
property of the mechanisms and scenarios, not a data-quality defect.

## H. Six-Policy Coverage: Now Dense for A/B, Still Sparse for C

Direct update to MF-PSD v1's §M. Family A and Family B scenarios are now
**fully dense** (6/6 anchors); Family C remains at its original native 2/6,
unchanged and untouched, blocked by the confirmed scenario-reconstruction
gap (§B Finding 2) rather than by policy invalidity.

## I. Frozen-Evidence Integrity Confirmation

- `experiments/mf_psd_v1/` (all 5 files): SHA-256 unchanged, matches the
  build manifest's recorded input checksums exactly.
- Three frozen source run directories
  (`policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/`,
  `policy_separation_prefill_decode_pilot_v2_20260817T024204Z/`,
  `kv_pressure_pilot_v2_20260817T165053Z/`): `git status --short` empty,
  both pre- and post-build.
- No historical audit, verdict, or CSV was modified.

## J. Verdict

**`UNIFIED_UTILITY_MATRIX_NEEDS_REFINEMENT`**

Justification:

- The matrix is **not** complete — 288/1,056 theoretical dense cells (all
  targeting Family C) remain unpopulated. This is a **recoverable** data
  issue (the pre-existing, separately-tracked KV v2 / BurstGPT
  reconstruction gap), not a structural incompatibility between policies
  and scenarios — hence `NEEDS_REFINEMENT` rather than
  `UTILITY_MATRIX_INVALID_FOR_UNIFIED_SELECTION`.
- Everything that *was* evaluated is scientifically sound: zero failures,
  zero leakage, zero frozen-artifact mutation, byte-exact scenario
  reconstruction verified for both families actually used, every
  degenerate/neutral-default cell explicitly tagged rather than silently
  blended in.
- Family A alone would independently justify `READY` (dense, genuine
  diversity, no unexplained degeneracy — §F/§G confirm real signal: 64.4%
  unique winners, meaningful oracle gain).
- Family B is dense but **low-diversity** by construction (§G2) — even once
  Family C is resolved, a future selector evaluation should not expect
  Family B to contribute much beyond the
  `chunked_prefill_small`-vs-everything-else contrast.
- Given the explicit instruction not to force a positive verdict, and that
  a materially large, well-understood fraction of the intended matrix is
  still missing pending a separate fix, `NEEDS_REFINEMENT` is the accurate
  label for the matrix as a whole.

## K. Remaining Blockers

1. **Family C / KV v2 scenario-reconstruction gap** (§B Finding 2) —
   unresolved, root cause not identified (per the pre-existing forensic
   audit). Resolving it is a prerequisite for the 288 currently-unsupported
   cells, not something this task attempted to fix (out of scope: data
   generation only).
2. **Family B's low cross-family diversity** (§G2) is not a blocker to
   completeness, but is a substantive caveat for whoever designs Step 3's
   selector evaluation: Family B's 6-wide coverage carries far less
   contextual-selection signal than its cell count suggests.

## L. Exact Next Steps (not started, not authorized here)

1. Investigate the Family C / KV v2 BurstGPT reconstruction gap directly
   (a dedicated task, separate from selector work) — until resolved, the
   288 Family-C cross-family cells cannot be added.
2. Once (1) is resolved, re-run `scripts/build_unified_utility_matrix_v1.py`
   (resume-safe — will only compute the remaining Family-C cells) and
   re-audit for a `READY`/`READY_LOW_DIVERSITY` verdict.
3. Independent of (1)/(2): **Step 3 — design the preregistered multi-family
   contextual-selector experiment** — not started, not authorized by this
   task.
