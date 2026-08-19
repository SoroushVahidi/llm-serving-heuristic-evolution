# Family-B-Balanced Replication v1 — Preregistration Completion

Date: 2026-08-19

## 0. Scope and Status

**DESIGN / PREREGISTRATION COMPLETION + READINESS PREPARATION ONLY.**

This document completes the "Family-B-Balanced Replication Design" already
frozen in Section 10 of
[`HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1.md`](HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1.md)
(commit `6c9ec36`), which specified the required properties (12/12/12
family balance, group-disjoint, no retuning) but not the exact mechanical
scenario-selection procedure. It does **not** authorize, launch, or report
any scientific result. No Stage-1 router is retrained, no Stage-2 selector
is retuned, no scenario in this replication set has been evaluated with
the live harness, and no ANWG/utility outcome was read to produce this
document.

The primary live re-evaluation result
(`HIERARCHICAL_ROUTER_NO_GO`, formally scored, documented in
[`hierarchical_regime_router_live_reeval_v1_20260818.md`](../audits/hierarchical_regime_router_live_reeval_v1_20260818.md))
and all frozen thresholds, models, and harness code remain unchanged and
are not re-examined here.

## 1. Scientific Question (Restated From SS 10, Unchanged)

> When prefill/decode contention (Family B) is represented in a genuinely
> held-out evaluation set, does the frozen live hierarchical router obtain
> practically meaningful value from the Family-B path?

This is a **replication of representation**, not a new router. Every
element in Section 2 of the parent design doc ("Frozen Elements") applies
identically here: Stage-1 inputs/model/thresholds, Stage-2 native pairs,
dwell=20, NONE/OVERLAP/no-confidence fallback to `weighted_fair_share`,
and the live closed-loop harness are all reused exactly as-is. No
threshold, feature, model, or fallback rule is altered for this
replication.

## 2. Exact Frozen 36-Scenario Selection

**Selection is 100% metadata-driven — no ANWG/utility/policy-performance
column was read to produce it.** The only inputs are `mf_psd_scenarios_v1.csv`'s
`mechanism_family`, `group_key`, `canonical_scenario_id`, `seed` columns
and each scenario's train/val/test split assignment from the already-frozen,
unmodified `build_splits()` (`src/llmserveopt/policy_separation/hierarchical_regime_router_v1.py`).

Implementation: `src/llmserveopt/policy_separation/family_b_balanced_replication_v1.py`.
Frozen output (materialized by `scripts/freeze_family_b_balanced_replication_v1_scenario_set.py`,
idempotent/deterministic): `experiments/family_b_balanced_replication_v1/frozen_scenario_selection_v1.json`.

### A. Family A (`RANKING_FAIRNESS`) — 12 scenarios

Pool = all Family-A rows with `split != "train"` (18 rows: 10 VAL + 8 TEST,
all of Family A's 36 groups split cleanly 27 train / 5 val / 4 test).
Selection: prefer VAL rows (never observed by any prior evaluation) over
TEST rows (already scored in the primary re-evaluation), each tier sorted
by `canonical_scenario_id` ascending, first 12 taken. **Result: 10 VAL + 2
TEST** — minimizes, without eliminating (VAL alone has only 10 rows),
overlap with the primary 32-scenario TEST set.

### B. Family B (`PREFILL_DECODE_CONTENTION`) — 12 scenarios

Pool = all Family-B rows with `split == "val"` (16 rows across the 4 of
Family B's 8 total groups that landed in VAL under the primary split;
the other 4 groups are in TRAIN; **0 are in TEST**, the already-documented
primary-split gap this replication exists to address). Selection: sorted
by `canonical_scenario_id` ascending, first 12 of 16. **Result: 12 VAL, 0
overlap with the primary TEST set** (impossible, since primary TEST has
zero Family-B rows).

### C. Family C (`KV_MEMORY_PRESSURE`) — 12 scenarios

Family C is **seed-partitioned, not group-partitioned** (`build_splits`
docstring; the same `group_key` archetype legitimately appears in both
TRAIN and TEST via different seeds — verified scenario-row-level, not
group-level, disjointness below). Pool = all Family-C rows with
`split == "test"` restricted to `seed == 20260914` — the **first of the
two already-frozen** Family-C held-out evaluation seeds named in
`configs/hierarchical_regime_router_v1_gates.json`'s
`splits.family_c_held_out.seeds` (`["20260914", "20260915"]`), not a new
ad hoc rule. This yields exactly one row per Family-C held-out group (12
groups → 12 rows). **Result: all 12 rows are already part of the primary
24-scenario Family-C TEST allocation** — Family C has no VAL rows at all,
so this is the only genuinely never-used-in-TRAIN pool available for it.

### D. Held-Out Integrity (Verified, Not Assumed)

`verify_no_train_leakage()` checks **scenario-row-level** (not
group-level) disjointness between the 36-scenario selection and the
primary TRAIN allocation: **empty overlap, confirmed**. (Group-key overlap
between the selection and TRAIN is expected and correct for Family C only,
per its seed-based partitioning — this is a property of the design, not a
leak; see `select_family_c_replication`'s docstring.) Since Stage-1/Stage-2
were fit only on `split == "train"` rows (`Stage1Router().fit(train_tel)`,
unchanged from both prior evaluation scripts), no scenario in this
replication set was used to fit the models it will be evaluated against.

### E. Summary Table

| Family | n | Source split(s) | Overlap with primary TEST | Overlap with primary TRAIN |
|---|---|---|---|---|
| A (`RANKING_FAIRNESS`) | 12 | 10 VAL + 2 TEST | 2/12 | 0/12 |
| B (`PREFILL_DECODE_CONTENTION`) | 12 | 12 VAL | 0/12 | 0/12 |
| C (`KV_MEMORY_PRESSURE`) | 12 | 12 TEST (seed 20260914 only) | 12/12 | 0/12 |
| **Total** | **36** | | 14/36 | **0/36 (verified)** |

Family C's full overlap with primary TEST is a structural consequence of
Family C having zero VAL rows, not a selection choice — it is disclosed
here, not hidden, and the replication report must not describe the Family-C
leg as independent evidence from the primary evaluation's Family-C result.
Family A and Family B legs are substantially or fully independent of the
primary TEST evaluation.

## 3. Models

Reused exactly, out-of-the-box, with **zero retuning**: Stage-1
(`Stage1Router().fit(train_tel)`) and Stage-2 (`fit_all_stage2_selectors(train_by_regime)`),
fit on the identical primary-split `train_tel`/`train_by_regime` used by
both prior evaluations (`21bfff1`'s approximate evaluation and the primary
live re-evaluation). Both prior scripts independently reproduced
bit-identical Stage-1 macro-F1 (0.9886585377383629) from this exact fit
procedure, confirming it is fully deterministic — refitting for the
replication run via the same code path on the same TRAIN data is
equivalent to reusing a persisted checkpoint, not retraining.

## 4. Metrics, Gates, Verdict Logic

Identical formulas and thresholds to the primary re-evaluation (Sections 6
and 8 of the parent design doc): mean ANWG live vs. approximate vs. best
global fixed vs. oracle, `delta_method`/`delta_fixed` with 90% group-
resampled bootstrap CIs, Stage-1 accuracy/macro-F1/catastrophic-misroute,
per-regime Stage-2 regret/epsilon-optimal-accuracy, routing/dwell/switching
dynamics. G1–G9 will be scored via the same canonical
`evaluate_all_gates`/`compute_verdict` implementation used for the formal
rescoring of the primary result — **not** a hand-rolled substitute (the
exact gap the primary re-evaluation's formal rescoring closed).

**Distinct verdict namespace**, to prevent any confusion with the primary
result: `FAMILY_B_BALANCED_REPLICATION_{SUPPORTS_HIERARCHY,
IMPROVES_METHOD_BUT_NO_END_TO_END_GAIN, CONFIRMS_NO_GO, INCONCLUSIVE}`,
computed with the same threshold logic as the parent doc's Section 8. Per
the parent design's "Strict Separation Rule" (SS 10B), this replication's
results and verdict must be reported in a wholly separate section/artifact
and **never merged or blended** into the primary 32-scenario result.

## 5. Pre-Launch Verification Plan (To Run Before Any Authorized Launch)

Mirrors the parent design's Section 13, scoped to this replication:

1. **Selection identity**: assert the 36 scenarios materialized at launch
   time exactly match `frozen_scenario_selection_v1.json`'s frozen IDs
   (byte-identical, via checksum).
2. **TEST/replication holdout**: assert none of the 36 replication
   scenarios was in the primary-split TRAIN allocation (Section 2D, already
   verified at freeze time; re-verified at launch time as a guard against
   drift).
3. **Model immutability**: Stage-1/Stage-2 refit from the identical TRAIN
   data reproduces the same macro-F1/hash as the primary re-evaluation.
4. **Harness integrity**: `tests/test_hierarchical_router_live_harness_v1.py`
   passes in full.
5. **No majority-vote leakage**: the replication runner never imports the
   majority-vote evaluation path (same AST guard pattern as
   `tests/test_hierarchical_router_live_harness_v1.py`).
6. **Forced-parent equivalence**: reused from the harness's own test suite,
   not re-derived here.
7. **Dwell/fallback constancy**: `dwell=20`, fallback→`weighted_fair_share`
   unchanged (asserted, not re-specified).
8. **Deterministic replay**: one scenario run twice produces bit-identical
   ANWG and trajectory.
9. **Frozen-artifact protection**: git diff check confirms no changes to
   `configs/hierarchical_regime_router_v1_gates.json`,
   `docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md`,
   `docs/design/HIERARCHICAL_REGIME_ROUTER_LIVE_REEVAL_V1.md`, or any
   `docs/audits/hierarchical_regime_router*.md` file.

## 6. Standing Long-Running Job Rule (Unchanged From Parent Design SS 14)

Any future authorized launch of the actual 36-scenario scientific
evaluation must run in a dedicated named tmux session, with at most a
3-minute initial health check before yielding control, exactly as the
parent design specifies. **This document does not authorize that launch.**

## 7. Explicit Non-Authorization

This document freezes the design and readiness tooling only. Launching the
scientific evaluation of the 36-scenario replication set — i.e., running
the live harness against
`experiments/family_b_balanced_replication_v1/frozen_scenario_selection_v1.json`'s
scenarios and inspecting the resulting ANWG/gate outcomes — requires
separate, explicit authorization and is **not started by this document or
any artifact it references**.
