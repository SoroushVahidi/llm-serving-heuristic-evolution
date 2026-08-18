# Hierarchical Regime Router v1 — First Held-Out TEST Scientific Evaluation

Date: 2026-08-18

**Verdict: `HIERARCHICAL_ROUTER_NO_GO`** (mechanical, computed once, not overridden).

This is the FIRST authorized held-out TEST evaluation of the design frozen in
[`HIERARCHICAL_REGIME_ROUTER_V1.md`](../design/HIERARCHICAL_REGIME_ROUTER_V1.md)
(commit `078f4f1`) and implemented in commit `2923087`
(`HIERARCHICAL_ROUTER_IMPLEMENTATION_COMPLETE`). Nothing in the design,
gate thresholds, or implementation was modified before, during, or after
this run. The evaluation ran exactly once; no result below was used to
adjust anything and re-run.

## 1. Preregistration confirmation

| Check | Result |
|---|---|
| HEAD at run time | `2923087` |
| Design doc SHA-256 | `c889e457e254183f16f79122967f253bccc756e971b55a2e2d91053ce8268f6e` (unchanged vs. commit) |
| Gates JSON SHA-256 | `b63265531a9687fa16bca5e990270427d11353fa42e00f2c5151ceb2cfc627ae` (unchanged vs. commit) |
| Implementation modules diff vs. `2923087` | none |
| Focused leakage/readiness test suite (84 tests) | all passed, re-run immediately before this evaluation |
| TEST split read by any fitting code | no — verified by code path (Stage-1 fit on `train_tel` only, Stage-2 fit on `train_by_regime` only; TEST rows first appear only after both are fit) |

`git_tree_dirty: true` at run time reflected only the not-yet-committed
evaluation script and results JSON themselves — not a change to any
frozen file (confirmed by the SHA-256 checks above).

## 2. Run provenance

- Script: `scripts/run_hierarchical_regime_router_v1_test_evaluation.py` (new, evaluation-only — does not modify any `hierarchical_regime_router_v1*` implementation module)
- Full results: `experiments/hierarchical_regime_router_v1_test_evaluation/test_evaluation_results.json`
- Ran once, locally, ~10 seconds wall time (no tmux/background needed)
- Source checksums recorded in the JSON's `preregistration_integrity` block (telemetry CSV, MF-PSD scenario table)

### Fit data

| | rows |
|---|---|
| Stage-1 TRAIN telemetry | 94,252 |
| Stage-1 TEST telemetry | 16,562 |
| Stage-2 TRAIN scenarios | 118 |
| Stage-2 TEST scenarios | 32 |

### TEST split composition — a structural finding, not a bug

```
test_regime_ground_truth_distribution: {"KV_MEMORY_PRESSURE": 24, "RANKING_FAIRNESS": 8}
```

**Family B (`PREFILL_DECODE_CONTENTION`) has zero TEST scenarios and zero
TEST telemetry rows.** Family B has only 8 groups; the frozen deterministic
hash (`sha256(group_key) mod 100`) happened to place none of them at or
above bucket 80 this time. This is exactly the caveat the design doc
itself flagged in §J ("Family B's small group count (8) means its val/test
partitions will be small... not glossed over") — it went further than a
small partition to an *empty* one here. The split builder was verified
deterministic and group-disjoint by the pre-run test suite; nothing was
adjusted to compensate. Consequence: Stage-2 Regime B and G4's Regime-B
fraction are `NOT_EVALUABLE` on this TEST split, and Stage-1's confusion
matrix has no ground-truth B or OVERLAP row to score against.

## 3. Stage-1 TEST metrics

| Metric | Value |
|---|---|
| Accuracy | 0.9922 |
| Macro-F1 (present classes: KV, NONE, RANKING only) | **0.9887** |
| Macro-F1 (all 5 classes, absent classes scored 0) | 0.5932 — reported for transparency, not used as G2's input (see note below) |
| Catastrophic misroute rate (A↔B↔C, excl. NONE/OVERLAP) | **0.0000** |
| NONE rate (predicted / true) | 0.5430 / 0.5508 |
| OVERLAP rate (predicted / true) | 0.0000 / 0.0000 |

**Confusion matrix** (rows = true, cols = predicted; only non-zero cells shown):

| True \ Pred | KV_MEMORY_PRESSURE | NONE | RANKING_FAIRNESS |
|---|---|---|---|
| KV_MEMORY_PRESSURE | 2356 | 0 | 0 |
| NONE | 130 | 8993 | 0 |
| RANKING_FAIRNESS | 0 | 0 | 5083 |

Every single TEST error (130/16,562 rows, 0.78%) is a NONE→KV_MEMORY_PRESSURE
confusion. Qualitative check (design doc G8(b) requirement): all 130 have
`kv_pressure ∈ [0.7922, 0.8195]` — **100% within 0.03 of the frozen 0.82
threshold**, a tight borderline band. Structurally, no forbidden field
could have caused this (Stage-1 only ever sees the 4 allowed columns,
mechanically enforced and test-covered). **G8(b) verdict: attributable to
Stage-1 input ambiguity near the frozen threshold, not to leakage.**

*Methodological note on macro-F1:* G2's threshold (≥0.90) was evaluated
using macro-F1 over the 3 classes actually present in TEST ground truth
(KV_MEMORY_PRESSURE, NONE, RANKING_FAIRNESS), not all 5 — scoring the
2 entirely-absent classes (PREFILL_DECODE_CONTENTION, OVERLAP) would
otherwise fold an artifact of the empty Family-B TEST partition into a
"router quality" number that has nothing to do with router quality on
those classes. The all-5-class number (0.593) is reported for
transparency but is not the number gates were evaluated against.

## 4. Stage-2 TEST metrics (standalone, given true regime)

| Regime | n | Mean regret | ε-optimal accuracy | Best fixed | Standalone gain vs. fixed |
|---|---|---|---|---|---|
| RANKING_FAIRNESS (A) | 8 | **0.0000** | 1.0 | weighted_fair_share | **+0.0302** |
| PREFILL_DECODE_CONTENTION (B) | 0 | NOT_EVALUABLE (0 TEST scenarios) | — | — | — |
| KV_MEMORY_PRESSURE (C) | 24 | **0.0000** | 1.0 | kv_constrained_online | 0.0000 (degenerate: `kv_constrained_online` is the oracle winner on all 24 TEST rows) |

Stage-2, in isolation (given the correct regime), is perfect on both
evaluable regimes. Regime C's zero standalone gain is not a Stage-2
failure — it reflects that, on this specific 24-scenario held-out-seed
subset, the native-pair oracle and the best-fixed policy coincide exactly
row-for-row (`kv_constrained_online` wins every time), leaving no gap for
any selector to close.

## 5. Seven-baseline table (mean ANWG, TEST, n=32)

| Baseline | Mean ANWG |
|---|---|
| A — Best global fixed (`weighted_fair_share`) | 0.8075 |
| B — Prior flat 6-policy selector | 0.8273 |
| C — Oracle regime router + oracle native-pair (audit) | 0.8244 |
| **D — Learned Stage-1 + learned Stage-2 (system under test)** | **0.8075** |
| E — Learned Stage-1 + regime-fixed-best (ablation) | 0.8075 |
| F — Hidden-family-aware selector (audit) | 0.8273 |
| G — Global six-policy oracle (audit) | 0.8506 |

**D = E = A exactly** on this TEST split (bit-identical means). B and F
are also bit-identical to each other.

## 6. End-to-end TEST metrics

| Metric | Value |
|---|---|
| Canonical ANWG (D) | 0.8075 |
| ΔANWG (D − A) | **0.0000** |
| Bootstrap 90% CI (group-resampled, 16 TEST groups, 5000 draws) | [0.0000, 0.0000] |
| Regret to global 6-policy oracle (G − D) | 0.0432 |
| Oracle gap closure (D−A)/(C−A) | 0.0000 |
| Per-regime ΔANWG (D−A) | RANKING_FAIRNESS: 0.0000, KV_MEMORY_PRESSURE: 0.0000 |
| Multi-regime benefit count | 0 / 2 evaluable regimes |
| Switching (total transitions, TEST telemetry) | 274 (16.5 / 1000 steps) |
| Fallback rate (TEST telemetry, per-step) | 66.5% |
| Dwell violations | **0** (independent check agrees) |

## 7. Blended-regime robustness microcases

| Case | n steps | True label mix | Active-pair steps | Catastrophic rate | Notable |
|---|---|---|---|---|---|
| A+B | 909 | NONE 888, PREFILL_DECODE_CONTENTION 20, RANKING_FAIRNESS 1 | 13 | 0.0000 | intended A-activation barely triggered |
| A+C | 1071 | NONE 1071 (100%) | 0 | n/a | **neither A nor C ever activated** — this microcase's parameters didn't cross either threshold |
| B+C | 10,056 | NONE 9886, KV 158, PREFILL_DECODE 6, **OVERLAP 6** | 80 | 0.0125 | **first empirical OVERLAP observation in this entire project lineage** (prior feasibility study: 0/127,319) |
| A+B+C (optional) | 866 | NONE 853, PREFILL_DECODE_CONTENTION 13 | 10 | 0.0000 | only B-component activated |

Weighted catastrophic rate across all 4 cases: **0.0097** (103 active-pair
comparable steps total, above the 30-step "too small" floor used here).
0 dwell violations in every case.

**Finding:** the B+C microcase is the first time OVERLAP has ever been
observed empirically in this lineage, directly answering the open
question the design doc (§L/§P) flagged — hard-routing's zero-overlap
assumption is *not* a universal property, it was an artifact of the three
frozen families being structurally distinct. When genuinely blended, the
router's fallback-to-fixed-on-OVERLAP behavior (§F) engaged correctly and
safely (catastrophic rate 0.0125, well inside the relaxed 0.10 G9(b)
threshold). A+C and A+B+C did not exercise their intended regimes — a
limitation of this task's quick microcase parameterization (built
correctly against the spec, per §5's instruction not to redesign after
the fact), not evidence against the router.

## 8. G1–G9 mechanical scoring

| Gate | Critical | Value | Threshold | Comparison | Result |
|---|---|---|---|---|---|
| G1 — Online input validity | Yes | 1.0 | 1.0 | `==` | **PASS** |
| G2 — Router quality (macro-F1) | Yes | 0.9887 | 0.90 | `>=` | **PASS** |
| G3 — Catastrophic misrouting | Yes | 0.0000 | 0.05 | `<=` | **PASS** |
| G4 — Stage-2 preservation | Yes | 0.0000 (binding: RANKING_FAIRNESS) | 0.90 | `>=` | **FAIL** |
| G5 — Beat global fixed | Yes | 0.0000 (mean); CI lower 0.0000 | 0.01 | `>` and CI>0 | **FAIL** (both sub-criteria) |
| G6 — Oracle gap closure | No | 0.0000 | 0.75 | `>=` | FAIL |
| G7 — Multi-regime benefit | No | 0 | 2 | `>=` | FAIL |
| G8 — Interpretable/non-leaking errors | Yes | (a) 0 leakage instances; (b) 100% of the one error cluster attributable to threshold ambiguity | (a) 0; (b) qualitative | `==`/review | **PASS** (a and b both hold; see §3) |
| G9 — Robustness | No | (a) ΔANWG=0.0 on Family-C held-out ≥0; (b) blended rate 0.0097 ≤ 0.10 | (a) ≥0; (b) ≤0.10 | `>=`/`<=` | PASS |

Regime B is `NOT_EVALUABLE` for G4 (0 TEST scenarios) — excluded from the
binding-minimum computation, not counted as a pass or a fail.

## 9. Mechanical final verdict

Per the frozen §O logic: G1 passes, G8(a) passes → not blocked there. G2
and G3 pass → not blocked there. **G4 fails → `HIERARCHICAL_ROUTER_NO_GO`**
(this branch fires before G5 is even consulted, per the frozen `elif`
chain — G5's independent failure is consistent with, not an additional
cause of, the verdict).

```
HIERARCHICAL_ROUTER_NO_GO
```

Not overridden. Verdict evaluator (`compute_verdict`) called exactly once.

## 10. Failure analysis

**This is not a Stage-1 competence failure.** Macro-F1 0.989, 0%
catastrophic misrouting, and the one real error mode (0.78% of rows) is
cleanly attributable to genuine threshold-boundary ambiguity in
`kv_pressure`, not a design or leakage defect.

**This is not a Stage-2 competence failure.** Given the true regime,
Stage-2 achieves exactly 0 regret against the native-pair oracle on both
evaluable regimes.

**This is an integration/measurement-methodology failure, traced to a
specific, previously-documented limitation of this task's offline
evaluation harness.** `hierarchical_router_evaluation_v1.py`'s own module
docstring states plainly: *"a scenario's end-to-end outcome is
approximated by the MAJORITY effective regime over its per-step online
telemetry... an offline scenario-level approximation... true per-step
live-simulation routing evaluation is deferred to the actual,
separately-authorized scientific run."* This TEST run is exactly where
that deferred limitation became load-bearing:

- **KV_MEMORY_PRESSURE (24 TEST scenarios):** `kv_pressure` is only
  active a *minority* of each scenario's steps (observed ranges: 4–49
  active steps out of 78–242 total, i.e. 8%–25%). A per-scenario majority
  vote over the *entire* trajectory therefore never elects
  KV_MEMORY_PRESSURE — **all 24 of these scenarios get routed to
  fallback**, identical to baseline A, regardless of Stage-1's accuracy
  at the steps where pressure genuinely is present. (Stage-2's own
  standalone gain here is 0 anyway — see §4 — so this specific regime
  had nothing to lose on this TEST subset, but the mechanism would zero
  out any regime whose activity is inherently a minority-of-steps
  phenomenon.)
- **RANKING_FAIRNESS (8 TEST scenarios):** majority-voted to
  RANKING_FAIRNESS for only 3/8 scenarios; the other 5 (including all 4
  "control", `tenant_weight_skew=1.0` scenarios, which structurally can
  *never* trigger `a_active` since 1.0 is not `> 1.05`) fell back.
  Critically, tracing Stage-2's own standalone predictions: **the 4
  scenarios where Stage-2 would have picked something other than the
  fixed policy (`estimated_service_time_first`, on the 4
  `skew=1.0`/control scenarios) are exactly the 4 scenarios Stage-1
  correctly, non-leakily never routes** (no priority heterogeneity to
  react to — this is *correct* Stage-1 behavior, not misrouting). On the
  4 `skew=10.0` stress scenarios, Stage-2 correctly identifies
  `weighted_fair_share` as the winner — but that is identical to the
  global-fixed baseline itself, so even scenarios that *did* get routed
  correctly showed zero measurable delta.

**Distinguishing the failure modes the task asked for:**
- Routing failure: no (macro-F1 0.99, 0% catastrophic).
- Selector failure: no (0 regret standalone on both evaluable regimes).
- **Integration failure: yes** — specifically, the scenario-level
  majority-vote dispatch approximation systematically under-counts
  minority-of-steps regime activity (structurally true for
  KV_MEMORY_PRESSURE) and, independently, this TEST split happens to put
  all of Stage-2's real headroom exactly on the scenarios Stage-1
  correctly declines to route.
- Insufficient oracle opportunity: yes, compounding — Regime C's
  standalone gain is 0 on these particular 24 held-out rows regardless of
  integration; Regime B has 0 TEST rows at all.
- Robustness failure: no (G9 passes; blended-microcase behavior was safe).

Gain/loss is entirely concentrated in **one regime's one sub-population**
(RANKING_FAIRNESS's control/`skew=1.0` scenarios) and is zero, not
negative, everywhere else — this is a "found nothing to gain," not a
"caused active harm," result.

## 11. Limitations

1. Family B has zero TEST representation — G4/Stage-2-B and G2's ground
   truth for B are structurally untested this run, not merely underpowered.
2. The offline scenario-level majority-vote dispatch approximation (not
   a live per-step simulation) is the primary reason G4/G5 read as flat
   zero rather than negative or positive — it is a conservative-to-neutral
   approximation of the true, not-yet-built live routing harness, not a
   measurement of what live per-step routing would actually achieve.
3. TEST is small (32 scenarios, 16 groups) — the bootstrap CI is
   degenerate ([0,0]) because D and A are bit-identical, not because of
   sample noise per se.
4. 3 of 4 blended microcases (A+B, A+C, A+B+C) did not exercise their
   intended regime combination under this task's quick parameterization.

## 12. Final readiness verdict

**`HIERARCHICAL_ROUTER_NO_GO`**

## 13. Exact single next scientific action

**Not started, not authorized by this task.** The evidence above points
at one specific, well-characterized next step: build a genuine per-step
live-simulation evaluation harness (replacing the scenario-level
majority-vote offline approximation) so that G4/G5 measure what the
router actually does moment-to-moment rather than a single dispatched
policy per whole scenario — and, if repeated, a TEST-split re-draw or a
larger Family-B allocation would be needed to get any Regime-B TEST
signal at all. Building that harness, and any consequent re-evaluation,
requires separate, explicit authorization, per this task's own stop
condition.
