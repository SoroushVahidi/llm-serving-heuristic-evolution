# Hierarchical Regime Router v1 — Design / Preregistration

Date: 2026-08-17/18

## 0. Scope

**DESIGN / PREREGISTRATION ONLY.** No Stage-1 router trained, no Stage-2
selector trained, no blended-regime experiment run, no threshold tuned
from any outcome, no frozen scientific evidence modified. This document
freezes every implementation-relevant scientific choice for the next,
separately authorized experiment named by
[`cross_family_transfer_wellposedness_reassessment_20260817.md`](../audits/cross_family_transfer_wellposedness_reassessment_20260817.md)
(verdict `CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY`) and
enabled by
[`online_regime_signal_feasibility_v1_20260817.md`](../audits/online_regime_signal_feasibility_v1_20260817.md)
(verdict `ONLINE_REGIME_SIGNALS_READY`).

**Scientific question this design will let a future experiment answer:**
can `online state → regime router → regime-specific selector → policy`
beat the best global fixed policy while preserving the strong within-family
selector behavior already demonstrated (Step-3 Regime A, 0 mean regret on
Family A/B holdouts)?

## A. Prior State This Design Builds On (not re-derived here)

| Result | Verdict | What it settles for this design |
|---|---|---|
| Flat 6-policy selector | `MULTIFAMILY_SELECTOR_NO_GO` | Universal single selector is not viable — motivates hierarchy |
| Shared feature schema | `SHARED_FEATURE_SCHEMA_NO_GO` | Scenario-level whole-trajectory features don't transfer — motivates per-step online inputs instead |
| Mechanism-choice target | `MECHANISM_TARGET_NO_GO` | Naive utility-gap contrasts are confound-prone (esp. KV contrast on Family A) — motivates native-pair-only Stage-2 sets (§G) rather than reusing that target |
| Cross-family reassessment | `CROSS_FAMILY_TRANSFER_DEMOTED_HIERARCHICAL_ROUTING_READY` | Named this exact hierarchical direction as the next defensible step, with 9 conceptual gates (§L) |
| Online regime-signal feasibility | `ONLINE_REGIME_SIGNALS_READY` | Validated the exact Stage-1 inputs/labels this design freezes (§C/§D) |

## B. Design Document Path

This document: `docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md`. Companion
machine-readable gate file: `configs/hierarchical_regime_router_v1_gates.json`
(§M).

## C. Stage-1 Inputs — Frozen

Exactly four fields, all already validated online-observable in the
feasibility study (`src/llmserveopt/policy_separation/online_regime_signals_v1.py`,
`TELEMETRY_LEARNABLE_SIGNAL_COLUMNS`) — canonical names confirmed to match
exactly, no mapping needed:

| Feature | Formula (from `compute_regime_signals`) | Regime it primarily signals |
|---|---|---|
| `contention_score_v2` | `min(prefilling_count, decoding_count) / max(1, prefilling_count + decoding_count)` (active-fraction framing) | B |
| `priority_skew` | `max(priority) / max(min(priority), eps)` over `state.waiting_queue` | A |
| `kv_pressure` | `max_gpu(current_kv_tokens / max_kv_tokens)` | C |
| `queue_length` | `len(state.waiting_queue)` | A (conflict-gate), general load context |

**Explicitly denied as Stage-1 inputs** (structural, not merely a style
preference — enforced the same way `TELEMETRY_IDENTITY_COLUMNS` is kept
disjoint from `TELEMETRY_LEARNABLE_SIGNAL_COLUMNS`, tested):
`mechanism_family`, `canonical_scenario_id`/`scenario_id`, `seed`, any
future arrival, `actual_output_tokens`, any `CompletedRequest` field
(`ttft`, `tpot`, `slo_violated`, final SLO outcome), any post-run
utility/ANWG value, any oracle information. No additional feature is added
beyond these four — the feasibility study's own diagnostics (AUROC 0.779 /
0.841 / 0.993 for the three primary signals individually) did not motivate
a larger input set, and adding features "casually" is exactly what this
design freezes against.

## D. Stage-1 Target — Frozen

4-way categorical, using the feasibility study's exact preregistered
activity-label formulas (`compute_activity_labels`,
`PRIORITY_SKEW_THRESHOLD=1.05`, `MIN_CONFLICT_QUEUE=2`,
`CONTENTION_SCORE_V2_THRESHOLD=0.20`, `KV_PRESSURE_THRESHOLD=0.82` — copied
by reference, not redefined):

```
a_active  -> RANKING_FAIRNESS
b_active_v2 -> PREFILL_DECODE_CONTENTION
c_active  -> KV_MEMORY_PRESSURE
none of the above -> NONE
```

The target is never derived from `mechanism_family` — it is entirely a
function of the four Stage-1 inputs, exactly as validated in the
feasibility study (§C of that audit: `compute_activity_labels` takes only
`RegimeSignals`, never family identity).

**Tie/overlap handling (empirically not observed in the clean feasibility
telemetry, §E of that audit — 0/127,319 rows had >1 label true — but a
rule is still frozen defensively, per §4 below):** if more than one of
`a_active`/`b_active_v2`/`c_active` is simultaneously true, the row is
labeled `OVERLAP`, a fifth outcome distinct from any of A/B/C/NONE — see
§F for its routing treatment.

## E. Stage-1 Architecture

A single multiclass classifier over `{RANKING_FAIRNESS,
PREFILL_DECODE_CONTENTION, KV_MEMORY_PRESSURE, NONE, OVERLAP}` (5 classes;
`OVERLAP` present in the label space even though it was never observed in
the clean feasibility data, so the classifier and downstream routing logic
have a well-defined action for it if it does appear during training data
collection or evaluation — not something invented post-hoc after seeing a
failure). Simple model class, consistent with §H.

## F. Overlap / NONE Handling — Frozen

| Router output | Routing action | Rationale |
|---|---|---|
| `RANKING_FAIRNESS` | Route to Stage-2 Regime-A selector | Primary case |
| `PREFILL_DECODE_CONTENTION` | Route to Stage-2 Regime-B selector | Primary case |
| `KV_MEMORY_PRESSURE` | Route to Stage-2 Regime-C selector | Primary case |
| `NONE` | **Fall back to best global fixed policy** (`weighted_fair_share`, §I row A) | No regime signal justifies specializing; matches the reassessment's own finding that a good fixed policy already captures 95–98% of within-family oracle value (reassessment §D4) |
| `OVERLAP` | **Fall back to best global fixed policy** (same as NONE) | Conservative v1 choice — overlap has not been empirically validated (feasibility study found exactly 0 occurrences in clean-regime data), so no evidence supports any particular precedence ordering among A/B/C; inventing one now would be an unvalidated assumption. Explicitly NOT a soft-MoE blend — this is a hard fallback to one deployable fixed policy, nothing is blended |

This is the **entire** v1 overlap policy. No precedence ranking among
regimes is defined, and none will be inferred from future results without
a separate, explicitly authorized redesign (§13 of the task: "If overlap
appears materially in the later experiment, that should trigger a separate
reassessment rather than post-hoc architecture changes").

**Hard routing is the frozen v1 architecture** — justified by the
feasibility study's zero-overlap finding (§L of that audit), with the
explicit, already-documented caveat (carried forward, not re-litigated
here) that this may partly reflect how structurally distinct the three
frozen scenario families are rather than a general property of blended
live traffic. §P's blended-regime microcases exist specifically to probe
this caveat empirically, in a future task.

## G. Stage-2 Policy Sets — Frozen (Native-Pair-Only)

| Regime | Candidate policies | Excluded (foreign-mechanism) |
|---|---|---|
| A — RANKING_FAIRNESS | `estimated_service_time_first`, `weighted_fair_share` | `kv_constrained_online` (numerically non-degenerate on Family A per the mechanism-choice-target audit's §K clustering, but excluded — see rationale below) |
| B — PREFILL_DECODE_CONTENTION | `full_prefill`, `chunked_prefill_small` | none non-degenerate outside B in the first place (`full_prefill`≡`chunked_prefill_small` everywhere else, mechanism-choice-target audit §K) |
| C — KV_MEMORY_PRESSURE | `kv_constrained_online`, `least_laxity_first` | `estimated_service_time_first`, `weighted_fair_share` (each non-degenerate on Family C per the same clustering, excluded for the same reason) |

**Rationale for native-pair-only, stated precisely (not merely asserted):**
[`mechanism_choice_target_feasibility_v1_20260817.md`](../audits/mechanism_choice_target_feasibility_v1_20260817.md)
§D found that the `kv` mechanism contrast (`kv_constrained_online` vs.
`least_laxity_first`) is *numerically* largest on Family A specifically
because `least_laxity_first` performs poorly there for reasons unconnected
to KV pressure (Family A has essentially no KV pressure —
`token_footprint_per_kv≈0.58` — yet the largest `gain_kv` of any family,
with a within-family dose-response check confirming no real KV-pressure
relationship on Family A: ρ=−0.13, p=0.28). The same general risk applies
symmetrically to including `estimated_service_time_first`/
`weighted_fair_share` in Regime C's candidate set, or
`kv_constrained_online` in Regime A's: a policy being *numerically
non-degenerate* outside its native regime is not evidence its mechanism is
*meaningfully* relevant there — it may simply reflect that policy's
general competence level under conditions its native regime never
designed it to face. Native-pair-only avoids importing this specific,
already-documented confound into Stage-2, at the cost of not exploring
whether a cross-regime policy could occasionally help — a defensible,
conservative choice for a first hierarchical falsification, not a claim
that cross-regime policies are never useful.

`fifo` and `aging_priority` (Family A's two non-anchor extra policies,
never cross-family evaluated, `is_canonical_anchor=False` in MF-PSD) are
excluded from every regime, consistent with the existing MF-PSD/unified-
matrix convention.

## H. Stage-2 Model Formulation — Frozen

**Logistic regression**, one independent binary classifier per regime
(native pair → 2-class problem), reusing the exact model class already
shown sufficient for within-family selection: Step-3's Regime A
(`multifamily_contextual_selector_v1_20260817.md` §F) found `logreg`
achieved **0.0000 mean regret** on both Family A and Family B held-out
test scenarios (tied with `tree`/`forest`/`utility_argmax` on B) — i.e.,
the simplest model already suffices where within-family signal is known to
be strong. No model zoo, no hyperparameter grid, no neural network. If a
future implementation finds logistic regression insufficient on Regime C
specifically (§F of Step-3 showed Family C's within-family regret was
competitive but not superior to best-fixed, 0.0319–0.0417 vs. 0.0221),
that is a finding to report, not grounds to silently swap in a more
complex model without re-authorization.

## I. Baselines — Frozen (7, Deployable vs. Audit-Only)

| ID | Baseline | Deployable? | Definition |
|---|---|---|---|
| A | Best global fixed policy | Yes | Single policy, chosen from TRAIN only, applied to every scenario regardless of regime — `weighted_fair_share` per the reassessment's frozen pooled-fixed-baseline finding (reassessment §D4: mean ANWG 0.7829 pooled) |
| B | Prior flat 6-policy selector | Yes | The already-frozen `multifamily_contextual_selector_v1` artifact (`MULTIFAMILY_SELECTOR_NO_GO`) — re-evaluated on this design's own splits for a fair comparison, not retrained |
| C | Oracle regime router + oracle native-pair selector | **AUDIT ONLY** | Uses `mechanism_family` (or equivalently the true activity label, since it partitioned identically in the feasibility telemetry) to route, and the true best-of-2 native-pair policy per scenario — an upper bound on what perfect Stage-1 + perfect Stage-2 could achieve, never deployable (uses hidden information) |
| D | Learned Stage-1 router + learned Stage-2 selectors | Yes | The actual system under test |
| E | Learned Stage-1 router + regime-specific fixed-best policy | Yes | Ablation isolating Stage-2's contribution: same router as D, but each regime dispatches to that regime's own single best-fixed native policy (from TRAIN) instead of a learned selector |
| F | Hidden-family-aware selector | **AUDIT ONLY** | Uses `mechanism_family` directly as a selector input — explicitly the kind of leakage this entire redesign lineage has excluded; included only as an upper-bound reference, never a candidate for deployment |
| G | Global six-policy oracle | **AUDIT ONLY** | Per-scenario max over all 6 policies' ANWG (already computed in `unified_utility_matrix_v2`) — the absolute ceiling, ignores regime structure entirely |

## J. Train / Val / Test Split — Frozen

Reuses MF-PSD's existing `group_key` (seed-stripped scenario-config
identity, `mf_psd_long_v1.csv`/`mf_psd_scenarios_v1.csv`) as the grouping
unit for a **group-aware** split — no two seed-variants of the same
underlying scenario config may land in different splits, matching the
pattern already named (not yet implemented) in the MF-PSD audit §N and
already used by Step-3's within-family evaluation.

- **Family C**: its existing native `held_out_eval_seed` designation
  (12/72 scenarios, seeds `20260914`/`20260915`) is preserved as-is and
  assigned entirely to **TEST** — a genuine pre-registered holdout,
  reused rather than re-split.
- **Family A** (36 groups, 2 seeds each) and **Family B** (8 groups, 4
  seeds each): group-aware **60/20/20** train/val/test split by
  `group_key`, deterministic assignment via `sha256(group_key) mod 100`
  bucketed at the 60/80 percentile boundaries (same deterministic-hash
  pattern already used elsewhere in this project for reproducible
  group assignment) — **frozen split seed: none needed**, since the hash
  is a pure deterministic function of `group_key`, not a random draw; this
  is itself the "exact deterministic split procedure" required by §7 of
  the task, with no separate RNG seed to record.
- **Stage-1 and Stage-2 use the identical split boundaries** — no
  scenario/group may be TRAIN for one stage and TEST for the other.
- **TEST is never touched** for router model selection, Stage-2 model
  selection, activity-label thresholds (already frozen from the
  feasibility study, §D — not re-derived from any split), dwell-time N
  (§K), or fallback logic (§F/§L).
- Family B's small group count (8) means its val/test partitions will be
  small (≈2 groups / 8 scenarios each) — flagged honestly as a
  small-sample caveat for any future confidence interval on Family-B-
  specific metrics, not glossed over.

## K. Routing Frequency / Dwell Rule — Frozen

Routing is evaluated **every scheduling step** (the Stage-1 inputs are
already validated as genuinely step-level online quantities, §C) — but a
minimum dwell time prevents thrashing:

- **Dwell rule**: once the active regime changes to X, the router's output
  is held at X for a minimum of **N=20 raw simulator steps**
  (`sample_stride_steps` default already established in the feasibility
  study's telemetry cadence, §D of that audit — reused for consistency
  rather than inventing a new constant) before it is allowed to change
  again, *except* that a transition into `NONE`/`OVERLAP` from any active
  regime is never delayed (a safety-relevant "step back to the safe
  fallback" transition is never held back by dwell logic — only
  transitions *between* active regimes, or *into* an active regime from
  NONE/OVERLAP, are subject to the dwell minimum).
- **N is fixed at 20, not tuned from any future result** — chosen because
  it is the same cadence already validated not to have missed any
  label-transition dynamics in the feasibility study (§D/§K of that audit:
  every transition was recorded exactly regardless of stride; the stride
  only thinned steady-state recording).
- **Diagnostics to report** (not gates, informational): total transition
  count, mean transitions per scenario per regime (directly comparable to
  the feasibility study's own §K table: A=5.4, B=10.1, C=39.4 transitions/
  scenario at raw-step granularity, expected to be somewhat lower here
  once the dwell filter is applied), switching rate (transitions per
  1000 steps), dwell-violation count (should be exactly 0 by construction
  — a correctness check, not a tunable outcome).

**This is a switching-EXPERT design, not a policy-composition design.**
The router selects which Stage-2 selector (and hence which of that
regime's 2 policies) is active; it never blends two policies' actions
within one step. This is stated explicitly because the project has a
documented, separate composition/synthesis line of work
(`COMPOSITION_DEMOTED`) that this design must not be confused with or
silently reintroduce.

## L. Fallback — Frozen

Already specified in full at §F (NONE and OVERLAP both fall back to the
single best global fixed policy, `weighted_fair_share`). **No confidence-
threshold fallback in v1** — the router's discrete 5-way output
(A/B/C/NONE/OVERLAP) already has a safe default for every non-primary
case; adding a *second*, separate confidence-based fallback on top of the
primary 3 cases would introduce an additional free parameter with no
current evidence to calibrate it against, contrary to §9's preference for
the simpler design absent strong prior evidence. Baseline E (§I) already
isolates whether Stage-2 specifically is adding value versus Stage-1
routing alone, which is the more informative ablation to run first.

## M. Nine GO/STOP Gates — Frozen with Exact Thresholds

Recovered from the reassessment's conceptual §M (numbered there without
exact thresholds — this section adds those, or supplies the exact prior
threshold verbatim where reused from an earlier frozen document) — same 9
gates, no invented replacements, no watering-down for expected difficulty:

| ID | Name | Metric | Threshold | Comparison | Critical | Split |
|---|---|---|---|---|---|---|
| G1 | Online input validity | Fraction of Stage-1 inputs that are validated online-observable fields | 1.0 (100%) | `==` | **Yes** | N/A (structural/code-review gate) |
| G2 | Router quality | Macro-F1 over {A, B, C, NONE, OVERLAP} | ≥ 0.90 | `>=` | **Yes** | TEST |
| G3 | Catastrophic misrouting | Rate of A↔B↔C wrong-active-regime routing (excludes NONE/OVERLAP fallback outcomes, which are safe by construction, §F) | ≤ 0.05 | `<=` | **Yes** | TEST |
| G4 | Stage-2 preservation | Fraction of each regime's standalone within-family selector mean-regret improvement (vs. that regime's best-fixed, per Step-3 Regime A numbers) retained after full hierarchy integration | ≥ 0.90 (90%) | `>=` | **Yes** | TEST |
| G5 | Beat global fixed | Mean ΔANWG (hierarchy − best global fixed, baseline A) | > 0.01, AND bootstrap 90% CI lower bound > 0 (group-resampled, matching Step-3's own CI convention) if TEST sample size supports it | `>` | **Yes** | TEST |
| G6 | Oracle gap closure | (Hierarchy mean ANWG − best-global-fixed mean ANWG) / (regime-aware-oracle baseline C mean ANWG − best-global-fixed mean ANWG) | ≥ 0.75 (75%) | `>=` | No | TEST |
| G7 | Multi-regime benefit | Number of regimes (of A/B/C) where hierarchy's per-regime ANWG exceeds best-global-fixed's per-regime ANWG by > 0 (practical, not necessarily large) | ≥ 2 of 3 | `>=` | No | TEST |
| G8 | Interpretable / non-leaking errors | (a) zero hidden-label leakage (code-review + test gate, same discipline as G1); (b) qualitative: every major misrouting error cluster (>10% of total errors) must be attributable to Stage-1 input ambiguity (e.g. borderline `contention_score_v2`), not to any forbidden field | (a) 0 leakage instances; (b) 100% of major clusters attributable | `==` / qualitative review | **Yes** | TEST + code review |
| G9 | Robustness | (a) direction of G5's ΔANWG remains ≥ 0 on Family-C's native held-out-seed subset specifically (the one genuine pre-registered holdout already in the split, §J); (b) no catastrophic (§G3-level) misrouting rate increase on the blended-regime microcases (§P) | (a) ΔANWG ≥ 0; (b) blended-case catastrophic rate ≤ 0.10 (relaxed vs. G3's 0.05, since blended cases are explicitly out-of-distribution for this v1 design) | `>=` / `<=` | No | Family-C held-out-seed TEST subset + blended microcases |

**Critical gates: G1, G2, G3, G4, G5, G8 (6 of 9).** Non-critical: G6, G7,
G9 (informative, contribute to verdict nuance but do not alone force
`NO_GO`).

## N. Machine-Readable Gate File

`configs/hierarchical_regime_router_v1_gates.json` — same 9 gates, same
thresholds, same critical designation, plus the mechanical verdict-mapping
logic (§O), in a structured, future-implementation-consumable form. See
that file directly; not duplicated verbatim here to avoid the two ever
silently diverging (this document is the authoritative prose explanation;
the JSON is the authoritative machine-checkable encoding of the same
frozen numbers).

## O. Verdict Logic — Frozen, Mechanical

```
if G1 fails or G8(a) fails (any leakage detected):
    HIERARCHICAL_ROUTER_NO_GO   # non-negotiable safety/validity floor

elif G2 fails or G3 fails:
    HIERARCHICAL_ROUTER_NO_GO   # routing itself is not safe/accurate enough

elif G4 fails:
    HIERARCHICAL_ROUTER_NO_GO   # integration destroyed the one thing already known to work

elif G5 fails:
    if G2 passes and G3 passes and G4 passes:
        HIERARCHICAL_ROUTER_ROUTING_WORKS_SELECTION_NO_GAIN
    else:
        HIERARCHICAL_ROUTER_NO_GO

elif (blended-microcase sample too small/unstable to evaluate G9(b) meaningfully)
     or (TEST sample size insufficient for G5's CI criterion):
    HIERARCHICAL_ROUTER_INCONCLUSIVE

else:
    HIERARCHICAL_ROUTER_GO
```

`G6`, `G7`, `G9` never independently force `NO_GO` or `INCONCLUSIVE` in
this mechanical mapping — they are reported alongside the verdict as
nuance (e.g. a `GO` with `G7` at only 1/3 regimes would be reported as
`GO` with an explicit caveat that the benefit is concentrated, not evenly
distributed), consistent with the reassessment's own precedent of
reporting nuanced findings rather than collapsing everything into the
top-line label.

## P. Blended-Regime Robustness Microcases — Specified, Not Executed

Compact, hand-built diagnostic set (not a fourth family) probing whether
the feasibility study's zero-overlap finding (§L of that audit) is a
general property or an artifact of the three families' structurally
distinct configs (§L of this document already flags this as the open
question). **Specification only — building/running these is a separate,
future, explicitly authorized step.**

| Case | Source template components | Controlled intervention | Expected active labels | Hard-router failure signature |
|---|---|---|---|---|
| A+B | `case_fairness_vs_size_v2` request generation (heterogeneous `priority`) + `case_prefill_decode_ttft_contention`'s `service_model_kwargs` (`enable_prefill_modeling=True`, `enable_decode_prefill_contention=True`) | Build requests with Family-A-style priority heterogeneity (`tenant_weight_skew>1`) AND run them under a GPUConfig/ServiceModel with prefill modeling enabled and a step-token-budget tight enough to create real prefill/decode contention | `a_active=True` and `b_active_v2=True` simultaneously for a non-trivial fraction of steps | If `a_active`/`b_active_v2` never co-occur here despite both conditions being genuinely, simultaneously present by construction, that would indicate the labels are somehow mutually suppressing rather than independently detecting their own mechanism — a correctness bug, not a real "no overlap" finding, and must be distinguished from the case where they DO co-occur (real overlap exists, and the hard router's fallback-to-fixed behavior in that state needs evaluating) |
| A+C | `case_fairness_vs_size_v2` request generation (heterogeneous `priority`) + tight `GPUConfig.max_kv_tokens` (Family-C-scale, e.g. 6,000 instead of Family A's native 200,000) | Same priority heterogeneity as A+B, but combined with a small `max_kv_tokens` and enough concurrent large-prompt requests to genuinely stress KV occupancy above 0.82 | `a_active=True` and `c_active=True` simultaneously | Same correctness-vs-genuine-overlap distinction as above |
| B+C | `case_prefill_decode_ttft_contention`'s prefill/decode config + tight `GPUConfig.max_kv_tokens` | Prefill/decode contention config combined with a small `max_kv_tokens` sized so that the same cohort that creates prefill/decode contention also pushes KV occupancy above 0.82 | `b_active_v2=True` and `c_active=True` simultaneously | Same distinction |
| A+B+C (optional stress case) | All three interventions combined | Heterogeneous priority + prefill modeling + tight KV, single scenario | Potentially all three labels true at overlapping times | Same distinction; also specifically checks the `OVERLAP` routing path (§F) is exercised at least once |

For every case: only online-observable conditions are manipulated (request
priorities, `service_model_kwargs`, `GPUConfig.max_kv_tokens`) — no
scenario ID, seed, or family label is used to force the outcome, and no
threshold from §D/§E is adjusted to make overlap appear or disappear. If
overlap is found to occur non-trivially in these microcases, that is
recorded as a finding requiring a **separate, explicitly authorized
reassessment** of the hard-routing architecture (§4 of the task) — not an
invitation to redesign the fallback rule inside this same document after
the fact.

## Q. Future Metric Set

**Stage 1**: accuracy, macro-F1, confusion matrix (5×5 over
A/B/C/NONE/OVERLAP), catastrophic misroute rate (G3), NONE rate, OVERLAP
rate.

**Stage 2** (per regime): selector regret vs. that regime's own 2-policy
oracle, ε-optimal accuracy (ε=0.01, matching Step-3's own convention),
fraction of standalone within-family gain retained (G4).

**End-to-end**: canonical ANWG, regret to the global 6-policy oracle
(baseline G), Δ vs. best global fixed (baseline A, G5), oracle-gap closure
fraction (G6), per-regime ANWG and macro-regime ANWG (mean over the 3
regimes, unweighted — surfaces whether gains are evenly distributed, G7),
switching count/rate, fallback (NONE+OVERLAP) rate.

## R. Future Test Plan

Required before any future implementation is considered complete (not
implemented in this design-only task):

1. Stage-1 input allowlist is exactly the 4 fields in §C — nothing more,
   asserted via `inspect.signature` or an explicit allowlist check (same
   pattern as `test_activity_label_computation_never_reads_family_or_scenario_identity`
   in the feasibility study's own test suite).
2. `mechanism_family`/`canonical_scenario_id`/`seed` never appear in any
   Stage-1 or Stage-2 training matrix.
3. No future-arrival, `actual_output_tokens`, or post-run metric leakage
   (extend the feasibility study's own temporal-causality test pattern).
4. Activity-label formulas match §D exactly (byte-for-byte reuse of
   `compute_activity_labels`, not a redefinition) — regression test
   against the frozen feasibility telemetry.
5. Deterministic split builder: same `group_key` set in → same
   TRAIN/VAL/TEST assignment out, every time (hash-based, no RNG state).
6. Group disjointness: no `group_key` appears in more than one split.
7. Stage-2 candidate sets are exactly the native pairs in §G — a test
   that fails loudly if a third policy is ever added to any regime's
   candidate set.
8. Foreign-policy exclusion: explicit test that `kv_constrained_online` is
   never a candidate for Regime A, and `estimated_service_time_first`/
   `weighted_fair_share` are never candidates for Regime C.
9. Deterministic routing: same input state → same router output, every
   time (no unseeded randomness in the router itself).
10. Dwell-time semantics: a regime change is never accepted before N=20
    steps have elapsed since the last change, except transitions into
    NONE/OVERLAP (§K) — tested with a scripted state sequence.
11. NONE fallback: router output NONE always dispatches to baseline A's
    policy, never to any Stage-2 selector.
12. OVERLAP fallback: same as NONE (§F) — tested with a scripted state
    where >1 activity label is simultaneously true.
13. Switching diagnostics: transition/switching-rate counters match a
    hand-computed reference on a scripted trajectory.
14. All 9 metric formulas (§Q) computed correctly against hand-computed
    reference values on a tiny synthetic example.
15. All 9 gates (§M) evaluate correctly against both a synthetic
    all-pass and a synthetic all-fail input, and the mechanical verdict
    logic (§O) produces the correct one of the 4 verdicts for each of
    several scripted gate-outcome combinations.
16. Provenance: every source file the future build reads is
    checksum-recorded, same discipline as every prior artifact in this
    lineage.
17. Frozen-source immutability: none of `mf_psd_v1/`,
    `unified_utility_matrix_v2/`, `shared_cross_family_features_v1/`,
    `mechanism_choice_target_feasibility_v1/`,
    `cross_family_transfer_wellposedness_reassessment_v1/`,
    `online_regime_signal_feasibility_v1/`, or any prior audit document
    is mutated by the future implementation.

## S. Files Changed (This Task)

**New (additive only):**
- `docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md` (this document)
- `configs/hierarchical_regime_router_v1_gates.json`

**Modified:** `docs/current/RESUME_HERE.md`, `docs/current/NEXT_ACTIONS.md`
(minimal — state that the router is designed/preregistered, launch remains
unauthorized; no prior verdict rewritten).

**Confirmed unmodified:** every experiment/audit artifact listed in §17
above.

## T. Exact Single Next Scientific Action

**Not the hierarchical experiment itself.** Implementing the future test
plan (§R) and the actual Stage-1/Stage-2 training pipeline against the
frozen gates (§M) and split spec (§J) is the next step, and requires
separate, explicit authorization before any TRAIN/TEST data is touched —
not started here.
