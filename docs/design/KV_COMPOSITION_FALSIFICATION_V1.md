# KV-Aware Composition Falsification v1

**Date:** 2026-08-17
**Status:** PREREGISTERED — frozen before any TEST/OOD score is inspected.
**Predecessor:** [`POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md`](POLICY_SEPARATION_FAMILY_KV_PRESSURE_V2.md)
(verdict `KV_FAMILY_COMPOSITION_READY`), audit
[`family_c_kv_pressure_pairwise_separation_v2_20260817.md`](../audits/family_c_kv_pressure_pairwise_separation_v2_20260817.md).
**Parents (unmodified):** `kv_constrained_online`, `least_laxity_first`.
**Precedent this mirrors:** Family B v2 PrefillControl composition falsification
(`docs/audits/family_b_v2_prefill_control_composition_falsification_20260817.md`),
verdict `SELECTION_SUFFICIENT_FOR_THIS_PAIR`.

This is the first composition falsification attempted for a pair whose
pairwise-separation pilot reached `_COMPOSITION_READY`. The central question
is **not** "can we predict which fixed parent wins a scenario" (already
answered — a selector could do that) but **"can a state-dependent child
exploit changes in online KV pressure *within* a single trajectory to
outperform both fixed parents and expand their held-out envelope?"**

## 1. Pre-composition audit of the frozen v2 evidence

### 1A. Exact v2 factors and held-out structure

`bulk_pressure` (low/high) × `urgent_arrival_phase` (early/middle/late) ×
`urgent_tightness` (loose/tight) × 6 seeds = 72 scenarios. In-sample seeds
(gates): `20260910-13`. Held-out seeds (G6 only): `20260914-15`.
`max_kv_tokens=6000`.

### 1B. Exact G1-G10 composition-readiness evidence

All 10 gates passed (v2 audit §H): bidirectional 29-vs-4/48 (G1), tie rate
31.2% (G2), seed-stable timing direction 3/4 (G3), reserve activation 48/48
(G4), timing/pressure interaction both bulk levels (G5), held-out replication
*stronger* than in-sample (G6), action disagreement co-occurs with every
nonzero delta (G7), LLF wins a nonzero count (G8), 0 failures/leakage (G9),
6/16 matched cells flip practical winner by phase alone (G10).

### 1C. Measured within-scenario timing/occupancy evidence (v2, frozen)

The v2 audit's own §D explicitly found **peak KV utilization is not a usable
regime differentiator — both `bulk_pressure` levels saturate ~0.97-1.10 once
any backlog exists.** v2 relied on *reserve-deferral counts*, not occupancy,
as its pressure-regime diagnostic.

### 1D/1F. Online-observable variables associated with parent disagreement /
### states where reserve admission differs from LLF

**New empirical work done in this task** (not in the frozen v2 CSV, which
only records scenario-level aggregates): re-simulated several v2 scenarios
directly (`kvp2.bulk24.phase{middle,late}.tighttight.s{20260911,20260912,20260913}`,
using the unmodified frozen `case_kv_pressure_reserve_contention_v2` /
`KVConstrainedOnlinePolicy` / `LeastLaxityFirstPolicy`, staged BurstGPT at
`.local_data/burstgpt_v2/`) and extracted `sim._gpus[0].step_kv_used`
(already-existing simulator instrumentation, no new mechanism) plus
per-request admission times and slack-based "urgent" classification (reusing
`KVConstrainedOnlinePolicy`'s own `urgent_laxity_seconds=0.25` /
`laxity_seconds()` helper — no new mechanism).

Findings:

1. **Confirmed, first-hand:** KV occupancy is high (~83-100%) across the
   *entire* trajectory in the signal regime, not just late — v2's own §D
   finding replicates. Raw `kv_util_ratio` therefore does **not** vary enough
   within one trajectory to serve as a useful switch signal.
2. **New finding:** the count of currently-*waiting* requests classified
   urgent by `KVConstrainedOnlinePolicy`'s own threshold
   (`laxity_seconds(...) <= 0.25`) genuinely varies *within a single
   trajectory* — measured directly on one scenario
   (`kvp2.bulk24.phasemiddle.tighttight.s20260912`): 0 at 16% of steps,
   ranging up to 13, with real mass across the whole 0-13 range (not
   bimodal-degenerate). This is a real, non-saturated, per-step observable.
3. **New finding — the actual mechanism:** `KVConstrainedOnlinePolicy._score`
   sorts on `(laxity > urgent_laxity_seconds, kv_cost/priority, laxity, id)` —
   a **strict two-tier** sort where every genuinely-urgent request
   (laxity ≤ 0.25s) is placed ahead of *every* non-urgent request regardless
   of the non-urgent request's own numeric laxity. `LeastLaxityFirstPolicy`
   sorts on continuous laxity alone, mixing urgent and bulk. When a bulk
   tenant's own laxity happens to be numerically lower than an urgent
   tenant's laxity at a given instant (both are waiting, one KV slot frees
   up), LLF can serve the bulk tenant first; `kv_constrained_online` never
   can. This — not the reserve/`target_kv_utilization` admission gate in
   isolation — is the dominant differentiator, and it is a genuinely
   per-arrival, per-instant property, not a scenario-level one.
4. **Direct within-trajectory heterogeneity, not just between-scenario
   averages:** ordering each scenario's 10 urgent requests by their own
   arrival time and recording per-request SLO success
   (1=met, 0=missed) in six re-simulated scenarios:

   | seed | phase | `kv_constrained` sequence | `least_laxity_first` sequence |
   |---|---|---|---|
   | 20260911 | middle | `1,1,1,1,1,0,1,0,0,1` | `0,0,0,0,0,0,0,0,0,0` |
   | 20260911 | late   | `1,1,0,1,1,0,1,0,0,1` | `0,0,0,0,0,0,1,0,0,0` |
   | 20260912 | middle | `1,1,1,1,0,1,1,1,0,0` | `1,0,0,0,0,0,0,0,1,0` |
   | 20260912 | late   | `1,0,1,0,0,1,1,1,0,1` | `0,0,0,0,0,1,0,0,1,0` |
   | 20260913 | middle | `1,1,0,0,1,1,1,0,1,0` | `1,0,0,0,1,1,1,0,0,0` |
   | 20260913 | late   | `1,1,0,0,1,1,1,0,1,0` | `0,0,0,0,0,0,1,0,0,0` |

   Within single scenarios (e.g. seed 20260912/middle), `least_laxity_first`
   succeeds on the *first* urgent arrival and then fails on most of the
   rest before an isolated later success — a real *within-run* pattern tied
   to each request's own position relative to the backlog, not a uniform
   scenario-level outcome.

### 1E. Variables forbidden (generator/scenario labels)

Unchanged from v2's leakage guard: `scenario_id`, `seed`, `bulk_pressure`,
`urgent_arrival_phase`, `urgent_tightness`, any `"phase*"`/`"tight*"`/`"kvp2."`
substring, `intended`/`winner`, `generator_version`, and any parent-policy
name string. Only fields on `ObservableRequest`/`ObservableState`/
`ObservableGPUState` may be used — mirrors
`src/llmserveopt/composition/prefill_control_features.py`'s
`FORBIDDEN_FEATURE_KEYS` pattern, extended with the KV-specific tokens above.

### 1G. Aggregate timing interaction vs. genuine within-trajectory switching
### opportunity — the decisive question

**Answer, stated precisely and without overclaiming:** v2's frozen gates
(G5/G10) are a **between-scenario, aggregate** finding — `urgent_arrival_phase`
is a scenario-generation parameter, so "phase=middle beats phase=early" is,
strictly, a comparison across different generated scenarios, not evidence
read off one continuous run. **However**, the new re-simulation work in this
section (§1D, point 4) shows the *same underlying mechanism* (urgent-vs-bulk
queue contention at the moment a KV slot frees up) produces **heterogeneous,
non-uniform outcomes for individual requests within a single trajectory**,
tied to each request's own arrival-relative-to-backlog position. This is
consequential: it means the v2 gate evidence is not merely an artifact of
comparing different scenario populations — the identified mechanism
plausibly operates continuously, moment-to-moment, within any one run
containing time-varying urgent contention (which every scenario in this
family does, since urgent arrivals are Poisson-spread over a
0.03s-mean-interarrival span, not simultaneous).

**Conclusion: the frozen evidence supports a *plausible* within-trajectory
switching opportunity (not a proven one) — proceeding is scientifically
justified, but item 5's non-degeneracy instrumentation below is the actual
confirming test, not an assumption.** This is stated explicitly so the
falsification below is understood as testing a hypothesis grounded in direct
re-simulation evidence, not merely relying on the `_COMPOSITION_READY` label.

## 2. Minimal composition hypothesis

The child does **not** invent a new admission mechanism. It literally
delegates every `select_action` call, unmodified, to one of the two frozen
parent policy instances (`KVConstrainedOnlinePolicy()`,
`LeastLaxityFirstPolicy()`), chosen fresh at every scheduling step from a
single online-observable trigger:

> **mode = "reserve" (delegate to `kv_constrained_online`) if the count of
> currently-*waiting* requests classified urgent by
> `KVConstrainedOnlinePolicy`'s own `urgent_laxity_seconds` threshold is ≥
> `tau_urgent`; otherwise mode = "llf" (delegate to `least_laxity_first`).**

This is deliberately **not** `kv_util_ratio`-gated, contrary to the generic
suggestion in the task prompt's item 4 list — §1D showed `kv_util_ratio`
saturates almost immediately in the signal regime and would produce a
near-degenerate (always-reserve) child there. The urgent-waiting-count
signal was verified (§1D point 2, and a second direct check) to vary
genuinely within one trajectory (0 to 13, mass across the whole range, 16%
of steps at exactly 0 in one representative scenario) and is mechanistically
tied to the actual differentiator identified in §1D point 3.

One free parameter: `tau_urgent ∈ {1, 2, 3}` (tiny preregistered grid).
Selected by mean TRAIN ANWG, confirmed not worse on VAL, **frozen before any
TEST/OOD score is computed.**

## 3. Baselines

A. `least_laxity_first` (parent, unmodified)
B. `kv_constrained_online` (parent, unmodified)
C. `best_fixed_parent` — whichever parent has higher mean ANWG on TRAIN,
   applied globally
D. `contextual_top1` — fitted selector (LogisticRegression / DecisionTree,
   model type chosen on VAL), scenario-level observable features, predicts
   which parent wins; reuses that parent's already-simulated score
E. `hard_conditional` — symbolic if/else over observable scenario features
   (fraction of requests with predicted-urgent laxity, mean/min slack) —
   mirrors `prefill_control_policy.hard_conditional_rule`'s pattern
F. `kv_adaptive_reserve_child` — the new within-scenario child (§2)
G. `parent_oracle` — `max(kv_constrained_online, least_laxity_first)` per
   scenario
H. `oracle_after_child` — `max(parent_oracle, kv_adaptive_reserve_child)` per
   scenario (computed analytically, not a new simulator run)

## 4. Threshold fitting procedure

`tau_urgent` candidates `{1, 2, 3}` are each simulated once on every TRAIN
scenario (mean ANWG computed per candidate), the arg-max candidate is then
evaluated on VAL only to confirm it is not worse than the runner-up TRAIN
candidate there; the frozen choice is written to
`run_dir/child_threshold.json` **before** TEST/OOD scenarios are simulated
for the child. No threshold search touches TEST or OOD data at any point —
enforced by a dataflow test (§11) asserting the fitting function's signature
carries no test/ood-labelled rows.

## 5. Preregistered hypotheses (H)

- **H1 (mechanism):** the urgent-waiting-count switch signal is non-constant
  within TEST/OOD trajectories (both modes occur on a non-trivial fraction
  of held-out scenarios) — §1G's plausibility claim, tested directly.
- **H2 (composition upside):** `kv_adaptive_reserve_child` beats
  `parent_oracle` (i.e., beats *both* fixed parents) on at least some
  held-out scenarios by more than ε=0.01 — the literal test of "exploits
  within-trajectory switching," as opposed to merely picking one parent for
  a whole scenario.
- **H3 (selector-sufficiency null):** `contextual_top1` alone already closes
  most of the gap to `parent_oracle` (as it did for ESTF/WFS and
  PrefillControl v2) — the default expectation this falsification is
  designed to challenge.
- **H4 (non-degeneracy):** the child is not equivalent to a scenario-level
  selector — on at least one held-out scenario it visits *both* modes and
  makes *at least one* admission decision that differs from what the
  single-mode-for-the-whole-scenario parent would have done.

## 6. Preregistered gates (G), frozen thresholds

Canonical metric `arrival_normalized_weighted_goodput`, practical threshold
ε=0.01 (inherited from the KV v2 design, itself inherited from Family A/B).
`E_P(x) = max(R_kv(x), R_llf(x))`;
`G(c;P) = mean_x[max(R_c(x) - E_P(x), 0)]`;
`G_ε(c;P) = mean_x[max(R_c(x) - E_P(x) - ε, 0)]`.

| Gate | Test | Threshold |
|---|---|---|
| G1 | Non-degeneracy (H4) | ≥1 held-out (TEST∪OOD) scenario has both modes active **and** ≥1 mode transition **and** ≥1 admission decision differing from both fixed-mode replays |
| G2 | TEST envelope expansion | `G_ε(child; P)` on TEST > 0 |
| G3 | CI support | Paired bootstrap 95% CI (2000 resamples, seed 20261201) lower bound on TEST per-scenario `child − E_P` > 0 |
| G4 | Beats both parents | ≥1 TEST scenario has `child > kv_constrained_online + ε` **and** `child > least_laxity_first + ε` simultaneously |
| G5 | Beats selector | mean(`child − contextual_top1`) on TEST > ε, **or** (if `contextual_top1` is already within 0.005 of `parent_oracle` on TEST) `child` still expands the envelope per G2 |
| G6 | OOD directional replication | sign of `mean(child − E_P)` on OOD matches TEST (not required to reach the same magnitude — directional check only, consistent with the smaller OOD sample) |
| G7 | Safety/feasibility | 0 failed evals, 0 duplicate `(scenario_id, method)` rows, 0 NaN/Inf, 0 completion-fraction regressions below `min(parent completion fractions) − 0.01` on any scenario, child peak-KV-ratio ≤ `max(parent peak-KV-ratios)` per scenario (§8b), leakage guard passes on all scenarios |
| G8 | Sample adequacy | TEST ≥ 8 scenarios, OOD ≥ 8 scenarios (both satisfied by construction, §7) |

`KV_COMPOSITION_GO` requires **all** of G1-G8.
`KV_SELECTION_SUFFICIENT_FOR_THIS_PAIR` if G7-G8 pass, `contextual_top1` is
within 0.005 of `parent_oracle` on TEST, and G2 or G4 fails (no meaningful
child-specific gain beyond selection).
`KV_COMPOSITION_INCONCLUSIVE` if G1 fails (child is degenerate — cannot
distinguish this from a disguised selector), G7 fails, or G3's CI is too
wide to resolve sign (crosses zero substantially in both directions with
TEST/OOD directions disagreeing).

## 7. TRAIN / VAL / TEST / OOD split

Reuses v2's exact scenario grid, generator, and seed set — no new factor
levels, no wider ranges (per task scope: reuse, don't invent). The seed
partition reuses v2's own preregistered in-sample/held-out boundary,
subdividing the 4 in-sample seeds further:

| Split | Seeds | Scenarios | Role |
|---|---|---|---|
| TRAIN | `20260910, 20260911` | 24 | threshold `tau_urgent` fitting, selector/hard-rule fitting |
| VAL | `20260912` | 12 | model-type selection for `contextual_top1`, threshold confirmation |
| TEST | `20260913` | 12 | primary held-out gate computation (G1-G5, G7-G8) |
| OOD | `20260914, 20260915` | 24 | G6 directional replication only — **never** enters any fitting or threshold decision |

Grouped by seed (no scenario-level leakage across splits, identical logic to
`prefill_control_splits.assign_family_b_v2_splits`). Enforced by
`assert_no_split_leakage` and a dataflow test.

## 8. Non-degeneracy definition (frozen, G1)

A held-out scenario counts toward non-degeneracy only if, on that scenario's
run, the child's own instrumentation log shows: (a) at least one step in
`"llf"` mode and at least one step in `"reserve"` mode; (b) at least one mode
transition; (c) at least one request whose admission step differs from what
*both* single-mode replays (pure `kv_constrained_online`, pure
`least_laxity_first`) produced on that same scenario. G1 requires this on
≥1 held-out scenario, not all — a child that is non-degenerate on some
scenarios and collapses to one mode on others (e.g. the `urgent_tightness
=loose` placebo cells, where no request is ever classified urgent) is
expected and fine, provided it is not *uniformly* degenerate.

## 8b. Safety/feasibility baseline note

Direct verification during test-writing found that **neither unmodified
parent** respects a hard `max_kv_tokens` ceiling on every simulator step
(KV usage grows during decode past the admission-time capacity check -- a
pre-existing simulator/policy property, not introduced here): on one signal
scenario, `least_laxity_first` peaks at 7194/6000 (120%) and
`kv_constrained_online` at 6722/6000 (112%). Since the child only ever
delegates to these two unmodified policies, G7's safety check is therefore
"child's peak KV ratio ≤ max(parent peak ratios) on the same scenario" (no
new unsafe behavior beyond what the accepted, frozen parents already
exhibit), not an absolute zero-overflow bound neither parent itself
satisfies.

## 9. Provenance discipline

No selector retraining beyond what §3D/§3E already specify. No MAP-Elites,
GP, symbolic distillation, or LLM-guided synthesis. Parent files
(`kv_constrained_online.py`, `least_laxity_first.py`) — zero changes,
verified by a contract test. All frozen v1/v2 KV artifacts and Family A/B
artifacts — untouched.
