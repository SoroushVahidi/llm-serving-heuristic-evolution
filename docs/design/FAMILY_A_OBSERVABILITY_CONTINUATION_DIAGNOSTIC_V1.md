# Family-A Observability / Continuation-Dependence Diagnostic v1 — TRAIN/VAL-Only Design and Preregistration

Date: 2026-08-20

## 0. Scope

**DESIGN / PREREGISTRATION ONLY, TRAIN/VAL METHODOLOGY DIAGNOSTIC.** This is the
`RUN_PARTIAL_OBSERVABILITY_OR_CONTINUATION_DIAGNOSTIC` step selected by
`docs/current/decision_criticality_timescale_trainval_v1_analysis_20260820.md` §11. It does not
retrain Stage-1/Stage-2, does not retune the FSM/dwell, does not add a new policy, does not touch
TEST, does not touch the Family-B held-out replication set, and computes no project-level
scientific verdict beyond the interpretation category and next-step recommendation named in §13/§14
below. It does not supersede or reopen `HIERARCHICAL_ROUTER_NO_GO` or `LIVE_REEVAL_CONFIRMS_NO_GO`.

**Central question**: for Family-A (`RANKING_FAIRNESS`) consequential policy disagreements — the
one family the completed decision-criticality study found a real, corroborated positive ceiling
for (+0.886 mean completions/branch, TEST-audit-corroborated +0.0302 ANWG) — is the direction of
that advantage (a) predictable from observable causal state at the disagreement point
(**observability**), (b) primarily an artifact of which policy controls subsequent decisions
(**continuation dependence**), or (c) neither cleanly (**irreducible / weakly observable**)?

Scope restriction (per the authorizing task): Family A only. Families B/C are not touched, rerun,
or used to rescue a weak Family-A result.

---

## A. Population

Only Family-A (`FAMILY_A_FAIRNESS_STARVATION_V2`) TRAIN/VAL scenarios from the exact same MF-PSD
population the completed decision-criticality study used
(`dcm.load_trainval_scenario_table()`, filtered to `mechanism_family ==
"FAMILY_A_FAIRNESS_STARVATION_V2"` — 64 scenarios: 54 train + 10 val, per the frozen
`build_splits`). No new workload is generated. No new scenario configuration is introduced. The
`assert_trainval_only` split guard is reused verbatim from
`llmserveopt.analysis.decision_criticality_timescale_trainval_v1`; any `"test"` split value raises
immediately. This diagnostic never imports `family_b_balanced_replication_v1` (structural check,
mirrors the parent diagnostic's own guard).

## B. Disagreement-state definition

Identical to the completed study (design doc `DECISION_CRITICALITY_TIMESCALE_TRAINVAL_V1.md` §5B):
a step where `effective_regime == REGIME_A` and the two frozen Family-A native candidates —
`estimated_service_time_first` (**ESTF**) and `weighted_fair_share` (**WFS**) — propose actions
whose canonical `{gpu_id: sorted(admit_ids)}` mappings differ. Both candidates are pure functions
of `ObservableState` (verified by reading `estimated_service_time_first.py` /
`weighted_fair_share.py`: neither policy accesses anything beyond the state object passed to
`select_action`), so both candidates' actions at a disagreement step are already fully determined
and reusable without re-deriving the live router's decision. Every disagreement state used here is
identified by literally re-running the live reference trajectory with the same shadow-disagreement
detection the completed study used (re-derived fresh in this diagnostic's own driver, not read back
from the completed study's CSV — this diagnostic performs its own independent instrumentation pass,
per the authorizing task's "richer causal state to log" requirement, since the original artifact did
not retain router-input state at the disagreement step).

## C. Causal observable-state schema

At every Family-A disagreement step, the diagnostic snapshots `ObservableState` **as already
available to the live router at that exact step** (no re-simulation, no future information) and
derives the following **strictly decision-time** feature groups. Every feature is computed from
fields already present on `ObservableState`/`ObservableRequest`/`ObservableGPUState`
(`src/llmserveopt/core/types.py`) — no new simulator instrumentation is added (§15 minimal-invasion
requirement is satisfied trivially: zero simulator hooks are extended).

- **Group A — snapshot workload**: `queue_length` (`len(waiting_queue)`), `active_count` (sum of
  `len(active_request_ids)` over GPUs), `completed_count`, `step`, `n_gpus`.
- **Group B — request-distribution** (over `waiting_queue`, quantiles p10/p50/p90 + mean): age
  (`state.time - arrival_time`), `predicted_output_tokens`, `prompt_tokens`, remaining estimated
  service time (`predicted_service_proxy`, the same ESTF-internal proxy, computed read-only here —
  reusing an existing scoring function, not adding new logic).
- **Group C — fairness/starvation**: per-`class_id` waiting-queue counts and per-`class_id` active
  counts (`queue_class_counts`, an existing read-only helper reused, not reimplemented); the
  weighted-fair-share "deficit" score's two raw ingredients (`demand` and `served_share` per class)
  aggregated as `max_class_deficit_ratio` = max over classes of `demand / max(1, served_share+1)`;
  longest-waiting request's age; count of distinct classes present in queue.
- **Group D — urgency/slack**: `slo_deadline - state.time` ("laxity") quantiles (p10/p50/p90 +
  mean) over the waiting queue; fraction of the waiting queue with laxity `< 0` (already breached)
  and `< HORIZON_H_STEPS_IN_TIME` (near-deadline, using the same `step_size` × `HORIZON_H=10`
  conversion already frozen in the parent diagnostic, so "near-deadline" means "would breach within
  the same 10-step counterfactual horizon the parent study used").
- **Group E — resource/KV**: mean/max KV utilization across GPUs (`current_kv_tokens /
  max_kv_tokens`), free KV capacity (sum), prefilling/decoding counts if `enable_prefill_modeling`
  is on for this scenario (Family A does not enable prefill/decode phase-splitting per its template
  — checked in code; if `prefilling_count`/`decoding_count` are always 0 for Family A, they are
  recorded as constant/dropped rather than fabricated).
- **Group F — pair-specific disagreement geometry**: ESTF's own sort-key ranking of the waiting
  queue vs. WFS's own score ranking of the waiting queue (both policies' scoring functions are
  read-only pure functions of `ObservableState`, called here exactly as the policies themselves
  call them — zero reimplementation); Spearman rank correlation between the two orderings restricted
  to the requests both policies would consider (top-`K` = the larger of the two candidates' actual
  admitted-set sizes, to keep the comparison decision-relevant rather than over the full queue tail);
  symmetric-difference size between ESTF's and WFS's admitted-ID sets (`|admit_ESTF Δ admit_WFS|`);
  whether the disagreement is "top-1-only" (the single highest-ranked-by-either-policy request
  differs in admit status) or "deeper" (symmetric difference `> 1`).
- **Group G — short history** (see §12 below; frozen `HISTORY_WINDOW = 10` steps, matching the
  parent study's own `HORIZON_H`): queue-length slope, mean-KV-utilization slope, and
  admitted-completed-count slope over the preceding `HISTORY_WINDOW` steps of the **real reference
  trajectory only** (never a fork/counterfactual trajectory), read from the trajectory rows already
  recorded by `LiveHierarchicalRouterPolicy` up to and including the disagreement step. If fewer
  than `HISTORY_WINDOW` prior steps exist in the scenario (early-scenario disagreement), the slope is
  computed over however many prior steps exist and the row is flagged `history_window_truncated`.

**Forbidden future-derived features** (never computed, never logged): any post-disagreement-step
queue/KV/completion state; `actual_output_tokens`; final scenario outcome; anything from a
counterfactual fork's own future (a fork's *own* rollout outcome is a *label*, §F below, never a
*feature* fed to the predictability models of §10).

## D. Immediate-action counterfactual

At each Family-A disagreement step `s_t`, both `action_ESTF(s_t)` and `action_WFS(s_t)` are already
fully determined (both policies are pure functions of the snapshotted `ObservableState`, §B) — no
counterfactual branching is needed to obtain the two candidate *actions themselves*, only to measure
their *downstream consequence* (§E).

## E. Continuation counterfactual — the core experiment

Reuses the parent diagnostic's exact, already-tested fork mechanism
(`fork_from_live_simulator`, `LiveFork.advance_one_step`, `run_bounded_rollout` — imported, not
reimplemented) with **one addition**: `run_bounded_rollout` already supports forking with any
`(policy, policy_id, first_action)` triple, so a "common continuation" branch is obtained simply by
passing a *different* policy instance than the one that produced `first_action` — no new fork
primitive is required.

For each sampled disagreement state `s_t` (§G sampling rule), **four** bounded rollouts are run,
each forked independently from the same pre-step simulator state (never from each other, never
mutating the real reference simulator — identical isolation guarantee as the parent diagnostic):

| Branch | first action applied | continuation policy | reused for |
|---|---|---|---|
| `br_ESTF_ESTF` | `action_ESTF(s_t)` | ESTF | native-ESTF outcome AND same-continuation(common=ESTF) ESTF side |
| `br_WFS_WFS`   | `action_WFS(s_t)`  | WFS  | native-WFS outcome AND same-continuation(common=WFS) WFS side |
| `br_WFS_ESTF`  | `action_WFS(s_t)`  | ESTF | same-continuation(common=ESTF) WFS side |
| `br_ESTF_WFS`  | `action_ESTF(s_t)` | WFS  | same-continuation(common=WFS) ESTF side |

Each branch is a `run_bounded_rollout(...)` call with `max_extra_steps =
FAMILY_A_DIAGNOSTIC_MAX_EXTRA_STEPS = 1500` (§ frozen constants below), reporting `completed_count`
(the same utility metric the parent study's ceiling diagnostic already uses — chosen for continuity
and comparability, not redefined here).

- **6A. Same-continuation action effect** (symmetrized, per the authorizing task — a single common
  continuation would privilege whichever candidate that continuation *is*, so both choices of common
  continuation are computed and reported separately, never averaged into a single hidden number):
  - `Δ_same_commonESTF(s_t) = U(br_ESTF_ESTF) − U(br_WFS_ESTF)`
  - `Δ_same_commonWFS(s_t)  = U(br_ESTF_WFS)  − U(br_WFS_WFS)`
  - `Δ_same(s_t)` (robust combination, reported alongside — never in place of — the two components)
    `= mean(Δ_same_commonESTF(s_t), Δ_same_commonWFS(s_t))`.
  - Justification for ESTF/WFS as the two common continuations (chosen **before** running, not
    after seeing results): they are the only two policies "already present in the family" (the
    authorizing task's explicit preference) — no third neutral policy exists in the Family-A native
    pair, so symmetrizing over both frozen candidates is the only design that avoids privileging one
    parent, and it directly answers "does the *immediate action* matter independent of which
    candidate happens to keep driving."

- **6B. Native-continuation effect**:
  `Δ_native(s_t) = U(br_ESTF_ESTF) − U(br_WFS_WFS)` — action and continuation coupled to the same
  candidate, i.e. exactly what the parent study's ceiling diagnostic already measured (reproduced
  fresh here, not read back from the completed artifact, so it is drawn from the same sample and
  bound as the new same-continuation branches for a valid within-event comparison).

- **6C. Continuation dependence**: `C(s_t) = Δ_native(s_t) − Δ_same(s_t)`, frozen exactly as stated.
  A robustness companion, `C_range(s_t) = Δ_native(s_t) − min/max(Δ_same_commonESTF(s_t),
  Δ_same_commonWFS(s_t))` (two values), is also reported so a large gap between the two
  same-continuation components (rather than agreement between them) is visible rather than
  averaged away.

All four differences are defined **ESTF − WFS**, uniformly, regardless of which candidate the live
router actually chose at `s_t` — this avoids conflating "sign relative to the router's choice" with
"sign relative to a fixed policy identity," which would make `C(s_t)` uninterpretable across events
where the router's chosen candidate differs.

## F. Labels/targets

Per sampled disagreement state, record (never overwriting the completed study's original
`disagreement_and_divergence_events.csv` — written to a new file under this diagnostic's own output
directory):
- `sign(Δ_same)`, `|Δ_same|`, `Δ_same_commonESTF`, `Δ_same_commonWFS`
- `sign(Δ_native)`, `|Δ_native|`
- `C`, `sign_same_eq_native` (bool: `sign(Δ_same) == sign(Δ_native)`, `0` treated as its own sign
  bucket, never silently merged into `+`/`-`)
- `router_chosen_policy_id` (ESTF or WFS — which one the live router actually picked at `s_t`, for
  context only, never used as a feature in §10's predictability models — it would leak the
  Stage-2 selector's own decision, which is exactly what this diagnostic must stay independent of)
- the full causal observable-state feature vector (§C)
- stable join key: `(canonical_scenario_id, step)`, joinable back to the completed study's
  `disagreement_and_divergence_events.csv` by the same key (not joined automatically by this
  diagnostic's code, but the key is preserved so a human/future analysis could)

## G. Split strategy / sampling rule

**Scenario population split**: TRAIN/VAL only, as in §A — Family-A scenarios' own `split` column
(train/val) is preserved per-event and used as the **grouping key** for §10's grouped
cross-validation (never split at the event level — no two disagreement events from the same
scenario are ever allowed on opposite sides of a train/test fold).

**Disagreement-event sampling rule** (frozen before running, outcome-blind): reuse the parent
study's exact rule — **the first `FULL_TRAJECTORY_BRANCHES_PER_SCENARIO = 3` disagreement steps
encountered, in trajectory order, per scenario** (chosen for direct comparability with the
completed study's own 132-branch Family-A ceiling sample, and because it is already a
preregistered, outcome-blind, computationally-validated-tractable rule from the frozen parent
design). Expected event count: **≤ 3 × 64 = 192** (fewer if a scenario has < 3 disagreement events
— the completed study found 132 actual branches attempted for Family A under this exact rule, so
132 is the expected realized count here too, modulo any nondeterminism — none is expected, since
both diagnostics use the same frozen models/scenario-rebuild path).

**Computational cost estimate**: 132 events × 4 branches = 528 bounded rollouts, each ≤ 1500 extra
steps (half the parent study's 3000-step bound — chosen here, before running, purely for
computational tractability given 4 branches/event vs. the parent's 2, and because Family A's own
episode-length p90 is 662 steps and median 223 steps §4 of the completed analysis — 1500 steps is
still ≥ 2× the p90 episode length, comfortably covering the timescales already found relevant. This
bound is not tuned on any output of this diagnostic).

## H. Analysis metrics

Q1–Q9 as posed by the authorizing task (materiality of `Δ_same`, mixedness of sign under same
continuation, sign(same) vs sign(native) disagreement rate, continuation-dependence magnitude vs.
local-action-effect magnitude, grouped-CV predictability of `sign(Δ_same)` and `sign(Δ_native)`
from §C features, strongest associated variables, sub-regime dominance patterns). Grouped
predictability analysis (§10 of the authorizing task): majority-class baseline, shallow decision
tree (`max_depth=3`), logistic regression, Random Forest (capacity diagnostic only) — grouped
`GroupKFold` by `canonical_scenario_id`, `n_splits = min(5, n_unique_scenarios_with_events)`.
Metrics: balanced accuracy, ROC-AUC (if both classes present per fold), macro F1, confusion
matrices, feature importances/coefficients, shallow-tree text rules — all reported against the
majority-class baseline explicitly, never accuracy alone (§10 imbalance guard).

## I. Falsification criteria

This diagnostic is designed so that **each** interpretation category (§J) has a distinct,
pre-specified evidence signature — it is not scored against an arbitrary pass/fail threshold:
- `LOCAL_ACTION_OBSERVABLE` is falsified if `Δ_same` is materially zero/noise-dominated, or if
  grouped held-out prediction of `sign(Δ_same)` does not clear the majority-class baseline by a
  visible margin on both balanced accuracy and AUC.
- `CONTINUATION_DOMINATED` is falsified if `sign(Δ_same) == sign(Δ_native)` for the large majority
  of events (i.e. continuation choice does not change the answer).
- `PARTIALLY_OBSERVABLE` is falsified if either the fully-local model (`LOCAL_ACTION_OBSERVABLE`
  criteria) or the fully-continuation-driven picture (`CONTINUATION_DOMINATED` criteria) cleanly
  fits instead.
- `NO_ROBUST_LOCAL_SYNTHESIS_SIGNAL` is the default only if `Δ_same` itself is not materially
  nonzero for a meaningful fraction of events, independent of any model's predictive power.

No numeric threshold is fixed in advance for "materially nonzero" or "visible margin" — per the
authorizing task's explicit instruction, classification is qualitative, evidence-based, and
justified in the analysis report's prose, not gated by an invented cutoff.

## J. Success/interpretation categories

Exactly as specified by the authorizing task (reproduced for freezing purposes):
- **A. `LOCAL_ACTION_OBSERVABLE`**: same-continuation advantage real, sign predictably related to
  causal state.
- **B. `CONTINUATION_DOMINATED`**: native advantage exists, but same-continuation effect
  weak/unstable or signs often change with continuation.
- **C. `PARTIALLY_OBSERVABLE`**: local advantage exists, but available causal state only weakly
  predicts direction; richer state/history helps but does not resolve it.
- **D. `NO_ROBUST_LOCAL_SYNTHESIS_SIGNAL`**: apparent prior complementarity does not survive clean
  local counterfactual isolation.

---

## Frozen constants (fixed here, before any run)

| constant | value | source |
|---|---|---|
| `FAMILY_A_DIAGNOSTIC_MAX_EXTRA_STEPS` | 1500 | new, justified §G |
| `FULL_TRAJECTORY_BRANCHES_PER_SCENARIO` | 3 | reused from parent diagnostic |
| `HISTORY_WINDOW` | 10 steps | matches parent's `HORIZON_H` |
| `dwell` (reference only, never modified) | 20 | `DWELL_MINIMUM_STEPS` |
| common continuations | `{estimated_service_time_first, weighted_fair_share}` | §E |
| utility metric `U` | bounded-rollout `completed_count` | reused from parent ceiling diagnostic |
| grouping key for CV | `canonical_scenario_id` | §H |
| sign convention | ESTF − WFS, always | §E |

## Provenance requirements

Every run records, in `experiments/family_a_observability_continuation_v1/`: git HEAD SHA and
dirty-tree flag at launch, exact invoking command, SHA-256 of this design doc, SHA-256 of
`experiments/mf_psd_v1/mf_psd_scenarios_v1.csv`, the frozen constants above, Python/numpy/pandas/
sklearn versions, UTC start/end timestamps, scenario counts processed (expected 64, asserted before
any result is written), and integrity counts (events sampled, branches attempted/completed).

## Pre-launch verification plan

1. Split guard: every scenario fed in has `split in {"train","val"}`; any `"test"` id raises.
2. No replication-module access (reused guard).
3. Fork isolation: reused, already tested by the parent diagnostic's own test suite — re-asserted
   here on this module's own fork call sites via a dedicated test.
4. Canonical-action purity: ESTF/WFS actions are admit-only (reused assertion).
5. Deterministic replay: running the same Family-A scenario's diagnostic twice yields bit-identical
   disagreement-step identification and branch outcomes.
6. Sign-convention unit test: a synthetic fixture where ESTF is known better under a fixed
   continuation, and a second fixture where native continuation flips or amplifies the result —
   independently verifies `Δ_same`/`Δ_native`/`C` arithmetic.
7. Focused test suite (`tests/test_family_a_observability_continuation_v1.py`) passes before the
   real 64-scenario run is launched.

## Standing long-running job rule

Launched in a dedicated, named tmux session `family_a_observability_continuation_v1`. Logs and
results under `experiments/family_a_observability_continuation_v1/` and
`logs/family_a_observability_continuation_v1.log`. Monitored until completion (this diagnostic's
own analysis report depends on its output, unlike the parent study's fire-and-forget launch rule).
No result interpretation, verdict, or synthesis-design step is taken until the run and its
integrity checks are both complete.
