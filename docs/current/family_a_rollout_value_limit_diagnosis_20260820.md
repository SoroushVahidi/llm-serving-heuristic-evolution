# Family-A Receding-Horizon Oracle: Rollout Value-Limit Diagnosis

Date: 2026-08-20

Diagnostic-only pass over the completed `family_a_receding_horizon_oracle_v1`
TRAIN/VAL run. No new controller was designed, no horizon/objective/fallback
was changed, no TEST data was read, and no prior frozen result was modified.
One diagnostic-only re-run of the already-frozen, unmodified controller was
performed solely to capture per-decision logs the original scientific run
did not persist (justified in §2; determinism across repeated runs is
already proven by `tests/test_family_a_receding_horizon_oracle_v1.py`, so
this is not a new experiment).

---

## 1. Executive Diagnosis

The oracle rollout controller loses to fixed WFS because its windowed
rollout objective and the raw-completion counterfactual signal it is built
on are **structurally biased toward ESTF-like throughput in exactly the
regime — `favlong` scenarios — where full-scenario, priority/SLO-weighted
ANWG structurally favors WFS**. This is not primarily a horizon-length
problem (H=20 does not fix it and does not outperform H=5) and not
primarily a plan-execution/replanning problem (commitment statistics are
mixed, not dominated by immediate reversal, and barely change across H).
It is an **objective/terminal-value definition problem**, compounded by a
**genuine structural WFS long-term invariant** (fairness-debt-driven
priority/SLO protection of `favlong` requests) that a short bounded window
cannot see.

Classification: **`ROLLOUT_OBJECTIVE_MISALIGNMENT` (primary) +
`WFS_LONG_TERM_INVARIANT` (primary, complementary) +
`HORIZON_TOO_SHORT` (secondary/contributing) +
`ORACLE_HEADROOM_CONCENTRATED` (partially true, but by *regime*, not by a
handful of scenarios)**. `PLAN_EXECUTION_MISMATCH` evidence is weak and not
the dominant driver.

Next step: **`JUSTIFY_TERMINAL_VALUE_REDESIGN`**.

## 2. Integrity

Reconfirmed independently from the raw artifacts (not re-derived from the
prior report's prose): 64/64 scenarios (54 train/10 val), 0 failures, 0
TEST rows, `H ∈ {1,5,20}`, WFS fallback, common-WFS continuation
(`COMMON_CONTINUATION_BUDGET=200`), 0 planning-cap hits (cap=150), and
`policy_id`-grouped mean ANWG exactly matching the prior report
(`weighted_fair_share`=0.747775, `h1`=0.735997, `h5`=0.737946,
`h20`=0.736072, `estimated_service_time_first`=0.729624,
`family_a_stateful_controller_v1`=0.728125). **No integrity failure —
`ROLLOUT_LIMIT_DIAGNOSIS_BLOCKED_BY_INTEGRITY` does not apply.**

One additional artifact was created for this diagnosis:
`scripts/diagnose_family_a_receding_horizon_oracle_v1_decision_logs.py`
re-runs the **unmodified** `FamilyARecedingHorizonOracleV1` (same code
path, same 64 scenarios, same 3 horizons, same objective/fallback/budget)
and persists `policy.decision_log()` — a list already computed and returned
by the existing class, just not written to disk by the original scientific
run. This is instrumentation, not a new experiment: no controller, design,
or scenario-generation code was touched. Output:
`experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_decision_logs.csv`
(6,151,248 rows — one row per real `select_action` call across all
scenarios × horizons; 8,195 of these are actual planning-call rows, matching
the 2,707+2,736+2,752=8,195 total planning calls already reported).

## 3. Scenario Loss Decomposition

For each horizon, scenarios were split into `beats WFS` / `ties WFS` /
`loses to WFS` and joined against parsed scenario metadata
(`utilization`, `skew`, `favlong`/`favshort`, `noise`):

| H | loses | ties | beats | losses that are `favlong` | beats that are `favshort` |
|---|---:|---:|---:|---:|---:|
| 1 | 25 | 25 | 14 | 22/25 (88%) | 10/14 (71%) |
| 5 | 24 | 22 | 18 | 21/24 (88%) | 14/18 (78%) |
| 20 | 23 | 18 | 23 | 23/23 (**100%**) | 18/23 (78%) |

At every horizon, essentially all losses are `favlong` scenarios, and this
gets **more**, not less, concentrated as `H` grows (100% of H=20's losses
are `favlong`). `favshort` scenarios are where the controller mostly ties or
beats WFS. This is not a broad, uniform failure — it is a **clean regime
split**.

## 4. Decision-Level Regret / ESTF Deviation Statistics

Per-scenario ESTF-choice fraction (from `controller_diagnostics`), by
regime:

| H | favlong mean ESTF-choice frac | favshort mean ESTF-choice frac |
|---|---:|---:|
| 1 | 26.8% | 32.9% |
| 5 | 27.6% | 43.7% |
| 20 | 47.2% | 65.5% |

ESTF-choice frequency roughly doubles in `favlong` between H=1 and H=20
(26.8%→47.2%), yet loss *count* in `favlong` does not improve (22→23 losses
out of 32) — more horizon makes the controller trust ESTF more in exactly
the regime where that trust is unwarranted, without correspondingly fixing
the outcome.

**Within-regime, decision-level regret is statistically significant**:
at `H=1`, the per-scenario ESTF-choice fraction is strongly and
significantly negatively correlated with `(controller_ANWG − WFS_ANWG)`
*within* `favlong` alone (Spearman ρ=−0.518, p=0.002, n=32) — i.e., this is
not a between-regime artifact; within `favlong`, choosing ESTF more often
causally tracks with doing worse. In `favshort`, the same correlation is
weak and non-significant (ρ=0.125, p=0.60). At `H=20` the within-`favlong`
correlation weakens and loses significance (ρ=0.158, p=0.39) — consistent
with (but not proof of) partial improvement at longer horizon.

## 5. Short- vs. Long-Horizon Preference Agreement (existing counterfactual data)

Cross-referencing the repaired `family_a_observability_continuation_v1`
events (91 disagreement events, four-branch bounded counterfactuals) by
regime:

| Regime | n events | mean `Delta_same` (local) | mean `Delta_native` (bounded continuation) | sign(`Delta_same`)==sign(`Delta_native`) |
|---|---:|---:|---:|---:|
| favlong | 60 | +0.050 | **+1.400** | 40.0% |
| favshort | 31 | +0.065 | +0.323 | **80.6%** |

A purely local (1-step) signal agrees with the native-continuation-preferred
direction only 40% of the time in `favlong` (worse than useful) vs. 81% in
`favshort`. This independently corroborates §3–4: `favlong` is exactly where
short-horizon/local signals are least trustworthy, and `favshort` is exactly
where they are most trustworthy — matching where the rollout controller
loses and wins, respectively.

## 6. Common-Continuation Bias

Using the existing four branches (`br_estf_estf`, `br_wfs_wfs`,
`br_wfs_estf`, `br_estf_wfs`, raw completed-count only — no weighted/SLO
field exists in this artifact, noted as a limitation):

- ESTF-first: `br_estf_estf − br_estf_wfs completed count`: mean **+0.901**,
  51/91 positive, **0/91 negative**. ESTF's first action always does at
  least as well completing more requests when ESTF itself continues, never
  when forced into a WFS continuation.
- WFS-first: `br_wfs_wfs − br_wfs_estf completed count`: mean **−1.055**,
  58/91 negative, only 1/91 positive. WFS's first action produces **fewer**
  raw completions when WFS itself continues than when forced into an ESTF
  continuation.

Both directions point the same way: **ESTF-style continuation always
produces more raw completions than WFS-style continuation, regardless of
which policy took the first action.** This is consistent with ESTF being a
throughput-maximizing (SJF-like) policy by construction. It means: (a) the
choice of `COMMON_CONTINUATION = WFS` in the V1 design is *not* what creates
an anti-ESTF bias — if anything, raw-throughput signals bias *toward* ESTF
regardless of continuation choice; but (b) it also means any bounded
counterfactual scored by raw/near-raw completions (as both
`family_a_stateful_controller_v1`'s training label `sign(Delta_native)` and,
functionally, the receding-horizon objective's within-window completion
tally were) will systematically over-credit ESTF relative to what a
priority/SLO-weighted, full-scenario objective rewards. **This is evidence
for objective misalignment, not for a specific common-continuation-arm
bias.**

## 7. Horizon / Truncation Analysis

Full scenarios run ~30,733–34,135 steps (mean 32,038). Prior
decision-criticality evidence: Family-A critical episodes have median
duration 223 steps, p90 662 steps. `H≤20` plus the 200-step common
continuation gives a total planning window of ≤220 steps — comparable to
one *median* episode, well under a p90 episode, and **0.7% of a full
scenario**. This is genuinely short relative to the timescale over which
WFS's fairness-debt/priority protection (§9) accumulates and pays off. This
supports `HORIZON_TOO_SHORT` as a contributing factor. However (§9 below),
extending `H` from 5→20 does not reduce `favlong` losses and does not raise
mean ANWG (H=5 > H=20 on mean ANWG) — so horizon length alone is not
sufficient to explain the full pattern; it interacts with, and is
subordinate to, the objective-definition issue (§6, §8, §10).

## 8. Replanning / Time-Inconsistency

Across consecutive *eligible* planning decisions within a scenario (in
occurrence order — note real steps between eligible decisions always
execute WFS directly, since ineligible steps fall back to WFS by
construction):

| H | flip rate (consecutive decisions change winner) | mean run length | median run length | fraction of runs length==1 |
|---|---:|---:|---:|---:|
| 1 | 28.4% | 3.36 | 2 | 49.6% |
| 5 | 29.4% | 3.27 | 2 | 49.1% |
| 20 | 26.6% | 3.58 | 2 | 47.5% |

This is **mixed, not dominant, evidence** for time-inconsistency: ~71–73%
of consecutive planning-decision pairs *keep* the same winner (the
controller is not chronically flip-flopping), but ~47–50% of "commitment
runs" are length 1 (immediately reconsidered at the very next opportunity).
Critically, **this pattern barely changes across H=1/5/20** — if
plan-execution mismatch (scoring an ambitious multi-step plan, then
abandoning it after one step) were the dominant driver of the H1-vs-H20
differences observed in the main study, this statistic should shift
substantially with H; it does not. This argues `PLAN_EXECUTION_MISMATCH` is
a real but secondary effect, not the primary cause.

## 9. Policy-Commitment Gap

(Same underlying data as §8, reframed.) Realized same-winner run lengths
average 3.3–3.6 eligible decisions regardless of `H`, i.e. the *realized*
commitment length does not scale with the *planned* horizon `H` at all —
consistent with commitment being driven by how often the *state itself*
re-enters/exits the eligible candidate region (a property of the scenario's
disagreement structure) rather than by how far ahead the planner looked.
This is suggestive that `H` mostly changes *what gets scored*, not *how long
a choice is executed for* — reinforcing that the objective/scoring window
(§6, §10), not commitment depth, is the more load-bearing knob.

## 10. Rollout-Objective / Final-ANWG Alignment

Scenario-level Spearman correlation between the rollout's own confidence
signal and the final `(controller_ANWG − WFS_ANWG)` outcome:

| H | ρ(mean window-objective margin, outcome) | ρ(ESTF-choice fraction, outcome) |
|---|---:|---:|
| 1 | −0.018 (p=0.90) | −0.256 (p=0.067) |
| 5 | −0.017 (p=0.90) | −0.158 (p=0.26) |
| 20 | **+0.276 (p=0.048)** | +0.246 (p=0.078) |

At `H=1` and `H=5`, the rollout's own score margin carries **no usable
signal** about final scenario-level outcome (ρ≈0, not significant), and
choosing ESTF more often is directionally *harmful* (negative, marginally
significant at H=1). Only at `H=20` does the margin become weakly,
marginally significantly *positively* correlated with outcome. **This is
direct evidence that the windowed rollout objective is poorly aligned with
final ANWG at short horizons, and only partially recovers at the longest
horizon tested.**

## 11. WFS Structural Advantage

Full-trajectory metrics by regime (mean over scenarios):

| Policy | favlong ANWG | favlong `priority_weighted_slo_goodput` | favlong `max_latency` | favshort ANWG |
|---|---:|---:|---:|---:|
| WFS | **0.6029** | **0.6029** | 25.81 | 0.8926 |
| ESTF | 0.5404 | 0.5404 | **23.71** | **0.9189** |
| RHO H=20 | 0.5587 | 0.5587 | 24.62 | 0.9134 |

In `favlong`, WFS achieves the **best** priority-weighted SLO outcome
despite having the **worst** (highest) `max_latency` — a clean signature of
WFS deliberately sacrificing some requests' raw latency to protect specific
(likely long-running, high-priority) requests' SLO compliance via its
fairness-debt bookkeeping, a mechanism that plays out over the full
~32,000-step scenario. The `H=20` rollout controller sits between ESTF and
WFS on every one of these metrics — it partially, not fully, replicates
WFS's protection. This is real, if indirect, evidence for
`WFS_LONG_TERM_INVARIANT`: a coherent full-trajectory invariant that no
bounded window (even `H=20`+200) fully captures.

## 12. ESTF-Useful Regimes

Native-envelope decomposition (per-scenario ESTF vs. WFS ANWG, all 64
scenarios): ESTF is the native winner in 23/64 scenarios, WFS in 26/64, 15
ties. **By regime this splits almost perfectly**: `favlong` → WFS wins 26/32
natively (ESTF only 4/32, mean native diff −0.063 — ESTF is actively
*harmful* here on average); `favshort` → ESTF wins 19/32 natively (WFS
**0/32**, mean native diff +0.026). 89% of all native ESTF-favoring mass
(`0.840` of `0.945` total) comes from `favshort` scenarios. **The genuine
ESTF-useful regime is essentially `favshort`, not scattered evenly across
Family A** — and this is exactly where the rollout controller performs
comparably well (§3). The apparent "ESTF benefit" the controller shows in
aggregate (beating fixed ESTF, beating the failed stateful controller) is
real and regime-specific, not a counterfactual artifact — but it is a
narrower opportunity than the aggregate mean ANWG numbers alone suggest.

## 13. Native-Envelope Decomposition

`oracle_gap = native_envelope_mean(0.7625) − best_fixed_mean(0.7478) =
+0.0148`. This gap is **not concentrated in a handful of scenarios**
(top-5 ESTF-win scenarios carry only 38% of the positive ESTF-favoring
mass; 23/64 scenarios contribute) — but it **is** concentrated by *regime*:
essentially all of it is `favshort`-sourced (§12). `favlong` contributes
**negative** average headroom under ESTF and is where WFS is natively
correct nearly everywhere. So "is the oracle gap broad and exploitable, or
narrow and hard to commit to" (§14 of the task) has a precise answer: it is
broad *within* `favshort` (19/32 scenarios, modest and fairly even
per-scenario gains) and essentially **absent or negative** in `favlong` —
a regime-conditional answer, not a single scalar one.

## 14. Upper-Bound Hierarchy (existing artifacts only)

| Level | Value (mean ANWG) | Source |
|---|---:|---|
| A. Best fixed parent globally | 0.7478 (WFS) | this study |
| B. Per-scenario native ESTF/WFS oracle | 0.7625 | this study |
| C. Receding-horizon oracle controller (best arm, H=5) | 0.7380 | this study |
| D. Episode-level oracle (directional only) | not apples-to-apples | `decision_criticality_timescale_trainval_v1`: bounded full-trajectory branch mean **+0.886 completions/branch** (raw units, sampled at first 3 disagreement events/scenario, ≤3000-step bound) — directionally corroborates nonzero Family-A headroom beyond B's 3-event sample, but is not convertible to an ANWG number without further simulation (not attempted, per task's anti-new-experiment instruction) |
| E. Local-action oracle (directional only) | not separately computed | repaired `Delta_same` mean +0.055 (tiny) vs. `Delta_native` mean +1.033 (large) — a hypothetical perfect local-action switch would capture very little value; almost all of B's headroom is continuation-dependent, matching the original `CONTINUATION_DOMINATED` finding |

The rollout controller (C) sits **below** A, not between A and B. Levels D
and E are directional corroboration only, not new precise bounds — computing
them properly would require new simulation, which this task prohibits.

## 15. Ranked Failure Hypotheses (by evidence strength)

1. **`ROLLOUT_OBJECTIVE_MISALIGNMENT`** — strongest evidence. §6 (raw
   completions structurally favor ESTF-style continuation regardless of
   which policy moves first), §10 (rollout score margin uncorrelated or
   negatively correlated with final outcome at H=1/H=5), §4 (statistically
   significant within-`favlong` negative correlation between ESTF-choice
   frequency and outcome, p=0.002).
2. **`WFS_LONG_TERM_INVARIANT`** — strong, complementary evidence. §11
   (WFS achieves the best priority-weighted SLO outcome in `favlong`
   *despite* the worst raw latency — a full-trajectory fairness mechanism
   no bounded window replicates).
3. **`HORIZON_TOO_SHORT`** — moderate, secondary evidence. §7 (H≤20+200 is
   ~0.7% of a full scenario, ~1 median episode), §10 (H=20's correlation is
   the only one that turns positive) — but §8/§9 show commitment/run-length
   statistics barely move with H, and H=20 does not outperform H=5 on mean
   ANWG, so horizon length alone does not explain the pattern; it is
   necessary but not sufficient.
4. **`ORACLE_HEADROOM_CONCENTRATED`** — partially true, regime-qualified.
   §12–§13: not concentrated by *scenario* (top-5 = 38% of mass), but
   sharply concentrated by *regime* (89% of headroom is `favshort`-sourced,
   `favlong` headroom is negative on average). The label as stated is
   imprecise; the regime-conditional version is well supported.
5. **`PLAN_EXECUTION_MISMATCH`** — weak evidence. §8–§9: real but mixed
   (49% immediate-reversal rate, but 71–73% same-winner persistence
   between consecutive decisions), and does not vary materially with `H`,
   which is inconsistent with this being the dominant driver of the
   H1-vs-H5-vs-H20 differences actually observed.

## 16. Exact Next-Step Decision

**`JUSTIFY_TERMINAL_VALUE_REDESIGN`**

The dominant, best-evidenced explanation is that the windowed rollout
objective (and the raw-completion counterfactual signal underlying the
prior stateful controller's training label too) is poorly aligned with
final priority/SLO-weighted ANWG, specifically in the `favlong` regime
where WFS's long-run fairness-debt protection is what actually matters.
`JUSTIFY_COMMITMENT_AWARE_ROLLOUT_TEST` is not supported (§8–§9: weak,
H-invariant evidence). `JUSTIFY_LONGER_HORIZON_VALUE_TEST` is not supported
on its own (§7/§14 of the task: only justified if objective/commitment
semantics otherwise look sound — they do not, per §10).
`JUSTIFY_ON_POLICY_DAGGER_DIRECTION` is premature — the primary open
problem is what is being scored, not a state-distribution-mismatch problem
that only a learned policy/value function would address.
`NO_FURTHER_FAMILY_A_CONSTRUCTIVE_TEST` is not supported — a specific,
identified, and plausibly fixable definitional flaw exists (§6, §10, §11),
not a broad structural dead end.

## 17. Limitations

- The common-continuation-bias check (§6) uses raw completed-count, the
  only signal in the existing counterfactual artifact — no weighted/SLO
  field exists there, so it cannot directly indict or exonerate the V1
  rollout's *weighted* window objective, only the raw-completion signal
  both this and the prior controller are partly built on.
- §5's cross-reference uses the repaired diagnostic's 91 sampled
  disagreement events (32/64 scenarios), a different (smaller, differently
  triggered) sample than the rollout's own 2,700+ eligible decisions per
  horizon — directionally consistent, not identically sampled.
- §14's levels D/E are explicitly not apples-to-apples ANWG bounds; they
  are directional corroboration only, per the task's prohibition on new
  expensive counterfactual simulation.
- §4/§10's within-regime correlations use scenario-level aggregates (n=32
  or n=20 per regime); this is suggestive, well-powered enough for the
  significant results reported, but not decision-level causal
  identification.
- No new TEST, public-trace, or real-serving analysis was performed (by
  design).

## 18. Reproducible Commands / Artifacts

- Prior scientific artifacts (read-only): `experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_results.json`,
  `..._per_scenario_results.csv`, `..._state_distribution.json`;
  `experiments/family_a_observability_continuation_v1/family_a_observability_continuation_events.csv`.
- New diagnostic-only re-run (unmodified controller, added logging):
  `python3 scripts/diagnose_family_a_receding_horizon_oracle_v1_decision_logs.py`
  → `experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_decision_logs.csv`
  (6,151,248 rows, 641.5s wall clock) and its planning-call-only filtered view
  `..._decision_logs_planning_only.csv` (8,195 rows).
- This report: `docs/current/family_a_rollout_value_limit_diagnosis_20260820.md`.
