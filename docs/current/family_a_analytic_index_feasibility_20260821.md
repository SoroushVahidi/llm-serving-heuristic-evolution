# Family-A Analytic Request-Index Feasibility Study

Date: 2026-08-21

Offline, diagnostic-only feasibility study for a **training-free analytic
request index** that trades completion value against SLO/deadline risk, over
the already-frozen 91 Family-A ESTF/WFS contested events. No new simulation
was run, no controller/policy/simulator code was touched, no TEST data was
read, no learned model was trained, no coefficient sweep was performed, and
nothing was staged/committed/pushed. All numbers come from one deterministic,
read-only script (`scripts/analyze_family_a_analytic_index_feasibility.py`)
over the already-extracted
`experiments/family_a_contested_request_value_diagnosis/constrained_formulation_event_table.csv`
(91 rows, produced by the prior constrained-formulation-feasibility audit).

---

## 1. Executive Verdict

Classification: **`ANALYTIC_INDEX_NO_GO`**

Next step: **`STOP_ANALYTIC_INDEX_DIRECTION`**

Five literature-motivated, coefficient-free index candidates were built from
the exact same causal state the existing ESTF/WFS policies already read
online (`priority`, `predicted_service_proxy`, unit-consistent `laxity_own`,
`queue_age`). None survives both of the following checks simultaneously:

1. **Not a regime-identity reconstruction.** Three of five candidates
   (`B` deadline-urgency, `C` generalized-cμ, `E` Whittle-inspired
   criticality-ratio) are **byte-for-byte identical** to a plain relative
   "higher-priority side wins" rule and to the hidden favlong→WFS /
   favshort→ESTF regime-equivalence rule — 100.0% prediction overlap with
   both, confirmed empirically, not estimated. This is exactly the collapse
   condition the task's `NO_GO` criterion names explicitly ("signal depends
   on regime identity").
2. **Correct two-risk quadrant semantics.** The two candidates that do not
   fully collapse (`A` weighted-shortest-service, `D` age-adjusted) get the
   *quadrant-conditional* direction backwards within `favlong`: both assign
   their **lowest** ESTF share to the `COMPLETION_ONLY` quadrant (events
   where ESTF should be preferred — no SLO risk, real completion benefit at
   stake) and a **higher** ESTF share to `SLO_RISK_ONLY` (events where ESTF
   should be suppressed) than to `COMPLETION_ONLY`. This is the opposite of
   the semantics defined in Task §10, which the task itself states matters
   more than aggregate accuracy.

The only rule in this study that gets the quadrant ordering right
(`COMPLETION_ONLY` highest, `SLO_RISK_ONLY`/`BOTH_RISKS` lowest, within
`favlong`) is the previously-built **`constrained_rule`** — which is *not*
training-free (grouped-CV logistic regression on the same two labels). No
coefficient-free analytic index tested here reproduces that ordering.

---

## 2. Preflight

| Check | Value |
|---|---|
| branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `8e1223beb58fd4d296061b6b48e3ba493714108f` |
| upstream | `origin/contextual-compositional-heuristics-20260731` |
| ahead/behind (upstream…HEAD) | 0 / 0 |
| worktrees | 1 (main only) |
| lock files | none found |
| tmux sessions | none (`no server running on /tmp/tmux-1000/default`) |
| active scientific jobs (`ps aux`) | none (only unrelated background processes: `unattended-upgrade-shutdown`, a user `uvicorn` web app, `update-manager`) |
| RAM | 24Gi free / 59Gi available of 62Gi |
| disk | 638G available of 835G (20% used) |
| load average | 0.30 / 0.13 / 0.04 |

`git status` shows one pre-existing modified test file
(`tests/test_decision_criticality_timescale_trainval_v1.py`) and a long list
of untracked files from prior sessions (docs/experiments/scripts/src),
including the analytic-index paths used here. This task updated only the
analytic-index report and deterministic offline analysis artifacts listed in
§16; all unrelated local changes were preserved. Nothing staged, stashed,
reset, cleaned, committed, or pushed by this task.

---

## 3. Reused Input, Not Re-Extracted

This study reuses `constrained_formulation_event_table.csv` (91 events,
already produced and integrity-checked by
`docs/current/family_a_constrained_formulation_feasibility_20260821.md`),
rather than re-deriving from the raw `contested_requests.csv`, because it
already carries, per event: per-side (`estf_`/`wfs_`) `priority`,
`predicted_service_proxy`, `queue_age`, `laxity_own` (unit-consistent,
verified below); the two frozen offline labels (`completion_benefit_label`,
`slo_risk_label`); the frozen `constrained_rule_pred`; the biased native
ground truth `gt_label`; the `fav` (favlong/favshort) stratum; and
predictions for every trivial-rule comparator already computed
(`pred_always_estf`, `pred_always_wfs`, `pred_priority_ge5`,
`pred_regime_equiv`, `pred_E`, `pred_A`). Reusing this avoids re-running the
four-branch bounded rollout extraction (already frozen, deterministic,
read-only) and keeps this study strictly analytic (no simulation of any
kind, not even a replay).

---

## 4. Candidate Index Formulas (coefficient-free, per Task §4–§5)

All five are computed independently for the ESTF-only and WFS-only request
of each contested pair; the side with the larger index is predicted.

| ID | Name | Formula | Free parameters |
|---|---|---|---|
| A | `WEIGHTED_SHORTEST_SERVICE` | `priority / predicted_service_proxy` | none |
| B | `DEADLINE_URGENCY_INDEX` | `priority / max(laxity_own, eps)` | `eps = 1e-9` (reused from `scoring.py::urgency_score`'s existing clamp, not invented) |
| C | `GENERALIZED_CMU_STYLE` | `priority × (1/max(laxity_own, eps)) / predicted_service_proxy` | same `eps` |
| D | `FAIRNESS_DEBT_ADJUSTED_INDEX` | `(priority/predicted_service_proxy) × (1 + queue_age)` | none (multiplicative "1 +" form chosen specifically so `queue_age = 0` is a no-op, avoiding any invented additive weight) |
| E | `WHITTLE_INSPIRED_DEADLINE_INDEX` | `priority × predicted_service_proxy / max(laxity_own, eps)` | same `eps` |

**Tie handling (Task §8, fixed in advance):** predict `ESTF` if
`I_estf > I_wfs`, `WFS` if `I_wfs > I_estf`, and resolve an exact tie to
`WFS` — matching the codebase's own existing fallback-to-WFS convention
(used by the receding-horizon oracle and stateful-controller candidate-region
fallback outside the eligible region). No exact ties occurred in the 91×5
= 455 index evaluations (all floats distinct).

No candidate was tuned against TRAIN/VAL outcomes; no threshold or
coefficient grid was swept.

---

## 5. Dimensional / Unit Audit (Task §6, P0)

| Quantity | Units | Notes |
|---|---:|---|
| `priority` / `weight` | unitless (relative weight) | safe to multiply/divide against anything |
| `predicted_service_proxy` | raw token-count proxy ("steps"); **not** converted to real time. `scoring.py::deadline_slack`'s own docstring: *"service_proxy is in steps; convert to seconds via step_size if needed. Phase 1 leaves it unit-less."* | contested-row means 228–960 |
| `slo_deadline`, `arrival_time`, `state.time` | real simulation time (same absolute clock) | contested-row means 2.5–18.7 |
| `laxity_own = slo_deadline − state.time` | real time | **dimensionally valid** — both terms real time, confirmed non-degenerate by the prior constrained-formulation audit (only 1.1–3.3% negative, vs. 100% negative for the unit-mixed `deadline_slack_if_admitted_now`) |
| `queue_age = state.time − arrival_time` | real time | valid |
| `max_class_deficit_ratio` | unitless ratio (`demand / (served+1)`) | event-level aggregate, not per-request (see §9) |

**Reject list, applied:** `deadline_slack_if_admitted_now` (real time minus a
raw token count, 1–2 orders of magnitude apart) is **not used in any
candidate** — this is exactly the previously-diagnosed
`INHERENTLY_DEGENERATE_FOR_ABSOLUTE_GATING_BY_DESIGN` quantity. All five
candidates instead use `laxity_own` for any deadline term.

**Caveat carried forward honestly (Candidates C and E):** both combine
`predicted_service_proxy` (steps) and `laxity_own` (real time) in a
**product or quotient**, not a subtraction. A ratio/product of differently-
scaled quantities is not the same error as subtracting them (it does not
silently compare "0 vs 0" across incompatible scales the way
`deadline_slack`'s `>= 0` gate does) — each term is individually well-defined
and the resulting quantity can still be used to *rank* two candidates
validly. But **its interpretation as a literal cμ-rule or Whittle-index
value in real physical units is not justified**, because no validated
steps→seconds conversion factor exists in this codebase (`scoring.py`
explicitly declines to provide one). C and E are therefore labeled
`_STYLE` / `_INSPIRED` rather than claimed as exact derivations, per Task §4
and §14's instruction.

No candidate index was rejected outright on unit grounds; C and E are
flagged, not excluded, and both nonetheless turn out to reduce to pure
regime reconstruction empirically (§8) — a distinct, independently
sufficient reason to disqualify them.

---

## 6. Online Causal Inputs Used

Every field used by every candidate — `priority`, `predicted_service_proxy`
(derived from `prompt_tokens`/`predicted_output_tokens`), `laxity_own`
(derived from `slo_deadline`/`arrival_time`/`state.time`), `queue_age` — is
`ONLINE_CAUSAL`, per the inventory already established in
`docs/current/family_a_constrained_formulation_feasibility_20260821.md` §4
and `docs/current/family_a_contested_request_value_diagnosis_20260821.md`
§12. None of `favlong`/`favshort` (used only as an analysis stratum, never
as a feature), `canonical_scenario_id`, `split`, `seed`, or any `br_*`
future-outcome field is read by any candidate formula. The two offline
labels (`completion_benefit_label`, `slo_risk_label`) and the biased native
`gt_label` are used **only as evaluation targets**, never as index inputs.

---

## 7. Reference Targets Used (Task §9)

| Ref | Source | Trustworthy for | Caveat |
|---|---|---|---|
| A. completion-benefit | `completion_benefit_label` (ESTF-only completes under `br_estf_estf`, not under `br_wfs_wfs`) | completion | frozen label from bounded (≤1500-step) branch rollouts, not full-scenario |
| B. SLO-protection | `slo_risk_label` (WFS-only not completed-and-SLO-safe under `br_estf_estf`) | SLO/timeliness | same rollout-window caveat |
| C. constrained rule | `constrained_rule_pred` (prior grouped-CV logistic-regression rule, 0.740/0.892 balanced accuracy on A/B respectively) | overall two-risk direction | **not training-free** — used only as an upper-reference, not a target this study's candidates are expected to match by construction |
| D. native/full-path | `gt_label` (sign of `delta_native_whole_branch_raw`) | **nothing reliably in `favlong`** | documented ESTF-biased: `favlong` ground truth is 58 ESTF / 2 TIE / **0 WFS** among 60 events (i.e. this signal structurally cannot validate a WFS-favoring choice in `favlong`, regardless of whether that choice is correct) |

No new ground truth was manufactured; all four references pre-exist this
study.

---

## 8. Results — Completion-Benefit / SLO-Protection / Overlap

### 8a. Headline metrics, `ALL` (n=91)

| Candidate | ESTF share | Completion-benefit recall | SLO-risk protection recall | False-ESTF rate | False-WFS rate | Bal.acc vs `gt_label` (biased) | Bal.acc vs `constrained_rule` |
|---|---:|---:|---:|---:|---:|---:|---:|
| A `WSS` | 45.1% | 0.300 | 0.861 | 0.122 | 0.560 | 0.086 | 0.610 |
| B `DUI` | 34.1% | 0.275 | 0.972 | 0.032 | 0.483 | 0.063 | 0.681 |
| C `CMU` | 34.1% | 0.275 | 0.972 | 0.032 | 0.483 | 0.063 | 0.681 |
| D `FDA` | 53.8% | 0.375 | 0.667 | 0.245 | 0.595 | 0.126 | 0.552 |
| E `WHT` | 34.1% | 0.275 | 0.972 | 0.032 | 0.483 | 0.063 | 0.681 |
| *constrained_rule (ref.)* | 23.1% | 0.325 | 0.944 | 0.095 | 0.386 | 0.402 | 1.000 |
| *always-ESTF* | 100% | 1.000 | 0.000 | 0.396 | — | 0.333 | — |
| *always-WFS* | 0% | 0.000 | 1.000 | — | 0.440 | 0.333 | — |

B, C, E are numerically identical in every column — confirmed to be the
**exact same predictions**, not merely similar performance (§8b).

### 8a-2. Stratified candidate metrics, `ALL` / `favlong` / `favshort`

| Candidate | Stratum | n | ESTF share | Completion-benefit recall | SLO-risk protection recall | False-ESTF rate | False-WFS rate | Bal.acc vs `gt_label` | Macro F1 vs `gt_label` | Bal.acc vs `constrained_rule` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A `WSS` | ALL | 91 | 0.451 | 0.300 | 0.861 | 0.122 | 0.560 | 0.086 | 0.101 | 0.610 |
| A `WSS` | favlong | 60 | 0.167 | 0.034 | 0.886 | 0.400 | 0.560 | 0.043 | 0.047 | 0.404 |
| A `WSS` | favshort | 31 | 1.000 | 1.000 | 0.000 | 0.032 | n/a | 0.333 | 0.175 | n/a |
| B `DUI` | ALL | 91 | 0.341 | 0.275 | 0.972 | 0.032 | 0.483 | 0.063 | 0.082 | 0.681 |
| B `DUI` | favlong | 60 | 0.000 | 0.000 | 1.000 | n/a | 0.483 | 0.000 | 0.000 | n/a |
| B `DUI` | favshort | 31 | 1.000 | 1.000 | 0.000 | 0.032 | n/a | 0.333 | 0.175 | n/a |
| C `CMU` | ALL | 91 | 0.341 | 0.275 | 0.972 | 0.032 | 0.483 | 0.063 | 0.082 | 0.681 |
| C `CMU` | favlong | 60 | 0.000 | 0.000 | 1.000 | n/a | 0.483 | 0.000 | 0.000 | n/a |
| C `CMU` | favshort | 31 | 1.000 | 1.000 | 0.000 | 0.032 | n/a | 0.333 | 0.175 | n/a |
| D `FDA` | ALL | 91 | 0.538 | 0.375 | 0.667 | 0.245 | 0.595 | 0.126 | 0.137 | 0.552 |
| D `FDA` | favlong | 60 | 0.300 | 0.138 | 0.686 | 0.611 | 0.595 | 0.117 | 0.113 | 0.327 |
| D `FDA` | favshort | 31 | 1.000 | 1.000 | 0.000 | 0.032 | n/a | 0.333 | 0.175 | n/a |
| E `WHT` | ALL | 91 | 0.341 | 0.275 | 0.972 | 0.032 | 0.483 | 0.063 | 0.082 | 0.681 |
| E `WHT` | favlong | 60 | 0.000 | 0.000 | 1.000 | n/a | 0.483 | 0.000 | 0.000 | n/a |
| E `WHT` | favshort | 31 | 1.000 | 1.000 | 0.000 | 0.032 | n/a | 0.333 | 0.175 | n/a |

`n/a` means the rule never predicted the relevant side in that stratum, so
the conditional false-rate denominator is zero; for example, B/C/E never
predict ESTF in `favlong`, and A/B/C/D/E never predict WFS in `favshort`.

### 8b. Overlap / triviality check (Task §13)

| Candidate | Overlap vs. relative-priority-only rule | Overlap vs. regime-equivalence rule (favlong→WFS, favshort→ESTF) | Overlap vs. always-WFS |
|---|---:|---:|---:|
| A `WSS` | **89.0%** | **89.0%** | 54.9% |
| B `DUI` | **100.0%** | **100.0%** | 65.9% |
| C `CMU` | **100.0%** | **100.0%** | 65.9% |
| D `FDA` | **80.2%** | **80.2%** | 46.2% |
| E `WHT` | **100.0%** | **100.0%** | 65.9% |

Additional confirmed identities in this dataset (not assumed, computed):

- `priority-only-relative` rule ≡ `priority≥5` rule ≡ `regime-equivalence`
  rule, **100.0%** mutual overlap. In this scenario family, priority is
  (by construction) an almost-perfect regime label: every `favlong`
  ESTF-only request has `priority=1.0`, every `favlong` WFS-only request has
  `priority≥5`, and the pattern flips exactly in `favshort` (already
  established in the prior contested-value diagnosis, §4). Any index whose
  ranking is dominated by the `priority` factor inherits this collapse for
  free — this is exactly what happened to B, C, E.
- `remaining-service-only` rule (shorter service wins) ≡ `always-ESTF`,
  **100.0%**, because the WFS-only request has strictly larger
  `predicted_service_proxy` than the ESTF-only request in **91/91** events
  (0 exceptions) — confirming Task §13's named trivial pattern "long
  request → WFS" is, in this dataset, indistinguishable from "always WFS."

**Verdict: B, C, and E add zero state-sensitive information beyond regime
identity in this dataset.** A and D are not literal regime reconstructions
(89.0% / 80.2% overlap, not 100%) but see §8c for why that partial
independence does not translate into correct decision behavior.

### 8c. Two-risk quadrant behavior (Task §10 — "more important than aggregate accuracy")

ESTF share by quadrant, `favlong` only (n=60; `BOTH_RISKS`=21,
`SLO_RISK_ONLY`=14, `COMPLETION_ONLY`=8, `NEITHER`=17):

| Rule | BOTH_RISKS (want: low) | SLO_RISK_ONLY (want: low) | COMPLETION_ONLY (want: **high**) | NEITHER |
|---|---:|---:|---:|---:|
| A `WSS` | 4.8% | 21.4% | **0.0%** | 35.3% |
| D `FDA` | 19.0% | 50.0% | **0.0%** | 41.2% |
| B/C/E | (identical to priority≥5: `favlong` ESTF share fixed at 0% in every quadrant — priority=1.0 for every `favlong` ESTF-only request, so the urgency/service term never overturns it) | | | |
| *constrained_rule (ref.)* | 9.5% | **0.0%** | **62.5%** | 5.9% |
| *age-only-relative* | 61.9% | 64.3% | 100.0% | 70.6% |

**This is the decisive finding.** Candidates A and D both assign their
**lowest** ESTF share to exactly the quadrant where ESTF should be *most*
preferred (`COMPLETION_ONLY`: real completion value at stake, no SLO risk),
and a **higher** ESTF share to `SLO_RISK_ONLY` than to `COMPLETION_ONLY` —
the reverse of the intended two-risk semantics. Only `constrained_rule`
(fitted, not training-free) gets the ordering right
(`COMPLETION_ONLY` 62.5% ≫ `SLO_RISK_ONLY` 0.0%). The naive `age-only`
comparator is uniformly high across all four quadrants (61.9–100%) — it
does not discriminate risk type at all, it just favors the ESTF-only side's
typically-shorter wait almost everywhere.

`favshort` quadrants are far less informative for this check: `n=1` for
`BOTH_RISKS`, `n=0` for `SLO_RISK_ONLY` (recall `favshort` has only 1 WFS
label in `gt_label` and near-zero SLO-risk prevalence, 3.2%, per the prior
contested-value diagnosis) — every candidate scores 100% ESTF in
`favshort`'s `COMPLETION_ONLY`/`NEITHER` quadrants, which is directionally
fine there but not a meaningful discrimination test (favshort's risk
quadrants are nearly empty by construction).

---

## 9. Virtual-Debt / Primal-Dual Diagnostic (Task §15)

Constructed exactly one diagnostic form, as instructed: completion-efficiency
index (`A`) adjusted by the existing fairness-debt pressure signal
(`max_class_deficit_ratio`, already computed online by
`family_a_observability_continuation_v1.py` and structurally identical to
`WeightedFairSharePolicy._score`'s own live deficit term):

```
I_D_aggregate(side) = (priority/service_proxy) × (1 + max_class_deficit_ratio_event)
```

**Result: 0 / 91 decision flips relative to `A` alone.** This is not a weak
correlation — it is a mathematical certainty, verified rather than argued:
`max_class_deficit_ratio` is computed **once per event** (an aggregate max
over all classes in the queue at that decision point), so it is
**identical for the ESTF-only and WFS-only side of the same event**.
Multiplying both sides of a pairwise `argmax` comparison by the same
positive scalar cannot change which side is larger. **The only existing
fairness-debt signal in the causal feature set is structurally incapable of
informing this specific per-event pairwise choice**, independent of whether
deficit is "the right idea" — the signal would need to be computed
*per-request/per-class*, which is not available in the existing extraction
without new simulation (out of scope here; flagged as a limitation, §12).

Comparison "does it handle SLO-risk events better than the pure
service-time index": **no differently at all** — it is byte-identical to
`A` on every metric in §8, by construction.

---

## 10. Indexability / Whittle-Theory Audit (Task §14)

| Assumption | Status |
|---|---|
| Separable per-request state | Satisfied — each request's (laxity, remaining service) evolves independently once queued |
| Binary serve/passive action per arm | Satisfied — each request is either admitted (served) or left waiting, per step |
| Capacity coupling via Lagrangian relaxation | Not derived here — Family-A GPUs have `max_active_sequences=1` (hard capacity 1 per GPU, no fractional relaxation attempted) |
| Markov evolution | Plausible for the *passive* arm (deterministic laxity decay while waiting) but **no preemption exists once admitted** (confirmed empirically in the prior contested-value diagnosis: `br_wfs_wfs`≡`br_wfs_estf` and `br_estf_estf`≡`br_estf_wfs` for every one of 182 rows) — once served, a request's fate is already sealed, which is a *simpler* structure than the general restless-bandit "arm keeps evolving under either action" setting, not a strict violation but a structural simplification the classical theory does not need to assume |
| Reward structure suited to a subsidy/index decomposition | Plausible in form (binary weight-if-on-time reward) but no subsidized single-arm relaxed MDP was solved |
| Indexability (monotone passive set as subsidy varies) | **Not verified** — would require solving the per-arm subsidized MDP explicitly; not attempted (would require new derivation/simulation beyond this offline audit's scope) |
| A valid steps↔seconds conversion for `predicted_service_proxy` | **Does not exist in this codebase** (§5) — this alone blocks constructing a rigorous per-request completion-probability term, which any faithful Whittle derivation for deadline scheduling would need |

**Conclusion: `WHITTLE_MAPPING_NOT_JUSTIFIED`.** Neither indexability nor a
valid subsidized-MDP index was derived; the units gap in §5 is a concrete,
named blocker, not just an unproven-but-plausible gap. Candidate `E` is
therefore explicitly labeled `WHITTLE_INSPIRED` (a recognizable
"least-laxity-per-unit-processing, priority-weighted" criticality ratio from
the real-time/restless-bandit deadline-scheduling literature), never
`WHITTLE_OPTIMAL`. Empirically, `E` is in any case identical to `B`/`C`
(100% overlap, §8b) — the theoretical caveat is moot in practice here, since
`E` collapses to regime reconstruction regardless of its Whittle framing.

---

## 11. Comparison Against Trivial/Simple Rules (Task §12)

| Rule | ALL bal.acc vs `gt_label` | favlong | favshort | ALL bal.acc vs `constrained_rule` |
|---|---:|---:|---:|---:|
| always-ESTF | 0.333 | 0.500 | 0.333 | — |
| always-WFS | 0.333 | 0.000 | 0.333 | — |
| priority-only (relative) | 0.063 | 0.000 | 0.333 | 0.681 |
| remaining-service-only (relative) | 0.333 (≡always-ESTF) | 0.500 | 0.333 | — |
| age-only (relative) | 0.546 | 0.351 | 0.455 | 0.555 |
| prior contested proxy `E` (age-protection) | 0.063 | 0.043 | 0.212 | 0.617 |
| prior contested proxy `A` (completion-only; tautological vs `gt_label`, see prior report's caveat) | 1.000 | 1.000 | 1.000 | 0.293 |
| prior constrained rule | 0.402 | 0.075 | 0.485 | 1.000 |
| **A `WSS`** | 0.086 | 0.043 | 0.333 | 0.610 |
| **B/C/E** | 0.063 | 0.063 | 0.333 | 0.681 |
| **D `FDA`** | 0.126 | 0.117 | 0.333 | 0.552 |

None of the five analytic candidates beats the naive `age-only` baseline's
`ALL` balanced accuracy against `gt_label` (0.546); several fall below the
`priority-only`/regime-equivalent floor. Against the `constrained_rule`
reference, B/C/E (0.681) edge out A (0.610) and D (0.552) — but this is
exactly because B/C/E are the priority/regime-collapsed rules, and
`constrained_rule`'s own predictions correlate strongly with priority too
(the prior report's own §14 already found 71.4% overlap between
`constrained_rule` and `priority≥5`) — so this is not independent evidence
of analytic quality, it is two regime-correlated rules agreeing with each
other.

---

## 12. Is the Best-Looking Index Just Regime Reconstruction? (Task §13)

**Yes, for B, C, and E — proven, not inferred: 100.0% prediction overlap
with the plain relative-priority rule and with the hidden
favlong→WFS/favshort→ESTF regime-equivalence rule.** These three add no
information beyond "which side has the locally larger `priority`," which in
turn is (in this scenario family) an almost-deterministic function of
regime.

**A and D are not literal reconstructions** (89.0% / 80.2% overlap — a
genuine ~10–20% disagreement footprint with the priority/regime rule) but
that disagreement does not point in a decision-useful direction: within
`favlong`, both assign their *lowest* ESTF-share to the quadrant
(`COMPLETION_ONLY`) where ESTF is actually correct, and a higher share to
`SLO_RISK_ONLY` where it is not (§8c). The non-collapse is real but is not
evidence of a working two-risk index — it is evidence that `predicted_service_proxy`
and `queue_age` inject *some* per-request variation into the ranking, just
not variation aligned with the risk structure the task asked the index to
capture.

---

## 13. Classification

**`ANALYTIC_INDEX_NO_GO`**

- Three of five candidates are provable regime-identity reconstructions
  (100.0% overlap) — an explicit, named `NO_GO` trigger.
- The remaining two candidates do not reconstruct regime identity but fail
  the two-risk quadrant-semantics test in the *wrong direction*
  (§8c) — the task explicitly weights this check above aggregate accuracy.
- The one diagnostic that could in principle differentiate the pair on a
  fairness/debt axis (§9) is proven mathematically incapable of doing so
  with the only existing debt signal in the current online feature set.
- No candidate clears the naive `age-only` trivial baseline's aggregate
  accuracy either (§11), though that comparison is secondary to the
  quadrant finding.
- `STRONG` requires an index that "protects SLO-risk events while retaining
  meaningful ESTF completion gains... in both favshort and favlong... does
  not merely reconstruct regime identity." No candidate satisfies this
  jointly. `PARTIAL` ("useful structure exists but incomplete/unstable") is
  not chosen because the failure mode found (backwards quadrant ordering,
  provable regime collapse) is a structural mismatch between the available
  causal state and a training-free linear/multiplicative combination of it,
  not merely a noisy or unstable-but-promising signal.

---

## 14. Exact Next Step

**`STOP_ANALYTIC_INDEX_DIRECTION`**

Per the task's own instruction, this is named, not executed. The
constructive path that *does* show quadrant-correct behavior on this same
data — the previously-built `constrained_rule` — is a fitted grouped-CV
logistic regression, not a training-free index; this study does not
recommend reopening that direction, only records that it is the one
approach in the current diagnostic chain that clears the two-risk semantics
bar the analytic candidates here do not.

---

## 15. Limitations

- **Small n.** 91 events (32/64 scenarios); quadrant-level breakdowns
  within `favlong`/`favshort` have cells as small as `n=1` or `n=0`
  (`favshort`'s `BOTH_RISKS`/`SLO_RISK_ONLY`), limiting how much can be
  concluded from `favshort` specifically.
- **Ground-truth caveat repeated.** `gt_label` (native raw-completion sign)
  is independently documented as ESTF-biased and structurally unable to
  validate any WFS-favoring choice in `favlong` (0/60 WFS labels there);
  low agreement against it in `favlong` is therefore ambiguous on its own —
  the quadrant-semantics finding (§8c, using the two frozen risk labels,
  not `gt_label`) is the load-bearing evidence for this study's
  classification, not the `gt_label` agreement numbers.
- **Whittle audit is a theory/derivation audit, not an empirical
  optimality test** — `WHITTLE_MAPPING_NOT_JUSTIFIED` reflects that
  indexability was not verified and a units blocker exists, not that a
  Whittle index was derived and found to underperform.
- **Only one virtual-debt diagnostic form was tried** (§9), per the task's
  "construct only one" instruction; a per-request/per-class deficit
  variant (which would require new simulation to extract) might behave
  differently and is explicitly not tested here.
- **Five candidates, not an exhaustive search** of the classical
  index-scheduling literature — per the task's "implement only a SMALL
  set" instruction. Other coefficient-free forms (e.g., a true per-class
  deficit-weighted index, if extracted with new simulation) are not ruled
  out by this study.
- TRAIN/VAL only; no TEST, public-trace, or real-serving validation.

---

## 16. Reproducible Commands / Artifacts

Inputs read (all pre-existing, none modified):

- `experiments/family_a_contested_request_value_diagnosis/constrained_formulation_event_table.csv`
- `docs/current/family_a_constrained_formulation_feasibility_20260821.md`
- `docs/current/family_a_contested_request_value_diagnosis_20260821.md`
- `docs/current/family_a_terminal_value_v1_analysis_20260820.md`
- `docs/current/family_a_rollout_value_limit_diagnosis_20260820.md`
- `docs/current/family_a_receding_horizon_oracle_v1_analysis_20260820.md`
- `src/llmserveopt/policies/scoring.py`, `src/llmserveopt/policies/estimated_service_time_first.py`,
  `src/llmserveopt/policies/weighted_fair_share.py`, `src/llmserveopt/core/types.py`
  (read for exact semantics/units, not modified)

Artifacts created/updated for this study (deterministic, read-only over the
above; no simulation):

- `scripts/analyze_family_a_analytic_index_feasibility.py`
- `experiments/family_a_analytic_index_feasibility/analytic_index_feasibility_summary.json`
- `experiments/family_a_analytic_index_feasibility/analytic_index_event_table.csv`
- This report: `docs/current/family_a_analytic_index_feasibility_20260821.md`

Reproduce:
`python3 scripts/analyze_family_a_analytic_index_feasibility.py`
(pure pandas/numpy/scikit-learn over one existing CSV; no simulator import,
no RNG, no new scenario execution; wall clock < 2s).
