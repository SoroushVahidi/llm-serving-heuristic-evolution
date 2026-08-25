# Decision-Criticality Terminal-ANWG on Joint-240 v1 — Design / Preregistration

Date: 2026-08-25  
Status: **FROZEN BEFORE SCORING**  
Experiment: `decision_criticality_terminal_anwg_joint240_v1`  
Parent estimand: `docs/design/DECISION_CRITICALITY_TERMINAL_ANWG_V1.md`  
Workload source: `experiments/joint240_same_distribution_adaptive_exploitability_v1/`  
+ frozen joint scenarios from `experiments/joint_multimechanism_generalization_v1/` (`SEED=20260824`)

## 0. Scope and non-goals

Same-distribution falsification test: does the A/B/C TRAIN/VAL terminal-ANWG
criticality pattern reproduce on the **exact joint-240 workload** used for
Section 4.2 same-distribution exploitability (SBS/VBS/Ascen/Alive)?

- Does **not** overwrite `experiments/decision_criticality_terminal_anwg_v1/`.
- Does **not** retrain or modify the frozen Alive router beyond identical OOF
  Stage-1 refits required to reconstruct fold-conditional Alive.
- Does **not** edit the manuscript.
- Does **not** pool joint-240 with the A/B/C corpus.
- Does **not** invent a new H10 proxy if the old completed-count events cannot
  be joined on these states.

## 1. Primary scientific question

> On the exact joint-240 workload where SBS/VBS headroom is positive and
> Ascen/Alive both underperform SBS, are individual scheduler decisions also
> sparse and concentrated in terminal ANWG consequence?

Secondary (preregistered):

1. Prevalence of nonzero one-step terminal ANWG effects.
2. Effect-mass concentration over states and scenarios.
3. Whether P6 native disagreement predicts terminally consequential actions.
4. Whether the prior H10 completed-count proxy can be joined; if yes, association
   with terminal criticality; if no, report unavailable.
5. Pressure-regime stratification using **frozen** joint-240 pressure flags.
6. Downstream trajectory divergence among zero vs nonzero forks.
7. Side-by-side contrast with `decision_criticality_terminal_anwg_v1` (not pooled).

## 2. Estimand (continuation-policy-conditional)

For a selected decision state at time \(t\):

1. Clone the exact pre-decision simulator state.
2. **Reference branch:** take the Alive reference action \(a_{\mathrm{ref}}\).
3. **Counterfactual branch:** force one alternative genuine P6-native action
   \(a_{\mathrm{alt}}\).
4. After that one action, **both** branches return to the **same cloned**
   Alive continuation-policy state.
5. Identical future arrivals / randomness (fork copies the not-yet-enqueued
   pending suffix; deterministic simulator).
6. Run both branches to natural termination via `Simulator.continue_run`.
7. Measure

\[
\Delta_{\mathrm{terminal}} = \mathrm{ANWG}_{\mathrm{cf}} - \mathrm{ANWG}_{\mathrm{ref}}.
\]

**Label (mandatory):** this is a
**continuation-policy-conditional one-step terminal effect**,
**not** a policy-independent Q-value.

Primary reported CF contrast uses the untouched Alive reference terminal ANWG
as \(\mathrm{ANWG}_{\mathrm{ref}}\) (with mandatory REF-replay integrity checks).

## 3. Continuation policy = frozen Section 4.2 Alive

**Choice:** common continuation policy = exact OOF `A_live` /
`LiveP6DwellRouterPolicy` from
`joint240_same_distribution_adaptive_exploitability_v1`.

Rationale: Alive is the online adaptive method in Section 4.2; Ascen is
scenario-level (no per-step decision states). Using Alive makes the estimand
directly conditional on the same adaptive continuation that underperformed SBS.

Reconstruction protocol (identical to frozen joint-240 runner):

- Scenarios: `rebuild_all_scenarios()` with joint runner `SEED=20260824`.
- Folds: load frozen `split_oof_folds.csv` (seed `20260825`, stratified on
  `n_elevated_mechanisms`).
- Per fold \(k\): fit Stage-1 LogReg (`C=1.0`, balanced, max_iter=2000) on probe
  telemetry from train+val scenarios of that fold
  (`probe_policy=weighted_fair_share`, VBS-winner labels), dwell=20.
- For scenario \(s\) with OOF fold \(k\): run Alive with Stage-1 from fold \(k\).

Cloned continuation state includes: Stage-1 pipeline (shared read-only),
`PolicyDwellFSM` (deep-copied), native P6 policy instances (deep-copied),
`switch_count`, `_last_policy`, and `selected_policies` prefix.

No router weights are updated from this experiment’s outcomes.

## 4. Joint-240 extension of the native-pair rule

Family A/B/C used Stage-2 **native pairs**. Joint-240 Alive is **6-way P6**.
Documented extension (outcome-blind):

**P6 set (frozen order):**
`full_prefill`, `chunked_prefill_small`, `estimated_service_time_first`,
`weighted_fair_share`, `least_laxity_first`, `kv_constrained_online`.

**Action canonicalization:** admit-set only (same as parent terminal-ANWG v1).

**Eligible decision steps:** `len(waiting_queue) > 0` (admission opportunity).

**DISAGREEMENT:** at least one other P6 policy produces a different canonical
admit-set than Alive’s \(a_{\mathrm{ref}}\) on a deep-copied state.
\(a_{\mathrm{alt}}\) = action of the **first** disagreeing policy in frozen P6
order.

**AGREEMENT_CONTROL:** no other P6 policy disagrees on admit-set.
\(a_{\mathrm{alt}}\) = action of the **next** policy in cyclic P6 order after
Alive’s effective policy (partner-force control; may yield \(\Delta=0\)).

## 5. Sampling (preregistered; outcome-blind)

- Corpus: all **240** frozen joint scenarios.
- Per scenario, in trajectory order:
  - first `MAX_DISAGREEMENT_PER_SCENARIO = 10` disagreement states;
  - first `MAX_AGREEMENT_CONTROL_PER_SCENARIO = 5` agreement-control states.
- If fewer exist, take all.
- Theoretical cap: \(240 \times 15 = 3600\) intervention states.
- Seeds: `CONTROL_SEED = 20260825`, `BOOTSTRAP_SEED = 20260825`,
  experiment seed provenance `20260825`.
- Online evaluation cannot rewind; no outcome-dependent resampling.

## 6. Reference-replay integrity (mandatory)

Once per scenario with ≥1 acquired state: fork, apply \(a_{\mathrm{ref}}\),
continue with Alive clone. Require

\[
|\mathrm{ANWG}_{\mathrm{ref\_replay}} - \mathrm{ANWG}_{\mathrm{ref}}| \le 10^{-12}.
\]

Report n_checked, n_matched, max abs mismatch. If material failure: **STOP**;
do not interpret CF results.

## 7. Primary analyses

1. Acquired state counts; fractions \(\Delta=0\), \(\Delta>0\), \(\Delta<0\),
   \(|\Delta|\ge 0.01\).
2. Mean/median/quantiles of \(\Delta\) and \(|\Delta|\).
3. Scenario prevalence of ≥1 nonzero effect; effects/scenario.
4. State effect-mass concentration of \(|\Delta|\) at top 1/5/10%.
5. Positive-effect concentration of \(\max(\Delta,0)\) at top 1/5/10%.
6. Scenario-aggregated concentration: top 1, 2, 5 scenarios; top 5%/10% scenarios.
7. Pressure strata from frozen manifest flags (thresholds **not** retuned):
   `high_fairness_pressure`, `high_service_heterogeneity`,
   `high_prefill_decode_pressure`, `high_kv_pressure`, `high_urgency_pressure`,
   `high_burst_pressure`, and bins of `n_elevated_mechanisms` {0–1, 2, 3, ≥4}.

## 8. Disagreement as criticality proxy

Predict \(1[|\Delta_{\mathrm{terminal}}| > 10^{-12}]\) from
`acquisition_type == DISAGREEMENT`.

Report: prevalence in disagreement vs agreement; enrichment ratio; base rate;
AUROC; AUPRC (mandatory); scenario-grouped bootstrap CIs for AUROC/AUPRC/
enrichment.

## 9. H10 proxy

Attempt join only if the prior H10 completed-count events share
`(scenario_id, step)` with joint-240 Alive trajectories **without redefining**
H10. Expected outcome: **unavailable** (H10 corpus is A/B/C TRAIN/VAL). If
unavailable, state that explicitly — do not invent a joint-240 H10.

## 10. Closed-loop divergence

Per fork, record whether CF diverges downstream via completion-count delta,
sim-duration delta, or terminal utility effect (same conceptual proxies as
parent). Report divergence rates for nonzero vs zero \(|\Delta|\), extra steps,
and intervention step/time distributions.

## 11. Uncertainty (scenario-grouped bootstrap)

- \(B = 2000\), seed `20260825`.
- Resample **scenarios**, taking all states from selected scenarios.
- CIs (2.5/97.5%) for: nonzero prevalence; mean \(|\Delta|\); top-1/5/10% state
  mass shares; top-5 scenario concentration; disagreement enrichment; AUROC;
  AUPRC.
- **Zero-mass rule (preregistered):** if a bootstrap replicate has total
  \(\sum|\Delta|=0\), define all concentration shares as `0.0` for that
  replicate (not NaN; not dropped).

## 12. Verdict rules (frozen before outcomes)

Let \(p_{nz}\) = nonzero prevalence; \(m_{10}\) = top-10% state \(|\Delta|\) mass
share; \(n_{nz}\) = number of nonzero states; AUROC for disagreement→nonzero.

| Verdict | Criteria |
|---|---|
| `JOINT240_INSUFFICIENT_EFFECT_EVENTS` | \(n_{nz} < 10\) |
| `JOINT240_TERMINAL_CRITICALITY_REPLICATED` | \(n_{nz}\ge 10\) and \(p_{nz} < 0.15\) and \(m_{10} \ge 0.50\) and AUROC \(< 0.70\) |
| `JOINT240_CRITICALITY_PARTIAL_REPLICATION` | \(n_{nz}\ge 10\) and exactly one of (sparse \(p_{nz}<0.15\), concentrated \(m_{10}\ge 0.50\)) holds, or both hold but AUROC \(\ge 0.70\) |
| `JOINT240_CRITICALITY_NOT_REPLICATED` | \(n_{nz}\ge 10\) and (\(p_{nz} \ge 0.25\) or \(m_{10} < 0.30\)) |
| `JOINT240_DISAGREEMENT_PROXY_USEFUL` | enrichment \(> 1.5\) and AUROC \(\ge 0.70\) and AUROC CI excludes 0.5 |

Multiple labels may apply; primary criticality verdict is the first matching row
among the criticality rows (insufficient → replicated → partial → not).
Disagreement-proxy label is additive.

Do **not** require numerical equality with A/B/C point estimates.

## 13. Execution

- Prefer SLURM/Wulver if available; else local tmux
  `joint240_terminal_criticality_v1`.
- Log: `experiments/decision_criticality_terminal_anwg_joint240_v1/logs/full_run.log`.
- Dry-run 2–4 scenarios before full launch.
- After launch: ~3-minute health check only; hand off if still running.
