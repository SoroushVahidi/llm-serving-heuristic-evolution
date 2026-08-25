# Family-A Selector Closure And Pivot v1 Analysis

Date: 2026-08-24

## Executive Decision

Selector hypothesis status: `SELECTOR_HYPOTHESIS_FALSIFIED`

Chosen pivot: `PIVOT_TO_PORTFOLIO_POLICY_SYNTHESIS`

Anything heavy to launch next: **NO**

## Evidence Chain

- `pi0_offline`: `SUPPORTED_OFFLINE_SIGNAL` — selected pi0 DEV mean regret 1.115 vs WFS/majority 4.087 and ESTF 12.087; balanced sign accuracy 0.855 from frozen offline artifact
- `pi0_closed_loop`: `MIXED_SIGNAL` — closed-loop aggregate pi0 mean ANWG 0.854614 below WFS 0.859748; worst favored-long group regret 0.0463
- `D1`: `OFFLINE_NO_GO` — D1 status/classification: None / DAGGER_D1_OFFLINE_NO_GO; active acquisition did not pass frozen gates
- `D2_support_diagnosis`: `DO_NOT_RUN_D2` — scientific audit stop tree prohibited D2/oracle acquisition unless support gates passed; D1 did not solve support/offline transfer
- `bridge_pilot`: `FEATURE_ONLY_SIGNAL_NO_GATE` — 18 scenarios, 812 candidates, FEATURE_V1 0/104 DEV closer and favored-long 0/66 closer
- `configuration_state_surrogate`: `SURROGATE_GO_DIAGNOSTIC_ONLY` — surrogate predicted response/yield better than naive parameter NN, justifying target-free sweep but not scheduler success
- `wulver_target_free_sweep`: `MODERATE_TRAIN_EXPANSION` — 200 cells, 400 scenarios, 24,314 unique fingerprints, 156 novel TRAIN-side domains
- `frozen_dev_support_eval`: `NO_GO` — all 8 gates failed; 1/104 DEV rows closer; task subspace 0/104 rows closer

The frozen stop condition matched: target-free TRAIN expansion failed the frozen DEV support thresholds. Oracle acquisition is therefore not allowed next.

## Deepest Failure Modes

1. `CONFIGURATION_COMPOSITIONAL_GENERALIZATION_FAILURE` — implication: A non-DEV held-out group with matched configuration axes should remain poorly covered by target-free novelty unless the acquisition explicitly captures its compositional interaction.
2. `ENDOGENOUS_CLOSED_LOOP_SHIFT` — implication: A selector that looks good on static rows will generate a shifted trajectory distribution where its own high-confidence states have weak support or low utility contribution.
3. `POLICY_SELECTION_FORMULATION_ITSELF_IS_WRONG` — implication: Even with improved labels/support, a memoryless ESTF/WFS switcher will add little or negative portfolio value relative to a robust parent or a direct composite scheduler.
4. `NON_STATIONARY_OR_STATE_ALIASED_DECISION_PROBLEM` — implication: Adding causal context/history should reduce local label entropy only partially; exact or near-exact observable states will still sometimes imply opposite oracle parents.
5. `WRONG_FEATURES` — implication: Adding causal workload/history descriptors should reduce aliasing and improve static support, but not necessarily solve closed-loop value.
6. `WRONG_ACQUISITION` — implication: Another acquisition that is not DEV-aware would need a different mechanism, not simply more cells, to produce held-out support movement.
7. `INSUFFICIENT_DATA` — implication: If sheer volume were the main issue, broad target-free novelty should have produced broad DEV support movement; it did not.

## State Aliasing

Diagnostic rows: 998 total, 919 decisive, 79 ties; V2 DEV/FINAL excluded from label diagnostics.

- FEATURE_V1 kNN entropy mean: 0.0963; contradictory-neighbor fraction mean: 0.0427; any contradictory neighbor: 0.1882; regret-weighted contradictory fraction: 0.0212.
- FEATURE_V1 + configuration contradictory fraction: 0.0423; relative reduction vs state-only: 0.77%.
- FEATURE_V2 normalized contradictory fraction: 0.0457.
- Task subspace contradictory fraction: 0.0441.

Interpretation: observable-state aliasing is present. Configuration context helps but does not erase label conflicts, so the problem is not just missing scalar features.

## Parent Complementarity And Headroom

Oracle TRAIN/D1 best single parent by mean row utility: `WFS`.
Envelope advantage over WFS: 3.2585; over ESTF: 24.5511.
Rows with abs(delta)>=5: 86.37%; top 10% of decisions carry 58.39% of best-parent regret mass.

Closed-loop frozen aggregate still makes WFS the robust default, but oracle rows show real complementarity; the failed piece is statewise learned selection, not the existence of mechanisms.

## Configuration Stability

Scenarios: 262; configurations: 13.
Scenario-level dominant-parent median row fraction: 1.000; mixed-label scenario fraction: 0.065.
Configuration-level capture of pair-envelope gain over best single parent: 0.953.
Scenario-level capture of pair-envelope gain over best single parent: 0.979.

## Pivot Assessment

- Robust single scheduler: viable only as guarded mechanism synthesis; not the old static analytic index.
- Configuration-level selection: useful diagnostic baseline, but it remains parent selection and does not answer the new-scheduler objective.
- Portfolio-policy synthesis: best next direction because it can produce a genuinely new policy and use the ESTF/WFS mechanism evidence without reopening DEV-targeted support acquisition.

## Prior Analytic Index

The prior analytic-index study tried five coefficient-free static request indices. Three collapsed to priority/regime identity; the two non-collapsed forms inverted the critical favlong completion-vs-SLO quadrant semantics. Direct synthesis remains viable only with explicit guards/history/regime structure, not a single scalar index.

## Publication Value

Large target-free expansion of reachable TRAIN disagreement-state support did not resolve held-out decision-state support for statewise scheduling-policy selection.

Placement: central if paper is about certified oracle/support failure methodology; appendix if the final contribution becomes a new composite scheduler..

## Next Experiment Design

Name: `family_a_mechanism_composite_rule_static_feasibility_v1`

Hypothesis: A WFS-safe deterministic composite scheduler rule with explicit ESTF completion-release guards can recover meaningful oracle headroom on TRAIN/D1 contested states without collapsing to the failed static analytic-index patterns.

Benchmark: TRAIN/D1 only; no DEV, FINAL, TEST, Wulver jobs, new simulations, or oracle labels

Primary metric: regret to certified ESTF/WFS pair envelope on TRAIN/D1 oracle rows

GO: >=30% reduction in mean regret versus always-WFS on TRAIN/D1, no favored-side regret worse than WFS by >5%, correct quadrant ordering on contested events, and <80% prediction overlap with regime identity/static priority rule

NO_GO: mean regret reduction <10% versus WFS, any favored-side safety regression >5%, quadrant ordering inverted, or >=90% overlap with old regime/static-index predictions

Expected runtime: <2 minutes local CPU; pure CSV/JSON analysis

## Stop Conditions

- Stop if regret reduction versus WFS is <10%.
- Stop if any favored side regresses versus WFS by >5%.
- Stop if quadrant semantics are inverted or indistinguishable from the old analytic-index failure.
- Stop if the rule overlaps regime identity/static priority by >=90%.
- Stop if the only apparent gains depend on DEV/FINAL/TEST inspection or coefficient tuning.

## Confirmation

- No new selector training.
- No oracle acquisition.
- No DEV-driven redesign.
- No new FINAL evaluation or FINAL-row diagnostic analysis.
- No TEST.
- No new simulation.
- No Wulver jobs.
- No GPUs.
- No git mutation.
