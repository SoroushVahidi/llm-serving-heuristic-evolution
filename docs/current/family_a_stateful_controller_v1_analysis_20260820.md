# Family-A Stateful Controller V1 Analysis

Date: 2026-08-20

## Executive Verdict

Classification: `STATEFUL_CONTROLLER_NO_GO`.

Next step: `STOP_FAMILY_A_CONSTRUCTION`.

The V1 controller was evaluated strictly on repaired Family-A TRAIN/VAL inputs. TEST was not loaded.

The offline diagnostic signal was real enough to pass the preregistered go gate, but the executable controller did not convert it into held-out scheduling value. Mean controller ANWG was below fixed WFS and below the native ESTF/WFS scenario envelope; it had zero wins over the per-scenario best fixed parent.

## Controller

- Representation: `STATEFUL_TREE` shallow decision tree.
- Initial mode: `WFS_MODE`.
- Minimum dwell: `20` scheduler steps.
- Hysteresis: WFS to ESTF at `P(ESTF_MODE) >= 0.65`; ESTF to WFS at `P(ESTF_MODE) <= 0.35`.
- Candidate gate: switching evidence is evaluated only when ESTF and WFS produce different canonical actions on the current observable state.

## Target And Features

Supervision target: `ESTF_MODE` if repaired `Delta_native > 0`, otherwise `WFS_MODE`. Zero-native events are conservative WFS/default labels.

Online causal feature groups:

- request/service distribution: queue age, predicted output tokens, prompt tokens, estimated service-time quantiles/means
- fairness/starvation: max class deficit ratio, longest waiting age, distinct queued classes
- workload/queue pressure: step, queue length, active count, completed count, GPU count
- urgency/slack: laxity quantiles/means and near-deadline fractions
- resource/KV: mean/max KV utilization, free KV capacity, prefill/decode counts

Excluded runtime inputs: `favlong/favshort`, synthetic family labels, hidden generator identity, random seed, split labels, scenario ID, TEST indicators, future outcomes, and counterfactual branch outcomes. Short history and pair-rank geometry were not used as primary V1 model inputs; pair disagreement was used only as the causal candidate gate.

## Offline Feasibility

- Events: 91
- Scenarios with events: 32
- Class counts: {'wfs_or_zero': 33, 'estf': 58}
- Majority baseline balanced accuracy: 0.500000
- Tree balanced accuracy: 0.680251
- Majority baseline AUC: 0.4647335423197492
- Tree AUC: 0.7053291536050157
- Majority baseline macro F1: 0.389262
- Tree macro F1: 0.675000
- Confusion matrix tree [WFS/zero, ESTF]: `[[21, 12], [16, 42]]`

## Offline Dwell Replay

- Abstention rate: 0.659341
- ESTF event share: 0.582418
- WFS event share: 0.417582
- Switch count: 31
- Switch directions: {'ESTF_MODE->WFS_MODE': 5, 'WFS_MODE->ESTF_MODE': 26}
- Dwell violations: 0

## Offline Gate

- GO: `True`
- Reasons: none

## Full Simulation

- Scenario count: 64
- Split composition: 54 train, 10 validation
- Failures: 0
- Wall clock seconds: 348.983
- Mean ANWG by policy: `{'estimated_service_time_first': 0.7296244062499999, 'family_a_stateful_controller_v1': 0.7281250312499999, 'family_a_stateless_tree_controller_v1': 0.727541078125, 'weighted_fair_share': 0.7477746249999999}`
- Best fixed parent mean ANWG: `0.7477746249999999`
- Native-pair ESTF/WFS envelope mean ANWG: `0.76254734375`
- Paired differences vs stateful: `{'estimated_service_time_first': {'mean_diff': -0.001499374999999998, 'median_diff': 0.0, 'wins': 4, 'ties': 55, 'losses': 5}, 'weighted_fair_share': {'mean_diff': -0.01964959374999999, 'median_diff': 0.0, 'wins': 21, 'ties': 16, 'losses': 27}, 'best_fixed_parent_by_scenario': {'mean_diff': -0.034422312499999996, 'median_diff': 0.0, 'wins': 0, 'ties': 34, 'losses': 30}, 'stateless_tree': {'mean_diff': 0.0005839531249999998, 'median_diff': 0.0, 'wins': 3, 'ties': 61, 'losses': 0}}`
- Switch-count summary: `{'mean': 1.1875, 'median': 1.0, 'p25': 1.0, 'p75': 1.0, 'min': 0.0, 'max': 3.0}`
- ESTF occupancy summary: `{'mean': 0.671463608918055, 'median': 0.8256609587158394, 'p25': 0.5745765290906905, 'p75': 0.9122399903646722, 'min': 0.0, 'max': 0.982450486347123}`
- WFS occupancy summary: `{'mean': 0.3285363910819451, 'median': 0.17433904128416058, 'p25': 0.08776000963532778, 'p75': 0.42542347090930954, 'min': 0.017549513652877064, 'max': 1.0}`
- Safety metric means: `{'completion_fraction': {'estimated_service_time_first': 1.0, 'family_a_stateful_controller_v1': 1.0, 'family_a_stateless_tree_controller_v1': 1.0, 'weighted_fair_share': 1.0}, 'weighted_completion_fraction': {'estimated_service_time_first': 1.0, 'family_a_stateful_controller_v1': 1.0, 'family_a_stateless_tree_controller_v1': 1.0, 'weighted_fair_share': 1.0}, 'p95_latency': {'estimated_service_time_first': 14.7144675625, 'family_a_stateful_controller_v1': 14.71412503125, 'family_a_stateless_tree_controller_v1': 14.71412503125, 'weighted_fair_share': 16.031498125000002}, 'p95_queuing_delay': {'estimated_service_time_first': 14.179855921875001, 'family_a_stateful_controller_v1': 14.179855921875001, 'family_a_stateless_tree_controller_v1': 14.179855921875001, 'weighted_fair_share': 15.566722515625}, 'slo_violation_rate': {'estimated_service_time_first': 0.237239640625, 'family_a_stateful_controller_v1': 0.24088546875, 'family_a_stateless_tree_controller_v1': 0.24127612499999998, 'weighted_fair_share': 0.25403651562499996}}`
- Six-policy portfolio: `not_computed_in_v1_first_internal_run`

## Interpretation

The controller exercised both modes and avoided dwell violations, so this was not a fixed-parent collapse or a thrashing failure. The failure is scientific performance: WFS remained the stronger fixed parent on mean ANWG, and the controller did not add any scenario-level value over the native ESTF/WFS envelope. Safety proxies did not show a completion collapse, but the ANWG loss versus WFS means the controller cannot be considered a useful constructive scheduler.

The sign-flip regimes therefore did not generalize into an executable persistent controller in V1. The offline native-sign classifier was above baseline, but its event-region signal was insufficient when embedded in a causal online scheduler operating over full trajectories.

## Novelty Guard

This V1 is deliberately close to known state-dependent scheduling ideas. If it succeeds, novelty still remains at risk against FSP-style fairness/SRPT hybrids, T-SRPT-style state-dependent switching, VTC-style fairness baselines, vLLM-LTR service-time ranking, and PARS-like learned service-aware ranking. Those external baselines are not integrated in this first internal feasibility run.

Given the no-go result, the controller does not yet justify a novelty claim. Mechanistically it resembles a service-time/fairness-debt regime switch, which is exactly the risk area covered by FSP/T-SRPT/VTC/vLLM-LTR-style prior baselines.

## Limitations

- Only 91 repaired diagnostic events supervise the offline scorer.
- Only 32/64 Family-A scenarios have repaired events.
- TRAIN/VAL only.
- Grouped CV uncertainty remains high.
- Event-only labels are guarded by a candidate region but still differ from ordinary online state distribution.
- No TEST, public-trace, or real-serving validation has been run.
- Six-policy portfolio marginal contribution was not computed unless reported above.

## Tests

Focused deterministic tests passed with:

`python -m pytest -q tests/test_family_a_stateful_controller_v1.py`

Result: 11 passed.

## Artifacts And Reproducibility

- Design: `docs/design/FAMILY_A_STATEFUL_CONTROLLER_V1.md`
- Analysis: `docs/current/family_a_stateful_controller_v1_analysis_20260820.md`
- Experiment dir: `experiments/family_a_stateful_controller_v1/`
- Command: `scripts/run_family_a_stateful_controller_v1.py`
