# Industry Realism Causal Headroom Phase D V1

Status: preregistered design only. No terminal counterfactual outcomes, causal labels, or selector training are executed in this phase-freeze task.

## Scientific Question

When production-derived workloads enter valid constrained regimes where SBS and the P6 portfolio take different canonical actions, how often is SBS locally causally suboptimal, how much one-step headroom exists, and how does that headroom vary with workload and resource pressure?

## Estimand

`Q_SBS(s,a)` forces exactly one canonical action `a` at live state `s`, then immediately returns to fixed `kv_constrained_online` continuation with the same future arrivals and simulator semantics. `A_SBS(s,a)=Q_SBS(s,a)-Q_SBS(s,SBS)`.

Forbidden continuations: policy-forever switching, learned continuation, alternative-policy continuation, and closed-loop adaptive selector continuation.

## Populations

Prevalence population: all SBS decision states from Phase A/B, used for `P(D)`. Causal population: canonical disagreement states from selected valid Phase-D regimes, used for `P(B|D)` and headroom metrics.

## Coverage-Based Regime Selection

| Workload | Axis | Condition | Stage labels | Phase-B disagreements | P(D) |
|---|---|---|---|---:|---:|
| azure_2023_code | active_sequence_capacity | active_8 | onset | 1 | 1.00008e-05 |
| azure_2023_code | active_sequence_capacity | active_4 | sustained+high_valid_pressure | 100 | 0.000998732 |
| azure_2023_code | kv_capacity | kv_16000 | onset+sustained+high_valid_pressure | 596 | 0.00595315 |
| azure_2023_conv | active_sequence_capacity | active_4 | onset+sustained+high_valid_pressure | 31 | 6.69907e-05 |
| azure_2023_conv | kv_capacity | kv_16000 | onset | 18 | 3.89623e-05 |
| azure_2023_conv | kv_capacity | kv_8000 | sustained+high_valid_pressure | 10120 | 0.0218267 |
| burstgpt | active_sequence_capacity | active_16 | onset | 14 | 3.17683e-05 |
| burstgpt | active_sequence_capacity | active_8 | sustained | 76 | 0.000170991 |
| burstgpt | active_sequence_capacity | active_4 | high_valid_pressure | 244 | 0.000535496 |
| burstgpt | kv_capacity | kv_16000 | onset+sustained | 12 | 2.7244e-05 |
| burstgpt | kv_capacity | kv_8000 | high_valid_pressure | 116 | 0.000263343 |

Arrival-pressure cells have zero Phase-B support and remain structural null controls, not causal-label cells.

## Labeling Scope

Chosen path: `EXHAUSTIVE_SELECTED_REGIME_LABELING`.
Selected disagreement states: `11328`.
Unique non-SBS branches: `12169`.
SBS reference branches: `11328`.
Total expected terminal continuations: `23497`.

## Universe Reconciliation

All valid Phase-B support disagreement states reconstructed: `11328`.
Reconciles to Phase-B reported valid disagreement states: `True`.

## Statistics

Clustered resampling unit is faithful window, with `2000` percentile-bootstrap replicates, seed `20260920`, and minimum independent clusters `5` for inferential CI. Sparse regimes below that threshold are descriptive only.

## Practitioner Output

Report `ACTIONABLE_HEADROOM_INDEX = (P(D), P(B|D), mean_oracle_headroom)` per workload/regime, optionally with binding rate and downside prevalence. Do not present this tuple as deployable selector performance.

## Compute Budget

Recommended Wulver array tasks: `94`. Estimated storage: `93.99 MB`. GPU required: `False`.
