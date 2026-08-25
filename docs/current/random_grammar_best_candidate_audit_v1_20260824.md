# Random Grammar Best Candidate Audit V1 - 2026-08-24

Status: TRAIN-only candidate freeze audit. No new search, no candidate modification, no DEV/TEST/FINAL/OOD evaluation.

## Candidate Identity

- Treatment: `A_RANDOM_GRAMMAR_GP`
- Proposal index: `40`
- Seed: `20260824`
- Structural hash: `837e6de6d8a1ad8ab5ec3d6fd928495e7e972de5557c42dbe80a11ada7276d16`
- Behavioral fingerprint: `2065e43e809f2a586b7a805ae329e1acd742a048f0dd977f10b95819d8668889`
- AST depth: 2
- Node count: 4
- Free constants: `[{'module_id': np.str_('ranking.wfs_deficit_priority_service'), 'parameter': 'alpha', 'value': 0.25}, {'module_id': np.str_('ranking.wfs_deficit_priority_service'), 'parameter': 'beta', 'value': 1.25}, {'module_id': 'prefill.chunked_small', 'parameter': 'max_prefill_chunk_tokens', 'value': 128}]`
- Readable policy: Policy(WFS_deficit_priority_service(alpha=0.25,beta=1.25), round_robin_scan placement, chunked_prefill cap=128)

Canonical genome:

```json
{"metadata":{"index":40,"operator":"random_grammar_initialization","parent_seeded":false,"seed":20260824},"name":"random_grammar_gp::20260824::40","root":{"children":[{"children":[],"description":"","module_id":"ranking.wfs_deficit_priority_service","module_type":"RankingRule","parameters":{"_free_numeric_parameters":["alpha","beta"],"alpha":0.25,"beta":1.25}},{"children":[],"description":"","module_id":"placement.round_robin_scan","module_type":"PlacementRule","parameters":{}},{"children":[],"description":"","module_id":"prefill.chunked_small","module_type":"PrefillRule","parameters":{"_free_numeric_parameters":["max_prefill_chunk_tokens"],"decode_first":false,"max_prefill_chunk_tokens":128}}],"description":"","module_id":"policy.module_composition","module_type":"Policy","parameters":{"canonical_parent_id":null,"exactness_status":"COMPOSED_CANDIDATE"}},"schema_version":"PortfolioGuidedTypedGPGenomeV1"}
```

## Provenance

The candidate was produced by `random_grammar_initialization` with `parent_seeded=false`. It uses grammar primitives that also appear in parents, specifically WFS-style ranking and chunked-prefill control, but it is not structurally descended from a parent genome and has no crossover or mutation lineage.

## Deterministic Reproduction

- Same 24 TRAIN scenarios: true
- Reward exact match: True
- Max absolute reward diff: 0.000000
- Behavioral fingerprint exact match: True
- Parent-overlap exact match: True

## Metrics

- Mean MG: 0.011295
- Unique wins at epsilon 0.005: 6
- Positive-MG scenarios: 6
- Positive-MG families: `['FAMILY_A_FAIRNESS_STARVATION_V2', 'FAMILY_C_KV_PRESSURE_V2']`
- Max parent probe decision overlap: 0.000000
- Max parent reward correlation: 0.948424
- Worst family mean regression vs E6: 0.044972
- Worst group mean regression vs E6: 0.125000
- Top-one scenario MG share: 0.413612
- Top-one family MG share: 0.891501

## Gate Table

| Gate | Threshold | Observed | Result |
|---|---|---|---|
| mean_MG >= 0.005 ANWG | `0.005` | `0.011294934640522872` | PASS |
| >=3 unique wins at epsilon 0.005 | `3` | `6` | PASS |
| wins across >=2 regions | `2` | `2` | PASS |
| max parent decision overlap <=95% | `0.95` | `0.0` | PASS |
| max parent reward correlation <=0.985 | `0.985` | `0.9484239608915721` | PASS |
| no group regression >0.030 ANWG | `0.03` | `0.044971955128205135` | FAIL |
| top-one scenario MG share <=0.40 | `0.4` | `0.41361170475094516` | FAIL |
| top-one family MG share <=0.70 | `0.7` | `0.891500904159132` | FAIL |
| mechanism behavior coherent | `true` | `true` | PASS |
| parent reproduction gates remain PASS | `"all PASS"` | `{"chunked_prefill_small": "PARENT_REPRODUCTION_PASS", "estimated_service_time_first": "PARENT_REPRODUCTION_PASS", "full_prefill": "PARENT_REPRODUCTION_PASS", "kv_constrained_online": "PARENT_REPRODUCTION_PASS", "least_laxity_first": "PARENT_REPRODUCTION_PASS", "weighted_fair_share": "PARENT_REPRODUCTION_PASS"}` | PASS |

## Epsilon-Level Wins

- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.5000.skew10.0000.favlong.noise0.00.s20260816` (FAMILY_A_FAIRNESS_STARVATION_V2): E6 0.443939, candidate 0.556061, MG 0.112121, best parent `weighted_fair_share`
- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.1000.skew1.0000.favlong.noise0.00.s20260816` (FAMILY_A_FAIRNESS_STARVATION_V2): E6 0.750000, candidate 0.833333, MG 0.083333, best parent `weighted_fair_share`
- `FAMILY_C_KV_PRESSURE_V2::kvp2.bulk24.phaselate.tighttight.s20260910` (FAMILY_C_KV_PRESSURE_V2): E6 0.823529, candidate 0.852941, MG 0.029412, best parent `estimated_service_time_first`
- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.5000.skew1.0000.favlong.noise0.30.s20260816` (FAMILY_A_FAIRNESS_STARVATION_V2): E6 0.733333, candidate 0.758333, MG 0.025000, best parent `estimated_service_time_first;kv_constrained_online`
- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.1000.skew10.0000.favlong.noise0.30.s20260816` (FAMILY_A_FAIRNESS_STARVATION_V2): E6 0.769697, candidate 0.784848, MG 0.015152, best parent `weighted_fair_share`
- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.5000.skew10.0000.favshort.noise0.00.s20260816` (FAMILY_A_FAIRNESS_STARVATION_V2): E6 0.951515, candidate 0.957576, MG 0.006061, best parent `estimated_service_time_first`

## Positive-MG Scenarios

- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.5000.skew10.0000.favlong.noise0.00.s20260816`: MG 0.112121
- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.1000.skew1.0000.favlong.noise0.00.s20260816`: MG 0.083333
- `FAMILY_C_KV_PRESSURE_V2::kvp2.bulk24.phaselate.tighttight.s20260910`: MG 0.029412
- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.5000.skew1.0000.favlong.noise0.30.s20260816`: MG 0.025000
- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.1000.skew10.0000.favlong.noise0.30.s20260816`: MG 0.015152
- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.5000.skew10.0000.favshort.noise0.00.s20260816`: MG 0.006061

## Largest Regressions vs E6

- `FAMILY_B_PREFILL_DECODE_V2::pd2.hog12.late12.slohog_ttft.s20260820` (FAMILY_B_PREFILL_DECODE_V2): candidate minus E6 -0.125000
- `FAMILY_B_PREFILL_DECODE_V2::pd2.hog24.late12.slohog_ttft.s20260820` (FAMILY_B_PREFILL_DECODE_V2): candidate minus E6 -0.083333
- `FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.1000.skew1.0000.favshort.noise0.00.s20260816` (FAMILY_A_FAIRNESS_STARVATION_V2): candidate minus E6 -0.075000
- `FAMILY_B_PREFILL_DECODE_V2::pd2.hog12.late40.slohog_ttft.s20260820` (FAMILY_B_PREFILL_DECODE_V2): candidate minus E6 -0.057692
- `FAMILY_C_KV_PRESSURE_V2::kvp2.bulk10.phaseearly.tighttight.s20260910` (FAMILY_C_KV_PRESSURE_V2): candidate minus E6 -0.050000

## Parent Behavioral Comparison

| Parent | Probe decision overlap | Candidate-trajectory overlap | Reward correlation |
|---|---:|---:|---:|
| full_prefill | 0.000000 | 0.000000 | 0.485535 |
| chunked_prefill_small | 0.000000 | 0.000000 | 0.580996 |
| estimated_service_time_first | 0.000000 | 0.000000 | 0.920782 |
| weighted_fair_share | 0.000000 | 0.000000 | 0.948424 |
| least_laxity_first | 0.000000 | 0.000000 | 0.430359 |
| kv_constrained_online | 0.000000 | 0.000000 | 0.873954 |

## Module Activation

| Module | Type | Total activations | Nonempty waiting activations | Positive-MG scenario activations | Dead |
|---|---|---:|---:|---:|---|
| policy.module_composition | Policy | 273868 | 249138 | 160411 | False |
| ranking.wfs_deficit_priority_service | RankingRule | 273868 | 249138 | 160411 | False |
| placement.round_robin_scan | PlacementRule | 273868 | 249138 | 160411 | False |
| prefill.chunked_small | PrefillRule | 273868 | 249138 | 160411 | False |

There are no IF/conditional branches in this AST. No module is dead on TRAIN.

## Explanatory Ablation

These are TRAIN-only explanatory ablations, not new candidates and not used for selection:

- `remove_prefill_rule`: mean MG 0.011295, unique wins 6, positive scenarios 6
- `replace_round_robin_with_default_placement`: mean MG 0.011295, unique wins 6, positive scenarios 6
- `remove_prefill_and_use_default_placement`: mean MG 0.011295, unique wins 6, positive scenarios 6

## Leakage / Pathology

No future information, actual output length, scenario/family identifier, generator identifier, DEV/TEST/FINAL/OOD information, NaN artifact, impossible state value, no-op pathology, or metric artifact was found. The candidate uses online observable request, queue, class, active-load, and prefill-control fields.

## Freeze Decision

Decision: `RANDOM_GP_CANDIDATE_NOT_FREEZE_READY`

Reasons: `['no group regression >0.030 ANWG', 'top-one scenario MG share <=0.40', 'top-one family MG share <=0.70']`

The TRAIN gain is real within the completed screen, but it is selection-biased because the candidate was selected as best of 60 random-grammar candidates. It fails the preregistered freeze/substantive GO standard on group regression and gain concentration. It should not be frozen for held-out validation under the current contract.
