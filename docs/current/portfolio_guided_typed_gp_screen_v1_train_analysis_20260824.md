# Portfolio-Guided Typed GP Screen V1 TRAIN Analysis - 2026-08-24

Status: TRAIN-only scientific screen completed. No DEV, TEST, or FINAL data were used.

## Run Integrity

Command: `python scripts/run_portfolio_guided_typed_gp_screen_v1.py --mode screen --confirm-full-screen --candidates-per-treatment 60 --scenario-limit 24 --seed 20260824 --out-dir experiments/portfolio_guided_typed_gp_screen_v1/run_train_screen_v1`

- Mode: `screen`
- Seed: `20260824`
- TRAIN scenarios: 24 ({'FAMILY_A_FAIRNESS_STARVATION_V2': 8, 'FAMILY_B_PREFILL_DECODE_V2': 8, 'FAMILY_C_KV_PRESSURE_V2': 8})
- Candidate-scenario evaluations: 4320
- Equal evaluated-candidate budget: True
- Wall time: 1090.363236 s
- Note: runner artifact schema names still say smoke, but mode/budget/provenance identify this as the full TRAIN screen.

- `A_RANDOM_GRAMMAR_GP`: proposed 64, rejected 4, duplicates 0, valid 60, unique 60, evaluated 60.
- `B_PARENT_SEEDED_MUTATION_ONLY`: proposed 60, rejected 0, duplicates 0, valid 60, unique 60, evaluated 60.
- `C_PORTFOLIO_STRUCTURAL_CROSSOVER`: proposed 61, rejected 1, duplicates 0, valid 60, unique 60, evaluated 60.

## Treatment Results

- `A_RANDOM_GRAMMAR_GP`: best mean MG 0.011295, median candidate mean MG 0.000505, mean candidate mean MG 0.001980, best unique wins 6, positive-MG candidates 39/60, mean-MG>=0.005 candidates 7/60, >=2-region candidates 6/60, best max parent overlap 0.000000.
- `B_PARENT_SEEDED_MUTATION_ONLY`: best mean MG 0.002551, median candidate mean MG 0.000000, mean candidate mean MG 0.000492, best unique wins 4, positive-MG candidates 21/60, mean-MG>=0.005 candidates 0/60, >=2-region candidates 3/60, best max parent overlap 1.000000.
- `C_PORTFOLIO_STRUCTURAL_CROSSOVER`: best mean MG 0.000000, median candidate mean MG 0.000000, mean candidate mean MG 0.000000, best unique wins 0, positive-MG candidates 0/60, mean-MG>=0.005 candidates 0/60, >=2-region candidates 0/60, best max parent overlap 0.500000.

## Equal-Budget Comparison

- C best minus A best mean MG: -0.011295
- C best minus B best mean MG: -0.002551
- B best minus A best mean MG: -0.008744
- C produced overall best candidate: False
- A candidates equivalent/better than best C by mean MG: 60/60
- B candidates equivalent/better than best C by mean MG: 60/60

This screen is small and candidates share generators/scenarios, so no formal independence-based significance test is claimed.

## Best Crossover Child

- Treatment: `C_PORTFOLIO_STRUCTURAL_CROSSOVER`
- Proposal index: 2
- Structural hash: `dc4dbe92d481cdc6e1d7713760609e62857938744959e4a2267aef251da59d33`
- Behavioral fingerprint: `a1e9fde08bd1b86c2c8c79615e422189c132721df0457c1c02cf3c5a9a7b4dcd`
- Complexity: depth 2, nodes 3, free constants 2
- Parent behavior overlaps: `{'chunked_prefill_small': 0.0, 'estimated_service_time_first': 0.5, 'full_prefill': 0.0, 'kv_constrained_online': 0.0, 'least_laxity_first': 0.0, 'weighted_fair_share': 0.5}`
- Parent reward correlations: `{'full_prefill': 0.5784176883653072, 'chunked_prefill_small': 0.6003200604489017, 'estimated_service_time_first': 1.0, 'weighted_fair_share': 0.9774182419748961, 'least_laxity_first': 0.46533595428075586, 'kv_constrained_online': 0.954131354870442}`
- Human-readable policy: `policy.module_composition(ranking.estf_service_time, placement.default_gpu_pressure)`

Canonical genome:

```json
{"metadata":{"module_type":"RankingRule","operator":"typed_subtree_crossover","parent_a":"4ffe946951ddc7bec6b7988bd56d63cf0117f05221ff7705035975c104dede05","parent_b":"c0ba47a7cb96570adb66ac592c80865db15a556118960cb89a81b97ed3558114","parent_ids":["weighted_fair_share","estimated_service_time_first"],"path_a":[0],"path_b":[0],"seed":20260886,"treatment_id":"C_PORTFOLIO_STRUCTURAL_CROSSOVER","uses_crossover":true},"name":"structural_crossover::weighted_fair_share::estimated_service_time_first::2","root":{"children":[{"children":[],"description":"","module_id":"ranking.estf_service_time","module_type":"RankingRule","parameters":{"_free_numeric_parameters":["alpha","beta"],"alpha":0.5,"beta":1.0}},{"children":[],"description":"","module_id":"placement.default_gpu_pressure","module_type":"PlacementRule","parameters":{}}],"description":"Typed exact parent representation for weighted_fair_share","module_id":"policy.module_composition","module_type":"Policy","parameters":{"canonical_parent_id":null,"exactness_status":"COMPOSED_CANDIDATE"}},"schema_version":"PortfolioGuidedTypedGPGenomeV1"}
```

Unique wins at epsilon 0.005:

- None at epsilon 0.005.

Largest regressions to E6:

- FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.5000.skew10.0000.favlong.noise0.00.s20260816: family `FAMILY_A_FAIRNESS_STARVATION_V2`, regression 0.125758, candidate 0.318182, E6 0.443939
- FAMILY_B_PREFILL_DECODE_V2::pd2.hog24.late40.slolate_ttft.s20260820: family `FAMILY_B_PREFILL_DECODE_V2`, regression 0.078125, candidate 0.421875, E6 0.500000
- FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.1000.skew10.0000.favlong.noise0.30.s20260816: family `FAMILY_A_FAIRNESS_STARVATION_V2`, regression 0.057576, candidate 0.712121, E6 0.769697
- FAMILY_B_PREFILL_DECODE_V2::pd2.hog24.late12.slolate_ttft.s20260820: family `FAMILY_B_PREFILL_DECODE_V2`, regression 0.055556, candidate 0.833333, E6 0.888889
- FAMILY_A_FAIRNESS_STARVATION_V2::fs2.util1.1000.skew1.0000.favshort.noise0.00.s20260816: family `FAMILY_A_FAIRNESS_STARVATION_V2`, regression 0.050000, candidate 0.925000, E6 0.975000

## Gate Evaluation

| Metric | Threshold | Observed | Result |
|---|---|---|---|
| mean MG >= 0.005 ANWG | `0.005` | `0.0` | FAIL |
| >=3 unique wins at epsilon 0.005 | `3` | `0` | FAIL |
| wins across >=2 regions | `2` | `0` | FAIL |
| max parent decision overlap <=95% | `0.95` | `0.5` | PASS |
| max parent reward correlation <=0.985 | `0.985` | `1.0` | FAIL |
| no group regression >0.030 ANWG | `0.03` | `0.032291666666666656` | FAIL |
| gains not concentrated in one scenario/family | `{"top_scenario": 0.4, "top_family": 0.7}` | `{"top_scenario_share": 1.0, "top_family_share": 1.0}` | FAIL |
| mechanism behavior scientifically coherent | `true` | `false` | FAIL |
| all parent reproduction gates remain PASS | `"all PASS"` | `{"chunked_prefill_small": "PARENT_REPRODUCTION_PASS", "estimated_service_time_first": "PARENT_REPRODUCTION_PASS", "full_prefill": "PARENT_REPRODUCTION_PASS", "kv_constrained_online": "PARENT_REPRODUCTION_PASS", "least_laxity_first": "PARENT_REPRODUCTION_PASS", "weighted_fair_share": "PARENT_REPRODUCTION_PASS"}` | PASS |

- GO pass: False
- NO-GO reasons: `['mean_MG_below_0.001', 'no_unique_wins_eps_0.005', 'isolated_or_absent_gains']`
- Overall verdict: `SYNTHESIS_NO_GO`
- LLM 2026 story supported: diagnostic-only

## Interpretation

Treatment C did not discover an envelope-expanding candidate under the frozen equal TRAIN-only budget. The best crossover child was structurally valid, but it produced no positive MG and no epsilon-level unique envelope wins. Random grammar search did find positive TRAIN envelope movement, while parent-seeded mutation did not beat C by the primary mean-MG metric only because both were at zero. This supports the diagnostic-only path rather than a constructive crossover result.

Next scientific decision: do not launch another search; choose preregistered two-timescale fallback or diagnostic-paper path.

## Safety

No TEST/FINAL data, DEV-driven tuning, Wulver/SLURM, GPU, API call, git push, or destructive git operation was used for this analysis.
