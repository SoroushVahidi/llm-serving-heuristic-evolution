# JOINT240 SBS Advantage Learnability V1 Post-Run Analysis

Generated from frozen `run_v1` outputs only. No fitting, prediction regeneration, scheduler launch, real-vLLM run, or manuscript edit was performed.

## Verdict
- Duplication gate: `NOT_PREVIOUSLY_DONE`
- Saved final verdict: `HELD_OUT_SIGNAL_POSITIVE_BUT_UNCERTAIN`
- Validated final verdict: `HELD_OUT_SIGNAL_POSITIVE_BUT_UNCERTAIN`
- Closed-loop gate: `JOINT240_SBS_OVERRIDE_CLOSED_LOOP_V1 = BLOCKED`
- Bottleneck: scenario-bootstrap 95% CI lower bound for mean held-out realized gain is not greater than zero

## Primary Selector
- States: 8888
- Overrides: 3397 (0.382201)
- Beneficial / harmful / zero-effect overrides: 568 / 450 / 2379
- Precision among overrides: 0.167206
- Mean realized gain/state: 0.000195941320391
- Total realized gain: 1.74152645563
- Median / p5 / p10 realized gain: 0 / -2.77555756156e-17 / 0
- Worst harmful override: -0.103544241347

## Bootstrap Gate
- Mean realized gain/state 95% CI: [-0.000207662850129, 0.000585957743655], point 0.000198399830997
- Lower bound > 0: NO

## Scenario Generalization
- Positive / zero / negative scenarios: 80 / 99 / 57
- Mean / median / p10 / p5 scenario gain: 0.00737934938828 / 0 / -0.0721747742622 / -0.145400985263
- Worst scenario: joint_mm_0064 (-0.773824077795)
- Best scenario: joint_mm_0065 (0.445273226465)

## Oracle And Harm
- Oracle mean / total gain: 0.00466469375877 / 41.4597981279
- Learned mean / total gain: 0.000195941320391 / 1.74152645563
- Gap closure: 0.0420051841608
- Mean regret / p90 / p95: 0.00446875243837 / 0.0170095551515 / 0.0272333985716
- Harmful overrides: 450 across 87 scenarios and folds [0, 1, 2, 3, 4]
- Mean / median harmful loss: -0.0194774296352 / -0.0144804126038

## Open-Loop d_SBS Descriptive Estimate
- Disagreement rate: 0.0196196160842
- Descriptive mean gain per SBS decision state: 3.8442934811e-06

## Artifact Tables
- `fold_level_results.csv`
- `scenario_level_results.csv`
- `state_core_vs_state_action_ablation.csv`
- `model_family_results.csv`
- `fixed_threshold_harm_table.csv`
- `post_run_analysis_summary.json`
