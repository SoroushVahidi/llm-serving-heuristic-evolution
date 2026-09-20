# SBS Override V2 One-Shot Confirmatory Result

This directory records the single authorized Stage-B fresh confirmatory evaluation
for `SBS_OVERRIDE_CONSERVATIVE_SELECTOR_DEV_V2`.

## Label Access Boundary

Fresh terminal outcomes became scientifically accessed when the frozen evaluator
was executed on 2026-09-19 with the canonical Wulver job 1299925 label source.
No selector retraining, recalibration, threshold change, feature change, Stage-A
regeneration, or verdict-rule modification was performed.

## Command

```bash
PYTHONPATH=src python scripts/sbs_override_fresh_id_confirmatory_evaluate_v2.py \
  --expected-protocol-sha256 0e769f74d7c08c5deb19a0e8bb59ee4485045ad5a598a2c80b73b1e87b22187d \
  --expected-stage-a-freeze-sha256 1e32420f9588c7a00dcc38213bb03f831e557066c6f407018d23d89fe97055cb \
  --label-source .confirmatory_label_staging/v2_1299925/state_action_rows.all_shards.csv \
  --result-dir experiments/sbs_override_fresh_id_confirmatory_v2/stage_b/one_shot_20260919
```

## Primary Result

- Frozen disagreement states: 2,862
- Frozen overrides: 356
- Frozen abstentions: 2,506
- Total realized gain: -0.7450120737870998
- Mean realized gain over all 2,862 states: -0.0002603116959423829
- Scenario-clustered bootstrap 95% CI: [-0.0010167976331709404, 0.0005637780765335589]
- Primary verdict: `POSITIVE_GENERALIZATION_NOT_CONFIRMED`

The frozen primary criterion required both positive mean gain and a strictly
positive lower endpoint of the scenario-clustered 95% percentile bootstrap CI.
The criterion was not met.

## Override Outcomes

- Beneficial overrides: 122
- Harmful overrides: 111
- Zero-effect overrides: 123
- Beneficial fraction among overrides: 0.34269662921348315
- Harmful fraction among overrides: 0.31179775280898875
- Zero-effect fraction among overrides: 0.3455056179775281
- Override rate among fresh disagreement states: 0.1243885394828791

## Scenario Diagnostics

- Positive aggregate-gain scenarios: 18
- Negative aggregate-gain scenarios: 20
- Zero aggregate-gain scenarios: 40
- Worst override: -0.1081603647330569
- Worst scenario aggregate gain: -0.5248420075188774
- Top-1 share of positive scenario gain: 0.37242158972968986
- Top-5 share of positive scenario gain: 0.6889541287309866

The frozen evaluator did not emit a best-scenario value or a full scenario-gain
distribution table. These were not recomputed outside the evaluator, preserving
the evaluator-only fresh-label access boundary.

## Development Comparison

V2 development crossfit had 8,888 states, 1,008 overrides, total gain
2.036047175585053, mean gain 0.00022907821507482597, 5/5 positive folds, and
scenario-bootstrap CI [0.00001836079908326229, 0.0004364471081140889].

Fresh confirmation used 2,862 states and 356 frozen overrides. Its primary mean
was negative and its scenario-clustered CI crossed zero. This descriptive
comparison does not alter the frozen selector or create any alternate verdict.

## Scientific Conclusion

The prespecified fresh one-step SBS-relative confirmatory evaluation did not
establish positive generalization for the frozen V2 sparse SBS-override selector
under the frozen primary criterion.

This result does not establish closed-loop benefit, production deployment
benefit, real-vLLM benefit, broad natural-OOD generalization, or universal
scheduler superiority.
