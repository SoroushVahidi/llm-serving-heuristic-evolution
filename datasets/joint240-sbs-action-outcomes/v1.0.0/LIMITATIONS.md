# Limitations

* This is historical exploratory research in a deterministic synthetic
  simulator, not production telemetry or a deployed scheduler evaluation.
* The original study has 240 scenario-level independent units for scan
  prevalence; the released conditional population covers 236 scenarios.
  States and action forks within scenarios and episodes are correlated.
* Labels were acquired only where at least one policy disagreed with SBS.
  Outcome proportions therefore cannot be extrapolated to all SBS states.
* The 453,016 scan states are records, not independent experiments.
* A forced action is followed by fixed SBS continuation for one-step causal
  headroom. This does not evaluate a permanent policy switch or closed-loop
  controller.
* The offline selector's held-out mean gain was `0.0001959413` ANWG with
  scenario-bootstrap 95% interval `[-0.000207663, 0.000585958]`, including
  zero.
* The release preserves harmful and null outcomes as well as positive ones:
  2,573 positive, 3,624 negative, and 15,657 null ANWG effects at the
  original `1e-12` threshold.
* The 98 ambiguous historical-overlap keys are excluded because full hidden
  simulator identity cannot be resolved.
* Public reference utilities, public aggregates, dense legacy records, raw
  third-party traces, and private provenance are excluded.
* No production-calibrated latency claim, general scheduler superiority,
  generalization guarantee, or confirmed selector benefit follows from these
  tables.
