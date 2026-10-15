# Limitations

## Historical / exploratory status

This dataset is a historical exploratory synthetic-simulator supplement. It is
not a new confirmatory study and must not be presented as the primary evidence
of the active Performance Evaluation manuscript (PEVA-D-26-00474).

## Synthetic simulator setting

All observations come from a deterministic project simulator with synthetic
arrivals and token lengths. Results do not automatically transfer to production
serving stacks, hardware, or real user traffic.

## Scenario-level dependence

The defensible independent sampling units are scenarios: 240 for the original
scan prevalence question and 236 for the conditional labeled population. States
and action forks within a scenario or disagreement episode are correlated.
Treating the 8,886 state rows or 21,854 outcome rows as independent replicates
is incorrect.

## Disagreement-conditioned label acquisition

Terminal labels were acquired only where at least one candidate policy
disagreed with SBS. Outcome proportions therefore cannot be extrapolated to
all SBS states. Equal-action states have zero one-action override advantage by
construction under this estimand and were not terminal-labeled.

## No generalization guarantee

There is no guarantee that relationships observed on these disagreement states
generalize to other scheduler states, workloads, action libraries, or
simulators.

## No confirmed deployed improvement

The release does not demonstrate a deployed scheduler improvement, a permanent
policy switch benefit, or production SLO gains.

## No production-calibrated latency claim

Metrics are simulator utilities (primary: ANWG). They are not production-calibrated
latency measurements.

## No confirmed closed-loop selector benefit

Historical offline selector analysis produced a held-out mean gain of
`0.0001959413` ANWG with scenario-bootstrap 95% interval
`[-0.000207663, 0.000585958]`, which **includes zero**. The closed-loop gate
was blocked. This dataset alone does not establish a confirmed selector benefit.

## Negative and harmful effects

The release preserves harmful and null effects as scientific content:
**2,573 positive, 3,624 negative, and 15,657 null** ANWG effects at the original
`1e-12` threshold. Analyses that report only positive rows are incomplete.

## Exclusions

Previously public reference utilities, public learnability aggregates,
public-equivalent alternatives, dense legacy records, fitted predictions, raw
shards, third-party traces, and 98 historically ambiguous overlap keys are
excluded. Users must not treat absence from this package as absence from all
prior public artifacts without checking those sources.

## Portfolio and modeling limits

The action library is the six P6 policies above. Effects are one-step
intervention + fixed SBS continuation. The package does not evaluate multi-step
overrides, alternative continuations, or unrestricted action spaces.
