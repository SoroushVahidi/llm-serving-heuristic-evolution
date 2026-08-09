# Apt-Serve Phase G Analysis Audit — 2026-08-09

This is the concise scientific audit for the completed Apt-Serve Phase G
collection and posthoc analysis. Machine-readable tables live in
`results/apt_serve_phase_g_analysis_20260809_190000/`; this document records
the interpretation boundary.

## Provenance

| Field | Value |
|---|---|
| Branch | `contextual-compositional-heuristics-20260731` |
| Analysis code SHA | `bcff4d8fcb0ee71f033503ee3a0f26ad28d8e576` |
| Collection directory | `results/apt_serve_phase_g_resume_20260807_174028/` |
| Failed SS15 source run | `results/apt_serve_phase_g_overnight_20260807_011542/` |
| Canonical analysis directory | `results/apt_serve_phase_g_analysis_20260809_190000/` |
| Analysis command | recorded in `wrapper_meta.txt` and `analysis_manifest.json` |
| Bootstrap replicates | `2,000,000` |
| Analysis exit code | `0` |
| Completion timestamp | `2026-08-09T23:06:53Z` |

The earlier `results/apt_serve_phase_g_analysis_20260809_185700/` run used the
same dataset and analysis seed with `500,000` bootstrap replicates. It was a
successful calibration run but is superseded by the canonical `190000` run and
contains no unique scientific evidence.

## Collection Design

Stage 1 screening:

- 41 regimes;
- 39 seeds, range 1001-1039;
- 1599 experiment units;
- full policy portfolio for each unit;
- Apt-Serve evaluated at transition costs `0x_idealized`, `0.5x`, `1x`, `2x`,
  and `4x`.

Stage 2 confirmation:

- 16 selected regimes;
- 36 seeds, range 2001-2036;
- 576 experiment units;
- analytically separated from Stage 1 in `stage1_stage2_replication.csv`.

Total:

- 2175 experiment units;
- 36975 policy-cell rows;
- 13 policy labels, with Apt-Serve represented by 5 transition-cost variants.

## SS15 Incident And Fix Boundary

The original overnight run self-terminated at SS15 on a genuine invariant
violation. The incident report is
`docs/audits/apt_serve_phase_g_ss15_incident_20260807.md`.

The resumed run preserved the failed source run as provenance and completed
after the SS15 fix. `dataset_validation.json` reports:

- `source_critical_failures`: 1 in the source run;
- `source_keys_present_in_run`: 152;
- `source_keys_not_present_in_run`: 0;
- `critical_failures`: 0 in the completed analysis dataset;
- duplicate unit keys: 0;
- malformed units: 0;
- NaN/Inf/impossible values: 0.

## Output Integrity

The canonical analysis reports:

- `dataset_validation`: `STRUCTURALLY_VALID`;
- total units: 2175;
- failures: 0;
- duplicate unit keys: 0;
- malformed units: 0;
- impossible values: 0;
- NaN/Inf values: 0.

Primary output files:

- `final_summary.json`
- `dataset_validation.json`
- `global_policy_summary.csv`
- `regime_summary.csv`
- `grouped_bootstrap_results.csv`
- `marginal_contribution_summary.csv`
- `stage1_stage2_replication.csv`
- `transition_cost_analysis.csv`
- `mechanism_analysis.csv`
- `logs/analysis.log`

## Primary Metric And Method

The primary metric is `arrival_normalized_weighted_goodput` (ANWG). The analysis
uses grouped bootstrap estimates over regime/context groups for global,
stage-level, KV-pressure, and cache-use subsets.

The primary Apt-Serve comparison uses the `1x` transition-cost variant. Fixed
baselines are compared without transition-cost variants.

## Supported Findings

1. The Phase G dataset is structurally valid.

2. Apt-Serve has positive leave-one-out marginal portfolio contribution:

   - mean marginal contribution: `0.025219`;
   - median marginal contribution: `0.0`;
   - maximum marginal contribution: `0.814815`;
   - fraction positive / unique winner: `0.204138`;
   - grouped bootstrap CI: `[0.004099, 0.057757]`, excluding zero.

3. Apt-Serve's global average ANWG is above the best fixed baseline point
   estimate, but not with a superiority CI:

   - Apt-Serve `1x` mean ANWG: `0.224845`;
   - best fixed baseline: `scorpio_style_slo_guard`, mean ANWG `0.207310`;
   - global Apt-vs-best-fixed mean gap: `0.012032`;
   - grouped bootstrap CI: `[-0.013237, 0.046700]`, crossing zero.

4. Stage 1 and Stage 2 are mostly directionally consistent for selected
   regimes, with one important exception: `length_prompt_heavy` changes sign
   from positive in screening to negative in confirmation.

5. Strong positive and negative regimes coexist:

   - strongest positive: `pressure_low_baseline`, gap about `+0.586` in
     screening and `+0.624` in confirmation;
   - strongest negative: `cacheuse_thrash_risk_high`, confirmation gap about
     `-0.165` against `weighted_shortest_processing`.

6. Transition-cost sensitivity is small in aggregate between `0.5x` and `4x`.
   The idealized `0x` variant is slightly higher (`0.225930` vs `0.224845`
   mean ANWG), but the nonzero variants are numerically identical in the
   aggregate summary.

## Promising / Interpretive Findings

- Apt-Serve appears to add portfolio value in pockets even when it is not a
  globally dominant fixed choice.
- Low-pressure and relaxed-homogeneous contexts produce the largest positive
  Apt-Serve gaps in this dataset.
- Thrash-risk contexts are counterexamples where Apt-Serve loses materially to
  `weighted_shortest_processing`.
- Mechanism diagnostics show positive correlation between Apt-Serve gap and
  total transitions overall, but correlation is not causal evidence.

## Not Yet Established

- Global Apt-Serve superiority over the best fixed deployable baseline.
- That Apt-Serve should become the final scheduler.
- That Phase G proves contextual compositional synthesis works.
- That the observed transition/gap correlations are causal mechanisms.
- That Apt-Serve's monolithic behavior has been decomposed into reusable typed
  modules suitable for the DSL.

## Artifact Concerns

- `pressure_low_baseline` dominates the global positive point estimate and has
  only two regime groups across screening/confirmation; interpret its CI with
  that grouping limit in mind.
- Some cache-use groups share very similar or identical generated behavior in
  summaries, so mechanism-level claims need follow-up module tests.
- No result directory in `results/` is version-controlled; preserve local
  artifacts until archival decisions are made.

## Exact Next Scientific Question

Which Apt-Serve cache/tier-transition behaviors, if any, should be decomposed
into typed modules and evaluated for context-specific marginal library-envelope
gain against the broader policy library?

That question belongs to the broader module-decomposition and
library-envelope-expansion roadmap, not to another default Apt-Serve sweep.
