# Contextual Composition Query 4 CC1 Results - 2026-07-31

Branch: `contextual-compositional-heuristics-20260731`

Starting SHA: `0f64c04893b48151437fe533aecd352bc6dad1be`

Implementation SHA before final documentation amend:
`6fa3a7e29fd95f9480efe36af9565cf21182b86d`

Canonical issue: [#1](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1)

## Implementation Summary

Implemented the approved CC1 true simulator-executed weighted Borda
rank-aggregation experiment.

Created:

- `configs/cc1_composition_opportunity_smoke.yaml`
- `configs/cc1_composition_opportunity.yaml`
- `scripts/run_cc1_composition_opportunity.py`
- `src/llmserveopt/experiments/__init__.py`
- `src/llmserveopt/experiments/cc1_composition_opportunity.py`
- `tests/test_cc1_composition_opportunity.py`

The implementation reuses:

- `StaticRankEnsemblePolicy(method="borda")`
- `RankExpertSpec`
- `rank_with_named_expert` through the ensemble policy
- `InstrumentedPolicy` and `DecisionTraceSink` when traces are enabled
- existing `run_policy`, simulator, workload, metric, and trace-loading paths

No CC2 primitive refactoring or CC3 DSL work was implemented.

## Exact Commands

Starting validation:

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
git rev-list --left-right --count HEAD...@{u}
python scripts/check_contextual_composition_status.py
```

Dry-run and smoke:

```bash
python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1_composition_opportunity_smoke.yaml \
  --dry-run

python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1_composition_opportunity_smoke.yaml \
  --timestamp query4_smoke_20260731 \
  --allow-dirty
```

Focused validation:

```bash
python -m pytest tests/test_cc1_composition_opportunity.py -q
python -m pytest \
  tests/test_cc1_composition_opportunity.py \
  tests/test_policy_composition.py \
  tests/test_score_and_reciprocal_rank_composition.py \
  tests/test_contextual_composition_status_checker.py \
  -q
python scripts/check_contextual_composition_status.py
python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1_composition_opportunity.yaml \
  --dry-run
```

Full local run:

```bash
python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1_composition_opportunity.yaml \
  --full-run \
  --max-runs 200 \
  --timestamp query4_full_20260731
```

The full run was executed from a clean local implementation commit; generated
result files remain under ignored `results/`.

## Workloads And Search Space

Policy subset:

- `weighted_shortest_processing`
- `scorpio_style_slo_guard`
- `edf`
- `estimated_service_time_first`
- `fifo`

Weight search:

- deterministic simplex grid;
- step `0.5`;
- nonnegative weights normalized to sum to one;
- `top_k = 2`;
- 15 mixtures, including one-hot mixtures and all two-policy 50/50 mixtures.

Full-run windows:

- 10 total workload windows;
- development: `TRAIN`, `VALIDATION`, `ROBUST_DEV`;
- evaluation: `ID_TEST`, `OOD_TEST`;
- regimes covered: underloaded, saturated, mixed SLO, prefill heavy, decode
  heavy, KV pressure, BurstGPT real trace, Azure conversation real trace;
- real-trace-derived windows used available local files:
  `data/processed/burstgpt/burstgpt_moderate_exact_prediction.jsonl` and
  `data/processed/azure/azure_llm_2023_conv.jsonl`;
- missing real traces: none.

Full run count:

- fixed-policy simulator executions: 50;
- mixture simulator executions: 150;
- total simulator executions: 200.

## Tests

Passed:

- `python -m pytest tests/test_cc1_composition_opportunity.py -q`: 11 passed.
- `python -m pytest tests/test_cc1_composition_opportunity.py tests/test_policy_composition.py tests/test_score_and_reciprocal_rank_composition.py tests/test_contextual_composition_status_checker.py -q`: 68 passed.

Covered:

- weight validation and normalization;
- deterministic sparse simplex grid;
- true simulator execution;
- no reward-vector interpolation for mixture outcomes;
- oracle fixed calculation;
- oracle mixture calculation;
- near-tie filtering;
- completion-loss gate;
- verdict logic;
- smoke CLI dry-run;
- missing-trace skip behavior;
- reproducibility.

## Results

Full output directory:

```text
results/cc1_composition_opportunity/query4_full_20260731/
```

Required files created:

- `manifest.json`
- `config.yaml`
- `policy_execution_rows.csv`
- `per_window_summary.csv`
- `method_comparison.csv`
- `composition_weights.csv`
- `near_tie_sensitivity.csv`
- `subset_analysis.csv`
- `verdict.json`
- `cc1_report.md`

Key full-run metrics:

- best fixed policy: `fixed__weighted_shortest_processing`
- best global mixture: `mix__weighted_shortest_processing-1p000`
- oracle fixed mean ANWG on evaluation: `1.0`
- oracle mixture mean ANWG on evaluation: `1.0`
- composition-opportunity gap: `0.0`
- non-near-tie evaluation windows: `0`
- non-near-tie gap: `null`
- completion impact vs best fixed: `0.0`
- best regime gain: `0.0`
- near-tie fraction at thresholds `0.001`, `0.005`, `0.01`: `1.0`

The learned hard-selector baseline was compatible as a development-trained
regime lookup over fixed-policy simulator rows. It selected one complete policy
per window and achieved evaluation mean ANWG `1.0`, matching best fixed.

## Completion And Near-Tie Analysis

Completion constraint passed: oracle mixture mean completion fraction matched
best fixed with impact `0.0`.

However, every evaluation window was a near tie under the fixed-policy
top-two-margin criterion. This means the local representative suite did not
produce discriminative fixed-policy margins on held-out/OOD evaluation windows.

## Verdict

`STOP_OR_REDESIGN`

Reason:

```text
oracle mixture did not beat oracle fixed on average
```

The CC1 gate did not pass:

- aggregate non-near-tie ANWG gain was unavailable because there were zero
  non-near-tie evaluation windows;
- regime-specific gain was `0.0`;
- oracle mixture did not exceed oracle fixed;
- best global mixture collapsed to a one-hot WSP mixture.

## Limitations

- The full local suite was deliberately capped at 200 simulator executions for
  reproducibility and runtime control.
- Evaluation windows were all near ties, so this result is a negative gate for
  the approved local CC1 suite, not a proof that all future composition
  variants can never help.
- The run used the current simulator objective and local representative
  workloads; it did not launch live APIs, GPU jobs, real-vLLM, or cluster jobs.
- Generated result files are local ignored artifacts and are not committed.

## Query 5 Recommendation

Query 5 should not begin CC2 primitive-interface refactoring or CC3 DSL work.

Recommended Query 5 action:

- redesign or strengthen CC1 workload discriminativeness before reopening the
  composition gate; or
- document a pause/stop decision for this contextual-composition path if the
  project accepts the current negative gate.

Any redesigned CC1 follow-up should first create non-near-tie simulator
evaluation windows under ANWG, then rerun true simulator-executed mixtures.
