# CC1 Composition Opportunity Experiment Specification

Status: `SPECIFICATION_READY`

Canonical issue: [#1](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1)

Roadmap phase: `CC1 - Composition opportunity experiment`

This specification freezes the Query 4 implementation target. It does not
mark CC1 complete and does not authorize running the full experiment in Query
3.

## Scientific Question

Does true simulator-executed weighted composition over compatible causal
request-ranking experts provide measurable opportunity beyond per-window hard
selection of one existing policy?

The experiment answers this by executing composed policies in the simulator and
comparing their realized `arrival_normalized_weighted_goodput` to fixed-policy,
learned hard-selector, oracle fixed-policy, global-mixture, and oracle
per-window mixture baselines.

## Hypotheses

H1: A sparse weighted rank aggregation over complementary policies can improve
ANWG over the best fixed policy without reducing completion fraction beyond the
allowed tolerance.

H2: The oracle per-window mixture can exceed the oracle per-window fixed policy
on non-near-tie windows. A positive gap is required before broad primitive
refactoring is scientifically justified.

H3: If the oracle per-window mixture does not beat oracle per-window fixed
policy by the stop threshold, subsequent contextual predictor and DSL work
should pause or be revised.

## Minimal Representative Policy Subset

Primary subset:

- `weighted_shortest_processing`
- `scorpio_style_slo_guard`
- `edf`
- `estimated_service_time_first`
- `fifo`

Rationale:

- `weighted_shortest_processing` represents short-service/priority behavior.
- `scorpio_style_slo_guard` contributes SLO/admission filtering behavior and is
  rank-capable but not score-capable.
- `edf` represents deadline ordering.
- `estimated_service_time_first` is a causal service-time specialist.
- `fifo` is the arrival-order floor and tie-stability reference.

Optional smoke-only subset: `weighted_shortest_processing`, `edf`, `fifo`.

No oracle policy, future output-token field, completed latency, reward column,
or label column may be used by the composed policy at simulator execution time.

## Exact Composition Semantics

Chosen operator: weighted Borda rank aggregation through
`StaticRankEnsemblePolicy(method="borda")`.

At each simulator scheduling decision:

1. Read the current `ObservableState`.
2. For every active expert, call `rank_with_named_expert(expert, state)`.
3. Convert each expert ranking over the current waiting queue to normalized
   ranks where `1.0` is most preferred and `0.0` is least preferred.
4. Normalize nonnegative finite expert weights so active weights sum to one.
5. Compute each request's aggregate score as:

```text
aggregate_score(request) =
  sum_over_experts(normalized_weight[expert] * normalized_rank[expert, request])
```

6. Sort requests by descending aggregate score, descending expert support,
   ascending arrival time, and ascending request id.
7. Project the ranked requests through the existing deterministic feasibility
   and placement path.
8. If no feasible admission remains while the waiting queue is nonempty, use
   the deterministic fallback policy and log fallback.

`top_k` sparsity is permitted only as a fixed config value selected on
development data. Query 4 should default to `top_k: 2` for the full grid and
allow `top_k: all` only in dry-run/smoke or explicit config.

## Normalization Method

Rank normalization is per expert and per decision state:

- `n == 0`: empty output.
- `n == 1`: the single ranked request receives `1.0`.
- `n > 1`: rank position `0..n-1` maps to `1 - rank / (n - 1)`.

This is true composition because the simulator executes the composed scheduler's
actions. It is not valid to estimate a mixture's performance by linearly
combining stored per-policy reward vectors. Reward vectors may be used only to
report fixed-policy and hard-selector baselines or to choose development-only
candidate weights before simulator execution.

Guardrail: Reward vectors may be used only to report fixed-policy and hard-selector baselines; they must not be used as mixture outcomes.

## Simulator Execution Path

Query 4 must execute every non-oracle treatment by constructing a `BasePolicy`
and running the existing simulator path, not by interpolating metric rows.

Required implementation path:

- build policy objects with `make_policy(...)` for fixed baselines;
- build composed policies with
  `llmserveopt.policies.composition.StaticRankEnsemblePolicy`;
- wrap composed policies with `InstrumentedPolicy` only when trace output is
  requested;
- run through the same `run_policy`/simulator helper used by existing
  policy-comparison scripts;
- compute metrics through `llmserveopt.core.metrics`, with
  `arrival_normalized_weighted_goodput` as primary.

The implementation may reuse existing workload/config loaders but must not run
hosted APIs, GPU jobs, real-vLLM jobs, or cluster submission commands.

## Workloads And Splits

Query 4 should provide two modes.

Smoke mode:

- synthetic tiny workloads or existing local fixture windows only;
- at most 3 scenarios;
- at most 2 seeds;
- at most 3 experts;
- intended to finish in under 60 seconds on CPU.

Full local CC1 mode:

- use representative local simulator workloads that cover underloaded,
  saturated, mixed SLO, prefill-heavy, decode-heavy, KV-pressure, and real-trace
  derived windows when locally available;
- preserve split-group atomicity;
- select fixed policies, mixture weights, `top_k`, and near-tie thresholds on
  development splits only: `TRAIN`, `VALIDATION`, and optional `ROBUST_DEV`;
- evaluate frozen choices on `ID_TEST` and available OOD splits only after all
  development choices are written to the manifest.

If a required local dataset is absent, Query 4 must fail closed with a clear
missing-input message or run only smoke mode.

## Primary Metric: Arrival-Normalized Weighted Goodput

Primary metric:

```text
arrival_normalized_weighted_goodput
```

Report the repository's exact metric column name in CSV as:

```text
metric_arrival_normalized_weighted_goodput
```

Completed-request `weighted_goodput` may be included only as a secondary
conditional-quality metric and must not be used for treatment selection.

## Completion-Fraction Constraints

Every comparison must report:

- `metric_completion_fraction`;
- `metric_num_arrivals` or equivalent arrival count;
- completed-request quality if available;
- rejection fraction when available.

A composed treatment is invalid for a success claim if mean completion fraction
falls more than `0.005` below the best fixed policy on the same evaluation
slice, or if any configured safety/feasibility violation is nonzero.

## Near-Tie Handling

Near ties are windows where the oracle per-window fixed-policy ANWG margin is
less than `0.005` unless a development-only analysis selects a stricter
threshold and records it in the manifest.

Required reports:

- all-window results;
- non-near-tie results;
- near-tie fraction;
- count of meaningful windows;
- sensitivity table for thresholds `0.001`, `0.005`, and `0.01`.

Success claims must be based on non-near-tie windows or explicitly named
strategically important regimes, not on noise-dominated all-window averages.

## Baselines

Best fixed policy:

- choose one policy using development splits only;
- execute that policy in the simulator on evaluation windows.

Learned hard selector:

- use the existing best available feature-only selector path when an artifact
  or trainable local dataset exists;
- the selector chooses one complete policy per window, never a mixture;
- if no local artifact/dataset exists, report `UNAVAILABLE` and keep CC1 from a
  success verdict until it is supplied.

Oracle best fixed per window:

- for each window, use the best realized fixed-policy simulator result among
  the candidate fixed policies;
- this is an upper-bound baseline over fixed policies, not deployable.

Best global mixture:

- select a single weight vector on development data;
- execute that same composed policy on all evaluation windows.

Oracle best mixture per window:

- for each window, search the allowed mixture grid and execute each candidate
  mixture in the simulator;
- choose the best realized simulator outcome for that same window;
- report strictly as a hindsight upper bound.

## Composition-Opportunity-Gap Formula

Primary opportunity gap:

```text
composition_opportunity_gap =
  mean_ANWG(oracle_best_mixture_per_window, evaluation_slice)
  - mean_ANWG(oracle_best_fixed_policy_per_window, evaluation_slice)
```

Also report:

```text
gap_vs_best_global_mixture =
  mean_ANWG(oracle_best_mixture_per_window)
  - mean_ANWG(best_global_mixture)
```

```text
deployable_global_mixture_gain =
  mean_ANWG(best_global_mixture)
  - mean_ANWG(best_fixed_policy)
```

Every formula must be recomputed from simulator-executed rows.

## Success And Stop Thresholds

Proceed threshold:

- aggregate non-near-tie composition opportunity gap at least `0.005`; or
- regime-specific gap at least `0.01` on a known failure regime;
- no material completion-fraction loss beyond `0.005`;
- no safety or feasibility violations;
- result is reproducible across at least two seeds or split groups when those
  groups exist.

Stop threshold:

- non-near-tie composition opportunity gap less than `0.002`; or
- oracle per-window mixture fails to beat oracle per-window fixed policy after
  completion constraints; or
- gains only appear through reduced completion fraction, near-tie noise, or
  reward-vector interpolation.

In the stop case, Query 4 should update the roadmap to record a negative CC1
gate and should not proceed to broad primitive refactoring.

## Required Output Files

Default output root:

```text
results/cc1_composition_opportunity/<timestamp>/
```

Required files:

- `manifest.json`
- `config.yaml`
- `policy_execution_rows.csv`
- `method_comparison.csv`
- `composition_weights.csv`
- `near_tie_sensitivity.csv`
- `subset_analysis.csv`
- `decision_traces/` only when enabled
- `cc1_report.md`

Required manifest fields:

- git branch and SHA;
- dirty-worktree flag;
- config hash;
- random seeds;
- simulator/runtime settings;
- policy subset;
- composition operator and normalization;
- weight grid;
- split names and split-group leakage result;
- development-selection inputs;
- held-out evaluation freeze timestamp;
- no-live-API/no-GPU/no-real-vLLM declarations.

## Reproducibility Requirements

- Use fixed random seeds from config.
- Write all selected weights and treatment choices before evaluation rows.
- Fail if the working tree is dirty unless `--allow-dirty` is explicitly set
  for smoke mode.
- Never use held-out rows for expert selection, weight selection, thresholds,
  selector choice, or fallback tuning.
- Record exact command, config path, output directory, and git SHA.
- Ensure composed policies do not reference `actual_output_tokens`, completed
  metrics, labels, oracle fields, or future arrivals during `select_action`.

## Expected Runtime And Resource Limits

Smoke mode:

- CPU only;
- under 60 seconds expected;
- no network;
- no live APIs;
- no GPU;
- no real-vLLM.

Full local mode:

- CPU only;
- target under 2 hours on a normal workstation;
- default maximum 200 simulator executions unless `--max-runs` is increased;
- checkpoint after every `(scenario, seed, treatment)` row;
- fail closed before launching if the planned run count exceeds the configured
  cap.

## Query 4 File-By-File Implementation Plan

Files to create:

- `configs/cc1_composition_opportunity_smoke.yaml`
- `configs/cc1_composition_opportunity.yaml`
- `scripts/run_cc1_composition_opportunity.py`
- `src/llmserveopt/experiments/cc1_composition_opportunity.py`
- `tests/test_cc1_composition_opportunity.py`

Files to modify:

- `docs/contextual_composition_roadmap.md` after results exist;
- `docs/contextual_composition_decisions.md` only if CC1 changes a decision;
- `docs/START_HERE_CONTEXTUAL_COMPOSITION.md` after results exist;
- GitHub issue #1 with the Query 4 result link;
- `scripts/check_contextual_composition_status.py` only if roadmap status
  changes or new required evidence files are added.

Tests required:

- config parsing and schema validation;
- run-count safeguard and `--max-runs` failure;
- split-group leakage rejection;
- held-out selection guard;
- smoke-mode execution creates all required files;
- composed-policy rows are simulator executed, not reward-vector interpolated;
- no oracle or hidden fields in composition policy code path;
- deterministic replay with fixed seed;
- near-tie classification and opportunity-gap formula;
- learned hard-selector unavailable path is explicit and non-successful.

Config format:

```yaml
schema_version: 1
mode: smoke
seed: 20260731
policy_subset:
  - weighted_shortest_processing
  - edf
  - fifo
composition:
  operator: weighted_borda_rank_aggregation
  implementation: StaticRankEnsemblePolicy
  method: borda
  normalization: per_state_normalized_rank
  top_k: 2
  weight_grid_step: 0.5
metrics:
  primary: arrival_normalized_weighted_goodput
  completion_fraction_tolerance: 0.005
near_tie_thresholds: [0.001, 0.005, 0.01]
safeguards:
  max_runs: 50
  require_clean_git_for_full: true
  forbid_live_api: true
  forbid_gpu: true
  forbid_real_vllm: true
outputs:
  root: results/cc1_composition_opportunity
```

CLI behavior:

```bash
python scripts/run_cc1_composition_opportunity.py --config configs/cc1_composition_opportunity_smoke.yaml --dry-run
python scripts/run_cc1_composition_opportunity.py --config configs/cc1_composition_opportunity_smoke.yaml
python scripts/run_cc1_composition_opportunity.py --config configs/cc1_composition_opportunity.yaml --max-runs 200
```

`--dry-run` must validate inputs, enumerate planned simulator executions, write
no result rows, and never execute simulations.

Dry-run or smoke mode:

- must be the default Query 4 validation mode;
- may allow a dirty tree only with explicit `--allow-dirty`;
- must not claim CC1 success.

Full-run safeguards:

- reject live API, GPU, real-vLLM, and hosted-provider options;
- compute planned run count before execution;
- require explicit `--max-runs`;
- checkpoint rows incrementally;
- resume only from matching config hash;
- fail if output directory already exists unless `--resume` is used.

How results update roadmap and issue #1:

- If Query 4 only implements and smoke-validates, keep CC1 as `NEXT` and issue
  #1 open.
- If Query 4 runs the approved full CC1 experiment, update the roadmap evidence,
  issue #1, and audit report with the measured opportunity gap.
- Do not mark CC1 complete until the full simulator-executed result and
  decision gate are documented.

## Rejected Approaches

Reward-vector interpolation is rejected because it combines stored outcomes
from policies that were executed separately. Such arithmetic does not represent
actions that would have been taken by a composed scheduler, and it cannot
capture admission, queueing, batching, or placement interactions.

Raw score aggregation is deferred because only a subset of policies exposes
single comparable scalar scores and `scorpio_style_slo_guard` is rank/admission
capable but not score-capable.

Component-wise composition is deferred for CC1 because it mixes admission,
ranking, KV, prefill, and aging semantics. It is useful later, but weighted
rank aggregation is the smaller scientifically valid interface.
