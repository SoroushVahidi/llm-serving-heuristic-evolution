# Contextual Composition Query 5 Discriminativeness Review - 2026-07-31

Branch: `contextual-compositional-heuristics-20260731`

Starting SHA: `47277b0b0c035491929960522d67d717efb91850`

Canonical issue: [#1](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/1)

## Diagnosis

The Query 4 CC1 `STOP_OR_REDESIGN` verdict was a real result for that suite,
not a reward-vector interpolation or oracle-accounting bug.

Implementation checks:

- fixed policies and mixtures both executed through `run_policy`;
- mixture rows recorded `true_simulator_executed=True`;
- mixture rows recorded `reward_vector_interpolated=False`;
- oracle fixed and oracle mixture were recomputed from simulator-executed rows;
- verdict logic correctly stopped because oracle mixture did not beat oracle
  fixed and there were zero non-near-tie evaluation windows.

Why Query 4 ANWG was `1.0`:

- service capacity was high relative to the generated arrivals: two GPUs with
  eight active sequences each, short decode times, and most windows below the
  effective throughput limit;
- the default service model did not enable prefill modeling, so long prompts
  mostly affected ranks and KV feasibility, not service duration;
- SLO slack was usually much looser than observed latency. Evaluation latencies
  were commonly around `0.03s` to `0.20s`, while default SLO classes included
  `0.5s`, `2.0s`, and `10.0s` slack;
- full drain behavior let nearly all queued requests complete, so completion
  loss rarely affected ANWG;
- policy differences did exist in latency, TTFT, GPU utilization, and occasional
  Scorpio drops, but several fixed policies still completed all requests within
  SLO, causing oracle fixed and oracle mixture to tie at ANWG `1.0`.

The five-policy subset was not the primary failure. Under tighter SLOs and
prefill contention, the same subset produced clear fixed-policy spread and
nonzero mixture opportunity.

## CC1b Design

CC1b was justified because the simulator can produce discriminative,
causal, non-near-tie windows with existing capabilities.

Created:

- `configs/cc1b_composition_discriminative_smoke.yaml`
- `configs/cc1b_composition_discriminative.yaml`

Implementation additions:

- `mode: cc1b` in the existing runner;
- YAML-defined `slo_classes` for synthetic workloads;
- optional request transforms for real traces:
  `arrival_time_scale`, `slo_slack_scale`, `slo_slack_cap`, and
  `slo_slack_floor`;
- `fixed_policy_spread.csv`;
- a fixed-policy spread gate that runs before mixture evaluation in `cc1b`
  mode.

CC1b uses the same true simulator-executed weighted Borda composition path as
CC1. It does not interpolate reward vectors.

## Workloads And Search

Policy subset:

- `weighted_shortest_processing`
- `scorpio_style_slo_guard`
- `edf`
- `estimated_service_time_first`
- `fifo`

Search:

- weighted Borda rank aggregation;
- deterministic simplex grid;
- step `0.25`;
- `top_k = 2`;
- 35 mixtures;
- nonnegative normalized weights;
- deterministic tie-breaking.

Simulator settings:

- one local CPU simulator GPU config with four active sequences;
- prefill modeling enabled;
- decode/prefill contention enabled;
- `step_token_budget = 128`;
- no live APIs, GPU jobs, hosted-provider calls, or real-vLLM jobs.

Full CC1b workload count:

- 11 windows;
- 440 true simulator executions;
- runtime: `90.169s`.

Regimes:

- overload with tight SLOs;
- long prompts plus mixed tight SLOs;
- burst transition;
- KV pressure;
- selective-admission and priority conflict;
- Azure-conversation-like OOD real-trace-derived window with disclosed arrival
  and SLO-slack transforms.

Held-out evaluation windows:

- `cc1b_overload_id_test`
- `cc1b_long_prompt_id_test`
- `cc1b_burst_id_test`
- `cc1b_azure_conv_ood_test`

The fixed-policy spread gate required at least three evaluation windows with
fixed-policy spread >= `0.03` and top-two fixed-policy margin >= `0.005`.
The gate passed.

## Commands

```bash
python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1b_composition_discriminative.yaml \
  --dry-run

python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1b_composition_discriminative_smoke.yaml \
  --dry-run

python -m pytest tests/test_cc1_composition_opportunity.py -q

python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1b_composition_discriminative_smoke.yaml \
  --timestamp query5_cc1b_smoke_20260731 \
  --allow-dirty

python scripts/run_cc1_composition_opportunity.py \
  --config configs/cc1b_composition_discriminative.yaml \
  --full-run \
  --max-runs 500 \
  --timestamp query5_cc1b_full_20260731
```

The full run was executed from clean commit
`45e664ea7146425a7cc2aca35773f7e4b6d6dc14`. Generated result artifacts remain
under ignored `results/`.

## Results

Full output directory:

```text
results/cc1b_composition_discriminative/query5_cc1b_full_20260731/
```

Required output files include:

- `manifest.json`
- `config.yaml`
- `policy_execution_rows.csv`
- `fixed_policy_spread.csv`
- `per_window_summary.csv`
- `method_comparison.csv`
- `composition_weights.csv`
- `near_tie_sensitivity.csv`
- `subset_analysis.csv`
- `verdict.json`
- `cc1_report.md`

CC1b verdict: `PROCEED`

Key metrics:

- best fixed policy: `fixed__estimated_service_time_first`
- best global mixture: `mix__estimated_service_time_first-1p000`
- oracle fixed mean ANWG: `0.203773`
- oracle mixture mean ANWG: `0.220547`
- composition-opportunity gap: `0.0167735`
- non-near-tie composition-opportunity gap: `0.0167735`
- non-near-tie evaluation windows: `4`
- completion impact vs best fixed: `0.0`
- best regime gain: `0.05`
- near-tie fraction at threshold `0.005`: `0.0`

Per-regime held-out opportunity:

- `overload_tight_slo`: gap `0.05`
- `long_prompt_mixed_tight_slo`: gap `0.017094`
- `burst_transition`: gap `0.0`
- `azure_conversation_like_tight_slo`: gap `0.0`

The deployable global mixture did not beat the best fixed policy. The positive
gate is specifically an oracle composition-opportunity result, which justifies
CC2 primitive-interface work but does not yet justify a deployable contextual
predictor.

## Limitations

- CC1b is compact by design and should not be interpreted as a broad final
  benchmark.
- The Azure-conversation-like window is transformed real-trace-derived
  simulation, not a real-serving result.
- The strongest evidence is oracle per-window mixture opportunity, not a
  learned deployable policy.
- One-hot rank-expert mixtures are not identical to all native fixed policies
  when the native policy has admission state, especially
  `scorpio_style_slo_guard`; this is expected under the approved rank
  composition semantics and should inform CC2 interface design.

## Final Decision

Continue to CC2.

CC1b establishes enough non-near-tie simulator evidence to justify defining
the canonical primitive interface. Query 6 should begin CC2 by specifying and
testing reusable primitive contracts for ranking, admission, placement,
batching, and resource guards. Query 6 should not begin CC3 DSL extensions,
contextual predictor training, live APIs, GPU jobs, or real-vLLM work.
