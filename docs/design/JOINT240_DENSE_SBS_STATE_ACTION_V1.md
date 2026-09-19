# Joint240 Dense SBS State-Action Dataset v1

Status: preregistered implementation design before dense scoring.

## Estimand

For an acquired predecision state `s_t` from the frozen joint240 Alive trajectory
and a candidate policy `p` in P6:

```text
Q_SBS(s_t, p)
  = terminal utility after forcing p's native action at s_t
    and then following fixed SBS thereafter.

A_SBS(s_t, p)
  = Q_SBS(s_t, p) - Q_SBS(s_t, SBS).
```

`SBS = kv_constrained_online`. This is a one-decision override estimand. It is
not a permanent switch to `p`, an H-step switch, Alive continuation, or
scenario-level policy selection.

## State Universe

The universe is the unique `(scenario_id, step)` set reconciled from:

- `experiments/decision_criticality_terminal_anwg_joint240_v1/branches.csv`
- `experiments/decision_criticality_terminal_utility_joint240_v1/branches.csv`
- `experiments/joint240_terminal_criticality_sbs_continuation_v1/branches_sbs_continuation.csv`
- `experiments/joint240_same_distribution_adaptive_exploitability_v1/split_oof_folds.csv`

Stable state id:

```text
joint240::<scenario_id>::<step>
```

Duplicate acquisition rows with the same `(scenario_id, step)` are represented
once for simulation and retain provenance in the reconciled universe summary.

## LIVE_STATE_FEATURES_V1

Features are computed only from `ObservableState` before the forced action. They
exclude actual future output length, terminal utility, policy outcomes, scenario
IDs, and labels.

Groups:

- queue/load: waiting, migrating-ready, active, admissible, total in-system, token masses.
- KV/memory: used/free/capacity, mean/max utilization, waiting estimated demand, projected pressure.
- prefill/decode: active prefilling/decoding counts, queued prefill mass, active decode remaining mass, mix ratios.
- request-size distribution: prompt, predicted-output, predicted-total summaries.
- SLO/urgency: slack summaries, near-violation counts and token mass.
- fairness/priority: priority mass/max/entropy, class entropy/count, high-priority waiting prompt mass.
- recent dynamics: fixed windows 1, 5, 20 scheduling decisions for queue, KV, completions, queued-token mass, and first-seen request count.
- control history: trajectory-generation Alive effective policy index, steps since switch, recent switch count. These are prefixed `traj_` and should be analyzed separately from physical state.

Exact machine-readable metadata is written to `live_state_features_v1.json`.

## ACTION_DIFF_FEATURES_V1

For every policy `p`, compute `p`'s native action on a deep-copied predecision
state. Compute SBS action the same way. Features compare candidate action to SBS
before either action is executed:

- full action equality and admit-only equality.
- symmetric difference in admitted request ids.
- candidate/SBS admitted counts.
- candidate-only and SBS-only admitted token, priority, slack, age, urgency, prefill/decode summaries.
- preempt/swap/migrate/hold counts and prefill override equality.

Exact machine-readable metadata is written to `action_diff_features_v1.json`.

## Label Generation

For each unique state:

1. replay the frozen scenario under the frozen OOF Alive trajectory machinery;
2. stop at the target predecision step;
3. compute all six P6 native actions from independent copies of the same state;
4. for each policy, fork the live simulator, force that action once, and continue with fixed SBS;
5. compute ANWG and continuous terminal utility metrics where supported;
6. write one long-form row per `(state_id, candidate_policy_id)`.

Identical actions are not collapsed in the output. They are marked by action
equivalence features and should produce identical deterministic downstream
results.

Expected dense shape: `N_unique_states x 6` rows.

## Splitting And Leakage Rules

Folds are inherited from frozen `split_oof_folds.csv`. State/action rows must be
split by scenario/fold, never randomly across rows from the same scenario.
Training must not use `scenario_id`, terminal utility columns, old Alive/SBS
criticality labels, or outcome-derived metadata as model features.

## Execution

Interactive runs are limited to tests and small pilots. Full dense scoring is a
Slurm array job using `scripts/slurm/joint240_dense_sbs_state_action_v1.sbatch`
with shard sentinels and per-task logs.
