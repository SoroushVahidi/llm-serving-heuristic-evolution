# CC4 Report: Offline Oracle Composition Dataset

Date: 2026-08-03
Branch: `contextual-compositional-heuristics-20260731`
Starting SHA: `19708f741d0bfb944b4a11ff34572a811df94d66`
Canonical issue: [#4](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/4).
tmux session: `cc4_oracle_dataset`; log: `logs/cc4_oracle_dataset_20260803_163956.log`.
Reference dataset directory: `results/cc4_oracle_composition_dataset/20260803T170735Z/`
(untracked/local, per repository convention -- see §12; reproducible via
`bash results/cc4_oracle_composition_dataset/20260803T170735Z/replay_commands.sh`).

## 1. Goal

Build the first reproducible, resumable, simulator-derived oracle
composition dataset over the CC2 primitive registry and CC3 verified
compositional DSL, sized to support later CC5 contextual-predictor training,
without beginning CC5 training itself.

## 2. Reuse And Design Summary

Full design note is in the session log (written before implementation).
Summary of what was reused rather than re-derived, per the infrastructure
audit performed before writing any code:

* Workload-window construction (with its split-leakage assertion),
  GPU/service-model construction, git-state capture, and CSV writing are
  imported directly from `llmserveopt.experiments.cc1_composition_opportunity`
  -- no new simulator-invocation or workload-window code was written.
* `HeuristicPolicy` (CC3) is already a `BasePolicy`; a candidate composition
  is executed by `verify_heuristic() -> compile_heuristic() ->
  HeuristicPolicy() -> run_policy()`, the exact call CC1 makes for fixed
  policies. No new simulator-execution path exists.
* Resumability follows `scripts/run_module_credit_overnight.py`'s
  append-only-store-plus-heartbeat pattern, keyed by a composite
  `"{window_id}::{candidate_id}"` string instead of an integer trial
  counter (`CC4TrialStore` in the new module).
* Oracle/regret/near-tie definitions extend CC1's own codified definitions
  (`near_tie = top-2 ANWG margin < threshold`; oracle = post-hoc argmax over
  already-executed rows, requiring no extra simulator calls) generalized
  from "best fixed policy" to "best candidate among the full searched pool."
* Split assignment reuses CC1's simple explicit-per-workload-entry `split:`
  field plus its leakage check (`build_workload_windows`), not
  `selector/dataset_v2/splits.py`'s dynamic group-hash assignment -- CC4's
  window catalog is a small, fixed, hand-authored set like CC1b's, so there
  is no dynamic group set to hash. `dataset_v2/splits.py` remains available
  to CC5 if it needs group-level splitting over CC4's *output* rows.

New code, all in `src/llmserveopt/experiments/cc4_oracle_composition_dataset.py`
(+ `scripts/run_cc4_oracle_composition_dataset.py` CLI +
`configs/cc4_oracle_composition_dataset.yaml`): candidate generation (fixed
baselines, the CC1b weighted-Borda baseline replayed unchanged, bounded
weighted-primitive-mixture grid search, sparse top-k mixtures, admission-gate
and placement variants), verify-before-execute wrapper, `CC4TrialStore`,
oracle/regret/near-tie/completion-constraint computation, primitive-usage
and search-summary statistics, a clean CloudRift opt-in skip stub, and the
15-file dataset writer.

## 3. Search Design (Deterministic, No RNG)

All candidate generation is grid/enumeration -- no random search, no
reward-vector interpolation anywhere in the pipeline (`reward_vector_interpolated`
is `False` on every row by construction; every executed row's
`true_simulator_executed` is `True`).

| Family | Count | Construction |
|---|---|---|
| `fixed_policy` | 5 | `fifo, edf, weighted_shortest_processing, estimated_service_time_first, scorpio_style_slo_guard` via `make_policy` |
| `cc1b_borda_baseline` | 1 | CC1b's discovered weighted-Borda mixture, replayed unchanged (roadmap-required comparison, not re-searched) |
| `weighted_primitive_mixture` | 21 | `weighted_sum` DSL over a 6-primitive RANKING pool (`laxity_urgency, priority, queue_age, predicted_output_length, prompt_length, estimated_service_time`, each pre-oriented "higher is better"), simplex weight grid (step 0.5, top_k 2) |
| `sparse_topk_mixture` | 3 | `topk_mixture` DSL, same pool, k in {1, 2, 3} |
| `admission_gate_variant` | 2 | `laxity_urgency` ranking + `laxity_gate` primitive_gate admission_condition, `laxity_threshold` in {0.0, 0.05}, declared `fallback: fifo_like` + `on_no_admits: safe_fallback` |
| `placement_variant` | 2 | `laxity_urgency` ranking + `placement.keys` in `{[projected_gpu_load], [projected_gpu_load, kv_pressure]}` |
| **Total** | **34** | 0 rejected by the verifier |

Per-regime and per-window oracle candidates are **post-hoc selections** over
this one executed (12 windows x 34 candidates = 408 simulator executions)
grid -- not separate DSL programs or additional simulator calls, exactly
mirroring how CC1 computes `oracle_best_fixed_per_window` /
`oracle_best_mixture_per_window` without re-running anything.

Per CC3's documented unresolved risk (its report §9), `admission_budget` is
**not used at all** in this search space -- a deliberate scope decision, not
an oversight, so it is never combined with `on_no_admits: safe_fallback`.

## 4. Workloads And Splits

12 windows, 0 skipped (both required real-trace files -- Azure conversation
and BurstGPT -- are present locally):

| Split | Windows |
|---|---|
| TRAIN | `underloaded`, `saturated`, `mixed_slo` (3) |
| VALIDATION | `long_prompt`, `long_output`, `burst_transition` (3) |
| ID_TEST | `kv_pressure`, `prediction_noise`, `priority_conflict`, `selective_admission_trap` (4) |
| OOD_TEST | `azure_conversation_like` (real trace), `burstgpt_derived` (real trace) (2) |

All 12 required regime categories from the CC4 task (underloaded, saturated,
mixed SLO, long prompt, long output, burst transition, KV pressure,
prediction noise, priority conflict, selective-admission trap, Azure-like,
BurstGPT-derived) are covered by exactly one window each.

## 5. Bugs Found And Fixed During Development

1. **BurstGPT real-trace arrival-time-scale hang.** The first full run
   ballooned to 3.6 GB RSS at 100% CPU and stalled at 375/408 rows. Raw
   BurstGPT arrival timestamps are in seconds with a mean inter-arrival gap
   of ~171s (an 80-request window spans ~12,555s); the config's
   `request_transform.arrival_time_scale` was left at `1.0` (copied from
   CC1b's Azure entry without adjusting for BurstGPT's much larger native
   timescale). At `service_model.step_size=0.001` that implied ~12.5
   million simulation steps -- effectively a hang. Fixed by computing
   `arrival_time_scale=0.00012` to compress the window to a ~1.5s span
   matching the other windows. The corrupted partial output directory was
   discarded rather than resumed from (resuming across a config change that
   alters request content for an unchanged `window_id` would have mixed
   old- and new-scale rows under one nominal window -- a data-integrity
   hazard) and the search was re-run cleanly.
2. **Dev/eval split fields collected but unused.** `development_splits`/
   `evaluation_splits` existed in the config schema but `determine_dataset_verdict`
   originally computed `oracle_composition_gain`/`near_tie_fraction` over
   *all* 12 windows (dev+eval mixed), which would have let TRAIN-window
   signal certify the held-out claim -- the same mistake CC1's own
   dev/eval separation exists to prevent. Fixed by filtering the verdict
   computation to the 6 evaluation-split windows only (`oracle_labels.parquet`
   etc. still cover all 12 windows for CC5's own training use). Locked in
   by `test_dataset_verdict_ignores_development_split_windows`.
3. Resumability was deliberately stress-tested by killing a run mid-search
   (`timeout 100`) and restarting with `--resume-dir`: the `CC4TrialStore`
   correctly skipped every already-completed `(window_id, candidate_id)`
   pair and merged rows without duplication or loss. This is also covered
   by an automated test (`test_reproducible_shard_merging_via_resume`)
   using the real simulator, not a mock.

## 6. Dataset Outputs

All 15 required files present in the reference dataset directory:
`manifest.json`, `resolved_config.yaml`, `workload_windows.parquet`,
`causal_features.parquet`, `candidate_compositions.parquet`,
`per_window_results.parquet`, `oracle_labels.parquet`, `regret_matrix.parquet`,
`composition_parameters.parquet`, `near_tie_flags.parquet`,
`completion_constraints.parquet`, `primitive_usage_statistics.csv`,
`search_summary.csv`, `dataset_card.md`, `replay_commands.sh`, plus
`checkpoints/trial_results.jsonl` and `checkpoints/heartbeat.json` (resumability
state, not a required top-level file but kept for audit/debugging).

Every `per_window_results.parquet` row carries: window/split/regime/source/
seed, candidate_id/family/policy_name, git SHA, composition hash, DSL schema
version + compiler version, primitive weights + extra params (JSON), every
`metric_*` field from `metrics_to_dict` (ANWG, completion fraction, SLO
violation rate, mean/median/max latency, mean TTFT, mean TPOT, throughput,
GPU utilization, etc. -- whatever the simulator reports, none synthesized),
verification outcome, runtime, and a best-effort `fallback_activated_last_step`
flag (see §10 limitations).

## 7. Simulator Executions And Search Summary

```
n_windows: 12                              n_windows_skipped_real_trace: 0
n_candidates_total: 34                     n_candidates_rejected: 0
n_simulator_executions: 408                n_unique_verified_compositions: 28
n_candidates__admission_gate_variant: 2     n_candidates__cc1b_borda_baseline: 1
n_candidates__fixed_policy: 5               n_candidates__placement_variant: 2
n_candidates__sparse_topk_mixture: 3        n_candidates__weighted_primitive_mixture: 21
```

Full local run: 270.8s wall-clock, 408/408 simulator executions, 0 GPU, 0
live API calls, 0 real-vLLM. Reproducibility was verified twice: once via a
kill-mid-run-then-resume cycle (§5.3) and once via an independent from-scratch
re-run (`20260803T170735Z`) that reproduced byte-identical verdict numbers
to the earlier resumed run.

## 8. Quality Statistics

**Oracle composition gain** (evaluation-split windows only, n=6): a
composition-family candidate (not a plain fixed policy or the CC1b
baseline) is the oracle winner in **4/6 (66.7%)** evaluation windows
(`kv_pressure`, `prediction_noise`, `priority_conflict`,
`selective_admission_trap`); plain fixed policies win on `azure_conversation_like`
and `burstgpt_derived`.

**Near-tie fraction** (evaluation-split windows, primary threshold 0.005):
**50%** (3/6) -- stable across all three swept thresholds (0.001, 0.005,
0.01), i.e. the near-tie windows are near-tie by a wide margin, not
threshold-sensitive artifacts. Per roadmap invariant 9, near-tie windows are
reported here, not excluded, and do not dominate the aggregate signal (the
non-near-tie half still shows a clear, non-trivial composition advantage).

**Regret distribution by family** (regret = window's oracle ANWG − this
candidate's ANWG; lower is better; n = rows per family across all 12
windows):

| Family | mean | median | max | n |
|---|---|---|---|---|
| `admission_gate_variant` | 0.0371 | 0.0161 | 0.1493 | 24 |
| `cc1b_borda_baseline` | 0.0507 | 0.0481 | 0.1638 | 12 |
| `weighted_primitive_mixture` | 0.0874 | 0.0394 | 0.4837 | 252 |
| `fixed_policy` | 0.1047 | 0.0673 | 0.4837 | 60 |
| `sparse_topk_mixture` | 0.1408 | 0.0695 | 0.4837 | 36 |
| `placement_variant` | 0.1633 | 0.0695 | 0.4837 | 24 |

`admission_gate_variant` has the lowest mean and median regret of every
family, including the individually-tuned `weighted_primitive_mixture` grid
-- a legible early signal that admission-aware compositions are
disproportionately valuable relative to their small candidate count (2 of
34), worth weighting more heavily in a future CC4 iteration's search budget.

**Completion-fraction constraints**: `completion_ok = True` on all 12
windows (tolerance 0.005) -- the oracle composition never reduces completion
fraction relative to the window's best fixed policy; on 4 windows
(`burst_transition`, `kv_pressure`, `saturated_train`,
`selective_admission_trap`) it substantially *improves* completion fraction
(+0.32 to +0.81), the clearest evidence in this dataset that composition
helps under admission pressure, not just ranking-quality pressure.

**Primitive usage**: `laxity_urgency` is both the most-searched (13
candidates reference it) and most-often-oracle-selected primitive (3 of 12
windows); `predicted_output_length` and `queue_age` are each selected once.
`priority`, `prompt_length`, and `estimated_service_time` are searched (9
candidates each) but never individually part of an oracle-winning
composition in this run -- informative for narrowing a future CC4 pool.

**Flagged (not rejected) degenerate window**: `cc4_burstgpt_derived_ood_test`
has a very low completion fraction (0.0375) across *every* candidate and a
near-tie margin of exactly 0.0 -- the compressed real-trace window is close
to system collapse regardless of scheduling choice. It is retained (per
roadmap invariant 9, near-tie/degenerate windows must be reported, not
silently dropped) but its oracle label carries little discriminative
signal; a future CC4 iteration should either loosen its GPU capacity or
choose a less-saturating trace slice if more BurstGPT-derived signal is
wanted.

No window had all candidates rejected, an inconsistent completion count, or
an oracle label depending on a non-causal field (verification-before-execution
and the CC3 verifier's causal-only variable enforcement make the latter
structurally impossible, not just empirically absent here).

## 9. API Usage And Cost

CloudRift was not used. `config['cloudrift']` was not set (`enabled: false`
by omission), so `maybe_generate_cloudrift_candidates` returned an empty
list with `skip_reason: "cloudrift.enabled is false in config"` before ever
checking for credentials -- confirmed in `manifest.json`'s `cloudrift` block
(`used: false, calls: 0, cost_usd: 0.0`). No live API, GPU, or real-vLLM
execution occurred anywhere in this build (`no_live_api`/`no_gpu`/
`no_real_vllm` are all `true` in the manifest).

## 10. Limitations

* **Bounded search scale.** 12 windows and 34 candidates were chosen to
  keep a full local run to ~4-5 minutes without GPU/live API, consistent
  with the issue's own "first reproducible" framing. Expanding window count
  and candidate-pool breadth (especially more admission-gate and weighted-
  mixture variants, given §8's regret signal) is the natural next step
  before CC5 needs maximum training signal.
* **`fallback_activated_last_step` is best-effort, not a full-run
  aggregate.** `HeuristicPolicy.last_trace` (CC3) only reflects the most
  recent scheduling step, not whether fallback ever activated at any point
  during a run; extending CC3's instrumentation to accumulate this across a
  full run was out of CC4's scope (would touch CC3's frozen deliverable).
* **`admission_budget` was not searched** (§3) -- CC3's own documented risk
  about combining it with `on_no_admits: safe_fallback` made this the safer
  scope boundary for a first dataset; a future iteration can search it
  alone (without `safe_fallback`) if desired.
* **`placement.keys` composes lexicographically only** (inherited from
  CC3's own documented scope boundary, not newly introduced here).
* One evaluation window (`cc4_burstgpt_derived_ood_test`) is near-collapse
  and low-signal (§8) -- flagged, not excluded.

No critical correctness issue remains: verification runs before every
execution, the split-leakage check passes on the full 12-window catalog,
completion fractions are always reported and internally consistent, and
regret/near-tie/oracle-label computations are covered by both hand-computed
unit tests and full end-to-end reproducibility checks (§5.3, §7).

## 11. Tests And Exact Commands

```bash
python -m pytest tests/test_cc4_oracle_composition_dataset.py -q
# 20 passed
```

Covers: candidate-generation determinism (§3), DSL verification-before-execution
(valid candidates all pass; a deliberately-broken candidate is rejected
*before* any simulator call), split integrity (both the tiny test config and
the full 12-window production config build without leakage), no
reward-vector interpolation (`true_simulator_executed=True`,
`reward_vector_interpolated=False` structurally on every executed row),
resumable checkpointing (a fresh `CC4TrialStore` instance over the same
directory recovers exactly the prior completed-keys set), reproducible
shard merging (a real, non-mocked two-call `run_search` integration test:
the second call with `resume_dir` makes zero additional simulator calls and
produces byte-identical row counts), oracle-label/regret/near-tie/completion-
constraint correctness (hand-computed small examples), search-summary
counts, CloudRift clean-skip (both the "not enabled" and "enabled but no
key" paths), and verdict determination including the dev/eval separation
fix (§5.2).

## 12. Commit And Reproducibility Policy

Per repository convention (`results/*` is gitignored; CC1b's own
`results/cc1b_composition_discriminative/` directory has never been
committed), the dataset directory itself
(`results/cc4_oracle_composition_dataset/20260803T170735Z/`, ~1 MB) is
**not** committed. It is fully reproducible via:

```bash
bash results/cc4_oracle_composition_dataset/20260803T170735Z/replay_commands.sh
```

which re-invokes `scripts/run_cc4_oracle_composition_dataset.py` with the
exact config, and will complete instantly if the local `checkpoints/`
directory is intact (resumed), or in ~4-5 minutes from scratch.

## 13. CC4 Verdict

**CC4 exit gate: PASSED (dataset status `COMPLETE`).**

* Dataset is reproducible (byte-identical verdict across an independent
  from-scratch re-run) and resumable (verified via an interrupt-and-resume
  cycle plus an automated integration test).
* Manifests are complete (all 15 required files, `manifest.json` carries
  every required provenance field).
* Split integrity passes (`build_workload_windows`' leakage check; 6 of 12
  windows reserved for evaluation, never used to certify the aggregate
  verdict, matching CC1's dev/eval separation after the §5.2 fix).
* Oracle labels come from true simulator execution exclusively -- no
  reward-vector interpolation anywhere in the pipeline.
* Sufficient non-near-tie signal exists for CC5: 3 of 6 evaluation windows
  are clearly non-near-tie with a real (up to 66.7%-of-windows) composition
  advantage and, on several windows, a substantial completion-fraction
  improvement.
* No critical correctness issue remains (§10 lists scope boundaries and one
  flagged low-signal window, not defects).

## 14. Exact CC5 Entry Condition

CC5 (contextual composition predictor) is **not begun in this query**. A
future, explicitly authorized query should begin it by:

1. reading this report and `docs/architecture/contextual_composition_dsl.md`;
2. training against `oracle_labels.parquet`/`regret_matrix.parquet`/
   `causal_features.parquet` (joined on `window_id`), using
   `development_splits` windows for fitting and `evaluation_splits` windows
   exclusively for the reported validation claim;
3. considering widening the CC4 search first (more windows, more
   admission-gate and weighted-mixture variants per §8's regret signal) if
   CC5's own held-out evidence looks data-starved with only 6 evaluation
   windows;
4. resolving or explicitly re-scoping the `admission_budget` /
   `fallback_activated_last_step` limitations in §10 if CC5's predictor
   needs to condition on either.
