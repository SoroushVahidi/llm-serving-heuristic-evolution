# Policy Selector

The selector is a supervised portfolio policy that maps online-observable workload features to one deployable scheduling policy.

## What the selector is

The selector does **not** create new heuristics.  It chooses, at the granularity of a request window (W = 200 requests by default), which of the 18 registered deployable policies to use next.  This is a supervised classification problem: label = argmax policy by priority-weighted SLO goodput (internal name: `weighted_goodput`) on that window.

## What the selector is not

- It does not learn or synthesize new scheduling rules.
- It does not call any external APIs.
- `oracle_srtf` is never a label candidate — it is a non-deployable hindsight upper bound only.

## Selector candidate set

All 18 online-deployable baselines registered in `BASELINE_NAMES` (from `llmserveopt/policies/registry.py`) minus any entry in `ORACLE_POLICY_NAMES`.

```
fifo, edf, shortest_output_first, shortest_prompt_first,
greedy_token_fill, least_loaded, multi_bin_batching, random_feasible,
orca_style, vllm_style_token_budget, sarathi_style, splitfuse_style,
slo_slack_score, weighted_shortest_processing, first_fit, best_fit,
least_laxity_first, estimated_service_time_first
```

Source of truth: `src/llmserveopt/selector/candidates.py`.

## Features (18 total)

Features are extracted in `online_prefix` mode by default: using only information available at the start of the window (requests that arrived before or at window start time, plus queue state).  `actual_output_tokens` is **never** used.

| Feature | Description |
|---------|-------------|
| `queue_length` | Approximate number of queued requests at window start |
| `active_sequence_count` | GPU active sequences (0 in offline mode) |
| `kv_utilization` | KV-cache fill fraction (0.0 in offline mode) |
| `free_sequence_ratio` | Free GPU slots fraction (1.0 in offline mode) |
| `mean_prompt_tokens` | Mean prompt length in window |
| `p95_prompt_tokens` | 95th percentile prompt length |
| `mean_pred_output_tokens` | Mean predicted output length |
| `p95_pred_output_tokens` | 95th percentile predicted output |
| `pred_output_cv` | CV of predicted output (burstiness proxy) |
| `fraction_tight_slo` | Fraction of window requests with class_id="tight" |
| `mean_slack` | Mean SLO slack (deadline − arrival) |
| `p10_slack` | 10th percentile SLO slack |
| `min_slack` | Minimum SLO slack |
| `mean_waiting_time` | Mean waiting time of prefix requests |
| `p95_waiting_time` | 95th percentile waiting time of prefix requests |
| `arrival_rate_est` | Estimated arrival rate (req/s) from prefix |
| `burstiness_cv` | CV of inter-arrival times from prefix |
| `recent_slo_violation_rate` | SLO violation rate in recently completed requests |

## Label objective

```
label = argmax_{p in SELECTOR_CANDIDATES} weighted_goodput(p, window)
```

Tie-breaking (in order):
1. Lower `slo_violation_rate`
2. Lower `p95_ttft`
3. Lower `p95_latency`
4. Higher `request_throughput`
5. Alphabetical policy name (deterministic)

## Models

| Model | Use |
|-------|-----|
| `RuleBasedSelector` | Always returns "fifo" — baseline for comparison |
| `DecisionTreeSelector` | `max_depth=8, min_samples_leaf=20` — interpretable |
| `RandomForestSelector` | `n_estimators=200, max_depth=10` — main v1 selector |

## Window construction

Non-overlapping windows of W=200 requests.  Partial tail windows with fewer than 50 requests are dropped by default.

## Feature leakage guards

- `actual_output_tokens` is never accessed in feature extraction.
- Features depend only on requests arrived before or at `window_start_time`.
- Changing future requests (beyond window end) must not alter current window features.
- Tests in `tests/test_selector_no_leakage.py` enforce these invariants.

## Phase 2A.3 evaluation results

The Random Forest selector was evaluated on train/validation/test/sanity splits built from stressed synthetic regimes and real BurstGPT traces.

| Model | Split | Sel WG | Δ vs Best Fixed |
|-------|-------|--------|----------------|
| rule_based | test | 0.597 | -0.200 |
| decision_tree | test | 0.797 | -0.001 |
| **random_forest** | **test** | **0.828** | **+0.030** |

The Random Forest selector beats the single best fixed policy (`shortest_output_first`) by +3% weighted goodput on the test split.

**Limitations**: Dataset is small (19 train windows). Phase 2A.4 scales to full BurstGPT traces.

## Usage

```bash
# Build smoke dataset
python scripts/build_selector_dataset.py \
    --config configs/selector/selector_dataset_smoke.yaml \
    --output results/selector_dataset/smoke_selector_dataset.csv

# Build full Phase 2A.3 datasets
python scripts/build_selector_dataset.py \
    --config configs/selector/selector_dataset_train_phase2a3.yaml \
    --output results/phase2a3_selector_eval/datasets/train_selector_dataset.csv

# Train with separate validation set
python scripts/train_policy_selector.py \
    --dataset results/.../train.csv \
    --validation-dataset results/.../validation.csv \
    --output results/.../models/

# Evaluate on held-out splits
python scripts/evaluate_policy_selector.py \
    --models-dir results/.../models \
    --validation-dataset results/.../validation.csv \
    --test-dataset results/.../test.csv \
    --output results/.../evaluation/
```
