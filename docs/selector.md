# Policy Selector

The selector is a supervised portfolio policy that maps online-observable workload features to one deployable scheduling policy.

## What the selector is

The selector does **not** create new heuristics.  It chooses, at the granularity of a request window (W = 200 requests by default), which of the 18 registered deployable policies to use next.  This is a supervised classification problem: label = argmax policy by priority-weighted SLO goodput (internal name: `weighted_goodput`) on that window.

## What the selector is not

- It does not learn or synthesize new scheduling rules.
- It does not call any external APIs.
- `oracle_srtf` is never a label candidate — it is a non-deployable hindsight upper bound only.

## Selector candidate set

All 19 online-deployable baselines registered in `BASELINE_NAMES` (from `llmserveopt/policies/registry.py`) minus any entry in `ORACLE_POLICY_NAMES`.

```
fifo, edf, shortest_output_first, shortest_prompt_first,
greedy_token_fill, least_loaded, multi_bin_batching, random_feasible,
orca_style, vllm_style_token_budget, sarathi_style, splitfuse_style,
slo_slack_score, weighted_shortest_processing, first_fit, best_fit,
least_laxity_first, estimated_service_time_first, admission_control
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
| `RuleBasedSelector` | Deterministic feature-based dispatch. **Was FIFO-only placeholder before Phase 2B.5; repaired in Phase 2B.8 (KV-pressure guard + noise guard).** Is a baseline/adaptive method, not an oracle. Uses only online-observable features — no `actual_output_tokens`. See [Phase 2B.8 rule change](#phase-2b8-rule-selector-repair) below. |
| `DecisionTreeSelector` | `max_depth=8, min_samples_leaf=20` — interpretable |
| `RandomForestSelector` | `n_estimators=200, max_depth=10` — main v1 selector |

## Phase 2B.8 Rule Selector Repair

**Phase 2B.7 finding:** Rule 1 (`fraction_tight_slo > 0.4 OR min_slack < 1.0 → least_laxity_first`)
fired for ALL overloaded workloads, producing catastrophic WG loss in 3 of 4 test regimes:

| Failure | WG (old LLF) | WG (best fixed) | Delta |
|---|---|---|---|
| overloaded_mixed_slo | 0.474 | 0.905 (slo_slack_score) | −0.431 |
| high_prediction_noise | 0.584 | 0.988 (admission_control) | −0.404 |
| kv_pressure_decode_heavy | 0.101 | 0.477 (weighted_shortest_processing) | −0.376 |

**Phase 2B.8 repair:** Three changes to `RuleBasedSelector.predict_one()` (in rule priority order):

1. **Rule 1 (new, elevated):** `mean_pred_output_tokens > 200 OR kv_utilization > 0.7`
   → `weighted_shortest_processing`
   *KV pressure proxy: large outputs fill KV slots longest; WSP is short-job-first, which frees KV quickly. `kv_utilization` is available in online deployments; `mean_pred_output_tokens` works offline.*

2. **Rule 2 (new):** `pred_output_cv > 1.0`
   → `admission_control`
   *High prediction noise (CV > 1.0 indicates 70%+ noise): laxity estimates unreliable; AC with urgency sort is more robust.*

3. **Rule 4 (modified):** `fraction_tight_slo > 0.4 OR min_slack < 1.0`
   → `slo_slack_score` (was `least_laxity_first`)
   *Composite urgency+throughput score; avoids LLF throughput collapse under queue build-up.*

The selector uses only online-observable features and remains deterministic.
Any remaining failure cases will drive either modern baseline additions or LLM-assisted rule synthesis in future phases.

## Window construction

Non-overlapping windows of W=200 requests.  Partial tail windows with fewer than 50 requests are dropped by default.

## Feature leakage guards

- `actual_output_tokens` is never accessed in feature extraction.
- Features depend only on requests arrived before or at `window_start_time`.
- Changing future requests (beyond window end) must not alter current window features.
- Tests in `tests/test_selector_no_leakage.py` enforce these invariants.

## Phase 2A.4 evaluation results (current)

The selectors were retrained and evaluated on an expanded 18-policy dataset with 52 total windows across three splits.

| Split | Windows | Config |
|-------|---------|--------|
| Train | 30 | phase2a4_train_18policies.yaml |
| Validation | 13 | phase2a4_validation_18policies.yaml |
| Test | 9 | phase2a4_test_18policies.yaml |

| Model | Split | Sel WG | Accuracy | Δ vs Best Fixed |
|-------|-------|--------|----------|----------------|
| rule_based | test | 0.597 | 0.0% | -0.200 |
| decision_tree | test | 0.828 | 55.6% | **+0.030** |
| **random_forest** | **test** | **0.828** | **55.6%** | **+0.030** |

Best fixed baseline on test split: `shortest_output_first` (WG = 0.798). The RF/DT selectors both achieve approximately +3.0% improvement over the best fixed policy.

**Test trace exclusion**: `burstgpt_scaled_high_10k.jsonl` appears only in the test config — it is never used in train or validation splits. This prevents data leakage from the test trace into selector training.

## Phase 2A.3 evaluation results (superseded)

Phase 2A.3 used 16 policies and 19 train windows. Superseded by Phase 2A.4 (18 policies, 30 train windows).

| Model | Split | Sel WG | Δ vs Best Fixed |
|-------|-------|--------|----------------|
| rule_based | test | 0.597 | -0.200 |
| decision_tree | test | 0.797 | -0.001 |
| **random_forest** | **test** | **0.828** | **+0.030** |

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
