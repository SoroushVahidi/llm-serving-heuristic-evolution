# Policy Selector

The selector is a supervised portfolio policy that maps online-observable workload features to one deployable scheduling policy.

## What the selector is

The selector does **not** create new heuristics.  It chooses, at the granularity of a request window (W = 200 requests by default), which of the 20 registered deployable policies to use next.  This is a supervised classification problem: label = argmax policy by priority-weighted SLO goodput (internal name: `weighted_goodput`) on that window.

## What the selector is not

- It does not learn or synthesize new scheduling rules.
- It does not call any external APIs.
- `oracle_srtf` is never a label candidate — it is a non-deployable hindsight upper bound only.

## Selector candidate set

All 20 online-deployable baselines registered in `BASELINE_NAMES` (from `llmserveopt/policies/registry.py`) minus any entry in `ORACLE_POLICY_NAMES`.

```
fifo, edf, shortest_output_first, shortest_prompt_first,
greedy_token_fill, least_loaded, multi_bin_batching, random_feasible,
orca_style, vllm_style_token_budget, sarathi_style, splitfuse_style,
slo_slack_score, weighted_shortest_processing, first_fit, best_fit,
least_laxity_first, estimated_service_time_first, admission_control,
scorpio_style_slo_guard
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

## Phase 2B.11 SCORPIO Selector Integration

**Phase 2B.11 finding:** Adding `scorpio_style_slo_guard` (Phase 2B.10) made it the best fixed
baseline (WG=0.993 overall), but the Phase 2B.8 rule selector never dispatched to it (fail_005/006).
The selector gap vs best fixed widened to −0.042 overall.

**Phase 2B.11 repair:** Three new routing rules integrate SCORPIO into the rule selector:

0. **Rule 0 (new):** `(fraction_tight_slo > 0.4 OR min_slack < 1.0) AND recent_slo_violation_rate > 0.2`
   → `scorpio_style_slo_guard`
   *Overloaded tight-SLO with active violations: SCORPIO's admission budget + TTFT/laxity guard
   beats slo_slack_score. The violation rate guard ensures the rule fires only under genuine overload.*

2a. **Rule 2a (new, between existing 1 and 2b):** `pred_output_cv > 2.0`
    → `scorpio_style_slo_guard`
    *Very extreme noise (> 2.0 CV): SCORPIO beats admission_control. Evidence: heldout_very_high_noise_s4
    at 90% noise — AC=0.970, SCORPIO=1.000 (fail_004 resolution).*

3. **Rule 3 (modified):** `recent_slo_violation_rate > 0.3`
   → `scorpio_style_slo_guard` (was `admission_control`)
   *Standalone high violations: SCORPIO's targeted admission budget throttling is more expressive.*

All existing Phase 2B.8 rules (KV pressure → WSP; moderate noise → AC; tight SLO → slo_slack_score;
prefill-heavy → sarathi_style; short uniform → ESTF; bursty → slo_slack_score; default → EDF)
remain unchanged where the new rules do not fire.

Config: `configs/phase2b11_scorpio_selector_integration.yaml`
Runner: `scripts/run_phase2b11_scorpio_selector_integration.py`
Summary: `docs/audits/phase2b11_scorpio_selector_integration_summary.md`

**Phase 2B.11 label diversity finding:** SCORPIO wins as per-window oracle on all 60
Phase 2B.9/2B.10 windows.  Rule selector dispatches to SCORPIO 1/60 times.  RF/DT training
deferred — "always choose SCORPIO" is the only learnable function.  Phase 2B.12 broadens the
suite to expose regimes where other policies win.

## Phase 2B.12 Workload Diversity (Results)

**Goal:** Build ~200-window suite across diverse regimes to enable meaningful RF/DT training
(label diversity criterion: ≥3 policies winning ≥10 windows each, no single policy >85%).

**Config:** `configs/phase2b12_workload_diversity_selector_labels.yaml`  
**Runner:** `scripts/run_phase2b12_workload_diversity_selector_labels.py`  
**Design doc:** `docs/audits/phase2b12_workload_diversity_design.md`  
**tmux session:** `phase2b12_workload_diversity` (completed, EXIT_CODE=0, ~638s)  
**Results:** `results/phase2b12_workload_diversity_selector_labels/` (gitignored)

### Phase 2B.12 Results Summary

| Group | n_windows | Rule selector WG | Best fixed WG | Gap |
|-------|-----------|-----------------|---------------|-----|
| dev | 27 | 0.9168 | 0.9878 | −0.071 |
| heldout | 33 | 0.9803 | 0.9975 | −0.017 |
| regression | 60 | 0.9518 | 0.9932 | −0.041 |
| diversity | 112 | 0.9831 | 0.9969 | −0.014 |
| **overall** | **172** | **0.9721** | **0.9956** | **−0.024** |

Best fixed policy in all groups: `scorpio_style_slo_guard`  
Per-window oracle WG (overall): 0.9974

### Label Distribution (Phase 2B.12 overall, n=172)

| Policy | Oracle wins | Fraction |
|--------|------------|---------|
| `scorpio_style_slo_guard` | 79 | 45.9% ↓ from 100% in Phase 2B.11 |
| `admission_control` | 29 | 16.9% |
| `best_fit` | 14 | 8.1% |
| `edf` | 14 | 8.1% |
| `shortest_output_first` | 13 | 7.6% |
| `estimated_service_time_first` | 10 | 5.8% |
| `multi_bin_batching` | 9 | 5.2% |
| `random_feasible` | 3 | 1.7% |
| `shortest_prompt_first` | 1 | 0.6% |

**RF/DT feasibility:**
- Window count: 172 < 200 → **FAIL** (28 short)
- Policy spread: 6 policies ≥10 wins → **PASS**
- Concentration: top=45.9% < 85% → **PASS**
- **Decision: RF/DT NOT trained** — expand to ≥200 windows first

### Key Phase 2B.12 Findings

1. **Label diversity achieved:** SCORPIO wins 45.9% overall (vs 100% in Phase 2B.11).
   9 distinct policies appear as oracle labels.
2. **Prefill surprise:** `admission_control` wins all prefill-heavy diversity windows;
   `sarathi_style` rule target (Rule 5) is wrong for this WG objective.
3. **Missing rule coverage:** `best_fit`, `multi_bin_batching`, `estimated_service_time_first`,
   `shortest_output_first` each win 9–14 windows but are not in the rule selector's policy choices.
4. **All-complete diversity:** Many diversity workloads have WG=1.0 for all policies; label
   diversity reflects tie-breaking, not genuine differentiation.
5. **Rule selector dispatch:** slo_slack_score=93, WSP=29, AC=28, EDF=19, SCORPIO=2, sarathi=1.

### Phase 2B.12 Failure Cases

| ID | Description | Status |
|----|-------------|--------|
| fail_007 | Rule selector under-dispatches SCORPIO (2/172 vs 79/172 oracle) | Partially deferred |
| fail_008 | Missing rule targets: best_fit, multi_bin_batching, SOF, estST | Open |
| fail_009 | sarathi_style rule target wrong; AC wins prefill-heavy | Open |
| fail_010 | 172 < 200 window threshold; RF/DT training blocked | Open |
| fail_011 | All-complete diversity windows have tie-breaking labels | Open |

See `docs/audits/phase2b12_failure_cases_summary.md` and `docs/audits/phase2b12_selector_label_diversity_summary.md`
for full analysis.

## Phase 2B.13 Selector Training and SCORPIO Suspicion Audit (Results)

**Goal:** Extend to ≥200 windows; train selectors; audit whether SCORPIO dominance and
near-tie labels make selector learning meaningful; report always-SCORPIO baseline.

**Config:** `configs/phase2b13_selector_training_and_suspicion_audit.yaml`  
**Runner:** `scripts/run_phase2b13_selector_training_and_suspicion_audit.py`  
**Summary:** `docs/audits/phase2b13_selector_training_and_suspicion_audit_summary.md`  
**tmux session:** `phase2b13_selector_training`

### Phase 2B.13 Results Summary

| Group | n_windows | Rule WG | RF WG | always-SCORPIO WG | Best fixed WG |
|-------|-----------|---------|-------|-------------------|---------------|
| dev | 27 | 0.9168 | 0.9881 | 0.9878 | 0.9878 |
| heldout | 33 | 0.9803 | **0.9975** | **0.9975** | 0.9975 |
| regression | 60 | 0.9518 | 0.9933 | 0.9932 | 0.9932 |
| diversity | 259 | 0.8179 | 0.9837 | 0.9826 | 0.9826 |
| **overall** | **319** | **0.8431** | **0.9855** | **0.9846** | **0.9846** |

### Selectors Trained

Standard RF/DT, regret-weighted RF/DT, per-policy regression, KNN (k=5), safe-fallback-to-SCORPIO
(margins 0.001/0.005/0.010). All tie or lose to always-SCORPIO on held-out.

### Key Phase 2B.13 Findings

1. **RF ties always-SCORPIO on held-out** (WG=0.9975) but does not beat it; collapses to SCORPIO 32/33 windows.
2. **Near-tie / all-complete labels dominate** (~93% windows); regret-weighting does not change held-out conclusion.
3. **Label diversity sufficient** for training (43.8% SCORPIO, 7 policies ≥10 wins).
4. **always-SCORPIO within 0.2 pp of per-window oracle** on held-out — little selector headroom.
5. **No rule selector repair** — trained selectors evaluated first; RF failure suggests rule patches would be ad hoc.

### Phase 2B.13 Failure Cases

See `docs/audits/phase2b13_failure_cases_summary.md` (fail_012–fail_018).

**Selector claim requirement:** Any selector contribution must beat always-SCORPIO on held-out
or demonstrate statistically meaningful improvement. Near-tie/all-complete windows should not
be treated as strong labels.

## Phase 2B.9 Status and Caveats

**Phase 2B.9 adds the first held-out generalization test for the repaired rule selector.**

Key findings from Phase 2B.9 training audit (`docs/audits/phase2b9_selector_training_audit.md`):
- Phase 2A.4 RF/DT training used only **~30 windows** — insufficient for a 19-class problem.
- KV-pressure and high-noise regimes (the Phase 2B.7 failure cases) are **not in the Phase 2A.4 training set**.
- The Phase 2B.8 repaired rule selector was evaluated on the same 4 workloads that motivated the repair.
  Phase 2B.9 is the first evaluation on truly held-out workloads.
- **Final publication requires ≥200 training windows** covering all regime families.

Phase 2B.9 robustness experiment:
- Config: `configs/phase2b9_selector_robustness.yaml`
- Dev group: 4 workloads (same as Phase 2B.7/2B.8), seeds 0–2
- Heldout group: 5 new workloads (moderate KV, very-high-noise, fixed prefill-overloaded,
  bursty-mixed-SLO, BurstGPT smoke), seeds 3–5
- Results: `results/phase2b9_selector_robustness/` (gitignored); summary in
  `docs/audits/phase2b9_selector_robustness_summary.md`
- **Held-out generalization (rule selector):** WG=0.979 vs best fixed 0.970 (EDF), gap vs oracle −0.005
- **One unresolved failure:** `heldout_very_high_noise_s4` — see `docs/audits/phase2b9_failure_cases_summary.md`
- RF/DT were not re-evaluated in Phase 2B.9 (model artifacts absent); Phase 2A.4 numbers below are historical only

## Phase 2A.4 evaluation results (Phase 2A.4 dataset; see Phase 2B.9 for broader evaluation)

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
