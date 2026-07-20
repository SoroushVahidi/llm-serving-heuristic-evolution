# LLM Heuristic Generation Loop (Phase 2B.2 + 2B.3)

All LLM API calls happen **offline only**. No LLM is called at runtime during request scheduling.
Generated heuristics are compiled to a deterministic JSON expression tree evaluated without `eval`.

---

## Overview

```
design_target (one of 7) + temperature
    ↓
build_targeted_messages()
    ↓
LLM provider (offline)
    ↓
extract_json()
    ↓
verify_heuristic() — JSON DSL verifier
    ↓  (if invalid)
repair loop (up to max_repair_attempts)
    ↓
deduplicate_candidates() — remove exact SHA256 duplicates
    ↓
save to candidate archive
    ↓
evaluate_multi_regime() — evaluate across 4 train + 3 validation regimes
    ↓
aggregate_regime_results() — per-candidate train/val aggregation
    ↓
rank_search_results() — rank by val priority_weighted_slo_goodput
```

---

## Fitness oracle

The simulator objective `priority_weighted_slo_goodput` is the fitness signal:

```
Σ(priority_i × 1[completion_time_i ≤ deadline_i]) / Σ(priority_i)
```

This is also exposed as `weighted_goodput` (internal field name). The selector (RF portfolio)
is NOT the fitness oracle — it is an adaptive deployable baseline.

`oracle_srtf` is excluded from all deployable baseline comparisons.

---

## Module layout

```
src/llmserveopt/llm_generation/
  __init__.py                  — public API
  provider_base.py             — LLMResponse dataclass, LLMProvider protocol
  providers.py                 — CloudRiftProvider, CohereProvider, MistralProvider, MockProvider
  prompt_templates.py          — build_generation_messages(), build_repair_messages()
  diversity.py                 — DESIGN_TARGETS, build_targeted_messages(), deduplicate_candidates()
  candidate_io.py              — CandidateRecord, save_candidate(), load_verified_candidates()
  repair.py                    — extract_json(), run_repair_loop()
  generation_loop.py           — GenerationConfig, run_generation_loop()
  evaluation.py                — EvaluationConfig, evaluate_candidates() (single regime)
  ranking.py                   — rank_candidates(), save_ranking_csv() (single regime)
  multi_regime_evaluation.py   — evaluate_multi_regime(), aggregate_regime_results()
  search_ranking.py            — rank_search_results(), save_search_ranking_csv()
```

---

## Providers

Priority order: CloudRift → Cohere → Mistral → Mock (dry-run only).

| Provider | Env vars | Default model |
|---|---|---|
| CloudRift | `CLOUDRIFT_API_KEY`, `CLOUDRIFT_BASE_URL` | `Qwen/Qwen3.6-35B-A3B-FP8` |
| Cohere | `COHERE_API_KEY` | `command-r-plus-08-2024` |
| Mistral | `MISTRAL_API_KEY` | `mistral-large-latest` |
| Mock | — | `mock-v1` |

**CloudRift note**: The current available model (`Qwen/Qwen3.6-35B-A3B-FP8`) is a thinking model.
It requires `max_tokens ≥ 8000` to allow the thinking phase to complete before outputting JSON.
Use `--max-tokens 8000` or higher when calling real CloudRift endpoints.

---

## Candidate archive format

Each candidate gets its own timestamped directory under `output_dir/`:

```
output_dir/
  index.csv
  20260615_104404_cloudrift_Qwen-Qwen3.6-35B-A3B-FP8_c001/
    prompt.json             — messages sent to the LLM
    raw_response.txt        — full text returned by the LLM
    candidate.json          — extracted + verified JSON heuristic (if valid)
    verifier_result.json    — {valid, errors, warnings}
    metadata.json           — provider, model, timing, sha256, git_commit
    repaired_attempts/      — per-attempt repair attempts (if any)
      attempt_1_raw.txt
      attempt_1_candidate.json
```

`metadata.json` never contains API keys, passwords, or secrets.

---

## Design Targets (Phase 2B.3)

Seven named design emphases steer prompt diversity. The generation loop cycles
through targets and temperatures to maximize candidate diversity:

| Target | Emphasis |
|---|---|
| `slo_urgency` | Minimize SLO violations; prioritize tight-deadline requests |
| `kv_pressure` | KV-cache efficiency; penalize large KV footprints under pressure |
| `throughput_oriented` | Maximize request throughput; shortest-first |
| `prefill_heavy` | Avoid prefill-dominated head-of-line blocking |
| `mixed_slo` | Handle tight/medium/loose SLO tiers with priority weighting |
| `noisy_prediction_robust` | Robust to 35% output-length prediction noise |
| `balanced` | Multi-objective balance with high-load regime switching |

---

## Multi-Regime Evaluation (Phase 2B.3)

Candidates are evaluated on 4 train + 3 validation synthetic regimes.
Test regimes are held out (not used for candidate selection).

| Regime | Split | Description |
|---|---|---|
| `train_poisson_moderate` | train | Standard Poisson, rate=15/s |
| `train_bursty_moderate` | train | Bursty Poisson, rate=15/s |
| `train_overloaded` | train | Poisson rate=25/s, tight SLOs |
| `train_mixed_slo` | train | Mixed SLO tiers |
| `val_prefill_heavy` | validation | Long prompts (512 avg), short outputs |
| `val_decode_heavy` | validation | Short prompts, long outputs (384 avg) |
| `val_noisy_predictions` | validation | 35% output-length noise |

Ranking is by **validation** `priority_weighted_slo_goodput`, not training.

---

## Scripts

### Generate candidates

```bash
# Dry-run (mock provider, no API calls)
python scripts/generate_llm_heuristics.py \
    --providers mock --models mock \
    --max-candidates 4 --max-repair-attempts 2 \
    --dry-run \
    --output-dir results/phase2b2_llm_generation/mock_candidates

# Phase 2B.3 controlled search (CloudRift, thinking model)
python scripts/generate_llm_heuristics.py \
    --providers cloudrift \
    --models Qwen/Qwen3.6-35B-A3B-FP8 \
    --max-candidates 30 \
    --max-repair-attempts 3 \
    --temperatures 0.3,0.7,1.0 \
    --design-targets all \
    --max-tokens 9000 \
    --output-dir results/phase2b3_llm_search/candidates_main
```

### Evaluate single regime

```bash
python scripts/evaluate_generated_heuristics.py \
    --candidates-dir results/phase2b2_llm_generation/mock_candidates \
    --output-dir     results/phase2b2_llm_generation/mock_evaluation
```

### Evaluate multi-regime (Phase 2B.3)

```bash
python scripts/evaluate_multi_regime.py \
    --candidates-dir results/phase2b3_llm_search/candidates_main \
    --output-dir     results/phase2b3_llm_search/evaluation_train_validation
```

Multi-regime outputs:
- `ranking_overall.csv` — all candidates + baselines, ranked by val WG
- `ranking_heuristics.csv` — heuristics only
- `ranking_baselines.csv` — baselines only
- `candidate_metrics_by_regime.csv` — per-regime WG for each candidate
- `candidate_metrics_by_regime_flat.csv` — flat CSV with all per-regime results
- `evaluation_summary.md` — markdown summary with overfitting analysis
- `top_candidates/` — top 5 candidate JSON + markdown summary per candidate

---

## Adding a new provider

1. Implement `name`, `is_available()`, and `generate()` (matching `LLMProvider` protocol).
2. Add to `_PROVIDER_CLASSES` dict in `providers.py`.
3. Add priority-order docs above.
4. Test with a unit test in `tests/test_llm_provider_interface.py`.

---

## Constraints (enforced)

- No `eval`, `exec`, or `import` at runtime in the DSL evaluator.
- No `actual_output_tokens`, `future_*`, `oracle_*`, `ground_truth_*`, `hidden_*`,
  or `completion_time` in any allowed variable.
- All expressions limited by: max depth=6, max nodes=64, max terms=16.
- Runtime scheduling is fully deterministic — no LLM called during simulation.
- API keys never printed, logged, or committed.

---

## Phase 2B.4 — Final Evaluation (shortlist freeze + held-out test)

Phase 2B.4 completes the evaluation with strict train/validation/test discipline.

### Shortlist freeze procedure

1. Re-evaluate all 22 Phase 2B.3 candidates on train+val regimes with all 18 baselines.
2. Select frozen shortlist using train+validation only (test NOT inspected before freeze).
3. Freeze 7 candidates: top-5 by val WG + 1 robustness pick + 1 simplicity pick.
4. Evaluate frozen shortlist once on 3 held-out test regimes with oracle.

### Held-out test regimes

| Regime | Rate | Burst | Noise | SLO |
|--------|------|-------|-------|-----|
| test_very_overloaded | 35/s | — | 0.30 | tight |
| test_extreme_bursty | 25/s | 8.0× | 0.25 | mixed |
| test_high_noise | 15/s | — | 0.50 | mixed |

### Final test results

Best heuristic: `slo_kv_balance_heuristic`, mean WG=0.9595 across 3 test regimes.
Best fixed baseline: `weighted_shortest_processing`, mean WG=0.8602.
oracle_srtf: WG=0.8550 (non-deployable; not optimal for priority-weighted SLO goodput).

**Safe interpretation**: `slo_kv_balance_heuristic` shows a suggestive +9.9 pp improvement over best fixed on these regimes (95% CI [0.00, 0.27]). The result is exploratory: the CI is wide (3 regimes only) and the improvement is partly driven by selective request handling under extreme overload. 6/7 shortlisted heuristics regress vs best fixed on held-out test.

### Scripts

```bash
# Shortlist selection (train+val only)
python scripts/evaluate_multi_regime.py \
    --split train_validation --all-baselines --candidates results/phase2b3_llm_search/candidates_main

# Final held-out test (after freeze)
python scripts/evaluate_multi_regime.py \
    --split test --all-baselines --include-oracle --candidates results/phase2a4_2b4_final_eval/frozen_shortlist

# Bootstrap CI summary
python scripts/summarize_final_evaluation.py \
    --eval-dir results/phase2a4_2b4_final_eval/final_heldout_eval \
    --selector-eval-dir results/phase2a4_2b4_final_eval/selector_evaluation \
    --output-dir results/phase2a4_2b4_final_eval/final_summary

# Plots
python scripts/plot_final_evaluation.py \
    --summary-dir results/phase2a4_2b4_final_eval/final_summary \
    --output-dir results/phase2a4_2b4_final_eval/plots
```

## Notes for final paper

- RF/DT Selectors: +3.0% over best fixed on 18-policy test split (Phase 2A.4). Supersedes Phase 2A.3.
- `oracle_srtf` is a hindsight upper bound only — never included in deployable comparisons.
- `estimated_service_time_first` is a PARS-inspired proxy, not a PARS reproduction.
  PARS uses learning-to-rank; this policy uses token arithmetic only.
- Safe wording: "We evaluate LLM-generated deterministic heuristics under a calibrated simulator and held-out workload regimes."
- Do not claim: "The LLM scheduler beats production vLLM."
