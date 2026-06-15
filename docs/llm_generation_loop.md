# LLM Heuristic Generation Loop (Phase 2B.2)

All LLM API calls happen **offline only**. No LLM is called at runtime during request scheduling.
Generated heuristics are compiled to a deterministic JSON expression tree evaluated without `eval`.

---

## Overview

```
Prompt template
    ↓
LLM provider (offline)
    ↓
extract_json()
    ↓
verify_heuristic() — JSON DSL verifier
    ↓  (if invalid)
repair loop (up to max_repair_attempts)
    ↓
save to candidate archive
    ↓
evaluate_candidates() — simulator fitness oracle
    ↓
rank_candidates() — sort by priority_weighted_slo_goodput
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
  __init__.py            — public API
  provider_base.py       — LLMResponse dataclass, LLMProvider protocol
  providers.py           — CloudRiftProvider, CohereProvider, MistralProvider, MockProvider
  prompt_templates.py    — build_generation_messages(), build_repair_messages()
  candidate_io.py        — CandidateRecord, save_candidate(), load_verified_candidates()
  repair.py              — extract_json(), run_repair_loop()
  generation_loop.py     — GenerationConfig, run_generation_loop()
  evaluation.py          — EvaluationConfig, evaluate_candidates()
  ranking.py             — rank_candidates(), save_ranking_csv(), build_summary_md()
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

## Scripts

### Generate candidates

```bash
# Dry-run (mock provider, no API calls)
python scripts/generate_llm_heuristics.py \
    --providers mock \
    --models mock \
    --max-candidates 4 \
    --max-repair-attempts 2 \
    --dry-run \
    --output-dir results/phase2b2_llm_generation/mock_candidates

# Real API (CloudRift, thinking model — needs large token budget)
python scripts/generate_llm_heuristics.py \
    --providers cloudrift \
    --models auto \
    --max-candidates 2 \
    --max-repair-attempts 2 \
    --temperature 0.4 \
    --max-tokens 8000 \
    --output-dir results/phase2b2_llm_generation/real_api_candidates
```

### Evaluate candidates

```bash
python scripts/evaluate_generated_heuristics.py \
    --candidates-dir results/phase2b2_llm_generation/mock_candidates \
    --output-dir     results/phase2b2_llm_generation/mock_evaluation
```

Outputs:
- `ranking.csv` — all heuristics + baselines ranked by `priority_weighted_slo_goodput`
- `candidate_metrics.csv` — heuristics only
- `baseline_metrics.csv` — baselines only
- `evaluation_summary.md` — markdown table + generation stats

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

## Notes for final paper

- RF Selector from Phase 2A.3 was trained on 16-policy candidate set. Rerun with 18 policies
  required before final paper evaluation.
- `oracle_srtf` is a hindsight upper bound only — never included in deployable comparisons.
- `estimated_service_time_first` is a PARS-inspired proxy, not a PARS reproduction.
  PARS uses learning-to-rank; this policy uses token arithmetic only.
