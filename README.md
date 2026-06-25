# llm-serving-heuristic-evolution

**Verifier-Constrained, LLM-Assisted Generation of Scheduling Heuristics for Online LLM Inference Serving**

This project studies verifier-constrained, LLM-assisted generation of scheduling
heuristics for online LLM inference serving. It provides a GPU-calibrated discrete-event
simulator, a suite of 18 baseline policies, real-trace replay against BurstGPT arrival
data, a portfolio selector, and a restricted verifiable DSL for LLM-generated heuristics.

> **Current status:** Phase 2A.4/2B.4 complete. All components finalized: simulator,
> 18 baselines, GPU calibration, real-trace replay, 18-policy selector (Phase 2A.4),
> DSL/verifier stack (Phase 2B.1), offline LLM generation loop (Phase 2B.2),
> controlled LLM heuristic search (Phase 2B.3), and final held-out evaluation with
> bootstrap CIs and shortlist freeze (Phase 2A.4/2B.4). 656 tests pass.
> See [docs/roadmap.md](docs/roadmap.md).

---

## Motivation

LLM inference serving must make online scheduling decisions — which requests to batch,
which GPU to assign them to — with incomplete information (output lengths are unknown at
arrival) and tight latency/SLO constraints. Classical policies (FIFO, EDF, SRPT) provide
interpretable but suboptimal baselines. Serving systems like vLLM, Sarathi-Serve, and
DeepSpeed-FastGen each introduce specialized scheduling insights.

This project asks: can an LLM automatically generate scheduling heuristics, expressed
in a restricted verifiable DSL, that outperform all known baselines on the regimes where
baseline performance diverges?

---

## Problem statement

- Requests arrive online; each has a prompt length, predicted output length, SLO deadline,
  and priority class.
- A pool of GPUs with KV-cache and sequence-count constraints processes them.
- A policy decides: which waiting requests to admit, and to which GPU, at each time step.
- Goal: minimize mean latency and SLO violation rate while maximizing throughput.

See [docs/problem_formulation.md](docs/problem_formulation.md) for the formal definition.

---

## Implemented components

| Component | Status | Notes |
|---|---|---|
| Deterministic iteration-level simulator | Complete | `src/llmserveopt/simulator/` |
| Synthetic workload generators | Complete | Poisson, bursty, heavy-tail, mixed-SLO |
| 18 baseline policies | Complete | See [docs/baselines.md](docs/baselines.md) |
| GPU-calibrated service model | Complete | RTX 5060 Ti, Qwen2.5-0.5B, MAPE <13% |
| Real-trace replay (BurstGPT) | Complete | 7 experiments, Phase 1.7C |
| Prediction-noise sensitivity | Complete | 0%, 35%, 70% noise variants |
| Calibrated vs. synthetic comparison | Complete | Spearman ρ = 1.000 on moderate trace |
| `weighted_goodput` / `priority_weighted_slo_goodput` metric | Complete | Priority-weighted SLO goodput; both names present in metric dicts |
| TTFT reporting | Complete | `mean_ttft`, `p95_ttft` in all summary CSVs |
| oracle_srtf wiring | Complete | Non-deployable hindsight upper bound; separated from online baselines |
| Selector (Phase 2A.4) | Complete | RF/DT beat best fixed +3.0 pp WG on test split (18 policies, 52 windows) |
| 2A.3B hardened baselines (LLF + ESTF) | Complete | 18 policies; `priority_weighted_slo_goodput` alias |
| LLM heuristic DSL + verifier (Phase 2B.1) | Complete | Safe expression tree; 16 error codes; 4 examples |
| LLM generation loop (Phase 2B.2) | Complete | Mock + CloudRift/Cohere/Mistral; verify → repair → evaluate pipeline |
| Controlled LLM heuristic search (Phase 2B.3) | Complete | 7 design targets; multi-regime train/val eval; 22 verified candidates |
| Final held-out evaluation (Phase 2B.4) | Complete | 7-heuristic frozen shortlist; 3 held-out test regimes; bootstrap CIs |

---

## Quick start

```bash
# Install (editable)
pip install -e ".[dev]"

# Smoke test
python scripts/smoke_test.py

# Quick debug run (seconds)
python scripts/run_baseline_comparison.py --config configs/small_debug.yaml

# Full synthetic baseline comparison
python scripts/run_baseline_comparison.py --config configs/baseline_comparison.yaml

# Real-trace replay (requires data/processed/burstgpt/*.jsonl)
python scripts/run_real_trace_comparison.py --config configs/real_trace/burstgpt_scaled_moderate_calibrated.yaml
```

---

## Running tests

```bash
pytest                    # all 656 tests
pytest -m gpu             # GPU-only tests (requires RTX 5060 Ti or equivalent)
```

All 656 tests pass on the current commit.

---

## Synthetic experiment configs

| Config | Description |
|---|---|
| `configs/small_debug.yaml` | Tiny trace, fast; for smoke testing |
| `configs/baseline_comparison.yaml` | Standard Poisson workload |
| `configs/overloaded_comparison.yaml` | High-load regime |
| `configs/prefill_heavy_comparison.yaml` | Long-prompt, prefill-dominated |
| `configs/decode_heavy_comparison.yaml` | Long-output, decode-dominated |
| `configs/mixed_slo_comparison.yaml` | Mixed tight/relaxed SLO tiers |
| `configs/burst_heavy_tail_comparison.yaml` | Heavy-tail arrival bursts |

See [configs/README.md](configs/README.md) for the full list.

---

## Real-trace replay (Phase 1.7C)

BurstGPT traces are replayed against the GPU-calibrated service model. Download
data before running:

```bash
python scripts/download_burstgpt.py  # requires HF_TOKEN in environment
python scripts/convert_burstgpt.py
```

| Config | Description |
|---|---|
| `configs/real_trace/burstgpt_natural_calibrated.yaml` | Natural BurstGPT timing (~318ks span) |
| `configs/real_trace/burstgpt_scaled_moderate_calibrated.yaml` | Moderate-load scaled replay |
| `configs/real_trace/burstgpt_scaled_high_calibrated.yaml` | High-load scaled replay |
| `configs/real_trace/burstgpt_scaled_moderate_synthetic_service.yaml` | Moderate load, synthetic service model |
| `configs/real_trace/burstgpt_moderate_exact_prediction.yaml` | Zero prediction noise |
| `configs/real_trace/burstgpt_moderate_noise035.yaml` | Natural BurstGPT prediction noise |
| `configs/real_trace/burstgpt_moderate_noise070.yaml` | 70% amplified prediction noise |

See [docs/milestones/phase1_7c_calibrated_real_trace.md](docs/milestones/phase1_7c_calibrated_real_trace.md)
for full results.

---

## GPU calibration (Phase 1.7B)

Service curves (prefill and decode timing) were measured on an RTX 5060 Ti running
Qwen/Qwen2.5-0.5B:

```bash
python scripts/run_gpu_calibration.py --config configs/gpu_calibration/calibration_grid.yaml
```

Prefill MAPE: 9%. Decode MAPE: 12%. Curves stored in `results/gpu_calibration/service_curves.json`.
See [docs/gpu_calibration.md](docs/gpu_calibration.md).

---

## Baseline policies

18 policies are registered for online use. See [docs/baselines.md](docs/baselines.md) for
provenance, safe/unsafe labels, and the full table.

| Label | Policy | Inspired by |
|---|---|---|
| `fifo` | FIFO | Classical |
| `edf` | Earliest Deadline First | Classical real-time scheduling |
| `shortest_output_first` | SRPT-style (predicted length) | Classical |
| `shortest_prompt_first` | Shortest KV footprint first | Original |
| `greedy_token_fill` | Best-fit KV packing | Original |
| `least_loaded` | Least-busy GPU dispatch | Load balancing |
| `multi_bin_batching` | Multi-Bin-style grouping | Independent adaptation |
| `random_feasible` | Random feasible admission | Stochastic baseline |
| `first_fit` | First-fit KV bin packing | Classical bin packing |
| `best_fit` | Best-fit KV bin packing (tightest fit) | Classical bin packing |
| `orca_style` | Orca-style iteration-level scheduler | Yu et al., OSDI 2022 |
| `vllm_style_token_budget` | vLLM-inspired token-budget / paged-KV proxy | Kwon et al., SOSP 2023 |
| `sarathi_style` | Sarathi-style chunked-prefill | Agrawal et al., arXiv 2023 / OSDI 2024 |
| `splitfuse_style` | Dynamic-SplitFuse-style chunked-prefill | Holmes et al., arXiv 2024 |
| `slo_slack_score` | SLO-slack composite scoring | Original |
| `weighted_shortest_processing` | WSPT composite | Original |
| `least_laxity_first` | LLF: deadline − now − est. service time | Classical real-time scheduling |
| `estimated_service_time_first` | Prompt-and-prediction-aware SJF proxy | PARS-inspired (not a PARS reproduction) |

All serving-style baselines are **original implementations** capturing the key
scheduling insight of each cited system. None reproduce the original system's code.

**Note:** `oracle_srtf` (hindsight SRTF) is in `ORACLE_POLICY_NAMES` — a non-deployable
upper-bound. It is NOT in `BASELINE_NAMES` or `SELECTOR_CANDIDATE_NAMES`. Use
`make_oracle_policy("oracle_srtf", requests)` to access it explicitly. Label all
oracle_srtf results as "hindsight upper bound" in reports.

---

## Repository structure

```
src/llmserveopt/
  core/             # Types, actions, metrics
  simulator/        # Deterministic step simulator + service models
  workloads/        # Synthetic generators, BurstGPT/ShareGPT loaders, trace I/O
  policies/         # 18 registered baselines + oracle + helpers
  evaluation/       # Run, compare, aggregate policies
  heuristics/       # JSON DSL, verifier, compiler, HeuristicPolicy wrapper
  llm_generation/   # Offline LLM generation loop (providers, repair, archive, ranking)
  plotting/         # Tables and figures
  utils/            # Seeding, JSONL helpers
scripts/         # CLI entry points (see scripts/README.md)
configs/         # YAML experiment configs (see configs/README.md)
docs/            # Design docs, milestones, claims, roadmap (see docs/README.md)
tests/           # pytest suite (656 tests)
data/            # Local datasets — not committed (see data/README.md)
results/         # Experiment outputs — not committed (see results/.gitkeep)
```

---

## Data policy

- `data/raw/` and `data/processed/` are gitignored.
- Results in `results/` are gitignored.
- See `data/README.md` and `.env.example` for download and credential instructions.
- **Never commit API keys, model weights, or raw dataset files.**

---

## Reproducibility

All randomness is seeded via `numpy.random.default_rng`. Same config + same seed =
identical results. The GPU calibration step requires a physical GPU; calibrated curves
are stored in `results/gpu_calibration/service_curves.json` (local only).

---

## Safe claims and limitations

- "We replay real BurstGPT arrival timestamps and token counts."
- "SLOs, priorities, and predicted output lengths are synthetically augmented."
- "Service curves are calibrated on an RTX 5060 Ti running Qwen2.5-0.5B."
- "Serving-style baselines are original implementations inspired by, not reproductions of, the cited systems."
- "We evaluate LLM-generated deterministic heuristics under a calibrated simulator and held-out workload regimes."

Do not claim: production vLLM reproduction, exact production latency, generalization
beyond the RTX 5060 Ti + Qwen2.5-0.5B calibration point, or that LLM heuristics
conclusively outperform all baselines (the CI is wide; only 1/7 heuristics clearly wins).

**Selective-admission caveat (headline result):** The best LLM heuristic
(`slo_kv_balance_heuristic`, mean WG=0.9595, 95% CI [0.00, 0.27]) achieved its score
on the `test_very_overloaded` regime by completing only **1240 of 2119 requests (58%)**
— i.e., via selective request dropping, not throughput improvement. All other heuristics
and baselines completed all 2119 requests on that regime. Do not present this result as
a full-admission win. See [docs/result_claims.md](docs/result_claims.md) §"What the final
evaluation numbers mean" for the exact per-regime `num_completed` breakdown and safe
claim language.

See [docs/result_claims.md](docs/result_claims.md) and [docs/gpu_validation_claims.md](docs/gpu_validation_claims.md).

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Simulator + classical baselines | Complete |
| 1.5 | Serving-style baselines (Orca/vLLM/Sarathi/SplitFuse) | Complete |
| 1.7A | BurstGPT + ShareGPT trace ingestion | Complete |
| 1.7B | GPU calibration (RTX 5060 Ti, Qwen2.5-0.5B) | Complete |
| 1.7C | Calibrated real-trace replay (7 experiments) | Complete |
| 2A.1 | Metric finalization + oracle wiring | Complete |
| 2A.2–2A.3 | Selector dataset + training + evaluation | Complete |
| 2A.3B | Hardened baselines (LLF, ESTF) + priority_weighted alias | Complete |
| 2B.1 | LLM heuristic DSL + verifier + policy wrapper | Complete |
| 2B.2 | LLM offline heuristic generation loop | Complete |
| 2B.3 | Controlled LLM heuristic search (multi-regime eval) | Complete |
| 2A.4/2B.4 | Final evaluation hardening (shortlist freeze, held-out test, bootstrap CIs) | Complete |
| 4 | LLM evolution loop (full) | Not started |
| 5 | Shifted-workload evaluation + paper write-up | Not started |

See [docs/roadmap.md](docs/roadmap.md) for details.

---

## License

MIT — see [LICENSE](LICENSE).
