# llm-serving-heuristic-evolution

**Verifiable LLM-Evolved Dispatching and Dynamic Batching Heuristics for Online LLM Serving**

> **Phase 1 only.**  This repository currently implements the simulator, synthetic
> workloads, classical baseline policies, and baseline comparison infrastructure.
> The LLM-evolved heuristic method is **not yet implemented**.
> See [docs/roadmap.md](docs/roadmap.md) for the full research plan.

---

## Motivation

LLM inference serving must make online decisions about which requests to batch
together and which GPU to send them to — with limited information (output lengths
are unknown at arrival) and tight latency/SLO constraints.

This project builds a research infrastructure to:
1. Evaluate classical dispatching and batching heuristics in a controlled simulator.
2. (Phase 2+) Use LLMs to generate and evolve novel policy code within a verified sandbox.
3. Benchmark evolved policies against classical baselines under realistic workloads.

---

## Problem statement

- Requests arrive online; each has a prompt length, predicted output length, SLO deadline, and priority class.
- A pool of GPUs with KV-cache and sequence-count constraints processes them.
- A policy decides: which waiting requests to admit, and to which GPU.
- Goal: minimize latency and SLO violations while maximizing throughput.

See [docs/problem_formulation.md](docs/problem_formulation.md) for the mathematical formulation.

---

## Quick start

```bash
# Install (editable)
pip install -e ".[dev]"

# Smoke test
python scripts/smoke_test.py

# Run full baseline comparison
python scripts/run_baseline_comparison.py --config configs/baseline_comparison.yaml

# Quick debug run (seconds)
python scripts/run_baseline_comparison.py --config configs/small_debug.yaml

# Generate and save synthetic traces
python scripts/generate_synthetic_traces.py --out-dir traces/ --seeds 0 1 2

# Run tests
pytest
```

---

## Baseline policies

| Name | Description | Type |
|---|---|---|
| `fifo` | Oldest request first | Classical |
| `edf` | Earliest Deadline First | Classical |
| `shortest_output_first` | Shortest predicted output first (SRPT-style) | Classical |
| `shortest_prompt_first` | Shortest prompt tokens first | Heuristic |
| `greedy_token_fill` | Best-fit KV packing | Heuristic |
| `least_loaded` | Assign to least-busy GPU | Load balancing |
| `multi_bin_batching` | Group by output-length bins (Multi-Bin-style) | Heuristic |
| `random_feasible` | Random admission, deterministic under seed | Baseline |
| `oracle_srtf` | Hindsight SRTF (non-deployable) | Oracle |

See [docs/baselines.md](docs/baselines.md) for policy descriptions and provenance.

---

## Reproducibility

The full baseline comparison can be reproduced exactly:

```bash
python scripts/run_baseline_comparison.py --config configs/baseline_comparison.yaml
```

All randomness is seeded via `numpy.random.default_rng`.  Same config + same seeds =
identical results.

---

## Example result table

Running `configs/small_debug.yaml` (single seed, small trace):

| Policy | Mean Lat (s) | P95 Lat (s) | SLO Viol. Rate | Req/s | GPU Util. |
|---|---|---|---|---|---|
| shortest_output_first | ~0.013 | ~0.020 | 0.00 | ~2.0 | ~0.3 |
| fifo | ~0.013 | ~0.022 | 0.00 | ~2.0 | ~0.3 |
| edf | ~0.014 | ~0.021 | 0.00 | ~2.0 | ~0.3 |
| random_feasible | ~0.015 | ~0.024 | 0.00 | ~2.0 | ~0.3 |

*(Approximate values; run the experiment for exact numbers.)*

---

## Result interpretation

- All numbers are from a **deterministic Phase 1 simulator** (see [docs/simulator_design.md](docs/simulator_design.md)).
- Do not compare directly to production vLLM or real-hardware benchmarks without additional validation.
- The oracle policy uses future information and is not deployable.
- Multi-Bin-style batching is an independent approximate implementation.
- See [docs/result_claims.md](docs/result_claims.md) for a full list of safe and unsafe claims.

---

## Repository structure

```
src/llmserveopt/
  core/          # Types, actions, metrics
  simulator/     # Deterministic iteration-level simulator
  workloads/     # Synthetic workload generators and trace I/O
  policies/      # All baseline policies + oracle
  evaluation/    # Run, compare, aggregate
  plotting/      # Tables and figures
  utils/         # Seeding, JSONL helpers
scripts/         # CLI entry points
configs/         # YAML experiment configs
docs/            # Problem formulation, design, claims, roadmap
tests/           # pytest test suite
results/         # Experiment outputs (gitignored except .gitkeep)
external/        # Placeholder for future external code with provenance notes
```

---

## Roadmap

- **Phase 1** (now): Simulator + baselines ✓
- **Phase 2**: LLM-generated restricted policy code
- **Phase 3**: Verifier and sandboxed executor
- **Phase 4**: LLM-evolution loop (Cohere, CloudRift)
- **Phase 5**: Shifted workload evaluation and paper write-up

See [docs/roadmap.md](docs/roadmap.md) for details.

---

## License

MIT — see [LICENSE](LICENSE).
