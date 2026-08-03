# llm-serving-heuristic-evolution

**Learning to select LLM-inference-serving scheduling policies, evaluated against
faithful external baselines in a GPU-calibrated discrete-event simulator.**

> ## Contextual composition branch (active, not paused)
> For the new contextual-compositional heuristic research path, use branch
> **`contextual-compositional-heuristics-20260731`** and start with
> **[docs/START_HERE_CONTEXTUAL_COMPOSITION.md](docs/START_HERE_CONTEXTUAL_COMPOSITION.md)**.
> Its authoritative technical roadmap is
> **[docs/contextual_composition_roadmap.md](docs/contextual_composition_roadmap.md)**.
> CC1-CC4 are complete; CC5 (contextual composition predictor) is
> `IN PROGRESS` -- see the
> **[latest CC5 report](docs/audits/contextual_composition_cc5_predictor_report_20260803.md)**
> and, for current status, the
> **[CC4b/CC5 retry report](docs/audits/contextual_composition_cc4b_cc5_retry_report_20260803.md)**.
> Resume from **[docs/RESUME_CONTEXTUAL_COMPOSITION.md](docs/RESUME_CONTEXTUAL_COMPOSITION.md)**;
> active issue: [#5](https://github.com/SoroushVahidi/llm-serving-heuristic-evolution/issues/5).
> This scoped roadmap does not replace the historical project-status documents
> for unrelated branches or earlier phases.

> ## ⏸ Project paused as of 2026-07-23 — resume here first
> **[docs/current/RESUME_HERE.md](docs/current/RESUME_HERE.md)** is the
> single entry point for resuming this project after the pause. It
> supersedes everything below this notice, including the policy/baseline
> counts and "current blocker" description in this file, which describe an
> **older, superseded** project phase (pre-Policy-Library-v2,
> pre-`slai_faithful`). Read `RESUME_HERE.md` before anything else.

> **New here? Start with [docs/current/README.md](docs/current/README.md).**
> That's the canonical, current-state documentation set. Everything below is a
> quick orientation; `docs/current/PROJECT_STATUS.md` is the authoritative
> source for exactly where this project stands right now.

---

## A. What this project does

This project studies **dynamic scheduling-policy selection** for online LLM
inference serving: requests arrive with unknown output length under tight
SLO constraints, and a policy must decide who to admit and in what order.
Concretely, the project:

1. Implements a GPU-calibrated discrete-event simulator and a 20-policy
   internal scheduling portfolio (classical, packing, composite, and
   literature-inspired "style" policies).
2. Implements 6 **faithful** external-system reimplementations (vLLM,
   vLLM-chunked-prefill, Sarathi-Serve, DistServe, TetriInfer, Llumnix), each
   pinned to an exact upstream commit and validated against real GPU
   hardware.
3. Trains a **selector** -- currently a supervised model -- that picks the
   best internal policy per workload window, and compares it against the
   strongest fixed internal policy and against the faithful external
   baselines (which are evaluation-only references, never selector
   actions).
4. Separately explores LLM-generated scheduling heuristics under a
   restricted, verifiable DSL (a secondary, currently dormant research
   track relative to the selector work).

See [docs/current/PROJECT_STATUS.md](docs/current/PROJECT_STATUS.md) for
what is scientifically complete today and what isn't yet.

## B. Current scientific architecture

- **Simulator**: deterministic, iteration-level, GPU-calibrated (RTX 5060 Ti
  / Qwen2.5-0.5B). [docs/current/ARCHITECTURE.md](docs/current/ARCHITECTURE.md)
- **Historical/internal policy portfolio**: 20 policies.
- **Selector v2 trainable action space**: exactly **8** of those 20 policies
  (`fifo`, `edf`, `scorpio_style_slo_guard`, `admission_control`,
  `weighted_shortest_processing`, `estimated_service_time_first`,
  `best_fit`, `multi_bin_batching`) -- an evidence-based scope decision
  ("Option B"), not an arbitrary subset.
- **Faithful external baselines are evaluation-only**: all 6 (3 monolithic,
  2 disaggregated, 1 migratory) are confirmed genuinely dominated under the
  current objective when tested as selector *actions*, but remain valuable,
  topology-aware, real-hardware-validated *comparison points*.
- Full inventory, exact names, and the "why 8" evidence:
  [docs/current/BASELINES.md](docs/current/BASELINES.md)

## C. Current objective

The primary objective is **`arrival_normalized_weighted_goodput`** (ANWG) --
weighted SLO goodput normalized by *all arriving requests*, not just
completed ones. An earlier metric (`weighted_goodput`, denominator =
completed requests only) was found to be biased toward policies that reject
or drop more work; that field is retained (renamed in interpretation, not
deleted) as a distinct "conditional quality of completions" metric, and ANWG
is the corrected primary objective for all current selector work. See
[docs/selector_objective_audit.md](docs/selector_objective_audit.md).

## D. Current status

- Internal policy portfolio and 6 faithful external baselines: **complete**,
  pinned, real-GPU-hardware-validated (local RTX 5060 Ti + Wulver A100).
- Selector Dataset v2 infrastructure (SLO calibration, leakage-safe splits,
  automated quality gates): **complete and in active use**.
- The most recent calibrated targeted pilot (250 windows, Option B scope)
  passed all of the pipeline's own automated quality gates, but an
  independent audit **confirmed a real leakage bug** in its non-OOD split
  construction (cross-transform row-range reuse -- the automated gate
  doesn't check for it). VALIDATION/ID_TEST are not trustworthy held-out
  splits as a result; on OOD_TEST, the one confirmed-clean split, the
  trained prototype selector loses to the best fixed policy. **Do not treat
  Selector v2 as a finished, working result.**
- **Current blocker**: fix the split-construction bug and regenerate a clean
  pilot. Until a clean pilot exists, comparing the selector against the
  faithful external baselines (Protocol C) is premature.

Full detail: [docs/current/SELECTOR_V2.md](docs/current/SELECTOR_V2.md).

## E. Quick start

```bash
# Install (editable) -- use python3 -m pip / python3 -m pytest throughout;
# the bare `pytest` on PATH may resolve to an interpreter missing pandas.
pip install -e ".[dev]"
python3 -c "import pandas"   # sanity check before trusting bare `pytest`

# Smoke test
python scripts/smoke_test.py

# Quick debug run (seconds)
python scripts/run_baseline_comparison.py --config configs/small_debug.yaml

# Full synthetic baseline comparison
python scripts/run_baseline_comparison.py --config configs/baseline_comparison.yaml

# Real-trace replay (requires data/processed/burstgpt/*.jsonl)
python scripts/run_real_trace_comparison.py --config configs/real_trace/burstgpt_scaled_moderate_calibrated.yaml
```

### Tests

```bash
python3 -m pytest                    # full non-GPU-safe suite
python3 -m pytest -m gpu             # GPU-only (requires a CUDA-capable GPU)
python3 -m pytest --collect-only -q  # count/list without running
```

Exact current test count and environment caveats:
[docs/current/PROJECT_STATUS.md](docs/current/PROJECT_STATUS.md) and
[docs/current/REPRODUCIBILITY.md](docs/current/REPRODUCIBILITY.md).

## F. Canonical documentation

Start with **[docs/current/README.md](docs/current/README.md)**, which
indexes:

- [PROJECT_STATUS.md](docs/current/PROJECT_STATUS.md) -- authoritative current state
- [ARCHITECTURE.md](docs/current/ARCHITECTURE.md) -- code architecture
- [BASELINES.md](docs/current/BASELINES.md) -- exact policy/baseline inventory
- [SELECTOR_V2.md](docs/current/SELECTOR_V2.md) -- full selector research narrative
- [EXPERIMENTS_AND_RESULTS.md](docs/current/EXPERIMENTS_AND_RESULTS.md) -- what's committed vs. local-only
- [REPRODUCIBILITY.md](docs/current/REPRODUCIBILITY.md) -- environment, tests, GPU workflows
- [NEXT_STEPS.md](docs/current/NEXT_STEPS.md) -- the exact next recommended action

The full legacy documentation index (~75 detailed design/audit documents) is
[docs/README.md](docs/README.md).

## G. Repository layout

```
src/llmserveopt/   # library code -- see docs/current/ARCHITECTURE.md for the module map
scripts/           # CLI entry points -- scripts/README.md covers Phase-1.7C-and-earlier
                   # scripts only; see docs/README.md §16A for the current Selector v2 scripts
configs/           # YAML experiment configs -- see configs/README.md
docs/              # docs/current/ = canonical current docs; docs/README.md = full legacy index
tests/             # pytest suite
data/              # local datasets -- gitignored (raw/processed), see data/README.md
results/           # experiment outputs -- gitignored except results/.gitkeep; local-only, large
logs/              # run logs -- gitignored entirely; local-only
experiments/       # small, curated experiment artifacts -- NOT gitignored, the only one of
                   # results/logs/experiments that is actually version-controlled
```

**`results/` and `logs/` are local-only.** A fresh clone will not have them.
`experiments/` is the committed, cloneable record of experiment outputs. See
[docs/current/EXPERIMENTS_AND_RESULTS.md](docs/current/EXPERIMENTS_AND_RESULTS.md)
for what's canonical, what's historical, and what's still local-only pending
a decision (e.g. the most recent calibrated pilot).

## H. Current next step

Fix the confirmed split-construction leakage bug and regenerate a clean
calibrated pilot, then proceed through the sequence in
[docs/current/NEXT_STEPS.md](docs/current/NEXT_STEPS.md) -- do not scale
Dataset v2 generation or claim selector superiority before clean, confirmed
held-out generalization exists.

---

## Historical: Phase 1-2B.4 headline result (kept for continuity, not current)

Prior to the Selector v2 / external-baseline program described above, this
project's furthest evaluated result (Phase 2A.4/2B.4, "Final held-out
evaluation") was:

- **Selector (RF/DT)**: +3.0 pp over best fixed on the then-current
  18-policy portfolio (52 windows).
- **Best LLM-generated heuristic** (`slo_kv_balance_heuristic`): mean
  WG=0.9595 on final held-out test regimes; 95% CI [0.00, 0.27] --
  achieved via **selective request dropping** (completed only 58% of
  requests on the hardest regime), not a full-admission throughput win.
  Do not present this as a full-admission result.

Safe claims for this historical result, and the exact per-regime breakdown:
[docs/result_claims.md](docs/result_claims.md),
[docs/gpu_validation_claims.md](docs/gpu_validation_claims.md).

For the full phase-by-phase roadmap (historical, superseded as the "current"
narrative by the unnumbered Selector v2 track): [docs/roadmap.md](docs/roadmap.md).

---

## Data policy

- `data/raw/` and `data/processed/` are gitignored.
- Results in `results/` are gitignored.
- See `data/README.md` and `.env.example` for download and credential
  instructions. **Never commit API keys, model weights, or raw dataset
  files.**

All randomness is seeded via `numpy.random.default_rng`; same config + same
seed = identical results.

## License

MIT -- see [LICENSE](LICENSE).
