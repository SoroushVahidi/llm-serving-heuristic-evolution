# Baseline Comparison Experiment

This directory stores results from the Phase 1 baseline comparison.

## How to run

```bash
# Full comparison (all workloads, 5 seeds)
python scripts/run_baseline_comparison.py --config configs/baseline_comparison.yaml

# Quick debug run (single workload, 1 seed)
python scripts/run_baseline_comparison.py --config configs/small_debug.yaml
```

## Output structure

```
results/baseline_comparison/<timestamp>/
  README.md               ← auto-generated run summary
  per_run.csv             ← one row per (policy, seed, workload)
  per_run.jsonl
  summary.csv             ← mean across seeds per policy
  summary.json
  medium_poisson/         ← per-workload results
    per_run.csv
    summary.csv
    figures/
      mean_latency.png
      slo_violation_rate.png
      ...
  bursty/
    ...
  heavy_tail/
    ...
```

## Interpreting results

- All numbers are from the **deterministic Phase 1 simulator**.
- See `docs/result_claims.md` for what can and cannot be claimed from these results.
- The oracle policy is excluded from the full comparison (run separately on small traces).
- Multi-Bin-style baseline is an approximate adaptation, not an official reproduction.
