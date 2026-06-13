# Real Trace Replay: Step-by-Step Guide

## Prerequisites

- Python 3.10+
- Project dependencies installed: `pip install -e .`
- Internet access (for download step only)

## 1. Download BurstGPT

```bash
python scripts/download_burstgpt.py --output data/raw/burstgpt/
```

If automated download fails:
```bash
wget 'https://raw.githubusercontent.com/HKUDS/BurstGPT/main/data/BurstGPT_without_fails.csv' \
    -O data/raw/burstgpt/BurstGPT_without_fails.csv
```

The downloaded file is approximately 50-200 MB. It is excluded from version
control by `.gitignore`.

## 2. Verify the Download

```bash
python scripts/summarize_trace.py \
    --input data/raw/burstgpt/BurstGPT_without_fails.csv \
    --out-dir /tmp/burstgpt_verify/
```

This will print token statistics and arrival rate without running the
full conversion. Check that the statistics look reasonable.

## 3. Convert to Simulator JSONL

```bash
python scripts/convert_burstgpt.py \
    --input data/raw/burstgpt/BurstGPT_without_fails.csv \
    --output data/processed/burstgpt/burstgpt_10k.jsonl \
    --max-requests 10000 \
    --seed 17 \
    --config configs/traces/burstgpt_conversion.yaml
```

This creates:
- `data/processed/burstgpt/burstgpt_10k.jsonl` — the processed trace
- `data/processed/burstgpt/burstgpt_10k.report.json` — conversion statistics

## 4. Inspect the Processed Trace

```bash
python scripts/summarize_trace.py \
    --input data/processed/burstgpt/burstgpt_10k.jsonl \
    --out-dir results/burstgpt_analysis/
```

Check the output for:
- N requests (should be ≤ 10,000)
- Mean arrival rate
- Token distribution percentiles
- SLO class proportions

## 5. Run the Baseline Comparison

```bash
python scripts/run_real_trace_comparison.py \
    --config configs/burstgpt_replay_comparison.yaml \
    2>&1 | tee results/burstgpt_replay.log
```

Results are saved to `results/burstgpt_replay_comparison/<timestamp>/`.

## 6. Run Scaled Load (Optional)

```bash
python scripts/run_real_trace_comparison.py \
    --config configs/burstgpt_replay_scaled_load.yaml \
    2>&1 | tee results/burstgpt_scaled.log
```

This uses `time_scale: 0.5` to compress interarrival times by 2x, doubling
the effective arrival rate.

## 7. Interpret Results

Key metrics to examine:

- **SLO violation rate**: fraction of requests completing after their
  deadline. Note: SLO deadlines are synthetic (see `docs/workload_realism.md`).
- **Mean latency**: average time from arrival to completion.
- **p95 latency**: 95th percentile latency, sensitive to tail behavior.
- **Request throughput**: requests completed per second.
- **GPU utilization**: fraction of time GPU is processing requests.

Compare against synthetic workload results to understand how real arrival
burstiness and token distributions affect policy rankings.

## When Dataset Is Unavailable

If the BurstGPT dataset cannot be downloaded (network restrictions, license
concerns, etc.):

1. Use the tiny fixture for development:
   ```bash
   python scripts/summarize_trace.py \
       --input tests/fixtures/burstgpt_tiny.csv \
       --out-dir /tmp/test_out/
   ```

2. Use synthetic workloads with bursty arrivals as an approximation:
   ```bash
   python scripts/run_baseline_comparison.py \
       --config configs/burst_heavy_tail_comparison.yaml
   ```

3. All tests pass without any dataset download (fixtures are in `tests/fixtures/`).
