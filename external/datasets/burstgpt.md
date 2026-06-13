# BurstGPT Dataset Provenance

## Citation

**Paper**: "BurstGPT: A Real-World Workload Dataset for LLM Serving Systems"
**Venue**: SIGMETRICS 2025
**arXiv**: https://arxiv.org/abs/2401.17644

## Official Sources

- **GitHub**: https://github.com/HKUDS/BurstGPT
- **HuggingFace**: https://huggingface.co/datasets/HKUDS/BurstGPT

## License

MIT License

## Dataset Description

BurstGPT contains real LLM serving request traces collected from production
systems. It captures temporal arrival patterns including bursty behavior that
is characteristic of real workloads.

## Fields Used

| Field | Source Column | Notes |
|---|---|---|
| `arrival_time` | `Timestamp` (unix seconds) | normalized so first request = 0.0 |
| `prompt_tokens` | `Request Token` | direct integer count |
| `actual_output_tokens` | `Response Token` | direct integer count |

## Synthetic Fields (NOT from dataset)

The following fields are **not present** in the BurstGPT dataset and are
generated synthetically by `augmentation.py`:

| Field | Generation Method |
|---|---|
| `predicted_output_tokens` | lognormal noise applied to `actual_output_tokens` |
| `class_id` | randomly sampled from SLO class distribution |
| `priority` | derived from assigned `class_id` |
| `slo_deadline` | `arrival_time + slo_slack` for the assigned class |

These fields are clearly labeled in trace metadata under `synthetic_fields`.

## Transformations Applied

1. Sort by timestamp
2. Normalize: subtract first timestamp so `arrival_time[0] = 0.0`
3. Apply `time_scale` to interarrival gaps (default 1.0 = no change)
4. Filter: drop rows with `prompt_tokens <= 0` or `output_tokens <= 0`
5. Clip prompt/output tokens to configured max values
6. Augment with synthetic SLO fields and prediction noise

## Download Date

<!-- Update this when you download the dataset -->
Downloaded: _not yet downloaded_

## Manual Download

If automated download fails:

```bash
# From GitHub
wget https://raw.githubusercontent.com/HKUDS/BurstGPT/main/data/BurstGPT_without_fails.csv \
    -O data/raw/burstgpt/BurstGPT_without_fails.csv

# Or from HuggingFace (requires git-lfs)
git clone https://huggingface.co/datasets/HKUDS/BurstGPT data/raw/burstgpt/hf/
```
