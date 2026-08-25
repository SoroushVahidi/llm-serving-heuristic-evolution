# Real Window Construction Summary

Validated overnight run (historical path may vanish):
`/mmfs1/project/ikoutis/sv96/llmserveopt-data/real_window_construction_20260725T035054Z`

- Git SHA: `4dd97eadd16aa65512db61af07f7750596c08d14`
- Status: `ALL_COMPLETE_VALID`
- Jobs: {
  "burstgpt": "1143271",
  "azure2023": "1143272",
  "azure2024": "1143273",
  "bailian": "1143274",
  "mooncake": "1143275",
  "pilot": "1143276",
  "morning": "1143277"
}
- Approximate size at pause: ~237 MB (window tree; Part 1 cited ~342 MB including nested pilots/logs)
- Azure 2024 chronological 164-inversion: corrected before window build
- Mooncake: real-only windows; redistribution prohibited until license clarified

## Dataset window counts
{
  "burstgpt_v2": {
    "n_windows": 168,
    "validation_ok": true
  },
  "azure_llm_2023": {
    "n_windows": 110,
    "validation_ok": true
  },
  "azure_llm_2024": {
    "n_windows": 140,
    "validation_ok": true
  },
  "bailian_qwen": {
    "n_windows": 190,
    "validation_ok": true
  },
  "mooncake": {
    "n_windows": 108,
    "validation_ok": true
  }
}

## Families
natural_replay, natural_busy_period, trace_derived_time_scaled (2x/4x/8x), trace_calibrated_synthetic

## Reconstruct
See `REPRODUCTION_COMMANDS.md` and `scripts/data/run_real_window_dataset_pipeline.py`.
