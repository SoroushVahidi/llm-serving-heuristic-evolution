# Real Dataset Expansion Status

**Updated:** 2026-07-25 (Part 2 pause preservation)

## Status

`STAGED_AND_WINDOW_READY` (external artifacts on Wolverine; compact summaries in Git).

Branch: `reality-grounded-dataset-expansion-20260724`

## Distinctions

| Class | Datasets | Role |
| --- | --- | --- |
| Real serving traces (Tier 1) | BurstGPT v1.0/v2 staging, Azure LLM 2023, Azure LLM 2024, Bailian/Qwen | Primary / supporting evaluation after windowing |
| Prompt corpora | (not the Tier 1 staging focus) | Separate from request-arrival traces |
| Internal-only | Mooncake (real-only) | `EVALUATION_ROLE=INTERNAL_OOD_ONLY`; redistribution prohibited until license clarified |
| Future candidates | SwissAI/TraceLab already explored separately | Not part of this Tier 1 staging claim |

## Completed

1. Tier 1 download/convert/validate (~26 GB under `llmserveopt-data/datasets`).
2. Azure 2024 chronological 164-inversion repair; BurstGPT metadata preservation repair.
3. Validated real-window construction (`ALL_COMPLETE_VALID`) with natural, busy, time-scaled (2×/4×/8×), and trace-calibrated synthetic families.
4. First capped pilot (flawed: Mooncake omitted) and repaired stratified pilot (`PARTIALLY_READY`).

## Not completed / not authorized

- Full 27-policy fingerprint sweep on the new windows.
- Mooncake public redistribution or GitHub upload of row-level data.
- Claiming natural-load discrimination is solved (near-tie rate still high).

## Git-safe summaries

See `docs/current/pause_2026_07_25/DATASET_PRESERVATION.md` and
`docs/current/pause_2026_07_25/REAL_WINDOW_SUMMARY.md`.

Machine-readable: `docs/current/real_dataset_expansion_status.json`.
