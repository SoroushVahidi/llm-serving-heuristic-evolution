# Dataset Preservation

Tier 1 staging root (external; ~26 GB; may vanish with Wolverine storage):
`/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets`

## Datasets
| Dataset | Role | Notes |
| --- | --- | --- |
| burstgpt_v2 | primary real serving | metadata-preservation repair applied |
| azure_llm_2023 | primary real serving | |
| azure_llm_2024 | primary real serving | chronological sort / 164-inversion repaired |
| bailian_qwen | primary/supporting real | |
| mooncake | **internal OOD only** | see license block |

## Mooncake (prominent)
```
DATA_LICENSE = NOT_EXPLICITLY_SPECIFIED
REDISTRIBUTION = PROHIBITED_UNTIL_CLARIFIED
EVALUATION_ROLE = INTERNAL_OOD_ONLY
```
Do **not** commit Mooncake data or row-level derivatives to GitHub.

## Reacquisition
Use download/convert scripts under `scripts/data/` with pinned sources recorded in each dataset’s staging manifests on the shared data root. Prefer verifying `checksums.sha256` after re-download.

Machine-readable: `dataset_preservation.json`.
