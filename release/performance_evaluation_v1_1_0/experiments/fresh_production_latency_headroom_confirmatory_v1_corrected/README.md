# Corrected derivative: fresh causal state-level artifact

This directory is a **non-destructive derivative** of
`../fresh_production_latency_headroom_confirmatory_v1/FRESH_LATENCY_STATE_LEVEL_V1.csv`.
The frozen original is unchanged and remains the confirmatory record.

| File | Content |
|---|---|
| `FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv` | Same 720 rows, identifiers and column order as the original. Only `mean_ref_latency` and `p95_ref_latency` differ: they now hold the SBS-reference branch values. |
| `STATE_LEVEL_CORRECTION_AUDIT_V1.csv` | Per-state original vs corrected values, deltas, change flags, and the counterfactual branch the original value came from. |
| `SBS_REFERENCE_ROWS_V1.csv` | The 720 `SBS_REFERENCE` rows extracted from the hash-verified continuation shards (source of the corrected values). |
| `CORRECTION_PROVENANCE_V1.json` | Hashes (source, outputs, frozen inputs, shards), replay proof, validation, generator. |
| `ROBUSTNESS_AND_590_589_RECHECK_V1.json` | Recheck of the robustness commit and the 590 vs 589 classification. |

Regenerate: `python scripts/fresh_causal_correct_state_level_v1.py --archive-root <local-provenance-archive>`.
Documentation: `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`.
