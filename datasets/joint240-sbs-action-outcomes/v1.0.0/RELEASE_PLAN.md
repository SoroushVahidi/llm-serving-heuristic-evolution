# Release plan (executed under Phase 6E)

This file records the authorized publication route for version 1.0.0.

## Authorization basis

Phase 6E author preferences authorize publication of the verified historical
Joint-240 supplement under CC BY 4.0 with:

* Author: Soroush Vahidi
* Affiliation: New Jersey Institute of Technology
* Version: 1.0.0
* Title: Joint-240 SBS Disagreement and One-Step Counterfactual Action Outcomes
* Destinations: GitHub source tree, distinct Zenodo dataset record, Hugging Face
  identical mirror

## Editorial relationship

Manuscript PEVA-D-26-00474 (Performance Evaluation; reported under review
2026-10-07) remains unmodified. This dataset is labeled historical exploratory
synthetic evidence and is kept separate from the manuscript’s production-derived
confirmatory archive (existing Zenodo v1.1.0). Performance Evaluation research
data Option C requires depositing research data; this independent dataset
deposit satisfies sharing without attaching the historical tables as
manuscript supplementary files unless the author later chooses to cite them
there.

## Sequence

1. Freeze scientific CSVs at the Phase 6C hashes.
2. Complete public documentation and public-only validation.
3. Publish one directory under
   `datasets/joint240-sbs-action-outcomes/v1.0.0/` on GitHub (isolated branch,
   review, merge). Do not create a GitHub Release that would duplicate archives.
4. Publish one distinct Zenodo **Dataset** deposit (not a new version of the
   manuscript package), version 1.0.0, CC BY 4.0.
5. Mirror identical files to Hugging Face
   `SoroushVahidi/joint240-sbs-action-outcomes`.
6. Verify public hashes, DOIs, and load examples externally.

## Metric interpretation

`q_sbs_*` = terminal metric after candidate action + fixed SBS continuation  
`q0_sbs_*` = same-state SBS reference  
`a_sbs_* = q_sbs_* - q0_sbs_*` for ANWG/soft/WCG  
`a_sbs_*_improvement = q0_sbs_* - q_sbs_*` for WMT/WNT
