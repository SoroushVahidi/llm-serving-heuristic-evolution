# Provenance and lineage

## Release identity

* Title: **Joint-240 SBS Disagreement and One-Step Counterfactual Action Outcomes**
* Version: `1.0.0`
* Author: Soroush Vahidi, New Jersey Institute of Technology
* Source repository: https://github.com/SoroushVahidi/llm-serving-heuristic-evolution
* Dataset path in repository:
  `datasets/joint240-sbs-action-outcomes/v1.0.0/`
* Data license: CC BY 4.0 (dataset files in this directory only)
* Software license: repository MIT license (separate)

Public provenance uses repository-relative paths and git commit hashes. Private
absolute filesystem paths are not part of this release.

## Experiment lineage (verifiable)

The preserved study used 240 independently seeded synthetic scenarios from
`experiments/joint_multimechanism_generalization_v1/scenario_manifest.csv`.

| Stage | Branch | Launch / HEAD (verifiable commit) | Notes |
|---|---|---|---|
| SBS disagreement scan | `experiment/joint240-sbs-disagreement-scan-v1` | Launch `11e9ac10898242a5c5fa95ac27174b9bba85339e`; post-freeze source `072e1a555e75ec7c732e6bf888026ef9850c6c54` | Design: `docs/design/JOINT240_SBS_DISAGREEMENT_SCAN_V1.md` |
| Dense acquisition (context only) | `experiment/joint240-dense-sbs-state-action-v1` | Launch `94f4621bb6610c2b426e365f659636b1a48a89f5`; post-freeze source `11e9ac10898242a5c5fa95ac27174b9bba85339e` | Not redistributed as dense tables |
| Targeted terminal labels (full) | `experiment/joint240-sbs-targeted-terminal-label-full-v1` | Launch `a1fc763b8b04dabf76b32ae2f30052cfd4a794e7`; source `ff34f6fa0303b6f21e13277553f2d5da789b01d4`; analysis `0900ee7b565ee1cbfffe86272e30e4d0db13483c` | Design: `docs/design/JOINT240_SBS_TARGETED_TERMINAL_LABEL_FULL_V1.md` |
| Advantage learnability | `experiment/joint240-sbs-advantage-learnability-v1` | Historical offline analysis | Public aggregates already published; excluded here |

Frozen input content digests recorded in the Phase 6B audit:

```text
scenario_manifest.csv     9c1319eec8a107ced39277504caa17ffe47e3b3dc3ab866771470dd223363782
split_oof_folds.csv       e676e785d9a56ce829cd317e29e04655e92140206b1936339e18263cf48e6708
utility_matrix_wide.csv   99bc4303b69b90a56d475db2e708bc8adf62a8931f6d11cfba9b8d3ff0f9b679
```

Canonical source table digests used when building the private candidate
(not redistributed here):

```text
state_rows.csv                         e3808ffbb69bfba41c940bfc899db191ef1b642c49748a9021e959e20ea5a966
manifest_all_disagreements.csv         4c41b5c2f279a914b89f384722ceca76cc03bb78145c81f02b790c324baa1df9
state_action_rows_full.csv             a96af891b70bcd0895440207df6febb5d1c1545af4eaaf455367e56152950dca
state_policy_action_map_full.csv       7a4118b65e4222e189aa1139ba548ade99951ae285a50b7b193fec6ef55d1324
```

Seeds in `metadata/scenario_metadata.csv` are synthetic scenario configuration
identifiers (236 unique seeds in this release). They are not upstream request
timestamps.

## Scenario generation

The generator builds synthetic arrivals and token lengths from seeded NumPy
draws and frozen parameters. It does not load BurstGPT, Azure, Mooncake, or
prompt corpora into these released tables. Mechanism and pressure parameters
are stored per scenario in `metadata/scenario_metadata.csv`.

## Processing and exclusion steps (release transform)

1. Start from the full SBS scan (453,016 unique predecision states).
2. Select the 8,888 disagreement states for terminal labeling.
3. Force each unique full canonical action once; continue with fixed SBS.
4. Retain only alternative-action rows (exclude SBS reference action rows as
   novel utilities).
5. Exclude all 98 historically ambiguous `(scenario_id, step)` keys whose 95
   numeric features matched earlier Alive-acquired states while hidden-state
   identity remained unresolved (removes 2 labeled states and 4 terminal rows).
6. Exclude already-public SBS reference utilities, 260 public learnability
   aggregate rows, and public-equivalent alternative outcomes.
7. Emit normalized `state_context.csv`, `alternative_action_outcomes.csv`, and
   scenario/policy metadata.

Release counts after transform: **8,886** states, **21,854** alternatives,
**236** scenarios, **6** policies.

## Public overlap analysis

Bounded comparison against named GitHub branches, Zenodo manuscript packages
v1.0.0/v1.1.0, and existing Hugging Face project datasets found no prior
equivalent Joint-240 SBS state/action release. That is not a proof against every
unlisted third-party mirror. See `PUBLIC_RECORD_OVERLAP.md`.

## Differences from public aggregate studies

* Public learnability CSVs are scenario/fold/model aggregates, not state-action
  terminal tables.
* Public utility-matrix SBS values are scenario-level references; this release
  embeds per-state `q0_*` only to define within-state contrasts and does not
  reissue the utility matrix.
* The active PE confirmatory archive uses production-derived replay and latency
  headroom; this release uses synthetic Joint-240 ANWG outcomes.

## Historical gaps (explicitly unresolved)

* Original experiment worktrees are not present as live checkouts in the
  canonical repository; preserved private provenance archives are not public
  dependencies.
* Full hidden simulator-state identity for the 98 matched-feature keys remains
  unresolved; conservative exclusion is intentional.
* Exact live editorial status of PEVA-D-26-00474 beyond the 2026-10-07
  under-review correspondence is not independently reconfirmed in this package.

## ORCID

Authoritative public Zenodo creator metadata for related deposits by
Soroush Vahidi list affiliation (NJIT) but do not include a verified ORCID.
No ORCID is asserted in this release.
