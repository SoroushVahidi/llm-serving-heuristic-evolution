# Provenance and lineage

## Release identity

* Title: **Joint-240 SBS Disagreement and One-Step Counterfactual Action Outcomes**
* Version: `1.0.0`
* Author: Soroush Vahidi, New Jersey Institute of Technology
* Source repository: https://github.com/SoroushVahidi/llm-serving-heuristic-evolution
* Software license: repository MIT license; this dataset is separately proposed
  under CC BY 4.0.

## Experiment lineage

The preserved study used 240 independently seeded synthetic scenarios from
the `joint_multimechanism_generalization_v1` scenario manifest. The SBS scan
was frozen on branch `experiment/joint240-sbs-disagreement-scan-v1` at
`11e9ac10898242a5c5fa95ac27174b9bba85339e`, with post-freeze source commit
`072e1a555e75ec7c732e6bf888026ef9850c6c54`. Dense acquisition used
`experiment/joint240-dense-sbs-state-action-v1` at launch
`94f4621bb661c2b426e365f659636b1a48a89f5` and post-freeze source commit
`11e9ac10898242a5c5fa95e`; the targeted-label full campaign used launch
`a1fc763b8b04dabf76b32ae2f30052cfd4a794e7`, source
`ff34f6fa0303b6f21e13277553f2d5da789b01d4`, and analysis
`0900ee7b565ee1cbfffe86272e30e4d0db13483c`. The five-fold learnability
analysis is historical context only and is not part of this release.

The frozen input hashes were recorded as:

```text
scenario_manifest.csv  9c1319eec8a107ced39277504caa17ffe47e3b3dc3ab866771470dd223363782
split_oof_folds.csv    e676e785d9a56ce829cd317e29e04655e92140206b1936339e18263cf48e6708
utility_matrix_wide.csv 99bc4303b69b90a56d475db2e708bc8adf62a8931f6d11cfba9b8d3ff0f9b679
```

Seeds are the synthetic scenario seeds in `metadata/scenario_metadata.csv`;
they are configuration identifiers, not upstream timestamps or private
request identifiers.

## Processing and exclusions

The scan produced 453,016 unique SBS trajectory states. The disagreement
manifest selected 8,888 states for full terminal labeling. Each unique full
canonical action was forced once, followed by fixed SBS continuation. The
release retains only alternative rows linked to the 8,886 retained labeled
states.

The release excludes all 98 state keys with unresolved hidden-state overlap,
their 2 labeled states and 4 linked terminal rows, all 8,888 public SBS
reference utilities, 260 exact public learnability aggregates, and
previously public/equivalent action outcomes. Raw shards, logs, fitted
predictions, and third-party trace files were never copied.

## Public overlap and historical gaps

The bounded overlap audit compared the named repository, GitHub branches,
Zenodo v1.0/v1.1 archives, and existing Hugging Face project datasets. It did
not find an equivalent Joint-240 release. This is not a proof against every
unlisted mirror. Full hidden simulator-state identity for the 98 matched
feature keys remains unresolved; conservative exclusion is intentional.

The original experiment worktrees are not available in the canonical
repository. The preserved local provenance archive is not a public release
dependency and is not redistributed.

## Relationship to other studies

The dataset's original SBS reference utilities equal already public
scenario-level utilities, so those references are excluded rather than
reissued as novel findings. The dataset is distinct from the active
Performance Evaluation manuscript's production-derived replay and
confirmatory latency analysis. It is historical exploratory Joint-240
synthetic ANWG research used to motivate the SBS reference, not confirmation
of the manuscript's production-derived result.
