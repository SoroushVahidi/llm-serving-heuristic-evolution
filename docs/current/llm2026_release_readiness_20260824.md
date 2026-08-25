# LLM 2026 Release Readiness

Date: 2026-08-24

Scope: author metadata and public-release preparation for
`The Exploitability Gap in LLM-Serving Scheduler Portfolios`. This pass did not
run experiments, change scientific results, publish artifacts, create tags, or
push to GitHub/Hugging Face.

## Pre-Flight

- Branch: `contextual-compositional-heuristics-20260731`
- HEAD: `2987b7181efa2bc550d8a894c537eca8f6393eb6`
- Upstream: `origin/contextual-compositional-heuristics-20260731`
- Ahead/behind: `2 ahead / 0 behind`
- Worktrees: one worktree at `/home/soroush/llm-serving-heuristic-evolution`
- Git lock state: `.git/index.lock` was present and was not removed.
- Current PDF page count: 15 LNCS pages.
- Compile command: `cd paper/llm2026 && tectonic --keep-logs main.tex`
- Compile status: exit 0; no fatal errors, undefined citations, or undefined
  references observed in the final compile.

Tracked dirty files and many untracked scientific/manuscript artifacts remain.
They were preserved.

## Author Block

The current manuscript remains anonymous:

```tex
\author{Anonymous Author(s)\inst{1}}
\authorrunning{Anonymous Author(s)}
\institute{Anonymous Institution, City, Country\\
\email{anonymous@example.com}}
```

Use the following identified LNCS block only after confirming that the submission
does not require the PC-conflict double-blind route:

```tex
% Normal identified submission; use only after review-mode confirmation.
\author{Soroush Vahidi\inst{1}\orcidID{[ORCID_IF_USED_OR_REMOVE]}}
\authorrunning{S. Vahidi}
\institute{[AFFILIATION], [CITY], [STATE_OR_PROVINCE_IF_APPLICABLE], [COUNTRY]\\
\email{[EMAIL]}}
```

Missing author metadata:

- exact submission author name spelling;
- confirmation that there is only one author;
- affiliation;
- city/state-or-province/country;
- email;
- ORCID or confirmation to remove `\orcidID{...}`;
- corresponding-author marker, if required by the submission form;
- confirmation that no PC-conflict double-blind route applies.

## Public-Release Safety

### Safe to Publish, Subject to Final Author Review

- Source code under `src/`, `scripts/`, and `tests/` that does not contain
  literal credentials.
- Manuscript sources under `paper/llm2026/`, including official Springer
  template provenance, figures, bibliography, and compiled PDF.
- Compact paper-relevant experiment summaries under:
  - `experiments/joint_multimechanism_generalization_v1/`
  - `experiments/portfolio_guided_typed_gp_screen_v1/`
  - `experiments/real_vllm_mechanism_validation_v1/`
- Paper planning and audit documents under `docs/current/`, after author review
  for tone and identifying metadata.
- Public-trace corpus metadata files such as
  `data/public_trace_corpus_v1/manifest.json`, `schema.json`,
  `source_coverage.csv`, and `distribution_stats.json`.

### Needs Review Before Public Release

- Secret-like strings in logs/docs/scripts. Most source-code hits are placeholder
  environment-variable names, but the following paths need manual review because
  they are logs, generated data, or audit outputs:
  - `results/apt_serve_phase_g_analysis_20260809_190000/environment.txt`
  - `results/baseline_api_audit/provider_identity_audit.log`
  - `results/baseline_api_audit/token_env_audit.log`
  - `results/baseline_api_audit/token_file_audit.log`
  - `dataset_staging/_audit/security_scan_report.md`
  - `data/processed/wildchat/wildchat_eval_prompts_by_id.json`
  - `data/processed/wildchat/wildchat_eval_sharegpt_shaped.json`
  - `results/pars_official/data_preprocess/alpaca_gpt4/val_data.json`
- API setup documentation should be checked for example values and publication
  tone:
  - `docs/api_provider_setup.md`
  - hosted-provider calibration scripts/tests under `scripts/` and `tests/`
- Wulver/Vulver bundle manifests and docs with local absolute paths or remote
  host details should be reviewed for disclosure appropriateness:
  - `artifacts/wulver_family_a_medium_sweep_bundle_v1/`
  - `docs/wulver_sarathi_vllm_repeated_validation.md`
- Raw or processed third-party trace files under `data/raw/`, `data/processed/`,
  and `data/public_trace_corpus_v1/*/records.parquet` should be reviewed against
  license and redistribution requirements before inclusion.
- Large generated artifacts should be intentionally included, excluded, or moved
  to an artifact release:
  - `experiments/family_a_pi0_closed_loop_final_v1/decision_rows.csv`
  - `experiments/family_a_receding_horizon_oracle_v1/family_a_receding_horizon_oracle_v1_decision_logs.csv`
  - `experiments/family_a_wulver_medium_sweep_v1/merged/candidate_states.jsonl`
  - `experiments/decision_criticality_timescale_trainval_v1/disagreement_and_divergence_events.csv`
  - `results/module_credit_overnight/module_credit_overnight_20260722T000121/trial_results.jsonl`

### Must Not Be Made Public As-Is

- Any file containing actual API tokens, access tokens, SSH credentials,
  service-account secrets, cookies, or private keys if found during manual
  review.
- Local model checkpoints and training outputs not needed for the paper:
  - `results/pars_official/predictor_train/alpaca_gpt4_bert/last_model.pt`
  - `results/pars_official/predictor_train/alpaca_gpt4_bert/best_model.pt`
- Large raw/generated third-party-derived files unless redistribution is
  explicitly intended, licensed, attributed, and documented.
- Private or unrelated datasets not used by the final manuscript, including
  WildChat/ShareGPT/alpaca-derived working files unless separately licensed and
  intentionally released.

No SSH private-key filenames were found by the release scan. `.env.example` was
the only `.env`-style file found at shallow repository depth.

## Third-Party Data Redistribution

Final manuscript results use BurstGPT and the Azure LLM Inference Trace 2023
conversation/code splits.

- BurstGPT source: `https://github.com/HPMLL/BurstGPT`. The repository exposes
  public releases for research and academic use and is marked CC-BY-4.0 in the
  GitHub license panel. The README requests citation of the KDD 2025 BurstGPT
  paper.
- Azure LLM Inference Trace 2023 source:
  `https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md`.
  The source page states that the data are made available under a CC-BY
  Attribution License and requests citation of the Splitwise ISCA 2024 paper.

Recommended release policy:

- Publish preprocessing scripts, checksums, source URLs, manifests, and compact
  derived summaries.
- Do not include raw third-party CSVs or processed per-request trace copies in
  the public repository unless the author explicitly decides to redistribute
  them with license files and attribution.
- If processed trace derivatives remain public, clearly distinguish original
  external traces, timestamp scaling/windowing, synthetic SLO/priority labels,
  and project-generated annotations.

## Reproducibility Status

Paper-relevant implementation/artifact coverage is present:

- six scheduler implementations and policy modules are present under `src/`;
- workload generation and ingestion scripts are present under `scripts/` and
  `src/`;
- joint 240-scenario generator, manifest, utility matrices, SBS/VBS/oracle
  summaries, and decision artifacts are present under
  `experiments/joint_multimechanism_generalization_v1/`;
- selector/router/support/composition/synthesis summaries and audit documents
  are present under `docs/current/` and `experiments/`;
- typed-GP screen and random-candidate audit artifacts are present under
  `experiments/portfolio_guided_typed_gp_screen_v1/`;
- real-vLLM validation, fidelity diagnosis, and native token-budget probe
  artifacts are present under `experiments/real_vllm_mechanism_validation_v1/`;
- manuscript figures are present under `paper/llm2026/figures/`;
- Python packaging metadata is present in `pyproject.toml`.

Documentation gaps before public release:

- Add a paper-specific reproduction guide or README section mapping each
  manuscript figure/table to exact source artifacts and scripts.
- Document environment setup for CPU simulator analyses separately from optional
  local-vLLM replication.
- Mark raw third-party trace download/preprocessing steps without bundling raw
  data unless redistribution is confirmed.
- Identify a stable paper commit or tag after final author metadata/release
  cleanup.
- Decide whether large untracked generated artifacts should be excluded,
  summarized, or moved to an external artifact archive.

Suggested README addition:

```markdown
## LLM 2026 Paper Artifact

The LLM 2026 manuscript is under `paper/llm2026/`. Paper-relevant derived
artifacts are under:

- `experiments/joint_multimechanism_generalization_v1/`
- `experiments/portfolio_guided_typed_gp_screen_v1/`
- `experiments/real_vllm_mechanism_validation_v1/`
- `docs/current/llm2026_*_20260824.md`

External traces are BurstGPT and Azure LLM Inference Trace 2023. To reproduce
trace-derived artifacts, download them from their original sources, verify
checksums, and run the preprocessing scripts under `scripts/data/`. The
repository may include compact derived summaries; raw trace redistribution
depends on source licenses and release policy.
```

## Release Identifier Strategy

Recommended publication identifier:

- Git tag/release: `llm2026-submission`
- Data and Code Availability should cite:
  - repository URL: `https://github.com/SoroushVahidi/llm-serving-heuristic-evolution`
  - exact release tag: `llm2026-submission`
  - exact commit SHA after final cleanup and author approval
- Optional archival DOI: create a Zenodo archive after the public GitHub release
  if the submission process or artifact policy benefits from a DOI.

Do not create the tag/release until author metadata, anonymization status,
third-party-data policy, and release cleanup are complete.

## Hugging Face Release Decision

Recommendation: no dedicated Hugging Face dataset is required before submission;
the GitHub repository is sufficient if cleaned and tagged.

Optional post-review artifact if desired:
`SoroushVahidi/llm-serving-scheduler-portfolio-results`.

If created later, include only redistributable paper artifacts:

- joint 240-scenario workload descriptors;
- six-policy utility matrix and SBS/VBS/oracle summaries;
- selector/router frozen-gate summaries;
- typed-GP summary outputs;
- real-vLLM aggregate measurements and mechanism summaries;
- provenance metadata;
- README;
- license and attribution files.

Exclude raw third-party traces, private/unrelated datasets, model checkpoints,
credential logs, and any artifact whose redistribution is uncertain.

## Final Verdict

`READY_AFTER_MINOR_RELEASE_FIXES`

The manuscript is ready for author-side release preparation, but immediate public
release should wait until the author:

1. confirms author metadata and review/anonymization mode;
2. reviews/removes any actual secrets in generated logs and environment audits;
3. decides raw/processed third-party trace redistribution policy;
4. excludes large non-paper or non-redistributable artifacts;
5. adds or approves the paper-specific reproduction guide;
6. creates a clean release commit/tag.
