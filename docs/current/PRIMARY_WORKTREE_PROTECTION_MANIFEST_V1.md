# Primary Worktree Protection Manifest V1

This manifest records the dirty primary checkout observed before repository
finalization. It is a protection record, not an instruction to import or
delete any listed content.

Primary checkout: `/home/soroush/llm-serving-heuristic-evolution`

Observed branch: `contextual-compositional-heuristics-20260731`

Observed HEAD: `38e02bcdaeb8f9cec2289b05dea9a1406318d14f`

Canonical comparison commit: `e71e538301350b089be591e0cb1b91ed3476e750`

## Tracked modified files

The SHA-256 values below were measured in the primary checkout. `ABSENT` means
the path does not exist at the canonical comparison commit.

| Path | Status | Bytes | Primary SHA-256 | Canonical SHA-256 | Disposition |
|---|---:|---:|---|---|---|
| `.gitignore` | M | 3588 | `adf8d37f6a3eb3627e85ae8e6171d1b813f68a155314d705f3439ca9c1927cd3` | `0ebf4f360a22c06b5d705d69f2d1f339c893956743daa4fcfa1d6d1bc1fd5562` | PRESERVE_FOR_QUERY_3_REVIEW |
| `CITATION.cff` | M | 803 | `639307d60a972e14756e888cc42f68721fbbb8f3ecb0356127b408911b782c66` | `9a256ee113934605368c9ff3be11eec2a46303e6a53b8f966640dc292b5aad5c` | POSSIBLE_UNINTEGRATED_WORK |
| `CONTRIBUTING.md` | M | 1526 | `a5fbcf4a888f7b3a2980b98efa3eb49045add7ef7b16f5b18de88bc72c4f023f` | `f250cf7a03881330f941ce1a31ee27c68097fb5bf0b47e2cb65043275d0b2039` | POSSIBLE_UNINTEGRATED_WORK |
| `LICENSE` | M | 1078 | `4f7c3cbbd12439c2d6416b5186539f684bab3d4eab296c30a06e85442b394346` | `06dcddbb6908a0c6dd4a9e8ec822eea41d5a460a53089fecccc8a68049e99241` | POSSIBLE_UNINTEGRATED_WORK |
| `README.md` | M | 7268 | `6128424372beeab65b9cffc71b8cdb98c45a593910aec017f0cbcd4a9249c146` | `649d7896b9ac0913ccf0f5b63315c100beb9a60f6375409c7ee321fb5fc9f376` | PRESERVE_FOR_QUERY_3_REVIEW |
| `docs/PROJECT_MAP.md` | M | 23123 | `2f8c5f2e22596811cc6833b187e91d03dc65b79c3e452c6560bbf10e9b35f425d` | `841fb86a5923e74cfd35a6e1c3ddaa4eefae5b2ff6bd1f26c9e8f6ce3274cfed` | POSSIBLE_UNINTEGRATED_WORK |
| `docs/README.md` | M | 2544 | `7c0dfad1b7e95dd258cb8615775b1f934b56f5eefbd2066bdb455649490ab0db` | `b621c619760d9a64c93974bd0ec34125acfe3804e79e1ca115b1c3f35f32de36` | PRESERVE_FOR_QUERY_3_REVIEW |
| `docs/current/PROJECT_MAP.md` | M | 13794 | `092ae6cbddcfc4e4c7883b187e91d03dc65b79c3e452c6560bbf10e9b35f425d` | `a51b62bc2e48f97be58e28a5fc19cf4e503ce127dee9b398b377299e0341c51b` | POSSIBLE_UNINTEGRATED_WORK |
| `experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_SUPPORT_RESULT_V1.json` | M | 6708 | `612e80fac12136edea43ef613c9860d40b89b59f7120b17fc6ab9185438bc3c7` | `b2a0281577111f8d175ae1cd0846e0364ed581780df5069cd1045cf2930a7ffc` | PRESERVE_FOR_QUERY_3_REVIEW |
| `experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv` | M | 600848 | `8b701bb977062429e307d5b58ec45dc349769d41203716f6cd838feb959f2b54` | `35496946031c427fae14c885135eb40332f831a6fac6bb0b53068abd16716e7a` | PRESERVE_FOR_QUERY_3_REVIEW |
| `experiments/fresh_production_latency_headroom_confirmatory_v1/FRESH_SUPPORT_WORKLOAD_AXIS_SUMMARY_V1.csv` | M | 20867 | `d9559cd2794d6f492f71b93650b47ee03fc06a11c2bc15620a25883b2aaea2e6` | `86b13c987bbbee6d510fca0cf768e8a7aff3bf64cf7452d274a2bd994dd328f4` | PRESERVE_FOR_QUERY_3_REVIEW |
| `scripts/fresh_production_support_mapping_v1.py` | M | 30326 | `00fc707d35d2c8f55abe14712a6a9c93ef99bca361a25dbc7822a63a50ca95e1` | `95ea98c32bd7681e32f1fd200c2a1dbfbe8b58a4294a542c9968c874ba5c1ab5` | POSSIBLE_UNINTEGRATED_WORK |
| `scripts/phase17c_postprocess.sh` | M | 5916 | `58eb9ce4eefa7a09590b88993a69acba17c0502bc8026b653de3350c128a2adc` | `dfd8e3afdffc9bd19392785d3501873a37dc67468e0a3ab7156889d3cf65e4d8` | UNKNOWN |

## Scientifically relevant untracked content

These paths remain in the primary checkout and were not copied into this
cleanup branch:

- `experiments/fresh_production_latency_headroom_confirmatory_v1/` staging and
  raw-run material; approximately 2.4 MB in the primary checkout.
- `worktrees/`, including fresh-latency execution worktrees and their raw
  outputs.
- `experiments/family_a_pi0_closed_loop_final_v1/decision_rows.csv`, about
  1.7 GB.
- `experiments/family_a_receding_horizon_oracle_v1/`, including decision logs
  of about 827 MB.
- `experiments/family_a_wulver_medium_sweep_v1/` and related family-A outputs.
- `experiments/decision_criticality_timescale_trainval_v1/`.
- `experiments/sbs_override_fresh_confirmatory_corpus_v1/support_scan/`.
- `experiments/sbs_override_fresh_ood_support_expansion_v1/`, including support
  scans and zero-support diagnosis tables.
- `experiments/joint240_alive_underperformance_decomposition_v1/` and
  `experiments/joint240_guarded_abstaining_selector_v1/`.
- `paper/llm2026/archive/`, additional historical figures, templates, and
  `main.blg`.
- `artifacts/` and `datasets/`, local data/artifact bundles totaling roughly
  102 MB.
- `REPRODUCIBILITY.md`, an untracked primary-worktree document not present at
  the canonical comparison commit.

The separately registered unintegrated worktrees also retain untracked result
paths and are protected by this audit:

- `/home/soroush/llm-serving-heuristic-evolution/worktrees/fresh-latency-confirmatory-v1/experiments/fresh_production_latency_headroom_confirmatory_v1/fresh_latency_causal_run_v1/`
- `/home/soroush/llm-serving-heuristic-evolution/worktrees/fresh-latency-execution-v1/experiments/fresh_production_latency_headroom_confirmatory_v1/fresh_latency_causal_run_v1/`
- `/home/soroush/llm-serving-heuristic-evolution-joint240-dense-sbs-state-action-v1/experiments/joint240_dense_sbs_state_action_v1/`
- `/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-advantage-learnability-v1/experiments/joint240_sbs_advantage_learnability_v1/run_v1/`
- `/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-disagreement-scan-v1/experiments/joint240_sbs_disagreement_scan_v1/`
- `/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1/experiments/joint240_sbs_targeted_terminal_label_full_v1/`
- `/home/soroush/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-pilot-v1/experiments/joint240_sbs_targeted_terminal_label_pilot_v1/`

Ordinary caches and logs were observed but are not individually hashed here:
`.pytest_cache/`, `__pycache__/`, `.coverage`, `logs/`, `data/raw/`, and ignored
runtime logs. Query 3 must make the final disposition decisions for all
scientifically relevant paths before any cleanup of the primary checkout.

## Protection rule

This manifest does not authorize reset, restore, clean, stash, deletion,
checkout, merge, or import of primary-worktree content. It exists so Query 3
can reconcile the primary checkout without guessing.
