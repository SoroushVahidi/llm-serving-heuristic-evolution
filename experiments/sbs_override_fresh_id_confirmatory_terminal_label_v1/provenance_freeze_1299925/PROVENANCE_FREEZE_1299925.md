# Fresh Label Campaign Provenance Freeze: Job 1299925

This directory preserves the exact untracked source/design/slurm files located on NJIT Wulver for fresh terminal-label job `1299925`.

Remote source checkout:

`/mmfs1/project/ikoutis/sv96/llm-serving-heuristic-evolution-joint240-sbs-targeted-terminal-label-full-v1-src`

Source branch:

`experiment/joint240-sbs-targeted-terminal-label-full-v1`

Committed source base:

`ff34f6fa0303b6f21e13277553f2d5da789b01d4`

The preserved files are copied without modification. SHA-256 values are recorded in `local_sha256.txt`; the remote and local hashes matched at copy time.

The generated fresh labels themselves were not copied, edited, opened for scientific inspection, or altered by this provenance freeze.

Run notes:

- The raw fresh manifest had 56 duplicate state IDs.
- The actual run used the clean `full_support_only` input.
- The actual confirmatory universe is 2,862 unique SBS-vs-P6 disagreement states across 78 supported fresh scenarios, with 6,996 unique non-SBS actions and 9,858 total terminal continuations.
- The remote/local `preregistration_boundary.json` differed and is documented as provenance context only. It was not an active run input for job `1299925`.

Commit suitability:

These source/design/slurm files are suitable for committing on a provenance branch because they are code and metadata, not confirmatory outcome labels.

## Frozen File Checksums

The following files were copied from the Wulver source tree without
modification:

| SHA-256 | Frozen copy |
| --- | --- |
| `b8ffddca8f27ebed5ebe947d59740935f90ccf0537f4319d2337bef8d2dc8a3f` | `docs_design/SBS_OVERRIDE_FRESH_CONFIRMATORY_CORPUS_V1.md` |
| `0bb79fe8a2ca1872feec566e38b74a47c7e481bc3b52cb71c4afe155e5e295ad` | `docs_design/SBS_OVERRIDE_FRESH_ID_CONFIRMATORY_TERMINAL_LABEL_V1.md` |
| `db136b49b4b56d142947c625ee560b893dacfae0f587e20e10d079516aed9ed4` | `docs_design/SBS_OVERRIDE_FRESH_OOD_SUPPORT_EXPANSION_V1.md` |
| `31d9ac06391f2510698ab0f365d26b8235f93e66dea3958ba6a6135477ba40ca` | `scripts/sbs_override_fresh_confirmatory_corpus_v1.py` |
| `216cb60e0ca6ec9a0a3ef31f6e478272efaae05e94df3676b26bc8391bc6e70b` | `scripts/sbs_override_fresh_id_confirmatory_terminal_label_v1.py` |
| `1509728e32505b31cebb8cd6e2b87c4170fbf8c889067ab8ea8186acb27ee21c` | `scripts/sbs_override_fresh_ood_support_expansion_v1.py` |
| `ea213a4ce4f5b01284ee99389af0e44f7de2f9cdbaaa8275b187ab812dd3505a` | `scripts_slurm/sbs_override_fresh_id_confirmatory_terminal_label_v1.sbatch` |

The same values are recorded in `local_sha256.txt`.
