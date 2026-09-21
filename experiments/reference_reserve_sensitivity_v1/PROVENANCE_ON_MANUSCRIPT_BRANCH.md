# Provenance of the reserve-sensitivity artifacts on the manuscript branch

This branch (`revise/peva-metadata-consistency-20260921`) carries the reserve-sensitivity experiment only as far as the manuscript's
claims need it. It was assembled from `experiment/reference-reserve-sensitivity-20260921` and is not a full copy of that branch.

## Present here (scientifically necessary provenance)

| Path | Role | sha256 (first 16) |
|---|---|---|
| `PROTOCOL_V1.json` | Frozen protocol, committed before the sensitivity jobs ran | `97b06cf1c96fe6b3` |
| `IMPLEMENTATION_V1.md` | Implementation notes for the frozen runner | `4f05992ca0edb9c2` |
| `DENOMINATOR_COMPLETION_V1.md` | Post hoc amendment frozen before the denominator pass (gates, counting rule) | `c58734e974b8c2cb` |
| `denominator_completion_v1/EXISTING_SENSITIVITY_RESULT_HASHES_V1.json` | Frozen sha256 of all 21 sensitivity result files | `9bd8677933e0818f` |
| `results_v1/`, `denominator_completion_v1/`, `DENOMINATOR_COMPLETION_RESULT_V1.md`, `PRESERVATION_MANIFEST_V1.json` | Results, counts, derived tables, sha256 manifest | see manifest |

The four documents above were imported byte-identically from commit `ec19963`. The 18 preserved result files verify against the frozen
hashes. The three `action_effects.csv` files are identified by hash only and stay in cluster scratch.

## Not present here, and why

* **Runner scripts and their test** are not on this branch. They are identified by immutable commits and recorded sha256:
  `scripts/reference_reserve_sensitivity_v1.py` at `8addcdb` (sha256 `555337c1435200a1...`, the executed sensitivity runner) and
  `scripts/reference_reserve_denominator_completion_v1.py` at `ec19963` (sha256 `d7c2b2985828a9ee...`, run as Slurm job 1303832). Both hashes are
  recorded in `denominator_completion_v1/count_execution_provenance.json`. Importing them would add unrelated experiment code and a test to the
  manuscript branch without changing any manuscript claim.
* **Slurm job logs** (`slurm/logs/*`, 8 files, listed by sha256 in `PRESERVATION_MANIFEST_V1.json`) are operational records that the repository's
  `.gitignore` (`logs/`) excludes. They are not scientifically necessary: the outcome of each job is recorded in the provenance JSON files
  (`execution_provenance.json`, `count_execution_provenance.json`) and in `DENOMINATOR_COMPLETION_RESULT_V1.md`.

The manifest therefore lists 8 files that are absent from a clean checkout of this branch; those are exactly the ignored logs.
