# Data Directory

> For the current study's data sources and what is or is not redistributed, see
> [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) and the current-scope note in
> [`../docs/DATA_RELEASE_POLICY.md`](../docs/DATA_RELEASE_POLICY.md).

## Structure

- `raw/` — unmodified downloaded datasets as obtained from original sources
- `processed/` — JSONL files in the simulator's canonical schema (one Request per line)

## Version Control

Neither `raw/` nor `processed/` is committed (see `.gitignore`). The only tracked
data files are the metadata of the public trace corpus under
`public_trace_corpus_v1/` (`manifest.json`, `schema.json`,
`distribution_stats.json`, `source_coverage.csv`); its parquet window tables are
built locally and not committed. Do not commit raw CSV, JSONL, Parquet, or any
file containing private user data.

**Never commit API keys, tokens, or credentials anywhere under this directory.**
If a download script needs authentication (e.g., `HF_TOKEN`), put the key in
`.env` (gitignored) and load it from the environment.

## Download Scripts

- `scripts/download_burstgpt.py` — downloads BurstGPT from HuggingFace (requires `HF_TOKEN`)
- ShareGPT: download manually from the original source; see `docs/milestones/phase1_7a_real_traces.md`

## Conversion Scripts

- `scripts/convert_burstgpt.py` — converts raw BurstGPT CSV to JSONL in the simulator schema
- `scripts/convert_sharegpt.py` — converts raw ShareGPT JSON to JSONL

## Field Provenance

See `docs/data_field_provenance.md` for which fields come from the original dataset
and which are synthetically augmented (SLOs, priorities, predicted output lengths).

## Licensing

- **BurstGPT**: CC-BY-4.0 (license of the upstream repository
  <https://github.com/HPMLL/BurstGPT>, checked 2026-10-04; an earlier version of
  this file said MIT). Wang et al., "BurstGPT: A Real-World Workload Dataset to
  Optimize LLM Serving Systems," KDD 2025, doi:10.1145/3711896.3737413.
- **Azure LLM inference traces 2023**: CC-BY-4.0 (license of the upstream
  repository <https://github.com/Azure/AzurePublicDataset>, checked 2026-10-04).
- **ShareGPT**: Community-collected conversational data. Check the original
  distribution source for current license terms before redistribution.
