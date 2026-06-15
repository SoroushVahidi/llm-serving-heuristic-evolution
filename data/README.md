# Data Directory

## Structure

- `raw/` — unmodified downloaded datasets as obtained from original sources
- `processed/` — JSONL files in the simulator's canonical schema (one Request per line)

## Version Control

Neither `raw/` nor `processed/` is committed (see `.gitignore`). Only `.gitkeep`
placeholder files are tracked. Do not commit raw CSV, JSONL, Parquet, or any
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

- **BurstGPT**: MIT License. Wang et al., "BurstGPT: A Real-World Workload Dataset
  for LLM Serving Systems," arXiv 2401.17644, SIGMETRICS 2025.
- **ShareGPT**: Community-collected conversational data. Check the original
  distribution source for current license terms before redistribution.
