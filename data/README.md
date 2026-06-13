# Data Directory

## Structure

- `raw/` — unmodified downloaded datasets as obtained from original sources
- `processed/` — JSONL files in the simulator's canonical schema (one Request per line)

## Version Control

Neither `raw/` nor `processed/` is version-controlled by default (see `.gitignore`).
Only `.gitkeep` placeholder files are committed.

## Download Scripts

Download scripts are in `scripts/`:
- `scripts/download_burstgpt.py` — downloads BurstGPT from HuggingFace
- ShareGPT must be downloaded manually (see `external/datasets/sharegpt.md`)

## Conversion Scripts

- `scripts/convert_burstgpt.py` — converts raw BurstGPT CSV to JSONL
- `scripts/convert_sharegpt.py` — converts raw ShareGPT JSON to JSONL

## Provenance Documentation

Dataset provenance and field descriptions are in `external/datasets/`.
