# Reality-Grounded Dataset Expansion Status

**Date:** 2026-07-24
**Branch:** `reality-grounded-dataset-expansion-20260724`
**Machine-readable companion:** `docs/current/real_dataset_expansion_status.json`
**Local staging root (outside git):** `~/llmserveopt-dataset-staging`
**Proposed Wolverine root:** `/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/`

This document records dataset discovery, access/licensing verification, schema
inspection, bounded sample ingestion, converter/loader preparation, and
provenance rules for the next phase of reality-grounded workload expansion.

It does **not** claim that conversation corpora are production serving traces.

---

## Overall status

`REAL_DATASET_EXPANSION_STATUS = PARTIALLY_READY_WITH_ACCESS_GAPS`

Hugging Face authentication is available for user `SoroushVahidi`. LMSYS-Chat-1M
data access is still gated (terms not granted for this account). All other Tier
1 candidates audited below are publicly reachable.

---

## Taxonomy (do not collapse)

| Class | Meaning |
|---|---|
| True serving trace | System-level request arrivals with prompt/output token lengths |
| Prompt/conversation corpus | Chat text or turns; arrivals must be synthesized or borrowed |
| Benchmark prompt corpus | Evaluation tasks (e.g. LongBench); not traffic |
| Synthetic / trace-calibrated | Generated arrivals and/or lengths, possibly fit to real stats |
| Aggregate statistics only | Summaries without per-request rows |

---

## Canonical ingestion schema

Module: `src/llmserveopt/workloads/canonical_schema.py`

Core fields: `request_id`, `arrival_time`, `prompt_tokens`,
`actual_output_tokens`, `predicted_output_tokens`, `slo_deadline`, `priority`,
`class_id`.

Optional: `session_id`, `tenant_id`, `model_id`, `prefix_id`, `source_dataset`,
`source_split`, `source_record_id`.

Every field carries provenance: `observed` | `derived` | `synthesized` |
`unavailable`. SLO/priority synthesis must be disclosed. Scheduler-visible
fields exclude `actual_output_tokens` (`OBSERVABLE_REQUEST_FIELDS` /
`ObservableRequest`).

Time scaling uses explicit `--time-scale`. Labels:

- `time_scale == 1.0` → `natural_trace_replay`
- otherwise → `trace-derived, time-scaled`

---

## License matrix (code repo vs data)

| Dataset | Code-repo license | Data / trace license | Explicit data license? | Redistribution of raw/processed | Attribution required | Official pin |
|---|---|---|---|---|---|---|
| BurstGPT | CC-BY-4.0 (`LICENSE`) | CC-BY-4.0 | yes | allowed with attribution | yes | release **v2.0** |
| Azure 2023 | CC-BY-4.0 | CC-BY Attribution (dataset page + LICENSE) | yes | allowed with attribution | yes | AzurePublicDataset `AzureLLMInferenceDataset2023.md` |
| Azure 2024 | CC-BY-4.0 | CC-BY Attribution | yes | allowed with attribution | yes | release `dataset-llm-2024` |
| Azure 2025 LMM | CC-BY-4.0 | CC-BY Attribution | yes | allowed with attribution | yes | `AzureLMMInferenceDataset2025.md` |
| Bailian/Qwen | Apache-2.0 | Apache-2.0 (README §License + LICENSE) | yes | allowed under Apache-2.0 | preserve notices + cite paper | commit `5f7439c51ec2…` (main tip 2026-07-24) |
| Mooncake | Apache-2.0 (`LICENSE-APACHE`) | **NOT_EXPLICITLY_SPECIFIED** (no dedicated dataset-license notice beyond project Apache file) | no (separate) | treat cautiously; cite FAST'25 / arXiv | cite papers | FAST25-release traces README |
| WildChat-1M | n/a (HF) | ODC-BY | yes | per ODC-BY; do not commit conversations | yes | HF revision pin |
| LMSYS-Chat-1M | n/a (HF) | LMSYS gated agreement | yes (gated) | **prohibited** to third parties | per agreement | gated; access not granted |
| LongBench | check GitHub | not declared on HF card | unclear on card | verify before redistribute | cite paper | HF `data.zip` |
| TraceLab | Apache-2.0 (code) | CC-BY-4.0 (public traces; prior docs) | yes (per prior docs) | with attribution | yes | uw-syfi/TraceLab |
| SwissAI | unknown in-repo | unknown in-repo | no | unknown | unknown | cluster staging only |

---

## Code-level inventory

| Dataset | Loader | Converter | Schema | Raw location | Processed location | Tests | Limitations |
|---|---|---|---|---|---|---|---|
| BurstGPT | `workloads/burstgpt.py` | `scripts/convert_burstgpt.py` | Timestamp + tokens; optional Session/Model/Log Type/Elapsed | external / cluster | processed on cluster | burstgpt + session fixtures | full CSV loaded via pandas; document memory |
| Azure 2023/2024 | `workloads/azure.py` + `scripts/data/convert_azure_llm_trace.py` | same | TIMESTAMP, ContextTokens, GeneratedTokens | `data/raw/azure/` | `data/processed/azure/` | azure tiny fixture | no function-calling subset |
| Bailian/Qwen | `workloads/bailian.py` | `scripts/data/convert_bailian_trace.py` | JSONL + hash_ids (16-token blocks) | Git LFS | processed on cluster | bailian fixture | two-hour samples |
| Mooncake | `workloads/mooncake.py` | `scripts/data/convert_mooncake_trace.py` | timestamp(**ms**), lengths, hash_ids (512-token blocks) | FAST25-release | processed on cluster | mooncake fixture | DATA_LICENSE not separately specified; synthetic split separate |
| ShareGPT | `workloads/sharegpt.py` | `scripts/convert_sharegpt.py` | conversations | manual JSON | processed | sharegpt tests | not a serving trace |
| WildChat / LMSYS / LongBench | `workloads/prompt_corpora.py` | length adapter | corpus-specific | HF hub | numeric lengths only | prompt-corpus tests | not serving traces; LMSYS gated |
| SwissAI | external only | external | total_buckets proxy | cluster | cluster | none in-repo | tokens mostly missing |
| TraceLab | external only | external | agent session tokens | cluster | cluster | none in-repo | no new loader this task |

---

## Access matrix (2026-07-24)

| Dataset | Type | Official source verified | Access | License summary | Sample inspected | Integration status |
|---|---|---|---|---|---|---|
| BurstGPT | true serving trace | yes (HPMLL/BurstGPT **v2.0**) | PUBLIC | CC-BY-4.0 | header/partial | extended loader |
| Azure 2023 | true serving trace | yes | PUBLIC | CC-BY-4.0 | 500-row samples | converter ready |
| Azure 2024 | true serving trace | yes (DynamoLLM / HPCA 2025) | PUBLIC | CC-BY-4.0 | metadata | download+convert ready |
| Azure 2025 LMM | multimodal serving trace | yes (ModServe / SoCC 2025) | PUBLIC | CC-BY-4.0 | metadata | future candidate only |
| Bailian/Qwen | true serving trace | yes | PUBLIC | Apache-2.0 (data+repo) | ~89 rows | converter ready |
| Mooncake | true serving (+ synthetic split) | yes | PUBLIC | repo Apache-2.0; data NOT_EXPLICITLY_SPECIFIED | ~181 rows | converter ready (ms→s) |
| TraceLab | agent/serving | yes | PUBLIC | CC-BY-4.0 traces | no (reuse cluster) | external staging |
| SwissAI | serving (token-limited) | cluster only | UNAVAILABLE locally | unknown in-repo | no | external only |
| LMSYS-Chat-1M | prompt/conversation | yes | GATED_ACCESS_NOT_GRANTED | gated; no redistribute | no | adapter ready |
| WildChat-1M | prompt/conversation | yes | PUBLIC | ODC-BY | 20 streamed rows | length adapter |
| LongBench | benchmark prompt | yes | PUBLIC | verify GitHub | metadata (`data.zip` ~109MB) | length adapter |
| ShareGPT | prompt/conversation | yes | PUBLIC | verify CC-BY | fixture | existing loader |

---

## Key audit findings

### BurstGPT
- Official repo: **HPMLL/BurstGPT**, release **v2.0**.
- Assets (bytes): `BurstGPT_1.csv` 52283111; `_2` 144819209; `_3` 231682327;
  cleaned counterparts similar (~50–217 MB).
- Schema: Timestamp (seconds from local midnight day-0), Model, Request/Response/Total
  tokens, Log Type; BurstGPT_3 adds Session ID and Elapsed time.
- Prefer `without_fails_*` for full-trace work; chronological day/night splits.

### Azure 2023 / 2024
- 2023 subsets: **code** and **conversation** only (collected 2023-11-11 per docs).
- Schema: TIMESTAMP (ISO wall-clock), ContextTokens, GeneratedTokens.
- **No public function-calling subset** (function* filenames 404).
- 2024: one-week code (~692 MB) + conv (~1.08 GB), same schema; DynamoLLM (HPCA 2025).
- Download script `scripts/data/download_azure_llm_2024.py` covers both splits;
  converter is schema-compatible with 2023.

### Azure 2025 multimodal (future)
- `AzureLMMInferenceDataset2025.md`: Oct 15–22 2024 multimodal cluster sample.
- Schema adds `NumImages`; ContextTokens includes text+image.
- File: `data/AzureLMMInferenceTrace_multimodal.csv.gz`.
- CC-BY. Recorded as Tier 2 future work; **no converter in this branch**.

### Bailian / Qwen
- `alibaba-edu/qwen-bailian-usagetraces-anon` (tip `5f7439c5…`).
- DATA_LICENSE = Apache-2.0 (explicit README §License + LICENSE file).
- Relative **seconds** timestamps; sessions; types; 16-token hash blocks.
- Splits: To-C (A), To-B (B), Thinking, Coder (Git LFS, ~28–132 MB each).

### Mooncake
- FAST'25 traces under `FAST25-release/traces/`; arxiv single-file is historical.
- Real: conversation (12031 req), toolagent (23608); Synthetic (3993, Poisson).
- Official timestamp unit: **milliseconds** (converter derives seconds).
- Prefix blocks: 512 tokens per hash id.
- DATA_LICENSE = NOT_EXPLICITLY_SPECIFIED (separate from repo Apache-2.0).
- `require_real_only=True` refuses synthetic paths/splits.

### Prompt / benchmark corpora
- WildChat, LongBench, ShareGPT, LMSYS are **not** serving-arrival traces.
- LMSYS remains gated and excluded until access is granted.

---

## Integrity rules for real traces

1. Retain original request order and inter-arrival timing unless `--time-scale`
   is set and recorded.
2. Chronological train/validation/test splits only.
3. Never expose `actual_output_tokens` to policies.
4. Do not silently synthesize SLOs/priorities without provenance.
5. Do not commit downloaded conversation text or multi-GB shards.
6. Never mix Mooncake real and synthetic rows without an explicit source-type.

---

## Storage estimates (Wolverine)

Root: `/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/`

| Dataset | Raw | Extracted | Processed JSONL (order-of-mag) | Tokenization cache | Parallel preprocess? |
|---|---|---|---|---|---|
| BurstGPT full cleaned | ~410MB | same | ~0.5–1.5GB | n/a | yes (per file) |
| Azure 2023 | <50MB | same | <200MB | n/a | no need |
| Azure 2024 | ~1.8GB | same | ~2–4GB | n/a | yes (per split) |
| Bailian all splits | ~312MB LFS | same | ~0.5–1GB | n/a | yes (per split) |
| Mooncake FAST25 | inspect on download | same | similar | n/a | yes |
| Azure 2025 LMM | gz multimodal | larger | future | n/a | later |
| WildChat-1M | multi-GB parquet | same | lengths-only smaller | optional | streaming |
| LongBench | ~109MB zip | larger | lengths JSON small | optional | low |

---

## Recommended tiers

**Tier 1 — primary real serving traces**

1. BurstGPT full (`without_fails_1/2/3`, release v2.0)
2. Azure 2023 code + conversation
3. Azure 2024 one-week code + conversation
4. Bailian/Qwen anonymous traces (Apache-2.0 data license documented)
5. Mooncake real production-derived traces (`conversation_trace`, `toolagent_trace`)

**Tier 2 — specialized evidence**

1. TraceLab (cluster staging; agentic/prefix)
2. SwissAI (cluster staging; KV proxy only — disclose limits)
3. Azure 2025 multimodal LMM trace (future; schema differs via NumImages)
4. Mooncake `synthetic_trace.jsonl` (labeled synthetic; keep separate)

**Tier 3 — empirical content and length corpora**

1. WildChat-1M length pools
2. LongBench long-context stress
3. ShareGPT (legacy compatibility)
4. LMSYS-Chat-1M — only after gated access + license review

Tier 3 does **not** supply genuine serving arrival processes.

**Inaccessible / rejected for this phase**

- Azure “function-calling” subset: **not published**
- LMSYS data shards: **GATED_ACCESS_NOT_GRANTED**
- Any NLP accuracy benchmark presented as a serving trace

---

## Wolverine next commands (do not run locally in this task)

```bash
# Directories
mkdir -p /mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/{raw,processed,manifests,checksums}/{burstgpt,azure,bailian,mooncake,wildchat,longbench}

# HF auth: use existing login or HF_TOKEN env — never echo the token
hf auth whoami

# BurstGPT v2.0 release assets → raw/burstgpt, then sha256sum
# Bailian via git lfs / media.githubusercontent.com
# Mooncake FAST25-release/traces (exclude synthetic unless explicitly labeled)
# Azure 2024:
python scripts/data/download_azure_llm_2024.py \
  --output-dir /mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/raw/azure

python scripts/convert_burstgpt.py \
  --input .../BurstGPT_without_fails_1.csv \
  --output .../processed/burstgpt/burstgpt_without_fails_1.jsonl \
  --time-scale 1.0 --seed 17

python scripts/data/convert_bailian_trace.py \
  --input .../qwen_traceA_blksz_16.jsonl \
  --output .../processed/bailian/traceA.jsonl \
  --source-split to_c_traceA --time-scale 1.0

python scripts/data/convert_mooncake_trace.py \
  --input .../conversation_trace.jsonl \
  --output .../processed/mooncake/conversation.jsonl \
  --source-split conversation_trace
```

---

## Risks and unresolved decisions

- LMSYS gated access and redistribution ban.
- LongBench license not declared on the HF card — confirm on GitHub before
  redistribution.
- Mooncake DATA_LICENSE = NOT_EXPLICITLY_SPECIFIED (repo is Apache-2.0).
- SwissAI token reconstruction remains proxy-only.
- BurstGPT full CSV loads into memory via pandas — plan chunked preprocessing
  on Wolverine for multi-million-row files.
- Azure function-calling subset does not exist publicly.

Do not push this branch until the final review checks in this task pass.
