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
1/2 candidates audited below are publicly reachable.

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

## Code-level inventory (pre-existing + this branch)

| Dataset | Loader | Converter | Schema | Raw location | Processed location | Tests | Limitations |
|---|---|---|---|---|---|---|---|
| BurstGPT | `workloads/burstgpt.py` | `scripts/convert_burstgpt.py` | Timestamp + token cols; optional Session/Model/Log Type | `data/raw/burstgpt/` or cluster overnight raw | `data/processed/burstgpt/` or cluster processed | `tests/test_burstgpt_loader.py`, session fixture tests | Historical 10k subsets; SLO synthesized |
| Azure 2023/2024 | `workloads/azure.py` (new) + `scripts/data/convert_azure_llm_trace.py` | same | TIMESTAMP, ContextTokens, GeneratedTokens | `data/raw/azure/` | `data/processed/azure/` | azure tiny fixture | **No function-calling subset**; SLO synthesized |
| Bailian/Qwen | `workloads/bailian.py` | `scripts/data/convert_bailian_trace.py` | JSONL + hash_ids | external / LFS | processed on cluster | `bailian_tiny.jsonl` | Two-hour samples; Git LFS |
| Mooncake | `workloads/mooncake.py` | `scripts/data/convert_mooncake_trace.py` | timestamp, input/output, hash_ids | Mooncake repo traces | processed on cluster | `mooncake_tiny.jsonl` | Label `synthetic_trace` separately |
| ShareGPT | `workloads/sharegpt.py` | `scripts/convert_sharegpt.py` | conversations | manual JSON | processed | `test_sharegpt_loader.py` | Not a serving trace |
| WildChat / LMSYS / LongBench | `workloads/prompt_corpora.py` | length adapter only | corpus-specific | HF hub | numeric lengths only | prompt-corpus tests | Not serving traces; LMSYS gated |
| SwissAI | external cluster scripts only | external | `total_buckets` proxy | cluster staging | cluster | none in-repo | Tokens mostly missing |
| TraceLab | external cluster scripts only | external | agent session tokens | cluster staging | cluster | none in-repo | No new loader this task |
| Synthetic | `workloads/synthetic.py` + `augmentation.py` | generators | Request schema | n/a | configs | workload tests | Not real traffic |

---

## Access matrix (2026-07-24)

| Dataset | Type | Official source verified | Access | License | Sample inspected | Integration status |
|---|---|---|---|---|---|---|
| BurstGPT | true serving trace | yes (HPMLL/BurstGPT v2.0) | PUBLIC | CC-BY-4.0 | header/partial | extended loader |
| Azure 2023 | true serving trace | yes | PUBLIC | CC-BY | 500-row samples | converter ready |
| Azure 2024 | true serving trace | yes | PUBLIC | CC-BY | metadata | download+convert ready |
| Bailian/Qwen | true serving trace | yes | PUBLIC | Apache-2.0 | ~89 rows | converter ready |
| Mooncake | true serving trace (+ synthetic split) | yes | PUBLIC | Apache-2.0 | ~181 rows | converter ready |
| TraceLab | true serving / agent | yes (prior docs + GitHub) | PUBLIC | CC-BY-4.0 traces | no (reuse cluster) | external staging |
| SwissAI | true serving (limited tokens) | cluster only | UNAVAILABLE locally | unknown in-repo | no | external only |
| LMSYS-Chat-1M | prompt/conversation | yes | GATED_ACCESS_NOT_GRANTED | LMSYS agreement (no redistribution) | no | adapter ready |
| WildChat-1M | prompt/conversation | yes | PUBLIC | ODC-BY | 20 streamed rows | length adapter |
| LongBench | benchmark prompt | yes | PUBLIC | check GitHub | metadata (`data.zip` ~109MB) | length adapter |
| ShareGPT | prompt/conversation | yes | PUBLIC | verify CC-BY | fixture | existing loader |

---

## Key audit findings

### BurstGPT
- Official repo is **HPMLL/BurstGPT** (not the older HKUDS mirror references).
- Release `v2.0` assets: ~52MB + ~145MB + ~232MB (raw with fails); cleaned
  counterparts similar.
- `BurstGPT_1/2` headers: Timestamp, Model, Request tokens, Response tokens,
  Total tokens, Log Type.
- `BurstGPT_3` adds Session ID and Elapsed time.
- Existing processed 10k subsets are compatible with the core three-column
  converter; full-trace work should prefer `without_fails_*` and chronological
  day/night splits. Natural diurnal structure exists across multi-month spans.

### Azure
- Official 2023 subsets: **code** and **conversation** only.
- Historical plan mentions of a function-calling subset are **not** supported by
  the published AzurePublicDataset files (function* filenames 404).
- 2024 one-week code (~692MB) and conv (~1.08GB) share the same schema.
- Existing `convert_azure_llm_trace.py` remains valid; `workloads/azure.py`
  adds canonical provenance metadata.

### Bailian / Qwen
- Verifiably public: `alibaba-edu/qwen-bailian-usagetraces-anon`.
- Classification: **accessible** (corrects older “maybe request-only” docs).
- Git LFS; raw JSONL sizes ~28–132MB per split.
- Genuine relative serving timestamps, input/output lengths, sessions, request
  types, and prefix-hash blocks.

### Mooncake
- Independent production-derived traces with prefix hashes.
- `synthetic_trace.jsonl` must be labeled synthetic, not natural replay.

### LMSYS vs WildChat
- LMSYS: gated; this account cannot load data shards yet. License forbids
  redistribution.
- WildChat: public ODC-BY; preferable while LMSYS access is blocked. Both are
  **conversation corpora**, not serving traces. Conversation timestamps ≠
  system arrivals.

### LongBench
- Long-context **benchmark** corpus (`data.zip` ~109MB). Use with
  trace-calibrated or synthetic arrivals only.

---

## Integrity rules for real traces

1. Retain original request order and inter-arrival timing unless
   `--time-scale` is set and recorded.
2. Chronological train/validation/test splits only.
3. Never expose `actual_output_tokens` to policies.
4. Do not silently synthesize SLOs/priorities without provenance.
5. Do not commit downloaded conversation text or multi-GB shards.

---

## Storage estimates (Wolverine)

Root: `/mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/`

| Dataset | Raw | Extracted | Processed JSONL (order-of-mag) | Tokenization cache | Parallel preprocess? |
|---|---|---|---|---|---|
| BurstGPT full cleaned | ~410MB | same | ~0.5–1.5GB | n/a | yes (per file) |
| Azure 2023 | <50MB | same | <200MB | n/a | no need |
| Azure 2024 | ~1.8GB | same | ~2–4GB | n/a | yes (per split) |
| Bailian all splits | ~312MB LFS | same | ~0.5–1GB | n/a | yes (per split) |
| Mooncake traces | hundreds of MB (inspect) | same | similar | n/a | yes |
| WildChat-1M | multi-GB parquet | same | lengths-only much smaller | optional tokenizer | yes streaming |
| LongBench | ~109MB zip | larger | lengths JSON small | optional | low |

---

## Recommended tiers

**Tier 1 — primary real serving traces**

1. BurstGPT full (`without_fails_1/2/3`)
2. Azure 2023 code + conversation (already core)
3. Bailian/Qwen anonymous traces
4. Mooncake conversation + toolagent (not synthetic)

**Tier 2 — specialized real / application-derived**

1. Azure 2024 one-week (temporal OOD)
2. TraceLab (cluster staging; agentic/prefix)
3. SwissAI (cluster staging; KV proxy only — disclose limits)

**Tier 3 — prompt/benchmark corpora for trace-calibrated workloads**

1. WildChat-1M length pools
2. LongBench long-context stress
3. ShareGPT (legacy compatibility)
4. LMSYS-Chat-1M — after gated access + license review

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

# BurstGPT (example)
cd /mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/raw/burstgpt
# download release assets via gh or browser_download_url, then:
sha256sum BurstGPT_without_fails_*.csv | tee ../../checksums/burstgpt/sha256.txt

# Bailian via LFS media URLs or git lfs pull
# Mooncake from FAST25-release/traces
# Azure 2024:
python scripts/data/download_azure_llm_2024.py \
  --output-dir /mmfs1/project/ikoutis/sv96/llmserveopt-data/datasets/raw/azure

# Preprocess examples (Slurm wrappers recommended)
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

# Copy/update manifests from docs/current/real_dataset_expansion_status.json
```

---

## Risks and unresolved decisions

- LMSYS gated access and redistribution ban.
- LongBench license not declared on the HF card — confirm on GitHub before
  redistribution.
- Mooncake timestamp numeric units need confirmation in manifests.
- SwissAI token reconstruction remains proxy-only.
- Output-length leakage risk if any future loader bypasses `ObservableRequest`.
- Historical docs mentioning Bailian as unavailable are superseded by the
  public Alibaba Edu repository verified here.
- Azure function-calling subset does not exist publicly.

Do not push this branch until the user explicitly requests it.
