# Data Release Policy

Date: 2026-08-24  
Scope: local evidence only (no internet lookups in this pass).

## Summary categories

### INCLUDED (default public tree)

Project-generated / metadata-only materials that support reproduction without
redistributing third-party raw traces:

| Material | Local path(s) | Notes |
|---|---|---|
| Corpus build metadata | `data/public_trace_corpus_v1/manifest.json`, `schema.json`, `distribution_stats.json`, `source_coverage.csv` | Safe metadata; may contain absolute build paths as provenance |
| Public-trace replay manifests | `experiments/public_trace_replay_v1/` layer manifests / provenance JSON | Derived experiment descriptors |
| Synthetic Family A/B/C / joint workloads | Generators + frozen summaries under `experiments/` | Project-generated |
| Unified utility matrix | `experiments/unified_utility_matrix_v2/` | Project-generated |
| Real-vLLM request manifests (project-built) | under `experiments/real_vllm_mechanism_validation_v1/` | Project-generated for local Qwen2.5-0.5B probes |
| Data directory README | `data/README.md` | Documents BurstGPT MIT claim + ShareGPT caution |

### EXCLUDED; DOWNLOAD FROM SOURCE

| Material | Local path(s) | Why |
|---|---|---|
| Raw BurstGPT CSV | `data/raw/burstgpt/BurstGPT_1.csv` | Third-party raw; obtain from upstream; local `data/README.md` states MIT but release policy still prefers source download for raw files |
| Raw Azure 2023 CSVs | `data/raw/azure/AzureLLMInferenceTrace_{conv,code}_2023.csv` | Third-party raw; do not ship in default public tree |
| Normalized parquet tables | `data/public_trace_corpus_v1/*/records.parquet` | Bulky derived tables; rebuild/download policy preferred until author approves redistribution |
| External validation metadata dumps | `data/public_trace_corpus_v1/external_validation_metadata/` | Gitignored / staging |
| ShareGPT / WildChat / other raw trees | under `data/raw/` | Not used in final manuscript; do not redistribute |
| HF / model caches | `hf_cache/`, `hf_datasets_cache/`, `.local_data/` | Caches |

### AUTHOR_EXTERNAL_LICENSE_CHECK_REQUIRED

Local evidence is **insufficient to clear redistribution** of these without
author confirmation against upstream terms:

| Source | Local license evidence | Included raw in repo? | Recommended action |
|---|---|---|---|
| **BurstGPT** | `data/README.md` asserts MIT; provenance audit cites GitHub/DOI | Yes locally under `data/raw/` (gitignored from commits typically) | Confirm MIT redistribution of raw CSV vs metadata-only; default public tree excludes raw |
| **Azure LLM Inference Trace 2023 (conv/code)** | `docs/PROJECT_MAP.md` asserts CC-BY-4.0; provenance audit cites AzurePublicDataset URL | Yes locally under `data/raw/azure/` | Confirm CC-BY-4.0 attribution/redistribution for raw CSV and any derived parquet before shipping |
| **Azure 2024 traces** | Manifest marks `REFERENCE_ONLY_THIS_PASS` | Downloader exists; not manuscript-used | Keep excluded |
| **AgentPerfBench** | Manifest classifies `REAL_SYSTEM_VALIDATION_SOURCE` with `source_license: Apache-2.0` in manifest fragment | Not ingested as workload corpus | Keep excluded from manuscript package |
| **ShareGPT** | `data/README.md`: check original terms before redistribution | May exist under `data/raw/sharegpt/` | Exclude; not final-paper source |
| **`datasets/` Family-A oracle/relabel trees** | Project-generated but large | Untracked local | Author decide include vs regenerate instructions |
| **`artifacts/` Wulver bundles** | Cluster transfer packages | Untracked local | Author decide |

## Manuscript-facing statement (safe wording)

Public-trace replay uses BurstGPT and Azure 2023 LLM-inference traces, with
project-generated SLO/priority/prediction overlays where needed. The default
public release ships **manifests and provenance**, not raw third-party CSVs.
Users should download raw traces from the upstream sources documented in
`docs/current/llm2026_dataset_provenance_audit_20260824.md` and `data/README.md`.

## Decision rule for release builders

1. Never copy `data/raw/**` into the public tree unless the author explicitly
   checks the corresponding license box and chooses redistribution.
2. Prefer metadata + download scripts + frozen experiment summaries.
3. If unsure → `AUTHOR_EXTERNAL_LICENSE_CHECK_REQUIRED` and EXCLUDE.
