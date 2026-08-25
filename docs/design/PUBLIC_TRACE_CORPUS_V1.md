# Public Trace Corpus v1 — Design & Preregistration

**Date:** 2026-08-19
**Branch:** `contextual-compositional-heuristics-20260731`
**Status:** PREREGISTERED (frozen before the long-running build)
**Predecessor context:** `docs/dataset_workload_decision.md` (Phase 2B.9),
`docs/selector_dataset_v2.md` §"Source Acquisition Plan",
`docs/current/REAL_DATASET_EXPANSION_STATUS.md`

## 0. Purpose (frozen)

We are constructing a **workload-input corpus**, not policy labels.

This document scopes the reuse of PUBLIC, real-world LLM-serving workload
traces (arrivals, prompt/output token counts, timing, session/context
metadata) as raw material for a future policy-separation annotation layer.
It explicitly does **not**:

- evaluate any of the six/twenty-seven deployable policies on these workloads,
- create oracle labels, policy winners, or counterfactual action labels,
- synthesize new policies,
- run held-out scientific replication,
- tune router models,
- use Cohere, CloudRift, or any paid/commercial LLM API.

Those are reserved for later, separately-preregistered phases (see §11,
Layers 3-6).

## 1. Existing state — do not re-derive

**This is not a cold start.** Prior phases already built substantial public
real-trace infrastructure. This design must extend, consolidate, and
audit-fill gaps in that infrastructure, not duplicate it.

Already implemented (`src/llmserveopt/workloads/`):

| Module | Source | Status |
|---|---|---|
| `canonical_schema.py` | N/A (schema itself) | `CanonicalIngestRecord`, `FieldProvenance` (OBSERVED/DERIVED/SYNTHESIZED/UNAVAILABLE), `DatasetType`, `ReplayLabel` already exist and are the canonical intermediate representation used by every converter below |
| `burstgpt.py` | BurstGPT | Full raw CSV present at `data/raw/burstgpt/BurstGPT_1.csv` (1,429,737 rows) |
| `azure.py` | Azure LLM Inference 2023 + 2024 (Splitwise/DynamoLLM) | 2023 code+conv raw CSVs present at `data/raw/azure/`; 2024 not locally present (see §6) |
| `bailian.py` | Qwen/Bailian anonymized traces | Converter exists; raw not locally present |
| `mooncake.py` | Mooncake/Kimi FAST'25 traces | Converter exists; raw not locally present; `EVALUATION_ROLE=INTERNAL_OOD_ONLY`, **redistribution prohibited until license clarified** — do not add to a distributable corpus |
| `sharegpt.py` | ShareGPT | Loader exists; raw not present (license gating retained from Phase 2B.9) |

Already audited as system artifacts (`docs/audits/`):

| Source | Audit | License | Role |
|---|---|---|---|
| Sarathi-Serve | `sarathi_official_artifact_audit_20260805.md` | Apache-2.0 (code) | POLICY_IMPLEMENTATION_SOURCE; paper's own OSDI traces are per-experiment, not a standalone redistributable corpus |
| DistServe | `distserve_existing_implementation_audit_20260806.md` | Apache-2.0 (confirmed via GitHub API this pass; LLMServe/DistServe) | POLICY_IMPLEMENTATION_SOURCE |
| Llumnix | `llumnix_official_artifact_audit_20260806.md` | Apache-2.0 (confirmed; canonical repo now `llumnix-project/llumnix`, OSDI artifact `alibaba/llm-scheduling-artifact`) | POLICY_IMPLEMENTATION_SOURCE + REAL_SYSTEM_VALIDATION_SOURCE (production-style traces used as baselines, not released as a standalone dataset) |

Already flagged as a known gap, unresolved before this document:

- **"SwissAI"** appears in `docs/result_claims.md`, `docs/slai_faithful_scheduler_reference.md`,
  and `docs/audits/swissai_v2_policy_sweep_reanalysis_20260809.md` as an
  in-repo workload family with `feat_swiss_kv_proxy_p95` / `feat_swiss_high_reuse_fraction`
  / `feat_swiss_low_reuse_fraction` features and a completed 512-window × 27-policy
  sweep. **No repository, HuggingFace dataset, or paper named "SwissAI serving
  trace" exists** (verified this pass: GitHub search for `swissai` returns 0
  hits; the `swiss-ai` GitHub org — makers of the Apertus LLM — has 75 public
  repos, none of them a serving trace; the reanalysis audit itself says
  "Bucket-reuse output lengths and SLOs are partly reconstructed or
  synthetic"). **Verdict: `SwissAI` is an internal project codename for an
  HPC-side (Wulver/Wolverine) synthetic KV/cache/reuse workload family, not a
  distinct public trace dataset.** It is out of scope for this corpus and
  must not be listed as a Tier-1 public source in any future document without
  a corrected citation. See §5 for the formal classification entry.

Local real-trace files currently on disk (`data/raw/`):

```
data/raw/burstgpt/BurstGPT_1.csv        1,429,737 rows  (sha256 in data/raw/burstgpt/checksums.sha256)
data/raw/azure/AzureLLMInferenceTrace_code_2023.csv
data/raw/azure/AzureLLMInferenceTrace_conv_2023.csv
```

## 2. What is genuinely new in this task

1. Confirm/close the SwissAI ambiguity (§1, done above).
2. Audit **AgentPerfBench** (Tier 2) — not previously investigated in this
   repo. Result (verified this pass against the live HF dataset file, not
   just its card prose, see §5): real, legitimate artifact — but every one of
   its four configs (`trace_replay`, `synthetic_distributional`,
   `per_layer_kernel`, `mse_validation`) is a **run-level aggregate
   performance summary table** (request_throughput, TTFT/TPOT/ITL/E2E
   latency percentiles measured on real vLLM 0.19.0/SGLang 0.5.9 GPUs), not
   per-request prompt/output-token rows. The dataset card's prose
   ("Replays exact ISL/OSL sequences from recorded agent sessions")
   describes the *methodology* used to generate the underlying workload, not
   a published per-request artifact — no `isl`/`osl`/token-count column
   exists in any config. **Reclassified from an initial TRACE_SOURCE guess
   to REAL_SYSTEM_VALIDATION_SOURCE only; not ingested into the
   workload-input corpus** (see §5, §9).
3. Audit **JITServe** and **PEACE** (Tier 3) — not previously investigated.
   Result: no discoverable public code/data artifact under either name;
   classified `INSUFFICIENT_EVIDENCE` / `NOT_USEFUL_FOR_CORPUS` (§5, §8).
4. Build the **unified canonical corpus artifact** — `data/public_trace_corpus_v1/`
   with `manifest.json`, `schema.json`, `source_coverage.csv`, and
   per-source Parquet partitions. No such consolidated, checksummed,
   provenance-complete corpus currently exists; each source has its own
   converter but no unified output layer.
5. Re-verify BurstGPT's license: `external/datasets/burstgpt.md` claims
   **MIT**; GitHub's license-detection API on the current authoritative
   repo (`HPMLL/BurstGPT`, the maintained successor to `HKUDS/BurstGPT`,
   which now 404s / has moved) reports **CC-BY-4.0**. This discrepancy is
   flagged, not silently resolved — see §5. Practically the two licenses
   agree on the operative point (attribution-friendly reuse permitted); the
   corpus records both claims and treats CC-BY-4.0 as authoritative pending
   a manual check of the dataset card/paper text, since it comes from the
   live repository rather than a possibly-stale local note.

## 3. Candidate source tiers

### Tier 1 — highest priority

| Source | Status this pass |
|---|---|
| BurstGPT | Already local; re-verify license (§2.5) |
| Azure LLM Inference traces (2023, 2024) | 2023 already local; 2024 not local — small fetch attempted in this build (§6) |
| ~~SwissAI serving trace~~ | **Rejected** — not a real distinct public source (§1) |

### Tier 2 — benchmark/real-engine sources

| Source | Status this pass |
|---|---|
| AgentPerfBench | New — accepted, see §5 |

### Tier 3 — system artifacts for workload/config reuse

| Source | Status this pass |
|---|---|
| Sarathi-Serve | Already audited; POLICY_IMPLEMENTATION_SOURCE, not a trace source |
| JITServe | New — no discoverable artifact; `INSUFFICIENT_EVIDENCE` |
| PEACE | New — no discoverable artifact; `INSUFFICIENT_EVIDENCE` |
| DistServe | Already audited; POLICY_IMPLEMENTATION_SOURCE |
| Llumnix | Already audited; POLICY_IMPLEMENTATION_SOURCE + REAL_SYSTEM_VALIDATION_SOURCE |

No additional sources are added beyond this list in this pass, per the
"do not add dozens of arbitrary datasets" instruction. Sources adjacent to
this list that already exist in-repo (Bailian, Mooncake, TraceLab, ServeGen,
ShareGPT, WildChat) are treated as **already-classified reference context**
in §5's table but are not re-audited here; their existing classification and
license notes in `docs/selector_dataset_v2.md` §"Source Acquisition Plan"
stand.

## 4. Canonical raw-workload schema (v1)

Extends `src/llmserveopt/workloads/canonical_schema.py` at the
**corpus/provenance layer** rather than replacing it. `CanonicalIngestRecord`
remains the per-request payload; this corpus adds a wrapping
identity/provenance/license record per source, defined in
`data/public_trace_corpus_v1/schema.json` (§9), covering:

- IDENTITY/PROVENANCE: `source_dataset`, `source_version`, `source_record_id`,
  `source_url_or_repo`, `source_license`, `source_file_sha256`
- REQUEST TIMING: `arrival_timestamp`, `interarrival_time`,
  `relative_arrival_time`
- REQUEST SIZE: `prompt_tokens`, `output_tokens`, `total_tokens` (derived
  where not native)
- MODEL CONTEXT: `model_name`, `model_family`, generation parameters (only
  where the source discloses them — none of the accepted Tier-1/2 sources do)
- WORKLOAD CONTEXT: `session_id`, `request_type`, concurrency/load metadata,
  prefix/cache-reuse identifiers (native only for Bailian/Mooncake/
  AgentPerfBench-style sources; BurstGPT/Azure have none)
- OPTIONAL SLO/PRIORITY: `priority`, `deadline`, `slo`, `tenant_class` — none
  of the accepted public sources natively provide these; they remain
  `UNAVAILABLE` in this corpus (SLO/priority synthesis, if ever performed, is
  a downstream simulator-input concern per `docs/data_field_provenance.md`,
  out of scope here)
- SOURCE-SPECIFIC: preserved under a namespaced `extra` map, never dropped
- AUDIT FIELDS: `field_provenance` map (NATIVE / DETERMINISTIC_DERIVED /
  SOURCE_SPECIFIC / UNAVAILABLE) per canonical field, per source

No field is fabricated. Missing fields are `null` plus an explicit
`UNAVAILABLE` provenance tag — the same invariant `canonical_schema.py`
already enforces via `validate_canonical_record`.

## 5. License / redistribution + role classification matrix

| Source | Authoritative URL | License | Redistribution class | Role(s) | Ingest this pass? |
|---|---|---|---|---|---|
| BurstGPT | `github.com/HPMLL/BurstGPT` (successor to `HKUDS/BurstGPT`, which now redirects/404s) | CC-BY-4.0 (GitHub API, this pass) vs. MIT (stale local note) — CC-BY-4.0 treated as authoritative | REDISTRIBUTABLE (attribution required) | TRACE_SOURCE | Yes — already local |
| Azure LLM Inference 2023 | `github.com/Azure/AzurePublicDataset` | CC-BY-4.0 | REDISTRIBUTABLE (attribution required) | TRACE_SOURCE | Yes — already local |
| Azure LLM Inference 2024 | `github.com/Azure/AzurePublicDataset` (`AzureLLMInferenceDataset2024.md`) | CC-BY-4.0 | REDISTRIBUTABLE | TRACE_SOURCE | Attempt small fetch (§6); reference-only if large |
| AgentPerfBench | code: `github.com/booth-algo/AgentPerfBench` (MIT); data: `huggingface.co/datasets/agent-perf-bench/AgentPerfBench` (Apache-2.0) | Apache-2.0 (data) | REDISTRIBUTABLE | REAL_SYSTEM_VALIDATION_SOURCE only (`trace_replay` config: 2,932 run-level rows of aggregate throughput/TTFT/TPOT/ITL/E2E-latency, measured replaying real SWE-Bench/TerminalBench agent-session workloads on vLLM 0.19.0/SGLang 0.5.9 across 9 models × 14 GPU configs; **no per-request prompt/output-token column in any config** — verified against the live parquet file, not just the card prose) | Metadata inspected/hashed, not ingested as workload input (small parquet, downloaded for the record into `external_validation_metadata/`) |
| Sarathi-Serve | `github.com/microsoft/sarathi-serve` | Apache-2.0 | REDISTRIBUTABLE (code); traces are per-experiment OSDI artifacts, not a standalone release | POLICY_IMPLEMENTATION_SOURCE | No (not a trace source) |
| DistServe | `github.com/LLMServe/DistServe` | Apache-2.0 | REDISTRIBUTABLE (code) | POLICY_IMPLEMENTATION_SOURCE | No |
| Llumnix | `github.com/llumnix-project/llumnix`; OSDI artifact `github.com/alibaba/llm-scheduling-artifact` | Apache-2.0 | REDISTRIBUTABLE (code) | POLICY_IMPLEMENTATION_SOURCE + REAL_SYSTEM_VALIDATION_SOURCE (paper tables only, not a raw trace release) | No |
| JITServe | none discoverable (0 GitHub search hits for the name; not previously implemented per `docs/audits/project_pause_reconciliation_query1_20260806.md`) | N/A | LICENSE_UNCLEAR / no artifact | NOT_USEFUL_FOR_CORPUS | No — `INSUFFICIENT_EVIDENCE` |
| PEACE | none discoverable (0 GitHub search hits for LLM-serving-context "PEACE") | N/A | N/A | NOT_USEFUL_FOR_CORPUS | No — `INSUFFICIENT_EVIDENCE` |
| SwissAI serving trace | N/A — does not exist as a distinct public dataset (§1) | N/A | N/A | NOT_USEFUL_FOR_CORPUS | No — rejected |
| Bailian/Qwen (reference) | `github.com/alibaba-edu/qwen-bailian-usagetraces-anon` | Apache-2.0 | REDISTRIBUTABLE | TRACE_SOURCE | Not re-audited; not ingested this pass (raw not local, outside this pass's Tier scope) |
| Mooncake/Kimi (reference) | `github.com/kvcache-ai/Mooncake` | Apache-2.0 (code); data license unspecified per in-repo audit | DERIVATIVE_ONLY (per existing `EVALUATION_ROLE=INTERNAL_OOD_ONLY`) | TRACE_SOURCE | No — existing internal-only restriction stands |
| TraceLab (reference) | `github.com/uw-syfi/TraceLab` | Apache-2.0 (code + data per repo) | REDISTRIBUTABLE | TRACE_SOURCE (coding-agent sessions) | Not ingested this pass (outside stated Tier scope; noted for a future pass) |
| ServeGen (reference) | `github.com/alibaba/ServeGen` | Apache-2.0 | REDISTRIBUTABLE (generator code, not raw traces) | WORKLOAD_GENERATOR | Not ingested (generator, not a trace) |

## 6. Download plan for this build

- BurstGPT: already local, re-hash only.
- Azure 2023 code + conv: already local, re-hash only.
- Azure 2024: **already has a downloader** at
  `scripts/data/download_azure_llm_2024.py` (found this pass; previously
  overlooked in the initial state check), explicitly documented there as
  "large files — user-initiated only" (code ~692 MB, conv ~1.1 GB from
  `Azure/AzurePublicDataset` release `dataset-llm-2024`). This build honors
  that existing repo convention: it records the downloader path and known
  sizes as `REFERENCE_ONLY_USER_INITIATED` in the manifest and does **not**
  auto-fetch ~1.8 GB in an unattended background job. A human should run the
  existing script deliberately in a future pass.
- AgentPerfBench: fetch `trace_replay/summary.parquet` (and the paired
  main-sweep table, if separately small) directly via the HF `resolve/main`
  URL. This is Apache-2.0 and explicitly small.
- No other downloads. Bailian/Mooncake/TraceLab/ServeGen remain
  reference-only pointers in this pass.

## 7. Layer architecture (target; only Layers 0-1 built this pass)

```
LAYER 0  immutable source provenance        <- built this pass (manifest.json, checksums)
LAYER 1  normalized public workload inputs  <- built this pass (canonical parquet per source)
LAYER 2  canonical scenarios/replay manifests            <- future
LAYER 3  multi-policy execution results                  <- future
LAYER 4  step-level trajectory/action data                <- future
LAYER 5  counterfactual/decision-criticality annotations   <- future
LAYER 6  small real-LLM validation subset (Cohere/CloudRift permitted here only) <- future
```

## 8. Novelty recheck (preliminary; formalized after the build in §16/§18
of the task instructions)

`RAW_INPUTS_EXIST_BUT_POLICY_SEPARATION_LAYER_MISSING` — every accepted
source here provides workload inputs and, for AgentPerfBench and
Llumnix/Sarathi/DistServe, some real-system performance metadata, but none
provide the same scenario replayed across multiple scheduler policies with
comparable regret/utility and causal disagreement annotations. This
confirms rather than assumes the conclusion the task instructions expect;
it will be re-verified against actual schema coverage after ingestion.

## 9. Deliverables of this pass

- `docs/design/PUBLIC_TRACE_CORPUS_V1.md` (this file)
- `data/public_trace_corpus_v1/schema.json` — canonical corpus schema
- `data/public_trace_corpus_v1/manifest.json` — built by the corpus builder
- `data/public_trace_corpus_v1/source_coverage.csv` — schema-coverage matrix
- `scripts/build_public_trace_corpus_v1.py` — deterministic, streaming build
- `src/llmserveopt/workloads/public_trace_corpus.py` — thin adapter layer
  reusing `burstgpt.py` / `azure.py` plus an AgentPerfBench inspector that
  correctly routes it to `external_validation_metadata/` instead of the
  workload-input table (§2, §5)
- Tests under `tests/test_public_trace_corpus_v1.py`
