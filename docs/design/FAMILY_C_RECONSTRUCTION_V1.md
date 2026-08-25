# Family C Reconstruction v1 — Preregistration

Date: 2026-08-17

## 0. Scope and Identity

Per [`family_c_step2_reconstruction_audit_20260817.md`](../audits/family_c_step2_reconstruction_audit_20260817.md)
§9/§14 (`FAMILY_C_RECONSTRUCTION_BOUNDED`): exact historical replay of
Family C / KV v2 is confirmed structurally impossible (the historical
runner never serialized request-level data). This document freezes a
**new, explicitly-versioned evaluation layer**:

**`CURRENT_RECONSTRUCTED_FAMILY_C_V1`**

It is **not** historical KV v2 replay, **not** a replacement for or
correction of the frozen historical run, and **not** used to reinterpret
`KV_FAMILY_COMPOSITION_READY`. It is a new Step-2 data layer: the same
scenario generator, config, and code path, run once in the current
deterministic environment, with **all 6** canonical anchors (not just the 2
native ones) evaluated on the exact same frozen request-level input per
scenario. Data generation only — no selector, no composition/synthesis, no
mechanism attribution.

## 1. Frozen Parameters

| Field | Value |
|---|---|
| Scenario count | 72 |
| Scenario factors | `bulk_pressure` ∈ {low, high}, `urgent_arrival_phase` ∈ {early, middle, late}, `urgent_tightness` ∈ {loose, tight}, `seed` ∈ {20260910..20260915} (2×3×2×6=72) |
| Config | `configs/kv_pressure_pilot_v2.yaml` (unchanged since `6be526e`) |
| Generator | `case_kv_pressure_reserve_contention_v2` (`templates_kv_pressure_v2.py`, `generator_version="kv_pressure_v2"`) via `build_scenarios(cfg, template_version="v2", ...)` (`scripts/run_policy_separation_kv_pressure_pilot_v1.py`, unmodified, imported by path) |
| BurstGPT dataset | resolved via `resolve_burstgpt_path(datasets_root=".local_data")` → `.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv` (same file used throughout this project's current-environment work; SHA-256 recorded in the build manifest) |
| Reconstruction-version identifier | `CURRENT_RECONSTRUCTED_FAMILY_C_V1` |
| Six canonical policies | identical to `docs/design/UNIFIED_UTILITY_MATRIX_STEP2_V1.md` §1 — `estimated_service_time_first`, `weighted_fair_share`, `full_prefill`, `chunked_prefill_small`, `least_laxity_first`, `kv_constrained_online`. **Not redefined here.** |
| Primary metric | `arrival_normalized_weighted_goodput` |
| Expected row count | 72 × 6 = **432** |
| Current git HEAD at generation time | recorded in the build manifest |

## 2. Generate-Once, Serialize, Replay-From-Disk Discipline

**Critical design constraint, stated explicitly:** the 72 scenarios are
generated **exactly once** (one call to `build_scenarios`, which calls the
generator 72 times, each with its own `np.random.default_rng(seed)` per
the forensic audit's §6 finding of local-generator determinism). The full
request-level result (every `Request` field, every `GPUConfig` field, the
resolved `service_model_kwargs`, and `params`) is serialized to
`experiments/family_c_reconstruction_v1/family_c_reconstruction_v1_scenarios.jsonl`
immediately after generation. **All 432 policy evaluations read scenarios
back from this serialized file, not from a second independent call to the
generator.** This guarantees that all 6 policies — including the 2 that
are native to Family C — see byte-identical request-level input for a
given scenario, and decouples evaluation from any further BurstGPT access:
the loader (`load_serialized_scenarios`) imports nothing from
`templates_kv_pressure_v2`, `templates_prefill_decode`, or any BurstGPT
resolution/loading code, and this is enforced by a dedicated test
(§7 below) that fails if the load path is ever wired back to a resampling
call.

## 3. Serialization Format

One JSON object per line (JSONL), one line per scenario:

```
{
  "scenario_id": str, "seed": int, "params": {...},
  "service_model_kwargs": {...},
  "gpu_configs": [{gpu_id, max_active_sequences, max_batch_tokens,
                    max_kv_tokens, role, hybrid_cache_enabled, ...}, ...],
  "requests": [{request_id, arrival_time, prompt_tokens,
                 predicted_output_tokens, actual_output_tokens,
                 slo_deadline, priority, class_id}, ...]
}
```

`gpu_configs` via `dataclasses.asdict(GPUConfig)` (every field, including
defaults — matches `PolicySeparationScenario.gpu_configs_as_dicts()`'s own
convention). `requests` via `dataclasses.asdict(Request)` (all 8 fields,
including `actual_output_tokens` — present in the frozen record for
completeness/audit purposes even though policies never see it, matching
`Request`'s own documented visibility contract enforced by `ObservableRequest.from_request`).

This differs deliberately from Family A/B's own `scenarios.jsonl`
convention (`PolicySeparationScenario.to_manifest_dict()`, which stores
only `params`+`seed` for later regeneration, not raw requests) — that
convention is sufficient for A/B, whose regeneration was independently
verified byte-exact against frozen history
([`unified_policy_utility_matrix_v1_20260817.md`](../audits/unified_policy_utility_matrix_v1_20260817.md)
§B). Family C has no such external-history anchor to verify against on
each reload, so this layer instead freezes its own request-level ground
truth once, directly, so no later step can silently diverge from it.

## 4. Historical vs. Reconstructed Separation (frozen rule)

- `HISTORICAL_FAMILY_C_KV_V2` (MF-PSD v1's 144 frozen native rows,
  `experiments/mf_psd_v1/`) is **read-only reference evidence**, used only
  for the diagnostic crosswalk (§6). It is never merged row-wise with this
  layer's cells, and its own file is never modified.
- `CURRENT_RECONSTRUCTED_FAMILY_C_V1` (this layer, 432 rows, all 6
  anchors including a **freshly re-evaluated** `kv_constrained_online` and
  `least_laxity_first`) is a wholly separate artifact,
  `experiments/family_c_reconstruction_v1/`.
- Any downstream Step-2 rebuild that includes Family C must use this
  layer's values for **all 6** Family-C columns uniformly — never a mix of
  historical-native-for-2-columns and reconstructed-for-4-columns on the
  same scenario row (the exact hybrid this document exists to prevent).

## 5. Success/Failure/Determinism Criteria

- `status="success"` on a clean run, `status="failed"` (never silently
  dropped) on exception.
- Matrix completeness criterion: 432/432 rows present, one per (scenario,
  canonical_policy_id) pair, zero duplicates.
- Determinism criterion: reloading the serialized JSONL and reconstructing
  `Request`/`GPUConfig` objects must reproduce every field exactly
  (verified by a dedicated equality test against the original in-memory
  generation, §7); this is stronger than "the simulator produces the same
  ANWG" — it is exact, field-by-field structural equality of the frozen
  input data itself.
- Stop conditions: regenerated scenario count ≠ 72; regenerated scenario
  ID set ≠ MF-PSD v1's Family-C `source_scenario_id` set; any leakage-guard
  assertion fails; any frozen artifact (MF-PSD v1, historical KV v2 run
  dirs, `unified_utility_matrix_v1`) shows a git diff after the build.
