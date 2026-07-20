# Experiments and Results (Canonical)

## The three directories, and why they're different

```
results/       96M   gitignored except results/.gitkeep -- local-only, large/regeneratable
logs/          480K  gitignored entirely -- local-only, raw run logs
experiments/   20M   NOT gitignored -- the only one of the three that is actually
                      version-controlled; small, curated, committed artifacts
```

**If a document cites a `results/...` or `logs/...` path as evidence, that
path exists only on the machine that generated it -- it is not part of the
committed repository and a fresh clone will not have it.** `experiments/...`
paths, by contrast, are (mostly) real, cloneable, committed artifacts.

## Major result families and their status

| Family | Location | Canonical (committed)? | Status | Safe for manuscript claims? |
|---|---|---|---|---|
| Phase 1-2B.16 historical results | `results/phase2*` | No (gitignored) | Complete, historical | Only via the docs that cite specific numbers (`docs/result_claims.md`), not by pointing at the local directory |
| Real-vLLM external serving comparison | `experiments/real_llm/` | **Yes** | Complete; selector arm corrected (`docs/vllm_real_serving_scaled_comparison_corrected.md`) | Yes, per that doc's safe-claim language |
| Wulver A100 KV-pressure validation | `docs/wulver_vllm_kv_pressure_results.md` (results referenced, not all committed) | Partial | Complete | Yes, per that doc |
| Real Sarathi runtime validation | referenced from `docs/wulver_sarathi_vllm_repeated_validation.md` | Partial | Complete (N=5 repeated trials) | Yes, per that doc |
| Runtime validation benchmark pack | `experiments/runtime_validation_benchmark_pack/` | **Yes** | Complete, checksummed | Yes -- this is the acceptance target for `vllm_chunked_prefill_faithful` |
| Selector Dataset v2 pilots (v1/redesigned) | `experiments/selector_v2_contention_frontier_search/`, `experiments/selector_v2_slo_calibrated_frontier_search/`, `experiments/selector_v2_faithful_baseline_scope_audit/` | **Yes** | Historical intermediate steps, superseded by the calibrated targeted pilot | Historical only -- cite the superseding doc for current claims |
| **Selector Dataset v2 calibrated targeted pilot (current)** | `experiments/selector_v2_calibrated_pilot_20260720T163235Z/` | **No -- untracked, local-only** | Finished, all pipeline quality gates passed, held-out results mixed (see [SELECTOR_V2.md](SELECTOR_V2.md)) | **No.** Do not cite as a canonical or final result. See below. |

## The calibrated targeted pilot, specifically

`experiments/selector_v2_calibrated_pilot_20260720T163235Z/` is:
- a finished local experiment output (generation completed, prototype
  selector trained and evaluated);
- **not committed to git** (untracked as of this writing);
- **not canonical** -- it is the most recent evidence, not a settled result;
- reporting mixed held-out performance (loses on VALIDATION and OOD_TEST for
  both trained model variants -- see [SELECTOR_V2.md](SELECTOR_V2.md) §9);
- subject to an **open, unverified leakage question** on its non-OOD splits
  (the pipeline's own gate says no leakage; this has not been independently
  audited -- see [SELECTOR_V2.md](SELECTOR_V2.md) §10 and
  [PROJECT_STATUS.md](PROJECT_STATUS.md) §3).

Treat it as current local evidence informing the next research step, not as
a citable result. Whether/how to commit it is deferred to a later cleanup
query (do not commit it as-is without resolving the open question above).

## Orphaned / undocumented `results/` directories

Roughly a third of `results/`'s ~68 subdirectories are not cited by literal
path from any doc. Some are prior runs of repo-hygiene/audit workflows
(dated mid-June 2026, plausible ancestors of this very cleanup process);
others are genuine research-phase results simply not linked by path. None
were moved or deleted by this documentation pass -- that is Query 4 scope.

## Datasets backing these results

- **BurstGPT**: raw + processed present, MIT-licensed, actively used.
- **Azure LLM 2023** (conv + code): raw + processed present, CC BY 4.0,
  actively used in the Selector v2 pipeline.
- **ShareGPT**: loader code exists, raw data not acquired.
- **Azure LLM 2024/2025, Mooncake/Kimi, ServeGen, TraceLab**: acquisition
  candidates only, documented in
  `src/llmserveopt/selector/dataset_v2/workload_sources.py`, not downloaded.
  That module's docstring explicitly forbids any code from silently
  downloading them.
