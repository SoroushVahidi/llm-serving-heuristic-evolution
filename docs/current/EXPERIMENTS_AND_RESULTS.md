# Experiments and Results (Canonical)

## Current Wulver Addendum (2026-07-22)

For current Wulver experiment roots and conclusions, prefer
`EXPERIMENT_INDEX.md`. Since this document was first written, the project has
completed:

- V2 real-OOD 27-policy library audit: strong V2 oracle-envelope gain.
- 27-policy selector/regret benchmark: useful suitability signals but OOD
  oracle-gap capture remains limited.
- Module intervention/structural credit: sparse positive single-module transfer
  but weak module-credit generalization.
- SwissAI staging and 27-policy sweep: novel KV/cache/reuse metadata, but
  saturated ANWG and zero strict V2 marginal gain.
- TraceLab staging and 27-policy sweep: novel long-context/agentic/prefix
  metadata, but saturated ANWG and zero strict V2 marginal gain.
- SLO/deadline augmentation: useful synthetic pressure and partial class-balance
  improvement.
- Simulator discriminative-power audit: current primary bottleneck is
  simulator/objective reward separation, not generic dataset volume.

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
| **Selector Dataset v2 calibrated targeted pilot (historical)** | `experiments/selector_v2_calibrated_pilot_20260720T163235Z/` | **Partial** -- small provenance/summary/audit files committed, large raw CSVs local-only | Finished; pipeline quality gates passed but a real split-construction leakage bug was independently confirmed (see below) | **No.** Confirmed data-quality problem, not a citable result. See below. |

## The calibrated targeted pilot, specifically

`experiments/selector_v2_calibrated_pilot_20260720T163235Z/` is:
- a finished local experiment output (generation completed, prototype
  selector trained and evaluated);
- **partially committed**: small provenance/summary/audit files
  (`manifest.json`, `provenance.json`, `quality_gates.json`,
  `split_manifest.json`, `final_summary.md`, `selector_metrics.json`,
  `retained_windows.csv`, `LEAKAGE_AUDIT.md`, `leakage_audit.json`) are
  committed; the large raw `full_policy_vectors.csv` (884K) and
  `window_features.csv` (52K) remain local-only and regeneratable;
- **not canonical** -- it is evidence of a confirmed problem, not a settled
  result;
- reporting mixed held-out performance (see [SELECTOR_V2.md](SELECTOR_V2.md) §9);
- subject to a **CONFIRMED, independently-audited leakage bug** in its split
  construction (VALIDATION/ID_TEST are not clean held-out splits; OOD_TEST
  is confirmed leakage-free and separately loses to best-fixed) -- see
  `LEAKAGE_AUDIT.md` in that directory and
  [SELECTOR_V2.md](SELECTOR_V2.md) §10.

Treat it as evidence motivating the next pilot's split-construction fix, not
as a citable selector result.

## Artifact commit policy

When finishing an experiment, decide what (if anything) belongs in git:

- **Commit small, structured provenance/summary files** whenever an
  experiment produces a result worth citing later: `manifest.json`,
  `provenance.json`, `quality_gates.json` / gate-check output,
  `split_manifest.json` or similar split-summary, `selector_metrics.json`
  or similar metrics summary, `final_summary.md`. These are what a doc can
  safely point to and what a future audit can re-derive conclusions from
  without re-running anything.
- **Keep large, raw, regeneratable files local-only**: full per-window
  policy-vector CSVs, full feature CSVs, raw server logs
  (`experiments/**/server.log` is gitignored -- see `.gitignore`), request
  dumps. If a specific raw file is needed to *reproduce* a specific
  committed finding (e.g. `retained_windows.csv` is needed to re-run the
  leakage audit script), commit that one file even if "large-ish" -- the
  test is "does an independent check need this file," not "is it small."
- **How to mark an experiment invalid/superseded**: add a dedicated,
  clearly-named markdown file inside that experiment's own directory (e.g.
  `LEAKAGE_AUDIT.md`) rather than only noting it in a separate doc --
  co-locate the finding with the evidence. Cross-link it from
  [SELECTOR_V2.md](SELECTOR_V2.md) or the relevant canonical doc so it's
  discoverable without knowing the experiment directory name in advance.
- **How to preserve provenance without committing giant files**: write a
  small, reproducible audit *script* (e.g.
  `scripts/audit_selector_v2_calibrated_pilot_leakage.py`) that regenerates
  the finding from the committed small files, rather than committing a
  giant intermediate file "just in case." This keeps the repository small
  while keeping every claim independently re-checkable.
- **Never commit**: model weights/caches, HuggingFace/venv caches, `.env`
  files or credentials, or anything already covered by `data/raw/`,
  `data/processed/`, `results/`, `logs/`'s gitignore rules.

## Orphaned / undocumented `results/` directories

Roughly a third of `results/`'s ~68 subdirectories are not cited by literal
path from any doc. Classified into research-still-useful,
checkpoint/resume-history, repo-maintenance-audit-output, and
duplicate/empty/unknown buckets in
[LOCAL_ARTIFACT_CLEANUP.md](LOCAL_ARTIFACT_CLEANUP.md) (not part of the
primary start-here set -- consult it when you need to decide what to do
with a specific local `results/` directory). Nothing has been moved or
deleted -- `results/` is entirely gitignored, so this is a local cleanup
plan, not a repository change.

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
