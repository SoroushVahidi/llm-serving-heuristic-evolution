# Agent Handoff (Operational)

Short and operational. This is the "what do I do right now" doc; for the
full story, use the docs it links to.

## A. Start here

1. [`README.md`](../../README.md) -- project orientation
2. [`PROJECT_STATUS.md`](PROJECT_STATUS.md) -- authoritative current state
3. [`NEXT_STEPS.md`](NEXT_STEPS.md) -- the exact next research step

## B. Current scientific state

- **Selector v2 is not final.** The most recent pilot's prototype selector
  shows mixed/weak held-out performance and its non-OOD splits have a
  **confirmed** leakage bug (independently audited, not just suspected).
- **A leakage fix is needed** before any further pilot's VALIDATION/ID_TEST
  numbers can be trusted: group by underlying source-trace row range across
  transforms, not just by transform name (see
  [SELECTOR_V2.md](SELECTOR_V2.md) §10 for the exact mechanism).
- **A clean pilot rerun is needed** after that fix, followed by retraining
  the same (unchanged) prototype selector and evaluating cleanly on
  VALIDATION/ID_TEST/OOD_TEST before deciding whether to scale.

## C. Current source-of-truth constants

```
Historical/internal policy portfolio:        20   (policies/registry.py::BASELINE_NAMES)
Selector v2 trainable action space (Option B): 8   (selector/dataset_v2/candidates.py::SELECTOR_V2_OPTION_B_POLICIES)
Faithful external baselines, total:            6   (policies/external_baselines_registry.py::EXTERNAL_BASELINE_NAMES)
  monolithic:    3   (vllm_faithful, vllm_chunked_prefill_faithful, sarathi_faithful)
  disaggregated: 2   (distserve_faithful, tetriinfer_paper_reimplementation)
  migratory:     1   (llumnix_faithful)
```
All four counts are programmatically guarded by
`tests/test_selector_v2_candidate_source_of_truth.py` -- if you change any
of these sets, run that file first.

## D. Current protected local artifacts / processes

- **Live vLLM process: STOPPED** as of this cleanup pass (2026-07-20). It
  had run idle for 17 days (0 requests in that window) with nothing
  depending on it; stopped gracefully via `SIGTERM`, port 8001 freed, GPU
  memory released, its log untouched (append-only, never truncated). If you
  need it again for a real-vLLM-serving pilot, the exact command (from its
  own log) is:
  ```bash
  cd /home/soroush/llm-serving-heuristic-evolution
  CUDA_HOME=/home/soroush/.venvs/vllm_baseline_pilot/lib/python3.12/site-packages/nvidia/cu13 \
  PATH="$CUDA_HOME/bin:/home/soroush/.venvs/vllm_baseline_pilot/bin:$PATH" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  /home/soroush/.venvs/vllm_baseline_pilot/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --port 8001 --gpu-memory-utilization 0.5 --max-model-len 4096 --enforce-eager
  ```
- **Local-only raw logs**: `experiments/**/server.log` files are gitignored
  by pattern (with two pre-existing, intentional tracked exceptions -- see
  `.gitignore`). Never truncate or force-add one without a specific reason.
- **Local pilot CSVs**: `experiments/selector_v2_calibrated_pilot_20260720T163235Z/full_policy_vectors.csv`
  and `window_features.csv` are large, regeneratable, and intentionally
  local-only. The pilot's small provenance/summary/audit files (including
  the leakage audit) are committed.
- **Orphaned branch** `phase2b13-selector-training-after-diversity`:
  **KEEP_AS_HISTORICAL**, not deleted. It's a same-day, narrower alternative
  to `phase2b13-selector-training-and-suspicion-audit` (the branch that's
  actually in the merged lineage) -- both diverge from one commit
  (`93b6da7`) with a single commit each; the merged sibling covers strictly
  more scope (319 vs. 256 windows, 4 selector-model types vs. 2, an
  explicit suspicion audit, source-code changes) and reaches the same
  headline result (RF test WG=0.9975). No unique finding would be lost by
  never merging it; kept anyway, consistent with this repo's existing
  convention of retaining every phase-numbered branch as a low-cost
  historical record.

## E. Exact next recommended task

1. Fix the raw-trace ancestry/cross-transform split-construction leakage
   (group by underlying row range, not transform name).
2. Regenerate a clean calibrated Selector v2 pilot (250-500 windows).
3. Verify zero leakage independently (`scripts/audit_selector_v2_calibrated_pilot_leakage.py`
   against the new pilot -- don't just trust the pipeline's own gate).
4. Retrain the same prototype selector (no hyperparameter tuning yet).
5. Evaluate clean VALIDATION/ID_TEST/OOD_TEST. See
   [NEXT_STEPS.md](NEXT_STEPS.md) for the full sequence and stop conditions.

## F. Long-running workflow rules

- **Local long-running work**: run under `tmux`, name the session
  descriptively (existing convention: `<purpose>-<date>` or similar).
- **Wulver A100 work**: submit via `sbatch` (`scripts/slurm/wulver_*.sbatch`);
  requires that specific SLURM account. See
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- **Sync**: GitHub is the primary path for code/docs/small committed
  experiment artifacts. Use `scp` (or equivalent cluster transfer tooling)
  for large raw artifacts that should stay off git (results/logs/raw data).

## G. Do-not-do list

- Do not use the contaminated (pre-leakage-fix) calibrated pilot for final
  scientific claims -- its VALIDATION/ID_TEST splits are confirmed leaky.
- Do not add faithful external baselines to the Selector v2 trainable action
  space -- they are confirmed genuinely dominated and are evaluation-only
  by an explicit, evidence-based decision (Option B).
- Do not use the legacy, completed-only-denominator `weighted_goodput` as
  the primary objective -- use `arrival_normalized_weighted_goodput` (ANWG).
- Do not flatten incompatible topology baselines into one comparison --
  monolithic, disaggregated, and migratory external baselines each need
  their own topology-aware comparison protocol (Protocol C,
  `docs/external_baseline_integration.md`).
- Do not trust a bare `pytest` invocation -- use `python3 -m pytest` (see
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).
- Do not treat `docs/research_status.md` or `docs/roadmap.md`'s historical
  sections as current -- [PROJECT_STATUS.md](PROJECT_STATUS.md) is the
  authority.
