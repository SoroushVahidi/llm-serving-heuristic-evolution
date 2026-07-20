# Project Status (Canonical)

**This is the single authoritative current-status document for this repository.**
`docs/research_status.md` is retained only as a historical redirect to this file.

**Status as of:** 2026-07-20, repository-audit pass (Queries 1-5 of a 5-query
cleanup, now complete). This document reflects the state at commit `3406bc0`
on branch `selector-v2-calibrated-targeted-pilot` (the scientific work),
plus documentation, source-of-truth, and artifact cleanup commits layered on
top across `repo-polish-query2-safe-cleanup` through
`repo-polish-query5-final-verification` (see that branch's final commit for
the full lineage; not yet merged to `main` as of this writing -- see the
integration recommendation in `docs/current/AGENT_HANDOFF.md`).

> Prefer commit hash over branch name when precision matters: branch names
> in this project have been renamed at least three times as work
> progressed (`phase2a4-2b4-final-eval` -> `phase2c1-real-trace-ingestion-validation`
> -> `selector-v2-calibrated-targeted-pilot`), so treat the commit hash above,
> not a hardcoded branch name, as the source of truth for "where we are."

---

## 1. Project objective

Learn a **selector** that dynamically chooses among LLM-inference-serving
scheduling policies per workload window, evaluated in a GPU-calibrated
discrete-event simulator, and compare it against (a) the strongest fixed
internal policy and (b) faithful reimplementations of real external serving
systems (vLLM, Sarathi-Serve, DistServe, TetriInfer, Llumnix). See
[ARCHITECTURE.md](ARCHITECTURE.md) for the system design and
[SELECTOR_V2.md](SELECTOR_V2.md) for the full selector research narrative.

## 2. Completed milestones (verified)

- Deterministic discrete-event simulator with a GPU-calibrated service model
  (RTX 5060 Ti / Qwen2.5-0.5B, prefill MAPE 9%, decode MAPE 12%).
- 20-policy historical/internal policy portfolio (Phase 1-2B.16).
- 6 faithful external baseline reimplementations, each pinned to an exact
  upstream commit, covering 3 topology classes (monolithic, disaggregated,
  migratory) -- see [BASELINES.md](BASELINES.md).
- A real bug in the legacy `weighted_goodput` objective (completed-request-only
  denominator, biased toward policies that reject/drop more work) was found
  and fixed via the arrival-normalized `arrival_normalized_weighted_goodput`
  (ANWG) objective. `docs/selector_objective_audit.md`.
- Real-hardware runtime validation on a local RTX 5060 Ti and on a Wulver
  A100 cluster (single-job and N=5-repeated-trial Sarathi-vs-vLLM
  comparisons; a committed, checksummed runtime-validation benchmark pack).
  `docs/wulver_sarathi_vllm_repeated_validation.md`, `docs/runtime_validation_benchmark_pack.md`.
- Selector Dataset v2 infrastructure: policy-independent per-request SLO
  calibration, group-aware leakage-safe split machinery, and an explicit,
  evidence-based scope decision (**Option B**) for the Selector v2 trainable
  action space. `docs/selector_v2_faithful_baseline_scope_audit.md`.
- Test suite: **2,501 tests collected**, 0 collection errors, under the
  correct interpreter (`python3 -m pytest --collect-only -q`; the bare
  `pytest` shim on `PATH` on this machine is missing `pandas` and
  undercounts -- see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

## 3. Current Selector v2 status (verifiable claims only)

The most recent pipeline run (`experiments/selector_v2_calibrated_pilot_20260720T163235Z/`,
small provenance/summary/audit files committed, large raw CSVs local-only)
generated 250 retained windows under the Option B 8-policy action space and
trained a prototype RF regressor/classifier.

**What the pipeline's own automated checks report:** all 10 quality gates
passed, including `no_leakage: {"passed": true, "detail": "verified"}`
(read directly from `quality_gates.json`) -- **now known to be an
incomplete check** (see the leakage finding below).

**What the held-out numbers actually show** (mean ANWG vs.
`weighted_shortest_processing`, the best fixed policy on this pilot):

| Split | n | Regressor vs. best-fixed | Classifier vs. best-fixed |
|---|---|---|---|
| TRAIN | 125 | +0.031 (expected; not held-out evidence) | +0.020 |
| VALIDATION | 51 | **-0.012 (loses)** | **-0.037 (loses)** |
| ID_TEST | 43 | +0.029 (wins) | -0.011 (loses) |
| OOD_TEST | 31 | **-0.010 (loses)** | **-0.031 (loses)** |

OOD_TEST is a markedly harder regime for every policy (mean ANWG ~0.14-0.26
across the board, including oracle at 0.263), not just the learned selector.
**VALIDATION and ID_TEST are also now known to be measured on a leaky
split** (see below), so their numbers cannot be trusted as held-out evidence
either -- OOD_TEST is the only split confirmed leakage-free, and it loses.

**Conclusion: this is not a clean, confirmed win, and the non-OOD splits
cannot be trusted as held-out evidence.** Full detail: [SELECTOR_V2.md](SELECTOR_V2.md).

**Leakage: CONFIRMED, independently reproduced.** A concern was raised about
this pilot's non-OOD splits; it has now been independently audited, not just
asserted. Finding: 19 cross-split row-range overlap pairs, 27 of 48
real-trace historical-pool windows (56%) involved, spanning
TRAIN-VALIDATION/ID_TEST-TRAIN/ID_TEST-VALIDATION. Mechanism: the same
underlying trace row range, drawn under different transforms, is treated as
independent groups by the split logic and can land in different splits.
**OOD_TEST is unaffected** (zero overlaps involve it; the OOD row-range
reservation's disjointness holds). Reproducible via
`scripts/audit_selector_v2_calibrated_pilot_leakage.py`; full writeup in
`experiments/selector_v2_calibrated_pilot_20260720T163235Z/LEAKAGE_AUDIT.md`.
This is no longer an open question -- it is a confirmed data-quality problem
in the split construction, with a known, fixable mechanism.

## 4. Current dataset status

- BurstGPT and Azure LLM 2023 (conv + code): raw + processed present, actively
  used.
- ShareGPT: loader code + tests exist; raw data not acquired.
- Azure LLM 2024/2025, Mooncake/Kimi, ServeGen, TraceLab: acquisition
  candidates only, no loader code beyond Azure 2024/2025's download script.
- See [EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md) for the full
  dataset/results inventory.

## 5. Current baseline status

20 historical/internal policies + 6 faithful external baselines (3
monolithic, 2 disaggregated, 1 migratory). All 6 external baselines are
pinned to an exact upstream commit and are **evaluation-only** -- none are
part of the Selector v2 trainable action space (Option B, 8 policies). Full
inventory: [BASELINES.md](BASELINES.md).

## 6. Current external validation status

Real-GPU-hardware validation is complete for the faithful baselines this
project currently ships: RTX 5060 Ti (local) and Wulver A100 (single-job and
N=5-repeated-trial) Sarathi-vs-vLLM comparisons, reconciled against a
committed runtime-validation benchmark pack. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for what is/isn't reproducible
without that specific hardware access.

## 7. Known issues / tracked follow-ups

- ~~`src/llmserveopt/selector/dataset_v2/candidates.py`'s stale 14-policy
  constant vs. the 8-policy Option B scope~~ -- **RESOLVED**: `candidates.py`
  now defines `SELECTOR_V2_OPTION_B_POLICIES` as the canonical 8-policy
  constant (import-time-asserted against both registries), and
  `calibrated_targeted_pilot.py::CANDIDATE_POLICIES` imports it directly
  instead of duplicating the list. The broader 14-policy pool is retained,
  unchanged, under the clearer alias `MONOLITHIC_DIAGNOSTIC_POLICY_POOL` for
  historical reproducibility and diagnostic exploration. See
  [BASELINES.md](BASELINES.md) §B.
- One orphaned branch, `phase2b13-selector-training-after-diversity`, has
  unique commits not merged into any current lineage (superseded by its
  sibling `phase2b13-selector-training-and-suspicion-audit`, which is
  merged). See [AGENT_HANDOFF.md](AGENT_HANDOFF.md) for the final
  classification and disposition decision.
- `experiments/selector_v2_calibrated_pilot_20260720T163235Z/`: small
  provenance/summary/audit files (including the leakage audit) are now
  committed; the large raw `full_policy_vectors.csv`/`window_features.csv`
  remain local-only and regeneratable -- see
  [EXPERIMENTS_AND_RESULTS.md](EXPERIMENTS_AND_RESULTS.md).

## 8. Active protected local artifacts / processes

- **Stopped as of this cleanup pass (2026-07-20).** The `vllm serve` process
  (Qwen2.5-0.5B-Instruct, port 8001) had run continuously since 2026-07-03
  but served zero requests in its last 17 days (last logged HTTP activity:
  2026-07-03 22:06); nothing in the current active research track depends on
  it. Stopped gracefully via `SIGTERM` (confirmed clean: port 8001 freed, GPU
  memory released 8269 MiB -> 15 MiB, its log
  `experiments/real_llm/vllm_healthcheck_20260703T171021Z/server.log` only
  appended to, never truncated -- `git diff --stat` shows insertions only).
  The log remains intentionally uncommitted (per this project's established
  policy for this file); restart command is documented in
  `docs/current/AGENT_HANDOFF.md` if needed again.
- The two raw GPU-stress server logs from 2026-07-18 are gitignored
  (`experiments/**/server.log`) and remain local-only by design -- their
  canonical structured summaries are committed alongside them. The
  calibrated pilot's two large raw CSVs (above) are the remaining
  local-only-by-design artifacts.

## 9. Current scientific blockers

1. **The calibrated pilot's split construction has a confirmed leakage bug**
   (§3) -- VALIDATION and ID_TEST results are measured on a leaky split and
   cannot be trusted. This must be fixed before those splits mean anything.
2. **Even on the one leakage-free split (OOD_TEST), the selector loses to
   best-fixed for both trained model variants.** Whether that is a
   too-small training set (125 windows) or a genuine regime-shift problem
   (OOD_TEST's collapse affects every policy, not just the selector) is
   **not yet diagnosed** -- and can't be, productively, until a clean pilot
   exists to re-test against.
3. Until #1-2 are resolved, comparing the trained selector against the 6
   faithful external baselines (Protocol C) is premature -- there is no
   confirmed-working selector yet to compare.

## 10. Next recommended action

See [NEXT_STEPS.md](NEXT_STEPS.md) for the full sequence. Immediate next
step: fix the split-construction bug identified in §3 (group by underlying
row range across transforms, not just by transform name) and regenerate a
clean calibrated pilot, before deciding whether to scale Dataset v2
generation or retrain.
