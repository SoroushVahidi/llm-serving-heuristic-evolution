# Wolverine Oracle-Mixture Sweep — Handoff

"Wolverine" here is this sweep's codename, not the cluster name -- the
actual SLURM cluster used throughout this repo is **Wulver** (see
`WULVER_HANDOFF.md`, branch names such as
`wulver-selector-v2-and-composition-integrated`, and the `/mmfs1/...` paths
below). This document hands off the large-scale composition sweep to a
future Wulver (SLURM) session. It does not launch anything. The
machine-readable spec is `docs/current/wolverine_oracle_mixture_spec.json`;
read that for exact fields.

**Evidence location note (updated during 2026-07-24 repository
reconciliation):** the `NATIVE_COMPOSITION_PILOT_DECISION = NO_GO` finding
attributed to job `1120123` is a real result from the real composition
harness (not a proxy). Its numeric artifacts live outside Git under the
durable Wulver data root
`/mmfs1/project/ikoutis/sv96/llmserveopt-data/native_composition_pilot_20260721T194929Z/`
(`pilot_report.md`, `method_comparison.csv`, `subset_analysis.csv`,
`pilot_manifest.json`, …). Those files were **verified readable** on
2026-07-24 from the authoritative checkout; they are still not vendored
into this git worktree (by design — large experiment roots stay on the
shared data filesystem). Optional future convenience import into
`results/wulver_imports/` remains available the way
`module_intervention_credit_20260721T224322Z` already was, but is no
longer required merely to *find* the artifacts.

## What changed since the last composition pilot

`tools/native_composition_pilot.py` (job `1120123`) already tested weighted
Borda rank ensembles and component-wise composition against a discrete
selector and found `NATIVE_COMPOSITION_PILOT_DECISION = NO_GO`. This session
added two operator families that pilot never exercised:

- **Weighted reciprocal-rank aggregation** (`composition.weighted_reciprocal_rank_aggregate`,
  `StaticRankEnsemblePolicy(method="reciprocal_rank")`) — weights top-rank
  agreement more heavily than normalized-rank (Borda) spacing does.
- **Weighted score aggregation** (`score_aggregation.py`) — combines genuine
  comparable scalar scores (not just ranks) from the 8 policies in
  `capabilities.SCORE_CAPABLE_EXPERTS`, after per-expert normalization
  (`none`, `min_max`, `zscore`, `robust_mad`).

Re-running the exact prior 5-treatment rank/component design is out of scope
for the future sweep; it would just reproduce the existing NO_GO finding.

## Required comparisons

Per the research brief this sweep exists to answer, every run must be able to
report all five of:

1. Best individual policy (`T0`)
2. Uniform composition (`T2`, equal weights, no tuning)
3. Random sparse composition (`T3`, dirichlet-random weights + random top-k —
   this is the null control; oracle/learned treatments must beat it by more
   than noise to mean anything)
4. Oracle sparse composition (`T4`, hindsight grid search on the *same*
   evaluation window — non-deployable upper bound only, exactly like the
   existing `oracle_srtf` convention in `registry.ORACLE_POLICY_NAMES`; never
   selected from or reported as beating a development-only baseline)
5. Learned-weight composition (`T5`, development-trained regressor weights,
   frozen before evaluation)

`T1` (the existing discrete `regression_anwg` selector) is included as the
non-composition reference point already in production.

## Readiness gate (unchanged)

Do not submit until both upstream final reports exist:

- `policy_frontier_cartography_20260721T154408Z/reports/FINAL_REPORT.md`
- `policy_library_v2_expanded_20260721T171933Z/reports/FINAL_POLICY_LIBRARY_REPORT.md`
  (or `reports/FINAL_REPORT.md`)

Check with `llmserveopt.selector.composition_experiment.check_upstream_readiness`.

## Estimated computational dimensions

- Policies: 8 score-capable + up to 12 rank-only experts (see
  `capabilities.RANK_CAPABLE_EXPERTS`); top-k in `{2, 3, 5, all}`.
- Scenarios: reuse `policy_library_v2_expanded_20260721T171933Z/design/policy_library_v2_design.csv`
  (same ~100-window design as the prior pilot, split across
  `train/validation/id_test/synthetic_ood`).
- Seeds: 1 scenario-sampling seed + 5 seeds for random-sparse draws.
- Composition methods: 3 (`weighted_borda`, `weighted_reciprocal_rank`,
  `weighted_score`) × up to 3 normalization modes for the score operator ×
  4 top-k values.
- Rough row count: `treatments(6) × composition_methods(~3) × top_k(4) ×
  scenarios(~100) × seeds(1 or 5 for T3)` — order of a few thousand rows, well
  within a single-node SLURM array, not a large training job.

## Parallelization

Embarrassingly parallel over `(scenario_id, seed, treatment_id)` — same
`ProcessPoolExecutor` pattern already used in `tools/native_composition_pilot.py`.
The only serial dependency is fitting `T5`'s development-only reward
regressor, which must finish before `T5` evaluation rows can be generated.

## Checkpoint and resume

Follow the convention already used by `scripts/run_module_credit_overnight.py`:
a `--resume-dir` flag, a `checkpoints/` subdirectory, a periodically-written
`heartbeat.json`, and per-row resume keyed on
`(scenario_id, seed, treatment_id, composition_method, normalization_mode, top_k)`
— skip any row already present in `policy_vectors.csv`.

## Expected output files

See `expected_result_schema` in the JSON spec. Top level:

```
composition_oracle_mixture_<timestamp>Z/
  logs/
  checkpoints/
  manifests/
  reports/FINAL_REPORT.md
  policy_vectors.csv
  composition_weights.csv
  method_comparison.csv
  subset_analysis.csv
  decision_traces/<treatment_id>/<scenario_id>_<seed>.jsonl
  pilot_manifest.json
```

`decision_traces/` uses `llmserveopt.policies.instrumentation.DecisionTraceSink`
(`DecisionTraceV1` schema) — enable it only for this Wulver run, not for
routine local development, since it does add allocation and I/O overhead when
turned on (it remains zero-overhead when left at its default `enabled=False`).

## Decisive question

Does any oracle-informed or learned-weight mixture (`T4`, `T5`) beat the best
individual policy (`T0`) by more than the random-sparse null margin (`T3`) on
held-out splits, using `weighted_reciprocal_rank` or `weighted_score` in
addition to the already-tested `weighted_borda`? If not, the existing
`COMPOSITION_AND_SYNTHESIS_ARCHITECTURE.md` recommendation to prioritize
structural symbolic synthesis over further composition search still stands.
