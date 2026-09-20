# Bootstrap Cluster-Key Correction (2026-09-20)

STATUS: CORRECTION APPLIED — bug fix implementing the pre-registered cluster unit.

## Classification

**BUG_FIX_TO_IMPLEMENT_DECLARED_METHOD** — not a post-hoc methodological
change. No new serving experiment, continuation, or state generation was run;
the correction recomputes only the bootstrap uncertainty interval from the
frozen 720-state artifact.

## Original execution identity

- Execution commit: `38e02bcdaeb8f9cec2289b05dea9a1406318d14f`
- Executing script: `scripts/fresh_latency_causal_confirmatory_v1.py`
  (commit `b196c3e`, sha256 `e4ef52c38ce0412674319cdf2997f1934ab5385c031b227010eb6a913f24815b`,
  as recorded in `FRESH_LATENCY_EXECUTION_PROVENANCE_V1.json`)
- Original cluster key: bare integer `window_index`
  (`sorted(df["window_index"].astype(int).unique())` in the original
  `bootstrap()` function)

## Why the original key was wrong

`window_index` is assigned independently within each workload (source-local
namespace), as frozen in `FRESH_WINDOW_UNIVERSE_V1.json` and
`OVERLAP_AUDIT_V1.json` (200-request windows per workload, overlap checked by
`source_record_id` within the same workload). Consequently, for example,
Azure-code window 7 and Azure-conversation window 7 are distinct physical
source windows that shared one bootstrap cluster. The original key produced
26 integer-index clusters from 36 distinct source windows, silently merging
10 cross-workload window pairs.

## Pre-registered cluster unit

- `PREREGISTRATION_V1.json` → `confirmation_rule.cluster_unit`:
  `"faithful source window"` (frozen before execution;
  `status: PREREGISTERED_NOT_EXECUTED`)
- `docs/design/FRESH_PRODUCTION_LATENCY_HEADROOM_CONFIRMATORY_V1.md`
  (frozen design): "cluster unit: faithful source window … Do not use
  state-level IID bootstrap"
- `docs/current/FRESH_LATENCY_HEADROOM_CONFIRMATORY_DESIGN_REPORT.md`:
  "faithful source window clusters"

## Corrected implementation

- Corrected cluster key: `(source_dataset, window_index)` — the faithful
  source-window identity.
- Corrected clusters: **36** (19 `azure_2023_code` + 17 `azure_2023_conv`)
- Original clusters: **26**
- Unchanged: same 720 frozen states, same `oracle_headroom` values, same
  estimator (state-weighted mean), same percentile cluster bootstrap,
  same 2,000 replicates, same seed 20260920.

## Results

| Quantity | Original | Corrected |
|---|---:|---:|
| Cluster count | 26 | 36 |
| States | 720 | 720 |
| Beneficial states | 590 | 590 |
| P(B_LAT \| D) | 0.819444… | 0.819444… |
| Observed mean headroom | 1.995791 ms | 1.995791 ms |
| Clustered 95% CI (ms) | [0.184589, 3.497418] | [0.190884, 3.546196] |
| Confirmatory verdict | POSITIVE | POSITIVE |

The confirmatory verdict is unchanged: positive mean headroom and clustered
95% CI lower endpoint > 0 under both keys.

## Changed artifacts (this directory)

- `FRESH_LATENCY_BOOTSTRAP_V1.json` — clusters 26 → 36; corrected CI
- `FRESH_LATENCY_CAUSAL_RESULT_V1.json` — `primary.clustered_ci95` only
- `FRESH_LATENCY_CONFIRMATORY_REPORT_V1.md` — CI line and gate wording
- `FRESH_LATENCY_ARTIFACT_HASHES_V1.json` — compact-artifact hashes for the
  three files above re-recorded; all other hashes unchanged and verified

`FRESH_LATENCY_EXECUTION_PROVENANCE_V1.json` deliberately still records the
original execution (commit `38e02bcd`, original script hash
`e4ef52c3…`); the original execution output is fully preserved in Git
history, and this note records the correction. `ARTIFACT_HASHES_V1.json`
(pre-execution freeze) and `FRESH_SUPPORT_RESULT_HASHES_V1.json` (support
stage) pin only unaffected files and are unchanged.
