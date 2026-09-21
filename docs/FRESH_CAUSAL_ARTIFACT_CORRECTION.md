# Fresh causal artifact: state-level reference-column correction

Frozen artifact: `experiments/fresh_production_latency_headroom_confirmatory_v1/` (unchanged).
Corrected derivative: `experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/`.
Generator: `scripts/fresh_causal_correct_state_level_v1.py`. Tests: `tests/test_fresh_causal_state_level_correction.py`.

## A. ISSUE

In `FRESH_LATENCY_STATE_LEVEL_V1.csv`, the columns `mean_ref_latency` and `p95_ref_latency` do not contain the SBS-reference latencies. They contain the mean and p95 latency of the **first counterfactual branch** of each state. All 720 rows carry the first-branch value (720/720 verified row by row).

## B. ROOT_CAUSE

`scripts/fresh_latency_causal_confirmatory_v1.py::analyze` builds the state table by looping over `cf.groupby("state_id")`, where `cf` holds counterfactual rows only, and takes `meta = g.iloc[0]`. The state row is filled with `float(meta["mean_latency"])` and `float(meta["p95_latency"])`. The SBS reference rows (`branch_type == "SBS_REFERENCE"`) live in the separate frame `ref`. That frame is used correctly for `a_lat` and for the regime-level relative and p95 effects, but not for these two state-level columns.

Proof: replaying that logic on the 96 hash-verified continuation shards reproduces the frozen `FRESH_LATENCY_STATE_LEVEL_V1.csv` **byte-for-byte**. The same replay reproduces the frozen `FRESH_LATENCY_ACTION_LEVEL_V1.csv` byte-for-byte. Changing `meta` to `ref.loc[state_id]` changes only these two columns.

## C. SCOPE

Correct source of the SBS reference values: the `SBS_REFERENCE` rows of the continuation shards (289 shard artifacts, all matching the frozen manifest `FRESH_LATENCY_ARTIFACT_HASHES_V1.json`). The shards are not tracked in git; they are in the local provenance archive (`fgcs-finalization-20260920/fresh_latency/fresh-latency-execution-v1`). The 720 reference rows are extracted into `SBS_REFERENCE_ROWS_V1.csv` so the correction is checkable from the repository alone.

| Column (state-level CSV) | Rows with a wrong value (of 720) | Rows wrong by more than 1e-12 s | Size of error (original minus true) |
|---|---:|---:|---|
| `mean_ref_latency` | 694 | 692 | median 0.27 ms, max 12.1 ms |
| `p95_ref_latency` | 224 | 224 | median 55 ms, max 239 ms |
| either column | 696 | | |
| neither column (values coincide) | 24 | | |

The 2 mean-latency rows that differ by less than 1e-12 s are the two floating-point-residue states of section F. The other 13 columns are **not affected**: `source_dataset`, `window_index`, `axis`, `condition_id`, `regime_stage`, `state_id`, `num_non_sbs_branches`, `max_a_lat`, `oracle_headroom`, `beneficial_opportunity`, `all_alternatives_harmful`, `all_alternatives_zero`, `mixed_beneficial_and_harmful`. Verified: the five taken from `meta` are constant across a state's branches and equal the values on the `SBS_REFERENCE` row and in `FRESH_ELIGIBLE_DISAGREEMENT_STATES_V1.csv`; `num_non_sbs_branches` equals the frozen branch manifest; the rest derive from `a_lat`.

Other artifacts: `FRESH_LATENCY_ACTION_LEVEL_V1.csv` holds counterfactual rows and correct `a_lat`; it is unaffected (byte-replayed). The regime CSV, result JSON, bootstrap JSON and confirmatory report are unaffected (section D).

## D. SCIENTIFIC_IMPACT

**No manuscript conclusion or primary number changes.** Mean oracle headroom (0.0019957911708132978 s = 1.9958 ms), 590 beneficial states, P(B|D) = 0.8194, the clustered CI [0.1909, 3.5462] ms over 36 windows, the regime table, and the fresh-result figure are unchanged.

| Artifact field | Downstream script / result | Affected? | Explanation |
|---|---|---|---|
| state `mean_ref_latency`, `p95_ref_latency` | `analyze()`: written to the state CSV only; no later statement reads them (`rg = ref.loc[...]` in the regime loop is assigned and unused) | **Column values wrong; nothing computed from them** | Replay with both columns removed reproduces every downstream output byte-for-byte |
| action-level `a_lat` | state `max_a_lat`, `oracle_headroom`, `beneficial_opportunity`, harm/zero/mixed flags | No | `a_lat = ref["mean_latency"] - cf mean_latency`, from `SBS_REFERENCE` rows |
| `oracle_headroom` | primary mean headroom, beneficial count, `P_B_given_D`, positive mean/median in `FRESH_LATENCY_CAUSAL_RESULT_V1.json` | No | Replay from shards gives identical values; the corrected file has an identical `oracle_headroom` column |
| `source_dataset`, `window_index`, `oracle_headroom` | `bootstrap()` clustered CI in `FRESH_LATENCY_BOOTSTRAP_V1.json` | No | The function reads only these three columns; the CI is reproduced bit-exactly with both columns dropped |
| state-level columns other than the two | regime rows in `FRESH_LATENCY_WORKLOAD_REGIME_V1.csv` (`states`, `P_B_given_D`, `mean_oracle_headroom`, `positive_headroom_*`, `contributing_windows`) | No | Regime CSV replayed byte-for-byte with both columns dropped |
| `ref` frame (`SBS_REFERENCE` rows) | regime `mean_relative_headroom`, `p95_effect_mean` | No | Computed from the correct reference; recomputed independently in the tests |
| result JSON / bootstrap JSON / confirmatory report | `paper/performance_evaluation/main.tex` abstract, Sec. "Fresh causal", Table `tab:fresh` (values transcribed) | No | Values match the frozen result and regime files |
| regime CSV columns `mean_oracle_headroom`, `P_B_given_D`, `source_dataset`, `condition_id` | `paper/performance_evaluation/scripts/plot_performance_evaluation_figures.py` (`pe_fresh_headroom.pdf`) | No | Only these regime columns are read; the state-level CSV is not read by any paper script |
| state CSV `oracle_headroom`, keys, `max_a_lat` | robustness commit `5c8a78c` (all headroom, threshold, weighting, leave-one-out, bootstrap, composition results) | No | See section H |
| a copy of the state CSV | `release/performance_evaluation_v1/experiments/fresh_production_latency_headroom_confirmatory_v1/` | Contains the same mislabeled columns | Byte-identical to the frozen file; not modified here |

Apart from the executor (which writes them) and the robustness/correction tooling (which only mention or drop them), no file in the repository reads `mean_ref_latency` or `p95_ref_latency` (repository-wide search of `.py`, `.tex`, `.md`, `.json`, `.yaml`, `.sh`). The defect is confined to two descriptive columns of one CSV.

## E. CORRECTION

`fresh_causal_correct_state_level_v1.py` (deterministic, no manual edits):

1. Verifies the frozen files against `FRESH_LATENCY_ARTIFACT_HASHES_V1.json` and all 289 shard artifacts against the frozen shard manifest; aborts on any mismatch.
2. Replays the executor aggregation from the shards and asserts byte identity with the frozen state-level, action-level and regime files, and equality of the primary block and bootstrap CI; aborts otherwise.
3. Builds `FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv` from the frozen CSV **at text level**: for each row only `mean_ref_latency` and `p95_ref_latency` are replaced by `repr(float(...))` of the `SBS_REFERENCE` row's `mean_latency` and `p95_latency`. All other fields are copied verbatim; the row set, order, identifiers, column set and column order are identical.
4. Cross-checks that this equals the replay with the corrected logic.
5. Writes `STATE_LEVEL_CORRECTION_AUDIT_V1.csv` (per state: original, corrected, delta, changed flags, and the counterfactual branch id the wrong value came from), `SBS_REFERENCE_ROWS_V1.csv`, and `CORRECTION_PROVENANCE_V1.json` (no timestamps, no absolute paths).
6. Hashes the frozen inputs before and after (`frozen_directory_unchanged_by_run = true`).

Side finding on numerical reading of these CSVs: pandas' default `read_csv` float parser is not correctly rounded. For the 720 `oracle_headroom` values, 448 differ from Python `float()` by up to 7.7e-13 relative. With `float_precision="round_trip"` the mean, positive mean and positive median equal the frozen JSON values bit-exactly. The frozen (post-correction) bootstrap CI is reproduced bit-exactly from the CSV as read by the default parser (consistent with the correction note, which recomputed it from the frozen artifact); an in-memory computation from exact floats differs by at most 3.0e-17 s at the endpoints. Use `float_precision="round_trip"` for exact comparisons.

## F. PROVENANCE

| Item | SHA-256 |
|---|---|
| Frozen `FRESH_LATENCY_STATE_LEVEL_V1.csv` (unchanged) | `d6c459ba382e0d8fdf5447d6b051aad878f7635b5dc348939b50418b851a7d8d` |
| Frozen `FRESH_LATENCY_ACTION_LEVEL_V1.csv` (unchanged) | `36f9105516a399cc41cac18867f0df18702170e52bc25477da5cab700d7c9dd4` |
| Frozen `FRESH_LATENCY_ARTIFACT_HASHES_V1.json` | `3e3c8a7a11e6d33abf9934f220dc6b4af7653c83fcf0246b64a162d019f7b0ee` |
| `FRESH_LATENCY_STATE_LEVEL_V1_CORRECTED.csv` | `23377f400426cc0f72911668a171e7da8ad5bc03c6f5620d294153bc98248242` |
| `SBS_REFERENCE_ROWS_V1.csv` | `7ed502941e9994f1685c87a6ae3afdba705947e2d9c49c12d3452ad5d81d4d0a` |
| `STATE_LEVEL_CORRECTION_AUDIT_V1.csv` | `da63788bdc11d16815281952b8c55940e627a1d92c1e9787c5962c71fa8c784f` |
| Shard manifest (289 entries, sorted-items digest) | `92fb31bf75cf353d1e2a238ba0a5578666b350e88c672084dda9c25295f0498a` |

Commits: original execution `38e02bcdaeb8f9cec2289b05dea9a1406318d14f` (executor script sha256 `e4ef52c38ce0412674319cdf2997f1934ab5385c031b227010eb6a913f24815b`); bootstrap cluster-key fix `a8fd735`; robustness analysis `5c8a78c4f1f4c9e92546c71bd3c0367cbb25b13a`. The full list of frozen-input hashes and the generator script hash are in `CORRECTION_PROVENANCE_V1.json`. A second local archive directory (`fresh-latency-confirmatory-v1`) holds only shard 0 (two of three files match the manifest; its `.json` differs); it is not used.

## G. CITATION / MANUSCRIPT GUIDANCE

* Cite the frozen files for all confirmatory numbers; nothing in the paper changes.
* Do not cite `mean_ref_latency` / `p95_ref_latency` from the frozen state-level CSV. Cite the corrected derivative if a per-state SBS reference latency is needed.
* Suggested sentence for Data/Code Availability or the repository README: "In the frozen state-level table, the descriptive columns `mean_ref_latency` and `p95_ref_latency` recorded the first counterfactual branch instead of the SBS reference; no reported result depends on them. A corrected derivative with per-state audit and provenance is provided in `experiments/fresh_production_latency_headroom_confirmatory_v1_corrected/`; the frozen confirmatory files are unchanged."
* If a public archive of the frozen files has already been published, add the corrected derivative and this note in its next version, and list the derivative in the release manifest. The release package under `release/performance_evaluation_v1/` still holds the original CSV.

## H. RECHECK OF THE ROBUSTNESS COMMIT (`5c8a78c`)

* The robustness module never reads `mean_ref_latency` or `p95_ref_latency`. Re-running the whole robustness pipeline on a copy of the frozen files with both columns removed reproduces **all 22 committed CSV/JSON/MD outputs byte-for-byte**.
* Where a reference latency was needed (supplementary relative headroom), it used `cf mean_latency + a_lat` from the action-level file. That differs from the true SBS reference rows by at most 1.94e-16 s. Recomputing the relative-headroom summary with the true references changes each value by at most 4.1e-16 (absolute; 5.7e-14 ms for the mean reference latency), so none of the reported numbers change.
* Threshold, distribution, weighting, leave-one-out, jackknife/bootstrap and composition results depend only on `oracle_headroom`, `beneficial_opportunity`, `max_a_lat`, and the window/regime keys, none of which are affected.
* Two wording corrections to the robustness report, applied in `docs/FRESH_CAUSAL_ROBUSTNESS_REPORT.md`: the state-level defect count is 694/720 exactly for the mean column (692 by more than 1e-12 s) and 224/720 for the p95 column; and the ten "within tolerance" reproduction items (max 5.6e-17 s) are caused by pandas' default float parsing, not by summation order.

## I. 590 vs 589 BENEFICIAL STATES

Independent verification from the raw shards (`A = SBS_REFERENCE.mean_latency - COUNTERFACTUAL.mean_latency`), agreeing with the state-level and action-level files:

* **Pre-specified rule, strict floating-point `max_a A > 0`: 590 / 720 = 0.819444.** This is the confirmatory result and is unchanged.
* One of the 590 states (`azure_2023_conv::w6::faithful::active_4::step22320`, one alternative) has `A = 5.551115123125783e-17` s. That is exactly **one unit in the last place** (`np.spacing`) of the 0.3571 s mean latencies being compared (SBS 0.3571018052261875 vs counterfactual 0.35710180522618745), and both branches complete the identical request set (same completed-ID hash). A second state has `A = -5.55e-17` s (residue of the same size).
* Latency differences in this simulator lie on a grid of `step/N` (step 0.001 s, N = 5 to 199 requests): the largest deviation of any branch from that grid is 3.8e-11 grid units, and the smallest genuine non-zero `|A|` is 5.15e-6 s. The residue is 11 orders of magnitude below it.
* Guarded count, `max_a A > 1e-12` s (post hoc numerical tolerance): **589 / 720 = 0.818056.** The residue's effect on the mean headroom is 7.7e-20 s.

**Recommended manuscript wording** (preserves both facts; the pre-specified analysis is reported as run):

> "Under the pre-specified criterion (strict `max_a A(s,a) > 0` in double precision), 590 of 720 disagreement states were beneficial (P = 0.8194). One of these 590 has an advantage of 5.6×10⁻¹⁷ s, a single unit in the last place of the latencies compared (both branches complete identical requests); all other non-zero advantages exceed 5×10⁻⁶ s. Excluding it with a post hoc numerical tolerance of 10⁻¹² s gives 589/720 (0.8181). The primary endpoint, mean oracle headroom, is unaffected (1.9958 ms)."

Report 590/720 as the headline in the abstract and results; give 589/720 only in this sentence or a table footnote, labeled post hoc.
