# Family-A Wulver DEV Support Evaluation v1 Analysis
Date: 2026-08-24
Scope: the single precommitted frozen DEV support evaluation after the Wulver target-free sweep. DEV was used only as a fixed support target; no labels, oracle acquisition, retraining, new simulations, FINAL, TEST, D2, GPUs, or git mutation were used.
## Classification
Final support classification: `WULVER_DEV_SUPPORT_NO_GO`
Oracle acquisition allowed next: NO
## Frozen Contract
Primary formal view: `FEATURE_V1`. Baseline: V2 `TRAIN_COLLECTION` + D1 TRAIN-only support. Scaler: baseline-only mean/std with std floor 1.0 for near-constant columns. Distance: Euclidean NN in standardized feature space. Local density: mean distance to 5 nearest support rows. Top-gap features are the five predeclared source features. Group improvement means lower group mean NN plus at least one row closer.
## Support Counts
- baseline_a_v2_train_rows: 748
- baseline_b_v2_train_plus_d1_rows: 998
- baseline_b_unique_fingerprints: 998
- wulver_raw_candidate_rows: 30039
- wulver_unique_fingerprints_total: 24314
- wulver_duplicate_fingerprints: 5725
- wulver_unique_new_rows_added: 24314
- wulver_unique_duplicates_of_baseline: 0
- expanded_support_rows: 25312
- expanded_support_unique_fingerprints: 25312
## DEV Set
- Rows: 104
- Groups: 4
- Favored-long rows: 66
- Favored-short rows: 38
- `util1.1000.skew5.0000.favlong.noise0.30.n120.maxseq1`: 66
- `util1.1000.skew5.0000.favshort.noise0.00.n120.maxseq1`: 23
- `util1.3000.skew10.0000.favshort.noise0.00.n120.maxseq1`: 7
- `util1.3000.skew10.0000.favshort.noise0.30.n120.maxseq1`: 8
## Primary FEATURE_V1 Results
- Mean NN: 1.776890 -> 1.771298 (0.315% improvement)
- Median NN: 1.702413 -> 1.702413
- p90 NN: 3.257824 -> 3.257824 (0.000% improvement)
- Rows closer/unchanged/worse: 1 / 103 / 0
- Nearest source after expansion: {'V2_TRAIN': 102, 'D1_TRAIN': 1, 'WULVER': 1}
## Favored Side
- Favored-long mean NN: 1.511606 -> 1.502795 (0.583%); closer 1/66.
- Favored-short mean NN: 2.237646 -> 2.237646 (0.000%); closer 0/38.
## Per-Group Results
- `util1.1000.skew5.0000.favlong.noise0.30.n120.maxseq1` (66 rows): mean 1.511606->1.502795, improvement 0.583%, rows closer 1, nearest-Wulver fraction 0.015.
- `util1.1000.skew5.0000.favshort.noise0.00.n120.maxseq1` (23 rows): mean 2.419971->2.419971, improvement 0.000%, rows closer 0, nearest-Wulver fraction 0.000.
- `util1.3000.skew10.0000.favshort.noise0.00.n120.maxseq1` (7 rows): mean 1.616350->1.616350, improvement 0.000%, rows closer 0, nearest-Wulver fraction 0.000.
- `util1.3000.skew10.0000.favshort.noise0.30.n120.maxseq1` (8 rows): mean 2.257096->2.257096, improvement 0.000%, rows closer 0, nearest-Wulver fraction 0.000.
## Density And Gap Overlap
- Local 5NN: 2.593093 -> 2.566156 (1.039% improvement).
- Top-gap overlap improved features: 1/5.
## Diagnostics
- V2-normalized: mean improvement 0.112%, p90 improvement 0.000%, closer rows 1, favored-long improvement 0.199%, improved groups 1.
- Task subspace: mean improvement 0.000%, p90 improvement 0.000%, closer rows 0, favored-long improvement 0.000%, improved groups 0.
## Wulver Nearest Source
- DEV rows whose nearest support became Wulver: 1. Improved DEV rows nearest to Wulver: 1. Unique responsible Wulver cells: 1. Movement shape: CONCENTRATED.
- `util1.0000.skew5.0000.favlong.noise0.30.n120.maxseq1`: 1 improved DEV rows.
## Gate Table
| Metric | Threshold | Observed | PASS/FAIL |
|---|---:|---:|---|
| DEV NN mean >= 5% improvement | >= 5% | 0.31468866951099234 | FAIL |
| DEV NN p90 >= 5% improvement | >= 5% | 0.0 | FAIL |
| favored-long DEV NN mean >= 7% improvement | >= 7% | 0.5828978057339095 | FAIL |
| DEV closer rows >= 15/104 | >= 15/104 | 1/104 | FAIL |
| favored-long closer rows >= 8/66 | >= 8/66 | 1/66 | FAIL |
| local 5NN density >= 3% improvement | >= 3% | 1.0388066421745037 | FAIL |
| top-gap overlap >= 3/5 features | >= 3/5 | 1/5 | FAIL |
| >= 2 DEV configuration groups improve | >= 2 groups | 1 | FAIL |
## Decision
The exact classification is `WULVER_DEV_SUPPORT_NO_GO`. The complete predeclared gate does not pass unless every table row passes. With only 1 / 104 DEV rows closer, 0.31% mean NN improvement, 0% p90 improvement, and only one DEV group improved, the target-free expansion did not materially move DEV support. Next scientific action: stop/reconsider the current Family-A selector-support direction; do not label automatically.
## Confirmation
- DEV used only once for frozen support evaluation.
- No DEV-driven redesign.
- No FINAL.
- No TEST.
- No oracle labels.
- No retraining.
- No D2.
- No new simulations.
- No GPUs.
- No git mutation.
