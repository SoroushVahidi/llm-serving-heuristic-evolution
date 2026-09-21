# DENOMINATOR_COMPLETION_RESULT_V1

**Classification: POST_HOC_DENOMINATOR_COMPLETION** (see `DENOMINATOR_COMPLETION_V1.md`; not preregistered, frozen after the
sensitivity results were observed). **Status: `DENOMINATOR_COMPLETION_SUCCESS`.**

| Item | Value |
|---|---|
| Denominator job | Slurm 1303832, `COMPLETED`, exit `0:0`, elapsed 00:22:40, MaxRSS 257,640 KB, node n0006 |
| Executed from | commit `ec1996354de0304dd6ea521c5d584a838401b06a` (clean tree, `git_dirty_tracked_files: []`) |
| Sensitivity implementation | commit `8addcdb1368f9cb3057d5d3823c8a273e52dcc1f`; jobs 1303715 / 1303716 / 1303717 (0.82 / 0.90 / 1.00), all `COMPLETED 0:0` |
| Protocol sha256 | `97b06cf1c96fe6b3f84aaf4355643192a0b400ce206f4c7c041b95d6bc417439` |
| Amendment sha256 | `c58734e974b8c2cbfca8cb3dcdff5a3aac392abe2e99d5f035427cce13793073` |
| Counting runner sha256 | `d7c2b2985828a9eef52e1067ad12fd6a945da77c49f0e15cc797eed90e7c2427` |
| Derive stage | same runner, unmodified, run locally on the hash-verified cluster copies; 21 existing-result hashes re-verified |

No counterfactual branch was executed at any stage (`no_counterfactual_execution: true`); every causal quantity below is read from the
completed sensitivity results.

## Gates

* **DENOMINATOR_TRAJECTORY_INTEGRITY_GATE = PASS.** Recounted disagreement states 720 / 466 / 198; 100 scenarios per reserve; 0 per-scenario
  mismatches; `decision_states` equals the simulator step-call invariant everywhere.
* **DENOMINATOR_REPRODUCTION_GATE = PASS** (reserve 0.82 vs exact canonical `FRESH_SUPPORT_WINDOW_CONDITIONS_V1.csv` /
  `FRESH_LATENCY_WORKLOAD_REGIME_V1.csv`). 100 scenarios compared, 0 window-level mismatches, 0 regime-level mismatches,
  total decisions 1,470,515 = 1,470,515, max |P_D difference| = 0.0.

## Final metrics (simulated ms; causal quantities from the completed results)

| Metric | 0.82 | 0.90 | 1.00 |
|---|---:|---:|---:|
| Total reference decision states | 1,470,515 | 1,470,447 | 1,470,427 |
| Disagreement states | 720 | 466 | 198 |
| P(D) | 0.04896% | 0.03169% | 0.01347% |
| Beneficial disagreement states | 590 | 353 | 99 |
| P(B_LAT given D) | 81.94% | 75.75% | 50.00% |
| State-weighted mean H_LAT | 1.996 | 3.022 | 0.119 |
| Clustered 95% CI (2,000 replicates, seed 42) | [0.182, 3.563] (36 clusters) | [0.067, 4.547] (34) | [0.042, 0.232] (32) |
| Median H_LAT (all disagreement states) | 0.197 | 0.172 | 0 (2.8e-14, floating residue) |
| Positive-state median H_LAT | 0.288 | 0.976 | 0.050 |
| Total headroom mass | 1,436.97 | 1,408.09 | 23.56 |
| Opportunity-weighted headroom (per decision) | 9.772e-4 | 9.576e-4 | 1.603e-5 |
| Beneficial decision rate | 0.04012% | 0.02401% | 0.006733% |
| All-alternatives-harmful | 102 (14.2%) | 91 (19.5%) | 82 (41.4%) |
| Dominant window headroom share | 88.4% (azure_code w11) | 92.8% (azure_code w11) | 36.8% (azure_conv w3) |
| Dominant regime headroom share | 96.8% (Azure-code KV 16,000) | 97.9% (Azure-code KV 16,000) | 58.9% (Azure-conv active cap 4) |

Ratios to reserve 0.82 (0.90 / 1.00): decision states 0.99995 / 0.99994; P(D) 0.647 / 0.275; beneficial decision rate 0.598 / 0.168;
opportunity-weighted headroom 0.980 / 0.0164; total headroom mass 0.980 / 0.0164.

Note on the 0.82 CI: this experiment bootstrapped with seed 42 and reports [0.182, 3.563] ms; the manuscript's confirmatory interval
[0.1909, 3.5462] ms uses its own pre-specified seed. The point estimates (mean 1.9958 ms, median 0.197 ms, shares) are identical.
Use the canonical interval for the primary result and this one only inside the reserve table, with its seed stated.

## Per-regime denominators (identical rows for the three active-cap regimes at every reserve)

| Regime | Reserve | Decision states | Disagreement | P(D) | Beneficial | Mean conditional H_LAT |
|---|---|---:|---:|---:|---:|---:|
| Azure code, active cap 8 | all | 94,407 | 10 | 0.01059% | 4 | 0.033 |
| Azure code, active cap 4 | all | 94,595 | 122 | 0.1290% | 64 | 0.077 |
| Azure conv, active cap 4 | all | 594,842 | 64 | 0.01076% | 31 | 0.217 |
| Azure code, KV 16,000 | 0.82 / 0.90 / 1.00 | 94,495 / 94,427 / 94,407 | 431 / 224 / 2 | 0.4561% / 0.2372% / 0.0021% | 398 / 208 / 0 | 3.227 / 6.156 / 0 |
| Azure conv, KV 16,000 | 0.82 / 0.90 / 1.00 | 592,176 (all) | 93 / 46 / 0 | 0.01570% / 0.00777% / 0% | 93 / 46 / 0 | 0.241 / 0.121 / 0 |

The three active-cap regimes are invariant across reserves in every recorded quantity (decisions, disagreements, beneficial states,
headroom mass). At reserve 1.00 their headroom (9.36 + 0.33 + 13.88 = 23.56 ms) is the entire remaining mass.

## Interpretation

1. P(D) falls 0.0490% -> 0.0317% -> 0.0135% (x0.647, x0.275).
2. Opportunity-weighted headroom is nearly unchanged at 0.90 (x0.980) and falls to 1.6% at 1.00.
3. The beneficial decision rate falls to 60% at 0.90 and 17% at 1.00.
4. The earlier sensitivity conclusion holds: the dominant KV-capacity opportunity survives at 0.90 (mass -0.9% in the Azure-code KV regime,
   with fewer but larger-value states: 431 -> 224 states, mean 3.2 -> 6.2 ms), nearly disappears at 1.00 (2 disagreement states, none beneficial, zero mass in both
   KV regimes), and the active-cap opportunity is reserve-invariant.
5. Total headroom mass was a directionally and quantitatively adequate surrogate: the decision denominator varies by < 0.006% across reserves,
   so mass ratios (0.9799, 0.01640) equal opportunity-weighted ratios (0.9799, 0.01640) to four significant figures.
6. The denominators normalize the result; they do not change it. They add scale: about one disagreement state per 2,000 reference decisions
   at 0.82. Pooled rates weight the two Azure-conversation regimes by their many more decisions (about 594,000 per regime versus 94,000),
   so regime-level rates are the more interpretable ones.

## Recommended manuscript wording (for later integration; manuscript not edited here)

> Varying only the reference's KV reserve, the opportunity is not specific to the default reserve of 0.82: at 0.90 the total oracle headroom
> mass is 98% of its 0.82 value (1,408 vs 1,437 sim. ms) although the disagreement rate P(D) and the beneficial-decision rate fall to 65% and 60%.
> It does depend strongly on the reference holding a sub-capacity reserve: at 1.00 the two KV-capacity regimes yield 2 disagreement states,
> none beneficial, and the total headroom mass falls to 1.6% of its 0.82 value (23.6 sim. ms), all from the three active-sequence-cap regimes,
> whose results are identical at every reserve. This experiment changes only the reserve; it does not isolate the KV-footprint ordering, and it
> does not establish that the opportunity is invariant to other reference schedulers.

## Preservation

* `results_v1/reserve_{082,090,100}/`: 6 small files each (summaries, provenance, regime/window tables, `disagreement_states.csv`). 18 files
  verified against `denominator_completion_v1/EXISTING_SENSITIVITY_RESULT_HASHES_V1.json`. The three `action_effects.csv` files stay in cluster
  scratch and are identified by hash only.
* `denominator_completion_v1/`: `counts_reserve_*.csv`, `count_execution_provenance.json`, `DENOMINATOR_*_V1.*`,
  `FINAL_RESERVE_METRICS_V1.json`, `FINAL_RESERVE_METRICS_BY_REGIME_V1.csv`, `final_metric_table_v1.py` (read-only post-derive statistics).
* `slurm/`: sbatch scripts and job logs for jobs 1303715-1303717 and 1303832.
* `PRESERVATION_MANIFEST_V1.json`: sha256 of every preserved file.

`SCIENTIFIC_EXPERIMENT_PHASE = CLOSED`. `NO_FURTHER_SCIENTIFIC_EXPERIMENTS_PLANNED`. The next step is manuscript integration and submission preparation.
