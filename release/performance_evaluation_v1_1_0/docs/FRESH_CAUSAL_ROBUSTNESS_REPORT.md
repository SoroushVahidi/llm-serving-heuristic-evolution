# Fresh Causal Latency-Headroom: Post-hoc Robustness Report

**Status of every analysis below: POST-HOC robustness / sensitivity.** None of the practical thresholds, weightings, leave-one-out, jackknife, BCa or extra-replicate bootstrap analyses were frozen in `PREREGISTRATION_V1.json` or `METRIC_PROTOCOL_V1.json`. They must not be called preregistered or confirmatory. The canonical confirmatory result is unchanged; nothing here redefines the regimes, the estimand, the policy portfolio, or the 720-state population. No manuscript edits were made.

Reproduce everything with `python scripts/fresh_causal_robustness_v1.py` (about 8 s, one core). Tests: `pytest tests/test_fresh_causal_robustness.py`. Machine-readable outputs are in `experiments/fresh_production_latency_headroom_confirmatory_v1_robustness/`.

---

## 1. CANONICAL_INPUTS

| Item | Value |
|---|---|
| Repository / branch | `SoroushVahidi/llm-serving-heuristic-evolution`, `main` |
| Git HEAD at analysis time | `f89749290af7cd452e746714a10357f2dfb0dde2` (`f897492`, "paper: fix table layout in final manuscript"); working tree clean before this work |
| Canonical artifact directory | `experiments/fresh_production_latency_headroom_confirmatory_v1/` (read-only) |
| Generating script | **`scripts/fresh_latency_causal_confirmatory_v1.py`** (original execution at commit `38e02bc`, script commit `b196c3e`; bootstrap cluster-key bug fix at `a8fd735`) |
| Frozen specification | `PREREGISTRATION_V1.json`, `METRIC_PROTOCOL_V1.json`, `CAUSAL_PROTOCOL_V1.json`, design doc `docs/design/FRESH_PRODUCTION_LATENCY_HEADROOM_CONFIRMATORY_V1.md` |
| Post-outcome correction note | `BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md` (bare `window_index` 26 clusters to `(source_dataset, window_index)` 36 clusters; verified below) |
| Manuscript | `paper/performance_evaluation/main.tex` (the `submission/main.tex` copy differs only in table layout; all fresh-result numbers are identical) |

**Identifiers.**

* Workload = `source_dataset` (`azure_2023_code`, `azure_2023_conv`; BurstGPT contributed no selected regime).
* Window = `(source_dataset, window_index)`, the faithful 200-request source window. `window_index` is source-local, so it must be qualified by workload. 36 windows: 19 code and 17 conversation.
* Regime = `(source_dataset, axis, condition_id)`. There are 5: code/active-4, code/active-8, code/KV-16,000, conv/active-4, conv/KV-16,000. A window can belong to several regimes (49 window-regime cells over 36 windows).
* State = `state_id` (`phase_a::FRESH_LATENCY_SUPPORT::<workload>::w<k>::faithful::<condition>::step<n>`), 720 total.
* Action = `branch_id` / `candidate_canonical_action_id`, 831 non-SBS counterfactual branches (1 to 3 per state).
* Outcome = mean request latency (seconds) over the request population not yet completed at the intervention step, with the SBS continuation after one forced action.

**Estimand (from `METRIC_PROTOCOL_V1.json` and the executor).**

* `A(s,a) = L_SBS(s) - L_CF(s,a)`. Positive means the alternative lowers latency.
* `H(s) = max(0, max_a A(s,a))` (`oracle_headroom`).
* `B(s) = 1[max_a A(s,a) > 0]`. The comparison is strict, with **no tolerance**.
* Primary endpoint = `mean_s H(s)` over all 720 states, zeros included.

**Canonical bootstrap (verified from the executor source).**

* Cluster = `(source_dataset, window_index)`, 36 clusters.
* Each replicate draws `rng.integers(0, 36, 36)` from `np.random.default_rng(20260920)`, with replacement at the cluster level. All states of each drawn window are retained with multiplicity.
* The statistic is the pooled mean of `oracle_headroom` over the concatenated sample. The CI is `np.quantile(vals, [.025, .975])` (linear interpolation) over 2,000 replicates.

**Input checksums (SHA-256, all match `FRESH_LATENCY_ARTIFACT_HASHES_V1.json` where that file lists them).**

| File | SHA-256 | in frozen hash file? |
|---|---|---|
| `BOOTSTRAP_CLUSTER_KEY_CORRECTION_20260920.md` | `0854a8ec7d8d19732e136b9d10ad1cc935075acd1d8df15f36633f3a997262d5` | n/a |
| `CAUSAL_PROTOCOL_V1.json` | `136bcc8aefa0589238f6fd7feecea8eeafa7aedd0e48b90168e57381e6e05e71` | n/a |
| `FRESH_LATENCY_ACTION_LEVEL_V1.csv` | `36f9105516a399cc41cac18867f0df18702170e52bc25477da5cab700d7c9dd4` | yes, matches |
| `FRESH_LATENCY_ARTIFACT_HASHES_V1.json` | `3e3c8a7a11e6d33abf9934f220dc6b4af7653c83fcf0246b64a162d019f7b0ee` | n/a |
| `FRESH_LATENCY_BOOTSTRAP_V1.json` | `92588fccfdb32b75c14804ed05068b9d102ee92ecc5975a23c32f2cee3096da2` | yes, matches |
| `FRESH_LATENCY_CAUSAL_RESULT_V1.json` | `c7104c50075ffefab87c2784adbe94fcc546cd220a9db69ac74dd46992cf57da` | yes, matches |
| `FRESH_LATENCY_CONFIRMATORY_REPORT_V1.md` | `06c15d5c4688319867a4d6388c6db3c7076b8131a46d5a7d0c2e1810caed94ab` | yes, matches |
| `FRESH_LATENCY_EXECUTION_PROVENANCE_V1.json` | `beec10ab87bb082bc89f8d6829a8849dd8dfea153ab12d428b67db39ff030800` | yes, matches |
| `FRESH_LATENCY_STATE_LEVEL_V1.csv` | `d6c459ba382e0d8fdf5447d6b051aad878f7635b5dc348939b50418b851a7d8d` | yes, matches |
| `FRESH_LATENCY_WORKLOAD_REGIME_V1.csv` | `503f2ba1b7a686b487f21deb2b920de3798332931e55a8f7323f0871b369a4dd` | yes, matches |
| `FRESH_REGIME_SELECTION_V1.json` | `d2e176a3e3d74090f0674ba141f60cbbdb0b53f6cad185126e927e71b82f45e7` | n/a |
| `METRIC_PROTOCOL_V1.json` | `aa71789590740553a3128062860066d2c51ff5c7940aed51a944f836817dc48e` | n/a |
| `PREREGISTRATION_V1.json` | `2fe1abb3d220050a69208eb93aeb19ac297785a5770d4d3c4196fb26691f4ed2` | n/a |

Canonical files were hashed before and after the analysis run and were byte-identical (`provenance.json: canonical_inputs_unchanged_by_run = true`). `git status` shows no modification under `experiments/fresh_production_latency_headroom_confirmatory_v1/`.

**Two artifact-level observations (documented, not fixed; canonical data must not be modified).**

1. **State-level `mean_ref_latency` and `p95_ref_latency` are mislabeled.** In the executor, `meta = g.iloc[0]` is taken from the *counterfactual* group, so these columns hold the *first counterfactual branch's* values, not the SBS reference. They match the first counterfactual branch exactly (max diff 0.0) and differ from the true SBS reference in 694 of 720 states for `mean_ref_latency` (692 by more than 1e-12 s) and 224 of 720 for `p95_ref_latency`; see `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`, which supersedes this note. **The primary endpoint is unaffected**, because `a_lat` was computed from the separate `ref` frame. The regime-level `p95_effect_mean` also uses the correct `ref`. I recovered the true reference as `cf_mean_latency + a_lat` (constant within each state to 2.2e-16 s; max 1.9e-16 s from the true SBS reference rows in the shards) and used that for the supplementary relative-headroom numbers only.
2. **The canonical strict `> 0` rule counts one floating-point-residue state as beneficial** (see section 3 for the tolerance audit).

**Compute estimate.** All analyses are vectorised numpy on 720 rows and 36 clusters. The heaviest step (3 x 10^6 bootstrap replicates plus 1,000 x 2,000 Monte-Carlo-noise replicates plus 200,000-replicate stages) took about 8 s of wall-clock time in total. This is far below the "few minutes" threshold, so no tmux session and no Wulver/SLURM job was launched (see the final response).

---

## 2. REPRODUCTION_CHECK

**Result: REPRODUCED. 42 checks, 32 bit-exact, 10 within floating-point tolerance (max difference 5.6e-17 s; cause: pandas' default CSV float parser is not correctly rounded, see `docs/FRESH_CAUSAL_ARTIFACT_CORRECTION.md`), 0 failed.** The analysis is coded to refuse to run the sensitivity analyses if this check fails.

| Reported (manuscript / frozen result) | Recomputed from canonical artifact | Status |
|---|---|---|
| 720 states; 590 beneficial; P = 590/720 = 0.8194 | 720; 590 (`beneficial_opportunity`); 590 (from `H > 0`) | exact |
| Mean H = 0.001995791 s = **1.9958 ms** | 0.0019957911708132692 s | within 2.9e-17 s (CSV float-parse rounding; bit-exact with `float_precision="round_trip"`); 4-dp ms value exact |
| Clustered 95% CI [0.1909, 3.5462] ms; 36 clusters | canonical loop re-run: [0.00019088421760325944, 0.003546195777055185] s, **bit-exact**; vectorised engine within 2e-18 s | exact / within tolerance |
| Positive-headroom mean 2.4355 ms, median 0.2879 ms | reproduced | within tolerance |
| 102 all-harmful / 26 all-zero / 22 mixed | reproduced | exact |
| Regime table (states, P(B), mean H, contributing windows), 5 regimes x 4 quantities | all reproduced | exact / within tolerance |
| `H_state` vs `max_a a_lat` from the *action-level* file (independent reconstruction) | max abs difference 0.0 | exact |
| Documented pre-correction key: 26 clusters, [0.184589, 3.497418] ms | bare `window_index` key gives 26 clusters, [0.184589, 3.497418] ms (to 1e-6 ms) | confirms the correction note |

The manuscript's numbers therefore match the canonical artifact, and the correction note's description of the pre-correction result is independently confirmed. Full detail is in `reproduction_check.json`.

---

## 3. PRACTICAL_THRESHOLD_RESULTS

**Units.** All artifact values are seconds; thresholds in ms were converted by x 1e-3 explicitly (`H_ms = 1e3 x oracle_headroom`). Unit and scale were verified from the data (means of about 0.1 s; smallest genuine advantage 5.15e-6 s).

**Numerical tolerance inspection (canonical result unchanged).**

* **The advantage lives on an exact grid.** Every `A(s,a)` equals `integer x step / N_s` (step = 0.001 s decode step, `N_s` = state request population, 5 to 199). Across all 831 branches the maximum deviation from the integer grid is 3.8e-11 grid units, and no branch is off-grid by more than 1e-6 units.
* **Genuine advantages are far above the noise floor.** The smallest genuine non-zero `|A|` is 5.15e-6 s (= 0.001/194) and the largest possible step/N is 5.03e-6 s. Values below about 5e-6 s can only be floating-point residue.
* **Floating-point residue does exist.** Exactly two branches carry `|A|` = 5.55e-17 s (= 2^-54): one is +5.55e-17 (state `azure_2023_conv::w6::...::step22320`), one is -5.55e-17 (state `azure_2023_conv::w26::...::step25692`).
* **Consequence for the canonical count.** The positive residue makes one state "beneficial" under the strict `> 0` rule (H = 5.55e-17 s), so **the canonical 590 includes 1 noise state; the tolerance-guarded count is 589 (81.81%).**
* **Ties at practical thresholds are real.** Several states sit within 1e-9 s of a threshold. One at 0.5 ms has H = 0.0005000000000001 s and a raw `>` comparison counts it as exceeding 0.5 ms, although it equals 0.5 ms. Others sit at 0.1, 0.25, 1 and 2 ms.
* **An epsilon is scientifically warranted for threshold comparisons.** I used ε = 1e-12 s (1e-9 ms), and "exceeds τ" means `H - τ > ε`. Any ε in [1e-15, 1e-7] s gives identical classifications, because there is an 11-orders-of-magnitude gap between the residue (5.6e-17) and the smallest genuine effect (5.2e-6).
* **Why the canonical value stays as reported.** The canonical `590/720 = 0.8194` is what was frozen. The guarded 589/720 is reported alongside it as a sensitivity. The pooled-mean difference is about 1e-17 s, so the primary endpoint is unaffected.

**Global exceedance fractions** (states whose best-alternative improvement exceeds τ; numerators and denominators shown; CI = post-hoc percentile window-cluster bootstrap, 200,000 replicates, seed 20260920).

| tau (ms) | states > tau | fraction | 95% window-cluster CI | windows with >=1 exceeding state | raw-FP strict count | ties (|H-tau|<=1e-9 s) |
|---:|---|---|---|---|---:|---:|
| 0 | 589 / 720 | 0.8181 | [0.699, 0.884] | 32 / 36 | 590 | 131 |
| 0.1 | 432 / 720 | 0.6000 | [0.371, 0.678] | 17 / 36 | 432 | 1 |
| 0.25 | 322 / 720 | 0.4472 | [0.237, 0.552] | 16 / 36 | 322 | 2 |
| 0.5 | 212 / 720 | 0.2944 | [0.030, 0.477] | 9 / 36 | 213 | 1 |
| 1 | 188 / 720 | 0.2611 | [0.019, 0.436] | 7 / 36 | 188 | 2 |
| 2 | 162 / 720 | 0.2250 | [0.003, 0.396] | 4 / 36 | 162 | 1 |
| 5 | 129 / 720 | 0.1792 | [0.000, 0.347] | 2 / 36 | 129 | 0 |

Column notes: "raw-FP strict count" is a plain `H > τ` floating-point comparison, shown only to expose the noise and tie effects (590 vs 589 at τ = 0; 213 vs 212 at 0.5 ms). The "ties" column counts states within 1e-9 s of τ (at τ = 0 these are the 131 states with H ≈ 0).

**By workload and by regime** (n/denominator, fraction). Regimes flagged "low support" have fewer than 30 states or fewer than 5 windows; they are shown for completeness, not omitted.

| scope | n | windows | >0 ms | >0.1 ms | >0.25 ms | >0.5 ms | >1 ms | >2 ms | >5 ms |
|---|---:|---:|---|---|---|---|---|---|---|
| Azure code | 563 | 19 | 466/563 (0.828) | 348/563 (0.618) | 269/563 (0.478) | 207/563 (0.368) | 185/563 (0.329) | 161/563 (0.286) | 128/563 (0.227) |
| Azure conversation | 157 | 17 | 123/157 (0.783) | 84/157 (0.535) | 53/157 (0.338) | 5/157 (0.032) | 3/157 (0.019) | 1/157 (0.006) | 1/157 (0.006) |
| Code, active cap 8 (low support) | 10 | 2 | 4/10 (0.400) | 2/10 (0.200) | 0/10 (0.000) | 0/10 (0.000) | 0/10 (0.000) | 0/10 (0.000) | 0/10 (0.000) |
| Code, active cap 4 | 122 | 15 | 64/122 (0.525) | 19/122 (0.156) | 10/122 (0.082) | 6/122 (0.049) | 0/122 (0.000) | 0/122 (0.000) | 0/122 (0.000) |
| Code, KV 16,000 | 431 | 14 | 398/431 (0.923) | 327/431 (0.759) | 259/431 (0.601) | 201/431 (0.466) | 185/431 (0.429) | 161/431 (0.374) | 128/431 (0.297) |
| Conv, active cap 4 | 64 | 17 | 30/64 (0.469) | 10/64 (0.156) | 8/64 (0.125) | 5/64 (0.078) | 3/64 (0.047) | 1/64 (0.016) | 1/64 (0.016) |
| Conv, KV 16,000 (low support) | 93 | 1 | 93/93 (1.000) | 74/93 (0.796) | 45/93 (0.484) | 0/93 (0.000) | 0/93 (0.000) | 0/93 (0.000) | 0/93 (0.000) |

**Reading the table.**

* The "81.9% beneficial" figure is dominated by *small* effects. 60.0% of states beat SBS by more than 0.1 ms, 29.4% by more than 0.5 ms, 22.5% by more than 2 ms and 17.9% by more than 5 ms.
* Large effects are geographically concentrated: only **4 of 36 windows** contain any state above 2 ms and only **2 of 36** contain a state above 5 ms (128 of the 129 states above 5 ms are in one window, `azure_2023_code:w11`; see section 8).
* The active-sequence-cap regimes almost never exceed 1 ms: 0/122, 0/10 and 3/64. The conv/KV-16,000 regime is beneficial in every one of its 93 states, but never by more than 0.48 ms and all from a single window.

---

## 4. DISTRIBUTIONAL_RESULTS

Oracle headroom `H(s)` over disagreement states (all values in ms; SD is the sample SD; percentiles are numpy-linear).

| scope | N | mean_ms | median_ms | sd_ms | p25_ms | p75_ms | p90_ms | p95_ms | max_ms | n_exact_zero | frac_exact_zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All 720 states | 720 | 1.996 | 0.1969 | 3.623 | 0.0228 | 1.186 | 8.405 | 11.3 | 12.12 | 130 | 0.1806 |
| Azure code | 563 | 2.488 | 0.2216 | 3.944 | 0.02685 | 3.409 | 10.38 | 11.4 | 12.12 | 97 | 0.1723 |
| Azure conversation | 157 | 0.2313 | 0.1289 | 0.6664 | 0.01176 | 0.3026 | 0.4226 | 0.4626 | 8.091 | 33 | 0.2102 |
| Code, active cap 8 | 10 | 0.03343 | 0 | 0.05169 | 0 | 0.05362 | 0.1227 | 0.1239 | 0.125 | 6 | 0.6 |
| Code, active cap 4 | 122 | 0.07668 | 0.01023 | 0.1883 | 0 | 0.05671 | 0.1797 | 0.2851 | 0.9756 | 58 | 0.4754 |
| Code, KV 16,000 | 431 | 3.227 | 0.3861 | 4.24 | 0.1108 | 7.366 | 11.13 | 11.54 | 12.12 | 33 | 0.07657 |
| Conv, active cap 4 | 64 | 0.2168 | 0 | 1.035 | 0 | 0.04545 | 0.264 | 0.6596 | 8.091 | 33 | 0.5156 |
| Conv, KV 16,000 | 93 | 0.2413 | 0.2423 | 0.1382 | 0.1237 | 0.359 | 0.4297 | 0.4533 | 0.4769 | 0 | 0 |

Among the states with strictly positive headroom (above ε):

| scope | n_positive_above_eps | pos_mean_ms | pos_median_ms | pos_p25_ms | pos_p75_ms | pos_p90_ms | pos_p95_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| All 720 states | 589 | 2.44 | 0.2887 | 0.08763 | 3 | 9.903 | 11.39 |
| Azure code | 466 | 3.006 | 0.3428 | 0.09845 | 6.94 | 11.04 | 11.5 |
| Azure conversation | 123 | 0.2953 | 0.2113 | 0.06443 | 0.3462 | 0.44 | 0.4713 |
| Code, active cap 8 | 4 | 0.08357 | 0.09314 | 0.05362 | 0.1231 | 0.1242 | 0.1246 |
| Code, active cap 4 | 64 | 0.1462 | 0.05424 | 0.02601 | 0.1275 | 0.2818 | 0.8148 |
| Code, KV 16,000 | 398 | 3.495 | 0.5455 | 0.1622 | 7.713 | 11.2 | 11.57 |
| Conv, active cap 4 | 30 | 0.4625 | 0.04572 | 0.01891 | 0.2468 | 0.7156 | 1.401 |
| Conv, KV 16,000 | 93 | 0.2413 | 0.2423 | 0.1237 | 0.359 | 0.4297 | 0.4533 |

**Zero accounting for the canonical `max(0, ·)` definition.** 130 of 720 states (18.06%) are exactly zero (non-beneficial). One further state is positive at noise level (5.55e-17 s), leaving 589 states (81.81%) with genuine positive headroom.

**Shape.** The distribution is extremely right-skewed. The pooled mean (1.996 ms) is about **10x the median (0.197 ms)**, and 77.4% of states (557 of 720) lie at or below the mean. The 90th percentile is 8.4 ms. State-level concentration: the top 10% of states carry 54.6% of all headroom, the top 20% carry 88.2%, and 65 states carry half of the total (Gini = 0.778).

**Figure 1** (`figures/fig1_headroom_distribution.pdf`): (a) ECDF of H by regime (Okabe-Ito colours, plus distinct line styles for grayscale), with threshold guide lines at 0.1, 0.25, 0.5, 1, 2, 5 ms and the pooled mean; the vertical rise at x = 0 is the exact-zero mass; (b) the state ECDF against the cumulative share of total headroom, which shows that mass sits in the far right tail.

**Signed advantage of alternatives before truncation** (831 branches; descriptive only, *not* the primary estimand):

| scope | n_actions | n_positive | n_negative | n_zero_within_eps | mean_ms | median_ms | p05_ms | p95_ms | best_gain_ms | worst_harm_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All 831 branches | 831 | 659 | 140 | 32 | 1.717 | 0.1804 | -0.362 | 11.18 | 12.12 | -7.568 |
| Code, active cap 8 | 16 | 5 | 6 | 5 | 0.008119 | 0 | -0.091 | 0.1231 | 0.125 | -0.1087 |
| Code, active cap 4 | 155 | 72 | 76 | 7 | -0.2642 | 0 | -1.164 | 0.372 | 0.9756 | -7.568 |
| Code, KV 16,000 | 498 | 458 | 27 | 13 | 2.903 | 0.3806 | -0.005719 | 11.48 | 12.12 | -1.467 |
| Conv, active cap 4 | 69 | 31 | 31 | 7 | -0.01247 | 0 | -1.224 | 0.6272 | 8.091 | -2.86 |
| Conv, KV 16,000 | 93 | 93 | 0 | 0 | 0.2413 | 0.2423 | 0.02887 | 0.4533 | 0.4769 | 0.005155 |

Of 831 forced alternatives, 659 improve latency, 140 worsen it and 32 are within ε of zero. The best single gain is 12.12 ms and the worst harm is -7.57 ms (code/active-4), so the harm tail is not negligible. Mean signed advantage in the two active-cap-4 regimes is about zero or negative (code -0.264 ms, conv -0.012 ms). The positive oracle headroom there arises because the oracle can abstain (`max(0, ·)` truncates harm to zero), not because alternatives are good on average.

**Relative scale (supplementary; SBS reference latency recovered exactly from the action-level file).** The state-weighted mean SBS mean-latency is about 100 ms. Pooled mean headroom / pooled mean SBS latency = **1.99%**. Per-state relative headroom has mean 2.75% and median 0.21%; 29.3% of states exceed 1% and 18.5% exceed 5%. By regime the mean relative headroom is 4.52% for code/KV-16,000 and at most 0.15% for every other regime.

---

## 5. WEIGHTING_SENSITIVITY

**Definitions.** Let `D` be the 720 states, `H(s)` the oracle headroom, `B(s)` the beneficial indicator. For a partition of `D` into units `u` (windows or regimes) with `D_u` the states in unit `u` and `n_u = |D_u|`:

* **State-weighted (canonical primary):** `θ_state = (1/|D|) Σ_s H(s)`. This is a ratio of totals, and each state has weight 1/720.
* **Equal-window:** `θ_win = (1/36) Σ_w [ (1/n_w) Σ_{s∈D_w} H(s) ]`. Each of the 36 windows has weight 1/36, and a window's mean pools all its states across regimes.
* **Equal-regime:** `θ_reg = (1/5) Σ_r [ (1/n_r) Σ_{s∈D_r} H(s) ]`. Each of the 5 regimes has weight 1/5.
* Supplementary: equal-workload (2 units) and equal window-x-regime cell (49 units).

The beneficial fraction `P` is aggregated identically, with pooled `P = Σ B / |D|` and equal-unit `P = mean_u(Σ_{D_u} B / n_u)`.

**On "averaging ratios".** Each equal-unit estimand is by definition a mean of per-unit ratios; that is a different, explicitly stated estimand, not an error. The pooled estimand and all bootstrap / jackknife / leave-one-out recomputations of it are computed as ratios of totals over the retained states (`Σ H / Σ n`), never as a mean of unit means. I implemented all of these and report all of them; I did not select the most favourable one. The state-weighted estimand is the preregistered primary and answers "expected headroom at a randomly chosen disagreement state." The equal-window and equal-regime estimands answer "expected headroom for a typical window / regime" and are more informative about generalisation, because states within a window are sequential snapshots of one trajectory and are not independent.

**Results** (CIs: post-hoc, 200,000 replicates, seed 20260920. State-weighted and equal-window resample the 36 windows. Equal-regime and equal-workload treat regimes/workloads as fixed strata and resample window-regime cells within each stratum, an approximation that treats cells of different strata as independent and is flagged as descriptive; the 1-window conv/KV regime has zero resampling variance).

| estimand | units | mean H (ms) | 95% CI (ms) | P(beneficial), canonical | P(beneficial), eps-guarded |
|---|---:|---|---|---|---|
| state-weighted (canonical primary) | 1 | 1.9958 | [0.184, 3.541] | 0.8194 | 0.8181 |
| equal-window | 36 | 0.3273 | [0.114, 0.644] | 0.6323 | 0.6295 |
| equal-regime | 5 | 0.7591 | [0.145, 1.161] | 0.6665 | 0.6634 |
| equal-workload (supplementary) | 2 | 1.3596 | [0.184, 2.094] | 0.8088 | 0.8056 |
| equal window x regime cell (supplementary) | 49 | 0.2901 | n/a | 0.5874 | 0.5854 |

The mean-headroom CIs are not directly comparable across rows (different estimands), but their signs are: **every estimand has a positive mean and a CI that excludes zero.** The magnitude varies about 7x, from 2.00 ms (state-weighted) to 0.33 ms (equal-window) and 0.29 ms (equal window-x-regime cell). The beneficial fraction falls from 0.819 to 0.632 (equal-window) and 0.666 (equal-regime), because the state-dense code/KV-16,000 regime (P = 0.923) is down-weighted relative to regimes whose benefit rates are 0.40 to 0.52 (the single-window conv/KV regime has P = 1.0).

---

## 6. LEAVE_ONE_OUT_RESULTS

Each row recomputes the pooled state-weighted mean and beneficial fraction on the remaining states (ratio of totals). "Share omitted" is the omitted unit's share of total summed headroom.

**Leave-one-regime-out (5 fits):**

| omitted | n_states_omitted | share_of_total_headroom_omitted | n_states | n_windows | mean_H_ms | delta_mean_H_ms | P_beneficial_canonical | P_beneficial_eps_guarded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Code, active cap 4 | 122 | 0.007 | 598 | 32 | 2.387 | 0.3915 | 0.8796 | 0.8779 |
| Code, active cap 8 | 10 | 0.000 | 710 | 36 | 2.023 | 0.02764 | 0.8254 | 0.8239 |
| Code, KV 16,000 | 431 | 0.968 | 289 | 32 | 0.1592 | -1.837 | 0.6644 | 0.6609 |
| Conv, active cap 4 | 64 | 0.010 | 656 | 20 | 2.169 | 0.1736 | 0.8521 | 0.8521 |
| Conv, KV 16,000 | 93 | 0.016 | 627 | 36 | 2.256 | 0.2602 | 0.7927 | 0.7911 |

* Mean-headroom range: **min 0.159 ms (omit code/KV-16,000), max 2.387 ms (omit code/active-4)**.
* Beneficial-fraction range: min 0.664 (omit code/KV-16,000), max 0.880 (omit code/active-4).

**Leave-one-window-out (36 fits; the 5 lowest and 5 highest estimates shown, full table in `leave_one_window_out.csv`):**

| omitted | n_states_omitted | share_of_total_headroom_omitted | n_states | n_windows | mean_H_ms | delta_mean_H_ms | P_beneficial_canonical | P_beneficial_eps_guarded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| azure_2023_code:w11 | 263 | 0.884 | 457 | 35 | 0.3647 | -1.631 | 0.8206 | 0.8184 |
| azure_2023_conv:w14 | 1 | 0.000 | 719 | 35 | 1.998 | 0.002012 | 0.8192 | 0.8178 |
| azure_2023_conv:w13 | 1 | 0.000 | 719 | 35 | 1.999 | 0.00275 | 0.8192 | 0.8178 |
| azure_2023_code:w17 | 1 | 0.000 | 719 | 35 | 1.999 | 0.002764 | 0.8192 | 0.8178 |
| azure_2023_conv:w26 | 1 | 0.000 | 719 | 35 | 1.999 | 0.002776 | 0.8206 | 0.8192 |
| azure_2023_code:w7 | 17 | 0.000 | 703 | 35 | 2.044 | 0.04795 | 0.8336 | 0.8321 |
| azure_2023_code:w21 | 32 | 0.002 | 688 | 35 | 2.084 | 0.08864 | 0.8198 | 0.8183 |
| azure_2023_code:w13 | 37 | 0.006 | 683 | 35 | 2.091 | 0.09471 | 0.8097 | 0.8082 |
| azure_2023_code:w31 | 80 | 0.010 | 640 | 35 | 2.224 | 0.228 | 0.8016 | 0.8 |
| azure_2023_conv:w18 | 95 | 0.016 | 625 | 35 | 2.263 | 0.2674 | 0.7952 | 0.7936 |

* Mean-headroom range: **min 0.365 ms (omit `azure_2023_code:w11`), max 2.263 ms (omit `azure_2023_conv:w18`)**.
* Beneficial-fraction range: min 0.7952 (omit `azure_2023_conv:w18`), max 0.8336 (omit `azure_2023_code:w7`).
* 35 of the 36 omissions give a mean within 0.27 ms of the full-sample 1.996 ms. **The single exception is `azure_2023_code:w11`, whose omission removes 1.63 ms (82%) of the estimate.** The beneficial fraction is stable to within about 0.025 under any single omission.

**Leave-one-group-out (supplementary, `leave_one_group_out_supplementary.csv`):**

| omitted | n_states_omitted | share_of_total_headroom_omitted | n_states | n_windows | mean_H_ms | delta_mean_H_ms | P_beneficial_canonical | P_beneficial_eps_guarded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Azure code | 563 | 0.975 | 157 | 17 | 0.2313 | -1.764 | 0.7898 | 0.7834 |
| Azure conversation | 157 | 0.025 | 563 | 19 | 2.488 | 0.492 | 0.8277 | 0.8277 |
| active_sequence_capacity | 196 | 0.016 | 524 | 15 | 2.697 | 0.7015 | 0.937 | 0.937 |
| kv_capacity | 524 | 0.984 | 196 | 32 | 0.1202 | -1.876 | 0.5051 | 0.5 |

**Does the result stay positive without the dominant unit?** (Post-hoc descriptive: window-cluster bootstrap CI of the pooled mean on the remainder, 200,000 replicates.)

| omitted | n_states | n_windows | mean_H_ms | boot_ci95_low_ms | boot_ci95_high_ms | ci_excludes_zero |
|---|---:|---:|---:|---:|---:|---:|
| Code, active cap 4 | 598 | 32 | 2.387 | 0.218 | 4.404 | True |
| Code, active cap 8 | 710 | 36 | 2.023 | 0.1857 | 3.618 | True |
| Code, KV 16,000 | 289 | 32 | 0.1592 | 0.05848 | 0.2341 | True |
| Conv, active cap 4 | 656 | 20 | 2.169 | 0.1848 | 3.749 | True |
| Conv, KV 16,000 | 627 | 36 | 2.256 | 0.1681 | 3.721 | True |

| omitted | n_states | n_windows | mean_H_ms | boot_ci95_low_ms | boot_ci95_high_ms | ci_excludes_zero |
|---|---:|---:|---:|---:|---:|---:|
| azure_2023_code:w11 | 457 | 35 | 0.3647 | 0.1578 | 0.6824 | True |
| azure_2023_conv:w14 | 719 | 35 | 1.998 | 0.1821 | 3.547 | True |
| azure_2023_conv:w13 | 719 | 35 | 1.999 | 0.1836 | 3.547 | True |
| azure_2023_code:w17 | 719 | 35 | 1.999 | 0.1851 | 3.544 | True |
| azure_2023_conv:w26 | 719 | 35 | 1.999 | 0.1842 | 3.542 | True |

**All 41 leave-one-out estimates are positive (5 regimes + 36 windows), and the remainder CI excludes zero in every one of the 41 cases.** After removing `azure_2023_code:w11` the pooled mean is 0.365 ms with CI [0.158, 0.682] ms. After removing the whole code/KV-16,000 regime it is 0.159 ms with CI [0.058, 0.234] ms.

**Figure 2** (`figures/fig2_leave_one_out.pdf`): (a, c) leave-one-regime-out mean and beneficial fraction (the hatched bar is the dominant regime; the dashed line is the full-sample value); (b, d) leave-one-window-out mean and beneficial fraction, sorted, with marker shape encoding workload.

---

## 7. CLUSTER_ROBUSTNESS

**Verification of the canonical procedure** (from source, and re-executed bit-exactly).

* **Independent cluster:** the faithful source window `(source_dataset, window_index)`, matching the preregistered cluster unit.
* **Number of clusters:** 36 (19 code, 17 conversation). Windows contain between 1 and 263 states (median 5).
* **Resampling:** with replacement at the cluster level; all states of a sampled cluster are retained together with multiplicity.
* **Seed / replicates / CI:** 20260920 / 2,000 / percentile 95% (numpy linear quantiles).
* **Cluster-key correction:** the original bare-`window_index` key gave 26 clusters and CI [0.1846, 3.4974] ms; the corrected key gives 36 clusters and [0.1909, 3.5462] ms. Both were reproduced. The verdict is identical under both, and the correction changed the CI by less than 0.06 ms.

**Sensitivity checks (post-hoc; not substitutes for the canonical procedure).**

| Procedure | 95% CI for pooled mean (ms) |
|---|---|
| Canonical: percentile, B = 2,000, seed 20260920 | [0.1909, 3.5462] |
| Percentile, B = 1,000,000, seed 20260920 | [0.1843, 3.5428] |
| Percentile, B = 1,000,000, seed 20260921 | [0.1842, 3.5427] |
| Percentile, B = 1,000,000, seed 20260922 | [0.1841, 3.5414] |
| Basic (reverse-percentile), B = 10^6 | [0.449, 3.807] |
| BCa, B = 10^6 (z0 = 0.092, acceleration = 0.149) | [0.239, 4.049] |
| Delete-one-window jackknife (SE = 1.651 ms), t(35) | **[-1.356, 5.348]** |

* **Bootstrap replicate count is not a concern.** Across 3 x 10^6 replicates the fraction of replicates with mean <= 0 is 0. Over 1,000 independent repeats of the canonical 2,000-replicate procedure, the lower endpoint has mean 0.1847 ms and SD 0.0041 ms (range 0.171 to 0.198) and every one is positive. The canonical lower endpoint 0.1909 is about 1.5 SD above the converged value, a difference of about 0.006 ms and immaterial.
* **The canonical CI is validated by the percentile, basic and BCa variants** (all exclude zero, lower endpoints 0.18 to 0.45 ms).
* **The delete-one-window jackknife t-interval includes zero.** All 36 delete-one estimates are positive (0.365 to 2.263 ms), but the jackknife SE (1.65 ms) is inflated by the single influential window. I report this as a warning about the dependence of the *magnitude* on one window, not as a competing primary CI: delete-one jackknife intervals are known to behave poorly with one dominant, skewed cluster, and the bootstrap and BCa intervals do exclude zero. A normal-theory reading that treats the 36 windows as exchangeable would not.
* **The bootstrap distribution is bimodal, and this is the key diagnostic.** `azure_2023_code:w11` is absent from 36.1% of replicates (analytic (35/36)^36 = 36.3%). Replicates **without** w11 have mean 0.362 ms (95% range [0.159, 0.681], max 1.07 ms); replicates **with** it have mean 2.43 ms (range [1.57, 3.64], min 1.09). The pooled percentile CI [0.18, 3.54] therefore spans the valley between two distinct populations. Its lower endpoint essentially reflects the "w11 not drawn" scenario. The positive-sign conclusion still holds because the other 35 windows also have positive mean headroom whose own CI excludes zero (section 6), but the width and center of the CI are driven by whether one window is drawn. See **Figure 4b**.
* **Effective number of independent windows.** By headroom mass, 1/Σp_w² = **1.27**; by state count, 5.68. The nominal "36 clusters" is correct for the resampling unit but overstates how many windows carry the magnitude.

**Figure 4** (`figures/fig4_weighting_and_cluster_robustness.pdf`): (a) weighting-sensitivity forest plot; (b) the 200,000-replicate bootstrap distribution with all interval procedures overlaid.

---

## 8. COMPOSITION_ANALYSIS

**By regime:**

| regime | n_states | share_of_states | n_windows | sum_H_s | share_of_total_headroom | mean_H_ms | contrib. to pooled mean (ms) | P_beneficial_canonical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Code, active cap 8 | 10 | 0.014 | 2 | 0.0003 | 0.000 | 0.03343 | 0.0004643 | 0.4 |
| Code, active cap 4 | 122 | 0.169 | 15 | 0.0094 | 0.007 | 0.07668 | 0.01299 | 0.5246 |
| Code, KV 16,000 | 431 | 0.599 | 14 | 1.3910 | 0.968 | 3.227 | 1.932 | 0.9234 |
| Conv, active cap 4 | 64 | 0.089 | 17 | 0.0139 | 0.010 | 0.2168 | 0.01927 | 0.4844 |
| Conv, KV 16,000 | 93 | 0.129 | 1 | 0.0224 | 0.016 | 0.2413 | 0.03117 | 1 |

**Top windows by total headroom** (all 36 in `composition_window.csv`):

| rank | id | n_states | share_of_states | sum_H_s | share_of_total_headroom | cum_share_of_total_headroom | mean_H_ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | azure_2023_code:w11 | 263 | 0.365 | 1.2703 | 0.884 | 0.884 | 4.83 |
| 2 | azure_2023_code:w37 | 53 | 0.074 | 0.0769 | 0.054 | 0.938 | 1.451 |
| 3 | azure_2023_conv:w18 | 95 | 0.132 | 0.0224 | 0.016 | 0.953 | 0.2363 |
| 4 | azure_2023_code:w35 | 18 | 0.025 | 0.0144 | 0.010 | 0.963 | 0.7993 |
| 5 | azure_2023_code:w31 | 80 | 0.111 | 0.0137 | 0.010 | 0.973 | 0.1716 |
| 6 | azure_2023_code:w1 | 11 | 0.015 | 0.0117 | 0.008 | 0.981 | 1.06 |
| 7 | azure_2023_code:w13 | 37 | 0.051 | 0.0092 | 0.006 | 0.987 | 0.2474 |
| 8 | azure_2023_conv:w3 | 7 | 0.010 | 0.0087 | 0.006 | 0.993 | 1.237 |
| 9 | azure_2023_code:w21 | 32 | 0.044 | 0.0029 | 0.002 | 0.995 | 0.09005 |
| 10 | azure_2023_conv:w11 | 11 | 0.015 | 0.0024 | 0.002 | 0.997 | 0.2213 |

**Findings (stated directly).**

* **The aggregate is dominated by one regime.** Code/KV-16,000 has 59.9% of states but **96.8% of total summed headroom**, and contributes **1.932 ms of the 1.996 ms pooled mean**. The other four regimes together contribute 0.064 ms. The effective number of regimes by headroom mass is 1.07.
* **Without that regime, the pooled mean over the other 289 states is 0.159 ms** (CI [0.058, 0.234] ms, still positive).
* **It is dominated even more strongly by one window.** `azure_2023_code:w11` has 263 states (36.5% of the 720) and **88.4% of total headroom**. The top 3 windows carry 95.3%, and one window is enough to reach both 50% and 80% of the total.
* **Anatomy of w11:** 205 code/KV-16,000 states (mean 6.16 ms), 51 code/active-4 states (mean 0.14 ms) and 7 code/active-8 states, spanning decision steps 545 to 3,230 of a single 200-request window. These are sequential decision points from at most three simulated trajectories (one per pressure condition) and are strongly dependent, which is why window-level clustering is essential and why the effective sample for the *magnitude* is small. Within the dominant regime, removing w11 leaves 226 states in 13 windows with mean 0.567 ms.
* **The high-headroom tail is nearly one window:** of the 129 states above 5 ms, 128 are in w11; of the 162 above 2 ms, 140 are in w11 (the rest in code:w37 (19), code:w1 (2) and conv:w3 (1)).

**Figure 3** (`figures/fig3_composition.pdf`): (a) share of states versus share of total headroom by regime; (b) window-level cumulative concentration (Lorenz-style) curve against equal contribution.

---

## 9. SCIENTIFIC_INTERPRETATION

**Does the positive latency-headroom conclusion survive reasonable weighting changes?**
The *sign* survives every weighting: state-weighted 1.996 ms, equal-workload 1.360 ms, equal-regime 0.759 ms, equal-window 0.327 ms and equal cell 0.290 ms, with CIs excluding zero where computed. The *magnitude* does not survive: it varies about 7x across defensible weightings, and the 2 ms headline is the largest of them.

**Does it depend strongly on one window?**
Its sign does not (all 36 leave-one-window-out estimates are positive and each remainder CI excludes zero). Its magnitude depends on one window very strongly. `azure_2023_code:w11` supplies 88.4% of total headroom; without it the mean is 0.365 ms [0.158, 0.682]. The delete-one jackknife t-interval, which treats windows as exchangeable, includes zero, and the bootstrap distribution is bimodal on the presence of this window.

**Does it depend strongly on one regime?**
Sign: no (remainder 0.159 ms, CI [0.058, 0.234]). Magnitude: yes. Code/KV-16,000 supplies 96.8% of total headroom and 1.932 of the 1.996 ms. This is the KV-capacity-16,000 pressure setting on the code trace; every other regime is at least 60x smaller in summed headroom.

**How much of the nominal 81.9% beneficial fraction remains at practically meaningful thresholds?**
Using tolerance-guarded numerators: 589/720 = 81.8% above 0 ms; 432/720 = 60.0% above 0.1 ms; 322/720 = 44.7% above 0.25 ms; 212/720 = 29.4% above 0.5 ms; 188/720 = 26.1% above 1 ms; 162/720 = 22.5% above 2 ms; 129/720 = 17.9% above 5 ms. Which threshold is "practically meaningful" depends on selector overhead, which this study does not measure; the results show that 40.0% of states offer 0.1 ms or less and 70.6% offer 0.5 ms or less. The 81.9% figure counts genuinely tiny effects and, in the canonical strict form, one floating-point-residue state.

**Is about 2 ms a representative effect or influenced by a high-headroom subgroup?**
It is *not* representative. The median state has 0.197 ms of headroom; 77.4% of states (557/720) are at or below the mean; and the mean is determined by a subgroup (code/KV-16,000, and within it one window). It is a valid state-weighted mean of a heavy-tailed distribution, but should not be read as a typical or per-window expectation. Relative to a roughly 100 ms mean SBS latency, the pooled headroom is about 2%, and at most 0.15% in every other regime.

**Is there anything requiring a change to the manuscript's central claim?**
My judgement, not a data-forced conclusion: **the central claim is supported as stated**, namely that fresh, preregistered replay shows a positive, statistically supported, operationally modest one-step oracle headroom under resource pressure, with all headline numbers reproduced. The manuscript already calls the result "operationally modest," conditional on disagreement, and not a deployment claim. What these analyses show is that the *supporting language* under-discloses concentration: "36 distinct source-window clusters" and a 2 ms pooled mean read as broader, more uniform evidence than the data provide (about 1.3 effective windows by headroom mass, one window carrying 88%, and an 81.9% beneficial fraction in which 40% of states offer at most 0.1 ms). I would add disclosure and sensitivity results (section 10), not change the claim. I would avoid any wording that implies a typical 2 ms improvement, and, if the reviewer-facing framing is "robust," I would specify that robustness is for the sign, not the magnitude.

**Caveats on these robustness analyses.**

* All are post-hoc, computed on the frozen 720 states after outcomes were known; none is a confirmatory test.
* Regimes, and hence the equal-regime weighting, rest on only 5 cells (one with a single window); equal-regime CIs assume independence across regime cells.
* Leave-one-out and jackknife are diagnostics of influence, not new inferential procedures.
* Only two workloads (Azure code and conversation) contribute; BurstGPT contributed no regime.

---

## 10. MANUSCRIPT_RECOMMENDATIONS

**Not edited in this task.** For later integration, in priority order:

1. **Sentence-level disclosure of concentration** (Results, Sec. `fresh`): one window (`azure_2023_code` window 11) accounts for 88.4% of total headroom (263/720 states); the code/KV-16,000 regime accounts for 96.8%; excluding that window the pooled mean is 0.365 ms (CI [0.158, 0.682]), excluding the regime it is 0.159 ms (CI [0.058, 0.234]). Source: `composition_summary.json`, `leave_one_out_summary.json`, `leave_one_regime_out_remainder_ci.csv`, `leave_one_window_out_remainder_ci.csv`.
2. **New table: practical-effect thresholds** (global row plus the 5 regimes; numerators and denominators): `practical_thresholds.csv` and the first two tables in section 3.
3. **Distribution table or panel** (median 0.197, p75 1.19, p90 8.4, p95 11.3, max 12.1 ms; zero mass 18.1%) with **Figure 1** (`fig1_headroom_distribution.pdf`), to replace or complement the regime bar chart's implication of typicality.
4. **Weighting-sensitivity table** (state-weighted vs equal-window vs equal-regime, with the estimand definitions of section 5): `weighting_sensitivity.csv` and `weighting_sensitivity_ci.csv`.
5. **Leave-one-out figure** (`fig2_leave_one_out.pdf`) with a one-sentence statement that all 41 leave-one-out estimates remain positive and the remainder CIs exclude zero.
6. **Cluster-robustness sentence and footnote:** percentile CI with B = 10^6 is [0.184, 3.543] ms (BCa [0.239, 4.049]) so replicate count is immaterial; note the bimodal bootstrap distribution (Figure 4b) and the effective window count (1.27 by headroom mass).
7. **Numerical-tolerance footnote:** the canonical strict `> 0` rule counts one floating-point-residue state (5.55e-17 s); with a 1e-12 s guard the beneficial count is 589/720 (81.8%). No conclusion changes.
8. **Reword** "36 distinct source-window clusters" so it describes the resampling unit rather than the amount of independent evidence for magnitude; add the qualifier that robustness is for sign, not magnitude.
9. **Repository-hygiene note (not manuscript):** document, without altering the frozen artifact, that state-level `mean_ref_latency` / `p95_ref_latency` in `FRESH_LATENCY_STATE_LEVEL_V1.csv` are the first counterfactual branch's values, not the SBS reference (section 1); the recovered reference is available from `ref_mean_latency` in the analysis module.
10. All of the above must be labelled **post-hoc sensitivity analyses**, in captions and text.

---

## Appendix: file inventory

`experiments/fresh_production_latency_headroom_confirmatory_v1_robustness/`:
`reproduction_check.json`, `numerical_tolerance_audit.json`, `practical_thresholds.csv`, `practical_thresholds_global_ci.csv`, `distribution_summary.csv`, `signed_action_advantage_summary.csv`, `relative_headroom_summary.json`, `weighting_sensitivity.csv`, `weighting_sensitivity_ci.csv`, `leave_one_window_out.csv`, `leave_one_regime_out.csv`, `leave_one_group_out_supplementary.csv`, `leave_one_regime_out_remainder_ci.csv`, `leave_one_window_out_remainder_ci.csv`, `leave_one_out_summary.json`, `cluster_robustness.json`, `composition_{regime,workload,axis,window}.csv`, `composition_summary.json`, `report_tables.md`, `provenance.json`, `figures/fig1..fig4 (.pdf + .png)`.
