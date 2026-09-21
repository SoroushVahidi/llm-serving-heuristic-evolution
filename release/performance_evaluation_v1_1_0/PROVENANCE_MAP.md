# Provenance map

| Paper element | Produced by | Frozen input(s) | Verified by |
|---|---|---|---|
| Native action-null result (Section 4, Figure 2) | `scripts/industry_realism_action_opportunity_phase_a_v1.py` | `experiments/industry_realism_action_opportunity_phase_a_v1/` | claim manifest `native.*`, `figure2.*` |
| Pressure-induced disagreement (Section 5, Figures 2-3) | `scripts/industry_realism_action_opportunity_phase_b_v2.py` | `experiments/industry_realism_action_opportunity_phase_b_v2/` | claim manifest `pressure.*`, `figure3.*` |
| Fresh causal headroom, primary endpoint (Section 6) | `scripts/fresh_latency_causal_confirmatory_v1.py` | `experiments/fresh_production_latency_headroom_confirmatory_v1/` | claim manifest `fresh.*`; `FRESH_LATENCY_ARTIFACT_HASHES_V1.json` |
| Robustness and concentration (Section 7, Tables 3-5, Figure 6) | `scripts/fresh_causal_robustness_v1.py` | `..._v1_robustness/` | claim manifest `robust.*`; `paper/performance_evaluation/scripts/robustness_numbers.py` |
| Corrected derivative | `scripts/fresh_causal_correct_state_level_v1.py` | continuation shards under `provenance_archive/` | `verify_release.py` (byte replay) |
| Current manuscript figures | `paper/performance_evaluation/scripts/plot_*.py` (shared style `figstyle.py`) | frozen CSVs above | `verify_release.py`, `tests/test_manuscript_figures.py` |
| vLLM correspondence probe (Section 9) | (probe summaries only) | `experiments/real_vllm_*` | claim manifest `vllm.*` |
| All 62 quoted numbers | `paper/performance_evaluation/scripts/build_claim_manifest.py` | all of the above | `build_claim_manifest.py --check` |

Built from git commit `e6fe60ac759b197cdbec0b92e4b3f9143a5ee0df` (branch `release/peva-v1-20260920`). Original execution commit of the fresh causal run: `38e02bcdaeb8f9cec2289b05dea9a1406318d14f`
(executor script sha256 `e4ef52c38ce0412674319cdf2997f1934ab5385c031b227010eb6a913f24815b`).
