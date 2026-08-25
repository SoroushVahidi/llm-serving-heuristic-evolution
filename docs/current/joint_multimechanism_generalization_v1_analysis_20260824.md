# joint_multimechanism_generalization_v1 Analysis (2026-08-24)

CPU-only six-policy workload-breadth experiment. No selector, search, DEV, TEST, FINAL, GPU, vLLM, or Wulver work was used.

## Verdict
- Scientific verdict: `JOINT_GENERALIZATION_STRONG`
- Best fixed policy: `kv_constrained_online`
- Best fixed mean ANWG: 0.314072
- Six-policy oracle mean ANWG: 0.333106
- Oracle headroom: 0.019034
- Unique-winner fraction at epsilon=0.01: 0.596
- Nontrivial policy-spread fraction: 0.917
- Multi-mechanism gain share (>=2 elevated pressures): 0.931
- Top-10% scenario gain share: 0.405

## Workload Coverage
- Scenarios: 240
- >=2 elevated mechanisms: 223 (0.929)
- >=3 elevated mechanisms: 175
- Elevated-mechanism counts: `{1: 17, 2: 48, 3: 90, 4: 64, 5: 20, 6: 1}`

## Winner Distribution
- Winner counts: `{'full_prefill': 46, 'chunked_prefill_small': 5, 'estimated_service_time_first': 45, 'weighted_fair_share': 35, 'least_laxity_first': 59, 'kv_constrained_online': 50}`
- Unique winner counts: `{'full_prefill': 3, 'chunked_prefill_small': 2, 'estimated_service_time_first': 28, 'weighted_fair_share': 25, 'least_laxity_first': 45, 'kv_constrained_online': 40}`

## Decision Disagreement
- Metric: closed_loop_action_set_disagreement_proxy
- Caveat: Policy trajectories can diverge; this is diagnostic, not a same-state selector disagreement estimate.
- Overall mean pairwise disagreement: 0.785294
- Correlation with policy utility range: 0.276575

## Local Winner Structure
- kNN winner consistency mean: 0.212917
- Local winner entropy mean bits: 1.942099

## Robustness
- Bootstrap headroom mean: 0.019088
- Bootstrap 95% CI: [0.015987777521472592, 0.022432582285730952]
- Best-fixed bootstrap counts: `{'full_prefill': 0, 'chunked_prefill_small': 0, 'estimated_service_time_first': 0, 'weighted_fair_share': 1, 'least_laxity_first': 0, 'kv_constrained_online': 999}`

## A/B/C Comparison
- Source: `experiments/unified_utility_matrix_v2/unified_utility_matrix_wide_v2.csv`
- FAMILY_A_FAIRNESS_STARVATION_V2: n=72, best_fixed=weighted_fair_share, headroom=0.021742, winners={'full_prefill': 0, 'chunked_prefill_small': 0, 'estimated_service_time_first': 37, 'weighted_fair_share': 23, 'least_laxity_first': 0, 'kv_constrained_online': 12}
- FAMILY_B_PREFILL_DECODE_V2: n=32, best_fixed=full_prefill, headroom=0.049433, winners={'full_prefill': 17, 'chunked_prefill_small': 15, 'estimated_service_time_first': 0, 'weighted_fair_share': 0, 'least_laxity_first': 0, 'kv_constrained_online': 0}
- FAMILY_C_KV_PRESSURE_V2: n=72, best_fixed=kv_constrained_online, headroom=0.020261, winners={'full_prefill': 32, 'chunked_prefill_small': 0, 'estimated_service_time_first': 17, 'weighted_fair_share': 5, 'least_laxity_first': 2, 'kv_constrained_online': 16}

## Figures
- `experiments/joint_multimechanism_generalization_v1/figures/winner_distribution.png`
- `experiments/joint_multimechanism_generalization_v1/figures/oracle_gain_histogram.png`
- `experiments/joint_multimechanism_generalization_v1/figures/mechanism_pressure_correlation.png`
- `experiments/joint_multimechanism_generalization_v1/figures/winner_map_prefill_kv.png`

## Claim Safety
- Safe: complementarity persists in this broader jointly varying synthetic workload distribution, if stated with the reported bounds.
- Safe: oracle headroom remains positive in scenarios combining multiple stress mechanisms.
- Unsafe: this proves complementarity in arbitrary production traffic.
- Unsafe: this proves an adaptive scheduler can exploit the oracle headroom.

## Runtime
- End-to-end experiment wall time: 89.90s
