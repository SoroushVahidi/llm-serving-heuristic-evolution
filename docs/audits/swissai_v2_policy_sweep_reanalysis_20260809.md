# SwissAI V2 Policy Sweep Reanalysis

## Provenance

- Original sweep: `/mmfs1/project/ikoutis/sv96/llmserveopt-data/swissai_v2_policy_sweep_20260722T184451Z/`
- Original code SHA: `e8bd759b6cdaa8a05096b0ceeb1c7684cfa07302`
- Repaired external report: `/mmfs1/project/ikoutis/sv96/llmserveopt-data/swissai_v2_policy_sweep_reanalysis_20260809_final/`
- Repair method: report-only reconstruction from existing CSV artifacts

## Integrity and repair

The original matrix contains 512 windows x 27 policies = 13,824 valid policy
cells. There are no missing or duplicate `(window_id, policy_name)` keys, and
all 512 causal-feature rows join one-to-one to the window summary.

The failed reporting stage expected `kv_proxy_p95`, `high_reuse_fraction`, and
`low_reuse_fraction` in `window_summary.csv`. The canonical fields were present
in `causal_features.csv` as `feat_swiss_kv_proxy_p95`,
`feat_swiss_high_reuse_fraction`, and `feat_swiss_low_reuse_fraction`. The
repair joined these fields by `window_id`; no policy evaluation was rerun and
the original sweep directory was not modified.

## Results

- Canonical mean oracle/best-fixed ANWG: `0.9925129331848189`
- Oracle gap over the best fixed policy set: `0` within floating-point tolerance
- Strict V2 marginal oracle gain: `0`
- Unique V2 wins: `0 / 512`
- All 512 windows are near-tied at epsilon `0.001`, `0.005`, and `0.01`
- High-KV/high-reuse stratum: 28 windows, zero marginal gain

The earlier `0.991726` value was stale prose/intermediate documentation. The
later simulator-audit CSV already records the canonical matrix aggregate
`0.9925129331848189`.

## Interpretation

SwissAI expands KV/cache/reuse and context feature coverage, but policy rewards
remain saturated under the current simulator and objective. The result does
not establish FIFO/EDF intrinsic optimality, causal real-system KV-reuse
effects, selector transfer, or that module composition is useless.

Bucket-reuse output lengths and SLOs are partly reconstructed or synthetic.
This is bounded simulator sensitivity evidence and should guide simulator
discrimination and contextual-composition work rather than end it.

