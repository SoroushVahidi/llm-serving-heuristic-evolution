# Repaired Pilot Summary (Git-safe)

Compact derivative of job `1143392` for durable pause preservation.
Absolute Wolverine paths below are **historical provenance** and may disappear.

## Decision
`LOAD_DISCRIMINATION_PILOT = PARTIALLY_READY`

**Do not** launch a full 27-policy fingerprint sweep based on this pilot alone.

## Provenance
- Slurm job: `1143392` (COMPLETED 0:0, ~13m, MaxRSS ~1.9G)
- Git SHA at run: `4dd97eadd16aa65512db61af07f7750596c08d14`
- Selection seed: `20260725`
- Windows: 250 (50 × 5 datasets, stratified quotas)
- Policies (8): fifo, edf, estimated_service_time_first, weighted_shortest_processing, scorpio_style_slo_guard, vllm_style_token_budget, aging_priority, adaptive_chunked_prefill
- Pilot root at pause: `/mmfs1/project/ikoutis/sv96/llmserveopt-data/real_window_construction_20260725T035054Z/pilot_repaired_20260725T124957Z`
- Source window root: `/mmfs1/project/ikoutis/sv96/llmserveopt-data/real_window_construction_20260725T035054Z`

## Key overall metrics
- saturated_rate: 0.0720
- exact_tie_rate: 0.6040
- near_tie_rate: 0.8040
- mean_margin: 0.012523
- winner_classes: 7
- best_fixed: vllm_style_token_budget (0.630945)
- oracle_gain: 0.002948
- Mooncake included: True (50 windows; internal OOD only)

## Readiness gates
- mandatory failed: []
- signal failed: ['near_tie_below_0_80', 'pct_margin_gt_0_02_above_0_20']
- mandatory_pass_count / signal_pass_count: 7 / 5

## Evidence roles (do not collapse)
- Primary natural/busy evidence remains weak on discrimination (high near-ties).
- Trace-derived scaled windows show stronger margins; treat as stress evidence, not natural proof.
- Synthetic is supporting only.

## Diagnostic limitations (non-negotiable)
- “Behavioral disagreement” / tie-cause labels use **outcome signatures**, not true scheduler action traces.
- Labels that mention “actions” are heuristic interpretations of outcome fields.
- True decision/action tracing remains desirable future work.

Machine-readable twin: `repaired_pilot_summary.json`.
Full external report (not in Git): `/mmfs1/project/ikoutis/sv96/llmserveopt-data/real_window_construction_20260725T035054Z/pilot_repaired_20260725T124957Z/reports/REPAIRED_PILOT_REPORT.md`.
