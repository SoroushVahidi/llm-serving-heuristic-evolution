# Phase 2C.2 Causal Selector Retraining

## Train / test design

| Split | Source | Feature mode | Used for |
|-------|--------|--------------|----------|
| Training | Phase 2B.13 synthetic workloads (`dev_*`, `div_*`) rebuilt with causal features | `causal` | Fit deployable selectors (dev + diversity seeds 6–10) |
| Validation | Diversity seed 11 (built but not used for final fit in default workflow) | `causal` | Optional diagnostics |
| Evaluation | Phase 2C.1 real-trace workloads (`burstgpt_*`, `azure_2023_*`) | `causal` | Held-out real-trace test only |

Real-trace workloads are **never** in the training split.

## Difference from Phase 2C.1 causal re-eval

- **Phase 2C.1 (causal rerun):** Frozen selectors trained on legacy lookahead features from Phase 2B.13 CSV; only evaluation used causal features.
- **Phase 2C.2:** Rebuilds synthetic training rows with causal features, retrains selectors, then evaluates on real traces.

## Metrics and selectors

- Primary rank metric: `mean_arrival_normalized_wg` (ANWG)
- Completed-only WG: diagnostic only
- `safe_fallback_wsp_margin*`: oracle-assisted, excluded from `deployable_selector_summary.csv`
- External-style baselines: orca, vLLM, Sarathi, SplitFuse, multi-bin, EST, SCORPIO-style guard

## Runner

```bash
python scripts/run_phase2c2_causal_selector_retraining.py --dry-run
python scripts/run_phase2c2_causal_selector_retraining.py --smoke
python scripts/run_phase2c2_causal_selector_retraining.py --allow-full-run
```
