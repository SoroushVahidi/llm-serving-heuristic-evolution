# ESTF↔WFS composition falsification pilot v1 — provenance

**Status:** execution COMPLETE; scientific analysis COMPLETE  
**Audit:** [`docs/audits/estf_wfs_composition_falsification_v1_20260816.md`](../../docs/audits/estf_wfs_composition_falsification_v1_20260816.md)  
**Design:** [`docs/design/ESTF_WFS_COMPOSITION_FALSIFICATION_V1.md`](../../docs/design/ESTF_WFS_COMPOSITION_FALSIFICATION_V1.md)  
**Verdict:** `SELECTION_SUFFICIENT_FOR_THIS_PAIR`  
**tmux:** `estf-wfs-comp-pilot`  
**Elapsed:** ≈559 s  
**Parents:** Family A v2 Job 1182377

## Integrity

- 252/252 child evaluations successful; 0 failures; 0 duplicates  
- Splits: train 30 / val 20 / test 10 / OOD 12  
- BurstGPT-required regeneration via `.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv`  
- Parent ANWG rows copied from Family A CSV (84 rows; bit-exact)

## Key negative result

Contextual rank composition does **not** beat contextual top-1 on TEST
(ΔANWG ≈ −0.003; envelope gain = 0). Symbolic distillation / MAP-Elites /
LLM synthesis are **not** justified from this pair alone.

## Preserved files

| File | Role |
|---|---|
| `composition_results.csv` | All method scores |
| `summary.json` | Auto verdict + split summaries |
| `splits.json` | Exact scenario IDs |
| `model_selection.json` | Val model choice |
| `analysis/analysis_summary.json` | Independent re-analysis |
| `run.log` / `console.log` | Execution log (gitignored `*.log`) |
| `README.md` | This note |
