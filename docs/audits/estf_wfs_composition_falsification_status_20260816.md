# ESTF↔WFS Composition Falsification — Status

**Date:** 2026-08-16  
**Design:** [`../design/ESTF_WFS_COMPOSITION_FALSIFICATION_V1.md`](../design/ESTF_WFS_COMPOSITION_FALSIFICATION_V1.md)  
**Pilot run dir:** `experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/`  
**tmux session:** `estf-wfs-comp-pilot`  
**Command:**

```bash
PYTHONPATH=src LLM_SERVEOPT_BURSTGPT_CSV=.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv \
  python scripts/run_estf_wfs_composition_pilot.py \
    --run-dir experiments/estf_wfs_composition_falsification_v1_20260816T222108Z \
    --datasets-root .local_data \
    --require-burstgpt
```

## Three-minute health check

- Splits: train=30, val=20, test=10, ood=12  
- Models fitted (selector logreg val_acc=0.30; alpha logreg proxy_acc=0.40)  
- Child evaluations progressing: **80/252** at ~3 minutes with no crash/traceback  
- BurstGPT path in use via local staged copy of `BurstGPT_without_fails_1.csv`  
- Monitoring stopped per long-running job rule; **scientific verdict pending completion**

## Next action after completion

1. Confirm `summary.json` exists and `composition_results.csv` has 252 success rows.  
2. Read `decisive_test.verdict` ∈ {`COMPOSITION_GO`, `SELECTION_SUFFICIENT_FOR_THIS_PAIR`, `INCONCLUSIVE`}.  
3. Write full audit; update handoff; do not escalate model complexity if selection wins.
