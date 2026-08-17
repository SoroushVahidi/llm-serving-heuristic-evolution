# Family B v1 full pilot — provenance

**Status:** execution COMPLETE; scientific analysis COMPLETE  
**Audit:** [`docs/audits/policy_separation_prefill_decode_pilot_v1_20260817.md`](../../docs/audits/policy_separation_prefill_decode_pilot_v1_20260817.md)  
**Design:** [`docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md`](../../docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md)  
**Verdict:** `USEFUL_BUT_NEEDS_REFINEMENT`; composition `PREFILL_COMPOSITION_NOT_YET_JUSTIFIED`  
**Run dir:** `experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z/`  
**tmux:** `policy-sep-prefill-decode-v1` (exited after completion)  
**Slurm:** none (local on `al-khwarizmi`)  
**Git HEAD at launch:** `ff78897a08a2dd4d04dd0317f79df9a0ba1485ba`  
**Elapsed:** 18.56 s  
**Integrity:** **720/720 success, 0 failures**; 144 unique scenarios × 5 policies; canonical ANWG finite in `[0.211, 1.000]`; BurstGPT `burstgpt_staged`.

Command:

```text
PYTHONPATH=src PYTHONUNBUFFERED=1 \
LLM_SERVEOPT_BURSTGPT_CSV=.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv \
python scripts/run_policy_separation_prefill_decode_pilot_v1.py \
  --config configs/policy_separation_prefill_decode_pilot_v1.yaml \
  --run-dir experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z \
  --workers 8 \
  --datasets-root .local_data \
  --require-burstgpt
```

Logs: `stdout.log` / `run.log` in the run dir (gitignored `*.log`).  
Analyzer: `scripts/analyze_policy_separation_prefill_decode_pilot_v1.py` → `analysis/`.  
Raw `per_policy_results.csv` is frozen evidence and was not rewritten.

Coarse integrity peek (reproduced in the audit): 58/144 cells have structural ANWG spread > 0.01. That spread is the full↔small pairwise contrast, not four distinct unique-winner niches.
