# Family B v2 full pilot — provenance

**Status:** execution COMPLETE; scientific analysis COMPLETE  
**Audit:** [`docs/audits/policy_separation_prefill_decode_pilot_v2_20260817.md`](../../docs/audits/policy_separation_prefill_decode_pilot_v2_20260817.md)  
**Design:** [`docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md`](../../docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md)  
**Verdict:** `FAMILY_B_COMPOSITION_READY`  
**Run dir:** `experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z/`  
**Smoke:** `experiments/policy_separation_prefill_decode_smoke_v2_20260817T024136Z/` (`SMOKE_GO`)  
**Slurm:** none (local on `al-khwarizmi`)  
**Git HEAD at launch:** `ecc0422286886c83d263e87655ed1123e62d2565`  
**Elapsed:** 0.17 s  
**Integrity:** **64/64 success, 0 failures**; 32 unique scenarios × 2 policies; canonical ANWG; BurstGPT staged prompts; short-output intervention labeled.

Command:

```text
PYTHONPATH=src PYTHONUNBUFFERED=1 \
LLM_SERVEOPT_BURSTGPT_CSV=.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv \
python scripts/run_policy_separation_prefill_decode_pilot_v2.py \
  --config configs/policy_separation_prefill_decode_pilot_v2.yaml \
  --run-dir experiments/policy_separation_prefill_decode_pilot_v2_20260817T024204Z \
  --workers 8 \
  --datasets-root .local_data \
  --require-burstgpt
```

Analyzer: `scripts/analyze_policy_separation_prefill_decode_pilot_v2.py` → `analysis/`.  
Raw `per_policy_results.csv` is frozen evidence after this analysis commit.
