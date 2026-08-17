# Family B v2 discriminative smoke — provenance

**Status:** COMPLETE; smoke gate **`SMOKE_GO`**  
**Design:** [`docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md`](../../docs/design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V2.md) §7  
**Run dir:** `experiments/policy_separation_prefill_decode_smoke_v2_20260817T024136Z/`  
**Integrity:** 8 scenarios × 2 policies = **16/16 success**, 0 failures  
**BurstGPT:** hog/late prompts `burstgpt_staged`; outputs labeled `synthetic_short_output_for_ttft_isolation`  
**Primary:** canonical `arrival_normalized_weighted_goodput`

Preregistered gate:

| Check | Result |
|---|---|
| S1 100% success | yes |
| S2 ≥1 full≻small by >0.01 | **4** cells (all `hog_ttft`) |
| S3 ≥1 small≻full by >0.01 | **4** cells (all `late_ttft`) |
| S4 two anchors only | yes |
| S5 BurstGPT + labeled short-output intervention | yes |

Mechanism on this smoke: full has lower hog-class TTFT (~0.054 s vs ~0.10 s); small has lower late-class TTFT. `decode_stalled_steps` remains 0 (expected FCFS equilibrium).

This smoke **justifies launching** the 32×2 full pilot. It is not the H1–H10 / composition-readiness result.

Command:

```text
PYTHONPATH=src PYTHONUNBUFFERED=1 \
LLM_SERVEOPT_BURSTGPT_CSV=.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv \
python scripts/run_policy_separation_prefill_decode_pilot_v2.py \
  --config configs/policy_separation_prefill_decode_smoke_v2.yaml \
  --run-dir experiments/policy_separation_prefill_decode_smoke_v2_20260817T024136Z \
  --workers 8 \
  --datasets-root .local_data \
  --require-burstgpt
```
