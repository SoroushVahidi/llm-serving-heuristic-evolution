# Family B v1 smoke / launch-gate calibration

**Run:** `experiments/policy_separation_prefill_decode_smoke_v1_20260817T020443Z/`  
**Config:** `configs/policy_separation_prefill_decode_smoke_v1.yaml`  
**BurstGPT:** `.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv` (`burstgpt_staged`)  
**Result:** 40/40 success in 2.0s. **GO** for full pilot.

Discriminative cells (ε=0.01 ANWG):

- `long × high × tbt_tight`: chunked_small / decode_priority / adaptive **0.918** vs full **0.864** (spread 0.055)
- `medium × low × ttft_tight`: full / chunked_large **0.381** vs small/decode-priority **0.333** (spread 0.048)
- `medium × high × ttft_tight`: full / chunked_large **0.764** vs others **0.745** (spread 0.018)

Easy cells (ANWG=1 or all-tied) are expected under loose deadlines / short prefills.

Verdict: no universal dominant policy; both a full/large-prefill TTFT-tight regime and a chunked/decode-priority high-overlap regime exist.
