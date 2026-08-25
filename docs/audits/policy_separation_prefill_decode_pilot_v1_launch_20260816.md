# Family B v1 Prefill–Decode Pilot — Launch Status

**Date:** 2026-08-16  
**Status:** `EXECUTED` (integrity-clean). Scientific H1–H10 audit is **not** in this note.  
**Design:** [`../design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md`](../design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md)  
**Run:** [`../../experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z/`](../../experiments/policy_separation_prefill_decode_pilot_v1_20260817T020803Z/)

| Field | Value |
|---|---|
| tmux | `policy-sep-prefill-decode-v1` (completed; session exited) |
| Slurm | none (local `al-khwarizmi`) |
| Git HEAD | `ff78897a08a2dd4d04dd0317f79df9a0ba1485ba` |
| Workers | 8 |
| Tasks | 144 × 5 = 720 |
| Success / fail | **720 / 0** |
| Elapsed | 18.56 s |
| BurstGPT | `.local_data/burstgpt_v2/raw/BurstGPT_without_fails_1.csv` (`burstgpt_staged`) |
| stdout/stderr | `experiments/…020803Z/stdout.log` (empty stderr) |
| Primary metric | canonical `arrival_normalized_weighted_goodput` |

Health check (~20 s into the 3-minute window): job had already completed with `EXIT=0`; no OOM, no path/config failure, ANWG finite.

**Next scientific action:** analyze the 720-row CSV against preregistered H1–H10 (CONFIRM / CONTRADICT / AMBIGUOUS / DESIGN_CONFOUND). Do not start MAP-Elites, distillation, or LLM synthesis.
