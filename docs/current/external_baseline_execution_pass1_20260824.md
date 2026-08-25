# External-Baseline Execution Pass 1 — 2026-08-24

**Mode:** Design freeze + safe launch of authorized cells only.  
**No commit/push. Dirty research worktree preserved.**  
**Verdict:** `EXTERNAL_BASELINE_EXECUTION_PASS1_LAUNCHED_SUCCESSFULLY`

---

## A. Preflight

| Field | Value |
|---|---|
| Host | `al-khwarizmi` |
| Repo | `/home/soroush/llm-serving-heuristic-evolution` |
| Branch | `contextual-compositional-heuristics-20260731` |
| HEAD | `2987b7181efa2bc550d8a894c537eca8f6393eb6` |
| Upstream | ahead 2 |
| Dirty | Yes — preserved |
| Wulver | Connected via `ssh -S ~/.ssh/wulver-control sv96@wulver.njit.edu` → `login02`; `squeue` empty |
| Decision | Long CPU cells run in **local tmux** (fast enough; no need to block on Wulver queue for these) |

---

## B. External-source fidelity

See `docs/current/external_baseline_fidelity_ledger_20260824.md`.

| Method | Finding |
|---|---|
| VTC | Official `Ying1123/VTC-artifact` @ `192c2e2` recloned to `~/.cache/external_baselines/VTC` |
| LTR | Official `hao-ai-lab/vllm-ltr` already integrated; **Class D** on joint/public (no prompt text) |
| SOLA | MLSys 2025 paper found; **no public code**; spec written; not implemented |
| vLLM proxy | Use `vllm_faithful` as `vllm_style_continuous_batching` (not native GPU; not SJF `vllm_style_token_budget`) |

---

## C. Stress protocol

- Docs: `docs/current/external_baseline_stress_protocol_20260824.md`
- JSON: `experiments/external_baseline_comparison_v1/stress_protocol.json`
- Calibration (FIFO-only, 12 windows × M∈{1,2,4,8,16,32}) completed.
- **Selected primary M = 32** via fallback (`fallback_max_queue_positive_among_noncatastrophic`): primary gates requiring `queue+ ≥ 0.10` never met (max queue+ ≈ 0.072 at M=32), but `max_active≈52` and completion≈0.93 are non-catastrophic.
- **Scientific note:** with frozen public capacity 512, even M=32 remains mild vs joint-scale contention. A future authorized pass may add a **predeclared capacity lever** (still policy-blind); do not retune M using ANWG winners.

---

## D. vLLM-style proxy

- Public name: `vllm_style_continuous_batching`
- Implementation: `VLLMFaithfulPolicy` (`vllm_faithful`)
- Joint SM overrides: `allow_chunked_prefill=False`, `decode_first=True`
- Manuscript wording must say **simulator proxy**, not native vLLM.

---

## E. VTC reuse

- Adapter reused; joint requires `batch_token_budget_override = max(step_token_budget, max_prompt)` due to documented units mismatch (count vs tokens).
- Label: `official_vtc_joint_token_budget_remap` (OFFICIAL_CODE_ADAPTED).

---

## F. LTR applicability

**D — not valid** on joint-240 / current public parquet (no prompt text). Not launched.

---

## G. SOLA

Spec only: `docs/current/sola_faithful_reimplementation_spec_20260824.md`. Not launched.

---

## H. Jobs launched

### 1. `extbase_stress_calibration`

- Host: local `al-khwarizmi`
- Command: `python3 experiments/external_baseline_comparison_v1/scripts/run_stress_calibration.py`
- Log: `experiments/external_baseline_comparison_v1/logs/stress_calibration/stdout_20260824T234945Z.log`
- Result: `experiments/external_baseline_comparison_v1/stress_calibration/calibration_summary.json`
- Health: **COMPLETED** EXIT=0; M=32 selected

### 2. `extbase_joint_vtc`

- Command: `run_joint_external_baseline.py --baseline official_vtc_joint_token_budget_remap`
- Result: `experiments/external_baseline_comparison_v1/results/joint_vtc/`
- Health: **COMPLETED** 240/240 success in ~36s; integrity PASS; mean ANWG ≈ 0.2549

### 3. `extbase_joint_vllm`

- First attempt failed 38/240 on metric serialization bug (`slo_violation_rate is None`) — engineering only; partial archived under `results/joint_vllm_style_partial_metricbug_20260824T234840Z/`
- Rerun after fix: tmux `extbase_joint_vllm`, log `.../stdout_20260824T235120Z.log`
- **COMPLETED** 240/240 success (~88s). Partial failed run preserved at `results/joint_vllm_style_partial_metricbug_20260824T234840Z/`.

---

## I. Deliberately not launched

- SOLA (spec incomplete)
- LTR joint/public (Class D)
- Family A/B/C external matrices
- Stressed public full Pext matrix (calibration just froze M; execution not authorized as Phase-2 follow-on in same breath beyond calibration)
- Optional WAIT/QLM/Sarathi/architecture baselines
- Native vLLM joint GPU sweep

---

## J. Unresolved scientific questions

1. Is M=32 + capacity 512 “enough” discrimination, or must a **policy-blind capacity scale** be added next?
2. Is VTC token-budget remap acceptable to authors/reviewers as the fair joint adapter?
3. SOLA: proceed with faithful reimplementation, or demote from mandatory?
4. LTR: extend simulator request model with prompt text for public traces, or restrict LTR to text corpora only?

---

## K. Unresolved engineering questions

1. Whether Wulver should host later stressed-public full matrix (larger than joint).
2. Optional follow-up: add policy-blind **capacity** grid if M=32 proves non-discriminative under ANWG (must remain winner-blind).

---

## L. Exact next action

1. Author review of M=32 stress sufficiency (queue+ still only ~7%).  
2. If accepted: launch stressed-public Pext cells (vLLM-style + VTC) under M=32.  
3. Begin `sola_faithful` implementation against the written spec (separate authorization).  
4. Do **not** compute SBS/VBS(Pext) until SOLA/LTR scope is resolved and stressed cells exist or are explicitly deferred.
