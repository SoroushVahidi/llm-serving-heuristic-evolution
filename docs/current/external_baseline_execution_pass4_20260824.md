# External-Baseline Execution Pass 4 — 2026-08-24

**Verdict:** `EXTERNAL_BASELINE_PASS4_COMPLETE`  
**HEAD:** `2987b7181efa2bc550d8a894c537eca8f6393eb6` (dirty preserved; no commit/push)

**Author scope accepted:** `Pext_common` = P6 + official VTC + `vllm_style_continuous_batching`. SOLA demoted to Related Work. LTR not in common matrix.

---

## Jobs (all completed within ~3 min health window)

| Job | tmux | Host | Result | Status |
|---|---|---|---|---|
| Family A VTC | `extbase_pass4_family_a_vtc` | al-khwarizmi | `results/family_a_vtc/` | 72/72 COMPLETE (~37s) |
| Family B vLLM-style | `extbase_pass4_family_b_vllm` | al-khwarizmi | `results/family_b_vllm_style/` | 32/32 COMPLETE (~45s) |
| Family C vLLM-style | `extbase_pass4_family_c_vllm` | al-khwarizmi | `results/family_c_vllm_style/` | 72/72 COMPLETE (~15s) |
| Stressed-public P6 | `extbase_pass4_stress_p6` | al-khwarizmi | `results/public_trace_stress_v1_p6/` | 360/360 COMPLETE (~35s) |

Backend: `LOCAL_TMUX_CPU`. Logs: `experiments/external_baseline_comparison_v1/logs/pass4/`.

---

## Integrity

- Joint P6 1440 cells; VTC 240; vLLM-style 240; scenario IDs align; canonical SBS/VBS/HR reproduced.
- Stressed public: P6 360 + VTC 60 + vLLM 60; M=16 C=32; sources 20×3; hash `stress_v1_M16_C32`.
- Family A/B/C: full success; rebuild matches unified WFS/ESTF on Family A spot-check.

---

## Analysis artifacts

Under `experiments/external_baseline_comparison_v1/analysis/`:

- `joint_240_pext_matrix.csv`, `joint_pext_summary.json`, `joint_pext_policy_summary.csv`, `joint_pext_winner_summary.csv`, `joint_pext_bootstrap.json`
- `public_stress_pext_summary.json`, `public_stress_pext_matrix.csv`
- `family_a_vtc_summary.json`, `family_b_vllm_style_summary.json`
- `ltr_separate_evidence_summary.json`

Narrative: `docs/current/external_baseline_comparison_analysis_20260824.md`

---

## Scientific headline

External baselines do **not** destroy portfolio complementarity on joint-240. SBS unchanged; VBS/headroom rise negligibly via VTC; all P6 policies retain envelope contribution. No adaptive selector retraining in this pass.

---

## Safety

No commit/push/reset; P6/joint matrices untouched; no SOLA implementation; no LTR fabrication; no post-hoc redesign after viewing external performance.
