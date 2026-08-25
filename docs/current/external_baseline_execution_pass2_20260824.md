# External-Baseline Execution Pass 2 — 2026-08-24

**Verdict:** `EXTERNAL_BASELINE_PASS2_PARTIAL_PENDING_RUNNING_JOBS`  
**HEAD:** `2987b7181efa2bc550d8a894c537eca8f6393eb6` (dirty tree preserved; no commit/push)

---

## Preflight

- Host: `al-khwarizmi`; branch `contextual-compositional-heuristics-20260731`; ahead 2; Wulver SSH OK (`login02`, `squeue` empty via `/apps/slurm/current/bin/squeue`).
- Canonical P6 joint matrix untouched; Pass-1 joint VTC/vLLM results not overwritten.

## M=32 diagnosis (policy-blind)

At C=512, M=32: max_active≈52 (≈10% of capacity), queue+≈0.072.  
**Conclusion:** arrival compression alone insufficient **and** active capacity unrealistically generous for 200-req windows (A+B).

## Capacity grid + criteria (predeclared)

- Grid: M∈{8,16,32} × C∈{256,128,64,32}; FIFO only; completion≥0.80; queue+≥0.10; util_max≥0.25 OR util_p99≥0.20; pick smallest M then largest C.
- Result: **(M=16, C=32)** selected. Meeting also (32,64) and (32,32).
- Artifact: `stress_calibration/pass2_MxC/calibration_summary.json`
- Workload frozen: **`public_trace_stress_v1`**

## Joint integrity

| Baseline | n | unique | fails | manifest match |
|---|---|---|---|---|
| VTC remap | 240 | 240 | 0 | yes |
| vLLM-style | 240 | 240 | 0 | yes |

Partial metric-bug vLLM run archived (engineering only).

## VTC remap fidelity

**A — necessary dimensional adaptation** (official token budget vs simulator count-cap). Disclosed label retained. Usable with disclosure.

## LTR

**`LTR_NOT_VALID_FOR_CURRENT_WORKLOAD`** (no prompt text on joint/public). Paths A+C recommended. Not ESTF.

## SOLA

**Spec incomplete for implementation** (cost model, dual TTFT/TPOT SLOs, peak-memory fits). No code/tests this pass. Spec §6 updated.

## Stressed-public launches

| Method | Session | Log | Result | Health |
|---|---|---|---|---|
| VTC | `extbase_stress_pub_vtc` | `logs/public_trace_stress_v1/vtc_20260824T235948Z.log` | `results/public_trace_stress_v1_vtc/` | **COMPLETE 60/60** (~14s) |
| vLLM-style | `extbase_stress_pub_vllm` | `logs/public_trace_stress_v1/vllm_20260824T235948Z.log` | `results/public_trace_stress_v1_vllm_style/` | **RUNNING_HEALTHY** (progressing; leave running) |

Inspect: `tmux attach -t extbase_stress_pub_vllm` or `tail -f .../vllm_20260824T235948Z.log`.

## Not launched

SOLA (fidelity); LTR (representation); Pext envelope analysis (premature).

## Exact next action

1. Confirm vLLM stressed-public `summary.json` reaches 60/60.  
2. Author-accept `public_trace_stress_v1` (M=16,C=32).  
3. Complete SOLA mapping (dual SLO + cost-model defaults) before any `sola_faithful` code.  
4. Resolve LTR path (prompt-bearing only vs demote from Pext).  
5. Only then compute SBS/VBS(Pext) — still **without** cherry-picking.

## Safety

No commit/push/reset; P6 untouched; no winner-based stress selection; no scientific Pext conclusions drawn.
