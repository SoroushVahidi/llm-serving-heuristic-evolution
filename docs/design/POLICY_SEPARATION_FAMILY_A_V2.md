# Policy Separation Family A v2 — Design (Fairness vs Size)

**Date:** 2026-08-16  
**Status:** PILOT COMPLETE + ANALYZED (Job 1182377); verdict
`USEFUL_BUT_NEEDS_REFINEMENT`  
**Predecessor:** Family A v1 Job 1182306 —
[`docs/audits/policy_separation_fairness_starvation_pilot_v1_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v1_20260816.md)  
**Verdict of v1:** `USEFUL_DIAGNOSTIC_ONLY / REDESIGN_REQUIRED`

## 1. Scientific goal

Construct a controlled multi-tenant workload family where **size/service-time
optimization** and **tenant priority/fairness** can conflict, so that
`estimated_service_time_first` (ESTF) and `weighted_fair_share` (WFS) can each
win in interpretable regimes (bidirectional separation), without forcing any
single universal winner.

This is **not** a search for the best scheduler. It is a discriminative
dataset family for policy-separation / complementarity evidence.

## 2. What failed in v1 (must not repeat)

| Confound | v1 pattern | v2 fix |
|---|---|---|
| Size∥priority | interactive=short+high; bulk=long+low | Orthogonal `favored_tenant_size` ∈ {short, long} |
| Oracle ESTF | predicted == actual | Controlled prediction noise + accurate control |
| Aging ceiling | success=1.0 on all 120 | Calibrated util/SLO/`max_active_sequences` so Aging is imperfect |
| Synthetic lengths | BurstGPT path miss | Require staged BurstGPT in production |
| Noncanonical primary | CSV `anwg` = unweighted SLO-success | Canonical `arrival_normalized_weighted_goodput` |

## 3. Causal structure

Two tenants with equal volume (50/50):

| Tenant | `class_id` (observable) | `priority` (observable) | Size profile (treatment) |
|---|---|---|---|
| Favored | `tenant_favored` | `tenant_weight_skew` | short **or** long per factor |
| Other | `tenant_other` | `1.0` | the complementary size class |

**Key treatment:** `favored_tenant_size`

- `short`: favored tenant draws short BurstGPT lengths; other draws long  
  (recovers a v1-like alignment cell for contrast)
- `long`: favored tenant draws **long** lengths; other draws short  
  (**orthogonal conflict cell**: WFS prefers long+high-priority; ESTF prefers short+low-priority)

Equal-weight control: `tenant_weight_skew = 1.0` (both priorities 1.0; size
profiles still follow `favored_tenant_size` for structural contrast).

### Factor levels (pilot)

| Factor | Levels | Notes |
|---|---|---|
| `favored_tenant_size` | `short`, `long` | Orthogonality axis |
| `tenant_weight_skew` | `1.0`, `5.0`, `10.0` | 1.0 = equal-weight control |
| `target_utilization` | `1.10`, `1.30`, `1.50` | Calibrated pressure band (2026-08-16 smoke) |
| `prediction_noise_sigma` | `0.0`, `0.30` | 0 = accurate; 0.30 = moderate unbiased lognormal noise |
| `seed` | `20260816`, `20260817` | Two seeds |

Fixed (documented; calibrated 2026-08-16):

- `n_total_jobs = 120` (60 per tenant)
- `max_active_sequences = 1` (ordering contention; reduced Aging monopoly vs mas=2)
- Favored SLO slack = `1.0s`; other SLO slack = `8.0s`
- BurstGPT token shapes required in production

**Grid size:** \(2 \times 3 \times 3 \times 2 \times 2 = 72\) scenarios  
**Policies:** 4 → **288 evaluations** (~0.6× v1’s 480)

## 4. Field provenance

| Field | Kind | Policy-visible? | Selector-eligible? |
|---|---|---|---|
| `arrival_time` | derived (synthetic Poisson) | yes | yes |
| `prompt_tokens`, `actual_output_tokens` | real-trace-shaped (BurstGPT) | prompt yes; actual no | prompt yes |
| `predicted_output_tokens` | derived (actual × noise) | yes | yes |
| `priority` | synthetic intervention | yes | yes |
| `class_id` (`tenant_favored` / `tenant_other`) | synthetic intervention | yes | yes |
| `slo_deadline` | synthetic intervention | yes (as deadline) | yes |
| `favored_tenant_size`, util, skew, noise σ, seed | hidden generator metadata | **no** | **no** (train/eval split keys only) |
| `token_length_source`, BurstGPT path | provenance | no | no |

Anti-leakage: `class_id` must not embed util/skew/size labels beyond the two
tenant names. Size is only in token fields.

## 5. Metrics

**Primary:** `RunMetrics.arrival_normalized_weighted_goodput` (canonical ANWG)  
**Secondary:** unweighted SLO-success, completion fraction, per-tenant
violation rates, Jain index on tenant goodputs, mean TTFT.

Never name a simplified metric `anwg`.

## 6. Pre-registered hypotheses

| ID | Hypothesis | CONFIRM | CONTRADICT | DESIGN_CONFOUND |
|---|---|---|---|---|
| H1 | FIFO suffers under contention when order conflicts with urgency/size | FIFO interactive/favored viol. ↑ with util; often loses on ANWG | FIFO competitive with specialists under overload | Flat load / no SLO pressure |
| H2 | ESTF benefits short jobs regardless of tenant priority | Under `favored_tenant_size=long`, ESTF still favors short/`tenant_other` (lower other-violations or higher short-class success) | ESTF ignores size | Perfect noise-free + degenerate sizes |
| H3 | WFS responds to tenant weights even when favored has long jobs | Under `favored=long` and skew>1, raising skew ↓ favored violations and/or ↑ WFS ANWG vs skew=1 | Skew has no effect | WFS not reading priority |
| H4 | Cells where ESTF materially beats WFS (size dominates) | ESTF≻WFS count > 0 at ε=0.01 on canonical ANWG | Never | Grid never stresses size |
| H5 | Cells where WFS materially beats ESTF (priority dominates) | WFS≻ESTF count > 0 at ε=0.01, especially `favored=long` + high skew | Never (v1 failure mode) | Still size∥priority |
| H6 | ESTF↔WFS bidirectional at practical ε=0.01 | H4 and H5 both true | Either direction empty | Metric/ties mask |
| H7 | Aging avoids starvation but does not universally dominate | Aging unique-win rate ≪ 1; not ANWG=1 everywhere | Aging ANWG=1 on all scenarios | Under-stressed grid |
| H8 | Near-tie rate at ε=0.01 materially below v1’s 60% | Rate ≤ ~0.45 | Rate ≥ 0.55 | ε mismatch |
| H9 | Effects stable across seeds | Winner-set agree ≥ ~0.75 on cells | Agree ≪ 0.6 | Too few seeds |
| H10 | Canonical ANWG and fairness metrics show a real tradeoff | WFS/ESTF niches disagree in places; JFI not identical to ANWG ranking | Same unique winner everywhere on both | Metric collapse |

## 7. Policies

`fifo`, `estimated_service_time_first`, `aging_priority`, `weighted_fair_share`.  
VTC deferred (scope).

## 8. Success criteria (design GO)

- Canonical ANWG in results  
- BurstGPT path used (no silent synthetic in production)  
- Size⊥priority via `favored_tenant_size`  
- Prediction not always oracle  
- Smoke: ESTF and WFS differ; Aging not trivially perfect; some SLO failures; both ESTF>WFS and WFS>ESTF appear in at least one smoke cell  
- Equal-weight controls present  
- Reproducible seeds; no hidden-factor leakage  

## 9. Non-goals

MAP-Elites, selector retraining, module composition, symbolic distillation,
LLM evolution, large real-vLLM runs.

## 10. Smoke / calibration log (2026-08-16)

Local synthetic smoke (24 cells × settings) found:

| Setting (favored_slo, other_slo, mas) | Aging perfect | ESTF≻WFS | WFS≻ESTF | Aging unique | Mean Aging ANWG |
|---|---:|---:|---:|---:|---:|
| 2.0 / 12.0 / 2 (initial draft) | 22/24 | 3 | 5 | ~all-best | ~0.998 |
| 1.0 / 8.0 / 2 | 12/24 | 8 | 10 | 14 | 0.918 |
| **1.0 / 8.0 / 1 (chosen)** | **6/24** | **10** | **10** | **6** | **0.785** |
| 0.8 / 6.0 / 1 | 0/24 | 10 | 9 | 3 | 0.707 |

Chosen fixed parameters prioritize bidirectional ESTF↔WFS with imperfect Aging
without extreme under-SLO that would make all policies fail equally. Pilot util
levels set to `[1.10, 1.30, 1.50]`.

## 11. Cluster pilot launch record (2026-08-16)

### Attempt 1 — Job 1182373 — FAILED

| Field | Value |
|---|---|
| Git SHA at submit | `5461e51` (later cluster HEAD `65e0a1d`) |
| tmux session | `family-a-v2-pilot` |
| Slurm job ID | `1182373` |
| Submit command | `sbatch scripts/slurm/run_policy_separation_fairness_starvation_pilot_v2.sbatch` |
| Scratch | `/mmfs1/scratch/ikoutis/sv96/policy_separation_fairness_starvation_pilot_v2_20260816T215822Z_1182373` |
| Result | `FAILED` ExitCode `1:0` after ~23s once past CONFIGURING |
| Root cause | BurstGPT CSV headers are `Request tokens` / `Response tokens`; loader looked for `Request Token` / `request_token` → `KeyError` |
| Fix | Use shared `detect_burstgpt_schema` in v2 loader; add regression test |

### Attempt 2 — Job 1182377 — COMPLETE

| Field | Value |
|---|---|
| Git SHA at run | `16ad5d3e5af2e02516dfc42cc0825fa8eb7cbf38` |
| Slurm job ID | `1182377` |
| Submit command | `sbatch scripts/slurm/run_policy_separation_fairness_starvation_pilot_v2.sbatch` |
| Scratch | `/mmfs1/scratch/ikoutis/sv96/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377` |
| Result | `COMPLETED` ExitCode `0:0`; 288/288 successes; BurstGPT staged |
| Audit | [`docs/audits/policy_separation_fairness_starvation_pilot_v2_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md) |
| Verdict | `USEFUL_BUT_NEEDS_REFINEMENT` |

**Next WS-P action:** design/execute the next mechanism family (Family A
optional refinements are non-blocking).
