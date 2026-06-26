# Phase 2B.10 SCORPIO-Style SLO Guard Summary

**Phase:** 2B.10  
**Date:** 2026-06-25  
**Branch:** `phase2b10-scorpio-slo-guard`  
**Policy name:** `scorpio_style_slo_guard`  
**Config:** `configs/phase2b10_scorpio_slo_guard.yaml`  
**Runner:** `scripts/run_phase2b10_scorpio_slo_guard.py`  
**Log:** `logs/phase2b10/phase2b10_scorpio_slo_guard.log` (gitignored)  
**Results:** `results/phase2b10_scorpio_slo_guard/` (gitignored; summary in this doc)  
**tmux session:** `phase2b10_scorpio_slo_guard` (completed, EXIT_CODE=0, 243.8s)

---

## Design Abstraction

This policy is a **SCORPIO-inspired, simulator-compatible SLO guard baseline**.  
Safe manuscript wording: **“SCORPIO-style SLO guard”** or **“SCORPIO-inspired TTFT/TPOT guard baseline.”**  
It is **not** an official SCORPIO reproduction.

### Core ideas approximated

| SCORPIO concept | Simulator implementation |
|-----------------|------------------------|
| TTFT / deadline guard | Filter requests whose **TTFT proxy** (`α × prompt_tokens × step_size`) exceeds remaining deadline budget |
| TPOT / decode pressure guard | Penalize long predicted decode under high `decoding_count / max_active_sequences` and `kv_fill_ratio` |
| Credit / budget batching | Refilling **admission credit budget** throttles new admissions under pressure (token-bucket proxy) |
| Admission control | Skip requests with negative **laxity** (`deadline − now − est_service_seconds`) |

### TTFT / TPOT proxies (not direct metrics)

- **TTFT proxy** = `step_size × α × prompt_tokens`
- **TPOT / decode pressure proxy** = `decoding_count / max_active_sequences` + KV fill ratio

### Parts not modelled

- Full SCORPIO ML violation-rate predictor
- Rolling production telemetry violation rate (queue/KV/decode proxies used instead)
- Preemption, migration, multi-node coordination

---

## Registry Status

| Item | Count |
|------|-------|
| Deployable policies | **20** |
| Selector candidates | **20** |
| Oracle (`oracle_srtf`) | **1**, excluded from candidates |

`scorpio_style_slo_guard` **is** a selector candidate.

---

## Experiment

Phase 2B.9 workload suite (60 windows: 27 dev + 33 held-out), **20** deployable policies,
repaired rule selector (unchanged dispatch table from Phase 2B.8).

### Phase 2B.9 reference (19 policies)

| Group | Rule selector WG | Best fixed WG | Best fixed policy |
|-------|------------------|---------------|-------------------|
| Dev | 0.917 | 0.893 | `weighted_shortest_processing` |
| Held-out | 0.979 | 0.970 | `edf` |
| Overall | 0.951 | 0.922 | `weighted_shortest_processing` |

### Phase 2B.10 results (20 policies)

| Metric | Dev | Held-out | Overall |
|--------|-----|----------|---------|
| **SCORPIO-style WG** | **0.988** | **0.998** | **0.993** |
| Best fixed WG | 0.988 | 0.998 | 0.993 |
| Best fixed policy | `scorpio_style_slo_guard` | `scorpio_style_slo_guard` | `scorpio_style_slo_guard` |
| Rule selector WG | 0.917 | 0.979 | 0.951 |
| **Selector gap vs best fixed** | **−0.071** | **−0.019** | **−0.042** |
| Per-window oracle/reference WG | 0.988 | 0.999 | 0.994 |
| Selector gap vs oracle | −0.071 | −0.021 | −0.044 |

**SCORPIO-style ranks #1** on dev, held-out, and overall (see `policy_ranking.csv`).

### SLO violation and completion (aux metrics)

| Group | Policy | SLO violation rate | Completion fraction |
|-------|--------|-------------------|---------------------|
| Dev | SCORPIO-style | **0.009** | 0.928 |
| Dev | EDF | 0.132 | 1.000 |
| Dev | admission_control | 0.145 | 1.000 |
| Held-out | SCORPIO-style | **0.002** | 0.966 |
| Held-out | EDF | 0.026 | 1.000 |

SCORPIO-style trades some completion (guard throttling) for much lower SLO violation rate.

### High-noise failure case (`heldout_very_high_noise_s4`)

| Policy | WG |
|--------|-----|
| Rule selector (`admission_control`) | 0.970 |
| SCORPIO-style | **1.000** |
| Per-window best | 1.000 |

**SCORPIO-style fixes the Phase 2B.9 high-noise failure; the rule selector does not.**

---

## Interpretation

### Is SCORPIO-style competitive?

**Yes — it becomes the best fixed baseline on all groups**, beating Phase 2B.9 best fixed by
+0.095 dev, +0.028 held-out, +0.071 overall.

### Does the repaired rule selector still beat best fixed?

**No.** Selector WG is unchanged from Phase 2B.9 (no SCORPIO routing rule), but best fixed
now includes SCORPIO-style. Gaps: −0.071 dev, −0.019 held-out, −0.042 overall.

### Should SCORPIO-style remain a selector candidate?

**Yes** — it is deployable, online-observable, and top-ranked. The selector must learn to
dispatch to it under overload/tight-SLO/high-noise regimes.

### New failure cases

See `docs/audits/phase2b10_failure_cases_summary.md`:
- fail_005: selector loses to SCORPIO-style best fixed
- fail_006: selector never chooses SCORPIO-style
- fail_004: partially resolved for SCORPIO; rule selector still fails

---

## Recommended Next Step

**Update selector rule dispatch (or RF/DT training with 20-class labels)** to route to
`scorpio_style_slo_guard` under overload, tight-SLO, and high-noise regimes; re-run
Phase 2B.10 comparison. SCORPIO-style's strength makes selector integration the priority
before adding PARS-style LTR or WAIT/KV baselines.
