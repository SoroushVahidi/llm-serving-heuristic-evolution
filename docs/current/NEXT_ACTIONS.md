# Next Actions

This is the current prioritized action list. It must agree with
[`../PROJECT_MAP.md`](../PROJECT_MAP.md), [`RESUME_HERE.md`](RESUME_HERE.md),
[`WORK_STATUS.md`](WORK_STATUS.md), and [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md).

## P0 - Policy Separation (WS-P)

**Analyze Family B v1 (prefill/decode chunk-control) after the full pilot completes.**
This is the **next mechanism family** after Family A v2.

Design: [`../design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md`](../design/POLICY_SEPARATION_FAMILY_PREFILL_DECODE_V1.md)  
Smoke GO: `experiments/policy_separation_prefill_decode_smoke_v1_20260817T020443Z/`

ESTF↔WFS composition falsification is COMPLETE:
[`../audits/estf_wfs_composition_falsification_v1_20260816.md`](../audits/estf_wfs_composition_falsification_v1_20260816.md)  
**Verdict: `SELECTION_SUFFICIENT_FOR_THIS_PAIR`.** Contextual rank composition
did not beat contextual top-1 or expand the parent envelope. More complex
composition, symbolic distillation, MAP-Elites, and LLM synthesis are **not**
justified from that pair alone.

Family A v2 remains validated complementary-parent evidence
([`../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md)).

Stop conditions for this thread:

- Do not start MAP-Elites / CMA-ES / QD from ESTF/WFS composition.
- Do not start symbolic distillation from this teacher.
- Do not use Fireworks/Cloudrift LLM APIs yet.
- Do not escalate composition model complexity to rescue this NO_GO.

## P0 - Apt-Serve / module envelope (independent)

**post-Phase-G module-envelope interpretation.**

Inputs:

- Phase G collection:
  `results/apt_serve_phase_g_resume_20260807_174028/`
- Preserved failed SS15 run:
  `results/apt_serve_phase_g_overnight_20260807_011542/`
- Canonical Phase G analysis:
  `results/apt_serve_phase_g_analysis_20260809_190000/`
- Audit:
  [`../audits/apt_serve_phase_g_analysis_20260809.md`](../audits/apt_serve_phase_g_analysis_20260809.md)

Deliverable:

- Decide which Apt-Serve cache/tier-transition mechanisms are candidates for
  typed module decomposition.
- Decide whether those mechanisms should enter a library-envelope evaluation
  tool as module candidates.
- Define the next WS-H/WS-K experiment without launching another broad
  Apt-Serve sweep.

Existing typed DSL / module-composition work remains available as infrastructure;
it does not substitute for Family A v2 on the WS-P thread.

## P1 - Library-Envelope Tooling

Build a standing evaluator for:

- existing-policy marginal contribution `MC_i(x; P)`;
- candidate marginal gain `MG_c(x; P)`;
- grouped bootstrap CIs by regime/context family;
- clear win/tie/loss classification at practical epsilon thresholds.

This should generalize the Phase G analysis pattern beyond Apt-Serve.

## P2 - Module Decomposition

Select one or two mechanisms from the completed external-baseline work and map
them into the typed DSL/module vocabulary. Apt-Serve's tier-transition behavior
is now a candidate input, but it is not the only candidate; Sarathi, VTC,
Llumnix, and DistServe mechanisms should stay visible.

## P3 - CC6 Decision

Only after P0-P2, decide whether CC6 dynamic adaptation should start. If it
starts, keep it restricted to the previously trusted CC5 operating envelope and
retain safe fallback outside that envelope.

## Stop Conditions

- Do not claim Phase G proves global Apt-Serve superiority.
- Do not rerun Phase G collection unless a concrete missing/invalid cell is
  discovered.
- Do not delete Phase G artifacts.
- Do not convert Apt-Serve into the whole project narrative.
- Do not start broad symbolic synthesis until module contribution evidence is
  stronger.
- Do not treat Job 1182306 as BurstGPT-anchored or as using canonical ANWG.
- Do not treat Family A v1 as corpus-ready policy-separation evidence for QD.
