# Baseline Status Index

Canonical cross-baseline status table. Update this file whenever a faithful
external scheduler integration or evaluation changes status. Dated audit files
remain historical provenance, not live status.

Naming convention:

- **scheduler / policy**: executable scheduling algorithm;
- **heuristic**: decision rule or synthesized scheduling rule;
- **module / primitive**: reusable composable sub-policy behavior;
- **faithful external baseline**: pinned paper/artifact reimplementation or
  official-code adapter used for evaluation.

## Current Summary

| Baseline | Implementation Status | Evaluation Status | Current Classification | Next Action |
|---|---|---|---|---|
| vLLM-LTR | Complete checkpoint-backed adapter | Complete for tested regime | `EVALUATION_ONLY` | None queued |
| PARS-Serve-2026 | Complete official-code reproduction with locally trained checkpoint | Complete across WildChat/control and canonical-suite evidence | `EVALUATION_ONLY` | None queued |
| Sarathi-Serve | Complete faithful reimplementation plus Wulver real-GPU validation | Complete mechanism-level stress coverage; known simulator limit documented | Foundational internal comparison, with structural caveat | None queued |
| VTC | Complete official-code adapter around real `VTCReqQueue` | Fairness sweep complete | `FOUNDATIONAL_CANDIDATE` scientific / `EVALUATION_ONLY` deployable | Native non-wrapped implementation before deployable registration |
| Llumnix | Complete faithful reimplementation (`llumnix_faithful`) | Comparative evaluation and independent verification complete | `FOUNDATIONAL_CANDIDATE` | None queued |
| DistServe | Complete faithful reimplementation | Comparative evaluation complete | `FOUNDATIONAL_CANDIDATE_FOR_DISAGGREGATION_PRIMITIVES_ONLY` | None queued |
| Apt-Serve | Complete through Phase G collection and posthoc analysis | Phase G analysis complete; structurally valid dataset | `STRATEGY_C_VIABLE_WITH_LIMITATIONS`; positive marginal portfolio contribution; no global superiority claim | Use as input to module-envelope interpretation |

## Apt-Serve Current Record

Artifacts:

- failed SS15 source run:
  `results/apt_serve_phase_g_overnight_20260807_011542/`;
- completed resumed collection:
  `results/apt_serve_phase_g_resume_20260807_174028/`;
- canonical posthoc analysis:
  `results/apt_serve_phase_g_analysis_20260809_190000/`;
- scientific audit:
  [`docs/audits/apt_serve_phase_g_analysis_20260809.md`](audits/apt_serve_phase_g_analysis_20260809.md).

Supported by the completed analysis:

- dataset validation: `STRUCTURALLY_VALID`;
- total experiment units: `2175`;
- Stage 1 screening: `1599` units, 41 regimes, seeds 1001-1039;
- Stage 2 confirmation: `576` units, 16 regimes, seeds 2001-2036;
- Apt-Serve primary transition-cost setting: `1x`;
- Apt-Serve mean ANWG: `0.224845`;
- best fixed baseline by mean ANWG: `scorpio_style_slo_guard` at `0.207310`;
- global Apt-vs-best-fixed grouped bootstrap mean gap: `0.012032`, CI
  `[-0.013237, 0.046700]` - does **not** exclude zero;
- leave-one-out marginal contribution mean: `0.025219`, CI
  `[0.004099, 0.057757]` - excludes zero.

Interpretation:

- Positive marginal portfolio contribution; no global superiority claim.
- Apt-Serve appears useful as a portfolio member in specific contexts.
- Apt-Serve does not currently justify a global “beats best fixed baseline”
  claim.
- Apt-Serve is one external scheduler family and one mechanism source for the
  broader contextual-compositional system.

## Historical Detail

Use these docs for provenance:

- `docs/audits/apt_serve_official_artifact_audit_20260805.md`
- `docs/audits/apt_serve_strategy_c_wulver_probe_20260806.md`
- `docs/audits/apt_serve_phase_f_headroom_stress_validation_20260806.md`
- `docs/audits/apt_serve_phase_g_ss15_incident_20260807.md`
- `docs/audits/llumnix_first_comparative_evaluation_20260806.md`
- `docs/audits/distserve_first_comparative_evaluation_20260806.md`
- `docs/audits/vtc_fairness_comparative_evaluation_20260805.md`
- `docs/audits/pars_first_comparative_evaluation_20260804.md`
- `docs/wulver_sarathi_vllm_repeated_validation.md`

Do not rewrite those point-in-time audits just because the live status has
advanced.
