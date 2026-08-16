# Resume Here

**Shortest current operational entrypoint.** For the research roadmap, read
[`docs/PROJECT_MAP.md`](../PROJECT_MAP.md). For detailed status, read
[`WORK_STATUS.md`](WORK_STATUS.md). For ordered next actions, read
[`NEXT_ACTIONS.md`](NEXT_ACTIONS.md).

## Current State

| Field | Value |
|---|---|
| Repository | `llm-serving-heuristic-evolution` |
| Branch | `contextual-compositional-heuristics-20260731` |
| Last reconciled SHA | see `git rev-parse HEAD` (Family A v1 audit landed after `1be0056`) |
| Remote | `origin/contextual-compositional-heuristics-20260731` |
| Expected Git state | clean, 0 ahead / 0 behind after `git fetch --prune origin` |
| Canonical roadmap | `docs/PROJECT_MAP.md` |
| Cluster PSD worktree | `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1` |

Resume commands:

```bash
cd /home/soroush/llm-serving-heuristic-evolution
git fetch --prune origin
git status --short --branch
git rev-parse HEAD
git rev-list --left-right --count @{u}...HEAD
python scripts/check_project_handoff_consistency.py
```

## What This Project Is

This project builds toward a verified contextual compositional scheduler system
for LLM inference serving:

```text
context -> performance/marginal-contribution learning -> DSL/module selection
        -> verified composition/synthesis -> evaluation -> envelope expansion
```

The current primary metric is `arrival_normalized_weighted_goodput` (ANWG).

Typed DSL / module-composition infrastructure exists in-repo and is valuable,
but it does **not** by itself complete policy-separation decision-boundary
characterization (WS-P).

## Most Recently Completed Work (WS-P / Policy Separation)

**Family A Fairness and Starvation pilot Job 1182306 is EXECUTED and now
SCIENTIFICALLY ANALYZED.**

- Audit: [`../audits/policy_separation_fairness_starvation_pilot_v1_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v1_20260816.md)
- Provenance: [`../../experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/`](../../experiments/policy_separation_fairness_starvation_pilot_20260816T211029Z_1182306/)
- Analysis artifacts: `…/analysis/` (winners, pairwise, seed stability, fairness surfaces)
- Verdict: **`USEFUL_DIAGNOSTIC_ONLY` / `REDESIGN_REQUIRED`** — not corpus-ready
- Key negative result: ESTF↔WFS bidirectional fairness/size niche **absent**;
  Aging saturated (historical unweighted SLO-success = 1.0 on all 120 scenarios);
  size–priority collinearity confound
- Caveats unchanged: synthetic token lengths; historical CSV `anwg` =
  unweighted SLO-success (**not** canonical ANWG)

Predecessor: Sobol pilot v1 Job `1182183` (COMPLETE + analyzed) —
[`../audits/policy_separation_sobol_pilot_v1_20260816.md`](../audits/policy_separation_sobol_pilot_v1_20260816.md).

## Latest Major Result (Apt-Serve/CC thread)

**Apt-Serve Phase G completed.**

- Collection: complete.
- Posthoc analysis: complete with wrapper `exit_code=0`.
- Canonical collection output:
  `results/apt_serve_phase_g_resume_20260807_174028/`.
- Preserved failed SS15 source run:
  `results/apt_serve_phase_g_overnight_20260807_011542/`.
- Canonical analysis output:
  `results/apt_serve_phase_g_analysis_20260809_190000/`.
- Audit:
  [`../audits/apt_serve_phase_g_analysis_20260809.md`](../audits/apt_serve_phase_g_analysis_20260809.md).

Supported interpretation:

- The Phase G dataset is structurally valid.
- Apt-Serve has positive leave-one-out marginal contribution to the policy
  portfolio: mean `0.025219`, grouped bootstrap CI `[0.004099, 0.057757]`.
- Global Apt-vs-best-fixed superiority is not established: mean gap
  `0.012032`, grouped bootstrap CI `[-0.013237, 0.046700]`.
- The best fixed baseline by mean ANWG is `scorpio_style_slo_guard`.
- Apt-Serve is one evaluated external scheduler family and a potential source
  of cache/tier-transition modules, not the whole project.

## Current Project Position

- CC0-CC5: complete; CC5 remains `COMPLETE_REGIME_SPECIFIC`.
- CC6: not started; requires explicit authorization and a scoped design.
- External baselines: current status is centralized in
  [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md).
- Apt-Serve: Phase G analysis is complete; no new Apt-Serve collection job is
  queued.
- WS-P: Family A v1 analyzed; **Family A v2 redesign is the next PSD step**.

## Exact Next Tasks (two independent threads)

1. **WS-P:** Design/implement Family A v2 (orthogonal size×priority, BurstGPT
   anchoring, canonical ANWG, prediction noise, anti-Aging-saturation). Do not
   start MAP-Elites or selector retraining yet.
2. **Apt-Serve/CC:** Perform the post-Phase-G module-envelope interpretation and
   decide the next module-decomposition/compositional-learning step.

## Do Not Do By Default

- Do not claim Apt-Serve globally beats the best fixed baseline.
- Do not treat Apt-Serve as the project endpoint.
- Do not start CC6 without explicit authorization.
- Do not delete Phase G artifacts or historical negative-result audits.
- Do not start MAP-Elites, selector retraining, or broad synthesis from PSD yet.
- Do not train selectors on Family A v1 rows.
- Do not rewrite Job 1182306 CSV rows.
- Do not use local `results/` absence as proof an experiment never ran; check
  the audit trail.

## Navigation

- Public overview: [`../../README.md`](../../README.md)
- Research roadmap: [`../PROJECT_MAP.md`](../PROJECT_MAP.md)
- Detailed status: [`WORK_STATUS.md`](WORK_STATUS.md)
- Prioritized next actions: [`NEXT_ACTIONS.md`](NEXT_ACTIONS.md)
- External-baseline index: [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md)
- Documentation index: [`../README.md`](../README.md)
