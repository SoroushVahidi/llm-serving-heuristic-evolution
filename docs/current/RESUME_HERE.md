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

**Family A v2 fairness-vs-size pilot Job 1182377 is EXECUTED and SCIENTIFICALLY
ANALYZED.**

- Audit: [`../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v2_20260816.md)
- Provenance: [`../../experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/`](../../experiments/policy_separation_fairness_starvation_pilot_v2_20260816T220113Z_1182377/)
- Verdict: **`USEFUL_BUT_NEEDS_REFINEMENT`** — include in PSD; next mechanism family justified
- Key positive results: ESTF↔WFS bidirectional @ε=0.01 (**26 / 29**); near-tie
  **18%** (vs v1 60%); Aging perfect **8.3%**; BurstGPT staged; canonical ANWG
- Failed predecessor Job 1182373 (BurstGPT plural header) fixed before relaunch

Family A v1 Job 1182306 remains frozen diagnostic evidence
(`USEFUL_DIAGNOSTIC_ONLY` / `REDESIGN_REQUIRED`; historical CSV `anwg` =
unweighted SLO-success, not canonical ANWG):
[`../audits/policy_separation_fairness_starvation_pilot_v1_20260816.md`](../audits/policy_separation_fairness_starvation_pilot_v1_20260816.md).

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
- WS-P: Family A v2 analyzed (`USEFUL_BUT_NEEDS_REFINEMENT`); **next mechanism
  family is the next PSD step**.

## Exact Next Tasks (two independent threads)

1. **WS-P:** After ESTF↔WFS composition pilot
   (`experiments/estf_wfs_composition_falsification_v1_20260816T222108Z/`,
   tmux `estf-wfs-comp-pilot`) finishes, read the verdict and write the audit.
   Do not start MAP-Elites or LLM synthesis yet.
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
