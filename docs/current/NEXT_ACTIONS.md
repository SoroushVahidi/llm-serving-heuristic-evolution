# Next Actions

This is the current prioritized action list. It must agree with
[`../PROJECT_MAP.md`](../PROJECT_MAP.md), [`RESUME_HERE.md`](RESUME_HERE.md),
[`WORK_STATUS.md`](WORK_STATUS.md), and [`../BASELINE_STATUS.md`](../BASELINE_STATUS.md).

## P0 - Current Task

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

## P0 (Policy Separation thread, independent of the above)

**Design and validate Policy Separation Sobol Pilot v1; scientific execution
pending review.**

Inputs:

- Completed diagnostics: jobs `1170116`, `1171116`
- Audits: [`../audits/policy_separation_three_case_v1_20260810.md`](../audits/policy_separation_three_case_v1_20260810.md),
  [`../audits/policy_separation_boundary_refinement_v1_20260810.md`](../audits/policy_separation_boundary_refinement_v1_20260810.md),
  [`../audits/policy_separation_edf_admission_mechanism_20260810.md`](../audits/policy_separation_edf_admission_mechanism_20260810.md)
- Sobol pilot design: `configs/policy_separation_sobol_pilot_v1.yaml`,
  `scripts/run_policy_separation_sobol_pilot_v1.py`

Status: design, implementation, tests, and a local dry-run smoke are complete.
The scientific sweep has **not** been submitted to Slurm.

Deliverable before submission:

- Explicit review of the Sobol architecture (two subspaces -- prediction-sensitive
  and deadline/admission -- plus a categorical FCFS add-on) and scale (~1,000-1,800
  scenarios, ~7,000-12,000 evaluations).
- Do not mark the synthetic dataset complete, and do not start MAP-Elites or
  selector retraining, until this pilot has run and been analyzed.

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
