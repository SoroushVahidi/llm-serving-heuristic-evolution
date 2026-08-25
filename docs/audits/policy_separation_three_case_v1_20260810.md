# Policy Separation Three-Case Diagnostic v1 -- Scientific Audit

## Provenance

- Slurm job: `1170116`
- Run directory: `/mmfs1/scratch/ikoutis/sv96/policy_separation_three_case_20260810T041406Z_1170116/`
- Repo checkout: `/mmfs1/project/ikoutis/sv96/github/llm-serving-heuristic-evolution-policy-separation-v1`
- Git branch: `policy-separation-v1-wulver-20260809`
- Git HEAD: `2a56fa8fc39ce7fc93479f7400037037187f7f57`
- Script: `scripts/run_policy_separation_three_case.py`, config: `configs/policy_separation_three_case_v1.yaml`
- Wall time: 81s total (68.3s execution phase, ~59.1 tasks/s with 8 CPU workers), ~189 MB RAM
- Scope: exactly three theory-grounded scenario families (FCFS convoy / head-of-line
  blocking, SJF size-prediction inversion, EDF unsalvageable overload). NOT the full
  5-family/25-template Policy Separation Dataset v1 corpus, no Sobol search, no
  MAP-Elites, no selector training.

## Integrity

- 880 scenarios, 4,040 (scenario, policy) evaluations
- `n_completed=4040`, `n_failed=0` -- structurally valid, no crashed cells
- No duplicate `(scenario_id, policy_name)` keys (enforced at build time and by resume logic)
- No oracle leakage: every template's policy-visible fields
  (`predicted_output_tokens`, `slo_deadline`, arrival order) are constructed without
  reading `actual_output_tokens`; case 2's prediction inversion deliberately diverges
  the two, which is the mechanism under test, not leakage (covered by
  `tests/test_policy_separation_three_case.py`)

## Results by family

### FCFS convoy / head-of-line blocking (`fcfs_convoy`)

- `hypothesis_validation.csv` classification: `PARTIALLY_CONFIRMED` (180 comparisons,
  `fraction_direction_confirmed=0.522`)
- At `offset=0.0` (long job and short burst arrive simultaneously, so both are visible
  to the policy at its very first scheduling decision): clean, reproducible separation.
  `estimated_service_time_first - fifo` mean ANWG gap ~= **+0.42**, with the
  expected-direction sign holding in **90/90** seed replicates at this offset.
- At `offset > 0.0` (the config's only other offset, `0.05`): the mechanism becomes
  **structurally uninformative** under `max_active_sequences=1`. The simulator advances
  in discrete `step_size=0.001s` jumps and only enqueues a request once
  `current_time >= arrival_time`; with a single admission slot, the long job is already
  running by the time any positive-offset short burst becomes visible, so no policy --
  size-aware or not -- has a genuine choice left to make. This collapses the family's
  pooled `PARTIALLY_CONFIRMED` classification (which mixes both offsets) relative to the
  clean offset=0.0 result above.

### SJF / size-prediction inversion (`sjf_prediction_inversion`)

- `hypothesis_validation.csv` classification: `CONTRADICTED` at the pooled level (160
  comparisons, `fraction_direction_confirmed=0.10`, diagnosis: "stress-control margin
  change is negative or opposite the expected direction") -- but this pooled figure
  mixes multiple `(heterogeneity, load)` buckets and does not by itself mean the
  mechanism is absent; it means the mechanism is *not monotonically in the hypothesized
  direction* once averaged across all inversion levels.
- The genuinely interesting finding is the **prediction-inversion decision boundary**:
  under strong heterogeneity + high load, `estimated_service_time_first`'s advantage
  over `fifo` declines **monotonically** from approximately **+0.113** (accurate
  prediction, `inversion_fraction=0.0`) to **-0.010** (full inversion,
  `inversion_fraction=1.0`) -- i.e. size-aware scheduling's benefit shrinks and can
  cross into a net harm as prediction quality degrades, exactly as the family's stated
  hypothesis (`CASE2_HYPOTHESIS` in `templates_three_case.py`) predicts, just not
  uniformly across the pooled comparison set.
- Family-level oracle headroom (best achievable oracle ANWG minus the best single fixed
  policy's ANWG, averaged across `sjf_prediction_inversion` scenarios) ~= **0.0177**,
  with **51.5%** of scenarios showing headroom `> 0.005`. This is a real, non-trivial
  gap -- unlike the SwissAI reanalysis below.

### EDF unsalvageable overload (`edf_unsalvageable_overload`)

- `hypothesis_validation.csv` classification: `CONFIRMED` (160 comparisons,
  `fraction_direction_confirmed=0.925`, `mean_margin_change=0.386`)
- `scorpio_style_slo_guard` beats `edf` in **all 120** stressed cells containing
  impossible (unsalvageable) jobs, and ties `edf` in all controls (loosened-deadline
  cells where nothing is actually unsalvageable) -- the cleanest result of the three
  families.
- `admission_control` is **almost behaviorally identical to `edf`** in this experiment.
  This is a real, structural finding, not measurement noise -- see
  `docs/audits/policy_separation_edf_admission_mechanism_20260810.md` (written for the
  boundary-refinement experiment's Study C) for the code-level reason: as configured
  here, `AdmissionControlPolicy`'s `laxity_threshold` defaults to `float("inf")`, which
  makes its admission *filter* a no-op -- it only ever re-sorts the queue by laxity
  instead of `edf`'s raw deadline, and never actually rejects/deprioritizes an
  unsalvageable job the way `scorpio_style_slo_guard`'s guard-and-throttle logic does.

### SwissAI comparison

The independent SwissAI V2 policy-sweep reanalysis
(`docs/audits/swissai_v2_policy_sweep_reanalysis_20260809.md`, 512 windows x 27
policies, oracle vs. best-fixed) found the policy-reward landscape **saturated**: mean
oracle/best-fixed ANWG ~= 0.9925, strict oracle gain ~= 0, 0/512 unique wins, all 512
windows near-tied at every epsilon tested (0.001/0.005/0.01). This three-case
diagnostic's **targeted, theory-grounded synthetic scenarios broke that saturation** --
the FCFS convoy (offset=0.0), prediction-inversion, and EDF-overload families all
produce real, reproducible, interpretable separation that the SwissAI-derived windows
did not. This is the central justification for continuing with targeted synthetic
generator refinement (this audit's own successor experiment) rather than drawing more
real-trace-derived windows.

## Readiness assessment

- **Larger structured exploration is justified.** All three mechanisms produced at
  least one clean, reproducible separation regime; the immediate next step is
  characterizing *where* each regime's boundary sits (arrival-offset boundary for FCFS
  convoy, load x inversion-fraction x heterogeneity surface for SJF inversion, overload
  x impossible-fraction matrix for EDF), which is exactly the scope of the
  boundary-refinement experiment this document accompanies
  (`configs/policy_separation_boundary_refinement_v1.yaml`,
  `scripts/run_policy_separation_boundary_refinement.py`).
- **Module interventions are NOT yet justified.** No MAP-Elites, CMA-ES, Bayesian
  optimization, surrogate-assisted search, selector retraining, or module synthesis has
  been started or should be started until the decision boundaries found by the
  three-case diagnostic are mapped cleanly at finer resolution. This audit does not
  claim the full synthetic dataset (5-family/25-template Policy Separation Dataset v1
  corpus) exists -- only these three narrow mechanisms have been validated so far.
