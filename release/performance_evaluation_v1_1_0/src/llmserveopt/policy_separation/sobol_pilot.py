"""Policy Separation Sobol Pilot v1 -- scenario generation.

DESIGN-ONLY MODULE for the first space-filling exploration stage named by
`docs/design/POLICY_SEPARATION_DATASET_V1.md`'s roadmap (stage 2, "Sobol /
low-discrepancy exploration"), built strictly from the dimensions validated
by jobs 1170116 and 1171116 -- see `docs/design/POLICY_SEPARATION_SOBOL_PILOT_V1.md`
for the full architecture rationale. This module only builds scenarios; it
does not run them (see `scripts/run_policy_separation_sobol_pilot_v1.py`)
and no scientific sweep has been executed with it.

Two independent Sobol subspaces, NOT one merged hybrid space:

  Family B (prediction-sensitive scheduling): reuses
  `templates_boundary_refinement.case2_prediction_inversion_boundary`
  unmodified, Sobol-sampling (target_utilization, inversion_fraction)
  crossed with a categorical heterogeneity condition. `target_utilization`
  and `inversion_fraction` only make sense together with the SJF/inversion
  request-generation mechanism (heterogeneous job sizes + a controllable
  size-prediction error) -- `overload_factor`/`fraction_impossible` have no
  meaning in that generator.

  Family C (deadline/admission scheduling): reuses
  `templates_three_case.case3_edf_overload` unmodified, Sobol-sampling
  (overload_factor, fraction_impossible). No third "slack" dimension is
  added -- see job 1171116's own boundary-refinement config comment and
  `docs/audits/policy_separation_boundary_refinement_v1_20260810.md`:
  overload_factor already parameterizes deadline slack directly
  (`window_s` is inversely proportional to it inside `case3_edf_overload`),
  so a separate slack knob would double-count the same mechanism.

  FCFS categorical add-on (NOT Sobol): reuses
  `templates_three_case.case1_fcfs_convoy` unmodified at a small, fixed
  grid, because job 1171116 proved arrival_offset is a discontinuous
  mechanism switch (exact separation at offset=0.0 under
  max_active_sequences=1, exact zero at any offset>0), not a smooth
  numeric axis a space-filling sampler should waste budget exploring.

Anti-leakage: every field in a Sobol-generated `PolicySeparationScenario`'s
`params` (heterogeneity, target_utilization, inversion_fraction,
overload_factor, fraction_impossible, sobol_index, sobol_scramble_seed,
generator_family, ...) is generator-only bookkeeping recorded for later
analysis and held-out splitting -- NONE of it is passed into the
`Request`/`ObservableRequest`/`GPUConfig` objects a policy actually reads
(`predicted_output_tokens`, `slo_deadline`, arrival order, GPU capacity
fields only). See `classify_field` below and
`tests/test_policy_separation_sobol_pilot.py::test_no_generator_field_leaks_into_policy_visible_state`.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from .schema import PolicySeparationScenario
from .templates_boundary_refinement import case2_prediction_inversion_boundary
from .templates_three_case import case1_fcfs_convoy, case3_edf_overload

GENERATOR_VERSION = "sobol_pilot_v1"

# ---------------------------------------------------------------------------
# Sobol point generation
# ---------------------------------------------------------------------------

def sobol_unit_points(d: int, m: int, scramble_seed: int) -> np.ndarray:
    """Deterministic scrambled Sobol sequence in [0,1]^d, 2**m points.

    Requires scipy (already present in the environment both prior Slurm
    jobs ran in -- `scipy.stats.qmc.Sobol`, confirmed available at
    scipy>=1.10 -- not a new heavy dependency; declared in pyproject.toml
    as of this module).
    """
    from scipy.stats.qmc import Sobol  # local import: keep scipy optional at package-import time

    sampler = Sobol(d=d, scramble=True, seed=scramble_seed)
    return sampler.random_base2(m=m)


def _scale(u: float, lo: float, hi: float) -> float:
    return lo + u * (hi - lo)


# ---------------------------------------------------------------------------
# Field provenance classification (anti-leakage bookkeeping, section 13)
# ---------------------------------------------------------------------------

#: Generator-only / oracle fields: exist purely for analysis, held-out
#: splitting, and reproducibility. A selector trained on this corpus later
#: must NEVER read these as input features -- they either encode generator
#: identity directly or are computed from ground truth
#: (actual_output_tokens) that no online policy observes.
GENERATOR_ONLY_FIELDS = frozenset({
    "sobol_index", "sobol_scramble_seed", "generator_family", "generator_version",
    "template_name", "pair_id", "seed", "heterogeneity", "target_utilization",
    "overload_factor", "fraction_impossible", "rank_agreement_kendall_tau",
    "rank_agreement_spearman", "window_s", "n_normal", "n_impossible",
    "total_required_service_s", "role", "inversion_fraction", "ratio", "n_short",
    "offset", "max_active_sequences",
})

#: Simulator-derived but plausibly deployment-estimable online (a real
#: server could measure something like these at decision time, even though
#: this simulator does not currently expose them as policy input) --
#: recorded here as the DOCUMENTED set of future selector-safe proxies
#: (section 13), not fields this module currently emits as scenario params.
DEPLOYMENT_ESTIMABLE_PROXIES = (
    "queue_length", "realized_arrival_rate", "current_active_sequences",
    "kv_occupancy_fraction", "predicted_service_time_distribution_summary",
    "recent_prediction_residual_stats", "slo_slack_distribution_summary",
    "recent_drop_or_violation_rate",
)

#: Policy-visible fields: only ever populated on Request/ObservableRequest/
#: GPUConfig objects (predicted_output_tokens, prompt_tokens, arrival_time,
#: slo_deadline, priority, class_id, GPU capacity fields) -- never touched
#: by this module directly; enforced by reusing the existing template
#: functions' `req()`/`GPUConfig(...)` calls unmodified.
#:
#: NOTE two deliberate, harmless NAME collisions with GENERATOR_ONLY_FIELDS,
#: each a coincidence between an unrelated pre-existing GPUConfig field and
#: a generator-only scenario.params label with the same English name but a
#: different referent:
#:   - "max_active_sequences": GPUConfig's real capacity field (an integer
#:     the policy legitimately reads) vs. the FCFS add-on's generator-only
#:     params label for which convoy regime (mas=1 vs mas=4) a scenario
#:     belongs to. Both are real and both are correctly classified for
#:     their own object -- see test_no_generator_field_leaks_into_policy_visible_state.
#:   - "role": GPUConfig's unrelated disaggregated-prefill/decode field
#:     (core/types.py, always None in every Policy Separation scenario)
#:     vs. this pilot's generator-only stress/control role label.
POLICY_VISIBLE_FIELDS = frozenset({
    "request_id", "arrival_time", "prompt_tokens", "predicted_output_tokens",
    "slo_deadline", "priority", "class_id", "gpu_id", "max_active_sequences",
    "max_batch_tokens", "max_kv_tokens", "role",
})

#: The two coincidental name collisions documented above -- excluded from
#: the "must never appear on a GPUConfig" leak check specifically (they are
#: real GPUConfig fields, just not populated FROM the generator-only value
#: of the same name), not from the Request leak check (no collision there).
_GPU_CONFIG_NAME_COLLISIONS = frozenset({"max_active_sequences", "role"})


def classify_field(name: str) -> str:
    """Returns 'generator_only', 'policy_visible', or 'unknown' for a
    scenario-manifest field name -- used by tests to assert no
    generator-only field name collides with a policy-visible one."""
    if name in GENERATOR_ONLY_FIELDS:
        return "generator_only"
    if name in POLICY_VISIBLE_FIELDS:
        return "policy_visible"
    return "unknown"


# ---------------------------------------------------------------------------
# Family B: prediction-sensitive scheduling (Sobol)
# ---------------------------------------------------------------------------

FAMILY_B_RANGES = {
    "target_utilization": (0.50, 1.10),
    "inversion_fraction": (0.00, 1.00),
}


def generate_family_b_sobol_scenarios(
    m: int,
    scramble_seed: int,
    heterogeneity_levels: List[str],
    seeds: List[int],
) -> List[PolicySeparationScenario]:
    """Family B: (target_utilization, inversion_fraction) Sobol-sampled,
    crossed with a categorical heterogeneity condition and a seed list.
    `role`/`pair_id` are inherited from `case2_prediction_inversion_boundary`
    unmodified (role=="control" only in the near-impossible event a Sobol
    point lands exactly at inversion_fraction==0.0); this pilot does NOT
    build stress/control pairing summaries the way the diagnostic
    experiments did -- see this module's docstring and the design doc.
    """
    points = sobol_unit_points(d=2, m=m, scramble_seed=scramble_seed)
    scenarios: List[PolicySeparationScenario] = []
    for idx, (u_util, u_inv) in enumerate(points):
        target_utilization = round(_scale(float(u_util), *FAMILY_B_RANGES["target_utilization"]), 6)
        inversion_fraction = round(_scale(float(u_inv), *FAMILY_B_RANGES["inversion_fraction"]), 6)
        for heterogeneity in heterogeneity_levels:
            for seed in seeds:
                base = case2_prediction_inversion_boundary(
                    target_utilization, heterogeneity, inversion_fraction, seed
                )
                scenarios.append(_tag_sobol(
                    base, generator_family="sobol_family_b_prediction_sensitive",
                    sobol_index=idx, sobol_scramble_seed=scramble_seed,
                    scenario_id_prefix="sobolB",
                ))
    _assert_no_duplicate_ids(scenarios)
    return scenarios


# ---------------------------------------------------------------------------
# Family C: deadline / admission scheduling (Sobol)
# ---------------------------------------------------------------------------

FAMILY_C_RANGES = {
    "overload_factor": (0.85, 1.40),
    "fraction_impossible": (0.00, 0.80),
}


def generate_family_c_sobol_scenarios(
    m: int,
    scramble_seed: int,
    seeds: List[int],
) -> List[PolicySeparationScenario]:
    """Family C: (overload_factor, fraction_impossible) Sobol-sampled,
    role="stress" only -- the loosened-deadline control mechanism is
    already validated (job 1171116: exactly-0.0 margin in all 240 control
    cells) and does not need re-validation via Sobol sampling; spending
    Sobol budget on controls here would not add landscape-characterization
    information."""
    points = sobol_unit_points(d=2, m=m, scramble_seed=scramble_seed)
    scenarios: List[PolicySeparationScenario] = []
    for idx, (u_of, u_fi) in enumerate(points):
        overload_factor = round(_scale(float(u_of), *FAMILY_C_RANGES["overload_factor"]), 6)
        fraction_impossible = round(_scale(float(u_fi), *FAMILY_C_RANGES["fraction_impossible"]), 6)
        for seed in seeds:
            base = case3_edf_overload(overload_factor, fraction_impossible, seed, role="stress")
            scenarios.append(_tag_sobol(
                base, generator_family="sobol_family_c_deadline_admission",
                sobol_index=idx, sobol_scramble_seed=scramble_seed,
                scenario_id_prefix="sobolC",
            ))
    _assert_no_duplicate_ids(scenarios)
    return scenarios


# ---------------------------------------------------------------------------
# FCFS categorical add-on (NOT Sobol -- see module docstring)
# ---------------------------------------------------------------------------

def generate_fcfs_categorical_add_on(
    a1_ratios: List[int], a1_short_counts: List[int], a1_seeds: List[int],
    a2_ratio: int, a2_short_count: int, a2_offsets: List[float],
    a2_max_active_sequences: int, a2_seeds: List[int],
) -> List[PolicySeparationScenario]:
    """Template A1 (mas=1, offset=0.0 -- the proven genuine-choice regime)
    crossed with a small (ratio, n_short) grid and both roles; Template A2
    (mas=positive, a couple of representative positive offsets -- the
    proven general-convoy regime) at one representative (ratio, n_short)
    cell. Both reuse `case1_fcfs_convoy` unmodified. Not Sobol-sampled:
    job 1171116 proved offset is a discontinuous categorical switch, so a
    fixed, small, already-validated grid is used instead of continuous
    sampling."""
    scenarios: List[PolicySeparationScenario] = []
    for ratio in a1_ratios:
        for n_short in a1_short_counts:
            for seed in a1_seeds:
                for role in ("stress", "control"):
                    scenarios.append(case1_fcfs_convoy(ratio, n_short, 0.0, seed, role, max_active_sequences=1))
    for offset in a2_offsets:
        for seed in a2_seeds:
            for role in ("stress", "control"):
                scenarios.append(case1_fcfs_convoy(
                    a2_ratio, a2_short_count, offset, seed, role,
                    max_active_sequences=a2_max_active_sequences,
                ))
    _assert_no_duplicate_ids(scenarios)
    return scenarios


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tag_sobol(
    scenario: PolicySeparationScenario, generator_family: str,
    sobol_index: int, sobol_scramble_seed: int, scenario_id_prefix: str,
) -> PolicySeparationScenario:
    """Attaches Sobol provenance to a reused template's scenario without
    modifying the template function itself: rewrites scenario_id with a
    prefix (kept collision-free against the underlying template's own
    scenario_id, which already encodes every generation parameter) and
    adds Sobol-specific keys to `params`. Uses `dataclasses.replace` since
    `PolicySeparationScenario` is frozen."""
    new_params = dict(scenario.params)
    new_params["sobol_index"] = sobol_index
    new_params["sobol_scramble_seed"] = sobol_scramble_seed
    new_params["generator_family"] = generator_family
    new_scenario_id = f"{scenario_id_prefix}.idx{sobol_index}.{scenario.scenario_id}"
    return replace(
        scenario,
        scenario_id=new_scenario_id,
        generator_version=GENERATOR_VERSION,
        params=new_params,
    )


def _assert_no_duplicate_ids(scenarios: List[PolicySeparationScenario]) -> None:
    ids = [s.scenario_id for s in scenarios]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate scenario_id(s) generated: {sorted(dupes)[:10]}")


# ---------------------------------------------------------------------------
# Validity guards (section 10) -- deterministic checks, not statistical
# repair: both Sobol ranges above were chosen (from Study A/B/C's own
# calibration) to lie entirely within the domain where the reused,
# already-tested generator functions produce valid output at every point
# in the unit hypercube, so no rejection/repair branch should ever fire.
# This function exists as the required explicit safety net anyway.
# ---------------------------------------------------------------------------

def validate_scenario(scenario: PolicySeparationScenario) -> List[str]:
    """Returns a list of validity violations (empty if none). Never raises
    -- callers decide whether a violation is fatal."""
    problems: List[str] = []
    if not scenario.requests:
        problems.append("no requests")
        return problems
    for r in scenario.requests:
        if r.arrival_time < 0:
            problems.append(f"negative arrival_time on request {r.request_id}")
        if r.prompt_tokens <= 0:
            problems.append(f"non-positive prompt_tokens on request {r.request_id}")
        if r.predicted_output_tokens <= 0:
            problems.append(f"non-positive predicted_output_tokens on request {r.request_id}")
        if r.actual_output_tokens <= 0:
            problems.append(f"non-positive actual_output_tokens on request {r.request_id}")
        if r.slo_deadline < r.arrival_time:
            problems.append(f"slo_deadline before arrival_time on request {r.request_id}")
    for g in scenario.gpu_configs:
        if g.max_active_sequences <= 0 or g.max_batch_tokens <= 0 or g.max_kv_tokens <= 0:
            problems.append(f"non-positive capacity field on gpu {g.gpu_id}")
    for leaked in GENERATOR_ONLY_FIELDS:
        # generator-only fields must live in scenario.params, never as an
        # attribute name collision on the Request/GPUConfig payload itself
        for r in scenario.requests:
            if hasattr(r, leaked):
                problems.append(f"generator-only field {leaked!r} exposed on Request")
                break
    return problems
