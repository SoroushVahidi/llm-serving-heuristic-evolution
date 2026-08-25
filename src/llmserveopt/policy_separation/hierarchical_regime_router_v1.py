"""Hierarchical Regime Router v1 -- Stage-1 router, dwell/fallback routing
FSM, deterministic split builder, and blended-regime microcase builders.

IMPLEMENTATION + VALIDATION ONLY. Implements
docs/design/HIERARCHICAL_REGIME_ROUTER_V1.md and
configs/hierarchical_regime_router_v1_gates.json exactly as frozen. No
final scientific TEST evaluation is performed by this module -- see
`hierarchical_router_evaluation_v1.py` (offline TRAIN/VAL/TEST-capable
evaluation code) and `hierarchical_router_gates_v1.py` (gate/verdict
logic); running either against the real TEST split for a scientific
verdict is a separate, explicitly authorized step (design doc S13/T).

Stage-1 inputs, target formulas, and thresholds are all imported (never
redefined) from `online_regime_signals_v1`, which is listed as a frozen,
immutable source in the gates config's `frozen_source_immutable`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..core.types import GPUConfig, Request
from .builders import kv_scarce_gpu, req
from .online_regime_signals_v1 import (
    CONTENTION_SCORE_V2_THRESHOLD,
    KV_PRESSURE_THRESHOLD,
    MIN_CONFLICT_QUEUE,
    PRIORITY_SKEW_THRESHOLD,
    TELEMETRY_LABEL_COLUMNS,
)
from .schema import PolicySeparationScenario
from .templates_fairness_starvation_v2 import case_fairness_vs_size_v2
from .templates_prefill_decode_v2 import case_prefill_decode_ttft_contention

SCHEMA_VERSION = "hierarchical_regime_router_v1.0.0"

# ---------------------------------------------------------------------------
# Frozen constants -- design doc SS C, D, G, K
# ---------------------------------------------------------------------------

#: Exactly the 4 fields frozen at design doc SS C. No other column may ever
#: be passed to Stage-1 `fit`/`predict`.
STAGE1_INPUT_COLUMNS: Tuple[str, ...] = (
    "contention_score_v2",
    "priority_skew",
    "kv_pressure",
    "queue_length",
)

#: Explicitly denied Stage-1 inputs (design doc SS C / gates json
#: `stage_1_excluded`) -- used only by tests/assertions, never read as a
#: feature.
STAGE1_FORBIDDEN_COLUMNS: Tuple[str, ...] = (
    "mechanism_family",
    "canonical_scenario_id",
    "scenario_id",
    "seed",
    "actual_output_tokens",
    "ttft",
    "tpot",
    "slo_violated",
    "mechanism_utility",
    "utility_gap",
)

REGIME_A = "RANKING_FAIRNESS"
REGIME_B = "PREFILL_DECODE_CONTENTION"
REGIME_C = "KV_MEMORY_PRESSURE"
REGIME_NONE = "NONE"
REGIME_OVERLAP = "OVERLAP"
REGIME_CLASSES: Tuple[str, ...] = (REGIME_A, REGIME_B, REGIME_C, REGIME_NONE, REGIME_OVERLAP)
ACTIVE_REGIMES: Tuple[str, ...] = (REGIME_A, REGIME_B, REGIME_C)
FALLBACK_REGIMES: Tuple[str, ...] = (REGIME_NONE, REGIME_OVERLAP)

#: Regime -> family, for scenario-level ground truth (design doc S6.C:
#: "an upper bound... uses mechanism_family (or equivalently the true
#: activity label, since it partitioned identically in the feasibility
#: telemetry)").
FAMILY_OF_REGIME: Dict[str, str] = {
    REGIME_A: "FAMILY_A_FAIRNESS_STARVATION_V2",
    REGIME_B: "FAMILY_B_PREFILL_DECODE_V2",
    REGIME_C: "FAMILY_C_KV_PRESSURE_V2",
}
REGIME_OF_FAMILY: Dict[str, str] = {v: k for k, v in FAMILY_OF_REGIME.items()}

#: Frozen fallback policy for NONE/OVERLAP (design doc SS F/L).
FALLBACK_POLICY = "weighted_fair_share"

#: Frozen dwell rule (design doc SS K).
DWELL_MINIMUM_STEPS = 20

#: Frozen Stage-2 native-pair candidate sets (design doc SS G).
STAGE2_CANDIDATES: Dict[str, Tuple[str, str]] = {
    REGIME_A: ("estimated_service_time_first", "weighted_fair_share"),
    REGIME_B: ("full_prefill", "chunked_prefill_small"),
    REGIME_C: ("kv_constrained_online", "least_laxity_first"),
}
#: Explicit foreign-mechanism exclusions (design doc SS G) -- used only by
#: tests, mirrors STAGE2_CANDIDATES by construction but stated separately
#: so a test can fail loudly if the two ever silently diverge.
STAGE2_EXCLUDED_CROSS_REGIME: Dict[str, Tuple[str, ...]] = {
    REGIME_A: ("kv_constrained_online",),
    REGIME_B: (),
    REGIME_C: ("estimated_service_time_first", "weighted_fair_share"),
}


# ---------------------------------------------------------------------------
# SS D -- Stage-1 target: deterministic ground-truth regime label
# ---------------------------------------------------------------------------

def regime_label_from_activity(a_active: bool, b_active_v2: bool, c_active: bool) -> str:
    """Map the three (already frozen, byte-for-byte reused)
    `compute_activity_labels` booleans to the 5-way Stage-1 target (design
    doc SS D). Never recomputes the booleans itself -- callers pass values
    already produced by `online_regime_signals_v1.compute_activity_labels`."""
    n_active = int(a_active) + int(b_active_v2) + int(c_active)
    if n_active > 1:
        return REGIME_OVERLAP
    if a_active:
        return REGIME_A
    if b_active_v2:
        return REGIME_B
    if c_active:
        return REGIME_C
    return REGIME_NONE


def add_regime_labels(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Attach the deterministic `regime_label` column to a telemetry frame
    that already has `a_active`/`b_active_v2`/`c_active` (e.g. loaded from
    the frozen `online_regime_telemetry_v1.csv`). Does not mutate the
    input frame."""
    missing = [c for c in ("a_active", "b_active_v2", "c_active") if c not in telemetry.columns]
    if missing:
        raise ValueError(f"telemetry frame missing required activity columns: {missing}")
    out = telemetry.copy()
    out["regime_label"] = [
        regime_label_from_activity(bool(a), bool(b), bool(c))
        for a, b, c in zip(out["a_active"], out["b_active_v2"], out["c_active"])
    ]
    return out


# ---------------------------------------------------------------------------
# SS E -- Stage-1 architecture: single multiclass classifier, simple model
# ---------------------------------------------------------------------------

def _assert_stage1_input_frame(X: pd.DataFrame) -> None:
    cols = list(X.columns)
    if cols != list(STAGE1_INPUT_COLUMNS):
        raise ValueError(
            f"Stage-1 input frame must have exactly columns {STAGE1_INPUT_COLUMNS} "
            f"in that order, got {cols}"
        )
    forbidden_present = set(cols) & set(STAGE1_FORBIDDEN_COLUMNS)
    if forbidden_present:
        raise ValueError(f"Stage-1 input frame contains forbidden columns: {forbidden_present}")


class Stage1Router:
    """Single multiclass classifier over the 5-way regime label, using
    exactly the 4 frozen Stage-1 inputs (design doc SS C/E). Simple model
    class (logistic regression, multinomial) -- no model zoo."""

    def __init__(self, seed: int = 20260817) -> None:
        self.seed = seed
        self.model = LogisticRegression(max_iter=2000, random_state=seed)
        self._fitted_classes: Optional[List[str]] = None

    @staticmethod
    def extract_inputs(df: pd.DataFrame) -> pd.DataFrame:
        """Explicit allowlist selection -- the only place a raw telemetry
        frame's columns are narrowed down to the frozen Stage-1 inputs."""
        X = df[list(STAGE1_INPUT_COLUMNS)].copy()
        _assert_stage1_input_frame(X)
        return X

    def fit(self, df: pd.DataFrame, label_col: str = "regime_label") -> "Stage1Router":
        X = self.extract_inputs(df)
        y = df[label_col].to_numpy()
        present_classes = sorted(set(y))
        if len(present_classes) < 2:
            raise ValueError(
                f"Stage-1 fit requires >=2 distinct classes in training data, got {present_classes}"
            )
        self.model.fit(X.to_numpy(dtype=float), y)
        self._fitted_classes = list(self.model.classes_)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._fitted_classes is None:
            raise RuntimeError("Stage1Router.predict called before fit")
        X = self.extract_inputs(df)
        return self.model.predict(X.to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# SS K -- Routing frequency / dwell / fallback FSM
# ---------------------------------------------------------------------------

@dataclass
class DwellDiagnostics:
    total_transitions: int
    switches_per_regime: Dict[str, int]
    switching_rate_per_1000_steps: float
    dwell_violation_count: int
    fallback_rate: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_transitions": self.total_transitions,
            "switches_per_regime": dict(self.switches_per_regime),
            "switching_rate_per_1000_steps": self.switching_rate_per_1000_steps,
            "dwell_violation_count": self.dwell_violation_count,
            "fallback_rate": self.fallback_rate,
        }


def apply_dwell_and_fallback(
    raw_regimes: Sequence[str], dwell_steps: int = DWELL_MINIMUM_STEPS
) -> Tuple[List[str], DwellDiagnostics]:
    """Apply the frozen dwell/fallback rule (design doc SS K) to a raw
    per-step sequence of Stage-1 outputs.

    Rule: once the effective regime changes to X, it is held at X for a
    minimum of `dwell_steps` raw steps before being allowed to change
    again -- EXCEPT a transition into NONE/OVERLAP from any active regime
    is never delayed (instant, safety-relevant fallback). The dwell timer
    applies to: active-regime -> different-active-regime transitions, and
    NONE/OVERLAP -> active-regime transitions.
    """
    if not raw_regimes:
        return [], DwellDiagnostics(0, {r: 0 for r in ACTIVE_REGIMES}, 0.0, 0, 0.0)
    for r in raw_regimes:
        if r not in REGIME_CLASSES:
            raise ValueError(f"unknown raw regime {r!r}; must be one of {REGIME_CLASSES}")

    effective = raw_regimes[0]
    steps_since_change = 0
    out: List[str] = [effective]
    transitions = 0
    switches_per_regime = {r: 0 for r in REGIME_CLASSES}

    for r in raw_regimes[1:]:
        if r == effective:
            steps_since_change += 1
            out.append(effective)
            continue
        if r in FALLBACK_REGIMES:
            # Instant, dwell-exempt transition into NONE/OVERLAP.
            effective = r
            steps_since_change = 0
            transitions += 1
            switches_per_regime[r] += 1
            out.append(effective)
        else:
            # r is an active regime (A/B/C): transition into it (from any
            # state, active or fallback) requires the dwell minimum to
            # have elapsed since the last change.
            if steps_since_change >= dwell_steps:
                effective = r
                steps_since_change = 0
                transitions += 1
                switches_per_regime[r] += 1
                out.append(effective)
            else:
                steps_since_change += 1
                out.append(effective)

    violations = count_dwell_violations(out, dwell_steps)
    n_steps = len(out)
    fallback_rate = sum(1 for x in out if x in FALLBACK_REGIMES) / n_steps
    diagnostics = DwellDiagnostics(
        total_transitions=transitions,
        switches_per_regime={r: switches_per_regime[r] for r in REGIME_CLASSES},
        switching_rate_per_1000_steps=1000.0 * transitions / n_steps,
        dwell_violation_count=violations,
        fallback_rate=fallback_rate,
    )
    return out, diagnostics


def count_dwell_violations(effective_regimes: Sequence[str], dwell_steps: int = DWELL_MINIMUM_STEPS) -> int:
    """Independent correctness check (design doc SS K: "should be exactly
    0 by construction") -- scans an already-produced effective-regime
    sequence and counts any active<->active or fallback->active
    transition that occurred fewer than `dwell_steps` raw steps after the
    prior change. Transitions INTO NONE/OVERLAP are never violations."""
    if len(effective_regimes) < 2:
        return 0
    violations = 0
    last_change_index = 0  # index 0 counts as the initial "change" (FSM starts steps_since_change=0 there)
    for i in range(1, len(effective_regimes)):
        if effective_regimes[i] == effective_regimes[i - 1]:
            continue
        gap = i - last_change_index
        if effective_regimes[i] not in FALLBACK_REGIMES and gap < dwell_steps:
            violations += 1
        last_change_index = i
    return violations


def route_action(regime: str) -> str:
    """Which Stage-2 selector (or fallback policy) a router output
    dispatches to -- design doc SS F. Returns a Stage-2 regime key
    (REGIME_A/B/C) or the literal fallback policy id."""
    if regime in ACTIVE_REGIMES:
        return regime
    if regime in FALLBACK_REGIMES:
        return FALLBACK_POLICY
    raise ValueError(f"unknown regime {regime!r}")


# ---------------------------------------------------------------------------
# SS J -- Deterministic group-aware split builder
# ---------------------------------------------------------------------------

TRAIN, VAL, TEST = "train", "val", "test"

FAMILY_C = "FAMILY_C_KV_PRESSURE_V2"
FAMILY_C_HELD_OUT_SEEDS: Tuple[str, ...] = ("20260914", "20260915")


def _hash_bucket(group_key: str) -> int:
    """sha256(group_key) mod 100 -- pure deterministic function, no RNG
    state (design doc SS J)."""
    digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


def build_splits(scenario_df: pd.DataFrame) -> Dict[str, str]:
    """Deterministic group-aware TRAIN/VAL/TEST split (design doc SS J).

    Returns a mapping `canonical_scenario_id -> split_name`. Requires
    `scenario_df` to carry `canonical_scenario_id`, `mechanism_family`,
    `group_key`, and (for Family C) `seed` columns -- the exact schema of
    `mf_psd_scenarios_v1.csv`.

    Family C: rows whose `seed` is in the native held-out-eval-seed
    designation (20260914/20260915) go to TEST, preserved as-is; all other
    Family C rows (the calibration seeds) go to TRAIN -- the design doc
    does not describe a further hash-based val/test carve-out for Family C
    beyond this pre-registered binary partition (only Family A/B get the
    60/20/20 group-hash split). Family A/B: group-aware 60/20/20 by
    `sha256(group_key) mod 100`, boundaries 0-60/60-80/80-100.

    NOTE on the design doc's own count: SS J's prose says "12/72 scenarios"
    held out for Family C: the actual seed-based designation
    (`held_out_eval_seed`, seeds 20260914/20260915) covers 24/72 scenarios
    on this frozen MF-PSD data (2 of 6 seeds x 12 groups) -- the seed list
    itself (20260914/20260915) is unambiguous and is what is implemented
    here verbatim; the "12" appears to describe the 12 Family-C groups
    each contributing 2 held-out scenarios, not a 12-scenario total. Not
    reinterpreted beyond using the literal, named seeds.
    """
    required = {"canonical_scenario_id", "mechanism_family", "group_key"}
    missing = required - set(scenario_df.columns)
    if missing:
        raise ValueError(f"scenario_df missing required columns: {missing}")

    out: Dict[str, str] = {}
    non_c = scenario_df[scenario_df["mechanism_family"] != FAMILY_C]
    for group_key, bucket in ((g, _hash_bucket(g)) for g in sorted(non_c["group_key"].unique())):
        split = TRAIN if bucket < 60 else (VAL if bucket < 80 else TEST)
        rows = non_c[non_c["group_key"] == group_key]
        for cid in rows["canonical_scenario_id"]:
            out[cid] = split

    fam_c = scenario_df[scenario_df["mechanism_family"] == FAMILY_C]
    if len(fam_c) > 0:
        if "seed" not in fam_c.columns:
            raise ValueError("Family C rows present but no 'seed' column to apply held_out_eval_seed")
        seed_str = fam_c["seed"].astype(str)
        held_out_mask = seed_str.isin(FAMILY_C_HELD_OUT_SEEDS)
        for cid, held in zip(fam_c["canonical_scenario_id"], held_out_mask):
            out[cid] = TEST if held else TRAIN

    return out


def assert_group_disjoint(scenario_df: pd.DataFrame, split_map: Dict[str, str]) -> None:
    """Test-plan item 6: no `group_key` appears in more than one split.

    Applies only to Family A/B's group-hash split (design doc SS J general
    rule). Family C is a documented, pre-registered EXCEPTION: its
    held-out-eval-seed rows are "preserved as-is and assigned entirely to
    TEST" independent of `group_key`, which structurally means a Family C
    group can (and, on the frozen MF-PSD data, does) contribute both TRAIN
    (calibration-seed) and TEST (held-out-seed) rows -- not a violation of
    the general rule, since the design doc explicitly carves Family C out
    of it rather than silently breaking it."""
    df = scenario_df.copy()
    df["split"] = df["canonical_scenario_id"].map(split_map)
    if df["split"].isna().any():
        missing = df[df["split"].isna()]["canonical_scenario_id"].tolist()
        raise AssertionError(f"scenarios missing a split assignment: {missing}")
    non_c = df[df["mechanism_family"] != FAMILY_C]
    by_group = non_c.groupby("group_key")["split"].nunique()
    bad = by_group[by_group > 1]
    if len(bad) > 0:
        raise AssertionError(f"group_keys spanning >1 split: {bad.index.tolist()}")


# ---------------------------------------------------------------------------
# SS P -- Blended-regime robustness microcases (instantiate-only, this task)
# ---------------------------------------------------------------------------

def _priority_heterogeneous_requests(seed: int, n_total_jobs: int = 40) -> Tuple[Request, ...]:
    """Family-A-style priority heterogeneity, reusing `case_fairness_vs_size_v2`
    verbatim for request generation (design doc SS P: "only online-observable
    conditions are manipulated... no scenario ID, seed, or family label is
    used to force the outcome"). `allow_synthetic_tokens=True` -- this
    workstation has no staged BurstGPT data (see the environment-limitation
    note in the audit that authorized this task); token *lengths* are not
    part of what SS P's interventions manipulate (priority/service-model/
    max_kv_tokens are), so synthetic-token fallback does not touch the
    controlled variable."""
    scenario = case_fairness_vs_size_v2(
        target_utilization=1.2,
        tenant_weight_skew=5.0,
        favored_tenant_size="long",
        prediction_noise_sigma=0.0,
        seed=seed,
        n_total_jobs=n_total_jobs,
        allow_synthetic_tokens=True,
    )
    return scenario.requests


def build_blended_microcase_a_plus_b(seed: int = 90101) -> PolicySeparationScenario:
    """A+B: Family-A priority heterogeneity + prefill-modeling /
    tight-step-token-budget contention (design doc SS P row 1)."""
    requests = _priority_heterogeneous_requests(seed)
    gpu = GPUConfig(gpu_id=0, max_active_sequences=16, max_batch_tokens=256, max_kv_tokens=200_000)
    return PolicySeparationScenario(
        scenario_id=f"blended.a_plus_b.s{seed}",
        family="blended_microcase_v1",
        template_name="hierarchical_router_v1_blended_a_plus_b",
        generator_version=SCHEMA_VERSION,
        seed=seed,
        params={"components": ["case_fairness_vs_size_v2", "case_prefill_decode_ttft_contention_v2"]},
        requests=requests,
        gpu_configs=(gpu,),
        service_model_kwargs={
            "enable_prefill_modeling": True,
            "enable_decode_prefill_contention": True,
            "step_token_budget": 256,
        },
        target_policy_family="hierarchical_router_v1",
        target_mechanism="blended_a_plus_b_robustness_microcase",
        expected_qualitative_hypothesis=(
            "a_active=True and b_active_v2=True should co-occur for a "
            "non-trivial fraction of steps (design doc SS P)."
        ),
        stress_control_relationship="stress",
        pair_id=f"blended.a_plus_b.s{seed}",
        changed_parameters=("priority_heterogeneity", "prefill_decode_contention"),
    )


def build_blended_microcase_a_plus_c(seed: int = 90102) -> PolicySeparationScenario:
    """A+C: Family-A priority heterogeneity + Family-C-scale tight
    `max_kv_tokens` (design doc SS P row 2)."""
    requests = _priority_heterogeneous_requests(seed)
    gpu = kv_scarce_gpu(max_kv_tokens=6_000, max_active_sequences=16, max_batch_tokens=64)
    return PolicySeparationScenario(
        scenario_id=f"blended.a_plus_c.s{seed}",
        family="blended_microcase_v1",
        template_name="hierarchical_router_v1_blended_a_plus_c",
        generator_version=SCHEMA_VERSION,
        seed=seed,
        params={"components": ["case_fairness_vs_size_v2", "tight_max_kv_tokens"]},
        requests=requests,
        gpu_configs=(gpu,),
        service_model_kwargs={},
        target_policy_family="hierarchical_router_v1",
        target_mechanism="blended_a_plus_c_robustness_microcase",
        expected_qualitative_hypothesis=(
            "a_active=True and c_active=True should co-occur for a "
            "non-trivial fraction of steps (design doc SS P)."
        ),
        stress_control_relationship="stress",
        pair_id=f"blended.a_plus_c.s{seed}",
        changed_parameters=("priority_heterogeneity", "kv_pressure"),
    )


def build_blended_microcase_b_plus_c(seed: int = 90103) -> PolicySeparationScenario:
    """B+C: prefill/decode contention config + tight `max_kv_tokens`
    (design doc SS P row 3)."""
    scenario_b = case_prefill_decode_ttft_contention(
        hog_count="high",
        late_pressure="high",
        slo_emphasis="hog_ttft",
        seed=seed,
        allow_synthetic_tokens=True,
    )
    gpu = kv_scarce_gpu(max_kv_tokens=6_000, max_active_sequences=16, max_batch_tokens=64)
    return PolicySeparationScenario(
        scenario_id=f"blended.b_plus_c.s{seed}",
        family="blended_microcase_v1",
        template_name="hierarchical_router_v1_blended_b_plus_c",
        generator_version=SCHEMA_VERSION,
        seed=seed,
        params={"components": ["case_prefill_decode_ttft_contention_v2", "tight_max_kv_tokens"]},
        requests=scenario_b.requests,
        gpu_configs=(gpu,),
        service_model_kwargs=dict(scenario_b.service_model_kwargs),
        target_policy_family="hierarchical_router_v1",
        target_mechanism="blended_b_plus_c_robustness_microcase",
        expected_qualitative_hypothesis=(
            "b_active_v2=True and c_active=True should co-occur for a "
            "non-trivial fraction of steps (design doc SS P)."
        ),
        stress_control_relationship="stress",
        pair_id=f"blended.b_plus_c.s{seed}",
        changed_parameters=("prefill_decode_contention", "kv_pressure"),
    )


def build_blended_microcase_a_plus_b_plus_c(seed: int = 90104) -> PolicySeparationScenario:
    """Optional A+B+C stress case: all three interventions combined
    (design doc SS P row 4)."""
    requests = _priority_heterogeneous_requests(seed)
    gpu = kv_scarce_gpu(max_kv_tokens=6_000, max_active_sequences=16, max_batch_tokens=64)
    return PolicySeparationScenario(
        scenario_id=f"blended.a_plus_b_plus_c.s{seed}",
        family="blended_microcase_v1",
        template_name="hierarchical_router_v1_blended_a_plus_b_plus_c",
        generator_version=SCHEMA_VERSION,
        seed=seed,
        params={"components": ["case_fairness_vs_size_v2", "prefill_modeling", "tight_max_kv_tokens"]},
        requests=requests,
        gpu_configs=(gpu,),
        service_model_kwargs={
            "enable_prefill_modeling": True,
            "enable_decode_prefill_contention": True,
            "step_token_budget": 256,
        },
        target_policy_family="hierarchical_router_v1",
        target_mechanism="blended_a_plus_b_plus_c_robustness_microcase",
        expected_qualitative_hypothesis=(
            "Potentially all three labels true at overlapping times; "
            "specifically checks the OVERLAP routing path is exercised "
            "at least once (design doc SS P)."
        ),
        stress_control_relationship="stress",
        pair_id=f"blended.a_plus_b_plus_c.s{seed}",
        changed_parameters=("priority_heterogeneity", "prefill_decode_contention", "kv_pressure"),
    )


BLENDED_MICROCASE_BUILDERS = {
    "A_plus_B": build_blended_microcase_a_plus_b,
    "A_plus_C": build_blended_microcase_a_plus_c,
    "B_plus_C": build_blended_microcase_b_plus_c,
    "A_plus_B_plus_C_optional": build_blended_microcase_a_plus_b_plus_c,
}
