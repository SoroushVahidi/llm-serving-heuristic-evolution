"""SBS-continuation robustness for joint-240 terminal criticality.

Reuses exact acquisition keys from
`experiments/decision_criticality_terminal_anwg_joint240_v1/branches.csv`.
Does not overwrite the parent experiment.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..core.action import Action
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..policy_separation.hierarchical_regime_router_v1 import DWELL_MINIMUM_STEPS
from ..policy_separation.schema import PolicySeparationScenario
from ..policy_separation.unified_utility_matrix import _build_policy
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig

from . import decision_criticality_timescale_trainval_v1 as dcm
from .decision_criticality_terminal_anwg_joint240_v1 import (
    ANWG_EQ_ATOL,
    P6,
    TracingAliveRouter,
    build_p6_shadow_policies,
    fit_oof_alive_stage1_models,
    load_frozen_joint240_context,
    select_alt_action,
)
from .joint240_same_distribution_adaptive_v1 import LiveP6DwellRouterPolicy

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / "experiments" / "decision_criticality_terminal_anwg_joint240_v1"
DESIGN_DOC = ROOT / "docs" / "design" / "JOINT240_TERMINAL_CRITICALITY_SBS_CONTINUATION_V1.md"

SCHEMA_VERSION = "joint240_terminal_criticality_sbs_continuation_v1.0.0"
SBS_POLICY = "kv_constrained_online"
BOOTSTRAP_SEED = 20260827
N_BOOTSTRAP = 10_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_parent_acquisition_keys() -> pd.DataFrame:
    b = pd.read_csv(PARENT / "branches.csv")
    keys = b[
        [
            "scenario_id",
            "step",
            "acquisition_type",
            "alt_policy_id",
            "delta_anwg",
            "abs_delta_anwg",
            "canonical_ref_action",
            "canonical_alt_action",
            "chosen_policy_id",
            "fold",
            "n_elevated_mechanisms",
            "seed",
        ]
    ].copy()
    keys["alive_delta_anwg"] = keys["delta_anwg"].astype(float)
    keys["alive_abs_delta_anwg"] = keys["abs_delta_anwg"].astype(float)
    keys["alive_nonzero"] = keys["alive_abs_delta_anwg"] > ANWG_EQ_ATOL
    return keys


def run_one_step_then_fixed_policy(
    sim: Simulator,
    *,
    first_action: Action,
    continuation: BasePolicy,
    all_requests: Sequence,
    workload_tag: str,
    seed: int,
    branch_label: str,
) -> Dict[str, Any]:
    fp_before = dcm._state_fingerprint(sim)
    step_before = int(sim._step)
    cont = copy.deepcopy(continuation)
    cont.name = branch_label  # type: ignore[attr-defined]
    fork = dcm.fork_from_live_simulator(
        sim,
        policy=cont,
        policy_id=SBS_POLICY,
        first_action=copy.deepcopy(first_action),
    )
    metrics = fork.shell.continue_run(
        cont,
        workload_tag=workload_tag,
        seed=seed,
        num_total=len(all_requests),
        all_requests=all_requests,
    )
    fp_after = dcm._state_fingerprint(sim)
    return {
        "anwg": float(metrics.arrival_normalized_weighted_goodput),
        "num_completed": int(metrics.num_completed),
        "sim_duration": float(metrics.sim_duration),
        "extra_steps": int(fork.shell._step - step_before),
        "live_fingerprint_unchanged": fp_before == fp_after,
    }


@dataclass
class SBSContinuationObserver(BasePolicy):
    """Replay Alive; at parent acquisition keys, evaluate REF-SBS vs CF-SBS."""

    name = "joint240_sbs_continuation_observer_v1"
    sim_ref: Simulator
    tracing_router: TracingAliveRouter
    shadow_policies: Dict[str, BasePolicy]
    sbs_policy: BasePolicy
    all_requests: Sequence
    scenario_id: str
    seed: int
    # set of (step, acquisition_type, alt_policy_id)
    targets: Set[Tuple[int, str, str]]
    parent_by_key: Dict[Tuple[int, str, str], Dict[str, Any]]

    branch_rows: List[Dict[str, Any]] = field(default_factory=list)
    hit_keys: Set[Tuple[int, str, str]] = field(default_factory=set)

    def reset(self) -> None:
        self.tracing_router.reset()
        for p in self.shadow_policies.values():
            if hasattr(p, "reset"):
                p.reset()
        if hasattr(self.sbs_policy, "reset"):
            self.sbs_policy.reset()
        self.branch_rows = []
        self.hit_keys = set()

    def select_action(self, state: ObservableState) -> Action:
        real_action = self.tracing_router.select_action(state)
        if len(state.waiting_queue) == 0:
            return real_action

        effective = self.tracing_router.last_effective_policy or "weighted_fair_share"
        alt_id, alt_action, disagree, _ = select_alt_action(
            state=state,
            ref_action=real_action,
            effective_policy=effective,
            shadow_policies=self.shadow_policies,
        )
        acq = "DISAGREEMENT" if disagree else "AGREEMENT_CONTROL"
        key = (int(state.step), acq, str(alt_id))
        if key not in self.targets or key in self.hit_keys:
            return real_action

        # Evaluate paired SBS continuation contrast
        ref_sbs = run_one_step_then_fixed_policy(
            self.sim_ref,
            first_action=real_action,
            continuation=self.sbs_policy,
            all_requests=self.all_requests,
            workload_tag=self.scenario_id,
            seed=self.seed,
            branch_label="ref_sbs_cont",
        )
        cf_sbs = run_one_step_then_fixed_policy(
            self.sim_ref,
            first_action=alt_action,
            continuation=self.sbs_policy,
            all_requests=self.all_requests,
            workload_tag=self.scenario_id,
            seed=self.seed,
            branch_label="cf_sbs_cont",
        )
        parent = self.parent_by_key[key]
        delta = float(cf_sbs["anwg"] - ref_sbs["anwg"])
        row = {
            "scenario_id": self.scenario_id,
            "step": int(state.step),
            "acquisition_type": acq,
            "alt_policy_id": alt_id,
            "chosen_policy_id": effective,
            "canonical_ref_action": str(dcm.canonical_action(real_action)),
            "canonical_alt_action": str(dcm.canonical_action(alt_action)),
            "sbs_ref_anwg": ref_sbs["anwg"],
            "sbs_cf_anwg": cf_sbs["anwg"],
            "sbs_delta_anwg": delta,
            "sbs_abs_delta_anwg": abs(delta),
            "sbs_nonzero": abs(delta) > ANWG_EQ_ATOL,
            "alive_delta_anwg": float(parent["alive_delta_anwg"]),
            "alive_abs_delta_anwg": float(parent["alive_abs_delta_anwg"]),
            "alive_nonzero": bool(parent["alive_nonzero"]),
            "fold": int(parent["fold"]),
            "n_elevated_mechanisms": int(parent["n_elevated_mechanisms"]),
            "live_fingerprint_unchanged": bool(
                ref_sbs["live_fingerprint_unchanged"] and cf_sbs["live_fingerprint_unchanged"]
            ),
            "cf_extra_steps": int(cf_sbs["extra_steps"]),
            "ref_extra_steps": int(ref_sbs["extra_steps"]),
        }
        self.branch_rows.append(row)
        self.hit_keys.add(key)
        return real_action


def run_scenario_sbs_continuation(
    scenario: PolicySeparationScenario,
    *,
    stage1: Pipeline,
    seed: int,
    parent_rows: pd.DataFrame,
) -> Dict[str, Any]:
    sid = scenario.scenario_id
    targets: Set[Tuple[int, str, str]] = set()
    parent_by_key: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for r in parent_rows.itertuples(index=False):
        key = (int(r.step), str(r.acquisition_type), str(r.alt_policy_id))
        targets.add(key)
        parent_by_key[key] = {
            "alive_delta_anwg": float(r.alive_delta_anwg),
            "alive_abs_delta_anwg": float(r.alive_abs_delta_anwg),
            "alive_nonzero": bool(r.alive_nonzero),
            "fold": int(r.fold),
            "n_elevated_mechanisms": int(r.n_elevated_mechanisms),
        }

    alive = LiveP6DwellRouterPolicy(stage1, P6, dwell_steps=DWELL_MINIMUM_STEPS)
    tracing = TracingAliveRouter(inner=alive)
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scenario.requests))
    observer = SBSContinuationObserver(
        sim_ref=sim,
        tracing_router=tracing,
        shadow_policies=build_p6_shadow_policies(),
        sbs_policy=_build_policy(SBS_POLICY)[0],
        all_requests=list(scenario.requests),
        scenario_id=sid,
        seed=seed,
        targets=targets,
        parent_by_key=parent_by_key,
    )
    observer.sim_ref = sim
    sim.run(observer, workload_tag=sid, seed=seed)
    missing = targets - observer.hit_keys
    return {
        "scenario_id": sid,
        "n_targets": len(targets),
        "n_hit": len(observer.hit_keys),
        "n_missing": len(missing),
        "missing_keys": sorted(list(missing))[:20],
        "branch_rows": observer.branch_rows,
    }


def paired_bootstrap_mean(
    vals: np.ndarray, *, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
) -> Dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    boots = [float(vals[rng.integers(0, n, size=n)].mean()) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(vals.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def scenario_clustered_bootstrap_prevalence(
    df: pd.DataFrame,
    col: str,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    by = {sid: g[col].to_numpy(dtype=bool) for sid, g in df.groupby("scenario_id")}
    sids = list(by.keys())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(sids), size=len(sids))
        parts = [by[sids[i]] for i in draw]
        sample = np.concatenate(parts) if parts else np.asarray([], dtype=bool)
        boots.append(float(sample.mean()) if len(sample) else float("nan"))
    arr = np.asarray(boots, dtype=float)
    point = float(df[col].mean())
    return {
        "mean": point,
        "ci95_low": float(np.nanquantile(arr, 0.025)),
        "ci95_high": float(np.nanquantile(arr, 0.975)),
    }


def concentration_share(abs_vals: np.ndarray, frac: float) -> float:
    vals = np.asarray(abs_vals, dtype=float)
    if len(vals) == 0:
        return float("nan")
    total = float(vals.sum())
    if total <= 0:
        return 0.0
    k = max(1, int(np.ceil(frac * len(vals))))
    return float(np.sort(vals)[::-1][:k].sum() / total)


def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return float(len(a & b) / len(u))


def classify_continuation(summary: Dict[str, Any]) -> str:
    p_nz = float(summary["sbs_nonzero_prevalence"]["mean"])
    m10 = float(summary["sbs_top10pct_mass"])
    spear = summary.get("spearman_abs_alive_vs_sbs")
    spear = float(spear) if spear is not None and np.isfinite(spear) else 0.0
    jac5 = float(summary.get("jaccard_top5pct") or 0.0)
    if p_nz < 0.15 and m10 >= 0.50 and spear >= 0.3:
        return "ROBUST_SPARSE_CONCENTRATION"
    if jac5 < 0.2 or spear < 0.1:
        return "CONTINUATION_SENSITIVE"
    return "MIXED_CONTINUATION"
