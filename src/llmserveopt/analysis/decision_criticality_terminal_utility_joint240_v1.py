"""Terminal utility robustness on joint-240: replay frozen ANWG forks with traces.

Implements docs/design/DECISION_CRITICALITY_TERMINAL_UTILITY_JOINT240_V1.md.
"""
from __future__ import annotations

import copy
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..core.action import Action
from ..core.types import CompletedRequest, ObservableState, Request
from ..policies.base import BasePolicy
from ..policy_separation.hierarchical_regime_router_v1 import DWELL_MINIMUM_STEPS
from ..policy_separation.schema import PolicySeparationScenario
from ..policy_separation.unified_utility_matrix import _build_policy
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig

from . import decision_criticality_timescale_trainval_v1 as dcm
from .decision_criticality_terminal_anwg_joint240_v1 import (
    ANWG_EQ_ATOL,
    TracingAliveRouter,
    build_p6_shadow_policies,
    clone_alive_router,
    fit_oof_alive_stage1_models,
    load_frozen_joint240_context,
    run_one_step_then_alive_terminal,
    select_alt_action,
)
from .joint240_same_distribution_adaptive_v1 import (
    P6,
    LiveP6DwellRouterPolicy,
)

ROOT = Path(__file__).resolve().parents[3]
PARENT_EXP = ROOT / "experiments" / "decision_criticality_terminal_anwg_joint240_v1"
SCHEMA_VERSION = "decision_criticality_terminal_utility_joint240_v1.0.0"
BOOTSTRAP_SEED = 20260825
N_BOOTSTRAP = 2000
EPS_SLO = 1e-12
MEANINGFUL_EPS = 1e-9
PRACTICAL = 0.001


def request_weight(req: Request) -> float:
    return float(req.priority) if float(req.priority) > 0 else 1.0


def _completion_time_for_row(
    *,
    completed_map: Dict[int, CompletedRequest],
    dropped_ids: set,
    req: Request,
    sim_duration: float,
) -> Tuple[float, bool, bool]:
    """Return (C_i, completed, dropped) with unfinished -> C_i = sim_duration."""
    rid = int(req.request_id)
    if rid in completed_map:
        return float(completed_map[rid].completion_time), True, False
    if rid in dropped_ids:
        return float(sim_duration), False, True
    return float(sim_duration), False, False


def extract_request_rows(
    *,
    all_requests: Sequence[Request],
    completed: Sequence[CompletedRequest],
    dropped: Sequence[Request],
    sim_duration: float,
    branch_meta: Dict[str, Any],
    branch_role: str,
) -> List[Dict[str, Any]]:
    completed_map = {int(c.request.request_id): c for c in completed}
    dropped_ids = {int(r.request_id) for r in dropped}
    rows = []
    for req in all_requests:
        C, done, drop = _completion_time_for_row(
            completed_map=completed_map,
            dropped_ids=dropped_ids,
            req=req,
            sim_duration=sim_duration,
        )
        A = float(req.arrival_time)
        D = float(req.slo_deadline)
        w = request_weight(req)
        T = max(0.0, C - D)
        S = max(D - A, EPS_SLO)
        rows.append(
            {
                **branch_meta,
                "branch_role": branch_role,
                "request_id": int(req.request_id),
                "weight": w,
                "arrival_time": A,
                "deadline": D,
                "completion_time": C,
                "completed": bool(done),
                "dropped": bool(drop),
                "lateness": float(C - D),
                "tardiness": float(T),
                "slo_window": float(S),
                "class_id": str(getattr(req, "class_id", "")),
            }
        )
    return rows


def metrics_from_request_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "anwg": float("nan"),
            "wcg": float("nan"),
            "wmt": float("nan"),
            "wnt": float("nan"),
            "soft": float("nan"),
            "W": 0.0,
            "n_completed": 0,
            "n_dropped": 0,
        }
    W = float(sum(r["weight"] for r in rows))
    if W <= 0:
        W = float(len(rows))
    anwg = sum(
        r["weight"] * (1.0 if r["completed"] and r["completion_time"] <= r["deadline"] else 0.0)
        for r in rows
    ) / W
    wcg = sum(r["weight"] * (1.0 if r["completed"] else 0.0) for r in rows) / W
    wmt = sum(r["weight"] * r["tardiness"] for r in rows) / W
    wnt = sum(r["weight"] * (r["tardiness"] / r["slo_window"]) for r in rows) / W
    soft = 0.0
    for r in rows:
        if r["completed"]:
            soft += r["weight"] * float(np.exp(-r["tardiness"] / r["slo_window"]))
        # unfinished/dropped: credit 0
    soft /= W
    return {
        "anwg": float(anwg),
        "wcg": float(wcg),
        "wmt": float(wmt),
        "wnt": float(wnt),
        "soft": float(soft),
        "W": float(W),
        "n_completed": int(sum(1 for r in rows if r["completed"])),
        "n_dropped": int(sum(1 for r in rows if r["dropped"])),
    }


def collect_terminal_requests(shell: Simulator, all_requests: Sequence[Request]) -> Tuple[List[CompletedRequest], List[Request]]:
    completed = list(shell._completed)
    dropped = (
        [ir.request for ir in shell._waiting]
        + [ir.request for ir in shell._migrating]
        + [ir.request for ir in shell._relocating.values()]
    )
    return completed, dropped


def load_frozen_parent_branches(path: Optional[Path] = None) -> pd.DataFrame:
    p = path or (PARENT_EXP / "branches.csv")
    df = pd.read_csv(p)
    need = [
        "scenario_id",
        "fold",
        "step",
        "acquisition_type",
        "chosen_policy_id",
        "alt_policy_id",
        "reference_anwg",
        "cf_anwg",
        "delta_anwg",
        "seed",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(missing)
    df["branch_id"] = [
        f"{r.scenario_id}::step{int(r.step)}::{r.acquisition_type}::{r.alt_policy_id}"
        for r in df.itertuples()
    ]
    if df["branch_id"].duplicated().any():
        raise RuntimeError("duplicate branch_id in parent branches")
    return df


@dataclass
class FrozenUtilityObserver(BasePolicy):
    """Replay only frozen intervention steps; dump CF (+ optional REF-replay) traces."""

    name = "decision_criticality_terminal_utility_joint240_observer_v1"

    sim_ref: Simulator
    tracing_router: TracingAliveRouter
    shadow_policies: Dict[str, BasePolicy]
    all_requests: Sequence[Request]
    scenario_id: str
    fold: int
    seed: int
    frozen_by_step: Dict[int, Dict[str, Any]]
    write_trace: Any = None
    anwg_atol: float = ANWG_EQ_ATOL

    branch_rows: List[Dict[str, Any]] = field(default_factory=list)
    n_matched_steps: int = 0
    ref_replay_done: bool = False
    max_anwg_mismatch: float = 0.0

    def reset(self) -> None:
        self.tracing_router.reset()
        for p in self.shadow_policies.values():
            if hasattr(p, "reset"):
                p.reset()
        self.branch_rows = []
        self.n_matched_steps = 0
        self.ref_replay_done = False
        self.max_anwg_mismatch = 0.0

    def select_action(self, state: ObservableState) -> Action:
        real_action = self.tracing_router.select_action(state)
        step = int(state.step)
        if step not in self.frozen_by_step:
            return real_action

        frozen = self.frozen_by_step[step]
        effective = self.tracing_router.last_effective_policy
        alt_id, alt_action, disagree, _ = select_alt_action(
            state=state,
            ref_action=real_action,
            effective_policy=effective or str(frozen["chosen_policy_id"]),
            shadow_policies=self.shadow_policies,
        )
        # Prefer frozen alt policy action for exact reconstruction
        want_alt = str(frozen["alt_policy_id"])
        if want_alt in self.shadow_policies:
            shadow_state = copy.deepcopy(state)
            alt_action = self.shadow_policies[want_alt].select_action(shadow_state)
            alt_id = want_alt

        meta = {
            "branch_id": str(frozen["branch_id"]),
            "scenario_id": self.scenario_id,
            "fold": int(self.fold),
            "step": step,
            "seed": int(self.seed),
            "acquisition_type": str(frozen["acquisition_type"]),
            "chosen_policy_id": str(effective),
            "alt_policy_id": alt_id,
            "parent_chosen_policy_id": str(frozen["chosen_policy_id"]),
            "parent_alt_policy_id": str(frozen["alt_policy_id"]),
            "parent_reference_anwg": float(frozen["reference_anwg"]),
            "parent_cf_anwg": float(frozen["cf_anwg"]),
            "parent_delta_anwg": float(frozen["delta_anwg"]),
            "policy_match": bool(str(effective) == str(frozen["chosen_policy_id"])),
            "alt_match": bool(alt_id == str(frozen["alt_policy_id"])),
        }

        cont_cf = clone_alive_router(self.tracing_router.inner)
        # Run CF with continue_run path identical to parent helper, but capture shell
        fp_before = dcm._state_fingerprint(self.sim_ref)
        fork = dcm.fork_from_live_simulator(
            self.sim_ref,
            policy=cont_cf,
            policy_id="live_p6_dwell_router_continuation",
            first_action=copy.deepcopy(alt_action),
        )
        cont_cf.name = "cf_one_step_alt"  # type: ignore[attr-defined]
        metrics = fork.shell.continue_run(
            cont_cf,
            workload_tag=self.scenario_id,
            seed=self.seed,
            num_total=len(self.all_requests),
            all_requests=self.all_requests,
        )
        assert dcm._state_fingerprint(self.sim_ref) == fp_before
        cf_anwg = float(metrics.arrival_normalized_weighted_goodput)
        completed, dropped = collect_terminal_requests(fork.shell, self.all_requests)
        cf_rows = extract_request_rows(
            all_requests=self.all_requests,
            completed=completed,
            dropped=dropped,
            sim_duration=float(metrics.sim_duration),
            branch_meta=meta,
            branch_role="CF",
        )
        cf_m = metrics_from_request_rows(cf_rows)
        # Prefer metrics-module ANWG for integrity vs parent; also check trace ANWG
        anwg_mismatch = abs(cf_anwg - float(frozen["cf_anwg"]))
        self.max_anwg_mismatch = max(self.max_anwg_mismatch, anwg_mismatch)

        if self.write_trace is not None:
            self.write_trace(cf_rows)

        # One REF-replay per scenario for integrity + REF trace from fork (optional);
        # scenario-level REF dump comes from untouched run after select loop.
        ref_replay_anwg = None
        if not self.ref_replay_done:
            cont_rr = clone_alive_router(self.tracing_router.inner)
            fp_b = dcm._state_fingerprint(self.sim_ref)
            fork_r = dcm.fork_from_live_simulator(
                self.sim_ref,
                policy=cont_rr,
                policy_id="live_p6_dwell_router_continuation",
                first_action=copy.deepcopy(real_action),
            )
            cont_rr.name = "ref_action_replay"  # type: ignore[attr-defined]
            m_rr = fork_r.shell.continue_run(
                cont_rr,
                workload_tag=self.scenario_id,
                seed=self.seed,
                num_total=len(self.all_requests),
                all_requests=self.all_requests,
            )
            assert dcm._state_fingerprint(self.sim_ref) == fp_b
            ref_replay_anwg = float(m_rr.arrival_normalized_weighted_goodput)
            self.ref_replay_done = True

        row = {
            **meta,
            "cf_anwg_live": cf_anwg,
            "cf_anwg_from_traces": cf_m["anwg"],
            "cf_wcg": cf_m["wcg"],
            "cf_wmt": cf_m["wmt"],
            "cf_wnt": cf_m["wnt"],
            "cf_soft": cf_m["soft"],
            "cf_num_completed": int(metrics.num_completed),
            "cf_num_dropped": int(metrics.num_dropped),
            "cf_sim_duration": float(metrics.sim_duration),
            "cf_anwg_minus_parent": float(cf_anwg - float(frozen["cf_anwg"])),
            "ref_replay_anwg": ref_replay_anwg,
            "disagree_runtime": bool(disagree),
        }
        self.branch_rows.append(row)
        self.n_matched_steps += 1
        return real_action


def run_scenario_utility_replay(
    scenario: PolicySeparationScenario,
    *,
    stage1: Pipeline,
    fold: int,
    seed: int,
    frozen_rows: pd.DataFrame,
    write_trace=None,
) -> Dict[str, Any]:
    sid = scenario.scenario_id
    frozen_by_step: Dict[int, Dict[str, Any]] = {}
    for _, r in frozen_rows.iterrows():
        frozen_by_step[int(r["step"])] = dict(r)

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
    observer = FrozenUtilityObserver(
        sim_ref=sim,
        tracing_router=tracing,
        shadow_policies=build_p6_shadow_policies(),
        all_requests=list(scenario.requests),
        scenario_id=sid,
        fold=fold,
        seed=seed,
        frozen_by_step=frozen_by_step,
        write_trace=write_trace,
    )
    observer.sim_ref = sim
    ref_metrics = sim.run(observer, workload_tag=sid, seed=seed)
    ref_anwg = float(ref_metrics.arrival_normalized_weighted_goodput)
    completed, dropped = collect_terminal_requests(sim, list(scenario.requests))
    ref_rows = extract_request_rows(
        all_requests=list(scenario.requests),
        completed=completed,
        dropped=dropped,
        sim_duration=float(ref_metrics.sim_duration),
        branch_meta={
            "branch_id": f"{sid}::REFERENCE",
            "scenario_id": sid,
            "fold": int(fold),
            "step": -1,
            "seed": int(seed),
            "acquisition_type": "REFERENCE",
            "chosen_policy_id": "",
            "alt_policy_id": "",
        },
        branch_role="REF",
    )
    ref_m = metrics_from_request_rows(ref_rows)
    if write_trace is not None:
        write_trace(ref_rows)

    out_branches = []
    for br in observer.branch_rows:
        br["reference_anwg_live"] = ref_anwg
        br["reference_anwg_from_traces"] = ref_m["anwg"]
        br["reference_wcg"] = ref_m["wcg"]
        br["reference_wmt"] = ref_m["wmt"]
        br["reference_wnt"] = ref_m["wnt"]
        br["reference_soft"] = ref_m["soft"]
        br["delta_anwg_live"] = float(br["cf_anwg_live"] - ref_anwg)
        br["delta_anwg_from_traces"] = float(br["cf_anwg_from_traces"] - ref_m["anwg"])
        br["delta_wcg"] = float(br["cf_wcg"] - ref_m["wcg"])
        br["delta_wmt_improvement"] = float(ref_m["wmt"] - br["cf_wmt"])
        br["delta_wnt_improvement"] = float(ref_m["wnt"] - br["cf_wnt"])
        br["delta_soft"] = float(br["cf_soft"] - ref_m["soft"])
        br["abs_delta_anwg_live"] = abs(br["delta_anwg_live"])
        br["parent_delta_anwg_abs_err"] = abs(br["delta_anwg_live"] - float(br["parent_delta_anwg"]))
        br["parent_cf_anwg_abs_err"] = abs(br["cf_anwg_live"] - float(br["parent_cf_anwg"]))
        br["parent_ref_anwg_abs_err"] = abs(ref_anwg - float(br["parent_reference_anwg"]))
        if br.get("ref_replay_anwg") is not None:
            br["ref_replay_minus_reference"] = float(br["ref_replay_anwg"] - ref_anwg)
            br["ref_replay_matches"] = bool(abs(br["ref_replay_minus_reference"]) <= ANWG_EQ_ATOL)
        out_branches.append(br)

    return {
        "scenario_id": sid,
        "fold": int(fold),
        "seed": int(seed),
        "reference_anwg_live": ref_anwg,
        "reference_metrics": ref_m,
        "n_frozen_expected": int(len(frozen_rows)),
        "n_matched_steps": int(observer.n_matched_steps),
        "max_cf_anwg_mismatch_vs_parent": float(observer.max_anwg_mismatch),
        "branch_rows": out_branches,
    }


def scenario_top_k_share_mult(mass: np.ndarray, k: int) -> float:
    if len(mass) == 0:
        return 0.0
    total = float(np.sum(mass))
    if total <= 0:
        return 0.0
    k = max(1, min(int(k), len(mass)))
    return float(np.sort(mass)[::-1][:k].sum() / total)


def concentration_curve(vals: np.ndarray, fracs=(0.01, 0.05, 0.10)) -> dict:
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return {str(f): {"k": 0, "share": None} for f in fracs}
    order = np.argsort(-v)
    sorted_v = v[order]
    total = float(sorted_v.sum())
    cum = np.cumsum(sorted_v)
    out = {}
    for f in fracs:
        k = max(1, int(np.ceil(f * len(sorted_v))))
        out[str(f)] = {"k": k, "share": float(cum[k - 1] / total) if total > 0 else 0.0}
    return out


def bootstrap_scenario_stats(
    branches: pd.DataFrame,
    *,
    effect_col: str,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    meaningful_eps: float = MEANINGFUL_EPS,
) -> Dict[str, Any]:
    """Scenario-grouped bootstrap with multiplicity retained for scenario mass."""
    abs_col = f"abs__{effect_col}"
    df = branches.copy()
    df[abs_col] = df[effect_col].abs()
    scen_ids = df["scenario_id"].unique()
    by_scen = {sid: g for sid, g in df.groupby("scenario_id")}
    scen_mass = {sid: float(g[abs_col].sum()) for sid, g in by_scen.items()}
    rng = np.random.default_rng(seed)

    def _ci(xs):
        a = np.asarray(xs, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return {"mean": None, "ci95_low": None, "ci95_high": None}
        return {
            "mean": float(np.mean(a)),
            "ci95_low": float(np.quantile(a, 0.025)),
            "ci95_high": float(np.quantile(a, 0.975)),
        }

    b_prev, b_mean, b_t1, b_t5, b_t10, b_top5, b_top10sc = [], [], [], [], [], [], []
    for _ in range(n_boot):
        draw = scen_ids[rng.integers(0, len(scen_ids), size=len(scen_ids))]
        sample = pd.concat([by_scen[sid] for sid in draw], ignore_index=True)
        a = sample[abs_col].to_numpy(float)
        b_prev.append(float((a > meaningful_eps).mean()))
        b_mean.append(float(a.mean()))
        conc = concentration_curve(a)
        b_t1.append(float(conc["0.01"]["share"]))
        b_t5.append(float(conc["0.05"]["share"]))
        b_t10.append(float(conc["0.1"]["share"]))
        mass = np.asarray([scen_mass[sid] for sid in draw], dtype=float)
        b_top5.append(scenario_top_k_share_mult(mass, 5))
        b_top10sc.append(float(concentration_curve(mass, fracs=(0.10,))["0.1"]["share"]))

    return {
        "meaningful_prevalence": _ci(b_prev),
        "mean_abs": _ci(b_mean),
        "top1pct_state_mass": _ci(b_t1),
        "top5pct_state_mass": _ci(b_t5),
        "top10pct_state_mass": _ci(b_t10),
        "top5_scenario_mass": _ci(b_top5),
        "top10pct_scenario_mass": _ci(b_top10sc),
        "multiplicity_retained": True,
    }
