"""Terminal-ANWG one-step counterfactual criticality on joint-240 v1.

Implements docs/design/DECISION_CRITICALITY_TERMINAL_ANWG_JOINT240_V1.md.

Continuation policy = OOF Alive (`LiveP6DwellRouterPolicy`) from the frozen
Section 4.2 joint-240 experiment. Estimand is
continuation-policy-conditional one-step terminal ΔANWG — not a Q-value.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..core.action import Action
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..policy_separation.hierarchical_regime_router_v1 import DWELL_MINIMUM_STEPS
from ..policy_separation.online_regime_signals_v1 import compute_regime_signals
from ..policy_separation.schema import PolicySeparationScenario
from ..policy_separation.unified_utility_matrix import _build_policy
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig

from . import decision_criticality_timescale_trainval_v1 as dcm
from .joint240_same_distribution_adaptive_v1 import (
    P6,
    PROBE_POLICY,
    SPLIT_SEED,
    LiveP6DwellRouterPolicy,
    PolicyDwellFSM,
    collect_probe_telemetry,
    fit_live_stage1,
    freeze_oof_folds,
    load_utility_matrix,
    rebuild_all_scenarios,
)

ROOT = Path(__file__).resolve().parents[3]
JOINT240_EXP = ROOT / "experiments" / "joint240_same_distribution_adaptive_exploitability_v1"
JOINT_DIR = ROOT / "experiments" / "joint_multimechanism_generalization_v1"

SCHEMA_VERSION = "decision_criticality_terminal_anwg_joint240_v1.0.0"

MAX_DISAGREEMENT_PER_SCENARIO = 10
MAX_AGREEMENT_CONTROL_PER_SCENARIO = 5
CONTROL_SEED = 20260825
BOOTSTRAP_SEED = 20260825
N_BOOTSTRAP = 2000
ANWG_EQ_ATOL = 1e-12
PRACTICAL_THRESHOLDS = (0.001, 0.005, 0.01)

PRESSURE_FLAGS = (
    "high_fairness_pressure",
    "high_service_heterogeneity",
    "high_prefill_decode_pressure",
    "high_kv_pressure",
    "high_urgency_pressure",
    "high_burst_pressure",
)


def build_p6_shadow_policies() -> Dict[str, BasePolicy]:
    """Fresh P6 policy instances (Alive semantics: policy only, no SM override)."""
    return {pid: _build_policy(pid)[0] for pid in P6}


def clone_alive_router(router: LiveP6DwellRouterPolicy) -> LiveP6DwellRouterPolicy:
    """Clone Alive continuation state after step-t select_action.

    Shares frozen Stage-1 pipeline (read-only at inference). Deep-copies FSM,
    native policy instances, and counters so CF/REF-replay do not mutate the
    live reference router.
    """
    cloned = LiveP6DwellRouterPolicy(
        stage1=router.stage1,
        policy_ids=list(router.policy_ids),
        dwell_steps=int(router.fsm.dwell_steps),
    )
    cloned.fsm = copy.deepcopy(router.fsm)
    cloned._policies = copy.deepcopy(router._policies)
    cloned.switch_count = int(router.switch_count)
    cloned._last_policy = router._last_policy
    cloned.selected_policies = list(router.selected_policies)
    return cloned


def run_one_step_then_alive_terminal(
    sim: Simulator,
    *,
    first_action: Action,
    continuation_router: LiveP6DwellRouterPolicy,
    all_requests: Sequence,
    workload_tag: str,
    seed: int,
    branch_label: str,
) -> Dict[str, Any]:
    """Fork sim, force first_action, continue with Alive clone to terminal ANWG."""
    fp_before = dcm._state_fingerprint(sim)
    step_before = int(sim._step)
    fork = dcm.fork_from_live_simulator(
        sim,
        policy=continuation_router,
        policy_id="live_p6_dwell_router_continuation",
        first_action=copy.deepcopy(first_action),
    )
    continuation_router.name = branch_label  # type: ignore[attr-defined]
    metrics = fork.shell.continue_run(
        continuation_router,
        workload_tag=workload_tag,
        seed=seed,
        num_total=len(all_requests),
        all_requests=all_requests,
    )
    fp_after = dcm._state_fingerprint(sim)
    return {
        "anwg": float(metrics.arrival_normalized_weighted_goodput),
        "num_completed": int(metrics.num_completed),
        "num_dropped": int(metrics.num_dropped),
        "sim_duration": float(metrics.sim_duration),
        "extra_steps": int(fork.shell._step - step_before),
        "finished": True,
        "live_fingerprint_unchanged": fp_before == fp_after,
        "pre_fork_fingerprint": fp_before,
    }


def _state_features(state: ObservableState) -> Dict[str, Any]:
    q = len(state.waiting_queue)
    active = sum(len(g.active_request_ids) for g in state.gpu_states)
    kv_utils = [
        (g.current_kv_tokens / g.max_kv_tokens) if g.max_kv_tokens else 0.0
        for g in state.gpu_states
    ]
    kv = float(np.mean(kv_utils)) if kv_utils else 0.0
    return {
        "queue_size": int(q),
        "active_sequences": int(active),
        "kv_utilization_mean": kv,
        "completed_count_pre": int(state.completed_count),
        "sim_time": float(state.time),
    }


def _acquisition_priority(scenario_id: str, step: int, seed: int) -> float:
    h = hashlib.sha256(f"{seed}:{scenario_id}:{step}".encode()).hexdigest()
    return int(h[:16], 16) / float(16**16)


def next_p6_policy(current: str) -> str:
    idxs = {p: i for i, p in enumerate(P6)}
    if current not in idxs:
        return P6[0]
    return P6[(idxs[current] + 1) % len(P6)]


def select_alt_action(
    *,
    state: ObservableState,
    ref_action: Action,
    effective_policy: str,
    shadow_policies: Dict[str, BasePolicy],
) -> Tuple[str, Action, bool, List[str]]:
    """Return (alt_policy_id, alt_action, is_disagreement, disagreeing_ids).

    DISAGREEMENT: first P6 policy (frozen order) whose admit-set differs.
    AGREEMENT_CONTROL partner: next cyclic P6 policy after effective.
    """
    disagreeing: List[str] = []
    alt_by_pid: Dict[str, Action] = {}
    for pid in P6:
        if pid == effective_policy:
            continue
        shadow_state = copy.deepcopy(state)
        a = shadow_policies[pid].select_action(shadow_state)
        alt_by_pid[pid] = a
        if dcm.actions_disagree(ref_action, a):
            disagreeing.append(pid)

    if disagreeing:
        alt_id = disagreeing[0]
        return alt_id, alt_by_pid[alt_id], True, disagreeing

    partner = next_p6_policy(effective_policy)
    shadow_state = copy.deepcopy(state)
    partner_action = shadow_policies[partner].select_action(shadow_state)
    return partner, partner_action, False, []


@dataclass
class TracingAliveRouter(BasePolicy):
    """Alive wrapper that records last effective policy / signals for the observer."""

    inner: LiveP6DwellRouterPolicy
    name: str = "tracing_alive_joint240"
    last_effective_policy: Optional[str] = None
    last_raw_policy: Optional[str] = None
    last_signals: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        self.inner.fsm = PolicyDwellFSM(
            self.inner.policy_ids, dwell_steps=self.inner.fsm.dwell_steps
        )
        self.inner.switch_count = 0
        self.inner._last_policy = None
        self.inner.selected_policies = []
        self.last_effective_policy = None
        self.last_raw_policy = None
        self.last_signals = None

    def select_action(self, state: ObservableState) -> Action:
        sig = compute_regime_signals(state)
        x = np.asarray(
            [
                [
                    float(sig.contention_score_v2),
                    float(sig.priority_skew),
                    float(sig.kv_pressure),
                    float(sig.queue_length),
                ]
            ],
            dtype=float,
        )
        raw = str(self.inner.stage1.predict(x)[0])
        if raw not in self.inner._policies:
            raw = PROBE_POLICY
        effective = self.inner.fsm.step(raw)
        if self.inner._last_policy is not None and effective != self.inner._last_policy:
            self.inner.switch_count += 1
        self.inner._last_policy = effective
        self.inner.selected_policies.append(effective)
        self.last_effective_policy = effective
        self.last_raw_policy = raw
        self.last_signals = {
            "contention_score_v2": float(sig.contention_score_v2),
            "priority_skew": float(sig.priority_skew),
            "kv_pressure": float(sig.kv_pressure),
            "queue_length": float(sig.queue_length),
        }
        return self.inner._policies[effective].select_action(state)


@dataclass
class Joint240TerminalANWGObserver(BasePolicy):
    """Shadow observer: never alters Alive reference actions; evaluates terminal CFs."""

    name = "decision_criticality_terminal_anwg_joint240_observer_v1"

    sim_ref: Simulator
    tracing_router: TracingAliveRouter
    shadow_policies: Dict[str, BasePolicy]
    all_requests: Sequence
    scenario_id: str
    fold: int
    seed: int
    n_elevated_mechanisms: int
    pressure_row: Dict[str, Any]
    max_disagreement: int = MAX_DISAGREEMENT_PER_SCENARIO
    max_agreement_control: int = MAX_AGREEMENT_CONTROL_PER_SCENARIO
    control_seed: int = CONTROL_SEED
    run_ref_replay_once: bool = True

    n_disagreement_kept: int = 0
    n_agreement_kept: int = 0
    ref_replay_done: bool = False
    branch_rows: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.tracing_router.reset()
        for p in self.shadow_policies.values():
            if hasattr(p, "reset"):
                p.reset()
        self.n_disagreement_kept = 0
        self.n_agreement_kept = 0
        self.ref_replay_done = False
        self.branch_rows = []

    def select_action(self, state: ObservableState) -> Action:
        real_action = self.tracing_router.select_action(state)
        if len(state.waiting_queue) == 0:
            return real_action

        effective = self.tracing_router.last_effective_policy or PROBE_POLICY
        alt_id, alt_action, disagree, disagreeing = select_alt_action(
            state=state,
            ref_action=real_action,
            effective_policy=effective,
            shadow_policies=self.shadow_policies,
        )
        feats = _state_features(state)
        sigs = self.tracing_router.last_signals or {}
        meta = {
            "step": int(state.step),
            "chosen_policy_id": effective,
            "raw_stage1_policy_id": self.tracing_router.last_raw_policy,
            "alt_policy_id": alt_id,
            "disagree": bool(disagree),
            "n_disagreeing_p6": int(len(disagreeing)),
            "disagreeing_p6": ",".join(disagreeing),
            "canonical_ref_action": str(dcm.canonical_action(real_action)),
            "canonical_alt_action": str(dcm.canonical_action(alt_action)),
            **feats,
            **sigs,
            "fold": int(self.fold),
            "n_elevated_mechanisms": int(self.n_elevated_mechanisms),
            **{k: self.pressure_row.get(k) for k in PRESSURE_FLAGS},
            "fairness_pressure": self.pressure_row.get("fairness_pressure"),
            "service_heterogeneity": self.pressure_row.get("service_heterogeneity"),
            "prefill_decode_pressure": self.pressure_row.get("prefill_decode_pressure"),
            "kv_pressure_manifest": self.pressure_row.get("kv_pressure"),
            "urgency_pressure": self.pressure_row.get("urgency_pressure"),
            "burst_pressure": self.pressure_row.get("burst_pressure"),
        }

        if disagree:
            if self.n_disagreement_kept < self.max_disagreement:
                self._evaluate_intervention(
                    state=state,
                    real_action=real_action,
                    alt_action=alt_action,
                    acquisition="DISAGREEMENT",
                    meta=meta,
                )
                self.n_disagreement_kept += 1
        elif self.n_agreement_kept < self.max_agreement_control:
            meta["acquisition_priority"] = _acquisition_priority(
                self.scenario_id, int(state.step), self.control_seed
            )
            self._evaluate_intervention(
                state=state,
                real_action=real_action,
                alt_action=alt_action,
                acquisition="AGREEMENT_CONTROL",
                meta=meta,
            )
            self.n_agreement_kept += 1

        return real_action

    def _evaluate_intervention(
        self,
        *,
        state: ObservableState,
        real_action: Action,
        alt_action: Action,
        acquisition: str,
        meta: Dict[str, Any],
    ) -> None:
        cont_cf = clone_alive_router(self.tracing_router.inner)
        cf = run_one_step_then_alive_terminal(
            self.sim_ref,
            first_action=alt_action,
            continuation_router=cont_cf,
            all_requests=self.all_requests,
            workload_tag=self.scenario_id,
            seed=self.seed,
            branch_label="cf_one_step_alt",
        )

        ref_replay = None
        if self.run_ref_replay_once and not self.ref_replay_done:
            cont_rr = clone_alive_router(self.tracing_router.inner)
            ref_replay = run_one_step_then_alive_terminal(
                self.sim_ref,
                first_action=real_action,
                continuation_router=cont_rr,
                all_requests=self.all_requests,
                workload_tag=self.scenario_id,
                seed=self.seed,
                branch_label="ref_action_replay",
            )
            self.ref_replay_done = True

        row = {
            "scenario_id": self.scenario_id,
            "seed": int(self.seed),
            "acquisition_type": acquisition,
            "reference_policy": "live_p6_dwell_router_v1",
            "alternative_policy": meta["alt_policy_id"],
            "chosen_native_policy": meta["chosen_policy_id"],
            **meta,
            "cf_anwg": cf["anwg"],
            "cf_num_completed": cf["num_completed"],
            "cf_num_dropped": cf["num_dropped"],
            "cf_sim_duration": cf["sim_duration"],
            "cf_extra_steps": cf["extra_steps"],
            "cf_finished": cf["finished"],
            "live_fingerprint_unchanged": cf["live_fingerprint_unchanged"],
            "ref_replay_anwg": None if ref_replay is None else ref_replay["anwg"],
            "ref_replay_live_fingerprint_unchanged": (
                None if ref_replay is None else ref_replay["live_fingerprint_unchanged"]
            ),
        }
        self.branch_rows.append(row)


def _inner_train_val(train_ids: List[str], fold: int) -> Tuple[List[str], List[str]]:
    """Match joint240 runner exactly (seed SPLIT_SEED+100+fold)."""
    rng = np.random.default_rng(SPLIT_SEED + 100 + fold)
    ids = list(train_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(0.20 * len(ids))))
    if len(ids) - n_val < 1:
        n_val = max(0, len(ids) - 1)
    val_ids = ids[:n_val]
    tr_ids = ids[n_val:]
    return tr_ids, val_ids


def fit_oof_alive_stage1_models(
    scenarios: Dict[str, PolicySeparationScenario],
    matrix: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    fold_ids: Optional[Sequence[int]] = None,
) -> Dict[int, Pipeline]:
    """Fit one Alive Stage-1 model per OOF fold (train+val telemetry).

    `folds` must be the full frozen 240-scenario fold table so train pools match
    Section 4.2. `fold_ids` may restrict which Stage-1 models are materialised
    (e.g. dry-run); each fitted model still trains on the full complementary
    train+val pool.
    """
    lookup = matrix.set_index("scenario_id")
    models: Dict[int, Pipeline] = {}
    n_folds = int(folds["fold"].max()) + 1
    targets = list(range(n_folds)) if fold_ids is None else [int(f) for f in fold_ids]
    for fold in targets:
        train_pool = folds.loc[folds["fold"] != fold, "scenario_id"].tolist()
        tr_ids, val_ids = _inner_train_val(train_pool, fold)
        tele = []
        for sid in tr_ids + val_ids:
            vbs = str(lookup.loc[sid, "vbs_policy"])
            tele.extend(collect_probe_telemetry(scenarios[sid], vbs))
        models[fold] = fit_live_stage1(tele)
    return models

def load_frozen_joint240_context() -> Dict[str, Any]:
    matrix = load_utility_matrix()
    scenarios_list = rebuild_all_scenarios()
    scenarios = {s.scenario_id: s for s in scenarios_list}
    folds_path = JOINT240_EXP / "split_oof_folds.csv"
    if folds_path.exists():
        folds = pd.read_csv(folds_path)
    else:
        folds = freeze_oof_folds(
            matrix["scenario_id"].tolist(),
            matrix["n_elevated_mechanisms"].astype(int).tolist(),
        )
    recomputed = freeze_oof_folds(
        matrix["scenario_id"].tolist(),
        matrix["n_elevated_mechanisms"].astype(int).tolist(),
    )
    merged = folds.merge(recomputed, on="scenario_id", suffixes=("_frozen", "_re"))
    if not (merged["fold_frozen"] == merged["fold_re"]).all():
        raise RuntimeError("frozen split_oof_folds.csv does not match recomputed folds")
    manifest = pd.read_csv(JOINT_DIR / "scenario_manifest.csv")
    return {
        "matrix": matrix,
        "scenarios": scenarios,
        "folds": folds,
        "manifest": manifest,
    }


def run_scenario_terminal_anwg_joint240(
    scenario: PolicySeparationScenario,
    *,
    stage1: Pipeline,
    fold: int,
    seed: int,
    n_elevated_mechanisms: int,
    pressure_row: Dict[str, Any],
    max_disagreement: int = MAX_DISAGREEMENT_PER_SCENARIO,
    max_agreement_control: int = MAX_AGREEMENT_CONTROL_PER_SCENARIO,
) -> Dict[str, Any]:
    sid = scenario.scenario_id
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
    observer = Joint240TerminalANWGObserver(
        sim_ref=sim,
        tracing_router=tracing,
        shadow_policies=build_p6_shadow_policies(),
        all_requests=list(scenario.requests),
        scenario_id=sid,
        fold=fold,
        seed=seed,
        n_elevated_mechanisms=n_elevated_mechanisms,
        pressure_row=pressure_row,
        max_disagreement=max_disagreement,
        max_agreement_control=max_agreement_control,
    )
    observer.sim_ref = sim
    ref_metrics = sim.run(observer, workload_tag=sid, seed=seed)
    ref_anwg = float(ref_metrics.arrival_normalized_weighted_goodput)

    for br in observer.branch_rows:
        br["reference_anwg"] = ref_anwg
        br["reference_num_completed"] = int(ref_metrics.num_completed)
        br["reference_sim_duration"] = float(ref_metrics.sim_duration)
        br["delta_anwg"] = float(br["cf_anwg"] - ref_anwg)
        br["abs_delta_anwg"] = abs(float(br["delta_anwg"]))
        br["completion_count_delta"] = int(
            br["cf_num_completed"] - int(ref_metrics.num_completed)
        )
        br["sim_duration_delta"] = float(
            br["cf_sim_duration"] - float(ref_metrics.sim_duration)
        )
        br["terminal_utility_effect"] = bool(br["abs_delta_anwg"] > ANWG_EQ_ATOL)
        br["subsequent_trajectory_diverged"] = bool(
            br["terminal_utility_effect"]
            or br["completion_count_delta"] != 0
            or abs(br["sim_duration_delta"]) > 1e-12
        )
        if br.get("ref_replay_anwg") is not None:
            br["ref_replay_minus_reference_anwg"] = float(br["ref_replay_anwg"] - ref_anwg)
            br["ref_replay_matches_reference"] = bool(
                abs(br["ref_replay_minus_reference_anwg"]) <= ANWG_EQ_ATOL
            )

    return {
        "scenario_id": sid,
        "fold": int(fold),
        "seed": int(seed),
        "n_elevated_mechanisms": int(n_elevated_mechanisms),
        "reference_anwg": ref_anwg,
        "reference_num_completed": int(ref_metrics.num_completed),
        "n_branch_rows": len(observer.branch_rows),
        "n_disagreement_evaluated": observer.n_disagreement_kept,
        "n_agreement_evaluated": observer.n_agreement_kept,
        "alive_n_switches": int(alive.switch_count),
        "branch_rows": observer.branch_rows,
    }


def concentration_curve(
    abs_vals: np.ndarray, fracs: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.50)
) -> dict:
    vals = np.asarray(abs_vals, dtype=float)
    if len(vals) == 0:
        return {str(f): {"k": 0, "share": None} for f in fracs}
    order = np.argsort(-vals)
    sorted_v = vals[order]
    total = float(sorted_v.sum())
    cum = np.cumsum(sorted_v)
    out = {}
    for f in fracs:
        k = max(1, int(np.ceil(f * len(sorted_v))))
        out[str(f)] = {
            "k": k,
            "share": float(cum[k - 1] / total) if total > 0 else 0.0,
        }
    return out


def auroc_binary_score(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if len(y) == 0 or y.min() == y.max() or s.min() == s.max():
        return None
    s_pos = s[y == 1]
    s_neg = s[y == 0]
    gt = float(np.mean(s_pos[:, None] > s_neg[None, :]))
    eq = float(np.mean(s_pos[:, None] == s_neg[None, :]))
    return gt + 0.5 * eq


def auprc_binary_score(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return None
    order = np.argsort(-s)
    y_ord = y[order]
    tp = np.cumsum(y_ord)
    fp = np.cumsum(1 - y_ord)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(y.sum())
    return float(np.sum(precision * np.diff(np.concatenate([[0.0], recall]))))


def scenario_top_k_share(sc_mass: np.ndarray, k: int) -> float:
    if len(sc_mass) == 0:
        return 0.0
    total = float(sc_mass.sum())
    if total <= 0:
        return 0.0
    k = max(1, min(int(k), len(sc_mass)))
    return float(np.sort(sc_mass)[::-1][:k].sum() / total)


def assign_verdicts(summary: Dict[str, Any]) -> List[str]:
    prev = summary.get("prevalence") or {}
    n_states = int(prev.get("n_states") or 0)
    p_nz = float(prev.get("frac_nonzero") or 0.0)
    n_nz = int(round(p_nz * n_states)) if n_states else 0
    conc = summary.get("concentration_abs_delta_all_states") or {}
    m10 = conc.get("0.1", {}).get("share")
    m10 = float(m10) if m10 is not None else 0.0
    proxy = summary.get("disagreement_as_criticality_proxy") or {}
    auroc = proxy.get("auroc_disagreement_for_nonzero_abs_delta")
    auroc_ci = (summary.get("bootstrap") or {}).get("auroc", {})
    enrichment = proxy.get("enrichment_ratio")

    labels: List[str] = []
    if n_nz < 10:
        labels.append("JOINT240_INSUFFICIENT_EFFECT_EVENTS")
    else:
        sparse = p_nz < 0.15
        concentrated = m10 >= 0.50
        weak_proxy = auroc is None or float(auroc) < 0.70
        if sparse and concentrated and weak_proxy:
            labels.append("JOINT240_TERMINAL_CRITICALITY_REPLICATED")
        elif sparse ^ concentrated or (sparse and concentrated and not weak_proxy):
            labels.append("JOINT240_CRITICALITY_PARTIAL_REPLICATION")
        if p_nz >= 0.25 or m10 < 0.30:
            labels.append("JOINT240_CRITICALITY_NOT_REPLICATED")

    if (
        enrichment is not None
        and float(enrichment) > 1.5
        and auroc is not None
        and float(auroc) >= 0.70
        and auroc_ci.get("ci95_low") is not None
        and float(auroc_ci["ci95_low"]) > 0.5
    ):
        labels.append("JOINT240_DISAGREEMENT_PROXY_USEFUL")

    if not labels:
        labels.append("JOINT240_CRITICALITY_PARTIAL_REPLICATION")
    return labels
