"""Joint-240 same-distribution adaptive exploitability v1.

Implements docs/design/JOINT240_SAME_DISTRIBUTION_ADAPTIVE_EXPLOITABILITY_V1.md.
"""
from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..core.action import Action
from ..core.types import ObservableState
from ..policies.base import BasePolicy
from ..policy_separation.hierarchical_regime_router_v1 import DWELL_MINIMUM_STEPS
from ..policy_separation.online_regime_signals_v1 import compute_regime_signals
from ..policy_separation.schema import PolicySeparationScenario
from ..policy_separation.unified_utility_matrix import _build_policy
from ..simulator.service_model import ServiceModel
from ..simulator.simulator import Simulator, SimulatorConfig

ROOT = Path(__file__).resolve().parents[3]
JOINT_DIR = ROOT / "experiments" / "joint_multimechanism_generalization_v1"
JOINT_RUNNER = JOINT_DIR / "run_joint_multimechanism_generalization_v1.py"

P6: Tuple[str, ...] = (
    "full_prefill",
    "chunked_prefill_small",
    "estimated_service_time_first",
    "weighted_fair_share",
    "least_laxity_first",
    "kv_constrained_online",
)

FEATURE_ALLOWLIST: Tuple[str, ...] = (
    "offered_load",
    "burstiness",
    "long_fraction",
    "prompt_scale",
    "prompt_heterogeneity_sigma",
    "output_scale",
    "output_heterogeneity_sigma",
    "tenant_weight_skew",
    "class_share_skew",
    "slo_tightness",
    "prediction_noise_sigma",
    "kv_pressure_target",
    "late_pressure",
    "late_phase",
    "max_active_sequences",
    "step_token_budget",
    "n_requests",
)

ONLINE_FEATURES: Tuple[str, ...] = (
    "contention_score_v2",
    "priority_skew",
    "kv_pressure",
    "queue_length",
)

SPLIT_SEED = 20260825
BOOTSTRAP_SEED = 20260825
N_BOOTSTRAP = 1000
N_FOLDS = 5
CATASTROPHIC_EPS = 0.01
PROBE_POLICY = "weighted_fair_share"
SCHEMA_VERSION = "joint240_same_distribution_adaptive_exploitability_v1.0.0"


def _load_joint_runner():
    spec = importlib.util.spec_from_file_location("joint_mm_runner_v1", JOINT_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load joint runner from {JOINT_RUNNER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild_all_scenarios() -> List[PolicySeparationScenario]:
    mod = _load_joint_runner()
    rng = np.random.default_rng(int(mod.SEED))
    return [mod.build_scenario(mod.sample_params(rng, i)) for i in range(int(mod.N_SCENARIOS))]


def load_utility_matrix() -> pd.DataFrame:
    wide = pd.read_csv(JOINT_DIR / "utility_matrix_wide.csv")
    manifest = pd.read_csv(JOINT_DIR / "scenario_manifest.csv")
    df = wide.merge(
        manifest[["scenario_id", "n_elevated_mechanisms"]],
        on="scenario_id",
        how="inner",
    )
    if len(df) != 240:
        raise ValueError(f"expected 240 joint scenarios, got {len(df)}")
    rename = {f"anwg__{p}": p for p in P6}
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise KeyError(missing)
    df = df.rename(columns=rename)
    df["sbs_anwg"] = df[list(P6)].max(axis=1)
    df["sbs_policy"] = df[list(P6)].idxmax(axis=1)
    if "winner" in df.columns:
        df["vbs_policy"] = df["winner"].astype(str)
    else:
        df["vbs_policy"] = df[list(P6)].idxmax(axis=1)
    if "oracle" in df.columns:
        df["vbs_anwg"] = df["oracle"].astype(float)
    else:
        df["vbs_anwg"] = df[list(P6)].max(axis=1)
    return df.reset_index(drop=True)


def generator_feature_table(scenarios: Sequence[PolicySeparationScenario]) -> pd.DataFrame:
    rows = []
    for s in scenarios:
        row = {"scenario_id": s.scenario_id}
        for k in FEATURE_ALLOWLIST:
            if k not in s.params:
                raise KeyError(f"missing allowlisted feature {k} on {s.scenario_id}")
            row[k] = s.params[k]
        rows.append(row)
    return pd.DataFrame(rows)


def freeze_oof_folds(
    scenario_ids: Sequence[str],
    strata: Sequence[int],
    *,
    n_folds: int = N_FOLDS,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.asarray(list(scenario_ids))
    y = np.asarray(list(strata), dtype=int)
    fold = np.full(len(ids), -1, dtype=int)
    for stratum in sorted(set(y.tolist())):
        idx = np.where(y == stratum)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = int(j % n_folds)
    if np.any(fold < 0):
        raise RuntimeError("fold assignment incomplete")
    return pd.DataFrame(
        {"scenario_id": ids, "fold": fold, "n_elevated_mechanisms": y}
    )


def freeze_reference_split(
    scenario_ids: Sequence[str],
    strata: Sequence[int],
    *,
    seed: int = SPLIT_SEED,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17)
    ids = np.asarray(list(scenario_ids))
    y = np.asarray(list(strata), dtype=int)
    split = np.empty(len(ids), dtype=object)
    for stratum in sorted(set(y.tolist())):
        idx = np.where(y == stratum)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(n - 1, 0)
        n_val = (
            min(max(n_val, 1), n - n_train - 1)
            if n - n_train >= 2
            else max(n - n_train - 1, 0)
        )
        split[idx[:n_train]] = "TRAIN"
        split[idx[n_train : n_train + n_val]] = "VAL"
        split[idx[n_train + n_val :]] = "TEST"
    return pd.DataFrame(
        {"scenario_id": ids, "split": split, "n_elevated_mechanisms": y}
    )


def _make_logreg(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=SPLIT_SEED,
                ),
            ),
        ]
    )


def fit_scen_selector(X: np.ndarray, y: np.ndarray, C: float) -> Pipeline:
    classes = np.unique(y)
    if len(classes) < 2:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    DummyClassifier(strategy="most_frequent"),
                ),
            ]
        )
        pipe.fit(X, y)
        return pipe
    pipe = _make_logreg(C)
    pipe.fit(X, y)
    return pipe


def predict_policies(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X))


def majority_policy(y: Sequence[str]) -> str:
    vals, counts = np.unique(np.asarray(list(y)), return_counts=True)
    return str(vals[int(np.argmax(counts))])


def select_scen_model_on_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    ids_val: Sequence[str],
    matrix: pd.DataFrame,
) -> Tuple[str, float, Pipeline]:
    candidates = [("logreg_C1.0", 1.0), ("logreg_C0.5", 0.5)]
    best_name = candidates[0][0]
    best_score = -1e18
    best_model: Optional[Pipeline] = None
    lookup = matrix.set_index("scenario_id")
    for name, C in candidates:
        model = fit_scen_selector(X_train, y_train, C)
        preds = predict_policies(model, X_val)
        scores = [float(lookup.loc[sid, pred]) for sid, pred in zip(ids_val, preds)]
        mean_anwg = float(np.mean(scores)) if scores else float("nan")
        if mean_anwg > best_score:
            best_score = mean_anwg
            best_name = name
            best_model = model
    assert best_model is not None
    return best_name, best_score, best_model


def run_policy_anwg(scenario: PolicySeparationScenario, policy_id: str) -> float:
    mod = _load_joint_runner()
    metrics, _records, _meta = mod.run_policy_on_scenario(scenario, policy_id)
    return float(metrics["arrival_normalized_weighted_goodput"])


@dataclass
class TelemetryRow:
    scenario_id: str
    step: int
    contention_score_v2: float
    priority_skew: float
    kv_pressure: float
    queue_length: float
    vbs_policy: str


class TelemetryProbePolicy(BasePolicy):
    def __init__(
        self,
        inner: BasePolicy,
        scenario_id: str,
        vbs_policy: str,
        sink: List[TelemetryRow],
    ):
        self.inner = inner
        self.name = f"telemetry_probe[{inner.name}]"
        self.scenario_id = scenario_id
        self.vbs_policy = vbs_policy
        self.sink = sink

    def select_action(self, state: ObservableState) -> Action:
        sig = compute_regime_signals(state)
        self.sink.append(
            TelemetryRow(
                scenario_id=self.scenario_id,
                step=int(state.step),
                contention_score_v2=float(sig.contention_score_v2),
                priority_skew=float(sig.priority_skew),
                kv_pressure=float(sig.kv_pressure),
                queue_length=float(sig.queue_length),
                vbs_policy=self.vbs_policy,
            )
        )
        return self.inner.select_action(state)


class PolicyDwellFSM:
    """Simple dwell FSM over P6 policy ids (active-only; no NONE/OVERLAP)."""

    def __init__(self, allowed: Sequence[str], dwell_steps: int = DWELL_MINIMUM_STEPS):
        self.allowed = set(allowed)
        self.dwell_steps = int(dwell_steps)
        self._effective: Optional[str] = None
        self._steps_since_change = 0
        self.transitions = 0

    def step(self, raw: str) -> str:
        if raw not in self.allowed:
            raise ValueError(f"unknown policy token {raw!r}")
        if self._effective is None:
            self._effective = raw
            self._steps_since_change = 0
            return self._effective
        if raw == self._effective:
            self._steps_since_change += 1
        elif self._steps_since_change >= self.dwell_steps:
            self._effective = raw
            self._steps_since_change = 0
            self.transitions += 1
        else:
            self._steps_since_change += 1
        return self._effective


def collect_probe_telemetry(
    scenario: PolicySeparationScenario,
    vbs_policy: str,
    *,
    probe_policy: str = PROBE_POLICY,
) -> List[TelemetryRow]:
    sink: List[TelemetryRow] = []
    inner, sm_override = _build_policy(probe_policy)
    policy = TelemetryProbePolicy(inner, scenario.scenario_id, vbs_policy, sink)
    merged_sm = dict(scenario.service_model_kwargs)
    merged_sm.update(sm_override)
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**merged_sm),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scenario.requests))
    sim.run(policy)
    return sink


class LiveP6DwellRouterPolicy(BasePolicy):
    """Online 6-way selector with dwell; Stage-1 maps online features → P6 id."""

    def __init__(
        self,
        stage1: Pipeline,
        policy_ids: Sequence[str],
        dwell_steps: int = DWELL_MINIMUM_STEPS,
    ):
        self.stage1 = stage1
        self.policy_ids = list(policy_ids)
        self.name = "live_p6_dwell_router_v1"
        self.fsm = PolicyDwellFSM(self.policy_ids, dwell_steps=dwell_steps)
        self._policies = {pid: _build_policy(pid)[0] for pid in self.policy_ids}
        self.switch_count = 0
        self._last_policy: Optional[str] = None
        self.selected_policies: List[str] = []

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
        raw = str(self.stage1.predict(x)[0])
        if raw not in self._policies:
            raw = PROBE_POLICY
        effective = self.fsm.step(raw)
        if self._last_policy is not None and effective != self._last_policy:
            self.switch_count += 1
        self._last_policy = effective
        self.selected_policies.append(effective)
        return self._policies[effective].select_action(state)


def run_live_router_anwg(
    scenario: PolicySeparationScenario, stage1: Pipeline
) -> Tuple[float, int, List[str]]:
    policy = LiveP6DwellRouterPolicy(stage1, P6)
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scenario.gpu_configs),
            service_model=ServiceModel(**dict(scenario.service_model_kwargs)),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scenario.requests))
    metrics = sim.run(policy)
    return (
        float(metrics.arrival_normalized_weighted_goodput),
        int(policy.switch_count),
        list(policy.selected_policies),
    )


def fit_live_stage1(rows: Sequence[TelemetryRow]) -> Pipeline:
    if not rows:
        raise ValueError("empty telemetry")
    X = np.asarray([[getattr(r, c) for c in ONLINE_FEATURES] for r in rows], dtype=float)
    y = np.asarray([r.vbs_policy for r in rows])
    return fit_scen_selector(X, y, C=1.0)


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    boots = [float(np.mean(d[rng.integers(0, n, size=n)])) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(np.mean(d)), "ci95_low": float(lo), "ci95_high": float(hi)}


def summarize_oof(rows: pd.DataFrame, adaptive_col: str) -> Dict[str, Any]:
    """Aggregate same-X metrics.

    SBS is the single best *fixed* P6 policy on this row set (argmax mean ANWG),
    not the per-scenario max (that is VBS).
    """
    means = {p: float(rows[p].mean()) for p in P6 if p in rows.columns}
    if not means:
        # Fallback if policy columns absent: use provided sbs_anwg column.
        sbs = rows["sbs_anwg"].to_numpy(dtype=float)
        sbs_policy = "unknown"
    else:
        sbs_policy = max(means, key=means.get)
        sbs = rows[sbs_policy].to_numpy(dtype=float)
    vbs = rows["vbs_anwg"].to_numpy(dtype=float)
    ada = rows[adaptive_col].to_numpy(dtype=float)
    headroom = float(np.mean(vbs - sbs))
    realized = float(np.mean(ada - sbs))
    gap = float(np.mean(vbs - ada))
    closure = float(realized / headroom) if headroom > 0 else float("nan")
    regret = vbs - ada
    return {
        "n": int(len(rows)),
        "SBS_policy": sbs_policy,
        "R_SBS": float(np.mean(sbs)),
        "R_VBS": float(np.mean(vbs)),
        f"R_{adaptive_col}": float(np.mean(ada)),
        "headroom": headroom,
        "realized_gain": realized,
        "exploitability_gap": gap,
        "gap_closure": closure,
        "frac_beat_sbs": float(np.mean(ada > sbs)),
        "frac_lose_to_sbs": float(np.mean(ada < sbs)),
        "median_regret_vs_vbs": float(np.median(regret)),
        "p90_regret_vs_vbs": float(np.quantile(regret, 0.90)),
        "catastrophic_lt_sbs_minus_eps": int(np.sum(ada < sbs - CATASTROPHIC_EPS)),
        "bootstrap_adaptive_minus_sbs": paired_bootstrap(ada, sbs),
        "bootstrap_vbs_minus_adaptive": paired_bootstrap(vbs, ada),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_matrix_vs_live(
    scenarios: Dict[str, PolicySeparationScenario],
    matrix: pd.DataFrame,
    scenario_ids: Sequence[str],
    *,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    rows = []
    lookup = matrix.set_index("scenario_id")
    for sid in scenario_ids:
        s = scenarios[sid]
        mrow = lookup.loc[sid]
        for pid in P6:
            live = run_policy_anwg(s, pid)
            frozen = float(mrow[pid])
            err = abs(live - frozen)
            rows.append(
                {
                    "scenario_id": sid,
                    "policy": pid,
                    "live": live,
                    "frozen": frozen,
                    "abs_err": err,
                    "ok": err <= atol or err <= 1e-8 * max(1.0, abs(frozen)),
                }
            )
    return {
        "ok": all(r["ok"] for r in rows),
        "n_checks": len(rows),
        "max_abs_err": max((r["abs_err"] for r in rows), default=0.0),
        "rows": rows,
    }
