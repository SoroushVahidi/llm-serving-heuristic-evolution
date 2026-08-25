"""Focused tests for joint-240 terminal-ANWG decision criticality v1."""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llmserveopt.analysis import decision_criticality_terminal_anwg_joint240_v1 as jtan
from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.analysis.joint240_same_distribution_adaptive_v1 import (
    P6,
    LiveP6DwellRouterPolicy,
    rebuild_all_scenarios,
)
from llmserveopt.core.types import ObservableState
from llmserveopt.policies.base import BasePolicy
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

_SPEC = importlib.util.spec_from_file_location(
    "run_decision_criticality_terminal_anwg_joint240_v1",
    Path(__file__).parent.parent
    / "scripts"
    / "run_decision_criticality_terminal_anwg_joint240_v1.py",
)
_runner = importlib.util.module_from_spec(_SPEC)
sys.modules["run_decision_criticality_terminal_anwg_joint240_v1"] = _runner
_SPEC.loader.exec_module(_runner)


def _dummy_stage1() -> Pipeline:
    pipe = Pipeline([("clf", DummyClassifier(strategy="constant", constant=P6[3]))])
    # DummyClassifier needs fit
    X = np.zeros((4, 4))
    y = np.array([P6[3]] * 4)
    pipe.fit(X, y)
    return pipe


def test_clone_alive_router_does_not_alias_fsm():
    stage1 = _dummy_stage1()
    r = LiveP6DwellRouterPolicy(stage1, P6)
    r.fsm.step(P6[0])
    c = jtan.clone_alive_router(r)
    assert c.fsm is not r.fsm
    assert c._policies is not r._policies
    assert c.stage1 is r.stage1  # shared read-only


def test_concentration_zero_mass_rule():
    conc = jtan.concentration_curve(np.zeros(20), fracs=(0.01, 0.05, 0.10))
    assert conc["0.01"]["share"] == 0.0
    assert conc["0.05"]["share"] == 0.0


def test_auprc_and_auroc_basic():
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    auroc = jtan.auroc_binary_score(y, s)
    auprc = jtan.auprc_binary_score(y, s)
    assert auroc is not None and auroc > 0.5
    assert auprc is not None and auprc > 0.3


def test_select_alt_action_disagreement_uses_frozen_p6_order():
    scen = rebuild_all_scenarios()[0]
    stage1 = _dummy_stage1()
    policy = LiveP6DwellRouterPolicy(stage1, P6)
    shadows = jtan.build_p6_shadow_policies()
    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scen.gpu_configs),
            service_model=ServiceModel(**dict(scen.service_model_kwargs)),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scen.requests))

    class Drive(BasePolicy):
        name = "drive_alt"

        def __init__(self):
            self.inner = policy
            self.checked = False

        def reset(self):
            pass

        def select_action(self, state: ObservableState):
            a = self.inner.select_action(state)
            if (not self.checked) and len(state.waiting_queue) > 0:
                self.checked = True
                alt_id, alt_a, disagree, disagreeing = jtan.select_alt_action(
                    state=state,
                    ref_action=a,
                    effective_policy=self.inner._last_policy or P6[3],
                    shadow_policies=shadows,
                )
                assert alt_id in P6
                if disagree:
                    assert disagreeing
                    assert alt_id == disagreeing[0]
                    assert dcm.actions_disagree(a, alt_a)
                else:
                    assert alt_id == jtan.next_p6_policy(self.inner._last_policy or P6[3])
            return a

    d = Drive()
    sim.run(d, workload_tag=scen.scenario_id, seed=0)
    assert d.checked


def test_ref_action_replay_matches_reference_on_joint_scenario():
    scen = rebuild_all_scenarios()[0]
    stage1 = _dummy_stage1()
    sid = scen.scenario_id

    sim_ref = Simulator(
        SimulatorConfig(
            gpu_configs=list(scen.gpu_configs),
            service_model=ServiceModel(**dict(scen.service_model_kwargs)),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim_ref.load_trace(list(scen.requests))
    router_ref = LiveP6DwellRouterPolicy(stage1, P6)
    m_ref = sim_ref.run(router_ref, workload_tag=sid, seed=int(scen.params.get("seed", 0)))
    ref_anwg = float(m_ref.arrival_normalized_weighted_goodput)

    sim = Simulator(
        SimulatorConfig(
            gpu_configs=list(scen.gpu_configs),
            service_model=ServiceModel(**dict(scen.service_model_kwargs)),
            max_steps=80_000,
            drain_steps=20_000,
        )
    )
    sim.load_trace(list(scen.requests))
    router = LiveP6DwellRouterPolicy(stage1, P6)

    class Drive(BasePolicy):
        name = "drive_replay_j240"

        def __init__(self):
            self.inner = router
            self.sim = sim
            self.replay_anwg = None

        def reset(self):
            pass

        def select_action(self, state: ObservableState):
            a = self.inner.select_action(state)
            if self.replay_anwg is None and len(state.waiting_queue) > 0 and state.step >= 5:
                cont = jtan.clone_alive_router(self.inner)
                out = jtan.run_one_step_then_alive_terminal(
                    self.sim,
                    first_action=copy.deepcopy(a),
                    continuation_router=cont,
                    all_requests=list(scen.requests),
                    workload_tag=sid,
                    seed=int(scen.params.get("seed", 0)),
                    branch_label="ref_replay",
                )
                assert out["live_fingerprint_unchanged"]
                self.replay_anwg = out["anwg"]
            return a

    d = Drive()
    sim.run(d, workload_tag=sid, seed=int(scen.params.get("seed", 0)))
    assert d.replay_anwg is not None
    assert abs(float(d.replay_anwg) - ref_anwg) <= jtan.ANWG_EQ_ATOL


def test_analyze_bootstrap_and_verdict_helpers():
    rows = []
    for i in range(30):
        rows.append(
            {
                "scenario_id": f"joint_mm_{i % 5:04d}",
                "step": i,
                "acquisition_type": "DISAGREEMENT" if i % 2 == 0 else "AGREEMENT_CONTROL",
                "delta_anwg": 0.1 if i < 3 else 0.0,
                "abs_delta_anwg": 0.1 if i < 3 else 0.0,
                "n_elevated_mechanisms": 2,
                "subsequent_trajectory_diverged": i < 3,
                "cf_extra_steps": 100,
                "ref_replay_anwg": np.nan,
                "reference_anwg": 0.3,
            }
        )
    df = pd.DataFrame(rows)
    # Shrink bootstrap for unit test speed by monkeypatching
    old_b = jtan.N_BOOTSTRAP
    jtan.N_BOOTSTRAP = 20
    try:
        summary = _runner.analyze(df)
    finally:
        jtan.N_BOOTSTRAP = old_b
    assert summary["prevalence"]["n_states"] == 30
    assert "bootstrap" in summary
    assert "top1pct_state_mass_share" in summary["bootstrap"]
    assert isinstance(summary["verdicts"], list)


def test_frozen_folds_match_recompute():
    ctx = jtan.load_frozen_joint240_context()
    assert len(ctx["scenarios"]) == 240
    assert len(ctx["folds"]) == 240
