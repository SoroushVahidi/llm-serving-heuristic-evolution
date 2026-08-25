"""Focused tests for terminal-ANWG one-step decision criticality v1."""
from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llmserveopt.analysis import decision_criticality_terminal_anwg_v1 as tan
from llmserveopt.analysis import decision_criticality_timescale_trainval_v1 as dcm
from llmserveopt.core.types import ObservableState
from llmserveopt.policies.base import BasePolicy
from llmserveopt.policy_separation.hierarchical_router_live_harness_v1 import (
    LiveHierarchicalRouterPolicy,
    build_feature_rows_by_regime,
    build_native_policy_instances,
)
from llmserveopt.policy_separation.templates_fairness_starvation_v2 import case_fairness_vs_size_v2
from llmserveopt.simulator.service_model import ServiceModel
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

_SPEC = importlib.util.spec_from_file_location(
    "run_decision_criticality_terminal_anwg_v1",
    Path(__file__).parent.parent / "scripts" / "run_decision_criticality_terminal_anwg_v1.py",
)
_runner = importlib.util.module_from_spec(_SPEC)
sys.modules["run_decision_criticality_terminal_anwg_v1"] = _runner
_SPEC.loader.exec_module(_runner)


def _tiny_scenario():
    return case_fairness_vs_size_v2(
        target_utilization=0.8,
        tenant_weight_skew=2.0,
        favored_tenant_size="short",
        prediction_noise_sigma=0.0,
        seed=900001,
        allow_synthetic_tokens=True,
        n_total_jobs=20,
    )


def test_clone_live_router_does_not_alias_fsm():
    scen = _tiny_scenario()
    stage1, stage2 = dcm.fit_frozen_models()
    cid = "SYNTH::test"
    fr = build_feature_rows_by_regime(scen, cid)
    r = LiveHierarchicalRouterPolicy(
        scenario_id=cid, stage1=stage1, stage2_selectors=stage2,
        feature_rows_by_regime=fr, record_trajectory=False,
    )
    r.reset()
    c = tan.clone_live_router(r)
    assert c._fsm is not r._fsm
    assert c.native_policies is not r.native_policies


def test_ref_action_replay_matches_reference_anwg():
    """Mandatory equivalence: force a_ref on a mid-run fork, continue with router clone."""
    stage1, stage2 = dcm.fit_frozen_models()
    scen = _tiny_scenario()
    cid = "SYNTH::replay"
    fr = build_feature_rows_by_regime(scen, cid)
    # Untouched reference
    sim_ref = Simulator(SimulatorConfig(
        gpu_configs=list(scen.gpu_configs),
        service_model=ServiceModel(**scen.service_model_kwargs),
        max_steps=100_000,
        drain_steps=20_000,
    ))
    sim_ref.load_trace(list(scen.requests))
    router_ref = LiveHierarchicalRouterPolicy(
        scenario_id=cid, stage1=stage1, stage2_selectors=stage2,
        feature_rows_by_regime=fr, record_trajectory=True,
    )
    m_ref = sim_ref.run(router_ref, workload_tag=cid, seed=900001)
    ref_anwg = float(m_ref.arrival_normalized_weighted_goodput)

    # Second run: at step>=5, fork with reference action and continue
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scen.gpu_configs),
        service_model=ServiceModel(**scen.service_model_kwargs),
        max_steps=100_000,
        drain_steps=20_000,
    ))
    sim.load_trace(list(scen.requests))
    router = LiveHierarchicalRouterPolicy(
        scenario_id=cid, stage1=stage1, stage2_selectors=stage2,
        feature_rows_by_regime=fr, record_trajectory=True,
    )

    class Drive(BasePolicy):
        name = "drive_replay"

        def __init__(self):
            self.inner = router
            self.sim = sim
            self.replay_anwg = None

        def reset(self):
            self.inner.reset()

        def select_action(self, state: ObservableState):
            a = self.inner.select_action(state)
            if self.replay_anwg is None and state.step >= 5:
                cont = tan.clone_live_router(self.inner)
                out = tan.run_one_step_then_router_terminal(
                    self.sim,
                    first_action=copy.deepcopy(a),
                    continuation_router=cont,
                    all_requests=list(scen.requests),
                    workload_tag=cid,
                    seed=900001,
                    branch_label="ref_replay",
                )
                assert out["live_fingerprint_unchanged"]
                self.replay_anwg = out["anwg"]
            return a

    d = Drive()
    sim.run(d, workload_tag=cid, seed=900001)
    assert d.replay_anwg is not None
    assert abs(float(d.replay_anwg) - ref_anwg) <= tan.ANWG_EQ_ATOL


def test_fork_does_not_mutate_live_sim_fingerprint():
    stage1, stage2 = dcm.fit_frozen_models()
    scen = _tiny_scenario()
    cid = "SYNTH::iso"
    fr = build_feature_rows_by_regime(scen, cid)
    sim = Simulator(SimulatorConfig(
        gpu_configs=list(scen.gpu_configs),
        service_model=ServiceModel(**scen.service_model_kwargs),
    ))
    sim.load_trace(list(scen.requests))
    router = LiveHierarchicalRouterPolicy(
        scenario_id=cid, stage1=stage1, stage2_selectors=stage2,
        feature_rows_by_regime=fr, record_trajectory=True,
    )

    class Drive(BasePolicy):
        name = "drive"

        def __init__(self):
            self.inner = router
            self.sim = sim
            self.did = False

        def reset(self):
            self.inner.reset()

        def select_action(self, state: ObservableState):
            a = self.inner.select_action(state)
            if (not self.did) and state.step >= 2:
                self.did = True
                fp = dcm._state_fingerprint(self.sim)
                cont = tan.clone_live_router(self.inner)
                out = tan.run_one_step_then_router_terminal(
                    self.sim, first_action=copy.deepcopy(a), continuation_router=cont,
                    all_requests=list(scen.requests), workload_tag=cid, seed=1,
                    branch_label="iso",
                )
                assert out["live_fingerprint_unchanged"]
                assert dcm._state_fingerprint(self.sim) == fp
            return a

    sim.run(Drive(), workload_tag=cid, seed=1)


def _synthetic_branches(acquisition_types: list[str]) -> pd.DataFrame:
    # 10 rows; delta_anwg deliberately uncorrelated with acquisition_type so that
    # a variable-shadowing regression (analyze() reusing a local name across the
    # positive-gain and AUROC/disagreement-proxy blocks) would visibly change
    # concentration_positive_gain_all_states between the two acquisition-label
    # arrangements below, even though the underlying deltas are identical.
    delta = [0.5, 0.3, 0.2, -0.4, 0.0, 0.0, 0.1, -0.1, 0.05, 0.0]
    return pd.DataFrame({
        "canonical_scenario_id": [f"SCN::{i % 3}" for i in range(10)],
        "step": list(range(10)),
        "acquisition_type": acquisition_types,
        "mechanism_family": ["TESTFAM"] * 10,
        "delta_anwg": delta,
        "abs_delta_anwg": [abs(d) for d in delta],
        "ref_replay_anwg": [float("nan")] * 10,
    })


def test_concentration_positive_gain_uses_max_delta_and_ignores_auroc_block():
    """Regression test for a variable-shadowing bug: analyze() previously reused a
    local name for both the positive-gain array (max(delta_anwg, 0)) and an
    unrelated AUROC/disagreement-proxy scratch array, so
    concentration_positive_gain_all_states silently changed depending on whether
    the AUROC block executed. It must depend only on delta_anwg.
    """
    delta = pd.Series([0.5, 0.3, 0.2, -0.4, 0.0, 0.0, 0.1, -0.1, 0.05, 0.0])
    expected = _runner.concentration_curve(np.maximum(delta.to_numpy(dtype=float), 0.0))

    # Case A: acquisition_type constant -> disagreement AUROC block is skipped
    # (s.min() == s.max()), so the positive-gain array is never touched by it.
    constant_labels = ["DISAGREEMENT"] * 10
    result_a = _runner.analyze(_synthetic_branches(constant_labels))
    assert result_a["concentration_positive_gain_all_states"] == expected

    # Case B: acquisition_type varies -> AUROC/disagreement-proxy block executes.
    varying_labels = ["DISAGREEMENT", "AGREEMENT_CONTROL"] * 5
    result_b = _runner.analyze(_synthetic_branches(varying_labels))
    assert result_b["disagreement_as_criticality_proxy"]["available"] is True
    assert result_b["concentration_positive_gain_all_states"] == expected

    # Both arrangements must agree: positive-gain concentration is independent of
    # whether/how the AUROC block ran.
    assert result_a["concentration_positive_gain_all_states"] == result_b["concentration_positive_gain_all_states"]


def test_prevalence_is_full_summary_dict_not_last_burst_step():
    """Regression test for a second instance of the same shadowing bug: analyze()
    previously reused a local name for both the prevalence summary dict and a
    burst-length-tracking loop variable, so summary["prevalence"] silently ended
    up as an arbitrary trajectory step number instead of the intended stats dict.
    """
    labels = ["DISAGREEMENT", "AGREEMENT_CONTROL"] * 5
    result = _runner.analyze(_synthetic_branches(labels))
    prevalence = result["prevalence"]
    assert isinstance(prevalence, dict)
    assert prevalence["n_states"] == 10
    assert prevalence["frac_nonzero"] == pytest.approx(0.7)


def test_module_does_not_import_family_b_replication():
    src = inspect.getsource(tan)
    assert "family_b_balanced_replication_v1" not in src
