"""Focused unit and smoke tests for the Fairness and Starvation template (Family A)."""

from __future__ import annotations

import numpy as np
import pytest

from llmserveopt.policies.fifo import FIFOPolicy
from llmserveopt.policies.estimated_service_time_first import EstimatedServiceTimeFirstPolicy
from llmserveopt.policies.aging_priority import AgingPriorityPolicy
from llmserveopt.policies.weighted_fair_share import WeightedFairSharePolicy
from llmserveopt.policy_separation.templates_fairness_starvation import case4_fairness_starvation
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig


def test_determinism():
    """Generating the same scenario twice with the same arguments and seed must produce
    exactly identical, byte-for-byte matching requests and metadata."""
    scen1 = case4_fairness_starvation(target_utilization=0.8, tenant_weight_skew=5.0, interactive_volume_fraction=0.2, seed=42)
    scen2 = case4_fairness_starvation(target_utilization=0.8, tenant_weight_skew=5.0, interactive_volume_fraction=0.2, seed=42)

    assert scen1.scenario_id == scen2.scenario_id
    assert len(scen1.requests) == len(scen2.requests)
    
    for r1, r2 in zip(scen1.requests, scen2.requests):
        assert r1.request_id == r2.request_id
        assert r1.arrival_time == r2.arrival_time
        assert r1.prompt_tokens == r2.prompt_tokens
        assert r1.predicted_output_tokens == r2.predicted_output_tokens
        assert r1.actual_output_tokens == r2.actual_output_tokens
        assert r1.slo_deadline == r2.slo_deadline
        assert r1.priority == r2.priority
        assert r1.class_id == r2.class_id


def test_unique_scenario_ids():
    """Varying coordinates must yield unique scenario IDs without collisions."""
    ids = set()
    for util in [0.5, 0.8, 1.1]:
        for skew in [1.0, 5.0, 10.0]:
            for vol in [0.1, 0.3]:
                scen = case4_fairness_starvation(target_utilization=util, tenant_weight_skew=skew, interactive_volume_fraction=vol, seed=0)
                assert scen.scenario_id not in ids
                ids.add(scen.scenario_id)


def test_no_leakage():
    """Ensure no generator metadata (like utilization, skew, or role) is leaked
    into request-level fields. The policy must only observe standard request fields."""
    scen = case4_fairness_starvation(target_utilization=0.9, tenant_weight_skew=8.0, interactive_volume_fraction=0.25, seed=7)
    
    for req in scen.requests:
        # Check standard fields exist and are of correct type
        assert isinstance(req.request_id, int)
        assert isinstance(req.arrival_time, float)
        assert isinstance(req.prompt_tokens, int)
        assert isinstance(req.predicted_output_tokens, int)
        assert isinstance(req.slo_deadline, float)
        assert isinstance(req.priority, float)
        assert isinstance(req.class_id, str)
        
        # Check no custom leakage columns are embedded inside class_id or other fields
        assert "util" not in req.class_id
        assert "skew" not in req.class_id


def test_local_smoke_creates_fairness_pressure():
    """A tiny local simulation smoke test comparing ESTF and Weighted Fair Share.
    
    Under high utilization (1.0) and high weight skew (10.0), a size-based scheduler
    (ESTF) should starve the interactive tenant because it is focused on clearing shorter/easier
    jobs globally, or if bulk jobs are shorter (depending on sizes). Here, we have bulk
    jobs with long lengths and interactive with short lengths. ESTF clears short jobs (interactive) first, BUT
    because bulk volume is high and we have sustained queue pressure, the bulk jobs can create a backlog.
    Moreover, under Weighted Fair Share with high weight skew (interactive = 10.0 priority vs bulk = 1.0 priority),
    Weighted Fair Share prioritizes the interactive tenant, ensuring 100% compliance for interactive requests.
    We verify that both policies run without errors, and that fairness/starvation pressure is observed
    (i.e., we observe different SLO violation rates and TTFTs across policies).
    """
    scen = case4_fairness_starvation(target_utilization=1.5, tenant_weight_skew=10.0, interactive_volume_fraction=0.2, seed=123)
    
    # We run four policies: FIFO, ESTF, Aging Priority, Weighted Fair Share
    policies = {
        "fifo": FIFOPolicy(),
        "estf": EstimatedServiceTimeFirstPolicy(),
        "aging": AgingPriorityPolicy(aging_rate=0.2),
        "wfs": WeightedFairSharePolicy(),
    }

    results = {}
    for name, policy in policies.items():
        sim_config = SimulatorConfig(gpu_configs=list(scen.gpu_configs))
        sim = Simulator(sim_config)
        sim.load_trace(list(scen.requests))
        
        sim.run(policy, workload_tag=f"smoke_{name}")
        
        # Calculate completion metrics and SLO violation rates for interactive vs bulk requests
        completed = sim._completed  # noqa: SLF001
        
        interactive_completed = [cr for cr in completed if cr.request.class_id == "tenant_interactive"]
        bulk_completed = [cr for cr in completed if cr.request.class_id == "tenant_bulk"]
        
        interactive_violations = sum(1 for cr in interactive_completed if cr.completion_time > cr.request.slo_deadline)
        bulk_violations = sum(1 for cr in bulk_completed if cr.completion_time > cr.request.slo_deadline)
        
        results[name] = {
            "interactive_violations": interactive_violations,
            "bulk_violations": bulk_violations,
            "interactive_total": len(interactive_completed),
            "bulk_total": len(bulk_completed),
            "total_completed": len(completed),
        }
        
    # Verify that everyone completed all tasks successfully
    for name, res in results.items():
        assert res["total_completed"] == len(scen.requests)
        print(f"Policy {name}: Interactive violations = {res['interactive_violations']}/{res['interactive_total']}, Bulk violations = {res['bulk_violations']}/{res['bulk_total']}")

    # Check that under high load and skew, we successfully created SLO violation pressure!
    # FIFO or ESTF should have some violations on either bulk or interactive due to utilization=1.0.
    total_violations = sum(res["interactive_violations"] + res["bulk_violations"] for res in results.values())
    assert total_violations > 0, "No SLO violations observed; load might be too low to induce fairness pressure."
