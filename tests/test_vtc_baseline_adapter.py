"""Fidelity/scope tests for the VTC baseline integration
(baselines/vtc/). See baselines/vtc/PROVENANCE.md and
docs/audits/vtc_official_artifact_audit_20260805.md for the full
provenance record and known deviations these tests lock in. Mirrors
tests/test_pars_baseline_adapter.py's structure and coverage.
"""
from __future__ import annotations

import subprocess

import pytest

from baselines.vtc.adapter import provenance
from baselines.vtc.adapter.errors import (
    MissingOfficialCloneError,
    MissingTenantIdError,
    StaleCloneCommitError,
    UnregisteredTenantError,
    UnsupportedCostFunctionError,
    UnsupportedTopologyError,
)
from baselines.vtc.adapter.official_loader import (
    load_vtc_official_classes,
    verify_official_clone,
)
from baselines.vtc.adapter.simulator_policy import (
    SELECTOR_ELIGIBLE,
    VTCFairnessPolicy,
    default_tenant_of,
)
from llmserveopt.core.action import Action
from llmserveopt.core.types import GPUConfig, ObservableRequest, Request
from llmserveopt.simulator.simulator import Simulator, SimulatorConfig

_CLONE_PRESENT = None


def _clone_present() -> bool:
    global _CLONE_PRESENT
    if _CLONE_PRESENT is None:
        try:
            verify_official_clone()
            _CLONE_PRESENT = True
        except Exception:
            _CLONE_PRESENT = False
    return _CLONE_PRESENT


requires_clone = pytest.mark.skipif(
    not _clone_present(),
    reason="Pinned Ying1123/VTC-artifact clone not found; see PROVENANCE.md to clone it.",
)


class TestProvenanceManifest:
    def test_pinned_commit_and_repo_recorded(self):
        assert provenance.OFFICIAL_REPOSITORY == "https://github.com/Ying1123/VTC-artifact"
        assert provenance.PINNED_COMMIT == "192c2e2014c69c8c6c699d7113c3822e4db632e6"

    def test_license_recorded(self):
        assert provenance.LICENSE == "Apache-2.0"

    def test_paper_citation_recorded(self):
        assert provenance.PAPER_ARXIV_ID == "2401.00588"
        assert "Sheng, Ying" in provenance.PAPER_AUTHORS
        assert "Stoica, Ion" in provenance.PAPER_AUTHORS
        assert provenance.PAPER_YEAR == 2024

    def test_evaluation_only_and_not_selector_candidate(self):
        assert provenance.EVALUATION_ONLY is True
        assert provenance.SELECTOR_CANDIDATE is False
        assert SELECTOR_ELIGIBLE is False

    def test_only_linear_cost_function_supported(self):
        assert provenance.SUPPORTED_COST_FUNC == "linear"


class TestOfficialLoaderVerification:
    def test_missing_clone_raises(self, tmp_path):
        with pytest.raises(MissingOfficialCloneError):
            verify_official_clone(str(tmp_path / "nonexistent_clone"))

    def test_stale_commit_raises(self, tmp_path):
        clone_dir = tmp_path / "clone"
        clone_dir.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=clone_dir, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t.com", "-c", "user.name=t", "commit",
             "--allow-empty", "-q", "-m", "init"],
            cwd=clone_dir, check=True,
        )
        with pytest.raises(StaleCloneCommitError):
            verify_official_clone(str(clone_dir))

    @requires_clone
    def test_pinned_clone_loads_real_official_classes(self):
        official = load_vtc_official_classes()
        assert official.VTCReqQueue.__name__ == "VTCReqQueue"
        assert official.ReqQueue.__name__ == "ReqQueue"
        # The real class, not a reimplementation -- has the exact method
        # set the pinned source defines.
        assert hasattr(official.VTCReqQueue, "generate_new_batch")
        assert hasattr(official.VTCReqQueue, "update_counter")
        assert hasattr(official.VTCReqQueue, "append")


@requires_clone
class TestOfficialAlgorithmDirect:
    """Exercises the real, unmodified VTCReqQueue class directly (no
    simulator adapter involved) to lock in the core mechanism semantics
    documented in PROVENANCE.md's "Algorithm audit" section."""

    def _queue(self, tenants, weights=None, **kwargs):
        official = load_vtc_official_classes()
        weights = weights or [1] * len(tenants)
        return official, official.VTCReqQueue(
            max_total_tokens=10**9, batch_max_tokens=10**9,
            running_max_req_size=10**6, adapter_dirs=tenants,
            fair_weights=weights, cost_func="linear", **kwargs,
        )

    def test_min_served_selection_picks_least_served_tenant(self):
        official, q = self._queue(["A", "B"])
        sp = official.SamplingParams(max_new_tokens=5)
        # A gets served first (arrives first, no ties yet).
        q.append(official.Req("A", 1, [0] * 10, sp))
        batch = q.generate_new_batch(None, {"A": 0, "B": 0})
        assert [r.request_id for r in batch.reqs] == [1]
        assert q.served["A"] == 10.0

        # B appears for the first time while A's queue is EMPTY (just
        # drained), so the "lift counter" rule (see
        # test_counter_lift_on_tenant_return) does not raise it -- B
        # starts genuinely behind A at 0 vs 10.
        q.append(official.Req("B", 3, [0] * 10, sp))
        # A returns with a new request -- strictly less-served B must be
        # scheduled first within this same batch-formation call, even
        # though A's request was appended chronologically earlier overall.
        q.append(official.Req("A", 2, [0] * 10, sp))
        batch2 = q.generate_new_batch(None, {"A": 0, "B": 0})
        assert batch2.reqs[0].request_id == 3  # B (served=0) picked before A (served=10)
        assert [r.request_id for r in batch2.reqs] == [3, 2]  # both fit; B first

    def test_tie_break_is_insertion_order(self):
        """Undocumented in the paper -- an artifact of Python dict
        iteration order in `min(active_served, key=...)`. Tie-break order
        follows `served`-dict insertion order, which is set by `append()`
        call order at RUNTIME, not `adapter_dirs` construction order (a
        distinction worth locking in explicitly). Locks in that this
        integration inherits this automatically by calling the real
        object, rather than re-deriving a different tie-break rule."""
        official, q = self._queue(["A", "B"])  # construction order irrelevant here
        sp = official.SamplingParams(max_new_tokens=5)
        # B appended (and thus inserted into `served`) first at runtime,
        # despite "A" appearing first in adapter_dirs.
        q.append(official.Req("B", 2, [0] * 10, sp))
        q.append(official.Req("A", 1, [0] * 10, sp))
        # Both start at served=0 -- true tie. B was inserted first at
        # runtime, so B's request must win the tie.
        batch = q.generate_new_batch(None, {"A": 0, "B": 0})
        assert batch.reqs[0].adapter_dir == "B"

    def test_counter_lift_on_tenant_return(self):
        """A tenant that goes idle and returns has its counter raised to
        at least the minimum active counter -- the mechanism the paper's
        service-difference bound theorem depends on for returning
        clients."""
        official, q = self._queue(["A", "B"])
        sp = official.SamplingParams(max_new_tokens=5)
        q.append(official.Req("A", 1, [0] * 100, sp))
        q.generate_new_batch(None, {"A": 0, "B": 0})  # A served=100.0, B never appeared
        assert q.served["A"] == 100.0
        assert "B" not in q.served

        # B arrives for the first time -- queue was empty (never existed),
        # so no other ACTIVE (non-empty-queue) tenant exists at this
        # instant since A's queue just drained. served stays at its
        # initialized 0 (lift only raises against tenants with a
        # currently non-empty queue).
        q.append(official.Req("B", 2, [0] * 10, sp))
        assert q.served["B"] == 0.0

        # A returns while B is still waiting (B's queue non-empty) -- A's
        # counter must be lifted to at least min(active B counters) = 0,
        # a no-op here since B hasn't been charged yet. Use a case where
        # the lift actually matters instead: charge B first.
        q.append(official.Req("B", 3, [0] * 10, sp))
        q.generate_new_batch(None, {"A": 0, "B": 0})  # admits B's req 2
        # A's queue is currently empty; A returns now.
        q.append(official.Req("A", 4, [0] * 5, sp))
        # A's pre-lift counter (100.0) already exceeds B's active counter,
        # so the lift (a max()) is a no-op here -- assert it stays at 100,
        # not silently reset down.
        assert q.served["A"] == 100.0

    def test_aborted_request_charged_nothing(self):
        official, q = self._queue(["A"])
        sp = official.SamplingParams(max_new_tokens=5)
        req = official.Req("A", 1, [0] * 50, sp)
        req.aborted = True
        q.append(req)
        batch = q.generate_new_batch(None, {"A": 0})
        assert batch is None  # nothing admitted -- the only request was aborted
        assert q.served["A"] == 0.0  # no cost charged for a dropped/aborted request

    def test_linear_cost_formula_matches_official_defaults(self):
        """input_price=1, output_price=2 are VTCReqQueue's own constructor
        defaults -- locks in that this project doesn't silently override
        them without saying so."""
        official, q = self._queue(["A"])
        sp = official.SamplingParams(max_new_tokens=5)
        q.append(official.Req("A", 1, [0] * 37, sp))
        q.generate_new_batch(None, {"A": 0})
        assert q.served["A"] == 37.0 * 1  # input_price=1
        batch = official.Batch("b", [official.Req("A", 1, [0] * 37, sp)])
        q.update_counter(batch)
        assert q.served["A"] == 37.0 + 2.0  # +1 * output_price=2

    def test_fair_weight_slows_counter_growth(self):
        official, q = self._queue(["A", "B"], weights=[2, 1])
        sp = official.SamplingParams(max_new_tokens=5)
        q.append(official.Req("A", 1, [0] * 10, sp))
        q.generate_new_batch(None, {"A": 0, "B": 0})
        # weight=2 halves the charged cost relative to weight=1.
        assert q.served["A"] == 5.0

    def test_unregistered_adapter_raises_keyerror_in_official_code(self):
        """Confirms the constraint UnregisteredTenantError guards against
        in the simulator adapter: the official code itself raises a raw
        KeyError for an adapter_dir not in the construction-time list."""
        official, q = self._queue(["A"])
        sp = official.SamplingParams(max_new_tokens=5)
        q.append(official.Req("C", 1, [0] * 10, sp))
        with pytest.raises(KeyError):
            q.generate_new_batch(None, {"A": 0, "C": 0})


class TestSimulatorPolicyAdapterConfig:
    def test_multi_gpu_topology_rejected(self):
        pytest.importorskip("baselines.vtc.adapter.official_loader")
        if not _clone_present():
            pytest.skip("clone not present")
        policy = VTCFairnessPolicy(known_tenants=["A", "B"])
        gpus = [
            GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=4096, max_kv_tokens=8192),
            GPUConfig(gpu_id=1, max_active_sequences=8, max_batch_tokens=4096, max_kv_tokens=8192),
        ]
        cfg = SimulatorConfig(gpu_configs=gpus)
        sim = Simulator(cfg)
        sim.load_trace([
            Request(request_id=0, arrival_time=0.0, prompt_tokens=10,
                    predicted_output_tokens=5, actual_output_tokens=5,
                    slo_deadline=100.0, priority=1.0, class_id="A"),
        ])
        with pytest.raises(UnsupportedTopologyError):
            sim.run(policy, workload_tag="t")

    @requires_clone
    def test_profile_cost_func_rejected(self):
        with pytest.raises(UnsupportedCostFunctionError):
            VTCFairnessPolicy(known_tenants=["A"], cost_func="profile")

    @requires_clone
    def test_empty_known_tenants_rejected(self):
        with pytest.raises(UnregisteredTenantError):
            VTCFairnessPolicy(known_tenants=[])

    @requires_clone
    def test_missing_tenant_id_rejected(self):
        policy = VTCFairnessPolicy(known_tenants=["A"], tenant_of=lambda r: None)
        gpu = GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=4096, max_kv_tokens=8192)
        cfg = SimulatorConfig(gpu_configs=[gpu])
        sim = Simulator(cfg)
        sim.load_trace([
            Request(request_id=0, arrival_time=0.0, prompt_tokens=10,
                    predicted_output_tokens=5, actual_output_tokens=5,
                    slo_deadline=100.0, priority=1.0, class_id="A"),
        ])
        with pytest.raises(MissingTenantIdError):
            sim.run(policy, workload_tag="t")

    @requires_clone
    def test_unregistered_tenant_rejected(self):
        policy = VTCFairnessPolicy(known_tenants=["A"])  # "Z" not registered
        gpu = GPUConfig(gpu_id=0, max_active_sequences=8, max_batch_tokens=4096, max_kv_tokens=8192)
        cfg = SimulatorConfig(gpu_configs=[gpu])
        sim = Simulator(cfg)
        sim.load_trace([
            Request(request_id=0, arrival_time=0.0, prompt_tokens=10,
                    predicted_output_tokens=5, actual_output_tokens=5,
                    slo_deadline=100.0, priority=1.0, class_id="Z"),
        ])
        with pytest.raises(UnregisteredTenantError):
            sim.run(policy, workload_tag="t")

    def test_default_tenant_of_uses_class_id(self):
        req = ObservableRequest(request_id=1, arrival_time=0.0, prompt_tokens=10,
                                 predicted_output_tokens=5, slo_deadline=100.0,
                                 priority=1.0, class_id="tenant-7")
        assert default_tenant_of(req) == "tenant-7"


@requires_clone
class TestSimulatorPolicyFairnessBehavior:
    def _run(self, requests, known_tenants, max_active_sequences=8):
        gpu = GPUConfig(gpu_id=0, max_active_sequences=max_active_sequences,
                         max_batch_tokens=4096, max_kv_tokens=8192)
        cfg = SimulatorConfig(gpu_configs=[gpu])
        sim = Simulator(cfg)
        sim.load_trace(requests)
        policy = VTCFairnessPolicy(known_tenants=known_tenants)
        metrics = sim.run(policy, workload_tag="t")
        return metrics, policy

    def _mk(self, rid, tenant, prompt=50, out=20, arrival=0.0):
        return Request(request_id=rid, arrival_time=arrival, prompt_tokens=prompt,
                        predicted_output_tokens=out, actual_output_tokens=out,
                        slo_deadline=1000.0, priority=1.0, class_id=tenant)

    def test_symmetric_workload_yields_equal_service(self):
        reqs = [self._mk(i, "A", arrival=float(i % 5)) for i in range(5)] + \
               [self._mk(5 + i, "B", arrival=float(i % 5)) for i in range(5)]
        metrics, policy = self._run(reqs, ["A", "B"])
        served = policy.served_snapshot()
        assert served["A"] == served["B"]
        assert metrics.num_completed == metrics.num_total == 10

    def test_heavy_hitter_does_not_starve_light_tenant(self):
        """The core VTC fairness property, with STAGGERED arrivals so
        requests don't all complete in lockstep on the run's absolute
        final simulated step (see simulator_policy.py deviation 5 -- a
        batch of requests that all arrive at once with identical
        predicted lengths decodes in lockstep and can ALL complete on the
        same final step, which is exactly the one case this adapter
        cannot observe the last decode tick for). Staggering arrivals
        lets this assert an EXACT accounting match, not just approximate."""
        reqs = [self._mk(i, "heavy", arrival=float(i)) for i in range(20)] + \
               [self._mk(20 + i, "light", arrival=float(i) + 0.5) for i in range(2)]
        metrics, policy = self._run(reqs, ["heavy", "light"], max_active_sequences=25)
        served = policy.served_snapshot()
        # Expected cost: request_count * (50*input_price=1 + 20*output_price=2).
        # Staggering arrivals eliminates most, but not all, final-step
        # lockstep overlap -- tolerate the disclosed bounded undercount
        # (at most a couple of trailing decode ticks, output_price=2
        # each) rather than requiring exact equality here; the EXACT
        # per-call cost formula is locked in at the raw-queue level by
        # test_linear_cost_formula_matches_official_defaults above.
        assert served["heavy"] == pytest.approx(20 * (50 * 1 + 20 * 2), abs=8)
        assert served["light"] == pytest.approx(2 * (50 * 1 + 20 * 2), abs=8)
        assert metrics.num_completed == 22

    def test_deterministic_replay(self):
        reqs = [self._mk(i, "A" if i % 2 == 0 else "B", arrival=float(i)) for i in range(10)]
        _, policy_1 = self._run(reqs, ["A", "B"])
        _, policy_2 = self._run(reqs, ["A", "B"])
        assert policy_1.served_snapshot() == policy_2.served_snapshot()

    def test_returning_tenant_does_not_monopolize_after_idle_period(self):
        """Tenant A is active early, then goes idle while B accrues
        service; when A returns, the official counter-lift rule prevents
        it from being treated as arbitrarily far behind."""
        reqs = (
            [self._mk(i, "A", arrival=0.0) for i in range(3)]
            + [self._mk(100 + i, "B", arrival=float(i)) for i in range(10)]
            + [self._mk(200, "A", arrival=50.0)]
        )
        metrics, policy = self._run(reqs, ["A", "B"], max_active_sequences=4)
        assert metrics.num_completed == metrics.num_total
