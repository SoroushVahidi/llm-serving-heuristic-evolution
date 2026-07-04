"""Tests for scripts/run_vllm_external_baseline_comparison.py.

No hosted API calls, no real vLLM server, no GPU needed: --mock exercises
the full external-admission-controller loop against a local deterministic
stub, and a small stdlib http.server fake (mirroring
tests/test_run_vllm_serving_baseline_pilot.py's pattern) validates the real
HTTP/SSE dispatch path end-to-end.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_vllm_external_baseline_comparison",
        ROOT / "scripts" / "run_vllm_external_baseline_comparison.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fake vLLM-OpenAI-compatible server (reused pattern)
# ---------------------------------------------------------------------------

class _FakeVllmHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/health" or self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "fake-model"}]}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"choices": [{"text": "hi ", "finish_reason": None}]},
            {"choices": [{"text": "there.", "finish_reason": "stop"}]},
            {"choices": [{"text": "", "finish_reason": None}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture()
def fake_vllm_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeVllmHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Policy name handling
# ---------------------------------------------------------------------------

def test_policy_aliases_normalize_correctly():
    mod = _load_module()
    assert mod.normalize_policy_name("vllm_default") == "vllm_direct"
    assert mod.normalize_policy_name("llf") == "least_laxity_first"
    assert mod.normalize_policy_name("estf") == "estimated_service_time_first"
    assert mod.normalize_policy_name("fifo") == "fifo"  # passthrough


def test_unsupported_policy_fails_clearly(tmp_path):
    mod = _load_module()
    result = mod.main(["--mock", "--policies", "fifo,not_a_real_policy", "--output-dir", str(tmp_path)])
    assert result == 2
    assert not (tmp_path / "requests.jsonl").exists()


def test_generated_heuristic_rejected_with_explanation(tmp_path):
    mod = _load_module()
    result = mod.main(["--mock", "--policies", "fifo,generated_heuristic", "--output-dir", str(tmp_path)])
    assert result == 8
    result3 = mod.main(["--mock", "--policies", "best_generated", "--output-dir", str(tmp_path)])
    assert result3 == 8


def test_selector_without_artifact_fails_clearly(tmp_path):
    """'selector' is conditionally wired: requesting it without
    --selector-artifact must fail before any benchmark work happens."""
    mod = _load_module()
    result = mod.main(["--mock", "--policies", "selector", "--output-dir", str(tmp_path)])
    assert result == 9
    assert not (tmp_path / "requests.jsonl").exists()


# ---------------------------------------------------------------------------
# Fixed request plan reuse across policies
# ---------------------------------------------------------------------------

def test_request_plan_identical_across_policies(tmp_path):
    mod = _load_module()
    plan_a = mod.build_request_plan(["short", "medium"], [64, 128], [1, 2], 2, seed=20260703)
    plan_b = mod.build_request_plan(["short", "medium"], [64, 128], [1, 2], 2, seed=20260703)
    assert [r.prompt_text for r in plan_a] == [r.prompt_text for r in plan_b]
    assert [r.priority for r in plan_a] == [r.priority for r in plan_b]
    assert [r.slo_slack_seconds for r in plan_a] == [r.slo_slack_seconds for r in plan_b]


def test_end_to_end_uses_same_plan_for_every_policy(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock", "--policies", "fifo,edf,vllm_direct",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    rows = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    for policy in ("fifo", "edf", "vllm_direct"):
        policy_rows = [r for r in rows if r["policy"] == policy]
        ids = sorted(r["request_id"] for r in policy_rows)
        assert ids == [0, 1]  # same 2 request_ids for every policy


# ---------------------------------------------------------------------------
# External admission-controller behavior (regression test for the
# gpu_state-snapshot bug: concurrency=1 with 2+ pending requests must admit
# both, not leak a phantom slot from policy-internal bookkeeping mutation)
# ---------------------------------------------------------------------------

def test_concurrency_one_admits_all_pending_requests_eventually(tmp_path):
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 3, seed=1)  # 3 requests, 1 slot
    results = mod.run_cell_for_policy("fifo", plan, concurrency=1, model="m", base_url=None, mock=True, timeout_s=30)
    assert len(results) == 3
    assert all(r.status == "success" for r in results)


def test_concurrency_one_admits_all_requests_for_every_wired_policy(tmp_path):
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 4, seed=1)
    for policy in mod.WIRED_POLICIES:
        results = mod.run_cell_for_policy(policy, plan, concurrency=1, model="m", base_url=None, mock=True, timeout_s=30)
        assert len(results) == 4, f"{policy} dropped requests at concurrency=1"
        assert all(r.status == "success" for r in results), f"{policy} had failures"


def test_no_policy_sees_actual_output_tokens():
    for fname in ("fifo.py", "edf.py", "shortest_output_first.py", "least_laxity_first.py", "estimated_service_time_first.py"):
        src = (ROOT / "src" / "llmserveopt" / "policies" / fname).read_text()
        # Look for attribute access, not the plain-English docstring
        # sentence "actual_output_tokens is never accessed" that some of
        # these files carry as documentation.
        assert ".actual_output_tokens" not in src, f"{fname} must never access .actual_output_tokens"


# ---------------------------------------------------------------------------
# Arrival-normalized weighted goodput: denominator over ALL arrivals
# ---------------------------------------------------------------------------

def test_arrival_normalized_wg_counts_failures_as_zero():
    mod = _load_module()
    rows = [
        {"policy": "p", "status": "success", "priority": 1.0, "slo_violated": False,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
        {"policy": "p", "status": "success", "priority": 1.0, "slo_violated": False,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
        {"policy": "p", "status": "error", "priority": 1.0, "slo_violated": None,
         "server_request_latency_seconds": None, "ttft_seconds": None,
         "total_wall_time_seconds": 1.0, "output_tokens": None},
        {"policy": "p", "status": "timeout", "priority": 1.0, "slo_violated": None,
         "server_request_latency_seconds": None, "ttft_seconds": None,
         "total_wall_time_seconds": 30.0, "output_tokens": None},
    ]
    m = mod.compute_policy_metrics(rows, policy_wall_clock_s=10.0)
    assert m["n_total"] == 4
    assert m["n_completed"] == 2
    assert m["n_failed"] == 2
    # conditional_WG among completed only = 1.0 (both met SLO)
    assert m["conditional_weighted_goodput"] == pytest.approx(1.0)
    # completion_fraction = 2/4 = 0.5, so arrival-normalized = 0.5 * 1.0 = 0.5
    assert m["completion_fraction"] == pytest.approx(0.5)
    assert m["arrival_normalized_weighted_goodput"] == pytest.approx(0.5)


def test_arrival_normalized_wg_all_completed_equals_conditional():
    mod = _load_module()
    rows = [
        {"policy": "p", "status": "success", "priority": 2.0, "slo_violated": False,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
        {"policy": "p", "status": "success", "priority": 1.0, "slo_violated": True,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
    ]
    m = mod.compute_policy_metrics(rows, policy_wall_clock_s=10.0)
    assert m["completion_fraction"] == pytest.approx(1.0)
    assert m["arrival_normalized_weighted_goodput"] == pytest.approx(m["conditional_weighted_goodput"])
    # priority-weighted: (2*1 + 1*0) / 3 = 0.6667
    assert m["conditional_weighted_goodput"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Timeout / error handling
# ---------------------------------------------------------------------------

def test_run_cell_for_policy_records_dispatch_errors_not_crash():
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 2, seed=1)
    # base_url=None with mock=False forces query_vllm_completion to fail
    # (no server reachable), exercising the error path without crashing.
    results = mod.run_cell_for_policy(
        "fifo", plan, concurrency=1, model="m", base_url="http://127.0.0.1:1", mock=False, timeout_s=2,
    )
    assert len(results) == 2
    assert all(r.status in ("error", "timeout") for r in results)
    assert all(r.error_type is not None for r in results)


# ---------------------------------------------------------------------------
# Warm-up excluded from metrics
# ---------------------------------------------------------------------------

def test_warmup_writes_separate_files_and_excluded_from_requests_jsonl(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock", "--warmup",
        "--policies", "fifo",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    assert (tmp_path / "warmup_requests.jsonl").exists()
    assert (tmp_path / "warmup_summary.md").exists()

    warmup_rows = [json.loads(l) for l in (tmp_path / "warmup_requests.jsonl").read_text().strip().splitlines()]
    assert len(warmup_rows) == 2
    assert all(r["request_id"] < 0 for r in warmup_rows)  # negative IDs, never collide with real plan

    requests_rows = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert all(r["request_id"] >= 0 for r in requests_rows)  # no warmup rows leaked in

    summary = json.loads((tmp_path / "summary.json").read_text())
    for policy_metrics in summary["per_policy"].values():
        assert policy_metrics["n_total"] == 1  # only the real 1-request plan, not +2 warmup


def test_no_warmup_flag_skips_warmup_files(tmp_path):
    mod = _load_module()
    mod.main([
        "--mock", "--policies", "fifo",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert not (tmp_path / "warmup_requests.jsonl").exists()


# ---------------------------------------------------------------------------
# Corrected-objective selector artifact: loading, validation, wiring
# ---------------------------------------------------------------------------

def _write_valid_selector_artifact(tmp_path: Path, subdir: str = "artifact") -> Path:
    """Build a tiny, real (not mocked) PerPolicyRegressionAnwgSelector, fit on
    synthetic rows, and persist it with a valid corrected-objective manifest
    -- mirrors scripts/persist_corrected_selector_artifact.py's contract
    without depending on the real 18MB repo artifact."""
    pytest.importorskip("sklearn")
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    from llmserveopt.selector.features import FEATURE_NAMES
    from llmserveopt.selector.models import PerPolicyRegressionAnwgSelector

    rows = []
    for i in range(40):
        row = {f"feat_{name}": float((i * 3 + idx) % 7) for idx, name in enumerate(FEATURE_NAMES)}
        for p in SELECTOR_CANDIDATES:
            row[f"completion_{p}"] = 1.0
            row[f"reward_{p}"] = 0.5 + 0.01 * ((i + hash(p)) % 10)
        rows.append(row)

    sel = PerPolicyRegressionAnwgSelector(n_estimators=5, max_depth=3, random_state=0)
    sel.fit(rows)

    out_dir = tmp_path / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "regression_anwg_selector.joblib"
    sel.save(str(artifact_path))
    manifest = {
        "artifact_type": "selector",
        "selector_name": "regression_anwg",
        "selector_class": "PerPolicyRegressionAnwgSelector",
        "objective_definition": {"name": "arrival_normalized_wg"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return artifact_path


def test_selector_artifact_missing_manifest_rejected(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    (artifact_path.parent / "manifest.json").unlink()
    with pytest.raises(mod.SelectorArtifactError, match="manifest"):
        mod.load_and_validate_selector_artifact(artifact_path)


def test_selector_artifact_wrong_objective_rejected(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    manifest_path = artifact_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["objective_definition"]["name"] = "completed_request_quality"  # pre-correction metric
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(mod.SelectorArtifactError, match="arrival_normalized_wg"):
        mod.load_and_validate_selector_artifact(artifact_path)


def test_real_pre_correction_artifacts_rejected():
    """Every actual pre-Phase-2B.14 *.joblib on disk must be rejected: none
    of them ship a manifest.json (the validation contract postdates them)."""
    mod = _load_module()
    stale_paths = [
        ROOT / "results/phase2a2_selector_dataset/smoke_selector_model/model_random_forest.joblib",
        ROOT / "results/phase2a4_2b4_final_eval/selector_models/random_forest/model.joblib",
    ]
    for p in stale_paths:
        if not p.exists():
            continue  # environment without these large result files checked out
        with pytest.raises(mod.SelectorArtifactError):
            mod.load_and_validate_selector_artifact(p)


def test_selector_artifact_missing_file_rejected(tmp_path):
    mod = _load_module()
    with pytest.raises(mod.SelectorArtifactError, match="not found"):
        mod.load_and_validate_selector_artifact(tmp_path / "does_not_exist.joblib")


def test_selector_artifact_loads_and_runs_end_to_end(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path, subdir="artifact")
    out_dir = tmp_path / "run"
    result = mod.main([
        "--mock", "--policies", "fifo,edf,selector",
        "--selector-artifact", str(artifact_path),
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(out_dir),
    ])
    assert result == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert "selector" in summary["per_policy"]
    assert summary["per_policy"]["selector"]["n_completed"] > 0

    # Our method appears in aggregate_by_policy.csv
    agg = (out_dir / "aggregate_by_policy.csv").read_text()
    assert "\nselector," in agg or agg.startswith("selector,")

    # selector_chosen_policy is recorded and is always a real candidate policy
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    rows = [json.loads(l) for l in (out_dir / "requests.jsonl").read_text().strip().splitlines()]
    selector_rows = [r for r in rows if r["policy"] == "selector"]
    assert len(selector_rows) > 0
    for r in selector_rows:
        assert r["selector_chosen_policy"] in SELECTOR_CANDIDATES
    non_selector_rows = [r for r in rows if r["policy"] != "selector"]
    assert all(r["selector_chosen_policy"] is None for r in non_selector_rows)


def test_selector_choose_subpolicy_is_feature_only(tmp_path):
    """The selector's decision must come only from extract_features() output
    (online-observable state), never from hindsight fields like actual
    output length or completion status."""
    mod = _load_module()
    from llmserveopt.core.types import Request

    artifact_path = _write_valid_selector_artifact(tmp_path)
    selector, _ = mod.load_and_validate_selector_artifact(artifact_path)

    waiting_requests = [
        Request(
            request_id=i, arrival_time=0.0, prompt_tokens=100,
            predicted_output_tokens=64, actual_output_tokens=64,
            slo_deadline=10.0, priority=1.0, class_id="default",
        )
        for i in range(3)
    ]
    chosen = mod.selector_choose_subpolicy(
        selector, waiting_requests=waiting_requests, now=0.5,
        active_sequence_count=1, concurrency=2, recent_violation_rate=0.0,
    )
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    assert chosen in SELECTOR_CANDIDATES

    # Swapping in wildly different actual_output_tokens must not change the
    # decision -- the feature adapter never reads it (extract_features()
    # only reads prompt_tokens/predicted_output_tokens/slo_deadline/
    # priority/class_id). Request is frozen, so build a fresh list.
    mutated_requests = [
        Request(
            request_id=r.request_id, arrival_time=r.arrival_time,
            prompt_tokens=r.prompt_tokens, predicted_output_tokens=r.predicted_output_tokens,
            actual_output_tokens=999999.0, slo_deadline=r.slo_deadline,
            priority=r.priority, class_id=r.class_id,
        )
        for r in waiting_requests
    ]
    chosen_after_mutation = mod.selector_choose_subpolicy(
        selector, waiting_requests=mutated_requests, now=0.5,
        active_sequence_count=1, concurrency=2, recent_violation_rate=0.0,
    )
    assert chosen_after_mutation == chosen


def test_require_our_method_without_selector_in_policies_fails(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    result = mod.main([
        "--mock", "--policies", "fifo,edf", "--require-our-method",
        "--selector-artifact", str(artifact_path),
        "--output-dir", str(tmp_path / "run"),
    ])
    assert result == 9


def test_require_our_method_fails_before_benchmark_starts_on_bad_artifact(tmp_path):
    mod = _load_module()
    bad_path = tmp_path / "nonexistent.joblib"
    out_dir = tmp_path / "run"
    result = mod.main([
        "--mock", "--policies", "fifo,selector", "--require-our-method",
        "--selector-artifact", str(bad_path),
        "--output-dir", str(out_dir),
    ])
    assert result == 9
    assert not out_dir.exists() or not (out_dir / "requests.jsonl").exists()


def test_require_our_method_succeeds_with_valid_artifact(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    out_dir = tmp_path / "run"
    result = mod.main([
        "--mock", "--policies", "fifo,selector", "--require-our-method",
        "--selector-artifact", str(artifact_path),
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(out_dir),
    ])
    assert result == 0
    assert (out_dir / "requests.jsonl").exists()


# ---------------------------------------------------------------------------
# Multi-regime request plans
# ---------------------------------------------------------------------------

def test_legacy_plan_call_preserves_pure_burst_arrival():
    """Omitting `regimes` must reproduce the exact pre-regime-support plan:
    every row arrival_time == 0.0, regime label 'steady_moderate', using
    DEFAULT_SLO_CLASSES -- this is the already-committed tiny-pilot's plan
    shape, and must not silently change under it."""
    mod = _load_module()
    plan = mod.build_request_plan(["short", "medium"], [64, 128], [1, 2], 2, seed=20260703)
    assert all(row.arrival_time == 0.0 for row in plan)
    assert all(row.regime == "steady_moderate" for row in plan)


def test_explicit_steady_moderate_regime_spreads_arrivals():
    mod = _load_module()
    plan = mod.build_request_plan(
        ["short"], [64], [1], 10, seed=1, regimes=["steady_moderate"],
    )
    arrival_times = [row.arrival_time for row in plan]
    assert any(t > 0.0 for t in arrival_times), "steady_moderate should spread arrivals over time"
    assert sorted(arrival_times) == arrival_times, "arrival times within a cell should be sorted"
    assert all(0.0 <= t <= mod.STEADY_ARRIVAL_WINDOW_S for t in arrival_times)


def test_bursty_and_overloaded_regimes_are_pure_burst():
    mod = _load_module()
    plan = mod.build_request_plan(
        ["short"], [64], [1], 5, seed=1, regimes=["bursty_tight", "overloaded_mixed_priority"],
    )
    assert all(row.arrival_time == 0.0 for row in plan)
    regimes_seen = {row.regime for row in plan}
    assert regimes_seen == {"bursty_tight", "overloaded_mixed_priority"}


def test_regime_slo_classes_differ_from_each_other():
    mod = _load_module()
    for regime, classes in mod.REGIME_SLO_CLASSES.items():
        assert classes, f"{regime} has no SLO classes"
    slacks = {
        regime: sorted(c.slo_slack for c in classes)
        for regime, classes in mod.REGIME_SLO_CLASSES.items()
    }
    # bursty_tight and overloaded_mixed_priority must be tighter than steady_moderate
    assert max(slacks["bursty_tight"]) < max(slacks["steady_moderate"])
    assert max(slacks["overloaded_mixed_priority"]) < max(slacks["steady_moderate"])


def test_unknown_regime_raises():
    mod = _load_module()
    with pytest.raises(ValueError, match="Unknown regime"):
        mod.build_request_plan(["short"], [64], [1], 2, seed=1, regimes=["not_a_real_regime"])


def test_cli_rejects_unknown_arrival_regime(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock", "--policies", "fifo", "--arrival-regimes", "not_a_real_regime",
        "--output-dir", str(tmp_path),
    ])
    assert result == 2


def test_multi_regime_end_to_end_writes_regime_field(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock", "--policies", "fifo,edf",
        "--arrival-regimes", "steady_moderate,bursty_tight",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    rows = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    regimes_seen = {r["regime"] for r in rows}
    assert regimes_seen == {"steady_moderate", "bursty_tight"}
    assert (tmp_path / "aggregate_by_policy_and_regime.csv").exists()


# ---------------------------------------------------------------------------
# Dropped-request recording (regression test: a policy that legitimately
# never admits a request must not silently vanish it from requests.jsonl)
# ---------------------------------------------------------------------------

def test_dropped_request_is_recorded_not_silently_lost():
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 1, seed=1)
    # Force an unmeetable deadline deterministically (no timing flakiness):
    # slo_deadline = arrival_time(0) + slack, so a deeply negative slack
    # guarantees negative laxity regardless of wall-clock timing.
    impossible_row = dataclasses.replace(plan[0], slo_slack_seconds=-1000.0)
    results = mod.run_cell_for_policy(
        "scorpio_style_slo_guard", [impossible_row], concurrency=1,
        model="m", base_url=None, mock=True, timeout_s=5,
    )
    assert len(results) == 1, "the request must appear in results, not vanish"
    assert results[0].status == "dropped"
    assert results[0].slo_violated is True
    assert results[0].request_id == impossible_row.request_id


def test_dropped_requests_counted_in_policy_metrics_as_failed():
    mod = _load_module()
    rows = [
        {"policy": "p", "status": "success", "priority": 1.0, "slo_violated": False,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
        {"policy": "p", "status": "dropped", "priority": 1.0, "slo_violated": True,
         "server_request_latency_seconds": None, "ttft_seconds": None,
         "total_wall_time_seconds": None, "output_tokens": None},
    ]
    m = mod.compute_policy_metrics(rows, policy_wall_clock_s=10.0)
    assert m["n_total"] == 2
    assert m["n_completed"] == 1
    assert m["n_failed"] == 1  # dropped counts as failed
    # arrival-normalized WG denominator includes the dropped request as zero credit
    assert m["arrival_normalized_weighted_goodput"] == pytest.approx(0.5)


def test_dropped_request_never_calls_real_network():
    """The dropped-row path must not attempt dispatch -- confirms this is a
    pure admission-refusal, not a masked network failure. A supported policy
    that deliberately declines a doomed request is labeled
    PolicyDeclinedAdmission (intentional load-shed), NOT PolicyNeverAdmitted
    (reserved for a missing adapter, which cannot happen post-preflight)."""
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 1, seed=1)
    impossible_row = dataclasses.replace(plan[0], slo_slack_seconds=-1000.0)
    # base_url=None + mock=False would normally raise on dispatch; since the
    # request is dropped before ever being admitted, no dispatch happens and
    # no exception propagates.
    results = mod.run_cell_for_policy(
        "scorpio_style_slo_guard", [impossible_row], concurrency=1,
        model="m", base_url=None, mock=False, timeout_s=5,
    )
    assert len(results) == 1
    assert results[0].status == "dropped"
    # scorpio_style_slo_guard IS a supported/constructible policy; a decline is
    # an intentional load-shed, not a missing-adapter PolicyNeverAdmitted.
    assert results[0].error_type == "PolicyDeclinedAdmission"


# ---------------------------------------------------------------------------
# Decision divergence: Kendall tau, cell-level comparison, examples
# ---------------------------------------------------------------------------

def test_kendall_tau_identical_orders():
    mod = _load_module()
    tau = mod._kendall_tau([1, 2, 3], [1, 2, 3], {1, 2, 3})
    assert tau == pytest.approx(1.0)


def test_kendall_tau_fully_reversed_orders():
    mod = _load_module()
    tau = mod._kendall_tau([1, 2, 3], [3, 2, 1], {1, 2, 3})
    assert tau == pytest.approx(-1.0)


def test_kendall_tau_needs_at_least_two_common_ids():
    mod = _load_module()
    assert mod._kendall_tau([1], [1], {1}) is None
    assert mod._kendall_tau([], [], set()) is None


def _divergence_row(policy, request_id, admission_time_s, slo_violated, **overrides):
    base = {
        "policy": policy, "request_id": request_id, "regime": "steady_moderate",
        "prompt_bucket": "short", "target_output_tokens": 64, "concurrency_level": 1,
        "admission_time_s": admission_time_s, "slo_violated": slo_violated, "status": "success",
    }
    base.update(overrides)
    return base


def test_compute_decision_divergence_detects_slo_outcome_change():
    mod = _load_module()
    all_rows = [
        _divergence_row("selector", 1, 0.0, True),   # selector: violated
        _divergence_row("selector", 2, 0.1, False),
        _divergence_row("fifo", 1, 0.0, False),        # fifo: same request, NOT violated
        _divergence_row("fifo", 2, 0.1, False),
    ]
    divergence_rows, example_rows = mod.compute_decision_divergence(all_rows, baselines=("fifo",))
    assert len(divergence_rows) == 1
    assert divergence_rows[0]["n_slo_outcome_changed"] == 1
    assert divergence_rows[0]["kendall_tau"] == pytest.approx(1.0)  # same admission order
    assert len(example_rows) == 1
    assert example_rows[0]["request_id"] == 1


def test_compute_decision_divergence_no_selector_returns_empty():
    mod = _load_module()
    all_rows = [_divergence_row("fifo", 1, 0.0, False)]
    divergence_rows, example_rows = mod.compute_decision_divergence(all_rows, baselines=("fifo",))
    assert divergence_rows == []
    assert example_rows == []


def test_write_decision_divergence_outputs_handles_no_examples(tmp_path):
    mod = _load_module()
    mod.write_decision_divergence_outputs(tmp_path, [], [])
    assert (tmp_path / "decision_divergence.csv").exists()
    md = (tmp_path / "selector_vs_baselines_examples.md").read_text()
    assert "No SLO-outcome divergence found" in md


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def test_bootstrap_ci_basic_shape():
    mod = _load_module()
    all_rows = []
    for rid in range(20):
        all_rows.append({"policy": "fifo", "request_id": rid, "priority": 1.0,
                          "status": "success", "slo_violated": rid % 5 == 0})
        all_rows.append({"policy": "selector", "request_id": rid, "priority": 1.0,
                          "status": "success", "slo_violated": rid % 10 == 0})
    ci_rows = mod.compute_bootstrap_ci(all_rows, ["fifo", "selector"], n_boot=200, seed=1)
    by_policy = {r["policy"]: r for r in ci_rows}
    assert "fifo" in by_policy and "selector" in by_policy
    assert "selector_minus_fifo" in by_policy
    for r in ci_rows:
        assert r["ci_low_2.5pct"] <= r["ci_high_97.5pct"]
    # selector violates less often (1/10 vs 1/5) -> selector should score higher
    assert by_policy["selector_minus_fifo"]["point_estimate_wg"] > 0


def test_bootstrap_ci_empty_rows_returns_empty():
    mod = _load_module()
    assert mod.compute_bootstrap_ci([], [], n_boot=10) == []


# ---------------------------------------------------------------------------
# End-to-end: decision-divergence-report + bootstrap-ci flags wired into main()
# ---------------------------------------------------------------------------

def test_decision_divergence_and_bootstrap_flags_produce_populated_reports(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path, subdir="artifact")
    out_dir = tmp_path / "run"
    result = mod.main([
        "--mock", "--policies", "fifo,edf,selector",
        "--selector-artifact", str(artifact_path),
        "--arrival-regimes", "steady_moderate,bursty_tight",
        "--decision-divergence-report", "--bootstrap-ci",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(out_dir),
    ])
    assert result == 0
    assert (out_dir / "decision_divergence.csv").exists()
    assert (out_dir / "selector_vs_baselines_examples.md").exists()
    assert (out_dir / "bootstrap_confidence_intervals.csv").exists()
    divergence_content = (out_dir / "decision_divergence.csv").read_text()
    assert "kendall_tau" in divergence_content
    ci_content = (out_dir / "bootstrap_confidence_intervals.csv").read_text()
    assert "selector_minus_fifo" in ci_content


def test_no_decision_divergence_flag_writes_empty_placeholder(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock", "--policies", "fifo,edf",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    assert (tmp_path / "decision_divergence.csv").exists()
    assert "Not computed for this run" in (tmp_path / "selector_vs_baselines_examples.md").read_text()
    assert not (tmp_path / "bootstrap_confidence_intervals.csv").exists()


# ---------------------------------------------------------------------------
# GPU memory capture (best-effort, must never raise)
# ---------------------------------------------------------------------------

def test_capture_gpu_mem_never_raises_without_nvidia_smi(tmp_path, monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("PATH", "/nonexistent")  # nvidia-smi won't be found
    out_path = tmp_path / "gpu_mem.txt"
    mod.capture_gpu_mem(out_path)  # must not raise
    assert out_path.exists()


# ---------------------------------------------------------------------------
# Real HTTP/SSE dispatch against a fake server (not real vLLM)
# ---------------------------------------------------------------------------

def test_dispatch_against_fake_server_parses_usage_and_text(fake_vllm_server):
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 1, seed=1)
    out = mod._dispatch(plan[0], model="fake-model", base_url=fake_vllm_server, mock=False, timeout_s=10)
    assert out["text"] == "hi there."
    assert out["finish_reason"] == "stop"
    assert out["output_tokens"] == 2.0
    assert out["prompt_tokens"] == 10.0
    assert out["ttft_seconds"] is not None


def test_end_to_end_against_fake_server(tmp_path, fake_vllm_server):
    mod = _load_module()
    result = mod.main([
        "--allow-live-server", "--server-url", fake_vllm_server,
        "--policies", "fifo,edf",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1,2", "--requests-per-cell", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    assert (tmp_path / "server_status.json").exists()
    server_status = json.loads((tmp_path / "server_status.json").read_text())
    assert "data" in server_status
    rows = [json.loads(l) for l in (tmp_path / "requests.jsonl").read_text().strip().splitlines()]
    assert all(r["status"] == "success" for r in rows)


# ---------------------------------------------------------------------------
# No hosted provider SDKs, no secrets
# ---------------------------------------------------------------------------

def test_no_hosted_provider_sdks_imported(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, {str(ROOT / "scripts")!r})
import run_vllm_external_baseline_comparison as mod
mod.main(["--mock", "--policies", "fifo", "--output-dir", {str(tmp_path)!r}])
for forbidden in ("cohere", "google.genai", "openai", "azure"):
    assert forbidden not in sys.modules, f"unexpectedly imported {{forbidden}}"
print("OK")
""",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_no_secrets_written_to_outputs(tmp_path, monkeypatch):
    secret = "sk-SHOULD-NEVER-APPEAR-EXTBASE-12345"
    monkeypatch.setenv("COHERE_API_KEY", secret)
    mod = _load_module()
    mod.main([
        "--mock", "--policies", "fifo,edf",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert secret not in f.read_text(errors="ignore"), f"secret leaked into {f}"


# ---------------------------------------------------------------------------
# Required output files present
# ---------------------------------------------------------------------------

REQUIRED_FILES = (
    "request_plan.jsonl", "requests.jsonl", "summary.json", "summary.md",
    "aggregate_by_policy.csv", "aggregate_by_policy_and_regime.csv", "aggregate_by_concurrency.csv",
    "aggregate_by_target_output_tokens.csv", "aggregate_by_prompt_bucket.csv",
    "decision_divergence.csv", "selector_vs_baselines_examples.md",
    "manifest.json", "run_config.json", "reproducibility.md", "errors.jsonl",
)


def test_all_required_output_files_present(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--mock", "--policies", "fifo,edf",
        "--prompt-buckets", "short,medium", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(tmp_path),
    ])
    assert result == 0
    for fname in REQUIRED_FILES:
        assert (tmp_path / fname).exists(), f"missing {fname}"


# ---------------------------------------------------------------------------
# Selector action-space: every selector-emittable label is dispatchable, and
# the preflight aborts BEFORE any live request on an unsupported label.
# ---------------------------------------------------------------------------

def test_selector_dispatchable_set_covers_full_selector_output_range():
    """The selector's entire output range (SELECTOR_CANDIDATES) must be a
    subset of what the harness can dispatch (SELECTOR_DISPATCHABLE); otherwise
    a selector run could route to a policy the harness cannot execute."""
    mod = _load_module()
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    assert set(SELECTOR_CANDIDATES) <= set(mod.SELECTOR_DISPATCHABLE)


def test_every_dispatchable_policy_is_constructible_and_selects_without_error():
    """Every label the selector may emit must be make_policy-constructible and
    able to run select_action() on a realistic observable state -- this is the
    Option-1 guarantee that the harness can execute the whole action space."""
    mod = _load_module()
    from llmserveopt.core.types import (
        ObservableGPUState, ObservableRequest, ObservableState, Request,
    )
    reqs = [
        Request(request_id=i, arrival_time=0.0, prompt_tokens=100,
                predicted_output_tokens=64, actual_output_tokens=64,
                slo_deadline=10.0, priority=1.0, class_id="default")
        for i in range(3)
    ]
    waiting = [ObservableRequest.from_request(r) for r in reqs]
    gpu = ObservableGPUState(
        gpu_id=0, max_active_sequences=2, max_batch_tokens=10**9, max_kv_tokens=10**9,
        active_request_ids=[], active_requests_info=[], current_kv_tokens=0,
        tokens_decoded_per_request={},
    )
    state = ObservableState(time=0.0, waiting_queue=list(waiting), gpu_states=[gpu],
                            completed_count=0, step=0)
    for name in mod.SELECTOR_DISPATCHABLE:
        assert mod._policy_constructible(name), f"{name} not constructible"
        policy = mod.make_policy(name)
        action = policy.select_action(state)  # must not raise
        assert action is not None


def test_preflight_passes_for_valid_selector_over_plan(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    selector, _ = mod.load_and_validate_selector_artifact(artifact_path)
    plan = mod.build_request_plan(["short", "medium"], [64, 128], [1, 2], 2, seed=1)
    report = mod.preflight_selector_action_space(selector, plan, [1, 2])
    assert report["ok"] is True
    assert report["labels_unsupported_static"] == []
    assert report["labels_unsupported_dynamic"] == []
    assert report["n_cells_enumerated"] > 0
    # every emitted label is dispatchable
    for label in report["labels_emitted_over_plan"]:
        assert label in mod.SELECTOR_DISPATCHABLE


def test_preflight_aborts_when_selector_emits_unsupported_label(tmp_path):
    """If a selector could emit a label the harness cannot dispatch, the
    preflight must raise before any live request is sent."""
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    selector, _ = mod.load_and_validate_selector_artifact(artifact_path)
    plan = mod.build_request_plan(["short"], [64], [1], 2, seed=1)

    # Shrink the harness's dispatchable set so a normally-supported emitted
    # label becomes "unsupported", exercising the dynamic-unsupported abort.
    original = mod.SELECTOR_DISPATCHABLE
    try:
        mod.SELECTOR_DISPATCHABLE = ("fifo",)  # deliberately too small
        with pytest.raises(mod.SelectorActionSpaceError):
            mod.preflight_selector_action_space(selector, plan, [1])
    finally:
        mod.SELECTOR_DISPATCHABLE = original


def test_selector_action_space_error_is_selector_artifact_error():
    mod = _load_module()
    assert issubclass(mod.SelectorActionSpaceError, mod.SelectorArtifactError)


def test_main_writes_preflight_report_and_runs_with_valid_selector(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    out_dir = tmp_path / "run"
    result = mod.main([
        "--mock", "--policies", "fifo,selector", "--require-our-method",
        "--selector-artifact", str(artifact_path),
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(out_dir),
    ])
    assert result == 0
    preflight = json.loads((out_dir / "selector_action_space_preflight.json").read_text())
    assert preflight["ok"] is True
    assert preflight["labels_unsupported_static"] == []
    # run_config.json records the preflight result too
    cfg = json.loads((out_dir / "run_config.json").read_text())
    assert cfg["selector_action_space_preflight"]["ok"] is True


def test_main_aborts_before_live_request_when_preflight_fails(tmp_path, monkeypatch):
    """--require-our-method must fail before any request when the selector's
    action space is not fully dispatchable. No requests.jsonl is produced."""
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    out_dir = tmp_path / "run"
    monkeypatch.setattr(mod, "SELECTOR_DISPATCHABLE", ("fifo",))  # too small on purpose
    result = mod.main([
        "--mock", "--policies", "fifo,selector", "--require-our-method",
        "--selector-artifact", str(artifact_path),
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--output-dir", str(out_dir),
    ])
    assert result == 9
    assert not (out_dir / "requests.jsonl").exists()
    preflight = json.loads((out_dir / "selector_action_space_preflight.json").read_text())
    assert preflight["ok"] is False


# ---------------------------------------------------------------------------
# Load-shed vs missing-adapter taxonomy (Part E)
# ---------------------------------------------------------------------------

def test_declined_admission_labeled_distinctly_from_never_admitted():
    """A constructible policy that deliberately admits nothing is a
    PolicyDeclinedAdmission (intentional load-shed), never a
    PolicyNeverAdmitted (missing adapter)."""
    mod = _load_module()
    plan = mod.build_request_plan(["short"], [64], [1], 1, seed=1)
    impossible = dataclasses.replace(plan[0], slo_slack_seconds=-1000.0)
    results = mod.run_cell_for_policy(
        "scorpio_style_slo_guard", [impossible], concurrency=1,
        model="m", base_url=None, mock=True, timeout_s=5,
    )
    assert len(results) == 1
    assert results[0].status == "dropped"
    assert results[0].error_type == "PolicyDeclinedAdmission"
    assert results[0].error_type != "PolicyNeverAdmitted"


def test_metrics_report_declined_and_never_admitted_counts():
    mod = _load_module()
    rows = [
        {"policy": "p", "status": "success", "priority": 1.0, "slo_violated": False,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0, "error_type": None},
        {"policy": "p", "status": "dropped", "priority": 1.0, "slo_violated": True,
         "server_request_latency_seconds": None, "ttft_seconds": None,
         "total_wall_time_seconds": None, "output_tokens": None,
         "error_type": "PolicyDeclinedAdmission"},
    ]
    m = mod.compute_policy_metrics(rows, policy_wall_clock_s=10.0)
    assert m["n_dropped"] == 1
    assert m["n_declined_admission"] == 1
    assert m["n_never_admitted"] == 0
    assert m["arrival_normalized_weighted_goodput"] == pytest.approx(0.5)


def test_no_never_admitted_when_selector_runs_supported_action_space(tmp_path):
    """End-to-end mock selector run: because the preflight guarantees the whole
    action space is dispatchable, no request may be dropped with
    PolicyNeverAdmitted (the missing-adapter failure mode)."""
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    out_dir = tmp_path / "run"
    result = mod.main([
        "--mock", "--policies", "fifo,edf,selector",
        "--selector-artifact", str(artifact_path),
        "--arrival-regimes", "steady_moderate,bursty_tight",
        "--prompt-buckets", "short,medium", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--output-dir", str(out_dir),
    ])
    assert result == 0
    rows = [json.loads(l) for l in (out_dir / "requests.jsonl").read_text().strip().splitlines()]
    never_admitted = [r for r in rows if r.get("error_type") == "PolicyNeverAdmitted"]
    assert never_admitted == [], "no request may be dropped for a missing adapter post-preflight"
