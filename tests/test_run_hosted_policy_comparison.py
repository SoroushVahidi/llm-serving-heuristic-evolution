"""Tests for scripts/run_hosted_policy_comparison.py.

No hosted API calls anywhere in this file: --mock and --dry-run-cost-check
never touch a network, and cohere/google.genai SDK imports inside
run_cohere_api_calibration.py / run_gemini_real_llm_calibration.py are
lazy (only inside _build_client()), so importing this module never
requires those SDKs to be installed.

Sibling-script loading (vext/cohere_mod/gemini_mod, and the PROVIDER_CONFIG
dict built from them) is itself lazy: exec_module for those siblings only
runs on first actual use (a function call, or external attribute access via
the module-level __getattr__), never merely from importing this module. See
test_import_does_not_trigger_sibling_load / test_provider_config_lazy_load /
test_vext_attribute_lazy_load below.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_SIBLING_MODULE_NAMES = (
    "run_vllm_external_baseline_comparison",
    "run_cohere_api_calibration",
    "run_gemini_real_llm_calibration",
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_hosted_policy_comparison",
        ROOT / "scripts" / "run_hosted_policy_comparison.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _clean_sibling_modules_from_sys_modules():
    for name in _SIBLING_MODULE_NAMES:
        sys.modules.pop(name, None)
    yield
    for name in _SIBLING_MODULE_NAMES:
        sys.modules.pop(name, None)


def test_import_does_not_trigger_sibling_load():
    """Importing the module alone must not exec_module vext/cohere_mod/gemini_mod."""
    mod = _load_module()
    assert mod._SIBLINGS_LOADED is False
    for name in _SIBLING_MODULE_NAMES:
        assert name not in sys.modules


def test_provider_config_lazy_load():
    """`mod.PROVIDER_CONFIG` (external attribute access, no main() call) must
    still work via the module-level __getattr__, lazily loading siblings."""
    mod = _load_module()
    assert mod._SIBLINGS_LOADED is False
    pc = mod.PROVIDER_CONFIG
    assert mod._SIBLINGS_LOADED is True
    assert set(pc.keys()) == {"cohere", "gemini"}


def test_vext_attribute_lazy_load():
    mod = _load_module()
    assert mod._SIBLINGS_LOADED is False
    v = mod.vext
    assert mod._SIBLINGS_LOADED is True
    assert v.__name__ == "run_vllm_external_baseline_comparison"


def test_getattr_unknown_name_raises():
    mod = _load_module()
    with pytest.raises(AttributeError):
        mod.not_a_real_attribute


def test_main_still_triggers_lazy_load_and_runs(tmp_path):
    """The original runtime path (main()) must still work end-to-end."""
    mod = _load_module()
    assert mod._SIBLINGS_LOADED is False
    result = mod.main(["--provider", "cohere", "--dry-run-cost-check", "--output-dir", str(tmp_path)])
    assert result == 0
    assert mod._SIBLINGS_LOADED is True


def _write_valid_selector_artifact(tmp_path: Path, subdir: str = "artifact") -> Path:
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
        "artifact_type": "selector", "selector_name": "regression_anwg",
        "selector_class": "PerPolicyRegressionAnwgSelector",
        "objective_definition": {"name": "arrival_normalized_wg"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return artifact_path


# ---------------------------------------------------------------------------
# Provider gating
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", ["azure", "fireworks", "cloudrift", "openai"])
def test_unsupported_providers_rejected_clearly(tmp_path, provider):
    mod = _load_module()
    result = mod.main(["--provider", provider, "--dry-run-cost-check", "--output-dir", str(tmp_path)])
    assert result == 10


def test_missing_mode_flag_fails(tmp_path):
    mod = _load_module()
    result = mod.main(["--provider", "cohere", "--output-dir", str(tmp_path)])
    assert result == 2


def test_supported_providers_are_cohere_and_gemini():
    mod = _load_module()
    assert set(mod.PROVIDER_CONFIG.keys()) == {"cohere", "gemini"}
    assert mod.PROVIDER_CONFIG["cohere"]["default_model"] == "command-r7b-12-2024"
    assert mod.PROVIDER_CONFIG["gemini"]["default_model"] == "gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Dry-run cost cap enforcement (no network)
# ---------------------------------------------------------------------------

def test_dry_run_cost_check_passes_under_default_caps(tmp_path):
    mod = _load_module()
    result = mod.main(["--provider", "cohere", "--dry-run-cost-check", "--output-dir", str(tmp_path)])
    assert result == 0
    report = json.loads((tmp_path / "cost_report.json").read_text())
    assert report["cap_check_passed"] is True
    assert report["worst_case_estimated_cost_usd"] < 5.0
    assert not (tmp_path / "requests.jsonl").exists()  # no run happened


def test_dry_run_cost_check_fails_when_cost_cap_too_low(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--provider", "cohere", "--dry-run-cost-check",
        "--max-estimated-cost-usd", "0.0000001",
        "--output-dir", str(tmp_path),
    ])
    assert result == 4
    report = json.loads((tmp_path / "cost_report.json").read_text())
    assert report["cap_check_passed"] is False
    assert report["cap_violations"]


def test_dry_run_cost_check_fails_when_request_cap_too_low(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--provider", "gemini", "--dry-run-cost-check",
        "--max-total-requests", "1",
        "--output-dir", str(tmp_path),
    ])
    assert result == 4


def test_full_part_g_grid_matches_972_requests_and_passes_caps(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    result = mod.main([
        "--provider", "cohere", "--dry-run-cost-check",
        "--policies", "fifo,edf,least_laxity_first,estimated_service_time_first,shortest_output_first,selector",
        "--selector-artifact", str(artifact_path), "--require-our-method",
        "--output-dir", str(tmp_path / "run"),
    ])
    assert result == 0
    report = json.loads((tmp_path / "run" / "cost_report.json").read_text())
    assert report["planned_total_requests"] == 972


# ---------------------------------------------------------------------------
# Live calls require --allow-live-api explicitly
# ---------------------------------------------------------------------------

def test_mock_never_requires_allow_live_api(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--provider", "cohere", "--mock",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--arrival-regimes", "steady_moderate",
        "--policies", "fifo", "--output-dir", str(tmp_path),
    ])
    assert result == 0


# ---------------------------------------------------------------------------
# Selector artifact reuse: manifest verification / stale rejection
# ---------------------------------------------------------------------------

def test_selector_without_artifact_fails_clearly(tmp_path):
    mod = _load_module()
    result = mod.main(["--provider", "cohere", "--mock", "--policies", "selector", "--output-dir", str(tmp_path)])
    assert result == 9
    assert not (tmp_path / "requests.jsonl").exists()


def test_stale_selector_artifact_rejected(tmp_path):
    mod = _load_module()
    stale_path = ROOT / "results/phase2a4_2b4_final_eval/selector_models/random_forest/model.joblib"
    if not stale_path.exists():
        pytest.skip("stale artifact not present in this checkout")
    result = mod.main([
        "--provider", "cohere", "--mock", "--policies", "selector",
        "--selector-artifact", str(stale_path), "--output-dir", str(tmp_path),
    ])
    assert result == 9
    assert not (tmp_path / "requests.jsonl").exists()


def test_require_our_method_without_selector_in_policies_fails(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    result = mod.main([
        "--provider", "cohere", "--mock", "--policies", "fifo,edf",
        "--require-our-method", "--selector-artifact", str(artifact_path),
        "--output-dir", str(tmp_path / "run"),
    ])
    assert result == 9


def test_require_our_method_succeeds_with_valid_artifact(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    out_dir = tmp_path / "run"
    result = mod.main([
        "--provider", "cohere", "--mock", "--policies", "fifo,selector",
        "--require-our-method", "--selector-artifact", str(artifact_path),
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--arrival-regimes", "steady_moderate",
        "--output-dir", str(out_dir),
    ])
    assert result == 0
    assert (out_dir / "requests.jsonl").exists()


# ---------------------------------------------------------------------------
# Fail-fast
# ---------------------------------------------------------------------------

def test_fail_fast_flag_is_accepted_and_wired(tmp_path):
    mod = _load_module()
    result = mod.main([
        "--provider", "cohere", "--mock", "--fail-fast",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--arrival-regimes", "steady_moderate",
        "--policies", "fifo", "--output-dir", str(tmp_path),
    ])
    assert result == 0  # mock never fails, so fail-fast never triggers, but flag must not error


# ---------------------------------------------------------------------------
# No API key logging
# ---------------------------------------------------------------------------

def test_no_api_key_value_leaked_to_outputs(tmp_path, monkeypatch):
    secret = "sk-SHOULD-NEVER-APPEAR-HOSTED-98765"
    monkeypatch.setenv("COHERE_API_KEY", secret)
    monkeypatch.setenv("GOOGLE_API_KEY", secret)
    mod = _load_module()
    mod.main([
        "--provider", "cohere", "--mock",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "1",
        "--arrival-regimes", "steady_moderate",
        "--policies", "fifo,edf", "--output-dir", str(tmp_path),
    ])
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert secret not in f.read_text(errors="ignore"), f"secret leaked into {f}"
    cfg = json.loads((tmp_path / "run_config.json").read_text())
    assert cfg["api_key_present"] is True  # presence recorded, value never


def test_no_hosted_provider_sdks_imported_in_mock_mode(tmp_path):
    proc = subprocess.run(
        [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, {str(ROOT / "scripts")!r})
import run_hosted_policy_comparison as mod
mod.main(["--provider", "cohere", "--mock", "--policies", "fifo",
          "--prompt-buckets", "short", "--target-output-tokens-list", "64",
          "--concurrency-list", "1", "--requests-per-cell", "1",
          "--arrival-regimes", "steady_moderate",
          "--output-dir", {str(tmp_path)!r}])
for forbidden in ("cohere", "google.genai"):
    assert forbidden not in sys.modules, f"unexpectedly imported {{forbidden}} in --mock mode"
print("OK")
""",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# ---------------------------------------------------------------------------
# End-to-end mock run: required artifacts present
# ---------------------------------------------------------------------------

REQUIRED_FILES = (
    "request_plan.jsonl", "requests.jsonl", "summary.json", "summary.md",
    "aggregate_by_policy.csv", "aggregate_by_policy_and_regime.csv",
    "aggregate_by_concurrency.csv", "aggregate_by_target_output_tokens.csv",
    "aggregate_by_prompt_bucket.csv", "decision_divergence.csv",
    "selector_vs_baselines_examples.md", "manifest.json", "run_config.json",
    "reproducibility.md", "errors.jsonl", "cost_report.json",
)


def test_mock_end_to_end_writes_all_required_files(tmp_path):
    mod = _load_module()
    artifact_path = _write_valid_selector_artifact(tmp_path)
    out_dir = tmp_path / "run"
    result = mod.main([
        "--provider", "gemini", "--mock",
        "--policies", "fifo,edf,selector",
        "--selector-artifact", str(artifact_path),
        "--prompt-buckets", "short,medium", "--target-output-tokens-list", "64,128",
        "--concurrency-list", "1,2", "--requests-per-cell", "2",
        "--arrival-regimes", "steady_moderate,bursty_tight",
        "--decision-divergence-report", "--bootstrap-ci",
        "--output-dir", str(out_dir),
    ])
    assert result == 0
    for fname in REQUIRED_FILES:
        assert (out_dir / fname).exists(), f"missing {fname}"
    assert (out_dir / "bootstrap_confidence_intervals.csv").exists()


def test_mock_run_uses_identical_plan_across_policies(tmp_path):
    mod = _load_module()
    out_dir = tmp_path / "run"
    result = mod.main([
        "--provider", "cohere", "--mock", "--policies", "fifo,edf",
        "--prompt-buckets", "short", "--target-output-tokens-list", "64",
        "--concurrency-list", "1", "--requests-per-cell", "2",
        "--arrival-regimes", "steady_moderate",
        "--output-dir", str(out_dir),
    ])
    assert result == 0
    rows = [json.loads(l) for l in (out_dir / "requests.jsonl").read_text().strip().splitlines()]
    for policy in ("fifo", "edf"):
        ids = sorted(r["request_id"] for r in rows if r["policy"] == policy)
        assert ids == [0, 1]


def test_arrival_normalized_wg_denominator_includes_all_arrivals_hosted(tmp_path):
    """Reuses vext.compute_policy_metrics -- same corrected-objective
    contract applies to hosted results as to vLLM results."""
    mod = _load_module()
    rows = [
        {"policy": "p", "status": "success", "priority": 1.0, "slo_violated": False,
         "server_request_latency_seconds": 0.1, "ttft_seconds": 0.01,
         "total_wall_time_seconds": 0.1, "output_tokens": 10.0},
        {"policy": "p", "status": "dropped", "priority": 1.0, "slo_violated": True,
         "server_request_latency_seconds": None, "ttft_seconds": None,
         "total_wall_time_seconds": None, "output_tokens": None},
    ]
    m = mod.vext.compute_policy_metrics(rows, policy_wall_clock_s=10.0)
    assert m["n_total"] == 2
    assert m["arrival_normalized_weighted_goodput"] == pytest.approx(0.5)
