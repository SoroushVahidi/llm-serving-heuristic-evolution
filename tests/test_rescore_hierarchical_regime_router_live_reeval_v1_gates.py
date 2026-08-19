"""Regression tests for the formal G1-G9 rescoring of the completed live
closed-loop re-evaluation
(`scripts/rescore_hierarchical_regime_router_live_reeval_v1_gates.py`).

This script closes a methodology gap found by a repository-wide audit
(2026-08-19): the live-reeval run script printed its own verdict from a
hand-rolled if/else rather than the frozen `evaluate_all_gates`/
`compute_verdict` gate evaluator. These tests pin:

- the rescoring path is read-only over the completed result artifacts (it
  never invokes the simulator, Stage-1/Stage-2 fitting, or the live
  harness, and never opens either source result file for writing);
- it uses ONLY the canonical gate evaluator, never a hand-written
  substitute verdict rule;
- Family-B-dependent / persistence-gap-dependent gate components are
  reported NOT_EVALUABLE, never silently assumed passing;
- the computed verdict is deterministic and the source result artifacts
  are left byte-identical.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rescore_hierarchical_regime_router_live_reeval_v1_gates.py"
LIVE_REEVAL_RESULTS = REPO_ROOT / "experiments/hierarchical_regime_router_live_reeval_v1/live_reeval_results.json"
TEST_EVAL_RESULTS = REPO_ROOT / "experiments/hierarchical_regime_router_v1_test_evaluation/test_evaluation_results.json"

FORBIDDEN_IMPORT_NAMES = {
    # simulator / live-execution surface -- rescoring must never touch these
    "Simulator",
    "SimulatorConfig",
    "run_live_scenario",
    "LiveHierarchicalRouterPolicy",
    "LiveRunResult",
    # model fitting / raw workload surface
    "Stage1Router",
    "fit_all_stage2_selectors",
    "load_scenario_level_dataset",
    "case_fairness_vs_size_v2",
    "case_prefill_decode_ttft_contention",
    "case_kv_pressure_reserve_contention_v2",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree() -> ast.Module:
    return ast.parse(SCRIPT_PATH.read_text())


def _load_module():
    spec = importlib.util.spec_from_file_location("rescore_hier_live_reeval_gates", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec.loader.exec_module(module)
    return module


def test_script_exists():
    assert SCRIPT_PATH.is_file()


def test_rescore_script_never_imports_simulator_or_fitting_surface():
    """AST-level guard: the rescoring path must be pure post-hoc analysis
    over already-persisted results, never a re-execution of the live
    harness or model fitting."""
    tree = _tree()
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add((alias.asname or alias.name).split(".")[0])
    forbidden_hit = imported_names & FORBIDDEN_IMPORT_NAMES
    assert not forbidden_hit, f"rescoring script must not import simulator/fitting surface, found: {forbidden_hit}"


def test_rescore_script_only_imports_canonical_gate_functions_from_gates_module():
    """The script must call the frozen evaluator, not reimplement gate
    logic -- pin that it imports compute_verdict/evaluate_all_gates/
    load_gates_config from hierarchical_router_gates_v1 specifically."""
    tree = _tree()
    gate_module_imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("hierarchical_router_gates_v1"):
            for alias in node.names:
                gate_module_imports.add(alias.name)
    assert {"compute_verdict", "evaluate_all_gates", "load_gates_config"} <= gate_module_imports


def test_rescore_script_never_opens_source_result_files_for_writing():
    """The script may open its own new OUTPUT_PATH for writing, but must
    never write to either source result artifact it reads."""
    tree = _tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            args_src = ast.dump(node)
            if "w" in [a.value for a in node.args if isinstance(a, ast.Constant)]:
                # this open() call is a write; it must target OUTPUT_PATH, not
                # LIVE_REEVAL_RESULTS or TEST_EVAL_RESULTS
                assert "LIVE_REEVAL_RESULTS" not in args_src
                assert "TEST_EVAL_RESULTS" not in args_src


def test_no_hand_rolled_verdict_substitute_regression_guard():
    """Regression guard against reintroducing a hand-rolled if/else verdict
    (the exact bug this script exists to correct): every assignment to a
    variable literally named `verdict` in this script's source must come
    from calling the canonical `compute_verdict`, never from a string
    literal or a manual conditional expression."""
    tree = _tree()
    verdict_assigns = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "verdict" for t in node.targets)
    ]
    assert verdict_assigns, "expected at least one `verdict = ...` assignment"
    for node in verdict_assigns:
        assert isinstance(node.value, ast.Call), "verdict must be assigned from a function call, not a literal/conditional"
        assert isinstance(node.value.func, ast.Name) and node.value.func.id == "compute_verdict", (
            f"verdict must be assigned from compute_verdict(...), found call to "
            f"{ast.dump(node.value.func)}"
        )


def test_all_nine_gates_present_in_output_using_real_persisted_artifacts():
    if not LIVE_REEVAL_RESULTS.exists():
        import pytest
        pytest.skip("live_reeval_results.json not present in this checkout")
    module = _load_module()
    live = json.loads(LIVE_REEVAL_RESULTS.read_text())
    sibling = json.loads(TEST_EVAL_RESULTS.read_text()) if TEST_EVAL_RESULTS.exists() else {}
    metrics, _notes = module.build_metrics(live, sibling)
    config = module.load_gates_config()
    gates = module.evaluate_all_gates(metrics, config)
    assert set(gates.keys()) == {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"}


def test_persistence_gap_gates_are_not_evaluable_not_manufactured():
    """G4, G7, and G9(a) require a per-regime live-ANWG breakdown that the
    persisted live_reeval_results.json does not carry. This must surface as
    NOT_EVALUABLE (passed=None), never as a manufactured pass/fail value."""
    if not LIVE_REEVAL_RESULTS.exists():
        import pytest
        pytest.skip("live_reeval_results.json not present in this checkout")
    module = _load_module()
    live = json.loads(LIVE_REEVAL_RESULTS.read_text())
    sibling = json.loads(TEST_EVAL_RESULTS.read_text()) if TEST_EVAL_RESULTS.exists() else {}
    metrics, _notes = module.build_metrics(live, sibling)
    assert metrics["stage2_preservation_fraction_by_regime"] is None
    assert metrics["multi_regime_benefit_count"] is None
    assert metrics["family_c_held_out_delta_anwg"] is None

    config = module.load_gates_config()
    gates = module.evaluate_all_gates(metrics, config)
    assert gates["G4"].passed is None
    assert gates["G7"].passed is None
    assert gates["G9"].passed is None


def test_g5_uses_delta_fixed_not_delta_method():
    """G5's frozen metric definition is 'hierarchy minus best global fixed',
    which is primary_metrics.delta_fixed, not delta_method (hierarchy minus
    the old majority-vote approximation)."""
    if not LIVE_REEVAL_RESULTS.exists():
        import pytest
        pytest.skip("live_reeval_results.json not present in this checkout")
    module = _load_module()
    live = json.loads(LIVE_REEVAL_RESULTS.read_text())
    sibling = json.loads(TEST_EVAL_RESULTS.read_text()) if TEST_EVAL_RESULTS.exists() else {}
    metrics, _notes = module.build_metrics(live, sibling)
    assert metrics["mean_delta_anwg"] == live["primary_metrics"]["delta_fixed"]
    assert metrics["bootstrap_ci_lower"] == live["primary_metrics"]["delta_fixed_ci_90"][0]


def test_verdict_is_deterministic():
    if not LIVE_REEVAL_RESULTS.exists():
        import pytest
        pytest.skip("live_reeval_results.json not present in this checkout")
    module = _load_module()
    live = json.loads(LIVE_REEVAL_RESULTS.read_text())
    sibling = json.loads(TEST_EVAL_RESULTS.read_text()) if TEST_EVAL_RESULTS.exists() else {}

    verdicts = set()
    for _ in range(3):
        metrics, _notes = module.build_metrics(live, sibling)
        config = module.load_gates_config()
        gates = module.evaluate_all_gates(metrics, config)
        blended_summary = sibling.get("blended_microcase_summary", {})
        verdict = module.compute_verdict(
            gates,
            blended_microcase_sample_too_small=bool(blended_summary.get("sample_too_small", True)),
            test_sample_insufficient_for_g5_ci=False,
        )
        verdicts.add(verdict)
    assert len(verdicts) == 1


def test_formal_verdict_is_no_go_family_and_agrees_with_ad_hoc_verdict():
    """Pins the actual formal verdict computed from the frozen completed
    result: HIERARCHICAL_ROUTER_NO_GO, agreeing directionally with the
    ad-hoc script's own LIVE_REEVAL_CONFIRMS_NO_GO."""
    if not LIVE_REEVAL_RESULTS.exists():
        import pytest
        pytest.skip("live_reeval_results.json not present in this checkout")
    module = _load_module()
    live = json.loads(LIVE_REEVAL_RESULTS.read_text())
    sibling = json.loads(TEST_EVAL_RESULTS.read_text()) if TEST_EVAL_RESULTS.exists() else {}
    metrics, _notes = module.build_metrics(live, sibling)
    config = module.load_gates_config()
    gates = module.evaluate_all_gates(metrics, config)
    blended_summary = sibling.get("blended_microcase_summary", {})
    verdict = module.compute_verdict(
        gates,
        blended_microcase_sample_too_small=bool(blended_summary.get("sample_too_small", True)),
        test_sample_insufficient_for_g5_ci=False,
    )
    assert verdict == "HIERARCHICAL_ROUTER_NO_GO"
    assert live["live_re_evaluation_verdict"] == "LIVE_REEVAL_CONFIRMS_NO_GO"


def test_running_main_leaves_source_result_artifacts_byte_identical(tmp_path):
    if not LIVE_REEVAL_RESULTS.exists():
        import pytest
        pytest.skip("live_reeval_results.json not present in this checkout")
    module = _load_module()
    before_live = _sha256(LIVE_REEVAL_RESULTS)
    before_sibling = _sha256(TEST_EVAL_RESULTS) if TEST_EVAL_RESULTS.exists() else None
    before_output = _sha256(module.OUTPUT_PATH) if module.OUTPUT_PATH.exists() else None

    module.main(output_path=tmp_path / "gate_rescoring_v1.json")

    after_live = _sha256(LIVE_REEVAL_RESULTS)
    after_sibling = _sha256(TEST_EVAL_RESULTS) if TEST_EVAL_RESULTS.exists() else None
    after_output = _sha256(module.OUTPUT_PATH) if module.OUTPUT_PATH.exists() else None
    assert before_live == after_live, "rescoring must never modify the source live-reeval result"
    assert before_sibling == after_sibling, "rescoring must never modify the sibling TEST-evaluation result"
    assert before_output == after_output, "rescoring must never modify the tracked canonical gate_rescoring_v1.json"
