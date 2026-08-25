"""Regression test for the live re-evaluation analysis's use of the frozen
grouped bootstrap helper (group_resampled_bootstrap_ci).

This is a source-level (AST) test, not an execution test: the script's
main() requires the real telemetry/scenario datasets and runs the full
live-harness simulation, which is far too heavy for a unit test and is
scientifically out of scope here. This test only verifies the *call
interface* used against the frozen helper -- it deliberately does not read
or reason about any TEST-split scientific results.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from llmserveopt.policy_separation.hierarchical_router_evaluation_v1 import (
    group_resampled_bootstrap_ci,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_hierarchical_regime_router_live_reeval_v1.py"


def _bootstrap_ci_calls() -> list[ast.Call]:
    tree = ast.parse(SCRIPT_PATH.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "group_resampled_bootstrap_ci"
    ]


def test_frozen_helper_signature_is_df_hierarchy_fixed_n_boot_ci():
    """Pin the canonical interface so a future signature change is caught
    here rather than at a post-simulation crash."""
    params = list(inspect.signature(group_resampled_bootstrap_ci).parameters)
    assert params[:3] == ["df", "hierarchy", "fixed"]
    assert "n_boot" in params
    assert "n_resamples" not in params


def test_live_reeval_script_calls_bootstrap_helper_at_least_twice():
    calls = _bootstrap_ci_calls()
    assert len(calls) >= 2


def test_live_reeval_bootstrap_calls_use_supported_keyword_interface():
    for call in _bootstrap_ci_calls():
        kwarg_names = {kw.arg for kw in call.keywords}
        assert "n_resamples" not in kwarg_names, (
            "call uses unsupported keyword 'n_resamples'; frozen helper only accepts 'n_boot'"
        )
        assert "n_boot" in kwarg_names
        n_boot_node = next(kw.value for kw in call.keywords if kw.arg == "n_boot")
        assert isinstance(n_boot_node, ast.Constant) and n_boot_node.value == 5000

        assert "ci" in kwarg_names
        ci_node = next(kw.value for kw in call.keywords if kw.arg == "ci")
        assert isinstance(ci_node, ast.Constant) and ci_node.value == 0.90


def test_live_reeval_bootstrap_calls_pass_three_positional_args_grouped_by_frozen_test_df():
    """The frozen helper requires (df, hierarchy, fixed) positionally; `df`
    must carry the frozen TEST group_key so resampling stays grouped."""
    for call in _bootstrap_ci_calls():
        assert len(call.args) == 3, (
            "frozen helper requires 3 positional args (df, hierarchy, fixed); "
            f"got {len(call.args)}"
        )
        first_arg = call.args[0]
        assert isinstance(first_arg, ast.Name) and first_arg.id == "test_df", (
            "first positional arg must be the frozen TEST dataframe (test_df) "
            "so group_key-based grouped resampling is preserved"
        )
