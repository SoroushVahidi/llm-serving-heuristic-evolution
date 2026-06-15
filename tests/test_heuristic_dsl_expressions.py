"""Tests for the DSL expression evaluator."""
import math
import pytest
from llmserveopt.heuristics.expressions import evaluate_expression, ExpressionError

DUMMY_CTX = {
    "req.prompt_tokens": 128.0,
    "req.predicted_output_tokens": 64.0,
    "req.waiting_time": 1.5,
    "req.deadline_slack": 3.0,
    "req.deadline_urgency": 0.333,
    "req.priority_weight": 2.0,
    "req.estimated_prefill_cost": 128.0,
    "req.estimated_decode_cost": 64.0,
    "req.estimated_kv_cost": 192.0,
    "sys.queue_length": 10.0,
    "sys.kv_utilization": 0.6,
    "sys.free_sequence_ratio": 0.5,
    "batch.size": 2.0,
}


def test_const():
    assert evaluate_expression({"const": 5.0}, {}) == pytest.approx(5.0)


def test_const_zero():
    assert evaluate_expression({"const": 0}, {}) == pytest.approx(0.0)


def test_var_lookup():
    ctx = {"x": 42.0}
    assert evaluate_expression({"var": "x"}, ctx) == pytest.approx(42.0)


def test_var_missing_raises():
    with pytest.raises(ExpressionError, match="Unknown variable"):
        evaluate_expression({"var": "missing.var"}, {})


def test_add():
    expr = {"op": "add", "args": [{"const": 3.0}, {"const": 4.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(7.0)


def test_sub():
    expr = {"op": "sub", "args": [{"const": 10.0}, {"const": 3.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(7.0)


def test_mul():
    expr = {"op": "mul", "args": [{"const": 3.0}, {"const": 4.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(12.0)


def test_div_safe_normal():
    expr = {"op": "div_safe", "args": [{"const": 10.0}, {"const": 2.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(5.0)


def test_div_safe_zero_denominator():
    expr = {"op": "div_safe", "args": [{"const": 1.0}, {"const": 0.0}]}
    result = evaluate_expression(expr, {})
    assert math.isfinite(result)
    assert abs(result) > 1e6  # Should be large but finite


def test_min():
    expr = {"op": "min", "args": [{"const": 3.0}, {"const": 7.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(3.0)


def test_max():
    expr = {"op": "max", "args": [{"const": 3.0}, {"const": 7.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(7.0)


def test_neg():
    expr = {"op": "neg", "args": [{"const": 5.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(-5.0)


def test_abs_negative():
    expr = {"op": "abs", "args": [{"const": -7.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(7.0)


def test_clip():
    expr = {"op": "clip", "args": [{"const": 15.0}, {"const": 0.0}, {"const": 10.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(10.0)
    expr2 = {"op": "clip", "args": [{"const": -5.0}, {"const": 0.0}, {"const": 10.0}]}
    assert evaluate_expression(expr2, {}) == pytest.approx(0.0)


def test_sqrt_safe():
    expr = {"op": "sqrt_safe", "args": [{"const": 9.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(3.0)


def test_sqrt_safe_negative():
    expr = {"op": "sqrt_safe", "args": [{"const": -4.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(0.0)  # max(0, x)


def test_log1p_safe():
    expr = {"op": "log1p_safe", "args": [{"const": 0.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(0.0)


def test_log1p_safe_negative():
    expr = {"op": "log1p_safe", "args": [{"const": -100.0}]}
    assert evaluate_expression(expr, {}) == pytest.approx(0.0)  # max(0, x)


def test_weighted_sum():
    expr = {
        "op": "weighted_sum",
        "terms": [
            [{"const": 3.0}, 2.0],
            [{"const": 5.0}, 4.0],
        ],
    }
    assert evaluate_expression(expr, {}) == pytest.approx(3.0 * 2.0 + 5.0 * 4.0)


def test_if_then_else_true():
    expr = {
        "op": "if_then_else",
        "cond": {"const": 1.0},
        "then": {"const": 100.0},
        "else": {"const": -1.0},
    }
    assert evaluate_expression(expr, {}) == pytest.approx(100.0)


def test_if_then_else_false():
    expr = {
        "op": "if_then_else",
        "cond": {"const": -1.0},
        "then": {"const": 100.0},
        "else": {"const": -1.0},
    }
    assert evaluate_expression(expr, {}) == pytest.approx(-1.0)


def test_nested_expression():
    # sqrt_safe(add(3, mul(2, 2))) = sqrt(7) ≈ 2.6457...
    expr = {
        "op": "sqrt_safe",
        "args": [
            {
                "op": "add",
                "args": [
                    {"const": 3.0},
                    {"op": "mul", "args": [{"const": 2.0}, {"const": 2.0}]},
                ],
            }
        ],
    }
    assert evaluate_expression(expr, {}) == pytest.approx(math.sqrt(7.0))


def test_depth_limit():
    # Build a deeply nested add chain that exceeds max_depth=5
    expr = {"const": 1.0}
    for _ in range(10):
        expr = {"op": "add", "args": [expr, {"const": 0.0}]}
    with pytest.raises(ExpressionError, match="depth"):
        evaluate_expression(expr, {}, max_depth=5)


def test_unknown_op_raises():
    with pytest.raises(ExpressionError, match="Unknown operation"):
        evaluate_expression({"op": "banana"}, {})


def test_invalid_expr_type():
    with pytest.raises(ExpressionError):
        evaluate_expression("not a dict", {})


def test_non_finite_const():
    with pytest.raises(ExpressionError, match="(?i)non.?finite"):
        evaluate_expression({"const": float("inf")}, {})


def test_missing_op_and_const_and_var():
    with pytest.raises(ExpressionError):
        evaluate_expression({"something_else": 1}, {})
