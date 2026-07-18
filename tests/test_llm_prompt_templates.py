"""Tests for prompt template completeness and correctness."""
import pytest
from llmserveopt.llm_generation.prompt_templates import (
    build_generation_messages,
    build_repair_messages,
)
from llmserveopt.heuristics.dsl_schema import (
    FORBIDDEN_SUBSTRINGS, ALLOWED_TIE_BREAKERS,
)


@pytest.fixture
def gen_messages():
    return build_generation_messages()


@pytest.fixture
def full_prompt_text(gen_messages):
    return " ".join(m["content"] for m in gen_messages).lower()


def test_prompt_includes_objective(full_prompt_text):
    assert "priority" in full_prompt_text
    assert "slo goodput" in full_prompt_text or "priority_weighted_slo_goodput" in full_prompt_text


def test_prompt_includes_allowed_variables(full_prompt_text):
    # At least a sample of allowed variables should appear
    for v in ["req.prompt_tokens", "req.deadline_urgency", "sys.queue_length", "batch.size"]:
        assert v in full_prompt_text, f"Expected '{v}' in prompt"


def test_prompt_includes_forbidden_variables(full_prompt_text):
    for s in FORBIDDEN_SUBSTRINGS:
        assert s in full_prompt_text, f"Expected forbidden substring '{s}' mentioned in prompt"


def test_prompt_requires_json_only(full_prompt_text):
    assert "json" in full_prompt_text
    assert "no markdown" in full_prompt_text or "no explanation" in full_prompt_text


def test_prompt_says_no_runtime_llm_calls(full_prompt_text):
    assert "runtime" in full_prompt_text or "offline" in full_prompt_text


def test_prompt_has_system_and_user_roles(gen_messages):
    roles = [m["role"] for m in gen_messages]
    assert "system" in roles
    assert "user" in roles


def test_prompt_includes_tie_breakers(full_prompt_text):
    for tb in ALLOWED_TIE_BREAKERS:
        assert tb in full_prompt_text, f"Expected tie_breaker '{tb}' in prompt"


def test_prompt_includes_operations(full_prompt_text):
    for op in ["add", "sub", "mul", "div_safe", "weighted_sum", "if_then_else"]:
        assert op in full_prompt_text, f"Expected op '{op}' in prompt"


def test_repair_prompt_includes_errors():
    errors = [("UNKNOWN_VARIABLE", "var 'req.actual_output_tokens' not allowed")]
    messages = build_repair_messages(
        {"name": "bad", "tie_breaker": "arrival_order", "default": {"request_score": {"const": 1.0}}},
        errors,
    )
    full_text = " ".join(m["content"] for m in messages)
    assert "UNKNOWN_VARIABLE" in full_text
    assert "actual_output_tokens" in full_text


def test_repair_prompt_has_json_instruction():
    messages = build_repair_messages({}, [("SCHEMA_MISSING_FIELD", "name missing")])
    full_text = " ".join(m["content"] for m in messages)
    assert "json" in full_text.lower()


def test_repair_prompt_includes_candidate():
    candidate = {"name": "test_repair", "tie_breaker": "arrival_order"}
    messages = build_repair_messages(candidate, [("SCHEMA_MISSING_FIELD", "default missing")])
    full_text = " ".join(m["content"] for m in messages)
    assert "test_repair" in full_text


def test_repair_prompt_has_system_role():
    messages = build_repair_messages({}, [])
    roles = [m["role"] for m in messages]
    assert "system" in roles
