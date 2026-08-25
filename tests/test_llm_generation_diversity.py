"""Tests for design target diversity controls."""
from llmserveopt.llm_generation.diversity import (
    DESIGN_TARGETS,
    DEFAULT_TARGET_CYCLE,
    build_targeted_messages,
)


def test_design_targets_non_empty():
    assert len(DESIGN_TARGETS) >= 5


def test_default_cycle_covers_all_targets():
    assert set(DEFAULT_TARGET_CYCLE) == set(DESIGN_TARGETS.keys())


def test_build_targeted_messages_no_target():
    msgs = build_targeted_messages()
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_targeted_messages_with_target():
    msgs = build_targeted_messages("slo_urgency")
    full = " ".join(m["content"] for m in msgs)
    assert "slo_urgency" in full
    assert "deadline" in full.lower()


def test_build_targeted_messages_kv_pressure():
    msgs = build_targeted_messages("kv_pressure")
    full = " ".join(m["content"] for m in msgs)
    assert "kv" in full.lower() or "cache" in full.lower()


def test_build_targeted_messages_throughput():
    msgs = build_targeted_messages("throughput_oriented")
    full = " ".join(m["content"] for m in msgs)
    assert "throughput" in full.lower()


def test_build_targeted_messages_balanced():
    msgs = build_targeted_messages("balanced")
    full = " ".join(m["content"] for m in msgs)
    assert "balanced" in full.lower() or "balance" in full.lower()


def test_build_targeted_messages_unknown_target_no_crash():
    msgs = build_targeted_messages("totally_unknown_target_xyz")
    assert isinstance(msgs, list)
    assert len(msgs) == 2


def test_targeted_messages_still_include_objective():
    for target in DESIGN_TARGETS:
        msgs = build_targeted_messages(target)
        full = " ".join(m["content"] for m in msgs).lower()
        assert "priority" in full
        assert "slo" in full
        assert "json" in full


def test_targeted_messages_still_mention_forbidden():
    msgs = build_targeted_messages("mixed_slo")
    full = " ".join(m["content"] for m in msgs)
    assert "actual_output" in full or "forbidden" in full.lower()


def test_target_descriptions_are_nonempty():
    for name, desc in DESIGN_TARGETS.items():
        assert len(desc) > 20, f"Description for '{name}' too short"


def test_all_seven_targets_present():
    expected = {
        "slo_urgency", "kv_pressure", "throughput_oriented",
        "prefill_heavy", "mixed_slo", "noisy_prediction_robust", "balanced",
    }
    assert expected == set(DESIGN_TARGETS.keys())
