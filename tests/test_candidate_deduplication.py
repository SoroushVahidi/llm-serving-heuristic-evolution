"""Tests for candidate deduplication."""
from llmserveopt.llm_generation.diversity import deduplicate_candidates


def _make_record(name: str, cand: dict = None, cid: str = None) -> dict:
    if cand is None:
        cand = {
            "name": name,
            "tie_breaker": "arrival_order",
            "default": {"request_score": {"const": 1.0}},
        }
    return {
        "candidate": cand,
        "metadata": {"candidate_id": cid or name, "verification_ok": True},
    }


CAND_A = {"name": "a", "tie_breaker": "arrival_order", "default": {"request_score": {"const": 1.0}}}
CAND_B = {"name": "b", "tie_breaker": "earliest_deadline", "default": {"request_score": {"const": 2.0}}}
CAND_A_COPY = dict(CAND_A)  # same content


def test_dedup_no_duplicates():
    records = [_make_record("a", CAND_A), _make_record("b", CAND_B)]
    unique, removed = deduplicate_candidates(records)
    assert len(unique) == 2
    assert len(removed) == 0


def test_dedup_one_duplicate():
    records = [_make_record("a", CAND_A, "c001"), _make_record("a2", CAND_A_COPY, "c002")]
    unique, removed = deduplicate_candidates(records)
    assert len(unique) == 1
    assert len(removed) == 1


def test_dedup_keeps_first():
    records = [_make_record("a", CAND_A, "c001"), _make_record("a2", CAND_A_COPY, "c002")]
    unique, _ = deduplicate_candidates(records)
    assert unique[0]["metadata"]["candidate_id"] == "c001"


def test_dedup_three_duplicates():
    records = [
        _make_record("a", CAND_A, "c001"),
        _make_record("a2", CAND_A_COPY, "c002"),
        _make_record("a3", CAND_A_COPY, "c003"),
    ]
    unique, removed = deduplicate_candidates(records)
    assert len(unique) == 1
    assert len(removed) == 2


def test_dedup_empty_list():
    unique, removed = deduplicate_candidates([])
    assert unique == []
    assert removed == []


def test_dedup_single_record():
    records = [_make_record("a", CAND_A)]
    unique, removed = deduplicate_candidates(records)
    assert len(unique) == 1
    assert len(removed) == 0


def test_dedup_key_order_invariant():
    cand1 = {"b": 2, "a": 1}
    cand2 = {"a": 1, "b": 2}
    records = [_make_record("x", cand1, "c1"), _make_record("y", cand2, "c2")]
    unique, removed = deduplicate_candidates(records)
    # Different key order → same canonical JSON → duplicate
    assert len(unique) == 1
    assert len(removed) == 1


def test_dedup_different_constants_not_duplicate():
    cand1 = {"name": "a", "tie_breaker": "arrival_order", "default": {"request_score": {"const": 1.0}}}
    cand2 = {"name": "a", "tie_breaker": "arrival_order", "default": {"request_score": {"const": 2.0}}}
    records = [_make_record("a", cand1, "c1"), _make_record("a2", cand2, "c2")]
    unique, removed = deduplicate_candidates(records)
    assert len(unique) == 2
    assert len(removed) == 0


def test_dedup_verbose_no_crash():
    records = [_make_record("a", CAND_A, "c1"), _make_record("a2", CAND_A_COPY, "c2")]
    unique, removed = deduplicate_candidates(records, verbose=True)
    assert len(unique) == 1
