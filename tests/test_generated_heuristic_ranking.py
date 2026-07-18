"""Tests for ranking logic."""
import csv
import tempfile
from pathlib import Path
from llmserveopt.llm_generation.evaluation import CandidateResult
from llmserveopt.llm_generation.ranking import (
    build_summary_md,
    rank_candidates,
    save_ranking_csv,
)


def _make_result(name, wg, vr=0.1, ttft=0.1, lat=0.5, tput=10.0, source="heuristic", error=None):
    return CandidateResult(
        name=name,
        source=source,
        policy_name=name,
        weighted_goodput=wg,
        priority_weighted_slo_goodput=wg,
        slo_violation_rate=vr,
        p95_ttft=ttft,
        p95_latency=lat,
        request_throughput=tput,
        num_completed=100,
        error=error,
    )


def test_rank_orders_by_goodput():
    a = _make_result("a", wg=0.9)
    b = _make_result("b", wg=0.5)
    c = _make_result("c", wg=0.7)
    ranked = rank_candidates([a, b, c])
    assert ranked[0].name == "a"
    assert ranked[1].name == "c"
    assert ranked[2].name == "b"


def test_rank_tiebreak_violation_rate():
    a = _make_result("a", wg=0.8, vr=0.2)
    b = _make_result("b", wg=0.8, vr=0.1)
    ranked = rank_candidates([a, b])
    assert ranked[0].name == "b"  # lower violation rate wins


def test_rank_tiebreak_ttft():
    a = _make_result("a", wg=0.8, vr=0.1, ttft=0.5)
    b = _make_result("b", wg=0.8, vr=0.1, ttft=0.2)
    ranked = rank_candidates([a, b])
    assert ranked[0].name == "b"


def test_rank_tiebreak_latency():
    a = _make_result("a", wg=0.8, vr=0.1, ttft=0.2, lat=1.0)
    b = _make_result("b", wg=0.8, vr=0.1, ttft=0.2, lat=0.5)
    ranked = rank_candidates([a, b])
    assert ranked[0].name == "b"


def test_rank_tiebreak_throughput():
    a = _make_result("a", wg=0.8, vr=0.1, ttft=0.2, lat=0.5, tput=5.0)
    b = _make_result("b", wg=0.8, vr=0.1, ttft=0.2, lat=0.5, tput=10.0)
    ranked = rank_candidates([a, b])
    assert ranked[0].name == "b"  # higher throughput wins


def test_rank_nan_goodput_last():
    good = _make_result("good", wg=0.5)
    bad = _make_result("bad", wg=float("nan"), error="failed")
    ranked = rank_candidates([bad, good])
    assert ranked[0].name == "good"
    assert ranked[-1].name == "bad"


def test_rank_empty():
    assert rank_candidates([]) == []


def test_rank_single():
    r = _make_result("only", wg=0.5)
    ranked = rank_candidates([r])
    assert len(ranked) == 1
    assert ranked[0].name == "only"


def test_save_ranking_csv_creates_file():
    results = [_make_result("a", 0.8), _make_result("b", 0.6)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ranking.csv"
        save_ranking_csv(results, path)
        assert path.exists()


def test_save_ranking_csv_has_correct_columns():
    results = [_make_result("a", 0.8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ranking.csv"
        save_ranking_csv(results, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert "rank" in row
            assert "priority_weighted_slo_goodput" in row
            assert "weighted_goodput" in row


def test_save_ranking_csv_nan_written_as_empty():
    results = [_make_result("bad", float("nan"), error="oops")]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ranking.csv"
        save_ranking_csv(results, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["priority_weighted_slo_goodput"] == ""


def test_build_summary_md_contains_all_candidates():
    results = [
        _make_result("heuristic_a", 0.9, source="heuristic"),
        _make_result("fifo", 0.7, source="baseline"),
    ]
    md = build_summary_md(results, n_generated=2, n_verified=1, n_repaired=1, n_failed=0)
    assert "heuristic_a" in md
    assert "fifo" in md


def test_build_summary_md_has_oracle_note():
    md = build_summary_md([], n_generated=0, n_verified=0, n_repaired=0, n_failed=0)
    assert "oracle_srtf" in md


def test_build_summary_md_has_rf_selector_note():
    md = build_summary_md([], n_generated=0, n_verified=0, n_repaired=0, n_failed=0)
    assert "RF Selector" in md or "rf selector" in md.lower()


def test_build_summary_md_generation_stats():
    md = build_summary_md([], n_generated=4, n_verified=3, n_repaired=1, n_failed=1)
    assert "4" in md
    assert "Generated" in md
