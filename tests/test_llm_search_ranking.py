"""Tests for search-phase ranking (by validation performance)."""
import math
import tempfile
from pathlib import Path
import pytest
from llmserveopt.llm_generation.multi_regime_evaluation import AggregatedCandidateResult
from llmserveopt.llm_generation.search_ranking import (
    build_search_summary_md,
    rank_search_results,
    save_per_regime_csv,
    save_search_ranking_csv,
)


def _make_agg(
    name, source="heuristic",
    train_wg=0.8, val_wg=0.8, gap=0.0,
    worst_wg=0.7, worst_name="some_regime",
    val_vr=0.1, val_ttft=0.2,
    beats_fixed=0, beats_ss=0, beats_estf=0,
    n_train=4, n_val=3,
    per_regime=None,
    error=None,
) -> AggregatedCandidateResult:
    return AggregatedCandidateResult(
        name=name, source=source,
        train_mean_wg=train_wg, val_mean_wg=val_wg,
        overall_mean_wg=(train_wg + val_wg) / 2,
        train_val_gap=gap,
        worst_regime_wg=worst_wg, worst_regime_name=worst_name,
        train_violation_rate=0.1, val_violation_rate=val_vr,
        train_p95_ttft=0.2, val_p95_ttft=val_ttft,
        regimes_beating_best_fixed=beats_fixed,
        regimes_beating_slo_slack=beats_ss,
        regimes_beating_estf=beats_estf,
        n_train_regimes=n_train, n_val_regimes=n_val,
        per_regime=per_regime or {},
        error=error,
    )


def test_rank_orders_by_val_wg():
    a = _make_agg("a", val_wg=0.9)
    b = _make_agg("b", val_wg=0.5)
    c = _make_agg("c", val_wg=0.7)
    agg = {"a": a, "b": b, "c": c}
    ranked = rank_search_results(agg)
    assert ranked[0].name == "a"
    assert ranked[1].name == "c"
    assert ranked[2].name == "b"


def test_rank_tiebreak_val_violation_rate():
    a = _make_agg("a", val_wg=0.8, val_vr=0.2)
    b = _make_agg("b", val_wg=0.8, val_vr=0.1)
    ranked = rank_search_results({"a": a, "b": b})
    assert ranked[0].name == "b"


def test_rank_tiebreak_val_ttft():
    a = _make_agg("a", val_wg=0.8, val_vr=0.1, val_ttft=0.5)
    b = _make_agg("b", val_wg=0.8, val_vr=0.1, val_ttft=0.2)
    ranked = rank_search_results({"a": a, "b": b})
    assert ranked[0].name == "b"


def test_rank_nan_val_wg_last():
    good = _make_agg("good", val_wg=0.5)
    bad = _make_agg("bad", val_wg=float("nan"))
    ranked = rank_search_results({"good": good, "bad": bad})
    assert ranked[0].name == "good"
    assert ranked[-1].name == "bad"


def test_rank_source_filter_heuristic():
    h = _make_agg("h", source="heuristic")
    b = _make_agg("b", source="baseline")
    ranked = rank_search_results({"h": h, "b": b}, source_filter="heuristic")
    assert all(r.source == "heuristic" for r in ranked)


def test_rank_source_filter_baseline():
    h = _make_agg("h", source="heuristic")
    b = _make_agg("b", source="baseline")
    ranked = rank_search_results({"h": h, "b": b}, source_filter="baseline")
    assert all(r.source == "baseline" for r in ranked)


def test_rank_empty():
    assert rank_search_results({}) == []


def test_save_search_ranking_csv():
    r = _make_agg("test_cand")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ranking.csv"
        save_search_ranking_csv([r], path)
        assert path.exists()
        content = path.read_text()
        assert "rank" in content
        assert "val_mean_wg" in content
        assert "train_val_gap" in content


def test_save_per_regime_csv():
    r = _make_agg("test", per_regime={"r1": 0.8, "r2": 0.7})
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "by_regime.csv"
        save_per_regime_csv([r], ["r1", "r2"], path)
        assert path.exists()
        content = path.read_text()
        assert "r1" in content
        assert "r2" in content


def test_build_search_summary_md_has_ranking():
    h = _make_agg("my_heuristic", source="heuristic", val_wg=0.85)
    b = _make_agg("fifo", source="baseline", val_wg=0.75)
    md = build_search_summary_md(
        ranked_all=[h, b],
        ranked_heuristics=[h],
        ranked_baselines=[b],
        n_generated=5, n_verified=4, n_repaired=1, n_failed=1,
        n_duplicates=0,
        regime_names=["train_r", "val_r"],
        train_regime_names=["train_r"],
        val_regime_names=["val_r"],
    )
    assert "my_heuristic" in md
    assert "fifo" in md
    assert "oracle_srtf" in md


def test_build_search_summary_md_has_overfitting_section():
    # overfit candidate (big negative gap)
    h = _make_agg("overfit_cand", val_wg=0.7, train_wg=0.95, gap=-0.25)
    md = build_search_summary_md(
        ranked_all=[h], ranked_heuristics=[h], ranked_baselines=[],
        n_generated=1, n_verified=1, n_repaired=0, n_failed=0, n_duplicates=0,
        regime_names=["t"], train_regime_names=["t"], val_regime_names=[],
    )
    assert "overfit" in md.lower() or "gap" in md.lower()


def test_build_search_summary_md_rf_selector_note():
    md = build_search_summary_md(
        ranked_all=[], ranked_heuristics=[], ranked_baselines=[],
        n_generated=0, n_verified=0, n_repaired=0, n_failed=0, n_duplicates=0,
        regime_names=[], train_regime_names=[], val_regime_names=[],
    )
    assert "RF Selector" in md or "16-policy" in md
