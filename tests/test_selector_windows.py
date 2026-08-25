"""Tests for selector window construction."""

from llmserveopt.core.types import Request
from llmserveopt.selector.windows import make_windows


def _req(i: int, t: float = None) -> Request:
    return Request(
        request_id=i,
        arrival_time=t if t is not None else float(i),
        prompt_tokens=32,
        predicted_output_tokens=16,
        actual_output_tokens=16,
        slo_deadline=float(i) + 10.0,
        priority=1.0,
        class_id="medium",
    )


def _make_trace(n: int) -> list:
    return [_req(i, float(i) * 0.1) for i in range(n)]


# --- exact multiple ---

def test_exact_multiple_window_count():
    reqs = _make_trace(600)
    windows = make_windows(reqs, window_size=200)
    assert len(windows) == 3


def test_window_sizes_exact():
    reqs = _make_trace(600)
    windows = make_windows(reqs, window_size=200)
    for w in windows:
        assert w.num_requests == 200
        assert len(w.requests) == 200


def test_window_ids_monotone():
    reqs = _make_trace(400)
    windows = make_windows(reqs, window_size=200)
    ids = [w.window_id for w in windows]
    assert ids == list(range(len(windows)))


def test_window_indices_cover_trace():
    n = 600
    reqs = _make_trace(n)
    windows = make_windows(reqs, window_size=200)
    assert windows[0].start_request_index == 0
    assert windows[-1].end_request_index == n


# --- partial tail ---

def test_partial_tail_dropped_by_default_below_min():
    reqs = _make_trace(210)  # 200 + 10 tail
    windows = make_windows(reqs, window_size=200, min_partial=50)
    assert len(windows) == 1   # tail of 10 < min_partial=50: dropped


def test_partial_tail_kept_above_min():
    reqs = _make_trace(260)  # 200 + 60 tail
    windows = make_windows(reqs, window_size=200, min_partial=50)
    assert len(windows) == 2
    assert windows[1].num_requests == 60


def test_keep_partial_forced_true():
    reqs = _make_trace(205)
    windows = make_windows(reqs, window_size=200, keep_partial=True)
    assert len(windows) == 2
    assert windows[1].num_requests == 5


def test_keep_partial_forced_false():
    reqs = _make_trace(260)
    windows = make_windows(reqs, window_size=200, keep_partial=False)
    assert len(windows) == 1


# --- time ordering ---

def test_monotone_time_ordering():
    reqs = _make_trace(400)
    windows = make_windows(reqs, window_size=200)
    for w in windows:
        times = [r.arrival_time for r in w.requests]
        assert times == sorted(times)


def test_window_start_end_time():
    reqs = _make_trace(400)
    windows = make_windows(reqs, window_size=200)
    for w in windows:
        assert w.start_time == w.requests[0].arrival_time
        assert w.end_time == w.requests[-1].arrival_time


# --- determinism ---

def test_deterministic_window_ids():
    reqs = _make_trace(400)
    w1 = make_windows(reqs, window_size=200)
    w2 = make_windows(reqs, window_size=200)
    assert [w.window_id for w in w1] == [w.window_id for w in w2]
    for a, b in zip(w1, w2):
        assert a.start_request_index == b.start_request_index
        assert a.end_request_index == b.end_request_index


def test_trace_id_propagated():
    reqs = _make_trace(200)
    windows = make_windows(reqs, trace_id="my_trace", window_size=200)
    assert all(w.trace_id == "my_trace" for w in windows)


def test_empty_trace():
    windows = make_windows([], window_size=200)
    assert windows == []


def test_fewer_than_window_size_but_above_min():
    reqs = _make_trace(80)
    windows = make_windows(reqs, window_size=200, min_partial=50)
    assert len(windows) == 1
    assert windows[0].num_requests == 80
