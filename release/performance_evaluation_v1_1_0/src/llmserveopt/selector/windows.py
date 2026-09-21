"""
Non-overlapping request window construction for selector dataset building.

Windows partition the trace by request count.  Partial tail windows are kept
only when they have at least MIN_PARTIAL_WINDOW requests (default 50).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..core.types import Request

DEFAULT_WINDOW_SIZE: int = 200
MIN_PARTIAL_WINDOW: int = 50


@dataclass
class RequestWindow:
    """A non-overlapping slice of a trace."""
    window_id: int
    trace_id: str
    start_request_index: int
    end_request_index: int          # exclusive
    start_time: float
    end_time: float
    num_requests: int
    requests: List[Request]

    def __post_init__(self) -> None:
        assert self.num_requests == len(self.requests), (
            "num_requests must match len(requests)"
        )
        assert self.num_requests == self.end_request_index - self.start_request_index


def make_windows(
    requests: Sequence[Request],
    trace_id: str = "trace",
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_partial: int = MIN_PARTIAL_WINDOW,
    keep_partial: Optional[bool] = None,
) -> List[RequestWindow]:
    """Partition `requests` into non-overlapping windows of `window_size`.

    Parameters
    ----------
    requests : sequence of Request, must be sorted by arrival_time.
    trace_id : label for the trace (used in window metadata and dataset CSV).
    window_size : number of requests per window.
    min_partial : minimum size to keep a tail window that does not fill completely.
    keep_partial : if True, always keep the tail; if False, always drop it; if None
        (default), keep when len(tail) >= min_partial.
    """
    reqs = list(requests)
    if not reqs:
        return []

    windows: List[RequestWindow] = []
    n = len(reqs)
    w_id = 0

    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        chunk = reqs[start:end]
        size = len(chunk)

        is_partial = size < window_size
        if is_partial:
            if keep_partial is False:
                break
            if keep_partial is None and size < min_partial:
                break

        windows.append(RequestWindow(
            window_id=w_id,
            trace_id=trace_id,
            start_request_index=start,
            end_request_index=end,
            start_time=chunk[0].arrival_time,
            end_time=chunk[-1].arrival_time,
            num_requests=size,
            requests=chunk,
        ))
        w_id += 1

    return windows
