"""
Render comparison tables as LaTeX and plain text.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


DISPLAY_COLS = {
    "policy":                "Policy",
    "mean_latency":          "Mean Lat (s)",
    "p95_latency":           "P95 Lat (s)",
    "p99_latency":           "P99 Lat (s)",
    "mean_queuing_delay":    "Mean Q-Delay (s)",
    "slo_violation_rate":    "SLO Viol. Rate",
    "request_throughput":    "Req/s",
    "mean_gpu_utilization":  "GPU Util.",
    "mean_active_batch_size":"Avg Batch",
    "num_completed":         "Completed",
}


def format_summary(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Return a display-ready DataFrame with renamed columns."""
    if cols is None:
        cols = list(DISPLAY_COLS.keys())
    available = [c for c in cols if c in df.columns]
    out = df[available].copy()
    out = out.rename(columns={c: DISPLAY_COLS.get(c, c) for c in available})
    # Round floats
    for col in out.columns:
        if out[col].dtype == float:
            out[col] = out[col].round(4)
    return out


def to_latex(
    df: pd.DataFrame,
    path: Optional[Union[str, Path]] = None,
    caption: str = "Baseline policy comparison",
    label: str = "tab:baseline",
) -> str:
    tdf = format_summary(df)
    latex = tdf.to_latex(
        index=False,
        caption=caption,
        label=label,
        float_format="%.4f",
    )
    if path is not None:
        Path(path).write_text(latex)
    return latex


def to_markdown(
    df: pd.DataFrame,
    path: Optional[Union[str, Path]] = None,
) -> str:
    tdf = format_summary(df)
    md = tdf.to_markdown(index=False, floatfmt=".4f")
    if md is None:
        md = tdf.to_string(index=False)
    if path is not None:
        Path(path).write_text(md)
    return md
