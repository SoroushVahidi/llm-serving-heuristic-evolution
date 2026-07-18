"""
BurstGPT real-workload trace loader.

BurstGPT is a real LLM serving workload dataset from production systems.
Reference: arXiv 2401.17644, SIGMETRICS 2025.

Real fields (from dataset):      arrival_time, prompt_tokens, actual_output_tokens
Synthetic augmented fields:      predicted_output_tokens, class_id, priority, slo_deadline
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.types import Request
from .augmentation import AugmentationConfig, augment_trace


_REQUEST_TOKEN_VARIANTS = [
    "Request Token", "Request tokens", "request_token", "RequestTokens",
    "input_tokens", "prompt_tokens",
]
_RESPONSE_TOKEN_VARIANTS = [
    "Response Token", "Response tokens", "response_token", "ResponseTokens",
    "output_tokens",
]
_TIMESTAMP_VARIANTS = [
    "Timestamp", "timestamp", "Time", "time",
]


def _detect_column(df: pd.DataFrame, variants: List[str], label: str) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for v in variants:
        if v in df.columns:
            return v
        if v.lower() in cols_lower:
            return cols_lower[v.lower()]
    raise ValueError(
        f"Cannot find {label} column in DataFrame. "
        f"Available columns: {list(df.columns)}. "
        f"Tried: {variants}"
    )


@dataclass
class BurstGPTConversionConfig:
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    max_requests: Optional[int] = None
    time_scale: float = 1.0
    min_prompt_tokens: int = 1
    min_output_tokens: int = 1
    max_prompt_tokens: int = 32768
    max_output_tokens: int = 32768


@dataclass
class ConversionReport:
    rows_read: int
    rows_retained: int
    rows_dropped_zero_tokens: int
    rows_dropped_invalid: int
    time_range_seconds: float
    mean_arrival_rate: float
    prompt_tokens_mean: float
    prompt_tokens_p95: float
    output_tokens_mean: float
    output_tokens_p95: float
    schema_detected: Dict[str, str]
    seed: int
    augmentation_config_summary: Dict


def conversion_report_to_dict(report: ConversionReport) -> dict:
    return {
        "rows_read": report.rows_read,
        "rows_retained": report.rows_retained,
        "rows_dropped_zero_tokens": report.rows_dropped_zero_tokens,
        "rows_dropped_invalid": report.rows_dropped_invalid,
        "time_range_seconds": report.time_range_seconds,
        "mean_arrival_rate": report.mean_arrival_rate,
        "prompt_tokens_mean": report.prompt_tokens_mean,
        "prompt_tokens_p95": report.prompt_tokens_p95,
        "output_tokens_mean": report.output_tokens_mean,
        "output_tokens_p95": report.output_tokens_p95,
        "schema_detected": report.schema_detected,
        "seed": report.seed,
        "augmentation_config_summary": report.augmentation_config_summary,
    }


def load_burstgpt_raw(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BurstGPT file not found: {path}")
    return pd.read_csv(path)


def convert_burstgpt_to_requests(
    df: pd.DataFrame,
    config: Optional[BurstGPTConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], ConversionReport]:
    if config is None:
        config = BurstGPTConversionConfig()
    if augmentation_config is None:
        augmentation_config = AugmentationConfig()

    rows_read = len(df)
    rows_dropped_zero = 0
    rows_dropped_invalid = 0

    ts_col = _detect_column(df, _TIMESTAMP_VARIANTS, "timestamp")
    req_col = _detect_column(df, _REQUEST_TOKEN_VARIANTS, "request tokens")
    resp_col = _detect_column(df, _RESPONSE_TOKEN_VARIANTS, "response tokens")

    schema_detected = {
        "timestamp": ts_col,
        "request_tokens": req_col,
        "response_tokens": resp_col,
    }

    df = df[[ts_col, req_col, resp_col]].copy()
    df.columns = ["timestamp", "prompt_tokens", "output_tokens"]

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["prompt_tokens"] = pd.to_numeric(df["prompt_tokens"], errors="coerce")
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce")

    initial_len = len(df)
    df = df.dropna()
    rows_dropped_invalid += initial_len - len(df)

    df = df.sort_values("timestamp").reset_index(drop=True)

    if config.start_time is not None or config.end_time is not None:
        lo = config.start_time if config.start_time is not None else -np.inf
        hi = config.end_time if config.end_time is not None else np.inf
        df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].reset_index(drop=True)

    if config.max_requests is not None:
        df = df.iloc[: config.max_requests].reset_index(drop=True)

    zero_mask = (df["prompt_tokens"] <= 0) | (df["output_tokens"] <= 0)
    rows_dropped_zero = int(zero_mask.sum())
    df = df[~zero_mask].reset_index(drop=True)

    if len(df) == 0:
        report = ConversionReport(
            rows_read=rows_read,
            rows_retained=0,
            rows_dropped_zero_tokens=rows_dropped_zero,
            rows_dropped_invalid=rows_dropped_invalid,
            time_range_seconds=0.0,
            mean_arrival_rate=0.0,
            prompt_tokens_mean=0.0,
            prompt_tokens_p95=0.0,
            output_tokens_mean=0.0,
            output_tokens_p95=0.0,
            schema_detected=schema_detected,
            seed=seed,
            augmentation_config_summary={
                "noise_mode": augmentation_config.prediction_noise.mode,
                "slo_classes": [c.class_id for c in augmentation_config.slo.classes],
            },
        )
        return [], report

    timestamps = df["timestamp"].values.astype(float)

    if config.time_scale != 1.0 and len(timestamps) > 1:
        interarrivals = np.diff(timestamps)
        scaled_gaps = interarrivals * config.time_scale
        scaled_timestamps = np.concatenate([[0.0], np.cumsum(scaled_gaps)])
    else:
        scaled_timestamps = timestamps - timestamps[0]

    arrival_times = scaled_timestamps

    prompt_tokens = np.clip(
        df["prompt_tokens"].values.astype(int),
        config.min_prompt_tokens,
        config.max_prompt_tokens,
    )
    output_tokens = np.clip(
        df["output_tokens"].values.astype(int),
        config.min_output_tokens,
        config.max_output_tokens,
    )

    rng = np.random.default_rng(seed)
    augmented = augment_trace(output_tokens, arrival_times, augmentation_config, rng)

    requests: List[Request] = []
    for i in range(len(df)):
        req = Request(
            request_id=i,
            arrival_time=float(arrival_times[i]),
            prompt_tokens=int(prompt_tokens[i]),
            predicted_output_tokens=int(augmented["predicted_output_tokens"][i]),
            actual_output_tokens=int(output_tokens[i]),
            slo_deadline=float(augmented["slo_deadlines"][i]),
            priority=float(augmented["priorities"][i]),
            class_id=augmented["class_ids"][i],
        )
        requests.append(req)

    rows_retained = len(requests)
    time_range = float(arrival_times[-1] - arrival_times[0]) if len(arrival_times) > 1 else 0.0
    mean_rate = rows_retained / time_range if time_range > 0 else 0.0

    report = ConversionReport(
        rows_read=rows_read,
        rows_retained=rows_retained,
        rows_dropped_zero_tokens=rows_dropped_zero,
        rows_dropped_invalid=rows_dropped_invalid,
        time_range_seconds=time_range,
        mean_arrival_rate=mean_rate,
        prompt_tokens_mean=float(np.mean(prompt_tokens)),
        prompt_tokens_p95=float(np.percentile(prompt_tokens, 95)),
        output_tokens_mean=float(np.mean(output_tokens)),
        output_tokens_p95=float(np.percentile(output_tokens, 95)),
        schema_detected=schema_detected,
        seed=seed,
        augmentation_config_summary={
            "noise_mode": augmentation_config.prediction_noise.mode,
            "slo_classes": [c.class_id for c in augmentation_config.slo.classes],
        },
    )
    return requests, report


def load_burstgpt_trace(
    path: Union[str, Path],
    config: Optional[BurstGPTConversionConfig] = None,
    seed: int = 0,
    aug_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], ConversionReport]:
    df = load_burstgpt_raw(path)
    return convert_burstgpt_to_requests(df, config, seed, aug_config)
