"""
BurstGPT real-workload trace loader.

BurstGPT is a real LLM serving workload dataset from production systems.
Official source: https://github.com/HPMLL/BurstGPT (CC-BY-4.0).
Reference: arXiv 2401.17644, SIGMETRICS 2025.

Observed fields (from dataset):
  Timestamp, Request tokens, Response tokens
  Optional (v2 / BurstGPT_3): Session ID, Elapsed time, Model, Log Type

Synthetic augmented fields (disclosed):
  predicted_output_tokens, class_id, priority, slo_deadline
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.types import Request
from .augmentation import AugmentationConfig, augment_trace
from .canonical_schema import (
    CanonicalIngestRecord,
    DatasetType,
    FieldProvenance,
    default_provenance,
    records_to_requests_and_metadata,
    replay_label_for_time_scale,
    scale_interarrivals,
    validate_canonical_records,
)


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
_SESSION_VARIANTS = ["Session ID", "SessionID", "session_id", "session id"]
_MODEL_VARIANTS = ["Model", "model", "model_id"]
_LOGTYPE_VARIANTS = ["Log Type", "LogType", "log_type", "log type"]
_ELAPSED_VARIANTS = ["Elapsed time", "Elapsed Time", "elapsed_time", "elapsed"]


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


def _detect_optional_column(df: pd.DataFrame, variants: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for v in variants:
        if v in df.columns:
            return v
        if v.lower() in cols_lower:
            return cols_lower[v.lower()]
    return None


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


def convert_burstgpt_to_canonical(
    df: pd.DataFrame,
    config: Optional[BurstGPTConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[CanonicalIngestRecord], ConversionReport]:
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
    session_col = _detect_optional_column(df, _SESSION_VARIANTS)
    model_col = _detect_optional_column(df, _MODEL_VARIANTS)
    logtype_col = _detect_optional_column(df, _LOGTYPE_VARIANTS)
    elapsed_col = _detect_optional_column(df, _ELAPSED_VARIANTS)

    schema_detected = {
        "timestamp": ts_col,
        "request_tokens": req_col,
        "response_tokens": resp_col,
        "session_id": session_col,
        "model": model_col,
        "log_type": logtype_col,
        "elapsed_time": elapsed_col,
    }

    keep_cols = [ts_col, req_col, resp_col]
    rename = {ts_col: "timestamp", req_col: "prompt_tokens", resp_col: "output_tokens"}
    if session_col:
        keep_cols.append(session_col)
        rename[session_col] = "session_id"
    if model_col:
        keep_cols.append(model_col)
        rename[model_col] = "model_id"
    if logtype_col:
        keep_cols.append(logtype_col)
        rename[logtype_col] = "log_type"
    if elapsed_col:
        keep_cols.append(elapsed_col)
        rename[elapsed_col] = "elapsed_time"

    df = df[keep_cols].copy().rename(columns=rename)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["prompt_tokens"] = pd.to_numeric(df["prompt_tokens"], errors="coerce")
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce")

    initial_len = len(df)
    df = df.dropna(subset=["timestamp", "prompt_tokens", "output_tokens"])
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

    empty_report = ConversionReport(
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
    if len(df) == 0:
        return [], empty_report

    timestamps = df["timestamp"].values.astype(float)
    arrival_times = np.asarray(scale_interarrivals(timestamps, config.time_scale), dtype=float)
    replay_label = replay_label_for_time_scale(config.time_scale)

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

    records: List[CanonicalIngestRecord] = []
    for i in range(len(df)):
        session_id = None
        model_id = None
        extra: Dict[str, Any] = {"replay_label": replay_label}
        if "session_id" in df.columns:
            val = df.at[i, "session_id"]
            if pd.notna(val):
                session_id = str(val)
        if "model_id" in df.columns:
            val = df.at[i, "model_id"]
            if pd.notna(val):
                model_id = str(val)
        if "log_type" in df.columns and pd.notna(df.at[i, "log_type"]):
            extra["log_type"] = str(df.at[i, "log_type"])
        if "elapsed_time" in df.columns and pd.notna(df.at[i, "elapsed_time"]):
            extra["elapsed_time"] = float(df.at[i, "elapsed_time"])
        extra["original_timestamp"] = float(timestamps[i])

        prov = default_provenance(
            session_id=(
                FieldProvenance.OBSERVED.value
                if session_id is not None
                else FieldProvenance.UNAVAILABLE.value
            ),
            model_id=(
                FieldProvenance.OBSERVED.value
                if model_id is not None
                else FieldProvenance.UNAVAILABLE.value
            ),
        )
        records.append(
            CanonicalIngestRecord(
                request_id=i,
                arrival_time=float(arrival_times[i]),
                prompt_tokens=int(prompt_tokens[i]),
                actual_output_tokens=int(output_tokens[i]),
                predicted_output_tokens=int(augmented["predicted_output_tokens"][i]),
                slo_deadline=float(augmented["slo_deadlines"][i]),
                priority=float(augmented["priorities"][i]),
                class_id=str(augmented["class_ids"][i]),
                session_id=session_id,
                model_id=model_id,
                source_dataset="burstgpt",
                source_split="",
                source_record_id=str(i),
                field_provenance=prov,
                time_scale=config.time_scale,
                replay_label=replay_label,
                dataset_type=DatasetType.TRUE_SERVING_TRACE.value,
                extra=extra,
            )
        )

    validate_canonical_records(records)
    rows_retained = len(records)
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
            "time_scale": config.time_scale,
            "replay_label": replay_label,
        },
    )
    return records, report


def convert_burstgpt_to_requests(
    df: pd.DataFrame,
    config: Optional[BurstGPTConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], ConversionReport]:
    records, report = convert_burstgpt_to_canonical(df, config, seed, augmentation_config)
    requests, _metadata = records_to_requests_and_metadata(records)
    return requests, report


def convert_burstgpt_to_requests_with_metadata(
    df: pd.DataFrame,
    config: Optional[BurstGPTConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], List[Dict[str, Any]], ConversionReport]:
    records, report = convert_burstgpt_to_canonical(df, config, seed, augmentation_config)
    requests, metadata = records_to_requests_and_metadata(records)
    return requests, metadata, report


def load_burstgpt_trace(
    path: Union[str, Path],
    config: Optional[BurstGPTConversionConfig] = None,
    seed: int = 0,
    aug_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], ConversionReport]:
    df = load_burstgpt_raw(path)
    return convert_burstgpt_to_requests(df, config, seed, aug_config)
