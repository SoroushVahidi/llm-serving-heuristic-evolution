"""
Azure LLM Inference trace helpers with canonical provenance.

Wraps the CSV schema used by Azure 2023 (Splitwise) and Azure 2024 (DynamoLLM):
TIMESTAMP, ContextTokens, GeneratedTokens.

Official subsets: code and conversation only. No public function-calling
subset exists in AzurePublicDataset releases audited on 2026-07-24.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.types import Request
from .augmentation import AugmentationConfig, augment_trace
from .canonical_schema import (
    CanonicalIngestRecord,
    DatasetType,
    FieldProvenance,
    TIMESTAMP_UNIT_DATETIME_ISO,
    default_provenance,
    records_to_requests_and_metadata,
    replay_label_for_time_scale,
    scale_interarrivals,
    validate_canonical_records,
)


@dataclass
class AzureConversionConfig:
    max_requests: Optional[int] = None
    time_scale: float = 1.0
    min_context_tokens: int = 1
    min_generated_tokens: int = 1
    source_dataset: str = "azure_llm_inference"
    source_split: str = ""


@dataclass
class AzureConversionReport:
    rows_read: int
    rows_retained: int
    rows_dropped: int
    time_scale: float
    replay_label: str
    time_range_seconds: float
    mean_arrival_rate: float
    context_tokens_mean: float
    generated_tokens_mean: float
    source_split: str


def parse_azure_timestamp(ts: str) -> float:
    ts = ts.strip()
    if "." in ts:
        base, frac = ts.rsplit(".", 1)
        frac = frac[:6]
        ts = f"{base}.{frac}"
    ts = ts.replace(" ", "T")
    return datetime.fromisoformat(ts).timestamp()


def load_azure_csv_rows(
    path: Union[str, Path],
    max_requests: Optional[int] = None,
    min_context_tokens: int = 1,
    min_generated_tokens: int = 1,
) -> Tuple[List[Dict[str, Any]], int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Azure CSV not found: {path}")
    raw_rows: List[Dict[str, Any]] = []
    dropped = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Azure CSV missing required columns {required}; got {reader.fieldnames}"
            )
        for row in reader:
            try:
                ctx = int(row["ContextTokens"])
                gen = int(row["GeneratedTokens"])
            except (TypeError, ValueError, KeyError):
                dropped += 1
                continue
            if ctx < min_context_tokens or gen < min_generated_tokens:
                dropped += 1
                continue
            raw_rows.append(
                {
                    "ts_str": row["TIMESTAMP"],
                    "context": ctx,
                    "generated": gen,
                }
            )
            if max_requests is not None and len(raw_rows) >= max_requests:
                break
    return raw_rows, dropped


def convert_azure_rows(
    raw_rows: Sequence[Dict[str, Any]],
    config: Optional[AzureConversionConfig] = None,
    seed: int = 17,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[CanonicalIngestRecord], AzureConversionReport]:
    if config is None:
        config = AzureConversionConfig()
    if augmentation_config is None:
        augmentation_config = AugmentationConfig()

    rows_read = len(raw_rows)
    if not raw_rows:
        return [], AzureConversionReport(
            rows_read=0,
            rows_retained=0,
            rows_dropped=0,
            time_scale=config.time_scale,
            replay_label=replay_label_for_time_scale(config.time_scale),
            time_range_seconds=0.0,
            mean_arrival_rate=0.0,
            context_tokens_mean=0.0,
            generated_tokens_mean=0.0,
            source_split=config.source_split,
        )

    raw_ts = np.array([parse_azure_timestamp(r["ts_str"]) for r in raw_rows], dtype=float)
    arrival_times = np.asarray(scale_interarrivals(raw_ts, config.time_scale), dtype=float)
    context_tokens = np.array([r["context"] for r in raw_rows], dtype=int)
    generated_tokens = np.array([r["generated"] for r in raw_rows], dtype=int)

    rng = np.random.default_rng(seed)
    augmented = augment_trace(generated_tokens, arrival_times, augmentation_config, rng)
    replay_label = replay_label_for_time_scale(config.time_scale)

    records: List[CanonicalIngestRecord] = []
    for i, row in enumerate(raw_rows):
        prov = default_provenance(
            session_id=FieldProvenance.UNAVAILABLE.value,
            model_id=FieldProvenance.UNAVAILABLE.value,
            tenant_id=FieldProvenance.UNAVAILABLE.value,
            prefix_id=FieldProvenance.UNAVAILABLE.value,
        )
        records.append(
            CanonicalIngestRecord(
                request_id=i,
                arrival_time=float(arrival_times[i]),
                prompt_tokens=int(context_tokens[i]),
                actual_output_tokens=int(generated_tokens[i]),
                predicted_output_tokens=int(augmented["predicted_output_tokens"][i]),
                slo_deadline=float(augmented["slo_deadlines"][i]),
                priority=float(augmented["priorities"][i]),
                class_id=str(augmented["class_ids"][i]),
                source_dataset=config.source_dataset,
                source_split=config.source_split,
                source_record_id=str(i),
                field_provenance=prov,
                timestamp_unit=TIMESTAMP_UNIT_DATETIME_ISO,
                time_scale=config.time_scale,
                replay_label=replay_label,
                dataset_type=DatasetType.TRUE_SERVING_TRACE.value,
                extra={"original_timestamp": row["ts_str"]},
            )
        )

    validate_canonical_records(records)
    time_range = float(arrival_times[-1] - arrival_times[0]) if len(arrival_times) > 1 else 0.0
    report = AzureConversionReport(
        rows_read=rows_read,
        rows_retained=len(records),
        rows_dropped=0,
        time_scale=config.time_scale,
        replay_label=replay_label,
        time_range_seconds=time_range,
        mean_arrival_rate=(len(records) / time_range) if time_range > 0 else 0.0,
        context_tokens_mean=float(np.mean(context_tokens)),
        generated_tokens_mean=float(np.mean(generated_tokens)),
        source_split=config.source_split,
    )
    return records, report


def convert_azure_to_requests(
    path: Union[str, Path],
    config: Optional[AzureConversionConfig] = None,
    seed: int = 17,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], List[Dict[str, Any]], AzureConversionReport]:
    cfg = config or AzureConversionConfig()
    rows, dropped = load_azure_csv_rows(
        path,
        max_requests=cfg.max_requests,
        min_context_tokens=cfg.min_context_tokens,
        min_generated_tokens=cfg.min_generated_tokens,
    )
    records, report = convert_azure_rows(rows, cfg, seed, augmentation_config)
    report.rows_dropped = dropped
    requests, metadata = records_to_requests_and_metadata(records)
    return requests, metadata, report
