"""
Mooncake / Kimi serving-trace converter.

Official source
---------------
https://github.com/kvcache-ai/Mooncake (FAST25-release/traces)

Prefer FAST'25 files under ``FAST25-release/traces/``:
  conversation_trace.jsonl, toolagent_trace.jsonl, synthetic_trace.jsonl.

Dataset type: true serving trace (production-derived, anonymized) for the
conversation / tool-agent releases; the synthetic_trace.jsonl file is
constructed from public datasets with Poisson arrivals and must be labeled
synthetic.

Licensing
---------
CODE_REPO_LICENSE = Apache-2.0 (``LICENSE-APACHE`` in kvcache-ai/Mooncake).
DATA_LICENSE = NOT_EXPLICITLY_SPECIFIED
  Traces are released in-repo without a dedicated dataset-license notice
  separate from the project Apache-2.0 file. Record attribution/citation
  requirements from the FAST'25 / arXiv papers; do not assert a dedicated
  CC or dataset SPDX id for the JSONL files alone.

Timestamp semantics (official FAST'25 README)
---------------------------------------------
``timestamp`` is relative request arrival time in **milliseconds**.
This converter converts to seconds before writing ``arrival_time``.

Observed fields: timestamp, input_length, output_length, hash_ids
(prefix blocks of 512 tokens).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .augmentation import AugmentationConfig, augment_trace
from .canonical_schema import (
    CanonicalIngestRecord,
    DatasetType,
    FieldProvenance,
    TIMESTAMP_UNIT_SECONDS_RELATIVE,
    default_provenance,
    records_to_requests_and_metadata,
    replay_label_for_time_scale,
    require_mapping_fields,
    scale_interarrivals,
    validate_canonical_records,
)
from ..core.types import Request

_REQUIRED = ("timestamp", "input_length", "output_length")
_SOURCE_TIMESTAMP_UNIT = "milliseconds_relative"


@dataclass
class MooncakeConversionConfig:
    max_requests: Optional[int] = None
    time_scale: float = 1.0
    min_prompt_tokens: int = 1
    min_output_tokens: int = 1
    max_prompt_tokens: int = 262144
    max_output_tokens: int = 65536
    source_dataset: str = "mooncake"
    source_split: str = "conversation_trace"
    # synthetic_trace.jsonl is not a production serving trace.
    treat_as_synthetic: bool = False
    # Official FAST'25 traces use millisecond timestamps.
    source_timestamp_unit: str = _SOURCE_TIMESTAMP_UNIT
    # When True, refuse paths/splits that look synthetic.
    require_real_only: bool = False


@dataclass
class MooncakeConversionReport:
    rows_read: int
    rows_retained: int
    rows_dropped_invalid: int
    rows_dropped_zero_tokens: int
    time_range_seconds: float
    mean_arrival_rate: float
    prompt_tokens_mean: float
    output_tokens_mean: float
    time_scale: float
    replay_label: str
    source_split: str
    dataset_type: str


def load_mooncake_jsonl(
    path: Union[str, Path],
    max_requests: Optional[int] = None,
) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mooncake trace not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("version https://git-lfs"):
                raise ValueError(
                    f"{path} looks like a Git LFS pointer. "
                    "Use media.githubusercontent.com or git lfs pull."
                )
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Mooncake row must be an object, got {type(row)}")
            rows.append(row)
            if max_requests is not None and len(rows) >= max_requests:
                break
    return rows


def convert_mooncake_rows(
    rows: Sequence[Dict[str, Any]],
    config: Optional[MooncakeConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[CanonicalIngestRecord], MooncakeConversionReport]:
    if config is None:
        config = MooncakeConversionConfig()
    if augmentation_config is None:
        augmentation_config = AugmentationConfig()

    rows_read = len(rows)
    dropped_invalid = 0
    dropped_zero = 0
    cleaned: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        try:
            require_mapping_fields(row, _REQUIRED, f"mooncake[{i}]")
            prompt = int(row["input_length"])
            output = int(row["output_length"])
            float(row["timestamp"])
        except (TypeError, ValueError, KeyError):
            dropped_invalid += 1
            continue
        if prompt < config.min_prompt_tokens or output < config.min_output_tokens:
            dropped_zero += 1
            continue
        if prompt > config.max_prompt_tokens or output > config.max_output_tokens:
            dropped_invalid += 1
            continue
        cleaned.append(row)

    if config.max_requests is not None:
        cleaned = cleaned[: config.max_requests]

    dataset_type = (
        DatasetType.SYNTHETIC_OR_TRACE_CALIBRATED.value
        if config.treat_as_synthetic
        else DatasetType.TRUE_SERVING_TRACE.value
    )

    if not cleaned:
        report = MooncakeConversionReport(
            rows_read=rows_read,
            rows_retained=0,
            rows_dropped_invalid=dropped_invalid,
            rows_dropped_zero_tokens=dropped_zero,
            time_range_seconds=0.0,
            mean_arrival_rate=0.0,
            prompt_tokens_mean=0.0,
            output_tokens_mean=0.0,
            time_scale=config.time_scale,
            replay_label=replay_label_for_time_scale(config.time_scale),
            source_split=config.source_split,
            dataset_type=dataset_type,
        )
        return [], report

    cleaned = sorted(cleaned, key=lambda r: float(r["timestamp"]))
    raw_timestamps = [float(r["timestamp"]) for r in cleaned]
    if config.source_timestamp_unit == _SOURCE_TIMESTAMP_UNIT:
        timestamps_seconds = [t / 1000.0 for t in raw_timestamps]
    elif config.source_timestamp_unit in {"seconds", "seconds_relative"}:
        timestamps_seconds = list(raw_timestamps)
    else:
        raise ValueError(
            f"Unsupported Mooncake source_timestamp_unit={config.source_timestamp_unit!r}"
        )
    arrival_times = scale_interarrivals(timestamps_seconds, config.time_scale)
    prompt_tokens = np.array([int(r["input_length"]) for r in cleaned], dtype=int)
    output_tokens = np.array([int(r["output_length"]) for r in cleaned], dtype=int)

    rng = np.random.default_rng(seed)
    augmented = augment_trace(output_tokens, np.asarray(arrival_times), augmentation_config, rng)
    replay_label = replay_label_for_time_scale(config.time_scale)

    records: List[CanonicalIngestRecord] = []
    for i, row in enumerate(cleaned):
        hash_ids = row.get("hash_ids") or []
        prefix_id = None
        extra: Dict[str, Any] = {
            "n_hash_ids": len(hash_ids) if isinstance(hash_ids, list) else None,
            "raw_timestamp": float(row["timestamp"]),
            "source_timestamp_unit": config.source_timestamp_unit,
            "prefix_block_tokens": 512,
            "data_license": "NOT_EXPLICITLY_SPECIFIED",
            "code_repo_license": "Apache-2.0",
        }
        if isinstance(hash_ids, list) and hash_ids:
            prefix_id = f"mooncake_h0:{hash_ids[0]}:n{len(hash_ids)}"

        prov = default_provenance(
            arrival_time=FieldProvenance.DERIVED.value,  # ms → relative seconds
            prefix_id=(
                FieldProvenance.DERIVED.value
                if prefix_id is not None
                else FieldProvenance.UNAVAILABLE.value
            ),
            session_id=FieldProvenance.UNAVAILABLE.value,
            model_id=FieldProvenance.UNAVAILABLE.value,
            tenant_id=FieldProvenance.UNAVAILABLE.value,
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
                prefix_id=prefix_id,
                source_dataset=config.source_dataset,
                source_split=config.source_split,
                source_record_id=str(i),
                field_provenance=prov,
                timestamp_unit=TIMESTAMP_UNIT_SECONDS_RELATIVE,
                time_scale=config.time_scale,
                replay_label=replay_label,
                dataset_type=dataset_type,
                extra=extra,
            )
        )

    validate_canonical_records(records)
    time_range = float(arrival_times[-1] - arrival_times[0]) if len(arrival_times) > 1 else 0.0
    report = MooncakeConversionReport(
        rows_read=rows_read,
        rows_retained=len(records),
        rows_dropped_invalid=dropped_invalid,
        rows_dropped_zero_tokens=dropped_zero,
        time_range_seconds=time_range,
        mean_arrival_rate=(len(records) / time_range) if time_range > 0 else 0.0,
        prompt_tokens_mean=float(np.mean(prompt_tokens)),
        output_tokens_mean=float(np.mean(output_tokens)),
        time_scale=config.time_scale,
        replay_label=replay_label,
        source_split=config.source_split,
        dataset_type=dataset_type,
    )
    return records, report


def convert_mooncake_to_requests(
    rows: Sequence[Dict[str, Any]],
    config: Optional[MooncakeConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], List[Dict[str, Any]], MooncakeConversionReport]:
    records, report = convert_mooncake_rows(rows, config, seed, augmentation_config)
    requests, metadata = records_to_requests_and_metadata(records)
    return requests, metadata, report


def _looks_synthetic(path: Path, source_split: str, treat_as_synthetic: bool) -> bool:
    name = path.name.lower()
    split = (source_split or "").lower()
    return treat_as_synthetic or ("synthetic" in name) or ("synthetic" in split)


def load_mooncake_trace(
    path: Union[str, Path],
    config: Optional[MooncakeConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], List[Dict[str, Any]], MooncakeConversionReport]:
    cfg = config or MooncakeConversionConfig()
    path = Path(path)
    is_synthetic = _looks_synthetic(path, cfg.source_split, cfg.treat_as_synthetic)
    if cfg.require_real_only and is_synthetic:
        raise ValueError(
            f"Mooncake real-only load refused for synthetic source: path={path.name!r} "
            f"source_split={cfg.source_split!r}"
        )
    if is_synthetic:
        cfg = MooncakeConversionConfig(
            max_requests=cfg.max_requests,
            time_scale=cfg.time_scale,
            min_prompt_tokens=cfg.min_prompt_tokens,
            min_output_tokens=cfg.min_output_tokens,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_output_tokens=cfg.max_output_tokens,
            source_dataset=cfg.source_dataset,
            source_split=cfg.source_split or "synthetic_trace",
            treat_as_synthetic=True,
            source_timestamp_unit=cfg.source_timestamp_unit,
            require_real_only=False,
        )
    rows = load_mooncake_jsonl(path, max_requests=cfg.max_requests)
    return convert_mooncake_to_requests(rows, cfg, seed, augmentation_config)
