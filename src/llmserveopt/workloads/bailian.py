"""
Bailian / Qwen anonymized serving-trace converter.

Official source
---------------
https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon
Pinned tip inspected 2026-07-24: commit 5f7439c51ec248a0c585f7d90a41a6f57773b912

Dataset type: true serving trace (production-derived, anonymized).

Licensing
---------
CODE_REPO_LICENSE = Apache-2.0 (repository ``LICENSE`` file).
DATA_LICENSE = Apache-2.0
  This repository's primary content is the anonymized traces; README §License
  explicitly states Apache License 2.0 for the dataset release.

Observed fields: timestamp (seconds, relative to trace start), input_length,
output_length, chat_id / parent_chat_id (session), type, turn, hash_ids
(16-token KV/prefix blocks).

Synthesized fields (disclosed): predicted_output_tokens, class_id, priority,
slo_deadline.

Memory note: ``load_bailian_jsonl`` streams line-by-line and may stop early via
``max_requests``; it does not require loading multi-GB shards when limited.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .augmentation import AugmentationConfig, augment_trace
from .canonical_schema import (
    CanonicalIngestRecord,
    FieldProvenance,
    DatasetType,
    default_provenance,
    records_to_requests_and_metadata,
    replay_label_for_time_scale,
    require_mapping_fields,
    scale_interarrivals,
    validate_canonical_records,
)
from ..core.types import Request

_REQUIRED = ("timestamp", "input_length", "output_length")


@dataclass
class BailianConversionConfig:
    max_requests: Optional[int] = None
    time_scale: float = 1.0
    min_prompt_tokens: int = 1
    min_output_tokens: int = 1
    max_prompt_tokens: int = 131072
    max_output_tokens: int = 131072
    source_dataset: str = "qwen_bailian_usagetraces_anon"
    source_split: str = ""
    keep_hash_id_count: bool = True


@dataclass
class BailianConversionReport:
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
    type_counts: Dict[str, int]


def load_bailian_jsonl(
    path: Union[str, Path],
    max_requests: Optional[int] = None,
) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bailian trace not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Refuse Git LFS pointer files early.
            if line.startswith("version https://git-lfs"):
                raise ValueError(
                    f"{path} looks like a Git LFS pointer, not the real JSONL. "
                    "Download via media.githubusercontent.com or `git lfs pull`."
                )
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Bailian row must be an object, got {type(row)}")
            rows.append(row)
            if max_requests is not None and len(rows) >= max_requests:
                break
    return rows


def convert_bailian_rows(
    rows: Sequence[Dict[str, Any]],
    config: Optional[BailianConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[CanonicalIngestRecord], BailianConversionReport]:
    if config is None:
        config = BailianConversionConfig()
    if augmentation_config is None:
        augmentation_config = AugmentationConfig()

    rows_read = len(rows)
    dropped_invalid = 0
    dropped_zero = 0
    cleaned: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        try:
            require_mapping_fields(row, _REQUIRED, f"bailian[{i}]")
            ts = float(row["timestamp"])
            prompt = int(row["input_length"])
            output = int(row["output_length"])
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

    if not cleaned:
        report = BailianConversionReport(
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
            type_counts={},
        )
        return [], report

    # Preserve chronological order from the source file (already time-ordered
    # in official traces); re-sort defensively by timestamp.
    cleaned = sorted(cleaned, key=lambda r: float(r["timestamp"]))
    timestamps = [float(r["timestamp"]) for r in cleaned]
    arrival_times = scale_interarrivals(timestamps, config.time_scale)
    prompt_tokens = np.array([int(r["input_length"]) for r in cleaned], dtype=int)
    output_tokens = np.array([int(r["output_length"]) for r in cleaned], dtype=int)

    rng = np.random.default_rng(seed)
    augmented = augment_trace(output_tokens, np.asarray(arrival_times), augmentation_config, rng)
    replay_label = replay_label_for_time_scale(config.time_scale)

    type_counts: Dict[str, int] = {}
    records: List[CanonicalIngestRecord] = []
    for i, row in enumerate(cleaned):
        req_type = str(row.get("type", "unknown"))
        type_counts[req_type] = type_counts.get(req_type, 0) + 1
        chat_id = row.get("chat_id")
        parent = row.get("parent_chat_id")
        hash_ids = row.get("hash_ids") or []
        prefix_id = None
        extra: Dict[str, Any] = {
            "request_type": req_type,
            "turn": row.get("turn"),
            "parent_chat_id": parent,
        }
        if config.keep_hash_id_count:
            extra["n_hash_ids"] = len(hash_ids) if isinstance(hash_ids, list) else None
        if isinstance(hash_ids, list) and hash_ids:
            # Stable prefix fingerprint without storing the full block list.
            prefix_id = f"bailian_h0:{hash_ids[0]}:n{len(hash_ids)}"

        prov = default_provenance(
            session_id=(
                FieldProvenance.OBSERVED.value
                if chat_id is not None
                else FieldProvenance.UNAVAILABLE.value
            ),
            prefix_id=(
                FieldProvenance.DERIVED.value
                if prefix_id is not None
                else FieldProvenance.UNAVAILABLE.value
            ),
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
                session_id=str(chat_id) if chat_id is not None else None,
                prefix_id=prefix_id,
                source_dataset=config.source_dataset,
                source_split=config.source_split,
                source_record_id=str(i),
                field_provenance=prov,
                time_scale=config.time_scale,
                replay_label=replay_label,
                dataset_type=DatasetType.TRUE_SERVING_TRACE.value,
                extra=extra,
            )
        )

    validate_canonical_records(records)
    time_range = float(arrival_times[-1] - arrival_times[0]) if len(arrival_times) > 1 else 0.0
    report = BailianConversionReport(
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
        type_counts=type_counts,
    )
    return records, report


def convert_bailian_to_requests(
    rows: Sequence[Dict[str, Any]],
    config: Optional[BailianConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], List[Dict[str, Any]], BailianConversionReport]:
    records, report = convert_bailian_rows(rows, config, seed, augmentation_config)
    requests, metadata = records_to_requests_and_metadata(records)
    return requests, metadata, report


def load_bailian_trace(
    path: Union[str, Path],
    config: Optional[BailianConversionConfig] = None,
    seed: int = 0,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Tuple[List[Request], List[Dict[str, Any]], BailianConversionReport]:
    cfg = config or BailianConversionConfig()
    rows = load_bailian_jsonl(path, max_requests=cfg.max_requests)
    return convert_bailian_to_requests(rows, cfg, seed, augmentation_config)
