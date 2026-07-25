"""
Canonical ingestion schema for reality-grounded serving workloads.

This module defines the documented intermediate representation used when
converting heterogeneous traces and corpora into simulator ``Request`` objects.

Design rules
------------
1. Preserve original raw values in external staging, not in Git.
2. Record every transformation via per-field provenance.
3. Timestamp units must be explicit (seconds relative to first request unless
   documented otherwise).
4. Distinguish missing values from synthesized values.
5. ``actual_output_tokens`` is evaluation-only and must never appear in the
   scheduler-visible field set (see ``OBSERVABLE_REQUEST_FIELDS``).
6. SLO / priority synthesis must be disclosed; never silent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from ..core.types import ObservableRequest, Request


class FieldProvenance(str, Enum):
    """How a canonical field obtained its value."""

    OBSERVED = "observed"
    DERIVED = "derived"
    SYNTHESIZED = "synthesized"
    UNAVAILABLE = "unavailable"


class DatasetType(str, Enum):
    TRUE_SERVING_TRACE = "true_serving_trace"
    PROMPT_CONVERSATION_CORPUS = "prompt_conversation_corpus"
    BENCHMARK_PROMPT_CORPUS = "benchmark_prompt_corpus"
    SYNTHETIC_OR_TRACE_CALIBRATED = "synthetic_or_trace_calibrated"
    AGGREGATE_STATISTICS_ONLY = "aggregate_statistics_only"


class ReplayLabel(str, Enum):
    NATURAL_TRACE_REPLAY = "natural_trace_replay"
    TRACE_DERIVED_TIME_SCALED = "trace-derived, time-scaled"
    TRACE_CALIBRATED_SYNTHETIC_ARRIVALS = "trace-calibrated, synthetic_arrivals"
    FULLY_SYNTHETIC = "fully_synthetic"


# Scheduler-visible fields. Deliberately excludes actual_output_tokens.
OBSERVABLE_REQUEST_FIELDS: frozenset[str] = frozenset({
    "request_id",
    "arrival_time",
    "prompt_tokens",
    "predicted_output_tokens",
    "slo_deadline",
    "priority",
    "class_id",
})

CANONICAL_CORE_FIELDS: tuple[str, ...] = (
    "request_id",
    "arrival_time",
    "prompt_tokens",
    "actual_output_tokens",
    "predicted_output_tokens",
    "slo_deadline",
    "priority",
    "class_id",
)

CANONICAL_OPTIONAL_FIELDS: tuple[str, ...] = (
    "session_id",
    "tenant_id",
    "model_id",
    "prefix_id",
    "source_dataset",
    "source_split",
    "source_record_id",
)

TIMESTAMP_UNIT_SECONDS_RELATIVE = "seconds_relative_to_first_request"
TIMESTAMP_UNIT_SECONDS_ABSOLUTE = "seconds_absolute"
TIMESTAMP_UNIT_DATETIME_ISO = "datetime_iso_parsed_to_unix_seconds"


@dataclass
class CanonicalIngestRecord:
    """One request in the canonical ingestion schema."""

    request_id: int
    arrival_time: float
    prompt_tokens: int
    actual_output_tokens: int
    predicted_output_tokens: int
    slo_deadline: float
    priority: float
    class_id: str
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    model_id: Optional[str] = None
    prefix_id: Optional[str] = None
    source_dataset: str = ""
    source_split: str = ""
    source_record_id: Optional[str] = None
    field_provenance: Dict[str, str] = field(default_factory=dict)
    timestamp_unit: str = TIMESTAMP_UNIT_SECONDS_RELATIVE
    time_scale: float = 1.0
    replay_label: str = ReplayLabel.NATURAL_TRACE_REPLAY.value
    dataset_type: str = DatasetType.TRUE_SERVING_TRACE.value
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_request(self) -> Request:
        return Request(
            request_id=int(self.request_id),
            arrival_time=float(self.arrival_time),
            prompt_tokens=int(self.prompt_tokens),
            predicted_output_tokens=int(self.predicted_output_tokens),
            actual_output_tokens=int(self.actual_output_tokens),
            slo_deadline=float(self.slo_deadline),
            priority=float(self.priority),
            class_id=str(self.class_id),
        )

    def to_metadata(self) -> Dict[str, Any]:
        synthetic = sorted(
            k for k, v in self.field_provenance.items()
            if v == FieldProvenance.SYNTHESIZED.value
        )
        observed = sorted(
            k for k, v in self.field_provenance.items()
            if v == FieldProvenance.OBSERVED.value
        )
        derived = sorted(
            k for k, v in self.field_provenance.items()
            if v == FieldProvenance.DERIVED.value
        )
        unavailable = sorted(
            k for k, v in self.field_provenance.items()
            if v == FieldProvenance.UNAVAILABLE.value
        )
        md: Dict[str, Any] = {
            "field_provenance": dict(self.field_provenance),
            "synthetic_fields": synthetic,
            "observed_fields": observed,
            "derived_fields": derived,
            "unavailable_fields": unavailable,
            "timestamp_unit": self.timestamp_unit,
            "time_scale": self.time_scale,
            "replay_label": self.replay_label,
            "dataset_type": self.dataset_type,
            "source_dataset": self.source_dataset,
            "source_split": self.source_split,
            "source_record_id": self.source_record_id,
        }
        if self.session_id is not None:
            md["session_id"] = self.session_id
        if self.tenant_id is not None:
            md["tenant_id"] = self.tenant_id
        if self.model_id is not None:
            md["model_id"] = self.model_id
        if self.prefix_id is not None:
            md["prefix_id"] = self.prefix_id
        if self.extra:
            md["extra"] = dict(self.extra)
        return md

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def default_provenance(**overrides: str) -> Dict[str, str]:
    """Build a provenance map with UNAVAILABLE defaults for optional fields."""
    base = {
        "request_id": FieldProvenance.DERIVED.value,
        "arrival_time": FieldProvenance.OBSERVED.value,
        "prompt_tokens": FieldProvenance.OBSERVED.value,
        "actual_output_tokens": FieldProvenance.OBSERVED.value,
        "predicted_output_tokens": FieldProvenance.SYNTHESIZED.value,
        "slo_deadline": FieldProvenance.SYNTHESIZED.value,
        "priority": FieldProvenance.SYNTHESIZED.value,
        "class_id": FieldProvenance.SYNTHESIZED.value,
        "session_id": FieldProvenance.UNAVAILABLE.value,
        "tenant_id": FieldProvenance.UNAVAILABLE.value,
        "model_id": FieldProvenance.UNAVAILABLE.value,
        "prefix_id": FieldProvenance.UNAVAILABLE.value,
    }
    base.update(overrides)
    return base


def validate_canonical_record(rec: CanonicalIngestRecord) -> None:
    """Raise ``ValueError`` if the record violates canonical invariants."""
    if rec.request_id < 0:
        raise ValueError(f"request_id must be non-negative, got {rec.request_id}")
    if rec.arrival_time < 0:
        raise ValueError(f"arrival_time must be non-negative, got {rec.arrival_time}")
    if rec.prompt_tokens <= 0:
        raise ValueError(f"prompt_tokens must be positive, got {rec.prompt_tokens}")
    if rec.actual_output_tokens <= 0:
        raise ValueError(
            f"actual_output_tokens must be positive, got {rec.actual_output_tokens}"
        )
    if rec.predicted_output_tokens <= 0:
        raise ValueError(
            f"predicted_output_tokens must be positive, got {rec.predicted_output_tokens}"
        )
    if rec.slo_deadline < rec.arrival_time:
        raise ValueError(
            f"slo_deadline ({rec.slo_deadline}) must be >= arrival_time ({rec.arrival_time})"
        )
    if not rec.class_id:
        raise ValueError("class_id must be non-empty")
    if rec.time_scale <= 0:
        raise ValueError(f"time_scale must be positive, got {rec.time_scale}")

    for name in CANONICAL_CORE_FIELDS:
        if name not in rec.field_provenance:
            raise ValueError(f"missing provenance for core field {name!r}")
        prov = rec.field_provenance[name]
        try:
            FieldProvenance(prov)
        except ValueError as exc:
            raise ValueError(f"invalid provenance for {name}: {prov!r}") from exc

    # Silent SLO/priority synthesis is forbidden: if values are present they
    # must be labeled synthesized/observed/derived, never unavailable.
    for name in ("slo_deadline", "priority", "class_id"):
        if rec.field_provenance.get(name) == FieldProvenance.UNAVAILABLE.value:
            raise ValueError(
                f"{name} cannot be UNAVAILABLE on a fully-formed ingest record; "
                "either observe it or disclose SYNTHESIZED"
            )


def validate_canonical_records(records: Sequence[CanonicalIngestRecord]) -> None:
    if not records:
        return
    prev_t = -1.0
    for rec in records:
        validate_canonical_record(rec)
        if rec.arrival_time < prev_t:
            raise ValueError(
                f"records not chronologically ordered: "
                f"request_id={rec.request_id} arrival_time={rec.arrival_time} < {prev_t}"
            )
        prev_t = rec.arrival_time


def observable_request_fields() -> Set[str]:
    """Return the scheduler-visible field names."""
    return set(OBSERVABLE_REQUEST_FIELDS)


def assert_no_actual_output_leakage(
    observable_fields: Optional[Iterable[str]] = None,
) -> None:
    """Structural guard: actual outputs must not be scheduler-visible."""
    fields = set(observable_fields) if observable_fields is not None else set(
        OBSERVABLE_REQUEST_FIELDS
    )
    if "actual_output_tokens" in fields:
        raise AssertionError(
            "actual_output_tokens must not appear in observable request fields"
        )
    # Cross-check against the live ObservableRequest dataclass.
    obs_names = {f.name for f in ObservableRequest.__dataclass_fields__.values()}
    if "actual_output_tokens" in obs_names:
        raise AssertionError(
            "ObservableRequest dataclass unexpectedly exposes actual_output_tokens"
        )


def replay_label_for_time_scale(time_scale: float) -> str:
    if time_scale == 1.0:
        return ReplayLabel.NATURAL_TRACE_REPLAY.value
    return ReplayLabel.TRACE_DERIVED_TIME_SCALED.value


def scale_interarrivals(
    timestamps: Sequence[float],
    time_scale: float,
) -> List[float]:
    """Normalize to t0=0 and optionally scale inter-arrival gaps.

    ``time_scale`` multiplies gaps (values > 1 stretch traffic; values < 1
    compress it). Callers must record the scale and use
    ``replay_label_for_time_scale``.
    """
    if time_scale <= 0:
        raise ValueError(f"time_scale must be positive, got {time_scale}")
    if len(timestamps) == 0:
        return []
    arr = [float(t) for t in timestamps]
    t0 = arr[0]
    relative = [t - t0 for t in arr]
    if time_scale == 1.0 or len(relative) == 1:
        return relative
    gaps = [relative[i] - relative[i - 1] for i in range(1, len(relative))]
    scaled = [0.0]
    for g in gaps:
        scaled.append(scaled[-1] + g * time_scale)
    return scaled


def records_to_requests_and_metadata(
    records: Sequence[CanonicalIngestRecord],
) -> tuple[List[Request], List[Dict[str, Any]]]:
    validate_canonical_records(records)
    requests = [r.to_request() for r in records]
    metadata = [r.to_metadata() for r in records]
    return requests, metadata


def require_mapping_fields(row: Mapping[str, Any], required: Sequence[str], label: str) -> None:
    missing = [k for k in required if k not in row]
    if missing:
        raise ValueError(f"{label}: malformed record missing fields {missing}")
