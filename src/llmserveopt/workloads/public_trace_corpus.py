"""
Public Trace Corpus v1 — source adapters and canonical corpus records.

Design doc: docs/design/PUBLIC_TRACE_CORPUS_V1.md
Corpus schema: data/public_trace_corpus_v1/schema.json

This module builds the WORKLOAD-INPUT layer only (Layer 0/1 in the design
doc). It must never compute or embed policy outcomes, oracle labels, or
scheduling decisions. It is deliberately independent from
``canonical_schema.py``'s ``CanonicalIngestRecord`` (which is the
scheduler-facing, SLO-synthesizing intermediate representation used by the
simulator converters) — this module's ``CorpusRecord`` is a plain
descriptive record of what a public source actually contains, with no
synthesized fields at all.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

PROVENANCE_NATIVE = "NATIVE"
PROVENANCE_DERIVED = "DETERMINISTIC_DERIVED"
PROVENANCE_SOURCE_SPECIFIC = "SOURCE_SPECIFIC"
PROVENANCE_UNAVAILABLE = "UNAVAILABLE"

CANONICAL_FIELDS: Tuple[str, ...] = (
    "source_dataset", "source_version", "source_record_id",
    "source_url_or_repo", "source_license", "source_file_sha256",
    "arrival_timestamp", "relative_arrival_time", "interarrival_time",
    "prompt_tokens", "output_tokens", "total_tokens",
    "model_name", "model_family", "generation_params",
    "session_id", "request_type", "concurrency_metadata", "prefix_cache_id",
    "priority", "deadline", "slo", "tenant_class",
)


@dataclass
class CorpusRecord:
    source_dataset: str
    source_version: str
    source_record_id: str
    source_url_or_repo: str
    source_license: str
    source_file_sha256: str
    arrival_timestamp: Optional[float]
    relative_arrival_time: float
    interarrival_time: Optional[float]
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    model_name: Optional[str] = None
    model_family: Optional[str] = None
    generation_params: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    request_type: Optional[str] = None
    concurrency_metadata: Optional[Dict[str, Any]] = None
    prefix_cache_id: Optional[str] = None
    priority: Optional[float] = None
    deadline: Optional[float] = None
    slo: Optional[Dict[str, Any]] = None
    tenant_class: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    field_provenance: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngestReport:
    source_dataset: str
    source_file: str
    rows_read: int
    rows_retained: int
    rows_dropped_malformed: int
    time_range_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finalize_partition(
    rows: List[Dict[str, Any]],
    source_dataset: str,
    source_version: str,
    source_url_or_repo: str,
    source_license: str,
    source_file_sha256: str,
) -> List[CorpusRecord]:
    """Sort by arrival timestamp, assign relative_arrival_time/interarrival_time.

    ``rows`` items must carry ``_sort_key`` (float, native units) plus the
    canonical payload fields. Ordering is preserved from the source; this
    only normalizes t0=0, it never shuffles or re-derives values that were
    not present.
    """
    rows_sorted = sorted(rows, key=lambda r: r["_sort_key"])
    records: List[CorpusRecord] = []
    t0 = rows_sorted[0]["_sort_key"] if rows_sorted else 0.0
    prev_rel: Optional[float] = None
    for idx, row in enumerate(rows_sorted):
        rel = row["_sort_key"] - t0
        interarrival = None if prev_rel is None else rel - prev_rel
        prev_rel = rel
        prompt = row.get("prompt_tokens")
        output = row.get("output_tokens")
        total = row.get("total_tokens")
        provenance = dict(row.get("field_provenance", {}))
        if total is None and prompt is not None and output is not None:
            total = prompt + output
            provenance["total_tokens"] = PROVENANCE_DERIVED
        elif "total_tokens" not in provenance:
            provenance["total_tokens"] = (
                PROVENANCE_NATIVE if total is not None else PROVENANCE_UNAVAILABLE
            )
        provenance.setdefault("interarrival_time", PROVENANCE_DERIVED if interarrival is not None else PROVENANCE_UNAVAILABLE)
        provenance.setdefault("relative_arrival_time", PROVENANCE_DERIVED)
        # Partition-level identity/provenance constants are not per-row dict
        # keys (they are function parameters), so seed them explicitly
        # before the generic row.get() fallback below would mis-mark them
        # UNAVAILABLE.
        for name in ("source_dataset", "source_version", "source_url_or_repo", "source_license", "source_file_sha256"):
            provenance.setdefault(name, PROVENANCE_NATIVE)
        for name in CANONICAL_FIELDS:
            if name in provenance:
                continue
            provenance[name] = (
                PROVENANCE_NATIVE if row.get(name) is not None else PROVENANCE_UNAVAILABLE
            )
        records.append(
            CorpusRecord(
                source_dataset=source_dataset,
                source_version=source_version,
                source_record_id=str(row.get("source_record_id", idx)),
                source_url_or_repo=source_url_or_repo,
                source_license=source_license,
                source_file_sha256=source_file_sha256,
                arrival_timestamp=row.get("arrival_timestamp"),
                relative_arrival_time=rel,
                interarrival_time=interarrival,
                prompt_tokens=prompt,
                output_tokens=output,
                total_tokens=total,
                model_name=row.get("model_name"),
                model_family=row.get("model_family"),
                generation_params=row.get("generation_params"),
                session_id=row.get("session_id"),
                request_type=row.get("request_type"),
                concurrency_metadata=row.get("concurrency_metadata"),
                prefix_cache_id=row.get("prefix_cache_id"),
                priority=row.get("priority"),
                deadline=row.get("deadline"),
                slo=row.get("slo"),
                tenant_class=row.get("tenant_class"),
                extra=row.get("extra", {}),
                field_provenance=provenance,
            )
        )
    return records


def ingest_burstgpt(
    path: Union[str, Path],
    source_url_or_repo: str = "https://github.com/HPMLL/BurstGPT",
    source_license: str = "CC-BY-4.0",
    source_version: str = "BurstGPT_1.csv",
    max_rows: Optional[int] = None,
) -> Tuple[List[CorpusRecord], IngestReport]:
    from .burstgpt import detect_burstgpt_schema

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BurstGPT CSV not found: {path}")
    file_hash = sha256_file(path)
    rows: List[Dict[str, Any]] = []
    rows_read = 0
    dropped = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("BurstGPT CSV has no header row")
        schema = detect_burstgpt_schema(list(reader.fieldnames))
        for i, row in enumerate(reader):
            rows_read += 1
            try:
                ts = float(row[schema["timestamp"]])
                prompt = int(row[schema["request_tokens"]])
                output = int(row[schema["response_tokens"]])
            except (TypeError, ValueError, KeyError):
                dropped += 1
                continue
            if prompt <= 0 or output <= 0:
                dropped += 1
                continue
            rows.append({
                "_sort_key": ts,
                "arrival_timestamp": ts,
                "source_record_id": i,
                "prompt_tokens": prompt,
                "output_tokens": output,
                "model_name": (row.get(schema["model"]) or None) if schema["model"] else None,
                "request_type": (row.get(schema["log_type"]) or None) if schema["log_type"] else None,
                "session_id": (row.get(schema["session_id"]) or None) if schema["session_id"] else None,
            })
            if max_rows is not None and len(rows) >= max_rows:
                break

    records = _finalize_partition(
        rows, "burstgpt", source_version, source_url_or_repo, source_license, file_hash,
    )
    time_range = records[-1].relative_arrival_time if records else 0.0
    report = IngestReport(
        source_dataset="burstgpt",
        source_file=str(path),
        rows_read=rows_read,
        rows_retained=len(records),
        rows_dropped_malformed=dropped,
        time_range_seconds=time_range,
    )
    return records, report


def _parse_azure_timestamp(ts: str) -> float:
    ts = ts.strip()
    if "." in ts:
        base, frac = ts.rsplit(".", 1)
        frac = frac[:6]
        ts = f"{base}.{frac}"
    ts = ts.replace(" ", "T")
    return datetime.fromisoformat(ts).timestamp()


def ingest_azure(
    path: Union[str, Path],
    source_dataset: str,
    source_url_or_repo: str = "https://github.com/Azure/AzurePublicDataset",
    source_license: str = "CC-BY-4.0",
    source_version: str = "AzureLLMInferenceDataset2023",
    max_rows: Optional[int] = None,
) -> Tuple[List[CorpusRecord], IngestReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Azure CSV not found: {path}")
    file_hash = sha256_file(path)
    rows: List[Dict[str, Any]] = []
    rows_read = 0
    dropped = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Azure CSV missing required columns {required}; got {reader.fieldnames}")
        for i, row in enumerate(reader):
            rows_read += 1
            try:
                ts = _parse_azure_timestamp(row["TIMESTAMP"])
                ctx = int(row["ContextTokens"])
                gen = int(row["GeneratedTokens"])
            except (TypeError, ValueError, KeyError):
                dropped += 1
                continue
            if ctx <= 0 or gen <= 0:
                dropped += 1
                continue
            rows.append({
                "_sort_key": ts,
                "arrival_timestamp": ts,
                "source_record_id": i,
                "prompt_tokens": ctx,
                "output_tokens": gen,
                "request_type": "code" if "code" in source_dataset else "conversation",
            })
            if max_rows is not None and len(rows) >= max_rows:
                break

    records = _finalize_partition(
        rows, source_dataset, source_version, source_url_or_repo, source_license, file_hash,
    )
    time_range = records[-1].relative_arrival_time if records else 0.0
    report = IngestReport(
        source_dataset=source_dataset,
        source_file=str(path),
        rows_read=rows_read,
        rows_retained=len(records),
        rows_dropped_malformed=dropped,
        time_range_seconds=time_range,
    )
    return records, report


@dataclass
class ExternalValidationMetadata:
    """Reference record for a source that provides real-system PERFORMANCE
    outcomes (throughput/TTFT/TPOT/latency) rather than per-request workload
    INPUT fields. Deliberately not a ``CorpusRecord``: mixing these into the
    workload-input table would violate the Layer-1 no-outcome-leakage rule.
    """

    source_dataset: str
    source_url_or_repo: str
    source_license: str
    source_file_sha256: str
    columns: List[str]
    n_rows: int
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def inspect_agentperfbench_trace_replay(
    path: Union[str, Path],
    source_url_or_repo: str = "https://huggingface.co/datasets/agent-perf-bench/AgentPerfBench",
    source_license: str = "Apache-2.0",
) -> ExternalValidationMetadata:
    """Classify AgentPerfBench's ``trace_replay`` config.

    Verified against the live dataset card and file (2026-08-19): every
    config in this dataset (``trace_replay``, ``synthetic_distributional``,
    ``per_layer_kernel``, ``mse_validation``) contains only RUN-LEVEL
    aggregate performance summaries (request_throughput, TTFT/TPOT/ITL/E2E
    latency percentiles) keyed by (model, hardware, engine, profile,
    concurrency, num_requests, duration_s) — there is no per-request
    prompt/output token column anywhere in the release, despite the dataset
    card's prose description ("Replays exact ISL/OSL sequences from recorded
    agent sessions") describing the *methodology* used to generate the
    workload, not a published per-request artifact.

    This function therefore does NOT return ``CorpusRecord`` objects (there
    is no workload-input row to build one from). It returns
    ``ExternalValidationMetadata`` so the source is correctly filed as
    REAL_SYSTEM_VALIDATION_SOURCE, kept out of the workload-input corpus,
    per docs/design/PUBLIC_TRACE_CORPUS_V1.md.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AgentPerfBench parquet not found: {path}")
    file_hash = sha256_file(path)
    df = pd.read_parquet(path)
    return ExternalValidationMetadata(
        source_dataset="agentperfbench_trace_replay",
        source_url_or_repo=source_url_or_repo,
        source_license=source_license,
        source_file_sha256=file_hash,
        columns=list(df.columns),
        n_rows=len(df),
        note=(
            "Run-level aggregate real-system performance metrics only "
            "(vLLM 0.19.0 / SGLang 0.5.9 on real GPUs); no per-request "
            "prompt/output token fields present. Classified "
            "REAL_SYSTEM_VALIDATION_SOURCE, not TRACE_SOURCE. Not ingested "
            "into the workload-input corpus."
        ),
    )


def write_source_parquet(records: Sequence[CorpusRecord], out_path: Union[str, Path]) -> None:
    import pandas as pd

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        d = r.to_dict()
        for key in ("generation_params", "concurrency_metadata", "slo", "extra", "field_provenance"):
            if d.get(key) is not None:
                d[key] = json.dumps(d[key], sort_keys=True, default=str)
        rows.append(d)
    df = pd.DataFrame(rows, columns=list(CANONICAL_FIELDS) + ["extra", "field_provenance"])
    df.to_parquet(out_path, index=False)


def schema_coverage_row(source_dataset: str, records: Sequence[CorpusRecord]) -> Dict[str, str]:
    """One row for source_coverage.csv: per canonical field, the dominant provenance state."""
    row: Dict[str, str] = {"source_dataset": source_dataset, "n_records": str(len(records))}
    if not records:
        for name in CANONICAL_FIELDS:
            row[name] = PROVENANCE_UNAVAILABLE
        return row
    for name in CANONICAL_FIELDS:
        states = {r.field_provenance.get(name, PROVENANCE_UNAVAILABLE) for r in records}
        row[name] = states.pop() if len(states) == 1 else "MIXED:" + ",".join(sorted(states))
    return row
