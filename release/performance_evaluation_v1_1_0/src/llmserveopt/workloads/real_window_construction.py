"""
Streaming real-trace window construction for Tier-1 staged JSONL traces.

Designed for multi-million-row files: builds a lightweight catalog in one pass,
then materializes selected windows in a second pass without loading the full
trace into memory.

Scale convention
----------------
``load_factor`` k in {1,2,4,8} means inter-arrival gaps are divided by k
(higher k ⇒ denser arrivals). Natural replay uses k=1.
Stored ``time_scale`` equals ``1/k`` for compatibility with
``scale_interarrivals`` (which multiplies gaps by time_scale).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Request
from .canonical_schema import ReplayLabel


WINDOW_ORIGIN_NATURAL = "natural_replay"
WINDOW_ORIGIN_BUSY = "natural_busy_period"
WINDOW_ORIGIN_SCALED = "trace_derived_time_scaled"
WINDOW_ORIGIN_SYNTHETIC = "trace_calibrated_synthetic"

BUSY_SELECTION_RULE = "request_rate_ge_empirical_p80_among_natural_windows"
DEFAULT_REQUEST_WINDOW_SIZE = 800
DEFAULT_MIN_WINDOW_REQUESTS = 80
DEFAULT_LOAD_FACTORS = (1, 2, 4, 8)


@dataclass
class WindowCatalogEntry:
    window_id: str
    source_file: str
    source_family: str
    start_index: int
    end_index: int  # exclusive
    start_arrival: float
    end_arrival: float
    n_requests: int
    prompt_sum: float
    output_sum: float
    chronological_split: str = "unassigned"
    window_origin: str = WINDOW_ORIGIN_NATURAL
    load_factor: int = 1
    time_scale: float = 1.0
    busy_selection_rule: Optional[str] = None
    parent_window_id: Optional[str] = None
    session_preserving: bool = False
    redistribution: str = "allowed_with_dataset_license"
    evaluation_role: str = "train_val_test_candidate"


@dataclass
class MaterializedWindowMeta:
    catalog: WindowCatalogEntry
    fingerprint: Dict[str, Any] = field(default_factory=dict)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def record_to_request(rec: Dict[str, Any], request_id: int) -> Request:
    return Request(
        request_id=request_id,
        arrival_time=float(rec["arrival_time"]),
        prompt_tokens=int(rec["prompt_tokens"]),
        predicted_output_tokens=int(rec["predicted_output_tokens"]),
        actual_output_tokens=int(rec["actual_output_tokens"]),
        slo_deadline=float(rec["slo_deadline"]),
        priority=float(rec["priority"]),
        class_id=str(rec["class_id"]),
    )


def apply_load_factor(requests: Sequence[Request], load_factor: int) -> List[Request]:
    """Divide inter-arrival gaps by load_factor; preserve order and lengths."""
    if load_factor < 1:
        raise ValueError(f"load_factor must be >= 1, got {load_factor}")
    if not requests:
        return []
    if load_factor == 1:
        return [
            Request(
                request_id=i,
                arrival_time=float(r.arrival_time) - float(requests[0].arrival_time),
                prompt_tokens=r.prompt_tokens,
                predicted_output_tokens=r.predicted_output_tokens,
                actual_output_tokens=r.actual_output_tokens,
                slo_deadline=float(r.slo_deadline) - float(requests[0].arrival_time),
                priority=r.priority,
                class_id=r.class_id,
            )
            for i, r in enumerate(requests)
        ]
    t0 = float(requests[0].arrival_time)
    out: List[Request] = []
    prev_scaled = 0.0
    prev_raw = t0
    for i, r in enumerate(requests):
        raw = float(r.arrival_time)
        if i == 0:
            arr = 0.0
        else:
            gap = max(0.0, raw - prev_raw)
            arr = prev_scaled + gap / float(load_factor)
        # Preserve relative SLO slack when possible.
        slack = max(0.0, float(r.slo_deadline) - raw)
        out.append(
            Request(
                request_id=i,
                arrival_time=arr,
                prompt_tokens=r.prompt_tokens,
                predicted_output_tokens=r.predicted_output_tokens,
                actual_output_tokens=r.actual_output_tokens,
                slo_deadline=arr + slack / float(load_factor) if load_factor > 1 else arr + slack,
                priority=r.priority,
                class_id=r.class_id,
            )
        )
        prev_scaled = arr
        prev_raw = raw
    return out


def fingerprint_requests(
    requests: Sequence[Request],
    *,
    window_origin: str,
    load_factor: int,
    chronological_split: str,
    source_family: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    n = len(requests)
    if n == 0:
        return {"n_requests": 0}
    arrivals = np.asarray([r.arrival_time for r in requests], dtype=float)
    prompts = np.asarray([r.prompt_tokens for r in requests], dtype=float)
    outputs = np.asarray([r.actual_output_tokens for r in requests], dtype=float)
    preds = np.asarray([r.predicted_output_tokens for r in requests], dtype=float)
    prios = np.asarray([r.priority for r in requests], dtype=float)
    slacks = np.asarray([r.slo_deadline - r.arrival_time for r in requests], dtype=float)
    gaps = np.diff(arrivals) if n > 1 else np.asarray([], dtype=float)
    duration = float(arrivals[-1] - arrivals[0]) if n > 1 else 0.0
    pred_err = preds - outputs

    def pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q)) if len(arr) else float("nan")

    cv = float(np.std(gaps) / np.mean(gaps)) if len(gaps) and np.mean(gaps) > 0 else float("nan")
    corr = float("nan")
    if n >= 2 and np.std(prompts) > 0 and np.std(outputs) > 0:
        corr = float(np.corrcoef(prompts, outputs)[0, 1])

    fp: Dict[str, Any] = {
        "n_requests": n,
        "duration_s": duration,
        "request_arrival_rate": _safe_div(n, duration),
        "prompt_token_arrival_rate": _safe_div(float(np.sum(prompts)), duration),
        "output_token_arrival_rate": _safe_div(float(np.sum(outputs)), duration),
        "total_token_arrival_rate": _safe_div(float(np.sum(prompts) + np.sum(outputs)), duration),
        "interarrival_cv": cv,
        "interarrival_mean": float(np.mean(gaps)) if len(gaps) else float("nan"),
        "prompt_p50": pct(prompts, 50),
        "prompt_p90": pct(prompts, 90),
        "prompt_p95": pct(prompts, 95),
        "prompt_p99": pct(prompts, 99),
        "output_p50": pct(outputs, 50),
        "output_p90": pct(outputs, 90),
        "output_p95": pct(outputs, 95),
        "output_p99": pct(outputs, 99),
        "prompt_output_correlation": corr,
        "long_context_fraction_ge_4k": float(np.mean(prompts >= 4096)),
        "priority_mean": float(np.mean(prios)),
        "slo_slack_p50": pct(slacks, 50),
        "slo_slack_p95": pct(slacks, 95),
        "prediction_error_mean": float(np.mean(pred_err)),
        "prediction_error_mae": float(np.mean(np.abs(pred_err))),
        "window_origin": window_origin,
        "load_factor": load_factor,
        "time_scale": 1.0 / float(load_factor),
        "replay_label": (
            ReplayLabel.NATURAL_TRACE_REPLAY.value
            if load_factor == 1 and window_origin != WINDOW_ORIGIN_SYNTHETIC
            else "trace-derived, time-scaled"
            if window_origin == WINDOW_ORIGIN_SCALED
            else window_origin
        ),
        "chronological_split": chronological_split,
        "source_family": source_family,
    }
    if extra:
        fp.update(extra)
    return fp


def build_catalog_streaming(
    path: Path,
    *,
    source_family: str,
    request_window_size: int = DEFAULT_REQUEST_WINDOW_SIZE,
    min_window_requests: int = DEFAULT_MIN_WINDOW_REQUESTS,
    max_natural_windows: Optional[int] = None,
) -> Tuple[List[WindowCatalogEntry], Dict[str, Any]]:
    """One-pass catalog of contiguous request-count windows."""
    entries: List[WindowCatalogEntry] = []
    start_index = 0
    start_arrival = 0.0
    end_arrival = 0.0
    prompt_sum = 0.0
    output_sum = 0.0
    count = 0
    total_rows = 0
    first_arrival: Optional[float] = None
    last_arrival: Optional[float] = None
    prev_arrival: Optional[float] = None
    nondecreasing = True
    negative_arrivals = 0
    file_stem = path.stem

    def flush(end_index: int) -> None:
        nonlocal start_index, prompt_sum, output_sum, count, start_arrival, end_arrival
        if count < min_window_requests:
            start_index = end_index
            prompt_sum = 0.0
            output_sum = 0.0
            count = 0
            return
        wid = f"{file_stem}__w{len(entries):05d}"
        entries.append(
            WindowCatalogEntry(
                window_id=wid,
                source_file=str(path),
                source_family=source_family,
                start_index=start_index,
                end_index=end_index,
                start_arrival=float(start_arrival),
                end_arrival=float(end_arrival),
                n_requests=count,
                prompt_sum=prompt_sum,
                output_sum=output_sum,
            )
        )
        start_index = end_index
        prompt_sum = 0.0
        output_sum = 0.0
        count = 0

    for rec in iter_jsonl_records(path):
        arr = float(rec["arrival_time"])
        if arr < 0:
            negative_arrivals += 1
        if first_arrival is None:
            first_arrival = arr
        if prev_arrival is not None and arr < prev_arrival:
            nondecreasing = False
        prev_arrival = arr
        last_arrival = arr
        if count == 0:
            start_arrival = arr
        end_arrival = arr
        prompt_sum += float(rec["prompt_tokens"])
        output_sum += float(rec["actual_output_tokens"])
        count += 1
        total_rows += 1
        if count >= request_window_size:
            flush(total_rows)
            if max_natural_windows is not None and len(entries) >= max_natural_windows:
                break

    if count >= min_window_requests and (
        max_natural_windows is None or len(entries) < max_natural_windows
    ):
        flush(total_rows)

    # Chronological split by window mid-time within this file's span.
    if entries and first_arrival is not None and last_arrival is not None:
        span = max(last_arrival - first_arrival, 1e-9)
        for e in entries:
            mid = 0.5 * (e.start_arrival + e.end_arrival)
            frac = (mid - first_arrival) / span
            if frac < 0.60:
                e.chronological_split = "train"
            elif frac < 0.80:
                e.chronological_split = "validation"
            else:
                e.chronological_split = "heldout"

    report = {
        "source_file": str(path),
        "source_family": source_family,
        "rows_scanned": total_rows,
        "n_catalog_windows": len(entries),
        "first_arrival": first_arrival,
        "last_arrival": last_arrival,
        "negative_arrivals": negative_arrivals,
        "nondecreasing_arrivals": nondecreasing,
        "request_window_size": request_window_size,
        "min_window_requests": min_window_requests,
    }
    return entries, report


def mark_busy_windows(
    entries: Sequence[WindowCatalogEntry],
    *,
    quantile: float = 0.80,
) -> List[WindowCatalogEntry]:
    """Clone top-rate natural windows as busy-period variants."""
    rates = []
    for e in entries:
        dur = max(e.end_arrival - e.start_arrival, 1e-9)
        rates.append(e.n_requests / dur)
    if not rates:
        return []
    thr = float(np.quantile(np.asarray(rates, dtype=float), quantile))
    busy: List[WindowCatalogEntry] = []
    for e, rate in zip(entries, rates):
        if rate < thr:
            continue
        b = WindowCatalogEntry(**asdict(e))
        b.window_id = f"{e.window_id}__busy"
        b.window_origin = WINDOW_ORIGIN_BUSY
        b.busy_selection_rule = BUSY_SELECTION_RULE
        b.parent_window_id = e.window_id
        busy.append(b)
    return busy


def mark_scaled_windows(
    entries: Sequence[WindowCatalogEntry],
    *,
    load_factors: Sequence[int] = DEFAULT_LOAD_FACTORS,
    max_per_factor: int = 12,
) -> List[WindowCatalogEntry]:
    """Create disclosed scaled variants from natural train windows."""
    train = [e for e in entries if e.chronological_split == "train"]
    out: List[WindowCatalogEntry] = []
    for k in load_factors:
        if int(k) == 1:
            continue
        selected = train[:max_per_factor]
        for e in selected:
            s = WindowCatalogEntry(**asdict(e))
            s.window_id = f"{e.window_id}__x{int(k)}"
            s.window_origin = WINDOW_ORIGIN_SCALED
            s.load_factor = int(k)
            s.time_scale = 1.0 / float(k)
            s.parent_window_id = e.window_id
            out.append(s)
    return out


def extract_window_requests(
    path: Path,
    start_index: int,
    end_index: int,
) -> List[Request]:
    """Second-pass extract of [start_index, end_index) without full load."""
    reqs: List[Request] = []
    for i, rec in enumerate(iter_jsonl_records(path)):
        if i < start_index:
            continue
        if i >= end_index:
            break
        reqs.append(record_to_request(rec, request_id=len(reqs)))
    return reqs


def write_window_jsonl(
    requests: Sequence[Request],
    path: Path,
    *,
    meta: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "w", encoding="utf-8") as f:
        header = {"record_type": "window_meta", **meta}
        f.write(json.dumps(header) + "\n")
        for r in requests:
            f.write(
                json.dumps(
                    {
                        "record_type": "request",
                        "request_id": r.request_id,
                        "arrival_time": r.arrival_time,
                        "prompt_tokens": r.prompt_tokens,
                        "predicted_output_tokens": r.predicted_output_tokens,
                        "actual_output_tokens": r.actual_output_tokens,
                        "slo_deadline": r.slo_deadline,
                        "priority": r.priority,
                        "class_id": r.class_id,
                    }
                )
                + "\n"
            )
    tmp.replace(path)


def load_window_jsonl(path: Path) -> Tuple[Dict[str, Any], List[Request]]:
    meta: Dict[str, Any] = {}
    reqs: List[Request] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") == "window_meta":
                meta = rec
                continue
            reqs.append(record_to_request(rec, request_id=len(reqs)))
    return meta, reqs


def validate_window_requests(
    requests: Sequence[Request],
    *,
    expect_nondecreasing: bool = True,
) -> List[str]:
    issues: List[str] = []
    if not requests:
        issues.append("empty_window")
        return issues
    prev = -1.0
    ids = set()
    for r in requests:
        if r.arrival_time < 0:
            issues.append("negative_arrival")
        if r.prompt_tokens < 0 or r.actual_output_tokens < 0:
            issues.append("negative_length")
        if r.request_id in ids:
            issues.append("duplicate_request_id")
        ids.add(r.request_id)
        if expect_nondecreasing and r.arrival_time < prev:
            issues.append("non_chronological")
            break
        prev = r.arrival_time
    # Leakage protection: ObservableRequest excludes actual_output_tokens
    from ..core.types import ObservableRequest

    for r in requests[:3]:
        obs = ObservableRequest.from_request(r)
        if hasattr(obs, "actual_output_tokens"):
            issues.append("actual_output_leakage")
            break
    return sorted(set(issues))


def fit_and_sample_synthetic(
    train_windows: Sequence[Tuple[WindowCatalogEntry, List[Request]]],
    *,
    n_windows: int,
    window_size: int,
    seed: int,
    source_fit_dataset: str,
) -> List[Tuple[Dict[str, Any], List[Request]]]:
    """Empirical resample generator fit on training windows only."""
    if not train_windows:
        return []
    rng = np.random.default_rng(seed)
    gaps: List[float] = []
    prompts: List[int] = []
    outputs: List[int] = []
    preds: List[int] = []
    prios: List[float] = []
    slacks: List[float] = []
    classes: List[str] = []
    for _entry, reqs in train_windows:
        arr = [r.arrival_time for r in reqs]
        for i in range(1, len(arr)):
            gaps.append(max(0.0, arr[i] - arr[i - 1]))
        for r in reqs:
            prompts.append(int(r.prompt_tokens))
            outputs.append(int(r.actual_output_tokens))
            preds.append(int(r.predicted_output_tokens))
            prios.append(float(r.priority))
            slacks.append(max(0.0, float(r.slo_deadline) - float(r.arrival_time)))
            classes.append(str(r.class_id))
    if not gaps:
        gaps = [0.1]
    out: List[Tuple[Dict[str, Any], List[Request]]] = []
    for w_i in range(n_windows):
        n = int(window_size)
        g = rng.choice(np.asarray(gaps, dtype=float), size=max(n - 1, 0), replace=True)
        arrivals = np.concatenate([[0.0], np.cumsum(g)]) if n > 1 else np.asarray([0.0])
        reqs: List[Request] = []
        for i in range(n):
            p = int(rng.choice(prompts))
            o = int(rng.choice(outputs))
            pr = int(rng.choice(preds))
            pri = float(rng.choice(prios))
            sl = float(rng.choice(slacks))
            cls = str(rng.choice(classes))
            reqs.append(
                Request(
                    request_id=i,
                    arrival_time=float(arrivals[i]),
                    prompt_tokens=max(1, p),
                    predicted_output_tokens=max(1, pr),
                    actual_output_tokens=max(0, o),
                    slo_deadline=float(arrivals[i] + sl),
                    priority=pri,
                    class_id=cls,
                )
            )
        meta = {
            "window_id": f"synth_{source_fit_dataset}_{w_i:04d}",
            "window_origin": WINDOW_ORIGIN_SYNTHETIC,
            "source_fit_dataset": source_fit_dataset,
            "source_fit_split": "train",
            "load_factor": 1,
            "time_scale": 1.0,
            "seed": seed,
            "generator": "empirical_resample_v1",
        }
        out.append((meta, reqs))
    return out


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def write_marker(path: Path, status: str, **extra: Any) -> None:
    payload = {"status": status, **extra}
    atomic_write_json(path, payload)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
