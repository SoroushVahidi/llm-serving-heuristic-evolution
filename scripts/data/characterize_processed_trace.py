#!/usr/bin/env python3
"""
Characterize a processed extended JSONL trace (no policy runs).

Computes size/temporal coverage, token distributions, arrival stats,
optional session/class/prefix summaries. Does not print prompt text.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.workloads.trace_io_extended import load_extended_jsonl


def _pct(arr: np.ndarray, qs: List[float]) -> Dict[str, float]:
    if len(arr) == 0:
        return {f"p{int(q)}": float("nan") for q in qs}
    vals = np.percentile(arr, qs)
    return {f"p{int(q)}": float(v) for q, v in zip(qs, vals)}


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def characterize(requests, metadata_list: Optional[List[dict]] = None) -> Dict[str, Any]:
    n = len(requests)
    if n == 0:
        return {"n_requests": 0}

    arrivals = np.array([r.arrival_time for r in requests], dtype=float)
    prompts = np.array([r.prompt_tokens for r in requests], dtype=float)
    outputs = np.array([r.actual_output_tokens for r in requests], dtype=float)
    totals = prompts + outputs
    gaps = np.diff(arrivals) if n > 1 else np.array([])

    duration = float(arrivals[-1] - arrivals[0]) if n > 1 else 0.0
    mean_rate = n / duration if duration > 0 else float("nan")

    # Simple busy windows: 60s bins if span allows.
    busy = {}
    if duration > 0:
        bin_s = 60.0
        bins = np.floor(arrivals / bin_s).astype(int)
        counts = Counter(bins.tolist())
        top = counts.most_common(5)
        busy = {
            "bin_seconds": bin_s,
            "busiest_bins": [
                {"bin_index": int(i), "requests": int(c), "start_s": float(i * bin_s)}
                for i, c in top
            ],
            "peak_to_median_bin_rate": (
                float(max(counts.values()) / max(np.median(list(counts.values())), 1e-9))
            ),
        }

    cv_gap = float(np.std(gaps) / np.mean(gaps)) if len(gaps) and np.mean(gaps) > 0 else float("nan")
    # Index of dispersion for counts in 1s bins.
    iod = float("nan")
    if duration > 1:
        one_s = np.floor(arrivals).astype(int)
        c1 = np.bincount(one_s - one_s.min())
        if np.mean(c1) > 0:
            iod = float(np.var(c1) / np.mean(c1))

    sessions = Counter()
    classes = Counter()
    models = Counter()
    prefix_hashes = Counter()
    types = Counter()
    if metadata_list:
        for md in metadata_list:
            if md.get("session_id"):
                sessions[str(md["session_id"])] += 1
            if md.get("model_id"):
                models[str(md["model_id"])] += 1
            extra = md.get("extra") or {}
            if "log_type" in extra:
                types[str(extra["log_type"])] += 1
            if "request_type" in extra:
                types[str(extra["request_type"])] += 1
            if "type" in extra:
                types[str(extra["type"])] += 1
            hashes = extra.get("hash_ids") or extra.get("prefix_hash_ids")
            if isinstance(hashes, list) and hashes:
                prefix_hashes[str(hashes[0])] += 1
            elif md.get("prefix_id"):
                prefix_hashes[str(md["prefix_id"])] += 1
        for r in requests:
            classes[str(r.class_id)] += 1

    corr = float("nan")
    if n >= 2 and np.std(prompts) > 0 and np.std(outputs) > 0:
        corr = float(np.corrcoef(prompts, outputs)[0, 1])

    return {
        "n_requests": n,
        "duration_seconds": duration,
        "first_arrival": float(arrivals[0]),
        "last_arrival": float(arrivals[-1]),
        "mean_arrival_rate_per_s": mean_rate,
        "requests_per_minute": mean_rate * 60.0 if mean_rate == mean_rate else float("nan"),
        "prompt_tokens": {
            "mean": float(np.mean(prompts)),
            "std": float(np.std(prompts)),
            "max": float(np.max(prompts)),
            **_pct(prompts, [50, 90, 95, 99]),
            "zero_or_missing_fraction": float(np.mean(prompts <= 0)),
        },
        "output_tokens": {
            "mean": float(np.mean(outputs)),
            "std": float(np.std(outputs)),
            "max": float(np.max(outputs)),
            **_pct(outputs, [50, 90, 95, 99]),
            "zero_or_missing_fraction": float(np.mean(outputs <= 0)),
        },
        "total_tokens": {
            "mean": float(np.mean(totals)),
            **_pct(totals, [50, 95, 99]),
            "max": float(np.max(totals)),
        },
        "prompt_output_correlation": corr,
        "interarrival": {
            "mean": float(np.mean(gaps)) if len(gaps) else float("nan"),
            "std": float(np.std(gaps)) if len(gaps) else float("nan"),
            "cv": cv_gap,
            "index_of_dispersion_1s": iod,
            "p50": float(np.median(gaps)) if len(gaps) else float("nan"),
            "p95": float(np.percentile(gaps, 95)) if len(gaps) else float("nan"),
            "autocorr_lag_1": _autocorr(gaps, 1) if len(gaps) else float("nan"),
            "autocorr_lag_10": _autocorr(gaps, 10) if len(gaps) else float("nan"),
        },
        "busy_periods": busy,
        "token_arrival_rate_per_s": float(np.sum(totals) / duration) if duration > 0 else float("nan"),
        "prefill_token_arrival_rate_per_s": float(np.sum(prompts) / duration) if duration > 0 else float("nan"),
        "output_token_arrival_rate_per_s": float(np.sum(outputs) / duration) if duration > 0 else float("nan"),
        "long_context_fraction_ge_4k": float(np.mean(prompts >= 4096)),
        "long_context_fraction_ge_8k": float(np.mean(prompts >= 8192)),
        "long_context_fraction_ge_32k": float(np.mean(prompts >= 32768)),
        "sessions": {
            "n_sessions": len(sessions),
            "requests_per_session_p50": float(np.median(list(sessions.values()))) if sessions else None,
            "requests_per_session_p95": float(np.percentile(list(sessions.values()), 95)) if sessions else None,
        },
        "class_counts": dict(classes),
        "model_counts": dict(models.most_common(20)),
        "type_counts": dict(types.most_common(20)),
        "prefix_reuse": {
            "n_distinct_first_block_hashes": len(prefix_hashes),
            "reusable_prefix_proportion": (
                float(sum(1 for c in prefix_hashes.values() if c > 1) / max(len(prefix_hashes), 1))
                if prefix_hashes
                else None
            ),
            "top_prefix_frequency": prefix_hashes.most_common(5),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    requests, meta = load_extended_jsonl(path)
    report = characterize(requests, meta)
    report["source_file"] = str(path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote characterization → {out}")
    print(
        f"  n={report['n_requests']} duration_s={report.get('duration_seconds')} "
        f"rate={report.get('mean_arrival_rate_per_s')}"
    )


if __name__ == "__main__":
    main()
