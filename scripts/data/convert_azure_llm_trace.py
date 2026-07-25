#!/usr/bin/env python3
"""
Convert Azure LLM Inference trace CSV to simulator JSONL format.

Handles both 2023 and 2024 Azure LLM traces (schema is identical):
  TIMESTAMP, ContextTokens, GeneratedTokens

Augments with synthetic fields: predicted_output_tokens, priority, class_id, slo_deadline.
Follows the same conversion pattern as convert_burstgpt.py.

Usage
-----
python scripts/data/convert_azure_llm_trace.py \\
    --input data/raw/azure/AzureLLMInferenceTrace_code_2023.csv \\
    --output data/processed/azure/azure_llm_2023_code.jsonl \\
    --source azure_2023_code \\
    --time-scale 0.05 \\
    --max-requests 10000 \\
    --seed 17

Time-scale guidance
-------------------
Azure 2023 traces have natural arrival rates of ~2.6 req/s (code) and ~5.5 req/s (conv),
spanning ~1 hour each.  To produce ~50 req/s (comparable to BurstGPT scaled variants):
  code trace:  --time-scale 0.05  (2.6 / 0.05 = 52 req/s effective)
  conv trace:  --time-scale 0.10  (5.5 / 0.10 = 55 req/s effective)

Azure 2024 traces are 1-week files (691 MB + 1.1 GB); use --max-requests 10000.

Citations
---------
Azure 2023: Patel et al., "Splitwise" ISCA 2024. CC-BY License.
Azure 2024: Stojkovic et al., "DynamoLLM" HPCA 2025. CC-BY License.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.core.types import Request
from llmserveopt.workloads.augmentation import AugmentationConfig, augment_trace
from llmserveopt.workloads.trace_io_extended import save_extended_jsonl


def _parse_azure_timestamp(ts: str) -> float:
    """Parse Azure TIMESTAMP string (handles up to 7 fractional digits)."""
    ts = ts.strip()
    if "." in ts:
        base, frac = ts.rsplit(".", 1)
        frac = frac[:6]
        ts = f"{base}.{frac}"
    ts = ts.replace(" ", "T")
    return datetime.fromisoformat(ts).timestamp()


def load_azure_csv(
    path: Path,
    max_requests: Optional[int] = None,
    time_scale: float = 1.0,
    min_context_tokens: int = 1,
    min_generated_tokens: int = 1,
) -> Tuple[List[Request], dict]:
    """Load Azure LLM CSV and return (requests, report)."""
    raw_rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ctx = int(row["ContextTokens"])
            gen = int(row["GeneratedTokens"])
            if ctx < min_context_tokens or gen < min_generated_tokens:
                continue
            raw_rows.append({"ts_str": row["TIMESTAMP"], "context": ctx, "generated": gen})

    if not raw_rows:
        raise ValueError(f"No valid rows in {path}")

    if max_requests is not None:
        raw_rows = raw_rows[:max_requests]

    # Parse timestamps; if file order is not chronological, sort by wall-clock
    # (disclosed) so relative arrivals are nonnegative and monotonic.
    for i, r in enumerate(raw_rows):
        r["file_index"] = i
        r["ts_unix"] = _parse_azure_timestamp(r["ts_str"])
    inversions = sum(
        1
        for i in range(1, len(raw_rows))
        if raw_rows[i]["ts_unix"] < raw_rows[i - 1]["ts_unix"]
    )
    sorted_by_ts = inversions > 0
    if sorted_by_ts:
        raw_rows.sort(key=lambda r: (r["ts_unix"], r["file_index"]))

    raw_ts = np.array([r["ts_unix"] for r in raw_rows], dtype=float)
    t0 = raw_ts[0]
    raw_ts = raw_ts - t0

    if time_scale != 1.0 and len(raw_ts) > 1:
        interarrivals = np.diff(raw_ts)
        arrival_times = np.concatenate([[0.0], np.cumsum(interarrivals * time_scale)])
    else:
        arrival_times = raw_ts

    if np.any(arrival_times < 0):
        raise ValueError(
            "Azure conversion produced negative relative arrival times "
            f"(min={float(np.min(arrival_times))})"
        )

    context_tokens = np.array([r["context"] for r in raw_rows], dtype=int)
    generated_tokens = np.array([r["generated"] for r in raw_rows], dtype=int)

    # Augmentation will fill predicted_output_tokens, class_id, priority, slo_deadline
    # (actual_output_tokens = generated_tokens from the real trace)

    time_range = float(arrival_times[-1]) if len(arrival_times) > 1 else 0.0
    mean_rate = len(arrival_times) / time_range if time_range > 0 else 0.0

    report = {
        "rows_read": len(raw_rows),
        "rows_retained": len(raw_rows),
        "rows_dropped": 0,
        "time_scale": time_scale,
        "time_range_seconds": round(time_range, 3),
        "mean_arrival_rate": round(mean_rate, 3),
        "context_tokens_mean": round(float(np.mean(context_tokens)), 1),
        "context_tokens_p50": float(np.median(context_tokens)),
        "context_tokens_p95": float(np.percentile(context_tokens, 95)),
        "context_tokens_max": int(np.max(context_tokens)),
        "generated_tokens_mean": round(float(np.mean(generated_tokens)), 1),
        "generated_tokens_p50": float(np.median(generated_tokens)),
        "generated_tokens_p95": float(np.percentile(generated_tokens, 95)),
        "generated_tokens_max": int(np.max(generated_tokens)),
        "file_order_inversions": inversions,
        "sorted_by_wall_clock_timestamp": sorted_by_ts,
    }
    # Return raw data as placeholder requests (augment after)
    return arrival_times, context_tokens, generated_tokens, report


def convert_azure_to_requests(
    arrival_times: np.ndarray,
    context_tokens: np.ndarray,
    generated_tokens: np.ndarray,
    aug_cfg: AugmentationConfig,
    seed: int = 17,
) -> List[Request]:
    """Apply augmentation and build Request objects."""
    rng = np.random.default_rng(seed)
    augmented = augment_trace(generated_tokens, arrival_times, aug_cfg, rng)

    requests: List[Request] = []
    for i in range(len(arrival_times)):
        requests.append(Request(
            request_id=i,
            arrival_time=float(arrival_times[i]),
            prompt_tokens=int(context_tokens[i]),
            predicted_output_tokens=int(augmented["predicted_output_tokens"][i]),
            actual_output_tokens=int(generated_tokens[i]),
            slo_deadline=float(augmented["slo_deadlines"][i]),
            priority=float(augmented["priorities"][i]),
            class_id=augmented["class_ids"][i],
        ))
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Azure LLM inference CSV to simulator JSONL"
    )
    parser.add_argument("--input", required=True, help="Path to Azure LLM CSV file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--source", default="azure",
                        help="Source tag embedded in output JSONL (e.g. azure_2023_code)")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Multiply inter-arrival gaps by this factor (default 1.0)")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="Max requests to include (default: all)")
    parser.add_argument("--seed", type=int, default=17,
                        help="Random seed for SLO/noise augmentation")
    parser.add_argument("--min-context-tokens", type=int, default=1)
    parser.add_argument("--min-generated-tokens", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting Azure LLM trace: {input_path.name}")
    print(f"  Output:     {args.output}")
    print(f"  Source tag: {args.source}")
    print(f"  time_scale: {args.time_scale}")
    print(f"  Seed:       {args.seed}")

    arrival_times, context_tokens, generated_tokens, report = load_azure_csv(
        input_path,
        max_requests=args.max_requests,
        time_scale=args.time_scale,
        min_context_tokens=args.min_context_tokens,
        min_generated_tokens=args.min_generated_tokens,
    )

    print(f"\n  Loaded: {report['rows_retained']} requests")
    print(f"  Span:   {report['time_range_seconds']:.1f}s  rate: {report['mean_arrival_rate']:.2f} req/s")
    print(f"  Context tokens:   mean={report['context_tokens_mean']:.0f}  p95={report['context_tokens_p95']:.0f}")
    print(f"  Generated tokens: mean={report['generated_tokens_mean']:.0f}  p95={report['generated_tokens_p95']:.0f}")

    aug_cfg = AugmentationConfig()
    requests = convert_azure_to_requests(
        arrival_times, context_tokens, generated_tokens, aug_cfg, seed=args.seed
    )
    print(f"\nAugmented with synthetic SLO/priority/prediction fields (seed={args.seed})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_extended_jsonl(requests, output_path, source=args.source)
    print(f"Saved {len(requests)} requests → {output_path}")

    report["seed"] = args.seed
    report["source"] = args.source
    report["augmentation"] = {
        "prediction_noise": "lognormal sigma=0.35",
        "slo_classes": ["interactive(w=0.50,p=3,slack=2s)",
                        "standard(w=0.35,p=2,slack=6s)",
                        "batch(w=0.15,p=1,slack=20s)"],
    }
    report_path = output_path.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report → {report_path}")


if __name__ == "__main__":
    main()
