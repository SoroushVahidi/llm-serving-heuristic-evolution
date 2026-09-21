"""
Trace serialization: read/write requests as JSONL and CSV.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Union

from ..core.types import Request


def save_jsonl(requests: List[Request], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in requests:
            row = {
                "request_id": r.request_id,
                "arrival_time": r.arrival_time,
                "prompt_tokens": r.prompt_tokens,
                "predicted_output_tokens": r.predicted_output_tokens,
                "actual_output_tokens": r.actual_output_tokens,
                "slo_deadline": r.slo_deadline,
                "priority": r.priority,
                "class_id": r.class_id,
            }
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: Union[str, Path]) -> List[Request]:
    requests: List[Request] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            requests.append(Request(**d))
    return requests


def save_csv(requests: List[Request], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "request_id", "arrival_time", "prompt_tokens", "predicted_output_tokens",
        "actual_output_tokens", "slo_deadline", "priority", "class_id",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in requests:
            writer.writerow({
                "request_id": r.request_id,
                "arrival_time": r.arrival_time,
                "prompt_tokens": r.prompt_tokens,
                "predicted_output_tokens": r.predicted_output_tokens,
                "actual_output_tokens": r.actual_output_tokens,
                "slo_deadline": r.slo_deadline,
                "priority": r.priority,
                "class_id": r.class_id,
            })


def load_csv(path: Union[str, Path]) -> List[Request]:
    requests: List[Request] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            requests.append(Request(
                request_id=int(row["request_id"]),
                arrival_time=float(row["arrival_time"]),
                prompt_tokens=int(row["prompt_tokens"]),
                predicted_output_tokens=int(row["predicted_output_tokens"]),
                actual_output_tokens=int(row["actual_output_tokens"]),
                slo_deadline=float(row["slo_deadline"]),
                priority=float(row["priority"]),
                class_id=row["class_id"],
            ))
    return requests
