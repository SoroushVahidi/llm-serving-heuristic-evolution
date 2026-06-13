"""
Extended trace JSONL format with source metadata.

Reads/writes an extended JSONL format that includes source provenance and
a list of synthetic fields. The core 8 Request fields are always present.
The metadata field is optional and ignored by the base simulator.

Extended format example:
{
    "request_id": 0,
    "arrival_time": 0.125,
    "prompt_tokens": 512,
    "predicted_output_tokens": 180,
    "actual_output_tokens": 220,
    "priority": 1.0,
    "class_id": "interactive",
    "slo_deadline": 3.125,
    "source": "burstgpt",
    "metadata": {
        "original_timestamp": 1700000000.0,
        "synthetic_fields": ["predicted_output_tokens", "class_id", "priority", "slo_deadline"]
    }
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.types import Request

_CORE_FIELDS = [
    "request_id",
    "arrival_time",
    "prompt_tokens",
    "predicted_output_tokens",
    "actual_output_tokens",
    "slo_deadline",
    "priority",
    "class_id",
]


def save_extended_jsonl(
    requests: List[Request],
    path: Union[str, Path],
    source: str = "",
    metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, r in enumerate(requests):
            row: Dict[str, Any] = {
                "request_id": r.request_id,
                "arrival_time": r.arrival_time,
                "prompt_tokens": r.prompt_tokens,
                "predicted_output_tokens": r.predicted_output_tokens,
                "actual_output_tokens": r.actual_output_tokens,
                "slo_deadline": r.slo_deadline,
                "priority": r.priority,
                "class_id": r.class_id,
            }
            if source:
                row["source"] = source
            if metadata_list is not None and i < len(metadata_list):
                md = metadata_list[i]
                if md is not None:
                    row["metadata"] = md
            f.write(json.dumps(row) + "\n")


def load_extended_jsonl(
    path: Union[str, Path],
) -> Tuple[List[Request], List[Dict[str, Any]]]:
    requests: List[Request] = []
    metadata_list: List[Dict[str, Any]] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            core = {k: d[k] for k in _CORE_FIELDS}
            requests.append(Request(**core))

            extra: Dict[str, Any] = {}
            if "source" in d:
                extra["source"] = d["source"]
            if "metadata" in d:
                extra["metadata"] = d["metadata"]
            metadata_list.append(extra)

    return requests, metadata_list
