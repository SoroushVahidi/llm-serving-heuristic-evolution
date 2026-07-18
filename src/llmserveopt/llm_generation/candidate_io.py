"""
Candidate archive: save and load generation results.

Directory layout per candidate:
  <output_dir>/
    <timestamp>_<provider>_<model>_<id>/
      prompt.json
      raw_response.txt
      candidate.json
      verifier_result.json
      repaired_attempts/
        attempt_<n>_raw.txt
        attempt_<n>_candidate.json
      metadata.json

Index:
  <output_dir>/index.csv
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


DSL_VERSION = "2B.1"

_INDEX_FIELDS = [
    "candidate_id", "provider", "model", "temperature", "verification_ok",
    "repair_attempts", "sha256", "git_commit", "timestamp", "candidate_dir",
    "design_target",
]


@dataclass
class CandidateRecord:
    candidate_id: str
    provider: str
    model: str
    temperature: float
    max_tokens: int
    generation_time: float
    repair_attempt_count: int
    verification_ok: bool
    sha256: str
    git_commit: str
    dsl_version: str = DSL_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)


def _sha256(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def make_candidate_dir(
    output_dir: Path,
    provider: str,
    model: str,
    candidate_id: str,
) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_").replace(":", "_")
    name = f"{ts}_{provider}_{safe_model}_{candidate_id}"
    d = output_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "repaired_attempts").mkdir(exist_ok=True)
    return d


def save_candidate(
    candidate_dir: Path,
    *,
    prompt_messages: List[Dict],
    raw_response: str,
    candidate_json: Optional[Dict],
    verifier_result: Dict,
    metadata: CandidateRecord,
) -> None:
    (candidate_dir / "prompt.json").write_text(
        json.dumps(prompt_messages, indent=2), encoding="utf-8"
    )
    (candidate_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
    if candidate_json is not None:
        (candidate_dir / "candidate.json").write_text(
            json.dumps(candidate_json, indent=2), encoding="utf-8"
        )
    (candidate_dir / "verifier_result.json").write_text(
        json.dumps(verifier_result, indent=2), encoding="utf-8"
    )
    (candidate_dir / "metadata.json").write_text(
        json.dumps(
            {k: v for k, v in asdict(metadata).items() if k != "extra"} | metadata.extra,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_repair_attempt(
    candidate_dir: Path,
    attempt_n: int,
    raw_response: str,
    candidate_json: Optional[Dict],
) -> None:
    d = candidate_dir / "repaired_attempts"
    (d / f"attempt_{attempt_n}_raw.txt").write_text(raw_response, encoding="utf-8")
    if candidate_json is not None:
        (d / f"attempt_{attempt_n}_candidate.json").write_text(
            json.dumps(candidate_json, indent=2), encoding="utf-8"
        )


def update_index(output_dir: Path, record: CandidateRecord, candidate_dir: Path) -> None:
    index_path = output_dir / "index.csv"
    write_header = not index_path.exists()
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "candidate_id": record.candidate_id,
            "provider": record.provider,
            "model": record.model,
            "temperature": record.temperature,
            "verification_ok": record.verification_ok,
            "repair_attempts": record.repair_attempt_count,
            "sha256": record.sha256,
            "git_commit": record.git_commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "candidate_dir": str(candidate_dir.name),
            "design_target": record.extra.get("design_target", ""),
        })


def load_verified_candidates(candidates_dir: Path) -> List[Dict]:
    """Load all verified candidate JSONs from an archive directory."""
    results = []
    for d in sorted(candidates_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        cand_path = d / "candidate.json"
        if not meta_path.exists() or not cand_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            if not meta.get("verification_ok", False):
                # Check repaired candidates too
                repair_dir = d / "repaired_attempts"
                if repair_dir.exists():
                    for rfile in sorted(repair_dir.glob("attempt_*_candidate.json")):
                        vfile = d / "verifier_result.json"
                        if vfile.exists():
                            vr = json.loads(vfile.read_text())
                            if vr.get("valid", False):
                                cand = json.loads(rfile.read_text())
                                results.append({
                                    "candidate": cand,
                                    "metadata": meta,
                                    "source_dir": str(d.name),
                                    "repaired": True,
                                })
                                break
                continue
            cand = json.loads(cand_path.read_text())
            results.append({
                "candidate": cand,
                "metadata": meta,
                "source_dir": str(d.name),
                "repaired": False,
            })
        except Exception as e:
            print(f"  [WARN] Could not load candidate from {d.name}: {e}")
    return results
