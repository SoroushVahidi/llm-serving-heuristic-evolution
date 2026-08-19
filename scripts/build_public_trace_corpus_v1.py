#!/usr/bin/env python3
"""
Build the Public Trace Corpus v1 — workload-INPUT layer only.

Design doc: docs/design/PUBLIC_TRACE_CORPUS_V1.md

This script does NOT evaluate scheduling policies, compute oracle labels,
or touch Cohere/CloudRift. It ingests already-accepted public sources
(BurstGPT, Azure LLM Inference 2023/2024, AgentPerfBench trace_replay),
normalizes them into the canonical corpus schema
(data/public_trace_corpus_v1/schema.json), and writes:

  data/public_trace_corpus_v1/manifest.json
  data/public_trace_corpus_v1/source_coverage.csv
  data/public_trace_corpus_v1/distribution_stats.json
  data/public_trace_corpus_v1/<source>/records.parquet

Usage:
    python scripts/build_public_trace_corpus_v1.py --output-dir data/public_trace_corpus_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llmserveopt.workloads.public_trace_corpus import (  # noqa: E402
    CANONICAL_FIELDS,
    IngestReport,
    inspect_agentperfbench_trace_replay,
    ingest_azure,
    ingest_burstgpt,
    schema_coverage_row,
    sha256_file,
    write_source_parquet,
)

REPO_ROOT = Path(__file__).parent.parent
AGENTPERFBENCH_URL = (
    "https://huggingface.co/datasets/agent-perf-bench/AgentPerfBench/"
    "resolve/main/trace_replay/summary.parquet"
)
AGENTPERFBENCH_MAX_BYTES = 200 * 1024 * 1024  # 200 MB safety cap


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def try_download(url: str, out_path: Path, max_bytes: int) -> Optional[str]:
    """Best-effort small download. Returns an error string on failure, else None."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "public-trace-corpus-v1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            length = resp.headers.get("Content-Length")
            if length is not None and int(length) > max_bytes:
                return f"SKIPPED: remote size {length} exceeds cap {max_bytes}"
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                return f"SKIPPED: downloaded body exceeded cap {max_bytes}"
            out_path.write_bytes(data)
        return None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return f"FAILED: {exc}"


def check_azure_2024_availability() -> Dict[str, Any]:
    """Reference-only: Azure 2024 is already known (scripts/data/download_azure_llm_2024.py)
    and is explicitly documented there as "large files — user-initiated only"
    (code ~692MB, conv ~1.1GB). Per that existing repo convention, this build
    does not auto-download it; it records the known download script/sizes so
    a human can run it deliberately.
    """
    downloader = REPO_ROOT / "scripts" / "data" / "download_azure_llm_2024.py"
    return {
        "status": "REFERENCE_ONLY_USER_INITIATED",
        "downloader_script": str(downloader.relative_to(REPO_ROOT)) if downloader.exists() else None,
        "known_sizes_bytes": {"code": 691_989_454, "conv": 1_135_195_393},
        "reason": "repo convention: large (>500MB) trace downloads are user-initiated, not part of an unattended build",
    }


def percentiles(values: List[float], pcts: List[float]) -> Dict[str, float]:
    if not values:
        return {f"p{int(p * 100)}": None for p in pcts}
    s = sorted(values)
    n = len(s)
    out = {}
    for p in pcts:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        out[f"p{int(p * 100)}"] = s[idx]
    return out


def describe(values: List[float]) -> Dict[str, Any]:
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0}
    s = sorted(values)
    n = len(s)
    d = {
        "count": n,
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
    }
    d.update(percentiles(values, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Public Trace Corpus v1")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "public_trace_corpus_v1"))
    parser.add_argument("--burstgpt-csv", default=str(REPO_ROOT / "data" / "raw" / "burstgpt" / "BurstGPT_1.csv"))
    parser.add_argument("--azure-conv-csv", default=str(REPO_ROOT / "data" / "raw" / "azure" / "AzureLLMInferenceTrace_conv_2023.csv"))
    parser.add_argument("--azure-code-csv", default=str(REPO_ROOT / "data" / "raw" / "azure" / "AzureLLMInferenceTrace_code_2023.csv"))
    parser.add_argument("--skip-agentperfbench-download", action="store_true")
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    sources: Dict[str, Dict[str, Any]] = {}
    coverage_rows: List[Dict[str, str]] = []

    # --- BurstGPT ---
    bg_path = Path(args.burstgpt_csv)
    if bg_path.exists():
        log(f"Ingesting BurstGPT from {bg_path} ...")
        records, report = ingest_burstgpt(bg_path, max_rows=args.max_rows_per_source)
        write_source_parquet(records, out_dir / "burstgpt" / "records.parquet")
        sources["burstgpt"] = {"report": report.to_dict(), "n_records": len(records)}
        coverage_rows.append(schema_coverage_row("burstgpt", records))
        log(f"BurstGPT: {report.rows_retained}/{report.rows_read} rows retained "
            f"({report.rows_dropped_malformed} dropped)")
    else:
        log(f"BurstGPT CSV not found at {bg_path}; skipping")
        sources["burstgpt"] = {"status": "NOT_LOCAL"}

    # --- Azure 2023 (conv + code) ---
    for name, path_str in (("azure_2023_conv", args.azure_conv_csv), ("azure_2023_code", args.azure_code_csv)):
        path = Path(path_str)
        if not path.exists():
            log(f"{name} CSV not found at {path}; skipping")
            sources[name] = {"status": "NOT_LOCAL"}
            continue
        log(f"Ingesting {name} from {path} ...")
        records, report = ingest_azure(path, source_dataset=name, max_rows=args.max_rows_per_source)
        write_source_parquet(records, out_dir / name / "records.parquet")
        sources[name] = {"report": report.to_dict(), "n_records": len(records)}
        coverage_rows.append(schema_coverage_row(name, records))
        log(f"{name}: {report.rows_retained}/{report.rows_read} rows retained "
            f"({report.rows_dropped_malformed} dropped)")

    # --- Azure 2024 (metadata-only availability check; no bulk download here) ---
    log("Checking Azure 2024 availability (metadata-only, no bulk download)...")
    azure_2024_check = check_azure_2024_availability()
    sources["azure_2024"] = {"status": "REFERENCE_ONLY_THIS_PASS", "availability_check": azure_2024_check}
    log(f"Azure 2024 availability: {azure_2024_check.get('status')}")

    # --- AgentPerfBench trace_replay: REAL_SYSTEM_VALIDATION_SOURCE, not a
    # workload-input trace (verified this pass: every config in the dataset
    # is run-level aggregate performance metrics, no per-request token
    # fields). Downloaded and hashed for the record, classified into
    # external_validation_metadata/, and deliberately NOT added to the
    # per-request workload-input corpus or source_coverage.csv.
    apb_raw = out_dir.parent / "raw" / "agentperfbench" / "trace_replay_summary.parquet"
    if not args.skip_agentperfbench_download:
        log(f"Downloading AgentPerfBench trace_replay summary (cap {AGENTPERFBENCH_MAX_BYTES} bytes)...")
        err = try_download(AGENTPERFBENCH_URL, apb_raw, AGENTPERFBENCH_MAX_BYTES)
        if err:
            log(f"AgentPerfBench download issue: {err}")
            sources["agentperfbench_trace_replay"] = {"status": err}
        else:
            log(f"AgentPerfBench downloaded to {apb_raw} ({apb_raw.stat().st_size} bytes)")
            try:
                meta = inspect_agentperfbench_trace_replay(apb_raw)
                ev_dir = out_dir / "external_validation_metadata" / "agentperfbench_trace_replay"
                ev_dir.mkdir(parents=True, exist_ok=True)
                (ev_dir / "metadata.json").write_text(json.dumps(meta.to_dict(), indent=2))
                sources["agentperfbench_trace_replay"] = {
                    "role": "REAL_SYSTEM_VALIDATION_SOURCE",
                    "ingested_into_workload_corpus": False,
                    "external_validation_metadata": meta.to_dict(),
                }
                log(f"AgentPerfBench: classified REAL_SYSTEM_VALIDATION_SOURCE, "
                    f"{meta.n_rows} run-level rows, {len(meta.columns)} columns; "
                    f"not ingested into workload-input corpus")
            except Exception as exc:
                log(f"AgentPerfBench inspection failed: {exc}")
                sources["agentperfbench_trace_replay"] = {"status": f"INSPECTION_FAILED: {exc}"}
    else:
        sources["agentperfbench_trace_replay"] = {"status": "SKIPPED_BY_FLAG"}

    # --- source_coverage.csv ---
    if coverage_rows:
        cov_path = out_dir / "source_coverage.csv"
        fieldnames = ["source_dataset", "n_records"] + list(CANONICAL_FIELDS)
        with open(cov_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in coverage_rows:
                writer.writerow(row)
        log(f"Wrote {cov_path}")

    # --- distribution_stats.json ---
    import pandas as pd

    dist_stats: Dict[str, Any] = {}
    for name in list(sources.keys()):
        parquet_path = out_dir / name / "records.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        dist_stats[name] = {
            "n_requests": len(df),
            "duration_seconds": float(df["relative_arrival_time"].max()) if len(df) else 0.0,
            "prompt_tokens": describe(df["prompt_tokens"].dropna().astype(float).tolist()),
            "output_tokens": describe(df["output_tokens"].dropna().astype(float).tolist()),
            "total_tokens": describe(df["total_tokens"].dropna().astype(float).tolist()),
            "interarrival_time": describe(df["interarrival_time"].dropna().astype(float).tolist()),
            "model_distribution": df["model_name"].value_counts(dropna=True).to_dict() if "model_name" in df else {},
        }
    (out_dir / "distribution_stats.json").write_text(json.dumps(dist_stats, indent=2, default=str))
    log(f"Wrote {out_dir / 'distribution_stats.json'}")

    # --- manifest.json ---
    manifest = {
        "schema_version": "public_trace_corpus_v1.0.0",
        "design_doc": "docs/design/PUBLIC_TRACE_CORPUS_V1.md",
        "git_head_sha": git_head_sha(),
        "build_start_utc": start_iso,
        "build_end_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_duration_seconds": round(time.time() - start, 3),
        "sources": sources,
        "no_policy_evaluation": True,
        "no_paid_api_used": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"Wrote {out_dir / 'manifest.json'}")
    log("Build complete.")


if __name__ == "__main__":
    main()
