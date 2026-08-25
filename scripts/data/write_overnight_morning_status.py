#!/usr/bin/env python3
"""Write overnight morning status from durable job registry + sacct/squeue."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output or str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    registry_path = run_root / "manifests" / "job_ids.tsv"
    rows = []
    if registry_path.exists():
        for line in registry_path.read_text().splitlines():
            if not line.strip() or line.startswith("job_id"):
                continue
            parts = line.split("\t")
            rows.append(parts)

    job_ids = [r[0] for r in rows if r]
    squeue = run(
        ["squeue", "-j", ",".join(job_ids), "-o", "%i|%j|%T|%M|%D|%R"]
    ) if job_ids else ""
    sacct = run(
        [
            "sacct",
            "-j",
            ",".join(job_ids),
            "--format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS",
            "-P",
        ]
    ) if job_ids else ""

    artifacts = {}
    for ds in ["burstgpt_v2", "azure_llm_2023", "azure_llm_2024", "bailian_qwen", "mooncake"]:
        d = run_root / "windows" / ds
        artifacts[ds] = {
            "final_report": (d / "FINAL_DATASET_REPORT.json").exists(),
            "validation_ok_marker": (d / "markers" / "validate.ok").exists(),
            "construct_ok_marker": (d / "markers" / "construct.ok").exists(),
            "window_catalog": (d / "window_catalog.json").exists(),
        }
    pilot = {
        "pilot_results": (run_root / "pilot" / "pilot_results.json").exists(),
        "pilot_report": (run_root / "pilot" / "PILOT_REPORT.md").exists(),
    }
    status = {
        "utc": utc_now(),
        "run_root": str(run_root),
        "job_ids": job_ids,
        "squeue": squeue,
        "sacct": sacct,
        "artifacts": artifacts,
        "pilot": pilot,
        "registry_rows": rows,
    }
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "overnight_status.json").write_text(json.dumps(status, indent=2) + "\n")
    md = [
        "# Overnight status",
        "",
        f"- utc: {status['utc']}",
        f"- run_root: `{run_root}`",
        "",
        "## squeue",
        "```",
        squeue.strip() or "(empty)",
        "```",
        "",
        "## sacct",
        "```",
        sacct.strip() or "(empty)",
        "```",
        "",
        "## Dataset artifacts",
    ]
    for ds, art in artifacts.items():
        md.append(f"- {ds}: {art}")
    md.append("")
    md.append(f"- pilot: {pilot}")
    (reports / "OVERNIGHT_STATUS.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {reports / 'OVERNIGHT_STATUS.md'}")


if __name__ == "__main__":
    main()
