#!/usr/bin/env python3
"""Reconstruct real windows via the validated dataset pipeline CLI.

Uses scripts/data/run_real_window_dataset_pipeline.py (--dataset, --run-root, --git-sha).
Mooncake requires an explicit license acknowledgment and is omitted by default.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--git-sha", default="")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["burstgpt_v2", "azure_llm_2023", "azure_llm_2024", "bailian_qwen"],
    )
    ap.add_argument("--mooncake-license-ack", default="")
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Run the pipeline; default is dry-run (print commands only)",
    )
    args = ap.parse_args()

    if "mooncake" in args.datasets:
        if args.mooncake_license_ack != "I_ACKNOWLEDGE_REDISTRIBUTION_PROHIBITED_UNTIL_CLARIFIED":
            print(
                "Mooncake requires --mooncake-license-ack="
                "I_ACKNOWLEDGE_REDISTRIBUTION_PROHIBITED_UNTIL_CLARIFIED",
                file=sys.stderr,
            )
            return 2

    git_sha = args.git_sha or subprocess.check_output(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    pipeline = args.repo_root / "scripts/data/run_real_window_dataset_pipeline.py"

    for ds in args.datasets:
        cmd = [
            sys.executable,
            str(pipeline),
            "--dataset",
            ds,
            "--run-root",
            str(args.run_root),
            "--git-sha",
            git_sha,
        ]
        print("CMD:", " ".join(cmd))
        if args.execute:
            rc = subprocess.call(cmd)
            if rc != 0:
                return rc
    if not args.execute:
        print("Dry-run only. Pass --execute after datasets are staged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
