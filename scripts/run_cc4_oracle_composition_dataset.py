#!/usr/bin/env python3
"""Run the CC4 true simulator-executed oracle composition dataset build."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.experiments.cc4_oracle_composition_dataset import CC4Error, load_config, run_search


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full-run", action="store_true", help="Required for any non-dry-run build.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--resume-dir", default=None, help="Reuse an existing output_dir instead of creating a new timestamped one; already-completed (window, candidate) executions are skipped.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        result = run_search(
            config,
            config_path=args.config,
            dry_run=args.dry_run,
            full_run=args.full_run,
            max_runs=args.max_runs,
            allow_dirty=args.allow_dirty,
            timestamp=args.timestamp,
            resume_dir=args.resume_dir,
        )
    except CC4Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = {
        "verdict": result.verdict,
        "output_dir": str(result.output_dir) if str(result.output_dir) else None,
        "manifest": result.manifest,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
