#!/usr/bin/env python3
"""Run the CC1 true simulator-executed composition-opportunity experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.experiments.cc1_composition_opportunity import CC1Error, load_config, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full-run", action="store_true", help="Required when config mode is full or cc1b.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        result = run_experiment(
            config,
            config_path=args.config,
            dry_run=args.dry_run,
            full_run=args.full_run,
            max_runs=args.max_runs,
            allow_dirty=args.allow_dirty,
            timestamp=args.timestamp,
        )
    except CC1Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = {
        "verdict": result.verdict,
        "output_dir": str(result.output_dir) if str(result.output_dir) else None,
        "manifest": result.manifest,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
