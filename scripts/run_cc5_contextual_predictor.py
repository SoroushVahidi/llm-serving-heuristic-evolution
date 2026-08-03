#!/usr/bin/env python3
"""Run CC5: train and evaluate the deployable contextual composition predictor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.experiments.cc5_contextual_predictor import CC5Error, run_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="CC4 oracle-composition-dataset directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full-run", action="store_true", help="Required for any non-dry-run build.")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--resume-dir", default=None, help="Reuse an existing output_dir; a completed manifest.json short-circuits retraining.")
    parser.add_argument("--ood-z-threshold", type=float, default=2.0)
    parser.add_argument("--uncertainty-threshold-quantile", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps({"dry_run": True, "dataset_dir": str(args.dataset_dir)}, indent=2))
        return 0
    if not args.full_run:
        print("ERROR: non-dry runs require --full-run", file=sys.stderr)
        return 2

    try:
        result = run_training(
            dataset_dir=args.dataset_dir,
            timestamp=args.timestamp,
            resume_dir=args.resume_dir,
            ood_z_threshold=args.ood_z_threshold,
            uncertainty_threshold_quantile=args.uncertainty_threshold_quantile,
            seed=args.seed,
        )
    except CC5Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = {"verdict": result.verdict, "output_dir": str(result.output_dir), "manifest": result.manifest}
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
