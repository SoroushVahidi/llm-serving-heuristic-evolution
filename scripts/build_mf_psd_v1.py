#!/usr/bin/env python3
"""Build the canonical Multi-Family Policy Separation Dataset (MF-PSD) v1.

DATA UNIFICATION ONLY -- see
docs/audits/multi_family_policy_separation_dataset_v1_20260817.md and
docs/audits/reassessment_composition_hypothesis_20260817.md (revised
roadmap Step 1). Does not train a selector or run any composition/synthesis
experiment.

Usage:
    python scripts/build_mf_psd_v1.py [--output-dir DIR]

Default output directory: experiments/mf_psd_v1/ (a stable, non-timestamped
path -- MF-PSD is a deterministic rebuildable transform of already-frozen
evidence, not a new simulation run, so it is versioned by content/SHA-256 in
its own provenance manifest rather than by build wall-clock time; see the
audit doc for the rationale).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.policy_separation.mf_psd import build_mf_psd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "mf_psd_v1",
        help="Output directory for the MF-PSD v1 artifacts.",
    )
    args = parser.parse_args()

    manifest = build_mf_psd(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
