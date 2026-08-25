#!/usr/bin/env python3
"""Launch-gate smoke check for Family B v2 PrefillControl composition.

Builds the full scenario grid from a config (default: p2_config.yaml, the
preregistered 32-scenario grid), assigns splits, extracts features, and
verifies the canonical primary metric -- without launching any simulation.

This is a pure validation pass: it must not run the simulator and must not
write any experiment results. It exists to catch scenario-generation /
split / feature-schema regressions before a costly full launch.

Exit code 0 = all gates passed. Non-zero = a gate failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from p7_runner import build_scenarios_from_config, extract_scenario_features  # noqa: E402
from p3_chunk_control import (  # noqa: E402
    PRIMARY,
    FORBIDDEN_FEATURE_KEYS,
    assign_family_b_v2_splits,
    assert_no_split_leakage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "p2_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runner_cfg = cfg.get("runner", {})
    allow_synthetic = not runner_cfg.get("require_burstgpt", True)

    scenarios = build_scenarios_from_config(
        cfg, allow_synthetic_tokens=allow_synthetic, datasets_root=None,
    )
    sids = [s.scenario_id for s in scenarios]
    assert len(sids) == len(set(sids)), "duplicate scenario_id in generated grid"

    split = assign_family_b_v2_splits(sids)
    assert_no_split_leakage(split)

    all_assigned = split.train + split.val + split.test + split.ood
    assert sorted(all_assigned) == sorted(sids), "not every scenario assigned exactly once"

    print(
        f"Scenarios: {len(sids)}, Train: {len(split.train)}, Val: {len(split.val)}, "
        f"Test: {len(split.test)}, OOD: {len(split.ood)}"
    )

    if len(split.ood) == 0:
        print("FAIL: OOD split is empty -- launch gate blocked.", file=sys.stderr)
        return 1

    features = extract_scenario_features(scenarios)
    n_feats = len(next(iter(features.values()))) if features else 0
    for feat_dict in features.values():
        for key in FORBIDDEN_FEATURE_KEYS:
            assert key not in feat_dict, f"forbidden key {key!r} leaked into features"
    print(f"Features extracted: {n_feats} (no forbidden leakage)")

    # Canonical primary metric check -- PRIMARY is imported directly above,
    # so this is a direct value comparison, not a `p3.PRIMARY` module-attribute
    # lookup (which would require `p3` to be bound in this scope).
    assert PRIMARY == "arrival_normalized_weighted_goodput"
    print(f"Primary metric: {PRIMARY} (canonical, OK)")

    print("SMOKE OK: all launch gates in this script passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
