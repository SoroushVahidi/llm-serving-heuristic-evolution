#!/usr/bin/env python3
"""Hierarchical Regime Router v1 -- FORMAL mechanical G1-G9 rescoring of the
already-completed live closed-loop re-evaluation.

Closes a methodology gap identified by a repository-wide audit
(2026-08-19): `scripts/run_hierarchical_regime_router_live_reeval_v1.py`
printed its own `live_re_evaluation_verdict` from a hand-rolled 4-branch
if/else, not by invoking the frozen `evaluate_all_gates`/`compute_verdict`
implementation in `hierarchical_router_gates_v1.py` (the same canonical
gate evaluator `scripts/run_hierarchical_regime_router_v1_test_evaluation.py`
already used correctly for the prior approximate TEST evaluation).

This script is PURE POST-HOC RESCORING:

- reads the already-completed, immutable
  `experiments/hierarchical_regime_router_live_reeval_v1/live_reeval_results.json`
  (read-only; never writes to it)
- reads the frozen `configs/hierarchical_regime_router_v1_gates.json`
- reuses the G9(b) blended-microcase result from the already-completed,
  frozen `experiments/hierarchical_regime_router_v1_test_evaluation/test_evaluation_results.json`
  (read-only) -- this is valid because blended microcases exercise only
  the frozen Stage-1 router against synthetic FIFO-simulated probes; they
  do not depend on the live-vs-approximate Stage-2 evaluation contract,
  and Stage-1's implementation/training data is verified byte-identical
  between the two runs (see module docstring note below)
- calls ONLY the canonical `evaluate_all_gates`/`compute_verdict`
  functions -- no hand-written substitute verdict logic
- does NOT invoke the simulator, does NOT refit Stage-1/Stage-2, does NOT
  read raw TEST workload/scenario files, does NOT alter any persisted
  scientific metric

Byte-identity of the router implementation across both evaluations was
independently confirmed via
`git diff --stat 2923087 <live-reeval-HEAD> -- src/llmserveopt/policy_separation/hierarchical_regime_router_v1.py configs/hierarchical_regime_router_v1_gates.json`
(empty diff) as part of the audit that motivated this script.

Known, explicitly-reported gap: the persisted `live_reeval_results.json`
records only TEST-AGGREGATE `mean_anwg_live` (a single scalar over all 32
TEST scenarios), not a per-scenario or per-regime breakdown. G4 (Stage-2
preservation, per-regime), G7 (multi-regime benefit count), and G9(a)
(Family-C held-out delta) all require per-regime live ANWG and therefore
CANNOT be mechanically computed from this artifact without re-executing
the live harness -- which is out of scope here (this task must not rerun
the 32 scenarios). They are reported as NOT_EVALUABLE with this reason,
not silently assumed passing or manufactured from partial data.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llmserveopt.policy_separation.hierarchical_router_gates_v1 import (  # noqa: E402
    compute_verdict,
    evaluate_all_gates,
    load_gates_config,
)

LIVE_REEVAL_RESULTS = REPO_ROOT / "experiments/hierarchical_regime_router_live_reeval_v1/live_reeval_results.json"
TEST_EVAL_RESULTS = REPO_ROOT / "experiments/hierarchical_regime_router_v1_test_evaluation/test_evaluation_results.json"
GATES_JSON = REPO_ROOT / "configs/hierarchical_regime_router_v1_gates.json"
OUTPUT_DIR = REPO_ROOT / "experiments/hierarchical_regime_router_live_reeval_v1"
OUTPUT_PATH = OUTPUT_DIR / "gate_rescoring_v1.json"

MIN_TEST_GROUPS_FOR_G5_CI = 5  # same judgment-call threshold as the sibling TEST-evaluation script


def _git_head_sha() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True)
    return bool(out.strip())


def _sha256_of_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metrics(live: dict, sibling_test_eval: dict) -> tuple[dict, dict]:
    """Map persisted live-reeval fields onto the canonical gate-evaluator
    metrics contract. Returns (metrics, provenance_notes)."""
    stage1 = live["stage1_metrics"]
    primary = live["primary_metrics"]

    blended = sibling_test_eval.get("blended_microcase_summary", {})
    blended_reused_note = (
        "REUSED from experiments/hierarchical_regime_router_v1_test_evaluation/"
        "test_evaluation_results.json (not recomputed here). Valid because blended "
        "microcases exercise only the frozen Stage-1 router via a synthetic FIFO "
        "probe simulator, independent of the live-vs-approximate Stage-2 evaluation "
        "contract, and the router implementation is byte-identical between the two "
        "evaluation runs (verified via git diff)."
    )

    metrics = {
        # G1: structural/code-review gate, unchanged since the TEST evaluation
        # (hierarchical_regime_router_v1.py is byte-identical between commit
        # 2923087 and this live-reeval's HEAD -- extract_inputs() enforces the
        # 4-column online-observable allowlist).
        "stage1_input_validity_fraction": 1.0,
        "router_macro_f1": stage1["macro_f1_present_classes_only"],
        "catastrophic_misroute_rate": stage1["catastrophic_misroute_rate"],
        # G4: NOT_EVALUABLE -- see module docstring. live_reeval_results.json
        # only persists the TEST-aggregate mean_anwg_live, not a per-regime
        # breakdown, so the per-regime standalone-vs-integrated fraction this
        # gate requires cannot be recovered without re-executing the live
        # harness (out of scope for this rescoring task).
        "stage2_preservation_fraction_by_regime": None,
        # G5's metric is explicitly "hierarchy minus best GLOBAL FIXED"
        # (baseline A), i.e. `delta_fixed`, not `delta_method` (which
        # compares against the old majority-vote approximation, a different
        # comparator not used by any frozen gate).
        "mean_delta_anwg": primary["delta_fixed"],
        "bootstrap_ci_lower": primary["delta_fixed_ci_90"][0],
        "oracle_gap_closure": primary["oracle_gap_closure"],
        # G7: NOT_EVALUABLE for the same reason as G4 (needs per-regime
        # live-vs-fixed ANWG, not persisted).
        "multi_regime_benefit_count": None,
        # G8(a): structural, verified by the frozen allowlist tests; router
        # code unchanged since the TEST evaluation (see G1 note).
        "leakage_instance_count": 0,
        # G8(b): qualitative human review, not auto-computable -- same
        # treatment as the sibling TEST-evaluation script.
        "qualitative_all_clusters_attributable": None,
        # G9(a): NOT_EVALUABLE for the same reason as G4/G7 (Family C's
        # held-out delta requires per-regime live ANWG, not persisted).
        "family_c_held_out_delta_anwg": None,
        # G9(b): reused from the sibling TEST-evaluation artifact (see note).
        "blended_microcase_catastrophic_rate": (
            blended.get("weighted_catastrophic_misroute_rate")
            if not blended.get("sample_too_small", True)
            else None
        ),
    }

    notes = {
        "G4_not_evaluable_reason": (
            "live_reeval_results.json persists only TEST-aggregate mean_anwg_live "
            "(a single scalar over all 32 TEST scenarios), not a per-scenario or "
            "per-regime breakdown. G4 requires per-regime standalone-vs-integrated "
            "regret comparison, which cannot be recovered without re-executing the "
            "live harness. This is a distinct artifact-completeness gap from the "
            "already-known Family-B-has-0-TEST-scenarios limitation (Regime B would "
            "have been NOT_EVALUABLE either way; Regimes A and C are ALSO "
            "NOT_EVALUABLE here specifically because of this persistence gap, not "
            "because of a data-scarcity issue)."
        ),
        "G7_not_evaluable_reason": "Same persistence gap as G4 (needs per-regime live vs. fixed ANWG).",
        "G9a_not_evaluable_reason": "Same persistence gap as G4/G7 (needs per-regime live ANWG restricted to KV_MEMORY_PRESSURE).",
        "G9b_source": blended_reused_note,
        "G5_metric_disambiguation": (
            "Used primary_metrics.delta_fixed (hierarchy vs. best-global-fixed), "
            "matching the gate's own metric definition, NOT primary_metrics."
            "delta_method (hierarchy vs. the old majority-vote approximation, a "
            "different, non-gate comparator that happens to be numerically "
            "identical to delta_fixed in this run -- see the audit for why)."
        ),
    }
    return metrics, notes


def main(output_path: Path | None = None) -> int:
    """output_path overrides OUTPUT_PATH; tests should pass a tmp_path here
    instead of mutating the tracked canonical gate_rescoring_v1.json."""
    output_path = output_path or OUTPUT_PATH
    if not LIVE_REEVAL_RESULTS.exists():
        print(f"FATAL: {LIVE_REEVAL_RESULTS} does not exist.", file=sys.stderr)
        return 2

    live = json.loads(LIVE_REEVAL_RESULTS.read_text())
    sibling_test_eval = json.loads(TEST_EVAL_RESULTS.read_text()) if TEST_EVAL_RESULTS.exists() else {}

    metrics, notes = build_metrics(live, sibling_test_eval)
    config = load_gates_config()
    gates = evaluate_all_gates(metrics, config)

    # test_sample_insufficient_for_g5_ci: live_reeval_results.json does not
    # persist an exact unique-group count for TEST. It reports 32 TEST
    # scenarios (split_counts.test), which upper-bounds the group count and
    # is well above MIN_TEST_GROUPS_FOR_G5_CI=5; the live-reeval script also
    # successfully produced a non-degenerate bootstrap CI
    # (delta_fixed_ci_90 has nonzero width), which would not happen with a
    # near-1-group sample. This flag is therefore set False on inferred,
    # not independently-confirmed, grounds -- and it is moot for the final
    # verdict here regardless, because G5 fails on its mean criterion alone
    # (compute_verdict returns before ever consulting this flag).
    test_sample_insufficient_for_g5_ci = False
    blended_summary = sibling_test_eval.get("blended_microcase_summary", {})
    blended_microcase_sample_too_small = bool(blended_summary.get("sample_too_small", True))

    verdict = compute_verdict(
        gates,
        blended_microcase_sample_too_small=blended_microcase_sample_too_small,
        test_sample_insufficient_for_g5_ci=test_sample_insufficient_for_g5_ci,
    )

    report = {
        "schema_version": "hierarchical_regime_router_live_reeval_v1_gate_rescoring.1.0.0",
        "purpose": (
            "Formal mechanical G1-G9 rescoring of the already-completed live "
            "closed-loop re-evaluation, via the canonical evaluate_all_gates/"
            "compute_verdict implementation. Does not rerun any simulation."
        ),
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rescoring_provenance": {
            "git_head_sha": _git_head_sha(),
            "git_tree_dirty": _git_dirty(),
            "gates_json_sha256": _sha256_of_file(GATES_JSON),
            "live_reeval_results_sha256": _sha256_of_file(LIVE_REEVAL_RESULTS),
            "live_reeval_results_embedded_git_head_sha": live.get("preregistration_integrity", {}).get("git_head_sha"),
            "sibling_test_evaluation_results_sha256": (
                _sha256_of_file(TEST_EVAL_RESULTS) if TEST_EVAL_RESULTS.exists() else None
            ),
        },
        "source_live_re_evaluation_verdict_ad_hoc": live.get("live_re_evaluation_verdict"),
        "gate_metrics_input": metrics,
        "gate_metrics_input_notes": notes,
        "gate_results": {k: v.to_dict() for k, v in gates.items()},
        "blended_microcase_sample_too_small": blended_microcase_sample_too_small,
        "test_sample_insufficient_for_g5_ci": test_sample_insufficient_for_g5_ci,
        "formal_gate_verdict": verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\nWrote {output_path}")
    print(f"\nFORMAL_GATE_VERDICT: {verdict}")
    print(f"source ad hoc verdict was: {live.get('live_re_evaluation_verdict')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
