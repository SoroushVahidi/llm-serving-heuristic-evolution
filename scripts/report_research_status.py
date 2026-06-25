"""
Research status report — derives facts from code, not from manual docs.

Usage:
    python scripts/report_research_status.py
    python scripts/report_research_status.py --json
    python scripts/report_research_status.py --help

Exit codes:
    0 — success
    1 — invariant violation detected

This script is intentionally cheap: it imports the policy/selector modules and
introspects them.  No simulation, no API calls, no file I/O beyond the import.
"""
from __future__ import annotations

import argparse
import json
import sys


def gather_status() -> dict:
    """Return a structured status dict derived from live code."""
    import os
    from pathlib import Path
    from llmserveopt.policies.registry import (
        BASELINE_NAMES,
        ORACLE_POLICY_NAMES,
        SELECTOR_CANDIDATE_NAMES,
    )
    from llmserveopt.selector.candidates import SELECTOR_CANDIDATES
    from llmserveopt.selector.models import (
        DecisionTreeSelector,
        RandomForestSelector,
        RuleBasedSelector,
    )

    # --- deployable baselines ---
    n_deployable = len(BASELINE_NAMES)
    n_oracle = len(ORACLE_POLICY_NAMES)
    n_selector_candidates = len(SELECTOR_CANDIDATES)

    # --- invariant checks ---
    oracle_leak = [n for n in ORACLE_POLICY_NAMES if n in SELECTOR_CANDIDATES]
    candidates_match_baselines = sorted(SELECTOR_CANDIDATES) == sorted(BASELINE_NAMES)

    # --- admission_control registered? ---
    admission_control_registered = "admission_control" in BASELINE_NAMES

    # --- rule_based still FIFO-only? ---
    # A genuine feature-based selector will return different policies for
    # different inputs; a placeholder always returns the same thing.
    rb = RuleBasedSelector()
    tight_slo_pred = rb.predict_one({"fraction_tight_slo": 0.9, "min_slack": 0.1})
    normal_pred    = rb.predict_one({"fraction_tight_slo": 0.0, "mean_pred_output_tokens": 20.0,
                                     "pred_output_cv": 0.2})
    rule_based_is_fifo_placeholder = (tight_slo_pred == "fifo" and normal_pred == "fifo")

    # --- selector models ---
    selector_models = ["rule_based"]
    try:
        import sklearn  # noqa: F401
        selector_models += ["decision_tree", "random_forest"]
    except ImportError:
        pass

    # --- template files ---
    _repo_root = Path(__file__).parent.parent
    _templates = {
        "experiment_registry_template": "docs/templates/experiment_registry_template.csv",
        "failure_case_registry_template": "docs/templates/failure_case_registry_template.csv",
        "api_usage_ledger_template": "docs/templates/api_usage_ledger_template.csv",
    }
    templates_present = {k: (_repo_root / v).exists() for k, v in _templates.items()}
    all_templates_present = all(templates_present.values())

    # --- admission_control unit fix ---
    import inspect as _inspect
    from llmserveopt.policies.admission_control import AdmissionControlPolicy as _AC
    _ac_init_src = _inspect.getsource(_AC.__init__)
    _ac_laxity_src = _inspect.getsource(_AC._laxity)
    # Unit fix: __init__ accepts step_size; _laxity delegates to _est_seconds
    admission_control_unit_fixed = (
        "step_size" in _ac_init_src and "_est_seconds" in _ac_laxity_src
    )

    # --- multi_bin_batching unit tests ---
    _mb_test = _repo_root / "tests" / "test_multi_bin_batching_policy.py"
    multi_bin_tests_exist = _mb_test.exists()

    # --- failure case registry populated ---
    _fc_path = _repo_root / "results" / "failure_cases" / "failure_case_registry.csv"
    if _fc_path.exists():
        try:
            import csv as _csv
            with open(_fc_path) as _f:
                _rows = [r for r in _csv.DictReader(_f) if r.get("failure_id", "").strip()]
            num_failure_cases = len(_rows)
        except Exception:
            num_failure_cases = -1
    else:
        num_failure_cases = 0

    # --- API ledger empty? ---
    import csv as _csv
    _al_path = _repo_root / "results" / "api_usage" / "api_usage_ledger.csv"
    if _al_path.exists():
        try:
            with open(_al_path) as _f:
                _al_rows = [r for r in _csv.DictReader(_f) if r.get("call_id", "").strip()]
            api_ledger_entries = len(_al_rows)
        except Exception:
            api_ledger_entries = -1
    else:
        api_ledger_entries = 0

    return {
        "deployable_baselines": {
            "count": n_deployable,
            "names": BASELINE_NAMES,
        },
        "oracle_policies": {
            "count": n_oracle,
            "names": ORACLE_POLICY_NAMES,
        },
        "selector_candidates": {
            "count": n_selector_candidates,
            "names": SELECTOR_CANDIDATES,
            "equals_deployable_baselines": candidates_match_baselines,
        },
        "selector_models_available": selector_models,
        "admission_control_registered": admission_control_registered,
        "admission_control_unit_fixed": admission_control_unit_fixed,
        "multi_bin_tests_exist": multi_bin_tests_exist,
        "rule_based_is_fifo_placeholder": rule_based_is_fifo_placeholder,
        "failure_cases": {
            "count": num_failure_cases,
            "registry_path": str(_fc_path.relative_to(_repo_root)) if _fc_path.exists() else None,
        },
        "api_ledger": {
            "entries": api_ledger_entries,
            "is_empty": api_ledger_entries == 0,
        },
        "invariants": {
            "oracle_not_in_selector_candidates": len(oracle_leak) == 0,
            "oracle_leak_names": oracle_leak,
        },
        "templates": {
            "present": templates_present,
            "all_present": all_templates_present,
        },
    }


def print_text_report(status: dict) -> None:
    db = status["deployable_baselines"]
    oc = status["oracle_policies"]
    sc = status["selector_candidates"]

    print("=" * 60)
    print("LLM Serving Heuristic Evolution — Research Status")
    print("=" * 60)

    print(f"\nDeployable scheduling policies: {db['count']}")
    for name in db["names"]:
        print(f"  - {name}")

    print(f"\nNon-deployable oracle policies: {oc['count']}")
    for name in oc["names"]:
        print(f"  - {name}  [ORACLE — never in selector candidates]")

    print(f"\nSelector candidates: {sc['count']}")
    print(f"  Candidates == deployable baselines: {sc['equals_deployable_baselines']}")

    print(f"\nSelector models available: {status['selector_models_available']}")

    ac = status["admission_control_registered"]
    print(f"\nadmission_control registered: {'YES' if ac else 'NO (MISSING)'}")

    rb = status["rule_based_is_fifo_placeholder"]
    print(f"rule_based is FIFO-only placeholder: {'YES (STALE)' if rb else 'NO (feature-based)'}")

    inv = status["invariants"]
    print(f"\nInvariants:")
    print(f"  oracle not in selector candidates: {inv['oracle_not_in_selector_candidates']}")
    if inv["oracle_leak_names"]:
        print(f"  !! LEAK: {inv['oracle_leak_names']}")

    ac_fixed = status.get("admission_control_unit_fixed", "unknown")
    print(f"\nadmission_control unit fix applied: {'YES' if ac_fixed else 'NO (STALE)'}")

    mb = status.get("multi_bin_tests_exist", False)
    print(f"multi_bin_batching unit tests: {'EXISTS' if mb else 'MISSING'}")

    fc = status.get("failure_cases", {})
    print(f"\nFailure cases recorded: {fc.get('count', 0)}")
    if fc.get("registry_path"):
        print(f"  Registry: {fc['registry_path']}")

    al = status.get("api_ledger", {})
    print(f"API ledger entries: {al.get('entries', 0)} {'(empty — no paid API calls yet)' if al.get('is_empty') else ''}")

    tmpl = status["templates"]
    print(f"\nTemplate files:")
    for name, present in tmpl["present"].items():
        print(f"  {name}: {'OK' if present else 'MISSING'}")
    print(f"  all templates present: {tmpl['all_present']}")

    print("=" * 60)


def check_invariants(status: dict) -> list[str]:
    """Return list of violated invariants (empty = all OK)."""
    violations = []
    inv = status["invariants"]
    if not inv["oracle_not_in_selector_candidates"]:
        violations.append(f"Oracle policies leaked into selector candidates: {inv['oracle_leak_names']}")
    if not status["selector_candidates"]["equals_deployable_baselines"]:
        violations.append("SELECTOR_CANDIDATES does not equal BASELINE_NAMES")
    tmpl = status.get("templates", {})
    if tmpl and not tmpl.get("all_present", True):
        missing = [k for k, v in tmpl["present"].items() if not v]
        violations.append(f"Template files missing: {missing}")
    if not status.get("admission_control_unit_fixed", True):
        violations.append("AdmissionControlPolicy unit mismatch not fixed (laxity mixes seconds and steps)")
    if not status.get("multi_bin_tests_exist", True):
        violations.append("multi_bin_batching unit tests missing (test_multi_bin_batching_policy.py)")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report research status derived from code (cheap, no simulation)."
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if any invariant is violated")
    if "--help" not in (argv or sys.argv[1:]) and "-h" not in (argv or sys.argv[1:]):
        pass
    args = parser.parse_args(argv)

    status = gather_status()

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print_text_report(status)

    violations = check_invariants(status)
    if args.check and violations:
        for v in violations:
            print(f"VIOLATION: {v}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
