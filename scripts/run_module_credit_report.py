#!/usr/bin/env python3
"""Run a focused module-credit report from existing intervention artifacts.

No simulator or structural synthesis jobs are launched.  If Wolverine artifacts
are unavailable, pass ``--use-synthetic-fixture`` to exercise the pipeline on a
small deterministic fixture and mark empirical evaluation as pending.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES  # noqa: E402
from llmserveopt.policies.structural_synthesis import map_policy_to_genome  # noqa: E402
from llmserveopt.selector.module_credit import (  # noqa: E402
    ModuleCreditModel,
    ModuleGateConfig,
    build_intervention_dataset,
    evaluate_credit_predictions,
    evaluate_offline_synthesis_decisions,
    evaluate_topk_ranking,
    gate_candidates,
    load_intervention_artifacts,
    synthetic_intervention_fixture,
)
from llmserveopt.selector.suitability.dataset import rows_with_reward  # noqa: E402
from llmserveopt.selector.suitability.models import JointRewardModel  # noqa: E402

WOLVERINE_ROOT = Path("/mmfs1/project/ikoutis/sv96/llmserveopt-data/module_intervention_credit_20260721T224322Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(WOLVERINE_ROOT))
    parser.add_argument("--out-dir", default="results/module_credit_report/latest")
    parser.add_argument("--use-synthetic-fixture", action="store_true")
    parser.add_argument("--lambda-m", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.perf_counter()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_root = Path(args.artifact_root)
    artifact_available = artifact_root.exists()
    if artifact_available and not args.use_synthetic_fixture:
        raw_rows = load_intervention_artifacts(artifact_root)
        artifact_mode = "wolverine_artifacts"
    else:
        raw_rows = synthetic_intervention_fixture()
        artifact_mode = "synthetic_fixture_empirical_pending"

    suitability_model = _fit_small_suitability_prior(args.seed)
    rows = build_intervention_dataset(raw_rows, suitability_model=suitability_model, suitability_lambda=0.5)
    train = [r for r in rows if r["split"] in ("TRAIN", "VALIDATION")]
    test = [r for r in rows if r["split"] not in ("TRAIN", "VALIDATION")]
    if not test:
        test = list(rows)

    model_specs = {
        "identity": "identity",
        "structural": "structural",
        "state_conditioned_structural": "contextual",
        "suitability_augmented": "suitability_augmented",
    }
    models = {
        name: ModuleCreditModel(name=name, encoding=encoding, target="C_base", random_state=args.seed).fit(train)
        for name, encoding in model_specs.items()
    }
    prediction = {name: evaluate_credit_predictions(model, test, target="C_base") for name, model in models.items()}
    ranking = {name: evaluate_topk_ranking(model, test, lambda_m=args.lambda_m) for name, model in models.items()}
    offline = {
        name: evaluate_offline_synthesis_decisions(test, model, lambda_m=args.lambda_m, seed=args.seed)
        for name, model in models.items()
    }

    held_out_donors: dict[str, Any] = {}
    for donor in sorted({r["donor_policy"] for r in rows if map_policy_to_genome(r["donor_policy"]).metadata.get("mapping_status") in {"EXACT", "APPROXIMATE"}}):
        train_d = [r for r in rows if r["donor_policy"] != donor]
        test_d = [r for r in rows if r["donor_policy"] == donor]
        if len(train_d) < 4 or len(test_d) < 2:
            continue
        model = ModuleCreditModel(name=f"held_out_{donor}", encoding="suitability_augmented", target="C_base", random_state=args.seed).fit(train_d)
        held_out_donors[donor] = {
            "prediction": evaluate_credit_predictions(model, test_d, target="C_base"),
            "ranking": evaluate_topk_ranking(model, test_d, lambda_m=args.lambda_m, ks=(1, 3)),
        }

    held_out_modules: dict[str, Any] = {}
    for module_type in sorted({r["module_type"] for r in rows}):
        train_m = [r for r in rows if r["module_type"] != module_type]
        test_m = [r for r in rows if r["module_type"] == module_type]
        if len(train_m) < 4 or len(test_m) < 2:
            continue
        model = ModuleCreditModel(name=f"held_out_module_{module_type}", encoding="suitability_augmented", target="C_base", random_state=args.seed).fit(train_m)
        held_out_modules[module_type] = evaluate_credit_predictions(model, test_m, target="C_base")

    winner = min(prediction, key=lambda name: prediction[name]["mae"])
    gate_rows = gate_candidates(models[winner], test, ModuleGateConfig(lambda_m=args.lambda_m, max_uncertainty=0.5))
    status = _status_from_results(prediction, ranking, winner)
    readiness = "NOT_READY" if artifact_mode.startswith("synthetic") else ("READY_WITH_RESTRICTIONS" if status != "NO_SIGNAL" else "NOT_READY")

    report = {
        "artifact_root": str(artifact_root),
        "artifact_available": artifact_available,
        "artifact_mode": artifact_mode,
        "n_rows": len(rows),
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "module_types": sorted(Counter(r["module_type"] for r in rows)),
        "donor_policies": sorted(Counter(r["donor_policy"] for r in rows)),
        "base_policies": sorted(Counter(r["base_policy"] for r in rows)),
        "credit_targets": ["C_base", "C_parent", "C_env"],
        "prediction": prediction,
        "ranking": ranking,
        "offline_synthesis": offline,
        "held_out_donors": held_out_donors,
        "held_out_modules": held_out_modules,
        "winning_model": winner,
        "gate": {
            "n_candidates": len(gate_rows),
            "n_passed": sum(1 for r in gate_rows if r["passes"]),
        },
        "MODULE_CREDIT_MODEL_STATUS": status,
        "STRUCTURAL_SYNTHESIS_READINESS": readiness,
        "runtime_s": round(time.perf_counter() - t0, 3),
    }
    (out_dir / "module_credit_results.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "artifact_mode": artifact_mode,
        "n_rows": len(rows),
        "winning_model": winner,
        "MODULE_CREDIT_MODEL_STATUS": status,
        "STRUCTURAL_SYNTHESIS_READINESS": readiness,
        "results": str(out_dir / "module_credit_results.json"),
    }, indent=2))
    return 0


def _fit_small_suitability_prior(seed: int):
    fixture_path = ROOT / "results" / "state_policy_suitability_fixture" / "report_run_v2" / "long_format_rows.json"
    if not fixture_path.exists():
        return None
    rows = rows_with_reward(json.loads(fixture_path.read_text()))
    train = [r for r in rows if r["split"] in ("TRAIN", "VALIDATION")]
    if not train:
        return None
    return JointRewardModel(name="module_credit_prior", encoding="hybrid", all_policies=POLICY_LIBRARY_V2_NAMES, random_state=seed, n_estimators=80).fit(train)


def _status_from_results(prediction: dict[str, Any], ranking: dict[str, Any], winner: str) -> str:
    best_mae = prediction[winner]["mae"]
    top1 = ranking[winner]["top_1"]
    if top1.get("beats_both_parents_fraction", 0.0) >= 0.5 and top1.get("positive_transfer_precision", 0.0) >= 0.7:
        return "STRONG_SIGNAL" if best_mae is not None and best_mae <= 0.08 else "NICHE_SIGNAL"
    if prediction.get("suitability_augmented", {}).get("mae", 999.0) < prediction.get("identity", {}).get("mae", -1.0):
        return "WEAK_GENERALIZATION"
    return "NO_SIGNAL"


if __name__ == "__main__":
    raise SystemExit(main())
