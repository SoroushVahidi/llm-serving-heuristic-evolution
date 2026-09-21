"""State-conditioned module structural credit modeling.

This package is intentionally offline/CPU-only.  It ingests already-evaluated
module intervention rows, builds leakage-safe long-format credit targets, fits
small RandomForest baselines, and evaluates donor/module ranking decisions.  It
does not launch structural synthesis or simulator runs.
"""
from .dataset import (
    MODULE_CREDIT_COLUMNS,
    build_intervention_dataset,
    load_intervention_artifacts,
    validate_no_target_leakage,
    validate_split_integrity,
)
from .encoders import ModuleCreditEncoder, module_structural_features
from .evaluation import (
    evaluate_credit_predictions,
    evaluate_offline_synthesis_decisions,
    evaluate_topk_ranking,
)
from .fixtures import synthetic_intervention_fixture
from .gate import ModuleCandidateGate, ModuleGateConfig, gate_candidates
from .models import ModuleCreditModel
from .pairwise import build_pairwise_interaction_rows

__all__ = [
    "MODULE_CREDIT_COLUMNS",
    "ModuleCandidateGate",
    "ModuleCreditEncoder",
    "ModuleCreditModel",
    "ModuleGateConfig",
    "build_intervention_dataset",
    "build_pairwise_interaction_rows",
    "evaluate_credit_predictions",
    "evaluate_offline_synthesis_decisions",
    "evaluate_topk_ranking",
    "gate_candidates",
    "load_intervention_artifacts",
    "module_structural_features",
    "synthetic_intervention_fixture",
    "validate_no_target_leakage",
    "validate_split_integrity",
]
