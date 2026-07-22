"""Feature encoders for module-credit rows."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ...heuristics.dsl_schema import ALLOWED_OPS, ALLOWED_VARS
from ...policies.genome import GenomeModule
from ...policies.structural_synthesis import map_policy_to_genome
from ..suitability.encoders import structural_features

_STATUS = {"EXACT": 2.0, "APPROXIMATE": 1.0, "UNSUPPORTED": 0.0}
_MODULE_TYPES = ("admission_rule", "priority_rule", "prefill_rule", "kv_guard", "fairness_rule")
_SUITABILITY_FEATURES = (
    "donor_predicted_reward",
    "donor_uncertainty",
    "donor_conservative_suitability",
    "base_predicted_reward",
    "base_uncertainty",
    "base_conservative_suitability",
    "predicted_donor_vs_base_advantage",
)


def module_structural_features(module: GenomeModule | None) -> dict[str, float]:
    """Derive numeric features from a single genome module."""
    if module is None:
        return {
            "module_present": 0.0,
            "module_status_exact": 0.0,
            "module_status_approximate": 0.0,
            "module_status_unsupported": 0.0,
            "module_ast_node_count": 0.0,
            "module_ast_max_depth": 0.0,
        }
    feats = {
        "module_present": 1.0,
        "module_status_ordinal": _STATUS.get(module.status, -1.0),
        "module_status_exact": float(module.status == "EXACT"),
        "module_status_approximate": float(module.status == "APPROXIMATE"),
        "module_status_unsupported": float(module.status == "UNSUPPORTED"),
    }
    expr = module.expression
    nodes = list(_walk_ast(expr))
    feats["module_ast_node_count"] = float(len(nodes))
    feats["module_ast_max_depth"] = float(_ast_depth(expr))
    for op in ALLOWED_OPS:
        feats[f"module_op_count_{op}"] = float(sum(1 for n in nodes if n.get("op") == op))
    for var in ALLOWED_VARS:
        feats[f"module_uses_var_{var}"] = float(sum(1 for n in nodes if n.get("var") == var))
    for key, value in module.parameters.items():
        try:
            feats[f"module_param_{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return feats


class ModuleCreditEncoder:
    """Fit-once matrix builder for module credit model variants."""

    ENCODINGS = ("identity", "structural", "contextual", "suitability_augmented")

    def __init__(self, encoding: str, *, all_policies: Sequence[str]):
        if encoding not in self.ENCODINGS:
            raise ValueError(f"Unknown module-credit encoding {encoding!r}")
        self.encoding = encoding
        self.all_policies = sorted(all_policies)
        self.feature_names: list[str] = []
        self._fitted = False

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "ModuleCreditEncoder":
        feature_keys = set()
        for row in rows:
            feature_keys.update(self._features(row).keys())
        self.feature_names = sorted(feature_keys)
        self._fitted = True
        return self

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ModuleCreditEncoder must be fit before transform")
        out = np.zeros((len(rows), len(self.feature_names)), dtype=float)
        idx = {c: i for i, c in enumerate(self.feature_names)}
        for r, row in enumerate(rows):
            for key, value in self._features(row).items():
                col = idx.get(key)
                if col is not None:
                    out[r, col] = float(value)
        return out

    def fit_transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return self.fit(rows).transform(rows)

    def _features(self, row: Mapping[str, Any]) -> dict[str, float]:
        feats: dict[str, float] = {}
        if self.encoding == "identity":
            feats.update(_identity_features(row, self.all_policies))
            return feats

        feats.update(_prefixed("donor_policy", structural_features(map_policy_to_genome(row["donor_policy"]))))
        feats.update(_prefixed("base_policy", structural_features(map_policy_to_genome(row["base_policy"]))))
        feats.update(_prefixed("donor_module", row.get("donor_module_representation", {})))
        feats.update(_prefixed("base_module", row.get("base_module_representation", {})))
        feats.update(_prefixed("compat", row.get("compatibility_metadata", {})))
        for module_type in _MODULE_TYPES:
            feats[f"module_type_{module_type}"] = float(row["module_type"] == module_type)
        # EDF is a grounded diagnostic feature from the completed anomaly audit:
        # it marks an explicit donor whose usefulness is regime-specific.  It is
        # not a negative prior; the model can learn either sign from state.
        feats["donor_is_edf_regime_specific"] = float(row["donor_policy"] == "edf")

        if self.encoding in ("contextual", "suitability_augmented"):
            feats.update(_prefixed("state", row.get("state_features", {})))
        if self.encoding == "suitability_augmented":
            for key in _SUITABILITY_FEATURES:
                feats[f"suitability_{key}"] = float(row.get(key, 0.0))
        return feats


def _identity_features(row: Mapping[str, Any], all_policies: Sequence[str]) -> dict[str, float]:
    feats = {}
    for policy in all_policies:
        feats[f"donor_id_{policy}"] = float(row["donor_policy"] == policy)
        feats[f"base_id_{policy}"] = float(row["base_policy"] == policy)
    for module_type in _MODULE_TYPES:
        feats[f"module_type_{module_type}"] = float(row["module_type"] == module_type)
    feats["donor_is_edf_regime_specific"] = float(row["donor_policy"] == "edf")
    return feats


def _prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in values.items():
        try:
            out[f"{prefix}_{key}"] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _walk_ast(expr: Any):
    if not isinstance(expr, dict):
        return
    yield expr
    op = expr.get("op")
    if op == "weighted_sum":
        for term in expr.get("terms", []):
            if isinstance(term, (list, tuple)) and term:
                yield from _walk_ast(term[0])
    elif op == "if_then_else":
        for key in ("cond", "then", "else"):
            yield from _walk_ast(expr.get(key))
    else:
        for child in expr.get("args", []):
            yield from _walk_ast(child)


def _ast_depth(expr: Any) -> int:
    if not isinstance(expr, dict):
        return 0
    children = []
    op = expr.get("op")
    if op == "weighted_sum":
        children = [t[0] for t in expr.get("terms", []) if isinstance(t, (list, tuple)) and t]
    elif op == "if_then_else":
        children = [expr.get(k) for k in ("cond", "then", "else")]
    else:
        children = list(expr.get("args", []))
    if not children:
        return 1
    return 1 + max(_ast_depth(c) for c in children)
