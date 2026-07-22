"""Policy encodings for joint state-policy suitability modeling.

Three encodings, per docs/current/STATE_POLICY_SUITABILITY_SCHEMA.md:

  A. Identity   -- one-hot policy name. Strong baseline, no structural
                   generalization (cannot score an unseen policy name).
  B. Structural -- deterministic features derived from the canonical
                   SchedulerGenomeV1 representation only. Contains no
                   policy-name information.
  C. Hybrid     -- state features + identity + structural, concatenated.

All structural features are grounded directly in the genome: module
presence/status, tie-breaker, AST node count/depth, and operator-vocabulary
counts (bounded by heuristics.dsl_schema.ALLOWED_OPS). No semantic label is
invented that isn't mechanically derivable from the genome object. The raw
policy_hash string is intentionally never used as a numeric feature -- doing
so would let the "structural" encoding smuggle policy identity back in
through the hash bytes, defeating the point of testing whether structure
(not identity) carries predictive signal.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ...heuristics.dsl_schema import ALLOWED_OPS, ALLOWED_TIE_BREAKERS
from ...policies.genome import GenomeModule, SchedulerGenomeV1

_MODULE_SLOTS: Tuple[str, ...] = (
    "admission_rule", "priority_rule", "prefill_rule", "kv_guard", "fairness_rule",
)
_STATUS_ORDINAL = {"EXACT": 2.0, "APPROXIMATE": 1.0, "UNSUPPORTED": 0.0}


def _walk_ast(expr: Any):
    """Yield every node dict in a DSL expression tree (any shape)."""
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
    elif "args" in expr:
        for sub in expr.get("args", []):
            yield from _walk_ast(sub)


def _ast_depth(expr: Any) -> int:
    if not isinstance(expr, dict):
        return 0
    op = expr.get("op")
    children: List[Any] = []
    if op == "weighted_sum":
        children = [t[0] for t in expr.get("terms", []) if isinstance(t, (list, tuple)) and t]
    elif op == "if_then_else":
        children = [expr.get(k) for k in ("cond", "then", "else")]
    elif "args" in expr:
        children = list(expr.get("args", []))
    if not children:
        return 1
    return 1 + max((_ast_depth(c) for c in children), default=0)


def _root_op(expr: Any) -> str:
    if not isinstance(expr, dict):
        return "none"
    if "op" in expr:
        return str(expr["op"])
    if "var" in expr:
        return "var"
    if "const" in expr:
        return "const"
    return "unknown"


def structural_features(genome: SchedulerGenomeV1) -> Dict[str, float]:
    """Deterministic numeric structural feature dict for one genome.

    Every key is stable across calls for the same genome content (no
    randomness, no dependence on dict iteration order beyond Python's
    already-deterministic insertion order for the fixed key set below).
    """
    modules: Dict[str, GenomeModule | None] = {
        "admission_rule": genome.admission_rule,
        "priority_rule": genome.priority_rule,
        "prefill_rule": genome.prefill_rule,
        "kv_guard": genome.kv_guard,
        "fairness_rule": genome.fairness_rule,
    }

    feats: Dict[str, float] = {}
    n_present = 0
    n_exact = n_approx = n_unsupported = 0
    all_expressions: List[Any] = []
    for slot in _MODULE_SLOTS:
        module = modules[slot]
        present = module is not None
        feats[f"struct_has_{slot}"] = 1.0 if present else 0.0
        feats[f"struct_status_{slot}"] = _STATUS_ORDINAL.get(module.status, -1.0) if present else -1.0
        if present:
            n_present += 1
            n_exact += module.status == "EXACT"
            n_approx += module.status == "APPROXIMATE"
            n_unsupported += module.status == "UNSUPPORTED"
            if module.expression is not None:
                all_expressions.append(module.expression)

    feats["struct_num_modules_present"] = float(n_present)
    feats["struct_num_modules_exact"] = float(n_exact)
    feats["struct_num_modules_approximate"] = float(n_approx)
    feats["struct_num_modules_unsupported"] = float(n_unsupported)
    feats["struct_n_regime_conditions"] = float(len(genome.regime_conditions))
    feats["struct_has_regime_conditions"] = 1.0 if genome.regime_conditions else 0.0

    for regime in genome.regime_conditions:
        all_expressions.append(regime.condition)
        all_expressions.append(regime.priority_rule.expression)
        if regime.admission_rule is not None and regime.admission_rule.expression is not None:
            all_expressions.append(regime.admission_rule.expression)

    # Root-op ("what kind of function is this") for the two most
    # scheduling-relevant slots, one-hot over the bounded op vocabulary.
    admission_root = _root_op(genome.admission_rule.expression) if genome.admission_rule else "none"
    priority_root = _root_op(genome.priority_rule.expression) if genome.priority_rule else "none"
    for candidate_op in list(ALLOWED_OPS) + ["var", "const", "none"]:
        feats[f"struct_admission_root_op_{candidate_op}"] = 1.0 if admission_root == candidate_op else 0.0
        feats[f"struct_priority_root_op_{candidate_op}"] = 1.0 if priority_root == candidate_op else 0.0

    # Tie-breaker, one-hot over the bounded vocabulary.
    for tb in ALLOWED_TIE_BREAKERS:
        feats[f"struct_tie_breaker_{tb}"] = 1.0 if genome.tie_breaker == tb else 0.0

    # AST node count / depth / operator-vocabulary counts across every
    # expression the genome actually contains.
    node_count = 0
    max_depth = 0
    op_counts = {o: 0 for o in ALLOWED_OPS}
    for expr in all_expressions:
        for node in _walk_ast(expr):
            node_count += 1
            if "op" in node and node["op"] in op_counts:
                op_counts[node["op"]] += 1
        max_depth = max(max_depth, _ast_depth(expr))
    feats["struct_ast_node_count"] = float(node_count)
    feats["struct_ast_max_depth"] = float(max_depth)
    feats["struct_n_expressions"] = float(len(all_expressions))
    for op_name, count in op_counts.items():
        feats[f"struct_op_count_{op_name}"] = float(count)

    return feats


def identity_features(policy_name: str, all_policies: Sequence[str]) -> Dict[str, float]:
    """One-hot policy-name encoding over a fixed, deterministic vocabulary."""
    return {f"policyid_{name}": (1.0 if name == policy_name else 0.0) for name in sorted(all_policies)}


class PolicyEncoder:
    """Fit-once feature-space builder for one of the three encodings.

    Column layout is frozen at fit() time from the *union* of state-feature
    keys seen plus the full `all_policies` identity vocabulary (not just
    the policies present in the fit rows) -- so a held-out-policy row can
    still be transformed consistently (its identity column, if any, is
    simply always 0 for training rows of other policies).
    """

    ENCODINGS = ("identity", "structural", "hybrid")

    def __init__(self, encoding: str, all_policies: Sequence[str]):
        if encoding not in self.ENCODINGS:
            raise ValueError(f"Unknown encoding {encoding!r}; expected one of {self.ENCODINGS}")
        self.encoding = encoding
        self.all_policies = sorted(all_policies)
        self._state_cols: List[str] = []
        self._policy_cols: List[str] = []
        self._fitted = False

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "PolicyEncoder":
        state_cols = sorted({k for row in rows for k in row["state_features"].keys()})
        self._state_cols = state_cols

        if self.encoding == "identity":
            self._policy_cols = [f"policyid_{name}" for name in self.all_policies]
        elif self.encoding == "structural":
            struct_cols = sorted({k for row in rows for k in row["policy_representation"].keys() if k != "mapping_status_summary"})
            self._policy_cols = struct_cols
        else:  # hybrid
            struct_cols = sorted({k for row in rows for k in row["policy_representation"].keys() if k != "mapping_status_summary"})
            self._policy_cols = [f"policyid_{name}" for name in self.all_policies] + struct_cols

        self._fitted = True
        return self

    @property
    def feature_names(self) -> List[str]:
        if not self._fitted:
            raise RuntimeError("PolicyEncoder must be fit() before feature_names is available")
        if self.encoding == "structural":
            return list(self._policy_cols)
        if self.encoding == "identity":
            return list(self._state_cols) + list(self._policy_cols)
        return list(self._state_cols) + list(self._policy_cols)

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PolicyEncoder must be fit() before transform()")
        out = np.zeros((len(rows), len(self.feature_names)), dtype=float)
        col_index = {c: i for i, c in enumerate(self.feature_names)}
        for r, row in enumerate(rows):
            if self.encoding != "structural":
                for k, v in row["state_features"].items():
                    idx = col_index.get(k)
                    if idx is not None:
                        out[r, idx] = float(v)
            if self.encoding in ("identity", "hybrid"):
                idx = col_index.get(f"policyid_{row['policy_name']}")
                if idx is not None:
                    out[r, idx] = 1.0
            if self.encoding in ("structural", "hybrid"):
                for k, v in row["policy_representation"].items():
                    if k == "mapping_status_summary":
                        continue
                    idx = col_index.get(k)
                    if idx is not None:
                        out[r, idx] = float(v)
        return out

    def fit_transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return self.fit(rows).transform(rows)
