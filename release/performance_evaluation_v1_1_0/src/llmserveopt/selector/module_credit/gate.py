"""Reusable module-credit synthesis gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ModuleGateConfig:
    min_conservative_C_base: float = 0.0
    min_predicted_C_parent: float = -0.01
    max_uncertainty: float = 0.25
    prefer_positive_C_env: bool = False
    lambda_m: float = 0.5


class ModuleCandidateGate:
    """Decision function for offline synthesis proposal filtering."""

    def __init__(self, config: ModuleGateConfig | None = None) -> None:
        self.config = config or ModuleGateConfig()

    def score_rows(self, model, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        mu = model.predict_mean(rows)
        u = model.predict_uncertainty(rows)
        scored = []
        for row, mean, unc in zip(rows, mu, u):
            conservative = float(mean) - self.config.lambda_m * float(unc)
            passed, reasons = self._passes(row, float(mean), float(unc), conservative)
            scored.append({
                "row": row,
                "mu_C": float(mean),
                "u_C": float(unc),
                "S_C": conservative,
                "passes": passed,
                "reasons": reasons,
            })
        return scored

    def _passes(self, row: Mapping[str, Any], mu: float, u: float, conservative: float) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        compat = row.get("compatibility_metadata", {})
        if isinstance(compat, Mapping) and float(compat.get("compatible", 1.0)) <= 0.0:
            reasons.append("incompatible")
        if conservative <= self.config.min_conservative_C_base:
            reasons.append("low_conservative_C_base")
        predicted_parent = float(row.get("predicted_C_parent", mu))
        if predicted_parent <= self.config.min_predicted_C_parent:
            reasons.append("low_predicted_C_parent")
        if u > self.config.max_uncertainty:
            reasons.append("high_uncertainty")
        if self.config.prefer_positive_C_env and float(row.get("predicted_C_env", 0.0)) <= 0.0:
            reasons.append("nonpositive_predicted_C_env")
        if bool(row.get("verifier_violation", False)):
            reasons.append("verifier_violation")
        return not reasons, reasons


def gate_candidates(model, rows: Sequence[Mapping[str, Any]], config: ModuleGateConfig | None = None) -> list[dict[str, Any]]:
    return ModuleCandidateGate(config).score_rows(model, rows)
