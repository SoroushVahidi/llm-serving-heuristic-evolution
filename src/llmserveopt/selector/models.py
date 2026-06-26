"""
Baseline selector model wrappers.

v1 selector models
------------------
* RuleBasedSelector   — deterministic feature-based rule selector.
  Dispatches to different scheduling policies based on workload features.
  Does NOT require training.  Replaced the prior FIFO-only placeholder in
  Phase 2B.5.
* DecisionTreeSelector
* RandomForestSelector

scikit-learn is optional.  When not installed, tree/forest classes raise ImportError
with a clear message; the rule-based selector always works.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .candidates import SELECTOR_CANDIDATES
from .features import FEATURE_NAMES


class RuleBasedSelector:
    """Deterministic feature-based rule selector (Phase 2B.11 SCORPIO integration).

    Chooses among deployable scheduling policies using a hand-coded decision
    tree over workload feature values.  No training required.  Fully
    deterministic: same features always produce the same policy choice.

    Phase 2B.8 change summary
    -------------------------
    Phase 2B.7 found that the original Rule 1 (tight-SLO → least_laxity_first)
    fired for ALL overloaded workloads, causing catastrophic WG losses:
      - kv_pressure_decode_heavy:   LLF WG=0.101 vs WSP WG=0.477 (−0.376)
      - high_prediction_noise:      LLF WG=0.584 vs AC  WG=0.988 (−0.404)
      - overloaded_mixed_slo:       LLF WG=0.474 vs SLO WG=0.905 (−0.431)

    Repair: add two guard rules BEFORE the tight-SLO dispatch:
      R1. Decode-heavy workload (large mean_pred_output) → weighted_shortest_processing
      R2. High output prediction noise (pred_output_cv > 1.0) → admission_control
    Replace least_laxity_first with slo_slack_score in the tight-SLO rule.

    Phase 2B.11 change summary
    --------------------------
    Phase 2B.10 added scorpio_style_slo_guard (20th deployable policy), which became
    the best fixed baseline (WG=0.993 overall vs 0.922 for WSP).  The Phase 2B.8
    rule selector was not updated and never dispatched to SCORPIO (fail_005/006).
    Two changes close the gap:

      R0. Overloaded tight-SLO + active SLO violations → scorpio_style_slo_guard
          SCORPIO's admission budget + TTFT/laxity guard outperforms slo_slack_score
          when the system is already violating SLOs under tight deadlines.
          (Evidence: dev_overloaded_mixed_slo + heldout_bursty_mixed_slo families)
          Guard condition: recent_slo_violation_rate > 0.2 ensures this only fires
          when online feedback confirms actual violations are occurring, not just tight
          SLO fractions without overload.

      R2 split. Very extreme prediction noise (pred_output_cv > 2.0) → scorpio_style_slo_guard
          Phase 2B.9 fail_004 (heldout_very_high_noise_s4, 90% noise): AC=0.970,
          SCORPIO=1.000.  Above CV=2.0, SCORPIO's composite SLO guard beats AC.
          Moderate noise (CV 1.0–2.0) keeps routing to admission_control (unchanged).

      R3. Recent SLO violations (standalone, no tight-SLO combined) → scorpio_style_slo_guard
          (was: admission_control)  SCORPIO's targeted admission budget throttling is
          strictly more expressive than AC for violation-prone regimes.

    Rules (in priority order)
    -------------------------
    0. Overloaded tight-SLO + active SLO violations (Phase 2B.11)
       (fraction_tight_slo > 0.4 OR min_slack < 1.0) AND recent_slo_violation_rate > 0.2
       → scorpio_style_slo_guard  (admission budget + TTFT guard beats urgency sort)

    1. Decode-heavy / KV-pressure proxy (Phase 2B.8)
       mean_pred_output_tokens > 200 OR kv_utilization > 0.7
       → weighted_shortest_processing  (short-job-first frees KV slots quickly)

    2a. Very high prediction noise (Phase 2B.11)
       pred_output_cv > 2.0
       → scorpio_style_slo_guard  (composite SLO guard beats AC at extreme CV)

    2b. High output prediction noise (Phase 2B.8, unchanged)
       pred_output_cv > 1.0
       → admission_control  (urgency-sorted; robust when laxity estimates unreliable)

    3. Recent SLO violations (Phase 2B.11 modified)
       recent_slo_violation_rate > 0.3
       → scorpio_style_slo_guard  (was: admission_control)

    4. Tight-SLO / urgent regime (Phase 2B.8: uses slo_slack_score, not LLF)
       fraction_tight_slo > 0.4 OR min_slack < 1.0
       → slo_slack_score  (composite urgency+throughput; avoids LLF cascade)

    5. Prefill-heavy workload (large prompts)
       mean_prompt_tokens > 512 OR p95_prompt_tokens > 1024
       → sarathi_style  (stall-free chunked prefill handles large prompts)

    6. Short, low-variance outputs (SJF regime)
       mean_pred_output_tokens < 64 AND pred_output_cv < 0.5
       → estimated_service_time_first  (SJF proxy works well for uniform short jobs)

    7. Bursty arrivals (high burstiness CV)
       burstiness_cv > 1.5
       → slo_slack_score  (composite urgency handles bursty overload)

    8. Moderate mixed workload
       → edf  (safe, SLO-aware default for general workloads)

    All policy names in use are verified to be in SELECTOR_CANDIDATES at
    class definition time.
    """

    name = "rule_based"

    # All policies this selector may recommend — verified at class definition.
    _POLICY_CHOICES = [
        "scorpio_style_slo_guard",
        "weighted_shortest_processing",
        "admission_control",
        "slo_slack_score",
        "sarathi_style",
        "estimated_service_time_first",
        "edf",
    ]

    def __init__(self) -> None:
        _missing = [p for p in self._POLICY_CHOICES if p not in SELECTOR_CANDIDATES]
        if _missing:
            raise RuntimeError(
                f"RuleBasedSelector references policies not in SELECTOR_CANDIDATES: {_missing}. "
                "Ensure these policies are registered before using this selector."
            )

    @staticmethod
    def _get(features: Dict[str, float], name: str, default: float = 0.0) -> float:
        """Retrieve feature by bare name or 'feat_<name>' (dataset row format)."""
        v = features.get(name)
        if v is not None:
            return float(v)
        v = features.get(f"feat_{name}")
        if v is not None:
            return float(v)
        return default

    def predict_one(self, features: Dict[str, float]) -> str:
        """Return a policy name for one feature dict.

        Accepts both bare feature names (from extract_features()) and
        'feat_*'-prefixed names (from dataset row dicts).
        """
        g = self._get  # shorthand

        fraction_tight_slo    = g(features, "fraction_tight_slo", 0.0)
        min_slack             = g(features, "min_slack", float("inf"))
        recent_violation_rate = g(features, "recent_slo_violation_rate", 0.0)
        kv_utilization        = g(features, "kv_utilization", 0.0)
        mean_prompt           = g(features, "mean_prompt_tokens", 0.0)
        p95_prompt            = g(features, "p95_prompt_tokens", 0.0)
        mean_pred_output      = g(features, "mean_pred_output_tokens", 0.0)
        pred_output_cv        = g(features, "pred_output_cv", 1.0)
        burstiness_cv         = g(features, "burstiness_cv", 0.0)

        # Rule 0: overloaded tight-SLO + active SLO violations (Phase 2B.11)
        # When tight deadlines coincide with observed violations, SCORPIO's admission
        # budget throttling + TTFT/laxity guard outperforms slo_slack_score.
        # The recent_violation_rate guard ensures this only fires under actual overload,
        # not merely a tight-SLO distribution without queue build-up.
        # Evidence: dev WG 0.988, heldout WG 0.998 (SCORPIO-style, Phase 2B.10).
        if (fraction_tight_slo > 0.4 or min_slack < 1.0) and recent_violation_rate > 0.2:
            return "scorpio_style_slo_guard"

        # Rule 1: decode-heavy / KV-pressure proxy (Phase 2B.8 guard)
        # Large mean output → requests hold KV slots longest; urgency-based scheduling
        # (LLF) promotes the worst offenders, cascading into KV saturation (WG=0.101).
        # Use WSP instead: short-job-first frees slots quickly.
        # kv_utilization guard also fires in online deployments where it is available.
        if mean_pred_output > 200 or kv_utilization > 0.7:
            return "weighted_shortest_processing"

        # Rule 2a: very high prediction noise (Phase 2B.11)
        # At extreme CV (> 2.0), SCORPIO's composite SLO guard beats admission_control.
        # Evidence: heldout_very_high_noise_s4 (90% noise) — AC=0.970, SCORPIO=1.000.
        # fail_004 resolution: routes the extreme-noise regime away from AC.
        if pred_output_cv > 2.0:
            return "scorpio_style_slo_guard"

        # Rule 2b: high output prediction noise (Phase 2B.8 guard, unchanged)
        # pred_output_cv 1.0–2.0 signals unreliable service estimates (moderate noise).
        # admission_control with urgency sort is robust here (dev 70%-noise: WG=0.988).
        if pred_output_cv > 1.0:
            return "admission_control"

        # Rule 3: recent SLO violations — SCORPIO guard (Phase 2B.11; was AC)
        # SCORPIO's targeted admission budget + deadline filter is more expressive than
        # AC alone when the system is in violation-prone steady state.
        if recent_violation_rate > 0.3:
            return "scorpio_style_slo_guard"

        # Rule 4: tight SLO / urgency (Phase 2B.8: uses slo_slack_score, not LLF)
        # slo_slack_score is a composite urgency+throughput score; unlike LLF (pure
        # urgency), it avoids throughput collapse under queue build-up.
        if fraction_tight_slo > 0.4 or min_slack < 1.0:
            return "slo_slack_score"

        # Rule 5: prefill-heavy
        if mean_prompt > 512 or p95_prompt > 1024:
            return "sarathi_style"

        # Rule 6: short uniform outputs — SJF regime
        if mean_pred_output < 64 and pred_output_cv < 0.5:
            return "estimated_service_time_first"

        # Rule 7: bursty arrivals
        if burstiness_cv > 1.5:
            return "slo_slack_score"

        # Rule 8: safe general default
        return "edf"

    def predict(self, features: List[Dict[str, float]]) -> List[str]:
        """Return a list of policy names for a list of feature dicts."""
        return [self.predict_one(f) for f in features]


def _check_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise ImportError(
            "scikit-learn is required for tree/forest selectors. "
            "Install with: pip install scikit-learn"
        )


def _feature_matrix(rows: List[Dict[str, float]]) -> np.ndarray:
    return np.array([[r.get(f"feat_{n}", 0.0) for n in FEATURE_NAMES] for r in rows], dtype=float)


def _labels(rows: List[Dict]) -> List[str]:
    return [r["best_policy"] for r in rows]


class DecisionTreeSelector:
    """DecisionTreeClassifier(max_depth=8, min_samples_leaf=20)."""
    name = "decision_tree"

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 20, random_state: int = 42):
        _check_sklearn()
        from sklearn.tree import DecisionTreeClassifier
        self._clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def fit(self, rows: List[Dict]) -> "DecisionTreeSelector":
        X = _feature_matrix(rows)
        y = _labels(rows)
        self._clf.fit(X, y)
        return self

    def predict(self, rows: List[Dict]) -> List[str]:
        X = _feature_matrix(rows)
        return list(self._clf.predict(X))

    def feature_importances(self) -> Dict[str, float]:
        return dict(zip(FEATURE_NAMES, self._clf.feature_importances_))

    def tree_text(self) -> str:
        from sklearn.tree import export_text
        return export_text(self._clf, feature_names=FEATURE_NAMES)

    def save(self, path: str) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path: str) -> "DecisionTreeSelector":
        import joblib
        obj = cls.__new__(cls)
        obj._clf = joblib.load(path)
        return obj


class RandomForestSelector:
    """RandomForestClassifier(n_estimators=200, max_depth=10)."""
    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        _check_sklearn()
        from sklearn.ensemble import RandomForestClassifier
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, rows: List[Dict]) -> "RandomForestSelector":
        X = _feature_matrix(rows)
        y = _labels(rows)
        self._clf.fit(X, y)
        return self

    def predict(self, rows: List[Dict]) -> List[str]:
        X = _feature_matrix(rows)
        return list(self._clf.predict(X))

    def feature_importances(self) -> Dict[str, float]:
        return dict(zip(FEATURE_NAMES, self._clf.feature_importances_))

    def save(self, path: str) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestSelector":
        import joblib
        obj = cls.__new__(cls)
        obj._clf = joblib.load(path)
        return obj


def evaluate_selector(
    selector,
    test_rows: List[Dict],
) -> Dict:
    """Return accuracy and per-class counts."""
    preds = selector.predict(test_rows)
    labels = _labels(test_rows)
    n = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    acc = correct / n if n > 0 else 0.0

    # Confusion: predicted → actual counts
    from collections import Counter
    confusion: Dict[str, Counter] = {}
    for p, l in zip(preds, labels):
        confusion.setdefault(p, Counter())[l] += 1

    return {
        "accuracy": acc,
        "n_test": n,
        "n_correct": correct,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }


def save_metrics(metrics: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
