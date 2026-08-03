"""CC5 deployable contextual composition predictor.

Trains and evaluates causal models that select among CC4's already-verified,
already-simulator-executed candidate pool given a window's causal features.
Because CC4 exhaustively executed every candidate against every window,
CC5's "deployable predictor" reduces to a SELECTION function (causal
features -> which pre-verified candidate to use); evaluating a selection is
a table lookup into CC4's own results, never a new simulator run, and "no
model may execute an unverified composition" holds structurally since CC5
never synthesizes new DSL.

Split discipline mirrors CC4's own development_splits/evaluation_splits
fields verbatim (not re-derived): development windows are used for fitting
and leave-one-window-out cross-validation; evaluation windows are touched
exactly once, for the final reported verdict.
"""
from __future__ import annotations

import json
import hashlib
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from llmserveopt.experiments.cc1_composition_opportunity import ROOT, display_path, git_state, mean

PRIMARY = "arrival_normalized_weighted_goodput"
PRIMARY_COL = f"metric_{PRIMARY}"
COMPLETION_COL = "metric_completion_fraction"

# The only window-level columns CC5 may ever use as model inputs -- computed
# purely from window.requests before any policy is executed (see CC4's
# build_causal_features()). Never includes a metric_*/oracle_* column.
CAUSAL_FEATURE_COLUMNS = (
    "num_requests", "mean_prompt_tokens", "mean_predicted_output_tokens",
    "mean_slo_slack", "arrival_span_s", "arrival_rate_est", "num_slo_classes",
)

# The RANKING-primitive pool CC4's candidate_search used (dsl_schema order,
# fixed so the primitive-weight feature vector has a stable dimension).
PRIMITIVE_POOL = (
    "laxity_urgency", "priority", "queue_age",
    "predicted_output_length", "prompt_length", "estimated_service_time",
)

CANDIDATE_FAMILIES = (
    "fixed_policy", "cc1b_borda_baseline", "weighted_primitive_mixture",
    "sparse_topk_mixture", "admission_gate_variant", "placement_variant",
)


class CC5Error(ValueError):
    """Raised when the CC4 dataset fails validation or CC5 runtime state is invalid."""


# ---------------------------------------------------------------------------
# Dataset loading + validation (audit, enforced programmatically)
# ---------------------------------------------------------------------------


@dataclass
class CC4Dataset:
    dataset_dir: Path
    manifest: dict[str, Any]
    workload_windows: pd.DataFrame
    causal_features: pd.DataFrame
    candidate_compositions: pd.DataFrame
    per_window_results: pd.DataFrame
    oracle_labels: pd.DataFrame
    regret_matrix: pd.DataFrame
    composition_parameters: pd.DataFrame
    near_tie_flags: pd.DataFrame
    completion_constraints: pd.DataFrame
    development_splits: tuple[str, ...]
    evaluation_splits: tuple[str, ...]


def load_cc4_dataset(dataset_dir: str | Path) -> CC4Dataset:
    d = Path(dataset_dir)
    if not d.exists():
        raise CC5Error(f"CC4 dataset directory does not exist: {d}")
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        raise CC5Error(f"CC4 dataset is missing manifest.json: {d}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment") != "cc4_oracle_composition_dataset":
        raise CC5Error(f"manifest.json is not a CC4 oracle-composition-dataset manifest: {manifest.get('experiment')!r}")

    def _read(name: str) -> pd.DataFrame:
        path = d / f"{name}.parquet"
        if not path.exists():
            raise CC5Error(f"CC4 dataset is missing required table: {name}.parquet")
        return pd.read_parquet(path)

    tables = {name: _read(name) for name in (
        "workload_windows", "causal_features", "candidate_compositions", "per_window_results",
        "oracle_labels", "regret_matrix", "composition_parameters", "near_tie_flags", "completion_constraints",
    )}
    resolved_config = d / "resolved_config.yaml"
    dev_splits: tuple[str, ...] = ("TRAIN", "VALIDATION")
    eval_splits: tuple[str, ...] = ("ID_TEST", "OOD_TEST")
    if resolved_config.exists():
        import yaml
        cfg = yaml.safe_load(resolved_config.read_text())
        dev_splits = tuple(cfg.get("development_splits", dev_splits))
        eval_splits = tuple(cfg.get("evaluation_splits", eval_splits))

    return CC4Dataset(
        dataset_dir=d, manifest=manifest, development_splits=dev_splits, evaluation_splits=eval_splits,
        **tables,
    )


def validate_cc4_dataset(ds: CC4Dataset) -> list[str]:
    """Run the dataset audit programmatically; returns a list of findings
    (empty == clean). Raises CC5Error on any finding that would make
    training unsafe (leakage risk, missing data, invalid rows)."""
    findings: list[str] = []

    if ds.per_window_results.empty:
        raise CC5Error("per_window_results is empty -- nothing to train on")
    if not (ds.per_window_results["true_simulator_executed"] == True).all():  # noqa: E712
        raise CC5Error("per_window_results contains rows that were not true_simulator_executed")
    if (ds.per_window_results["reward_vector_interpolated"] == True).any():  # noqa: E712
        raise CC5Error("per_window_results contains reward_vector_interpolated rows -- must never train on these")
    if not (ds.per_window_results["verification_outcome"] == "valid").all():
        raise CC5Error("per_window_results contains rows with verification_outcome != 'valid'")
    if ds.per_window_results[PRIMARY_COL].isna().any() or ds.per_window_results[COMPLETION_COL].isna().any():
        raise CC5Error("per_window_results has null ANWG or completion_fraction values")

    missing_causal = set(CAUSAL_FEATURE_COLUMNS) - set(ds.causal_features.columns)
    if missing_causal:
        raise CC5Error(f"causal_features is missing required columns: {sorted(missing_causal)}")
    if ds.causal_features[list(CAUSAL_FEATURE_COLUMNS)].isna().any().any():
        raise CC5Error("causal_features has null values in a required feature column")

    dev_windows = set(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    eval_windows = set(ds.causal_features[ds.causal_features["split"].isin(ds.evaluation_splits)]["window_id"])
    overlap = dev_windows & eval_windows
    if overlap:
        raise CC5Error(f"development/evaluation split overlap detected for windows: {sorted(overlap)}")
    if not dev_windows:
        raise CC5Error("no development-split windows found")
    if not eval_windows:
        raise CC5Error("no evaluation-split windows found")
    findings.append(f"{len(dev_windows)} development windows, {len(eval_windows)} evaluation windows")

    rejected = ds.candidate_compositions[ds.candidate_compositions["verification_outcome"] != "valid"]
    if not rejected.empty:
        findings.append(f"{len(rejected)} candidate(s) with non-valid verification_outcome (excluded from training)")

    return findings


# ---------------------------------------------------------------------------
# Feature encoding (causal features + candidate's own declared recipe only)
# ---------------------------------------------------------------------------


@dataclass
class FeatureEncoder:
    causal_mean: np.ndarray
    causal_std: np.ndarray
    feature_names: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, causal_df: pd.DataFrame) -> "FeatureEncoder":
        X = causal_df[list(CAUSAL_FEATURE_COLUMNS)].to_numpy(dtype=float)
        mean_ = X.mean(axis=0)
        std_ = X.std(axis=0)
        std_[std_ < 1e-9] = 1.0
        names = (
            list(CAUSAL_FEATURE_COLUMNS)
            + [f"family__{f}" for f in CANDIDATE_FAMILIES]
            + [f"weight__{p}" for p in PRIMITIVE_POOL]
            + ["extra__k", "extra__laxity_threshold", "extra__n_placement_keys"]
        )
        return cls(causal_mean=mean_, causal_std=std_, feature_names=names)

    def causal_vector(self, causal_row: Mapping[str, Any]) -> np.ndarray:
        raw = np.array([float(causal_row[c]) for c in CAUSAL_FEATURE_COLUMNS])
        return (raw - self.causal_mean) / self.causal_std

    def candidate_vector(self, family: str, primitive_weights: Mapping[str, float], extra_params: Mapping[str, Any]) -> np.ndarray:
        family_vec = np.array([1.0 if family == f else 0.0 for f in CANDIDATE_FAMILIES])
        weight_vec = np.array([float(primitive_weights.get(p, 0.0)) for p in PRIMITIVE_POOL])
        k = float(extra_params.get("k", 0.0))
        laxity_threshold = float(extra_params.get("laxity_threshold", 0.0))
        n_placement = float(len(extra_params.get("placement_keys", []) or []))
        return np.concatenate([family_vec, weight_vec, [k, laxity_threshold, n_placement]])

    def row_vector(self, causal_row: Mapping[str, Any], family: str, primitive_weights: Mapping[str, float], extra_params: Mapping[str, Any]) -> np.ndarray:
        return np.concatenate([self.causal_vector(causal_row), self.candidate_vector(family, primitive_weights, extra_params)])


def build_regret_training_table(ds: CC4Dataset, encoder: FeatureEncoder, window_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y=regret, meta) for every (window, candidate) row whose
    window_id is in `window_ids`."""
    causal_by_window = ds.causal_features.set_index("window_id")
    regret = ds.regret_matrix.merge(
        ds.per_window_results[["window_id", "candidate_id", "primitive_weights_json", "extra_params_json"]],
        on=["window_id", "candidate_id"], how="left",
    )
    regret = regret[regret["window_id"].isin(window_ids)].reset_index(drop=True)
    X = np.zeros((len(regret), len(encoder.feature_names)))
    for i, row in regret.iterrows():
        causal_row = causal_by_window.loc[row["window_id"]]
        weights = json.loads(row["primitive_weights_json"]) if row["primitive_weights_json"] else {}
        extras = json.loads(row["extra_params_json"]) if row["extra_params_json"] else {}
        X[i] = encoder.row_vector(causal_row, row["family"], weights, extras)
    y = regret["regret"].to_numpy(dtype=float)
    return X, y, regret[["window_id", "candidate_id", "family"]]


def build_class_training_table(ds: CC4Dataset, encoder: FeatureEncoder, window_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = ds.oracle_labels[ds.oracle_labels["window_id"].isin(window_ids)].reset_index(drop=True)
    causal_by_window = ds.causal_features.set_index("window_id")
    X = np.array([encoder.causal_vector(causal_by_window.loc[wid]) for wid in labels["window_id"]])
    y = labels["oracle_family"].to_numpy()
    return X, y, list(labels["window_id"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_regret_regressor_factories(seed: int = 0) -> dict[str, Callable[[], Any]]:
    """Factories (not fitted instances) -- leave_one_window_out_cv needs a
    fresh, unfitted estimator per fold."""
    return {
        "ridge": lambda: Ridge(alpha=1.0, random_state=seed),
        "decision_tree": lambda: DecisionTreeRegressor(max_depth=4, random_state=seed),
        "random_forest": lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=seed),
        "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=seed),
        "knn": lambda: KNeighborsRegressor(n_neighbors=3),
    }


class NearestRegimeClassifier:
    """1-NN in causal-feature space over dev windows -- the simplest
    non-parametric composition-class baseline."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearestRegimeClassifier":
        self._X, self._y = X, y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = []
        for row in X:
            dists = np.linalg.norm(self._X - row, axis=1)
            out.append(self._y[int(np.argmin(dists))])
        return np.array(out)


# ---------------------------------------------------------------------------
# Non-learned baselines (mirroring CC1's own comparison set)
# ---------------------------------------------------------------------------


@dataclass
class LookupBaseline:
    name: str
    selection: dict[str, str] | str  # per-regime dict, or a single global candidate_id

    def select(self, regime: str) -> str:
        if isinstance(self.selection, str):
            return self.selection
        return self.selection.get(regime, next(iter(self.selection.values())))


def fit_best_fixed_policy(ds: CC4Dataset, window_ids: Sequence[str]) -> LookupBaseline:
    rows = ds.per_window_results[
        (ds.per_window_results["window_id"].isin(window_ids)) & (ds.per_window_results["family"] == "fixed_policy")
    ]
    means = rows.groupby("candidate_id")[PRIMARY_COL].mean()
    return LookupBaseline("best_fixed_policy", means.idxmax())


def fit_best_global_composition(ds: CC4Dataset, window_ids: Sequence[str]) -> LookupBaseline:
    rows = ds.per_window_results[ds.per_window_results["window_id"].isin(window_ids)]
    means = rows.groupby("candidate_id")[PRIMARY_COL].mean()
    return LookupBaseline("best_global_composition", means.idxmax())


def fit_existing_hard_selector(ds: CC4Dataset, window_ids: Sequence[str]) -> LookupBaseline:
    rows = ds.per_window_results[
        (ds.per_window_results["window_id"].isin(window_ids)) & (ds.per_window_results["family"] == "fixed_policy")
    ]
    by_regime: dict[str, str] = {}
    for regime, group in rows.groupby("regime"):
        by_regime[regime] = group.groupby("candidate_id")[PRIMARY_COL].mean().idxmax()
    return LookupBaseline("existing_hard_selector", by_regime)


# ---------------------------------------------------------------------------
# Uncertainty / OOD / abstention
# ---------------------------------------------------------------------------

UNCERTAINTY_SCHEMA_VERSION = 2
SUPPORTED_UNCERTAINTY_METHODS = (
    "normalized_split_conformal",
    "bootstrap_ensemble",
    "random_forest_per_tree_prediction_std",
)


@dataclass
class UncertaintyOODGate:
    dev_causal_mean: np.ndarray
    dev_causal_std: np.ndarray
    ood_z_threshold: float
    uncertainty_threshold: float

    @classmethod
    def fit(cls, encoder: FeatureEncoder, causal_df: pd.DataFrame, window_ids: Sequence[str], *, ood_z_threshold: float, uncertainty_threshold: float) -> "UncertaintyOODGate":
        dev = causal_df[causal_df["window_id"].isin(window_ids)]
        X = np.array([encoder.causal_vector(row) for _, row in dev.iterrows()])
        return cls(X.mean(axis=0), X.std(axis=0) + 1e-9, ood_z_threshold, uncertainty_threshold)

    def ood_score(self, causal_vector: np.ndarray) -> float:
        # Per-dimension z-scores are clipped to a fixed, interpretable
        # ceiling before taking the max: a dev-set feature dimension with
        # near-zero variance (e.g. every dev window happening to share the
        # same num_slo_classes) would otherwise produce an uninterpretable
        # near-infinite score for any eval window with a different value in
        # just that one dimension, drowning out the other dimensions'
        # signal. The clip changes no is_ood() decision at any reasonable
        # threshold (50 std-equivalents is already far past any sane
        # ood_z_threshold) -- it only keeps the *reported* score legible.
        z = np.abs((causal_vector - self.dev_causal_mean) / self.dev_causal_std)
        return float(np.clip(z, 0.0, 50.0).max())

    def is_ood(self, causal_vector: np.ndarray) -> bool:
        return self.ood_score(causal_vector) > self.ood_z_threshold

    def is_uncertain(self, uncertainty_score: float) -> bool:
        return bool(uncertainty_score > self.uncertainty_threshold)


@dataclass
class UncertaintyCalibrator:
    """Model-agnostic uncertainty with validation-only threshold calibration.

    Preferred methods:
    - normalized_split_conformal: residual magnitude model fit on TRAIN,
      nonconformity scores and conformal quantile on VALIDATION only.
    - bootstrap_ensemble: window-bootstrap refits on TRAIN, std of predictions
      as uncertainty, threshold calibrated on VALIDATION only.
    """

    method: str
    schema_version: int
    residual_model: Any | None
    conformal_quantile: float
    uncertainty_threshold: float
    target_coverage: float
    empirical_coverage: float
    calibration_error: float
    seed: int
    train_window_ids: list[str]
    calibration_window_ids: list[str]
    bootstrap_models: list[Any] | None
    n_bootstrap: int
    residual_eps: float
    point_model_train_only: Any | None
    runtime_overhead_s: float
    calibration_manifest: dict[str, Any]

    def score(self, X: np.ndarray, point_preds: np.ndarray | None = None) -> np.ndarray:
        if self.method == "normalized_split_conformal":
            if self.residual_model is None:
                raise CC5Error("normalized_split_conformal calibrator missing residual_model")
            scale = np.maximum(self.residual_model.predict(X), self.residual_eps)
            return self.conformal_quantile * scale
        if self.method == "bootstrap_ensemble":
            if not self.bootstrap_models:
                raise CC5Error("bootstrap_ensemble calibrator missing bootstrap_models")
            stacked = np.stack([m.predict(X) for m in self.bootstrap_models], axis=0)
            return stacked.std(axis=0)
        raise CC5Error(f"unsupported uncertainty method for scoring: {self.method!r}")

    def interval(self, point_preds: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = self.score(X, point_preds)
        return point_preds - half, point_preds + half


def _split_train_calibration_windows(ds: CC4Dataset, dev_window_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """VALIDATION-only calibration; TRAIN-only residual/bootstrap fitting.
    If VALIDATION is empty (should not happen on CC4/CC4b), fall back to a
    deterministic half-split of development windows so calibration still
    never touches evaluation splits."""
    causal = ds.causal_features[ds.causal_features["window_id"].isin(dev_window_ids)]
    train_ids = sorted(causal[causal["split"] == "TRAIN"]["window_id"])
    val_ids = sorted(causal[causal["split"] == "VALIDATION"]["window_id"])
    if train_ids and val_ids:
        return train_ids, val_ids
    ordered = sorted(dev_window_ids)
    if len(ordered) < 2:
        raise CC5Error("need at least 2 development windows to calibrate uncertainty")
    cut = max(1, len(ordered) // 2)
    return ordered[:cut], ordered[cut:]


def _fit_residual_model(X: np.ndarray, abs_resid: np.ndarray, seed: int) -> Any:
    model = Ridge(alpha=1.0)
    model.fit(X, abs_resid)
    return model


def fit_normalized_split_conformal(
    *,
    model_factory: Callable[[], Any],
    encoder: FeatureEncoder,
    ds: CC4Dataset,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    seed: int = 0,
    target_coverage: float = 0.8,
    residual_eps: float = 1e-3,
) -> UncertaintyCalibrator:
    t0 = time.perf_counter()
    X_train, y_train, _ = build_regret_training_table(ds, encoder, train_ids)
    point_model = model_factory()
    point_model.fit(X_train, y_train)
    train_abs = np.abs(y_train - point_model.predict(X_train))
    residual_model = _fit_residual_model(X_train, train_abs, seed=seed)

    X_val, y_val, _ = build_regret_training_table(ds, encoder, val_ids)
    val_preds = point_model.predict(X_val)
    scale = np.maximum(residual_model.predict(X_val), residual_eps)
    scores = np.abs(y_val - val_preds) / scale
    alpha = 1.0 - target_coverage
    # Finite-sample split-conformal quantile (Vovk et al.)
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n) if n else 1.0
    conformal_q = float(np.quantile(scores, q_level)) if n else 1.0
    half_widths = conformal_q * scale
    covered = (y_val >= val_preds - half_widths) & (y_val <= val_preds + half_widths)
    empirical_coverage = float(covered.mean()) if n else 0.0
    calibration_error = abs(empirical_coverage - target_coverage)
    # Default threshold: validation quantile of selected-candidate uncertainty;
    # refined later by calibrate_uncertainty_threshold_for_selection.
    selected_unc = []
    for wid in val_ids:
        causal_row = ds.causal_features.set_index("window_id").loc[wid]
        _, X_c = build_candidate_matrix(ds, encoder, causal_row)
        preds = point_model.predict(X_c)
        unc = conformal_q * np.maximum(residual_model.predict(X_c), residual_eps)
        selected_unc.append(float(unc[int(np.argmin(preds))]))
    default_threshold = float(np.quantile(selected_unc, 0.75)) if selected_unc else conformal_q
    overhead = time.perf_counter() - t0
    manifest = {
        "method": "normalized_split_conformal",
        "schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "target_coverage": target_coverage,
        "empirical_coverage": empirical_coverage,
        "calibration_error": calibration_error,
        "conformal_quantile": conformal_q,
        "n_calibration_rows": int(n),
        "n_train_windows": len(train_ids),
        "n_calibration_windows": len(val_ids),
        "seed": seed,
    }
    return UncertaintyCalibrator(
        method="normalized_split_conformal",
        schema_version=UNCERTAINTY_SCHEMA_VERSION,
        residual_model=residual_model,
        conformal_quantile=conformal_q,
        uncertainty_threshold=default_threshold,
        target_coverage=target_coverage,
        empirical_coverage=empirical_coverage,
        calibration_error=calibration_error,
        seed=seed,
        train_window_ids=list(train_ids),
        calibration_window_ids=list(val_ids),
        bootstrap_models=None,
        n_bootstrap=0,
        residual_eps=residual_eps,
        point_model_train_only=point_model,
        runtime_overhead_s=overhead,
        calibration_manifest=manifest,
    )


def fit_bootstrap_ensemble_uncertainty(
    *,
    model_factory: Callable[[], Any],
    encoder: FeatureEncoder,
    ds: CC4Dataset,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    seed: int = 0,
    n_bootstrap: int = 12,
    target_coverage: float = 0.8,
) -> UncertaintyCalibrator:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    models: list[Any] = []
    for b in range(n_bootstrap):
        sample = [train_ids[i] for i in rng.integers(0, len(train_ids), size=len(train_ids))]
        X_b, y_b, _ = build_regret_training_table(ds, encoder, sample)
        model = model_factory()
        model.fit(X_b, y_b)
        models.append(model)

    X_val, y_val, _ = build_regret_training_table(ds, encoder, val_ids)
    stacked = np.stack([m.predict(X_val) for m in models], axis=0)
    mean_pred = stacked.mean(axis=0)
    unc = stacked.std(axis=0)
    # Scale factor so mean_pred ± k*unc covers target_coverage on validation.
    abs_z = np.abs(y_val - mean_pred) / np.maximum(unc, 1e-9)
    alpha = 1.0 - target_coverage
    n = len(abs_z)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n) if n else 1.0
    k = float(np.quantile(abs_z, q_level)) if n else 1.0
    half = k * unc
    covered = (y_val >= mean_pred - half) & (y_val <= mean_pred + half)
    empirical_coverage = float(covered.mean()) if n else 0.0
    calibration_error = abs(empirical_coverage - target_coverage)

    selected_unc = []
    for wid in val_ids:
        causal_row = ds.causal_features.set_index("window_id").loc[wid]
        _, X_c = build_candidate_matrix(ds, encoder, causal_row)
        boot = np.stack([m.predict(X_c) for m in models], axis=0)
        preds = boot.mean(axis=0)
        selected_unc.append(float(boot.std(axis=0)[int(np.argmin(preds))]))
    default_threshold = float(np.quantile(selected_unc, 0.75)) if selected_unc else 1.0
    overhead = time.perf_counter() - t0
    manifest = {
        "method": "bootstrap_ensemble",
        "schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "target_coverage": target_coverage,
        "empirical_coverage": empirical_coverage,
        "calibration_error": calibration_error,
        "coverage_scale_k": k,
        "n_bootstrap": n_bootstrap,
        "n_calibration_rows": int(n),
        "n_train_windows": len(train_ids),
        "n_calibration_windows": len(val_ids),
        "seed": seed,
    }
    # Store k inside conformal_quantile field for interval construction parity.
    return UncertaintyCalibrator(
        method="bootstrap_ensemble",
        schema_version=UNCERTAINTY_SCHEMA_VERSION,
        residual_model=None,
        conformal_quantile=k,
        uncertainty_threshold=default_threshold,
        target_coverage=target_coverage,
        empirical_coverage=empirical_coverage,
        calibration_error=calibration_error,
        seed=seed,
        train_window_ids=list(train_ids),
        calibration_window_ids=list(val_ids),
        bootstrap_models=models,
        n_bootstrap=n_bootstrap,
        residual_eps=1e-9,
        point_model_train_only=None,
        runtime_overhead_s=overhead,
        calibration_manifest=manifest,
    )


def calibrate_uncertainty_threshold_for_selection(
    *,
    calibrator: UncertaintyCalibrator,
    point_model: Any,
    encoder: FeatureEncoder,
    ds: CC4Dataset,
    val_ids: Sequence[str],
    fallback: LookupBaseline,
    quantiles: Sequence[float] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 1.0),
) -> tuple[float, pd.DataFrame]:
    """Pick an uncertainty threshold on VALIDATION only by maximizing mean
    ANWG of (use model if selected-candidate uncertainty <= t else fallback)."""
    rows = []
    for wid in val_ids:
        causal_row = ds.causal_features.set_index("window_id").loc[wid]
        candidate_ids, X = build_candidate_matrix(ds, encoder, causal_row)
        preds = point_model.predict(X)
        unc = calibrator.score(X, preds)
        best_idx = int(np.argmin(preds))
        model_cand = candidate_ids[best_idx]
        regime = causal_row["regime"] if "regime" in causal_row.index else causal_row.get("regime", "")
        fallback_cand = fallback.select(str(regime))
        rows.append({
            "window_id": wid,
            "uncertainty": float(unc[best_idx]),
            "model_candidate": model_cand,
            "fallback_candidate": fallback_cand,
            "model_anwg": _actual_metrics(ds, wid, model_cand)[PRIMARY_COL],
            "fallback_anwg": _actual_metrics(ds, wid, fallback_cand)[PRIMARY_COL],
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return calibrator.uncertainty_threshold, table
    unc_values = table["uncertainty"].to_numpy()
    best_t, best_anwg = float(np.max(unc_values)), float("-inf")
    records = []
    for q in quantiles:
        t = float(np.quantile(unc_values, q))
        chosen = np.where(table["uncertainty"].to_numpy() <= t, table["model_anwg"], table["fallback_anwg"])
        mean_anwg = float(np.mean(chosen))
        records.append({"quantile": q, "threshold": t, "mean_anwg": mean_anwg, "abstention_rate": float((table["uncertainty"] > t).mean())})
        if mean_anwg > best_anwg or (mean_anwg == best_anwg and t > best_t):
            best_anwg, best_t = mean_anwg, t
    calibrator.uncertainty_threshold = best_t
    calibrator.calibration_manifest["selection_threshold"] = best_t
    calibrator.calibration_manifest["selection_threshold_grid"] = records
    calibrator.calibration_manifest["selection_threshold_val_anwg"] = best_anwg
    return best_t, pd.DataFrame(records)


def choose_uncertainty_method(
    candidates: Mapping[str, UncertaintyCalibrator],
) -> UncertaintyCalibrator:
    """Prefer lower |coverage - target|; break ties by lower runtime overhead."""
    if not candidates:
        raise CC5Error("no uncertainty calibrators to choose from")
    ordered = sorted(
        candidates.values(),
        key=lambda c: (c.calibration_error, c.runtime_overhead_s, c.method),
    )
    return ordered[0]


def assert_uncertainty_calibrator_compatible(calibrator: UncertaintyCalibrator | None) -> None:
    if calibrator is None:
        raise CC5Error("missing uncertainty calibrator -- stale artifact rejected")
    if calibrator.schema_version != UNCERTAINTY_SCHEMA_VERSION:
        raise CC5Error(
            f"stale uncertainty calibration schema {calibrator.schema_version} "
            f"(expected {UNCERTAINTY_SCHEMA_VERSION})"
        )
    if calibrator.method not in SUPPORTED_UNCERTAINTY_METHODS:
        raise CC5Error(f"unsupported uncertainty method in artifact: {calibrator.method!r}")


def fit_model_agnostic_uncertainty(
    *,
    model_name: str,
    model_factory: Callable[[], Any],
    encoder: FeatureEncoder,
    ds: CC4Dataset,
    dev_window_ids: Sequence[str],
    seed: int = 0,
    target_coverage: float = 0.8,
    n_bootstrap: int = 12,
    fallback_for_threshold: LookupBaseline | None = None,
) -> tuple[UncertaintyCalibrator, dict[str, UncertaintyCalibrator], pd.DataFrame]:
    """Fit conformal + bootstrap calibrators on TRAIN/VALIDATION only; retain
    the better-calibrated method. Does not change point-model selection."""
    train_ids, val_ids = _split_train_calibration_windows(ds, dev_window_ids)
    conformal = fit_normalized_split_conformal(
        model_factory=model_factory, encoder=encoder, ds=ds,
        train_ids=train_ids, val_ids=val_ids, seed=seed, target_coverage=target_coverage,
    )
    bootstrap = fit_bootstrap_ensemble_uncertainty(
        model_factory=model_factory, encoder=encoder, ds=ds,
        train_ids=train_ids, val_ids=val_ids, seed=seed, n_bootstrap=n_bootstrap,
        target_coverage=target_coverage,
    )
    candidates = {
        conformal.method: conformal,
        bootstrap.method: bootstrap,
    }
    # Threshold calibration needs the deployed point model fitted on all
    # development windows; caller passes fallback. We temporarily fit a
    # point model on all development for threshold search consistency with
    # the eventual deployed model.
    point_model = model_factory()
    X_dev, y_dev, _ = build_regret_training_table(ds, encoder, dev_window_ids)
    point_model.fit(X_dev, y_dev)
    fb = fallback_for_threshold or fit_best_fixed_policy(ds, train_ids)
    grids = []
    for cal in candidates.values():
        _, grid = calibrate_uncertainty_threshold_for_selection(
            calibrator=cal, point_model=point_model, encoder=encoder, ds=ds,
            val_ids=val_ids, fallback=fb,
        )
        grid = grid.copy()
        grid["method"] = cal.method
        grids.append(grid)
    selected = choose_uncertainty_method(candidates)
    selected.calibration_manifest["compared_methods"] = {
        name: {
            "empirical_coverage": c.empirical_coverage,
            "calibration_error": c.calibration_error,
            "runtime_overhead_s": c.runtime_overhead_s,
            "uncertainty_threshold": c.uncertainty_threshold,
        }
        for name, c in candidates.items()
    }
    selected.calibration_manifest["selected_for_model"] = model_name
    return selected, candidates, pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()


# ---------------------------------------------------------------------------
# Cross-validation (leave-one-window-out, dev windows only)
# ---------------------------------------------------------------------------


def leave_one_window_out_cv(
    models: Mapping[str, Any],
    ds: CC4Dataset,
    encoder: FeatureEncoder,
    dev_window_ids: Sequence[str],
) -> pd.DataFrame:
    """For each model, fit on all-but-one dev window and evaluate the
    argmin-predicted-regret selection on the held-out window; report mean
    ANWG achieved across the folds. Model selection must never touch
    evaluation-split windows -- this function only ever sees dev windows."""
    rows = []
    for held_out in dev_window_ids:
        train_windows = [w for w in dev_window_ids if w != held_out]
        X_train, y_train, _ = build_regret_training_table(ds, encoder, train_windows)
        for name, model_factory in models.items():
            model = model_factory()
            model.fit(X_train, y_train)
            selected = select_candidate_for_window(ds, encoder, model, held_out)
            actual = _actual_metrics(ds, held_out, selected)
            rows.append({"model": name, "held_out_window": held_out, "selected_candidate": selected, PRIMARY_COL: actual[PRIMARY_COL], "regret": actual["regret"]})
    return pd.DataFrame(rows)


def build_candidate_matrix(ds: CC4Dataset, encoder: FeatureEncoder, causal_row: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    """Build the (candidate_ids, feature_matrix) pair for every one of CC4's
    pre-verified candidates against a single window's causal features.
    Shared by `select_candidate_for_window`, `select_composition_with_fallback`,
    and any external caller (e.g. cc5_uncertainty_regime_refinement.py) that
    needs the same candidate encoding without duplicating this lookup logic."""
    candidates = ds.candidate_compositions
    candidate_ids: list[str] = []
    rows: list[np.ndarray] = []
    for _, cand in candidates.iterrows():
        weights = {p["primitive_name"]: p["weight"] for _, p in ds.composition_parameters[
            (ds.composition_parameters["candidate_id"] == cand["candidate_id"]) & ds.composition_parameters["primitive_name"].notna()
        ].iterrows()}
        extras_rows = ds.composition_parameters[ds.composition_parameters["candidate_id"] == cand["candidate_id"]]
        extras = json.loads(extras_rows.iloc[0]["extra_params_json"]) if not extras_rows.empty else {}
        candidate_ids.append(cand["candidate_id"])
        rows.append(encoder.row_vector(causal_row, cand["family"], weights, extras))
    return candidate_ids, np.stack(rows)


def select_candidate_for_window(ds: CC4Dataset, encoder: FeatureEncoder, model: Any, window_id: str) -> str:
    """Evaluate `model`'s predicted regret for every pre-verified candidate
    against this window's causal features and return the argmin (i.e. the
    recommended, already-verified composition)."""
    causal_row = ds.causal_features.set_index("window_id").loc[window_id]
    candidate_ids, X = build_candidate_matrix(ds, encoder, causal_row)
    preds = model.predict(X)
    return candidate_ids[int(np.argmin(preds))]


def _actual_metrics(ds: CC4Dataset, window_id: str, candidate_id: str) -> dict[str, float]:
    row = ds.per_window_results[(ds.per_window_results["window_id"] == window_id) & (ds.per_window_results["candidate_id"] == candidate_id)]
    regret_row = ds.regret_matrix[(ds.regret_matrix["window_id"] == window_id) & (ds.regret_matrix["candidate_id"] == candidate_id)]
    return {
        PRIMARY_COL: float(row.iloc[0][PRIMARY_COL]),
        COMPLETION_COL: float(row.iloc[0][COMPLETION_COL]),
        "regret": float(regret_row.iloc[0]["regret"]) if not regret_row.empty else float("nan"),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (honest about n=6 held-out windows)
# ---------------------------------------------------------------------------


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    vals = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(vals) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return {
        "mean": float(vals.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "n": len(vals),
    }


# ---------------------------------------------------------------------------
# Deployable predictor artifact + runtime wrapper
# ---------------------------------------------------------------------------


@dataclass
class PredictorArtifact:
    model_name: str
    model: Any
    encoder: FeatureEncoder
    gate: UncertaintyOODGate
    fallback: LookupBaseline
    supports_ensemble_uncertainty: bool
    dsl_schema_version: int
    compiler_version: str
    dataset_config_hash: str
    dataset_dir: str
    git_sha: str
    feature_schema: list[str]
    target_definition: str
    split_definition: dict[str, Any]
    hyperparameters: dict[str, Any]
    uncertainty_method: str
    ood_method: str
    objective_definition: str
    training_timestamp: str
    dependency_versions: dict[str, str]
    uncertainty_calibrator: UncertaintyCalibrator | None = None
    gate_mode: str = "ood_or_uncertainty"
    regime_fallback_rules: dict[str, str] | None = None


def _predict_with_uncertainty(artifact: PredictorArtifact, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_regret_per_candidate, per-candidate_uncertainty).

    Preference order:
    1. model-agnostic calibrator (normalized split-conformal or bootstrap);
    2. native random-forest per-tree std when no calibrator is attached;
    3. zeros only when neither is available (legacy / tests).
    """
    preds = artifact.model.predict(X)
    calibrator = getattr(artifact, "uncertainty_calibrator", None)
    if calibrator is not None:
        assert_uncertainty_calibrator_compatible(calibrator)
        uncertainty = calibrator.score(X, preds)
        return preds, uncertainty
    if artifact.supports_ensemble_uncertainty and hasattr(artifact.model, "estimators_"):
        tree_preds = np.stack([t.predict(X) for t in artifact.model.estimators_], axis=0)
        uncertainty = tree_preds.std(axis=0)
    else:
        uncertainty = np.zeros(len(X))
    return preds, uncertainty


def select_composition_with_fallback(
    artifact: PredictorArtifact,
    ds: CC4Dataset,
    causal_row: Mapping[str, Any],
    *,
    gate_mode: str | None = None,
    fallback_override: LookupBaseline | None = None,
) -> dict[str, Any]:
    """Runtime wrapper: (1) extract causal features, (2) predict a verified
    composition, (3) apply uncertainty/OOD checks, (4) fall back safely when
    needed, (5) return a decision record suitable for logging.

    gate_mode options (validation-tuned externally; default from artifact):
      - ood_only
      - uncertainty_only
      - ood_or_uncertainty
      - regime_aware
    """
    mode = gate_mode or artifact.gate_mode
    fallback = fallback_override or artifact.fallback
    causal_vec = artifact.encoder.causal_vector(causal_row)
    ood_score = artifact.gate.ood_score(causal_vec)
    is_ood = ood_score > artifact.gate.ood_z_threshold

    candidate_ids, X = build_candidate_matrix(ds, artifact.encoder, causal_row)
    t0 = time.perf_counter()
    preds, uncertainties = _predict_with_uncertainty(artifact, X)
    inference_overhead_s = time.perf_counter() - t0
    best_idx = int(np.argmin(preds))
    best_candidate = candidate_ids[best_idx]
    best_uncertainty = float(uncertainties[best_idx])
    has_uncertainty = getattr(artifact, "uncertainty_calibrator", None) is not None or artifact.supports_ensemble_uncertainty
    is_uncertain = has_uncertainty and best_uncertainty > artifact.gate.uncertainty_threshold

    regime = causal_row.get("regime", "") if hasattr(causal_row, "get") else causal_row["regime"]
    regime = str(regime)
    reasons: list[str] = []
    abstained = False
    if mode == "ood_only":
        abstained = bool(is_ood)
        if is_ood:
            reasons.append("ood")
    elif mode == "uncertainty_only":
        abstained = bool(is_uncertain)
        if is_uncertain:
            reasons.append("high_uncertainty")
    elif mode == "regime_aware":
        rules = artifact.regime_fallback_rules or {}
        if regime in rules and rules[regime] == "fallback":
            abstained = True
            reasons.append("regime_rule")
        else:
            abstained = bool(is_ood or is_uncertain)
            if is_ood:
                reasons.append("ood")
            if is_uncertain:
                reasons.append("high_uncertainty")
    else:  # ood_or_uncertainty
        abstained = bool(is_ood or is_uncertain)
        if is_ood:
            reasons.append("ood")
        if is_uncertain:
            reasons.append("high_uncertainty")

    selected = fallback.select(regime) if abstained else best_candidate

    return {
        "selected_candidate_id": selected,
        "model_recommended_candidate_id": best_candidate,
        "predicted_regret": float(preds[best_idx]),
        "uncertainty": best_uncertainty,
        "ood_score": ood_score,
        "abstained": abstained,
        "fallback_reason": ",".join(reasons) if reasons else None,
        "gate_mode": mode,
        "inference_overhead_s": inference_overhead_s,
        "fallback_policy_name": fallback.name,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_selector(
    selector: Callable[[Mapping[str, Any]], dict[str, Any]],
    ds: CC4Dataset,
    window_ids: Sequence[str],
) -> pd.DataFrame:
    """Run `selector` (any callable causal_row -> {"selected_candidate_id": ...})
    over every window in `window_ids` and look up each selection's actual
    (already-simulator-executed) outcome. No new simulator executions."""
    causal_by_window = ds.causal_features.set_index("window_id")
    rows = []
    for window_id in window_ids:
        causal_row = causal_by_window.loc[window_id]
        decision = selector(causal_row)
        actual = _actual_metrics(ds, window_id, decision["selected_candidate_id"])
        rows.append({
            "window_id": window_id, "split": causal_row["split"], "regime": causal_row["regime"],
            "selected_candidate_id": decision["selected_candidate_id"],
            PRIMARY_COL: actual[PRIMARY_COL], COMPLETION_COL: actual[COMPLETION_COL], "regret": actual["regret"],
            "abstained": decision.get("abstained", False), "fallback_reason": decision.get("fallback_reason"),
        })
    return pd.DataFrame(rows)


def regret_vs_oracle_fixed(ds: CC4Dataset, eval_df: pd.DataFrame) -> pd.DataFrame:
    fixed_rows = ds.per_window_results[ds.per_window_results["family"] == "fixed_policy"]
    oracle_fixed = fixed_rows.loc[fixed_rows.groupby("window_id")[PRIMARY_COL].idxmax()].set_index("window_id")[PRIMARY_COL]
    out = eval_df.copy()
    out["oracle_fixed_anwg"] = out["window_id"].map(oracle_fixed)
    out["regret_vs_oracle_fixed"] = out["oracle_fixed_anwg"] - out[PRIMARY_COL]
    return out


# ---------------------------------------------------------------------------
# Decision-gate verdict
# ---------------------------------------------------------------------------


def determine_cc5_verdict(
    predictor_eval: pd.DataFrame,
    best_fixed_eval: pd.DataFrame,
    best_global_eval: pd.DataFrame,
    hard_selector_eval: pd.DataFrame,
    near_tie_windows: set[str],
) -> dict[str, Any]:
    """Compute (not presuppose) PROCEED/REGIME_SPECIFIC_ONLY/STOP_OR_REDESIGN/
    INCONCLUSIVE from the actual held-out numbers. With only a handful of
    held-out windows, bootstrap CIs are wide by construction -- reported,
    not hidden."""
    n_eval = len(predictor_eval)
    predictor_ci = bootstrap_ci(predictor_eval[PRIMARY_COL].tolist())
    fixed_ci = bootstrap_ci(best_fixed_eval[PRIMARY_COL].tolist())
    global_ci = bootstrap_ci(best_global_eval[PRIMARY_COL].tolist())
    selector_ci = bootstrap_ci(hard_selector_eval[PRIMARY_COL].tolist())

    non_near_tie = predictor_eval[~predictor_eval["window_id"].isin(near_tie_windows)]
    non_near_tie_ci = bootstrap_ci(non_near_tie[PRIMARY_COL].tolist()) if not non_near_tie.empty else {"mean": None, "n": 0}

    completion_violations = int((predictor_eval[COMPLETION_COL] < best_fixed_eval.set_index("window_id").loc[predictor_eval["window_id"], COMPLETION_COL].to_numpy() - 0.05).sum())

    beats_fixed = predictor_ci["mean"] >= fixed_ci["mean"]
    beats_global = predictor_ci["mean"] >= global_ci["mean"]
    competitive_with_selector = predictor_ci["mean"] >= selector_ci["mean"] - 0.01
    cis_overlap_fixed = not (predictor_ci["ci_low"] > fixed_ci["ci_high"] or predictor_ci["ci_high"] < fixed_ci["ci_low"])

    if completion_violations > 0:
        status, reason = "STOP_OR_REDESIGN", f"{completion_violations} evaluation window(s) show a completion-fraction regression >0.05 vs best fixed"
    elif n_eval < 8:
        if beats_fixed and beats_global and competitive_with_selector:
            status = "INCONCLUSIVE"
            reason = (
                f"predictor point-estimate beats all baselines on only n={n_eval} held-out windows; "
                "bootstrap confidence intervals are too wide at this sample size to support a PROCEED "
                "claim of generalization -- point estimates are directional evidence, not proof"
            )
        else:
            status = "STOP_OR_REDESIGN" if not cis_overlap_fixed and not beats_fixed else "INCONCLUSIVE"
            reason = "predictor does not clearly beat best fixed policy on held-out windows"
    elif beats_fixed and beats_global and competitive_with_selector:
        status, reason = "PROCEED", "predictor beats best fixed and best global composition, competitive with hard selector, with sufficient held-out sample size"
    else:
        status, reason = "REGIME_SPECIFIC_ONLY", "predictor shows value on some but not all comparisons"

    return {
        "status": status,
        "reason": reason,
        "n_evaluation_windows": n_eval,
        "predictor_anwg": predictor_ci,
        "best_fixed_anwg": fixed_ci,
        "best_global_composition_anwg": global_ci,
        "existing_hard_selector_anwg": selector_ci,
        "non_near_tie_predictor_anwg": non_near_tie_ci,
        "completion_violations": completion_violations,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def heartbeat(out_dir: Path, stage: str, **payload: Any) -> None:
    path = out_dir / "checkpoints" / "heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **payload,
    }, indent=2, sort_keys=True, default=str))


def resolve_output_dir(root: str | Path, *, timestamp: str | None) -> Path:
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / str(root) / stamp


@dataclass
class CC5Result:
    output_dir: Path
    manifest: dict[str, Any]
    verdict: dict[str, Any]


def run_training(
    *,
    dataset_dir: str | Path,
    output_root: str = "results/cc5_contextual_composition_predictor",
    timestamp: str | None = None,
    resume_dir: str | Path | None = None,
    ood_z_threshold: float = 2.0,
    uncertainty_threshold_quantile: float = 0.75,
    seed: int = 0,
) -> CC5Result:
    ds = load_cc4_dataset(dataset_dir)
    audit_findings = validate_cc4_dataset(ds)

    output_dir = Path(resume_dir) if resume_dir is not None else resolve_output_dir(output_root, timestamp=timestamp)
    if (output_dir / "manifest.json").exists():
        manifest = json.loads((output_dir / "manifest.json").read_text())
        verdict = json.loads((output_dir / "verdict.json").read_text())
        return CC5Result(output_dir=output_dir, manifest=manifest, verdict=verdict)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_audit.json").write_text(json.dumps({"findings": audit_findings}, indent=2))
    heartbeat(output_dir, "loaded_and_audited_dataset")

    t0 = time.perf_counter()
    dev_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.development_splits)]["window_id"])
    eval_ids = sorted(ds.causal_features[ds.causal_features["split"].isin(ds.evaluation_splits)]["window_id"])

    encoder = FeatureEncoder.fit(ds.causal_features[ds.causal_features["window_id"].isin(dev_ids)])

    # --- Target 2 (primary): regret regression + leave-one-window-out model selection ---
    regressor_factories = build_regret_regressor_factories(seed=seed)
    cv_results = leave_one_window_out_cv(regressor_factories, ds, encoder, dev_ids)
    cv_results.to_csv(output_dir / "cv_model_selection.csv", index=False)
    heartbeat(output_dir, "cross_validated_models", n_models=len(regressor_factories))

    cv_mean_by_model = cv_results.groupby("model")[PRIMARY_COL].mean().sort_values(ascending=False)
    best_model_name = cv_mean_by_model.index[0]
    best_model = regressor_factories[best_model_name]()
    X_dev, y_dev, _ = build_regret_training_table(ds, encoder, dev_ids)
    best_model.fit(X_dev, y_dev)

    # --- Target 1: hard composition-family classification (reported, underpowered) ---
    X_class, y_class, class_window_ids = build_class_training_table(ds, encoder, dev_ids)
    nearest_regime = NearestRegimeClassifier().fit(X_class, y_class)
    class_tree = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(X_class, y_class)
    class_report = pd.DataFrame({
        "window_id": class_window_ids, "true_oracle_family": y_class,
        "nearest_regime_pred": nearest_regime.predict(X_class), "decision_tree_pred": class_tree.predict(X_class),
    })

    # --- Target 3 (direct parameter regression): assessed, not trained ---
    wmix_oracle = ds.oracle_labels[ds.oracle_labels["oracle_family"] == "weighted_primitive_mixture"]
    target3_note = (
        f"{len(wmix_oracle)} of {len(ds.oracle_labels)} windows have a weighted_primitive_mixture oracle "
        f"winner ({len(PRIMITIVE_POOL)}-dim weight target); direct parameter regression was not trained "
        "(fewer positive examples than target dimensions -- non-identifiable at this sample size)."
    )

    # --- Non-learned baselines (dev-fit) ---
    best_fixed = fit_best_fixed_policy(ds, dev_ids)
    best_global = fit_best_global_composition(ds, dev_ids)
    hard_selector = fit_existing_hard_selector(ds, dev_ids)

    # --- Model-agnostic uncertainty calibration (TRAIN fit / VALIDATION calibrate) ---
    # Point-model selection criteria are unchanged; uncertainty is attached after.
    supports_ensemble = best_model_name == "random_forest"
    calibrator, all_calibrators, threshold_grid = fit_model_agnostic_uncertainty(
        model_name=best_model_name,
        model_factory=regressor_factories[best_model_name],
        encoder=encoder,
        ds=ds,
        dev_window_ids=dev_ids,
        seed=seed,
        fallback_for_threshold=best_fixed,
    )
    assert_uncertainty_calibrator_compatible(calibrator)
    uncertainty_threshold = float(calibrator.uncertainty_threshold)
    gate = UncertaintyOODGate.fit(
        encoder, ds.causal_features, dev_ids,
        ood_z_threshold=ood_z_threshold, uncertainty_threshold=uncertainty_threshold,
    )
    heartbeat(
        output_dir, "calibrated_uncertainty_gate",
        best_model=best_model_name,
        uncertainty_method=calibrator.method,
        supports_ensemble_uncertainty=supports_ensemble,
        calibration_error=calibrator.calibration_error,
        empirical_coverage=calibrator.empirical_coverage,
    )
    (output_dir / "calibration_manifest.json").write_text(
        json.dumps(calibrator.calibration_manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    threshold_grid.to_csv(output_dir / "uncertainty_threshold_grid.csv", index=False)
    pd.DataFrame([
        {
            "method": name,
            "empirical_coverage": c.empirical_coverage,
            "calibration_error": c.calibration_error,
            "runtime_overhead_s": c.runtime_overhead_s,
            "uncertainty_threshold": c.uncertainty_threshold,
            "selected": name == calibrator.method,
        }
        for name, c in all_calibrators.items()
    ]).to_csv(output_dir / "uncertainty_method_comparison.csv", index=False)

    git = git_state()
    artifact = PredictorArtifact(
        model_name=best_model_name, model=best_model, encoder=encoder, gate=gate, fallback=best_fixed,
        supports_ensemble_uncertainty=supports_ensemble,
        dsl_schema_version=2, compiler_version="cc3.1",
        dataset_config_hash=ds.manifest.get("config_hash", ""), dataset_dir=display_path(ds.dataset_dir),
        git_sha=git["commit"], feature_schema=encoder.feature_names,
        target_definition="regret = window_oracle_anwg - candidate_anwg; argmin predicted regret over CC4's 34 pre-verified candidates",
        split_definition={"development_splits": list(ds.development_splits), "evaluation_splits": list(ds.evaluation_splits), "dev_windows": dev_ids, "eval_windows": eval_ids},
        hyperparameters={
            "ood_z_threshold": ood_z_threshold,
            "uncertainty_threshold": uncertainty_threshold,
            "uncertainty_threshold_quantile": uncertainty_threshold_quantile,
            "seed": seed,
            "uncertainty_schema_version": UNCERTAINTY_SCHEMA_VERSION,
        },
        uncertainty_method=calibrator.method,
        ood_method="max_abs_zscore_vs_dev_causal_feature_distribution",
        objective_definition=PRIMARY,
        training_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        dependency_versions={"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version()},
        uncertainty_calibrator=calibrator,
        gate_mode="ood_or_uncertainty",
    )

    # --- Evaluation (evaluation-split windows, touched exactly once) ---
    predictor_eval = evaluate_selector(lambda row: select_composition_with_fallback(artifact, ds, row), ds, eval_ids)
    predictor_eval = predictor_eval.rename(columns={"selected_candidate_id": "predictor_selected_candidate_id"})
    best_fixed_eval = evaluate_selector(lambda row: {"selected_candidate_id": best_fixed.select(row["regime"])}, ds, eval_ids)
    best_global_eval = evaluate_selector(lambda row: {"selected_candidate_id": best_global.select(row["regime"])}, ds, eval_ids)
    hard_selector_eval = evaluate_selector(lambda row: {"selected_candidate_id": hard_selector.select(row["regime"])}, ds, eval_ids)

    near_tie_eval_windows = set(ds.near_tie_flags[
        (ds.near_tie_flags["threshold"] == 0.005) & (ds.near_tie_flags["near_tie"] == True) & (ds.near_tie_flags["window_id"].isin(eval_ids))  # noqa: E712
    ]["window_id"])

    verdict = determine_cc5_verdict(predictor_eval, best_fixed_eval, best_global_eval, hard_selector_eval, near_tie_eval_windows)
    predictor_eval_with_regret = regret_vs_oracle_fixed(ds, predictor_eval)

    per_regime = predictor_eval.groupby("regime").agg(
        mean_anwg=(PRIMARY_COL, "mean"), mean_completion=(COMPLETION_COL, "mean"),
        mean_regret=("regret", "mean"), abstention_rate=("abstained", "mean"),
    ).reset_index()

    fallback_analysis = predictor_eval[predictor_eval["abstained"]][["window_id", "regime", "fallback_reason", PRIMARY_COL]]

    heartbeat(output_dir, "evaluated_on_held_out_windows", n_eval=len(eval_ids))

    # --- Write outputs ---
    predictor_eval_with_regret.to_csv(output_dir / "per_window_predictions.csv", index=False)
    per_regime.to_csv(output_dir / "per_regime_summaries.csv", index=False)
    fallback_analysis.to_csv(output_dir / "fallback_analysis.csv", index=False)
    class_report.to_csv(output_dir / "composition_class_predictions.csv", index=False)
    diag_rows = []
    for w in eval_ids:
        causal_row = ds.causal_features.set_index("window_id").loc[w]
        decision = select_composition_with_fallback(artifact, ds, causal_row)
        diag_rows.append({
            "window_id": w,
            "ood_score": decision["ood_score"],
            "is_ood": decision["ood_score"] > artifact.gate.ood_z_threshold,
            "uncertainty": decision["uncertainty"],
            "is_uncertain": decision["uncertainty"] > artifact.gate.uncertainty_threshold,
            "abstained": decision["abstained"],
            "fallback_reason": decision["fallback_reason"],
            "inference_overhead_s": decision["inference_overhead_s"],
        })
    pd.DataFrame(diag_rows).to_csv(output_dir / "uncertainty_ood_diagnostics.csv", index=False)

    baseline_comparison = pd.DataFrame([
        {"method": "predictor", **{f"anwg_{k}": v for k, v in verdict["predictor_anwg"].items()}},
        {"method": "best_fixed_policy", **{f"anwg_{k}": v for k, v in verdict["best_fixed_anwg"].items()}},
        {"method": "best_global_composition", **{f"anwg_{k}": v for k, v in verdict["best_global_composition_anwg"].items()}},
        {"method": "existing_hard_selector", **{f"anwg_{k}": v for k, v in verdict["existing_hard_selector_anwg"].items()}},
    ])
    baseline_comparison.to_csv(output_dir / "regret_tables.csv", index=False)

    (output_dir / "resolved_config.json").write_text(json.dumps({
        "dataset_dir": display_path(ds.dataset_dir), "ood_z_threshold": ood_z_threshold,
        "uncertainty_threshold_quantile": uncertainty_threshold_quantile, "seed": seed,
    }, indent=2))

    manifest = {
        "schema_version": 1,
        "experiment": "cc5_contextual_composition_predictor",
        "git_sha": git["commit"],
        "dataset_dir": display_path(ds.dataset_dir),
        "dataset_config_hash": artifact.dataset_config_hash,
        "feature_schema": artifact.feature_schema,
        "target_definition": artifact.target_definition,
        "target3_direct_parameter_regression": target3_note,
        "split_definition": artifact.split_definition,
        "model_type": best_model_name,
        "hyperparameters": artifact.hyperparameters,
        "uncertainty_method": artifact.uncertainty_method,
        "uncertainty_schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "uncertainty_calibration": calibrator.calibration_manifest,
        "ood_method": artifact.ood_method,
        "fallback_policy": best_fixed.selection,
        "gate_mode": artifact.gate_mode,
        "objective_definition": artifact.objective_definition,
        "training_timestamp": artifact.training_timestamp,
        "dependency_versions": artifact.dependency_versions,
        "no_live_api": True, "no_gpu": True, "no_real_vllm": True,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "cv_model_ranking": cv_mean_by_model.to_dict(),
    }
    manifest["verdict"] = verdict
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n")
    (output_dir / "model_card.md").write_text(render_model_card(manifest, verdict, per_regime))
    (output_dir / "replay_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(git rev-parse --show-toplevel)\"\n"
        f"python scripts/run_cc5_contextual_predictor.py --dataset-dir {display_path(ds.dataset_dir)} "
        f"--full-run --resume-dir {display_path(output_dir)}\n"
    )
    heartbeat(output_dir, "complete")

    return CC5Result(output_dir=output_dir, manifest=manifest, verdict=verdict)


def render_model_card(manifest: Mapping[str, Any], verdict: Mapping[str, Any], per_regime: pd.DataFrame) -> str:
    lines = [
        "# CC5 Contextual Composition Predictor Model Card",
        "",
        f"Model type: `{manifest['model_type']}`",
        f"Trained: {manifest['training_timestamp']}",
        f"Git SHA: `{manifest['git_sha']}`",
        f"Dataset: `{manifest['dataset_dir']}` (config hash `{manifest['dataset_config_hash']}`)",
        "",
        "## Verdict",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Reason: {verdict['reason']}",
        f"- Evaluation windows: {verdict['n_evaluation_windows']}",
        f"- Predictor ANWG: {verdict['predictor_anwg']}",
        f"- Best fixed policy ANWG: {verdict['best_fixed_anwg']}",
        f"- Best global composition ANWG: {verdict['best_global_composition_anwg']}",
        f"- Existing hard selector ANWG: {verdict['existing_hard_selector_anwg']}",
        f"- Completion-fraction violations: {verdict['completion_violations']}",
        "",
        "## Per-Regime Summary",
        "",
    ]
    for _, row in per_regime.iterrows():
        lines.append(f"- `{row['regime']}`: ANWG={row['mean_anwg']:.4f}, completion={row['mean_completion']:.4f}, abstention_rate={row['abstention_rate']:.2f}")
    lines += [
        "",
        "## Reproduction",
        "",
        "```bash",
        "bash replay_commands.sh",
        "```",
    ]
    return "\n".join(lines) + "\n"
