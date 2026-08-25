"""Structure-aware state-policy suitability models.

All models share the JointRewardModel-style interface (fit / predict_mean /
predict_uncertainty / predict_suitability) so they drop into the existing
selector/evaluation harness (selector.suitability.selector) unchanged.

Every model here explicitly exploits SchedulerGenomeV1 structural distance,
as distinct from selector.suitability.models.JointRewardModel(encoding=...),
which treats structure as just more RF input features. The key mechanism
all of these share: for query (x, pi_target), look up OTHER policies' TRUE
rewards at the SAME state x (available because this is full simulator
policy-vector data, not partial observability) and combine them weighted by
structural distance between pi_target and each neighbor -- i.e. genuine
nearest-neighbor-style transfer, not just "structure as a feature column."
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dataset import genome_table
from .encoders import structural_features

_EPS = 1e-9


def _structural_feature_matrix(policies: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Z-score-normalized structural feature matrix over `policies`
    (normalization fit only over this set, matching
    selector.pairwise_structural_distances's convention)."""
    genomes = genome_table(policies)
    feats = {name: structural_features(genomes[name]) for name in policies}
    cols = sorted(feats[policies[0]].keys())
    matrix = np.stack([[feats[name][c] for c in cols] for name in policies])
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / std
    return normalized, cols


class StructuralDistanceIndex:
    """Shared, fit-once structural distance lookup over a fixed policy
    universe (typically all 27 deployable policies, so a held-out policy's
    genome is always indexable even though it never appears in training
    rows)."""

    def __init__(self, all_policies: Sequence[str]):
        self.all_policies = list(all_policies)
        self._matrix, _cols = _structural_feature_matrix(self.all_policies)
        self._index = {name: i for i, name in enumerate(self.all_policies)}

    def distance(self, policy_a: str, policy_b: str) -> float:
        if policy_a == policy_b:
            return 0.0
        va = self._matrix[self._index[policy_a]]
        vb = self._matrix[self._index[policy_b]]
        return float(np.linalg.norm(va - vb))

    def nearest(self, target_policy: str, candidates: Sequence[str], k: int) -> List[Tuple[str, float]]:
        scored = sorted(
            ((c, self.distance(target_policy, c)) for c in candidates if c != target_policy),
            key=lambda t: t[1],
        )
        return scored[:k]


def _weights_from_distances(distances: Sequence[float], scheme: str, tau: float = 1.0) -> np.ndarray:
    d = np.asarray(distances, dtype=float)
    if scheme == "uniform":
        w = np.ones_like(d)
    elif scheme == "inverse_distance":
        w = 1.0 / (d + _EPS)
    elif scheme == "exponential":
        w = np.exp(-d / max(tau, _EPS))
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    total = w.sum()
    return w / total if total > _EPS else np.full_like(w, 1.0 / max(len(w), 1))


class _StateRewardLookup:
    """rows -> {state_id: {policy_name: reward}}, built once at fit time."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.table: Dict[str, Dict[str, float]] = {}
        self.state_features: Dict[str, Dict[str, float]] = {}
        for row in rows:
            if row.get("reward_anwg") is None:
                continue
            self.table.setdefault(row["state_id"], {})[row["policy_name"]] = float(row["reward_anwg"])
            self.state_features.setdefault(row["state_id"], row["state_features"])

    def rewards_at(self, state_id: str) -> Dict[str, float]:
        return self.table.get(state_id, {})


class StructuralKNNModel:
    """R_hat(x, pi) = weighted combination of the k structurally-nearest
    training policies' TRUE rewards at the same state x."""

    def __init__(
        self,
        *,
        name: str,
        all_policies: Sequence[str],
        k: int = 5,
        weighting: str = "inverse_distance",
        tau: float = 1.0,
        distance_index: Optional[StructuralDistanceIndex] = None,
    ):
        self.name = name
        self.all_policies = list(all_policies)
        self.k = int(k)
        self.weighting = weighting
        self.tau = float(tau)
        self.distance_index = distance_index or StructuralDistanceIndex(all_policies)
        self._lookup: Optional[_StateRewardLookup] = None
        self._train_policies: List[str] = []

    def fit(self, rows: Sequence[Mapping[str, Any]], *, lookup_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> "StructuralKNNModel":
        """`lookup_rows` (defaults to `rows`) supplies the state->policy->
        reward table used at predict time. This model is transductive by
        design (it needs *sibling* policies' true rewards at the exact
        query state) -- passing a wider `lookup_rows` (e.g. covering
        held-out TEST-split states too, via their *other* policies' rows)
        is a legitimate, different-but-fair use of already-known simulator
        ground truth, not target leakage: the target policy's own value at
        that state is never looked up (candidates always exclude it)."""
        self._lookup = _StateRewardLookup(lookup_rows if lookup_rows is not None else rows)
        self._train_policies = sorted({r["policy_name"] for r in rows})
        return self

    def _predict_one(self, state_id: str, target_policy: str) -> Tuple[float, float, int]:
        """Returns (mean, neighbor_disagreement_std, n_neighbors_used)."""
        assert self._lookup is not None
        rewards_here = self._lookup.rewards_at(state_id)
        candidates = [p for p in self._train_policies if p in rewards_here]
        if not candidates:
            return 0.0, 0.0, 0
        neighbors = self.distance_index.nearest(target_policy, candidates, self.k)
        if not neighbors:
            return 0.0, 0.0, 0
        names = [n for n, _ in neighbors]
        dists = [d for _, d in neighbors]
        weights = _weights_from_distances(dists, self.weighting, self.tau)
        values = np.asarray([rewards_here[n] for n in names])
        mean = float(np.sum(weights * values))
        disagreement = float(np.sqrt(np.sum(weights * (values - mean) ** 2)))
        return mean, disagreement, len(neighbors)

    def predict_mean(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray([self._predict_one(r["state_id"], r["policy_name"])[0] for r in rows])

    def predict_uncertainty(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """Neighbor disagreement: weighted std of the k neighbors' true
        rewards. Nonnegative by construction, policy-specific (depends on
        which neighbors and how spread out their rewards are)."""
        return np.asarray([self._predict_one(r["state_id"], r["policy_name"])[1] for r in rows])

    def predict_suitability(self, rows: Sequence[Mapping[str, Any]], *, lam: float = 0.5) -> np.ndarray:
        mu = self.predict_mean(rows)
        u = self.predict_uncertainty(rows)
        return mu - float(lam) * u

    def neighbor_diagnostics(self, target_policy: str, state_id: str) -> Dict[str, Any]:
        """For structural-extrapolation diagnostics: nearest-training-policy
        distance and average k-neighbor distance for one query."""
        assert self._lookup is not None
        rewards_here = self._lookup.rewards_at(state_id)
        candidates = [p for p in self._train_policies if p in rewards_here]
        all_dists = sorted(self.distance_index.distance(target_policy, c) for c in candidates)
        k_dists = all_dists[: self.k]
        return {
            "nearest_training_policy_distance": all_dists[0] if all_dists else None,
            "mean_k_neighbor_distance": float(np.mean(k_dists)) if k_dists else None,
        }


class KernelSuitabilityModel:
    """Nadaraya-Watson-style kernel regression: R_hat(x, pi) =
    sum_j K(pi, j) * R_true(x, j) / sum_j K(pi, j), K(pi_i, pi_j) =
    exp(-d_total(pi_i, pi_j) / tau). Uses ALL training policies (not top-k)
    as the neighbor pool -- the KNN model's exponential-weighting variant
    restricted to k neighbors is a special case of this."""

    def __init__(
        self,
        *,
        name: str,
        all_policies: Sequence[str],
        tau: float = 1.0,
        state_distance_weight: float = 0.0,
        distance_index: Optional[StructuralDistanceIndex] = None,
    ):
        self.name = name
        self.all_policies = list(all_policies)
        self.tau = float(tau)
        self.state_distance_weight = float(state_distance_weight)
        self.distance_index = distance_index or StructuralDistanceIndex(all_policies)
        self._lookup: Optional[_StateRewardLookup] = None
        self._train_policies: List[str] = []

    def fit(self, rows: Sequence[Mapping[str, Any]], *, lookup_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> "KernelSuitabilityModel":
        """See StructuralKNNModel.fit's `lookup_rows` docstring -- same
        transductive-lookup semantics."""
        self._lookup = _StateRewardLookup(lookup_rows if lookup_rows is not None else rows)
        self._train_policies = sorted({r["policy_name"] for r in rows})
        return self

    def _combined_distance(self, target_policy: str, neighbor_policy: str, state_features: Mapping[str, float]) -> float:
        d_policy = self.distance_index.distance(target_policy, neighbor_policy)
        # d_state is structurally 0 in this evaluation regime: every neighbor
        # lookup is at the *same* state_id as the query (full simulator
        # policy-vector data), so there is no second state to be distant
        # from. state_distance_weight is exposed for future cross-state
        # transfer use, but with the current same-state setup its
        # contribution is always exactly 0 -- documented, not silently
        # assumed away.
        d_state = 0.0
        return d_policy + self.state_distance_weight * d_state

    def _predict_one(self, state_id: str, target_policy: str, state_features: Mapping[str, float]) -> Tuple[float, float, int]:
        assert self._lookup is not None
        rewards_here = self._lookup.rewards_at(state_id)
        candidates = [p for p in self._train_policies if p in rewards_here]
        if not candidates:
            return 0.0, 0.0, 0
        dists = [self._combined_distance(target_policy, c, state_features) for c in candidates]
        weights = np.exp(-np.asarray(dists) / max(self.tau, _EPS))
        total = weights.sum()
        weights = weights / total if total > _EPS else np.full_like(weights, 1.0 / len(weights))
        values = np.asarray([rewards_here[c] for c in candidates])
        mean = float(np.sum(weights * values))
        disagreement = float(np.sqrt(np.sum(weights * (values - mean) ** 2)))
        return mean, disagreement, len(candidates)

    def predict_mean(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray([self._predict_one(r["state_id"], r["policy_name"], r["state_features"])[0] for r in rows])

    def predict_uncertainty(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray([self._predict_one(r["state_id"], r["policy_name"], r["state_features"])[1] for r in rows])

    def predict_suitability(self, rows: Sequence[Mapping[str, Any]], *, lam: float = 0.5) -> np.ndarray:
        mu = self.predict_mean(rows)
        u = self.predict_uncertainty(rows)
        return mu - float(lam) * u


class StateConditionedNeighborModel:
    """R_hat(x, pi) = sum_j alpha_j(x, pi) * R_hat_j(x), where alpha_j
    combines structural-distance kernel weight with an uncertainty discount
    from an independent per-policy reward model (policies whose own
    prediction is uncertain at this state contribute less as neighbors,
    even if structurally close) -- the "optionally uncertainty of policy
    j's predicted reward" formulation from the task brief."""

    def __init__(
        self,
        *,
        name: str,
        all_policies: Sequence[str],
        tau: float = 1.0,
        k: int = 5,
        uncertainty_discount: float = 1.0,
        distance_index: Optional[StructuralDistanceIndex] = None,
    ):
        self.name = name
        self.all_policies = list(all_policies)
        self.tau = float(tau)
        self.k = int(k)
        self.uncertainty_discount = float(uncertainty_discount)
        self.distance_index = distance_index or StructuralDistanceIndex(all_policies)
        self._lookup: Optional[_StateRewardLookup] = None
        self._train_policies: List[str] = []
        self._uncertainty_model = None

    def fit(self, rows: Sequence[Mapping[str, Any]], *, lookup_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> "StateConditionedNeighborModel":
        """See StructuralKNNModel.fit's `lookup_rows` docstring -- same
        transductive-lookup semantics. The uncertainty source model is
        still trained only on `rows` (never on `lookup_rows`), so it never
        learns from held-out-split data."""
        from .models import IndependentPerPolicyRewardModel

        self._lookup = _StateRewardLookup(lookup_rows if lookup_rows is not None else rows)
        self._train_policies = sorted({r["policy_name"] for r in rows})
        self._uncertainty_model = IndependentPerPolicyRewardModel(
            name=f"{self.name}_uncertainty_source", all_policies=self._train_policies, n_estimators=60, max_depth=4,
        ).fit(rows)
        return self

    def _policy_uncertainty_at(self, policy: str, state_features: Mapping[str, float]) -> float:
        if self._uncertainty_model is None or policy not in self._uncertainty_model.models:
            return 0.0
        fake_row = [{"state_features": state_features, "policy_name": policy}]
        try:
            return float(self._uncertainty_model.predict_uncertainty(fake_row)[0])
        except Exception:
            return 0.0

    def _predict_one(self, state_id: str, target_policy: str, state_features: Mapping[str, float]) -> Tuple[float, float, int]:
        assert self._lookup is not None
        rewards_here = self._lookup.rewards_at(state_id)
        candidates = [p for p in self._train_policies if p in rewards_here]
        if not candidates:
            return 0.0, 0.0, 0
        neighbors = self.distance_index.nearest(target_policy, candidates, self.k)
        if not neighbors:
            return 0.0, 0.0, 0
        names = [n for n, _ in neighbors]
        dists = np.asarray([d for _, d in neighbors])
        kernel_w = np.exp(-dists / max(self.tau, _EPS))
        discounts = np.asarray([
            1.0 / (1.0 + self.uncertainty_discount * self._policy_uncertainty_at(n, state_features)) for n in names
        ])
        alpha = kernel_w * discounts
        total = alpha.sum()
        alpha = alpha / total if total > _EPS else np.full_like(alpha, 1.0 / len(alpha))
        values = np.asarray([rewards_here[n] for n in names])
        mean = float(np.sum(alpha * values))
        disagreement = float(np.sqrt(np.sum(alpha * (values - mean) ** 2)))
        return mean, disagreement, len(neighbors)

    def predict_mean(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray([self._predict_one(r["state_id"], r["policy_name"], r["state_features"])[0] for r in rows])

    def predict_uncertainty(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray([self._predict_one(r["state_id"], r["policy_name"], r["state_features"])[1] for r in rows])

    def predict_suitability(self, rows: Sequence[Mapping[str, Any]], *, lam: float = 0.5) -> np.ndarray:
        mu = self.predict_mean(rows)
        u = self.predict_uncertainty(rows)
        return mu - float(lam) * u


class ResidualTransferModel:
    """R_hat(x, pi) = R_structural_neighbor(x, pi) + g(x, z_pi), where g is
    a small RF regressor over (state features + structural genome features)
    predicting the *residual* the neighbor estimate misses. g is trained
    leave-one-policy-out within the training set (each training row's
    neighbor estimate excludes that row's own policy) so it learns a
    genuine correction, not a pass-through of its own answer."""

    def __init__(
        self,
        *,
        name: str,
        all_policies: Sequence[str],
        k: int = 5,
        weighting: str = "inverse_distance",
        tau: float = 1.0,
        n_estimators: int = 100,
        max_depth: Optional[int] = 6,
        random_state: int = 42,
        distance_index: Optional[StructuralDistanceIndex] = None,
        weight_scheme: str = "uniform",
        weight_epsilon: float = 0.001,
    ):
        self.name = name
        self.all_policies = list(all_policies)
        self.knn = StructuralKNNModel(
            name=f"{name}_base_knn", all_policies=all_policies, k=k, weighting=weighting, tau=tau,
            distance_index=distance_index,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        # "uniform" | "margin" | "margin_plus_epsilon" -- reuses
        # selector.advanced.policy_margin_weights so the residual
        # correction is trained to matter most on states where getting the
        # policy choice right actually changes the outcome (the primary
        # selector objective is decision regret, not reward RMSE).
        self.weight_scheme = weight_scheme
        self.weight_epsilon = float(weight_epsilon)
        self._residual_model = None
        self._struct_cols: List[str] = []
        self._state_cols: List[str] = []

    def _matrix(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        out = np.zeros((len(rows), len(self._state_cols) + len(self._struct_cols)), dtype=float)
        for r, row in enumerate(rows):
            for c, col in enumerate(self._state_cols):
                out[r, c] = float(row["state_features"].get(col, 0.0))
            for c, col in enumerate(self._struct_cols):
                out[r, len(self._state_cols) + c] = float(row["policy_representation"].get(col, 0.0))
        return out

    def fit(self, rows: Sequence[Mapping[str, Any]], *, lookup_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> "ResidualTransferModel":
        """See StructuralKNNModel.fit's `lookup_rows` docstring. The
        residual-correction RF itself is always trained only on `rows`."""
        from sklearn.ensemble import RandomForestRegressor

        self.knn.fit(rows, lookup_rows=lookup_rows)
        self._state_cols = sorted({k for row in rows for k in row["state_features"].keys()})
        self._struct_cols = sorted({k for row in rows for k in row["policy_representation"].keys() if k != "mapping_status_summary"})

        # Leave-one-policy-out neighbor estimate for each training row, so g
        # learns a genuine residual rather than memorizing its own answer.
        residual_targets = []
        for row in rows:
            other_rows = [r for r in rows if r["policy_name"] != row["policy_name"]]
            loo_lookup = StructuralKNNModel(
                name="_loo", all_policies=self.all_policies, k=self.knn.k, weighting=self.knn.weighting,
                tau=self.knn.tau, distance_index=self.knn.distance_index,
            ).fit(other_rows)
            neighbor_est = loo_lookup._predict_one(row["state_id"], row["policy_name"])[0]
            residual_targets.append(float(row["reward_anwg"]) - neighbor_est)

        x = self._matrix(rows)
        y = np.asarray(residual_targets)
        sample_weight = None
        if self.weight_scheme != "uniform":
            # Margin-weighted training: reuses the top2-margin concept from
            # selector.suitability.selector (long-format-compatible) rather
            # than selector.advanced.policy_margin_weights, which expects
            # wide anwg_<policy> rows -- a different schema than these
            # (state, policy) long-format rows. Same idea, right shape.
            from .selector import top2_margin, true_reward_row

            lookup = self.knn._lookup
            margins_by_state = {
                sid: top2_margin(true_reward_row([{"policy_name": p, "reward_anwg": r} for p, r in rewards.items()]))
                for sid, rewards in lookup.table.items()
            }
            margins = np.asarray([margins_by_state.get(row["state_id"], 0.0) for row in rows])
            if self.weight_scheme == "margin":
                sample_weight = np.maximum(margins, 0.0)
            elif self.weight_scheme == "margin_plus_epsilon":
                sample_weight = margins + self.weight_epsilon
            else:
                raise ValueError(f"Unknown weight_scheme: {self.weight_scheme}")
            sample_weight = np.clip(sample_weight, self.weight_epsilon, None)
        self._residual_model = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, n_jobs=1,
        )
        self._residual_model.fit(x, y, sample_weight=sample_weight)
        return self

    def predict_mean(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        neighbor_est = self.knn.predict_mean(rows)
        residual = self._residual_model.predict(self._matrix(rows))
        return neighbor_est + residual

    def predict_uncertainty(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        # Combine neighbor disagreement with residual-model per-tree variance.
        neighbor_u = self.knn.predict_uncertainty(rows)
        x = self._matrix(rows)
        per_tree = np.stack([t.predict(x) for t in self._residual_model.estimators_], axis=0)
        residual_u = np.maximum(per_tree.std(axis=0), 0.0)
        return neighbor_u + residual_u

    def predict_suitability(self, rows: Sequence[Mapping[str, Any]], *, lam: float = 0.5) -> np.ndarray:
        mu = self.predict_mean(rows)
        u = self.predict_uncertainty(rows)
        return mu - float(lam) * u
