"""Joint discrete selector, margin-aware evaluation, the Delta_SCORPIO_WSP
pairwise-advantage diagnostic, and held-out-policy/family generalization
pilots.

pi_select(x) = argmax_i S(x, pi_i), S = mu - lambda * u.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MARGIN_THRESHOLDS: Tuple[float, ...] = (0.0, 0.001, 0.005, 0.010)


def true_reward_row(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    return {r["policy_name"]: float(r["reward_anwg"]) for r in rows if r.get("reward_anwg") is not None}


def top2_margin(reward_by_policy: Mapping[str, float]) -> float:
    values = sorted(reward_by_policy.values(), reverse=True)
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[1])


def oracle_best(reward_by_policy: Mapping[str, float]) -> Tuple[str, float]:
    policy = max(reward_by_policy, key=reward_by_policy.get)
    return policy, reward_by_policy[policy]


def joint_select(
    model,
    rows_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    lam: float = 0.5,
) -> Dict[str, str]:
    """pi_select(x) = argmax_i S(x, pi_i) for every state."""
    selections: Dict[str, str] = {}
    for state_id, rows in rows_by_state.items():
        usable = [r for r in rows if r.get("reward_anwg") is not None]
        if not usable:
            continue
        suitability = model.predict_suitability(usable, lam=lam)
        best_idx = int(np.argmax(suitability))
        selections[state_id] = usable[best_idx]["policy_name"]
    return selections


def evaluate_selection(
    rows_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    selections: Mapping[str, str],
    *,
    best_fixed_policy: str,
) -> Dict[str, Any]:
    """ANWG, regret-to-oracle, oracle-gap-closure, policy-match accuracy,
    all reported overall and stratified by top-2 true-reward margin."""
    per_state_records = []
    for state_id, selected_policy in selections.items():
        rewards = true_reward_row(rows_by_state[state_id])
        if selected_policy not in rewards:
            continue
        oracle_policy, oracle_reward = oracle_best(rewards)
        fixed_reward = rewards.get(best_fixed_policy, float("nan"))
        margin = top2_margin(rewards)
        per_state_records.append({
            "state_id": state_id,
            "selected_policy": selected_policy,
            "selected_reward": rewards[selected_policy],
            "oracle_policy": oracle_policy,
            "oracle_reward": oracle_reward,
            "fixed_reward": fixed_reward,
            "margin": margin,
            "is_match": selected_policy == oracle_policy,
        })

    def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {"n_states": 0}
        selected = np.asarray([r["selected_reward"] for r in records])
        oracle = np.asarray([r["oracle_reward"] for r in records])
        fixed = np.asarray([r["fixed_reward"] for r in records])
        regret = oracle - selected
        denom = float(np.nanmean(oracle) - np.nanmean(fixed))
        gap_closed = (
            float((np.mean(selected) - np.nanmean(fixed)) / denom) if abs(denom) > 1e-12 else None
        )
        return {
            "n_states": len(records),
            "mean_anwg": float(np.mean(selected)),
            "mean_oracle_anwg": float(np.mean(oracle)),
            "mean_fixed_anwg": float(np.nanmean(fixed)),
            "mean_regret_to_oracle": float(np.mean(regret)),
            "p95_regret_to_oracle": float(np.percentile(regret, 95)),
            "gap_closed_fraction": gap_closed,
            "policy_match_accuracy": float(np.mean([r["is_match"] for r in records])),
        }

    result: Dict[str, Any] = {"overall": _summarize(per_state_records)}
    for threshold in MARGIN_THRESHOLDS:
        meaningful = [r for r in per_state_records if r["margin"] > threshold]
        result[f"margin_gt_{threshold}"] = _summarize(meaningful)
    return result


def margin_weighted_regret(rows_by_state: Mapping[str, Sequence[Mapping[str, Any]]], selections: Mapping[str, str]) -> float:
    """Mean regret weighted by top-2 margin -- near-tie mistakes count less."""
    weighted_sum = 0.0
    weight_total = 0.0
    for state_id, selected_policy in selections.items():
        rewards = true_reward_row(rows_by_state[state_id])
        if selected_policy not in rewards:
            continue
        _, oracle_reward = oracle_best(rewards)
        margin = top2_margin(rewards)
        regret = oracle_reward - rewards[selected_policy]
        weighted_sum += margin * regret
        weight_total += margin
    return weighted_sum / weight_total if weight_total > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Delta_SCORPIO_WSP(x) = R_SCORPIO(x) - R_WSP(x)
# ---------------------------------------------------------------------------

def build_delta_rows(
    rows_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    policy_a: str = "scorpio_style_slo_guard",
    policy_b: str = "weighted_shortest_processing",
) -> List[Dict[str, Any]]:
    out = []
    for state_id, rows in rows_by_state.items():
        rewards = true_reward_row(rows)
        if policy_a not in rewards or policy_b not in rewards:
            continue
        state_features = next(r["state_features"] for r in rows if r["policy_name"] == policy_a)
        out.append({
            "state_id": state_id,
            "state_features": state_features,
            "delta": rewards[policy_a] - rewards[policy_b],
            "margin": top2_margin(rewards),
        })
    out.sort(key=lambda r: r["state_id"])
    return out


class DeltaModel:
    """State-only regressor for a fixed pairwise advantage Delta_A_B(x)."""

    def __init__(self, *, n_estimators: int = 200, max_depth: Optional[int] = 6, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._state_cols: List[str] = []
        self.model = None

    def fit(self, delta_rows: Sequence[Mapping[str, Any]]) -> "DeltaModel":
        from sklearn.ensemble import RandomForestRegressor

        self._state_cols = sorted({k for row in delta_rows for k in row["state_features"].keys()})
        x = self._matrix(delta_rows)
        y = np.asarray([float(r["delta"]) for r in delta_rows])
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            random_state=self.random_state, n_jobs=1,
        )
        self.model.fit(x, y)
        return self

    def _matrix(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        out = np.zeros((len(rows), len(self._state_cols)), dtype=float)
        col_index = {c: i for i, c in enumerate(self._state_cols)}
        for r, row in enumerate(rows):
            for k, v in row["state_features"].items():
                idx = col_index.get(k)
                if idx is not None:
                    out[r, idx] = float(v)
        return out

    def predict(self, delta_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DeltaModel must be fit() before predict()")
        return self.model.predict(self._matrix(delta_rows))


def evaluate_delta_model(
    model: DeltaModel,
    delta_rows: Sequence[Mapping[str, Any]],
    *,
    meaningful_margin: float = 0.005,
) -> Dict[str, Any]:
    preds = model.predict(delta_rows)
    actual = np.asarray([float(r["delta"]) for r in delta_rows])
    margins = np.asarray([float(r["margin"]) for r in delta_rows])
    err = preds - actual
    sign_match = np.sign(preds) == np.sign(actual)
    meaningful_mask = margins > meaningful_margin
    near_zero_mask = np.abs(actual) <= meaningful_margin

    result = {
        "n_states": len(delta_rows),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "sign_accuracy": float(np.mean(sign_match)),
        "sign_accuracy_meaningful_margin": (
            float(np.mean(sign_match[meaningful_mask])) if meaningful_mask.any() else None
        ),
        "n_meaningful_margin_states": int(meaningful_mask.sum()),
        "calibration_near_zero_mean_abs_pred": (
            float(np.mean(np.abs(preds[near_zero_mask]))) if near_zero_mask.any() else None
        ),
        "n_near_zero_states": int(near_zero_mask.sum()),
    }
    return result


def delta_consistency_with_joint_model(
    delta_model: DeltaModel,
    joint_model,
    delta_rows: Sequence[Mapping[str, Any]],
    rows_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    policy_a: str,
    policy_b: str,
) -> Dict[str, Any]:
    """Compare the direct Delta_A_B(x) model against Rhat_A(x) - Rhat_B(x)
    implied by the joint reward model."""
    direct_preds = delta_model.predict(delta_rows)
    implied_preds = []
    for row in delta_rows:
        state_rows = rows_by_state[row["state_id"]]
        row_a = next(r for r in state_rows if r["policy_name"] == policy_a)
        row_b = next(r for r in state_rows if r["policy_name"] == policy_b)
        mu_a = float(joint_model.predict_mean([row_a])[0])
        mu_b = float(joint_model.predict_mean([row_b])[0])
        implied_preds.append(mu_a - mu_b)
    implied_preds = np.asarray(implied_preds)
    diff = direct_preds - implied_preds
    corr = float(np.corrcoef(direct_preds, implied_preds)[0, 1]) if len(direct_preds) > 1 else None
    return {
        "mean_abs_disagreement": float(np.mean(np.abs(diff))),
        "rmse_disagreement": float(np.sqrt(np.mean(diff ** 2))),
        "pearson_correlation": corr,
    }


# ---------------------------------------------------------------------------
# Held-out-policy / held-out-family generalization pilots
# ---------------------------------------------------------------------------

def held_out_policy_split(
    rows: Sequence[Mapping[str, Any]],
    held_out_policies: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Train rows exclude every row for `held_out_policies`; test rows are
    exactly those excluded rows. No state's rows are partially split --
    only entire policies are held out, so split integrity (a state's TRAIN/
    VALIDATION/ID_TEST/OOD_TEST split membership) is untouched."""
    held = set(held_out_policies)
    train = [dict(r) for r in rows if r["policy_name"] not in held]
    test = [dict(r) for r in rows if r["policy_name"] in held]
    return train, test


def nearest_structural_policy_baseline(
    train_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    held_out_policy: str,
) -> np.ndarray:
    """For each held-out test row, predict the TRUE reward achieved by the
    structurally-nearest training policy at that exact same state (state
    features/reward are available from the same simulator run -- this is a
    fair, state-conditioned "copy the closest known behavior" baseline)."""
    held_out_repr = next(r["policy_representation"] for r in test_rows if r["policy_name"] == held_out_policy)
    struct_keys = sorted(k for k in held_out_repr.keys() if k != "mapping_status_summary")
    held_out_vec = np.asarray([float(held_out_repr[k]) for k in struct_keys])

    train_policy_reprs: Dict[str, np.ndarray] = {}
    for row in train_rows:
        name = row["policy_name"]
        if name in train_policy_reprs:
            continue
        train_policy_reprs[name] = np.asarray([float(row["policy_representation"].get(k, 0.0)) for k in struct_keys])

    nearest_policy = min(
        train_policy_reprs, key=lambda name: float(np.linalg.norm(train_policy_reprs[name] - held_out_vec))
    )
    rewards_by_state = {r["state_id"]: r["reward_anwg"] for r in train_rows if r["policy_name"] == nearest_policy}
    out = np.asarray([
        rewards_by_state.get(row["state_id"], np.nan) for row in test_rows
    ], dtype=float)
    return out, nearest_policy


def held_out_policy_pilot(
    rows: Sequence[Mapping[str, Any]],
    held_out_policy: str,
    *,
    all_policies: Sequence[str],
    encoding: str = "hybrid",
) -> Dict[str, Any]:
    """Leave one policy out of training; predict its reward from state +
    structure alone; compare against global mean, nearest-structural-policy,
    and an identity-encoding model (which cannot see this policy's identity
    at all during training and is expected to fail to generalize)."""
    from .models import JointRewardModel

    train_rows, test_rows = held_out_policy_split(rows, [held_out_policy])
    if not train_rows or not test_rows:
        return {"held_out_policy": held_out_policy, "status": "insufficient_data"}

    struct_model = JointRewardModel(name=f"structural_holdout_{held_out_policy}", encoding="structural", all_policies=all_policies).fit(train_rows)
    hybrid_model = JointRewardModel(name=f"hybrid_holdout_{held_out_policy}", encoding=encoding, all_policies=all_policies).fit(train_rows)
    identity_model = JointRewardModel(name=f"identity_holdout_{held_out_policy}", encoding="identity", all_policies=all_policies).fit(train_rows)

    actual = np.asarray([float(r["reward_anwg"]) for r in test_rows])
    global_mean = np.full(len(test_rows), float(np.mean([r["reward_anwg"] for r in train_rows])))
    nearest_preds, nearest_policy = nearest_structural_policy_baseline(train_rows, test_rows, held_out_policy)

    def _mae(preds: np.ndarray) -> Optional[float]:
        mask = ~np.isnan(preds)
        if not mask.any():
            return None
        return float(np.mean(np.abs(preds[mask] - actual[mask])))

    return {
        "held_out_policy": held_out_policy,
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "nearest_structural_policy": nearest_policy,
        "mae": {
            "structural_model": _mae(struct_model.predict_mean(test_rows)),
            "hybrid_model": _mae(hybrid_model.predict_mean(test_rows)),
            "identity_model_cannot_see_unseen_id": _mae(identity_model.predict_mean(test_rows)),
            "global_mean_baseline": _mae(global_mean),
            "nearest_structural_policy_baseline": _mae(nearest_preds),
        },
    }


def held_out_family_pilot(
    rows: Sequence[Mapping[str, Any]],
    family_policies: Sequence[str],
    *,
    all_policies: Sequence[str],
    family_name: str,
) -> Dict[str, Any]:
    """Same as held_out_policy_pilot but for an entire documented policy
    family (docs/current/policy_component_matrix.json component grouping),
    held out together -- a harder, more honest generalization test than
    holding out one policy at a time within a family of similar policies."""
    from .models import JointRewardModel

    train_rows, test_rows = held_out_policy_split(rows, family_policies)
    if not train_rows or not test_rows:
        return {"family": family_name, "status": "insufficient_data"}

    struct_model = JointRewardModel(name=f"structural_holdout_family_{family_name}", encoding="structural", all_policies=all_policies).fit(train_rows)
    hybrid_model = JointRewardModel(name=f"hybrid_holdout_family_{family_name}", encoding="hybrid", all_policies=all_policies).fit(train_rows)

    actual = np.asarray([float(r["reward_anwg"]) for r in test_rows])
    global_mean = np.full(len(test_rows), float(np.mean([r["reward_anwg"] for r in train_rows])))

    def _mae(preds: np.ndarray, actual_subset: np.ndarray) -> float:
        return float(np.mean(np.abs(preds - actual_subset)))

    per_policy = {}
    for policy in family_policies:
        policy_test = [r for r in test_rows if r["policy_name"] == policy]
        if not policy_test:
            continue
        policy_actual = np.asarray([float(r["reward_anwg"]) for r in policy_test])
        per_policy[policy] = {
            "n_test_rows": len(policy_test),
            "structural_mae": _mae(struct_model.predict_mean(policy_test), policy_actual),
            "hybrid_mae": _mae(hybrid_model.predict_mean(policy_test), policy_actual),
        }

    return {
        "family": family_name,
        "family_policies": list(family_policies),
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "overall_mae": {
            "structural_model": _mae(struct_model.predict_mean(test_rows), actual),
            "hybrid_model": _mae(hybrid_model.predict_mean(test_rows), actual),
            "global_mean_baseline": _mae(global_mean, actual),
        },
        "per_policy": per_policy,
    }


def load_policy_families(component: str, matrix_path: Optional[Path] = None) -> List[str]:
    """Load a documented policy family from docs/current/policy_component_matrix.json
    (the Policy Composition Readiness audit's machine-readable component
    matrix) -- never an invented grouping."""
    if matrix_path is None:
        matrix_path = Path(__file__).resolve().parents[4] / "docs" / "current" / "policy_component_matrix.json"
    data = json.loads(matrix_path.read_text())
    return sorted(p["name"] for p in data["policies"] if component in p.get("components", []))


# ---------------------------------------------------------------------------
# Structural-distance diagnostics
# ---------------------------------------------------------------------------

def pairwise_structural_distances(policies: Sequence[str]) -> Dict[Tuple[str, str], float]:
    """Z-score-normalized Euclidean distance between every pair of policies'
    structural feature vectors. Normalization is fit only over `policies`
    (typically the mapped, non-UNSUPPORTED subset) so the many constant-zero
    columns shared with unmapped placeholders don't distort scale."""
    from .dataset import genome_table
    from .encoders import structural_features

    genomes = genome_table(policies)
    feats = {name: structural_features(genomes[name]) for name in policies}
    cols = sorted(feats[policies[0]].keys())
    matrix = np.stack([[feats[name][c] for c in cols] for name in policies])
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    normalized = (matrix - matrix.mean(axis=0)) / std

    distances: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(policies):
        for j in range(i + 1, len(policies)):
            b = policies[j]
            distances[(a, b)] = float(np.linalg.norm(normalized[i] - normalized[j]))
    return distances


def structural_distance_vs_performance_disagreement(
    rows_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    policies: Sequence[str],
) -> Dict[str, Any]:
    """Check whether structurally similar policies behave similarly and
    structurally distant ones disagree more -- reports a Pearson correlation
    between structural distance and mean-absolute-reward-disagreement across
    every policy pair with overlapping states. This is a correlational
    diagnostic only; it does not establish that structural similarity
    *causes* behavioral similarity."""
    distances = pairwise_structural_distances(list(policies))
    rewards_by_policy: Dict[str, Dict[str, float]] = {p: {} for p in policies}
    for state_id, rows in rows_by_state.items():
        for row in rows:
            if row["policy_name"] in rewards_by_policy and row.get("reward_anwg") is not None:
                rewards_by_policy[row["policy_name"]][state_id] = float(row["reward_anwg"])

    struct_dist: List[float] = []
    perf_disagreement: List[float] = []
    pair_records: List[Dict[str, Any]] = []
    for (a, b), dist in distances.items():
        common_states = set(rewards_by_policy[a]) & set(rewards_by_policy[b])
        if not common_states:
            continue
        diffs = [abs(rewards_by_policy[a][s] - rewards_by_policy[b][s]) for s in common_states]
        disagreement = float(np.mean(diffs))
        struct_dist.append(dist)
        perf_disagreement.append(disagreement)
        pair_records.append({"policy_a": a, "policy_b": b, "structural_distance": dist, "mean_abs_reward_disagreement": disagreement, "n_common_states": len(common_states)})

    if len(struct_dist) < 2:
        return {"n_pairs": len(struct_dist), "pearson_correlation": None, "pairs": pair_records}

    correlation = float(np.corrcoef(struct_dist, perf_disagreement)[0, 1])
    ranked = sorted(pair_records, key=lambda r: r["structural_distance"])
    return {
        "n_pairs": len(struct_dist),
        "pearson_correlation": correlation,
        "note": "correlational only -- does not establish structural similarity causes behavioral similarity",
        "closest_pairs": ranked[:5],
        "farthest_pairs": ranked[-5:],
    }
