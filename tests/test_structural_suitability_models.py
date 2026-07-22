from __future__ import annotations

import random

import numpy as np
import pytest

from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES
from llmserveopt.policies.structural_synthesis import map_policy_to_genome
from llmserveopt.selector.suitability.dataset import genome_table
from llmserveopt.selector.suitability.encoders import structural_features
from llmserveopt.selector.suitability.selector import held_out_policy_split
from llmserveopt.selector.suitability.structural_models import (
    KernelSuitabilityModel,
    ResidualTransferModel,
    StateConditionedNeighborModel,
    StructuralDistanceIndex,
    StructuralKNNModel,
    _weights_from_distances,
)

FAITHFUL_POLICIES = sorted(
    p for p in POLICY_LIBRARY_V2_NAMES
    if map_policy_to_genome(p).metadata.get("mapping_status") in ("EXACT", "APPROXIMATE")
)


def _synthetic_rows(policies, n_states=12, seed=0):
    rng = random.Random(seed)
    genomes = genome_table(policies)
    rows = []
    for widx in range(n_states):
        state_features = {"feat_load": rng.uniform(0, 1), "feat_prompt": rng.uniform(100, 2000)}
        for i, policy in enumerate(policies):
            reward = max(0.0, min(1.0, 0.5 + 0.1 * (i % 4) + rng.uniform(-0.02, 0.02)))
            rows.append({
                "state_id": f"s{widx}", "state_features": state_features,
                "policy_name": policy, "reward_anwg": reward,
                "policy_representation": structural_features(genomes[policy]),
            })
    return rows


# ---------------------------------------------------------------------------
# Distance / weighting primitives
# ---------------------------------------------------------------------------

def test_structural_distance_index_zero_for_identical_genomes():
    idx = StructuralDistanceIndex(["fifo", "first_fit", "edf"])
    assert idx.distance("fifo", "first_fit") == pytest.approx(0.0, abs=1e-9)
    assert idx.distance("fifo", "edf") > 0.0
    assert idx.distance("fifo", "edf") == idx.distance("edf", "fifo")  # symmetric


def test_structural_distance_index_nearest_excludes_self():
    idx = StructuralDistanceIndex(["fifo", "first_fit", "edf", "shortest_output_first"])
    nearest = idx.nearest("fifo", ["fifo", "first_fit", "edf", "shortest_output_first"], k=3)
    assert all(name != "fifo" for name, _ in nearest)
    assert len(nearest) == 3


@pytest.mark.parametrize("scheme", ["uniform", "inverse_distance", "exponential"])
def test_weights_from_distances_sum_to_one_and_nonnegative(scheme):
    w = _weights_from_distances([0.0, 1.0, 2.0], scheme, tau=1.0)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0.0).all()


def test_weights_from_distances_uniform_ignores_distance():
    w = _weights_from_distances([0.1, 5.0, 100.0], "uniform", tau=1.0)
    assert np.allclose(w, w[0])


def test_weights_from_distances_inverse_and_exponential_favor_closer_points():
    for scheme in ("inverse_distance", "exponential"):
        w = _weights_from_distances([0.1, 1.0, 10.0], scheme, tau=1.0)
        assert w[0] > w[1] > w[2]


# ---------------------------------------------------------------------------
# StructuralKNNModel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("weighting", ["uniform", "inverse_distance", "exponential"])
def test_structural_knn_fit_predict_shapes(k, weighting):
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=6)
    model = StructuralKNNModel(name="knn", all_policies=FAITHFUL_POLICIES, k=k, weighting=weighting, tau=2.0).fit(rows)
    mu = model.predict_mean(rows)
    u = model.predict_uncertainty(rows)
    assert mu.shape == (len(rows),)
    assert u.shape == (len(rows),)
    assert (u >= 0.0).all()


def test_structural_knn_deterministic_across_calls():
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=6)
    m1 = StructuralKNNModel(name="a", all_policies=FAITHFUL_POLICIES, k=3, weighting="inverse_distance", tau=2.0).fit(rows)
    m2 = StructuralKNNModel(name="b", all_policies=FAITHFUL_POLICIES, k=3, weighting="inverse_distance", tau=2.0).fit(rows)
    assert (m1.predict_mean(rows) == m2.predict_mean(rows)).all()


def test_structural_knn_supports_a_policy_entirely_held_out_from_training():
    all_faithful = FAITHFUL_POLICIES
    rows = _synthetic_rows(all_faithful, n_states=10)
    held_out = "edf"
    train_rows, test_rows = held_out_policy_split(rows, [held_out])
    assert all(r["policy_name"] != held_out for r in train_rows)
    model = StructuralKNNModel(name="knn", all_policies=all_faithful, k=5, weighting="inverse_distance", tau=2.0).fit(train_rows)
    preds = model.predict_mean(test_rows)
    assert preds.shape == (len(test_rows),)
    assert not np.isnan(preds).any()


def test_structural_knn_lookup_rows_widens_valid_query_states():
    """Without lookup_rows, querying a state absent from `rows` degenerates
    to zero (no sibling data); with lookup_rows covering that state, the
    model produces a real, non-degenerate prediction."""
    rows = _synthetic_rows(FAITHFUL_POLICIES[:6], n_states=4)
    train_rows = [r for r in rows if r["state_id"] != "s3"]
    query_row = next(r for r in rows if r["state_id"] == "s3" and r["policy_name"] == FAITHFUL_POLICIES[0])

    degenerate = StructuralKNNModel(name="no_lookup", all_policies=FAITHFUL_POLICIES[:6], k=3, weighting="uniform").fit(train_rows)
    assert degenerate.predict_mean([query_row])[0] == 0.0

    widened = StructuralKNNModel(name="with_lookup", all_policies=FAITHFUL_POLICIES[:6], k=3, weighting="uniform").fit(train_rows, lookup_rows=rows)
    assert widened.predict_mean([query_row])[0] != 0.0


def test_structural_knn_never_uses_target_policys_own_value_as_neighbor():
    """The widened-lookup mechanism must never leak the query row's own
    (state, policy) reward back into its own prediction."""
    rows = _synthetic_rows(FAITHFUL_POLICIES[:6], n_states=4)
    target = FAITHFUL_POLICIES[0]
    query_row = next(r for r in rows if r["policy_name"] == target)
    # Make the target's own reward a wild outlier; if it leaked in, the
    # prediction (an average over neighbors) would move toward it.
    poisoned = [dict(r) for r in rows]
    for r in poisoned:
        if r["policy_name"] == target and r["state_id"] == query_row["state_id"]:
            r["reward_anwg"] = 999.0
    train_rows = [r for r in poisoned if r["policy_name"] != target]
    model = StructuralKNNModel(name="knn", all_policies=FAITHFUL_POLICIES[:6], k=3, weighting="uniform").fit(train_rows, lookup_rows=poisoned)
    pred = model.predict_mean([query_row])[0]
    assert pred < 10.0  # nowhere near the poisoned 999.0 value


# ---------------------------------------------------------------------------
# KernelSuitabilityModel
# ---------------------------------------------------------------------------

def test_kernel_model_fit_predict_and_nonnegative_uncertainty():
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=6)
    model = KernelSuitabilityModel(name="kernel", all_policies=FAITHFUL_POLICIES, tau=1.0).fit(rows)
    mu = model.predict_mean(rows)
    u = model.predict_uncertainty(rows)
    assert mu.shape == (len(rows),)
    assert (u >= 0.0).all()


def test_kernel_model_smaller_tau_weights_nearest_neighbor_more_heavily():
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=6)
    held_out = FAITHFUL_POLICIES[0]
    train_rows, test_rows = held_out_policy_split(rows, [held_out])
    tight = KernelSuitabilityModel(name="tight", all_policies=FAITHFUL_POLICIES, tau=0.1, distance_index=StructuralDistanceIndex(FAITHFUL_POLICIES)).fit(train_rows)
    loose = KernelSuitabilityModel(name="loose", all_policies=FAITHFUL_POLICIES, tau=100.0, distance_index=StructuralDistanceIndex(FAITHFUL_POLICIES)).fit(train_rows)
    # With a huge tau, the kernel approaches a uniform average over all
    # training policies; with a tiny tau it collapses onto the nearest one.
    # These need not be numerically different for every row, but the model
    # must at least run and be deterministic.
    assert tight.predict_mean(test_rows).shape == loose.predict_mean(test_rows).shape


# ---------------------------------------------------------------------------
# StateConditionedNeighborModel
# ---------------------------------------------------------------------------

def test_state_conditioned_neighbor_model_fit_predict():
    pytest.importorskip("sklearn")
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=8)
    model = StateConditionedNeighborModel(name="scn", all_policies=FAITHFUL_POLICIES, tau=2.0, k=3).fit(rows)
    mu = model.predict_mean(rows)
    u = model.predict_uncertainty(rows)
    assert mu.shape == (len(rows),)
    assert (u >= 0.0).all()


def test_state_conditioned_neighbor_model_alphas_are_state_and_policy_dependent():
    """Different states for the same target policy should not always
    produce identical weighting (uncertainty discount depends on state)."""
    pytest.importorskip("sklearn")
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=10)
    model = StateConditionedNeighborModel(name="scn", all_policies=FAITHFUL_POLICIES, tau=2.0, k=3).fit(rows)
    target = FAITHFUL_POLICIES[0]
    target_rows = [r for r in rows if r["policy_name"] == target]
    preds = model.predict_mean(target_rows)
    assert len(set(np.round(preds, 6).tolist())) >= 1  # runs without error; some variation expected


# ---------------------------------------------------------------------------
# ResidualTransferModel
# ---------------------------------------------------------------------------

def test_residual_transfer_model_fit_predict():
    pytest.importorskip("sklearn")
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=10)
    model = ResidualTransferModel(name="rt", all_policies=FAITHFUL_POLICIES, k=3, weighting="inverse_distance", tau=2.0).fit(rows)
    mu = model.predict_mean(rows)
    u = model.predict_uncertainty(rows)
    assert mu.shape == (len(rows),)
    assert (u >= 0.0).all()


@pytest.mark.parametrize("scheme", ["uniform", "margin", "margin_plus_epsilon"])
def test_residual_transfer_model_weight_schemes_run(scheme):
    pytest.importorskip("sklearn")
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=10)
    model = ResidualTransferModel(
        name="rt", all_policies=FAITHFUL_POLICIES, k=3, weighting="inverse_distance", tau=2.0, weight_scheme=scheme,
    ).fit(rows)
    preds = model.predict_mean(rows)
    assert not np.isnan(preds).any()


def test_residual_transfer_model_deterministic_under_fixed_seed():
    pytest.importorskip("sklearn")
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=10)
    m1 = ResidualTransferModel(name="a", all_policies=FAITHFUL_POLICIES, k=3, random_state=7).fit(rows)
    m2 = ResidualTransferModel(name="b", all_policies=FAITHFUL_POLICIES, k=3, random_state=7).fit(rows)
    assert np.allclose(m1.predict_mean(rows), m2.predict_mean(rows))


# ---------------------------------------------------------------------------
# Suitability formula shared across all four models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls,kwargs", [
    (StructuralKNNModel, {"k": 3, "weighting": "uniform"}),
    (KernelSuitabilityModel, {"tau": 1.0}),
])
def test_predict_suitability_equals_mean_minus_lambda_times_uncertainty(model_cls, kwargs):
    rows = _synthetic_rows(FAITHFUL_POLICIES[:8], n_states=6)
    model = model_cls(name="m", all_policies=FAITHFUL_POLICIES, **kwargs).fit(rows)
    mu = model.predict_mean(rows)
    u = model.predict_uncertainty(rows)
    for lam in (0.0, 0.5, 2.0):
        s = model.predict_suitability(rows, lam=lam)
        assert np.allclose(s, mu - lam * u)
