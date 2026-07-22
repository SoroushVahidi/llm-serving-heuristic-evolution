from __future__ import annotations

from llmserveopt.policies.registry import POLICY_LIBRARY_V2_NAMES
from llmserveopt.selector.suitability.dataset import genome_table
from llmserveopt.selector.suitability.encoders import (
    PolicyEncoder,
    identity_features,
    structural_features,
)


def _rows_for(policies, n_states=2):
    genomes = genome_table(policies)
    rows = []
    for widx in range(n_states):
        state_features = {"feat_a": float(widx), "feat_b": float(widx) * 2.0}
        for policy in policies:
            rows.append({
                "state_id": f"s{widx}", "state_features": state_features,
                "policy_name": policy, "reward_anwg": 0.5,
                "policy_representation": structural_features(genomes[policy]),
            })
    return rows


def test_structural_features_deterministic_across_calls():
    genomes = genome_table(POLICY_LIBRARY_V2_NAMES)
    for name in POLICY_LIBRARY_V2_NAMES:
        a = structural_features(genomes[name])
        b = structural_features(genome_table([name])[name])
        assert a == b


def test_structural_features_distinguish_exact_from_placeholder_policies():
    genomes = genome_table(POLICY_LIBRARY_V2_NAMES)
    wsp = structural_features(genomes["weighted_shortest_processing"])  # EXACT mapping
    fifo = structural_features(genomes["fifo"])  # UNSUPPORTED placeholder
    assert wsp != fifo
    assert wsp["struct_num_modules_exact"] >= 1.0
    assert fifo["struct_num_modules_unsupported"] >= 1.0


def test_structural_features_never_include_policy_hash():
    genomes = genome_table(POLICY_LIBRARY_V2_NAMES)
    feats = structural_features(genomes["scorpio_style_slo_guard"])
    for key in feats:
        assert "hash" not in key.lower()


def test_identity_features_one_hot_deterministic():
    policies = sorted(POLICY_LIBRARY_V2_NAMES)
    a = identity_features("fifo", policies)
    b = identity_features("fifo", list(reversed(policies)))
    assert a == b
    assert a[f"policyid_fifo"] == 1.0
    assert sum(a.values()) == 1.0
    assert len(a) == len(policies)


def test_policy_encoder_structural_excludes_state_and_identity():
    rows = _rows_for(["fifo", "edf", "weighted_shortest_processing"])
    enc = PolicyEncoder("structural", ["fifo", "edf", "weighted_shortest_processing"]).fit(rows)
    assert not any(name.startswith("feat_") for name in enc.feature_names)
    assert not any(name.startswith("policyid_") for name in enc.feature_names)
    x = enc.transform(rows)
    assert x.shape == (len(rows), len(enc.feature_names))


def test_policy_encoder_identity_excludes_structural():
    rows = _rows_for(["fifo", "edf"])
    enc = PolicyEncoder("identity", ["fifo", "edf"]).fit(rows)
    assert not any(name.startswith("struct_") for name in enc.feature_names)
    assert any(name.startswith("policyid_") for name in enc.feature_names)
    assert any(name.startswith("feat_") for name in enc.feature_names)


def test_policy_encoder_hybrid_includes_all_three():
    rows = _rows_for(["fifo", "edf"])
    enc = PolicyEncoder("hybrid", ["fifo", "edf"]).fit(rows)
    names = enc.feature_names
    assert any(n.startswith("feat_") for n in names)
    assert any(n.startswith("policyid_") for n in names)
    assert any(n.startswith("struct_") for n in names)


def test_policy_encoder_transform_deterministic():
    rows = _rows_for(["fifo", "edf", "scorpio_style_slo_guard"])
    enc = PolicyEncoder("hybrid", ["fifo", "edf", "scorpio_style_slo_guard"]).fit(rows)
    x1 = enc.transform(rows)
    x2 = enc.transform(rows)
    assert (x1 == x2).all()


def test_policy_encoder_handles_unseen_policy_row_without_crashing():
    """A held-out-policy row can be transformed even though that policy's
    identity column (if any) was never set to 1 during fit -- it just
    stays zero, which is exactly the "cannot see unseen identity" property
    being tested at the model level elsewhere."""
    all_policies = ["fifo", "edf", "weighted_shortest_processing"]
    train_rows = _rows_for(["fifo", "edf"])
    enc = PolicyEncoder("identity", all_policies).fit(train_rows)
    genomes = genome_table(all_policies)
    held_out_row = {
        "state_id": "s0", "state_features": {"feat_a": 0.0, "feat_b": 0.0},
        "policy_name": "weighted_shortest_processing", "reward_anwg": 0.5,
        "policy_representation": structural_features(genomes["weighted_shortest_processing"]),
    }
    x = enc.transform([held_out_row])
    assert x.shape[0] == 1
