from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from llmserveopt.core.types import GPUConfig
from llmserveopt.policies.registry import ORACLE_POLICY_NAMES, POLICY_LIBRARY_V2_NAMES
from llmserveopt.selector.advanced import validate_feature_columns
from llmserveopt.selector.dataset_v2.features import extract_selector_v2_features
from llmserveopt.selector.suitability.dataset import build_long_format_rows, group_by_state, rows_with_reward
from llmserveopt.selector.suitability.models import JointRewardModel
from llmserveopt.selector.suitability.selector import joint_select
from llmserveopt.selector.windows import make_windows
from llmserveopt.simulator.service_model import ServiceModel


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_local_e2e_smoke.py"
    spec = importlib.util.spec_from_file_location("run_local_e2e_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tiny_long_format_dataset(seed: int):
    """real trace -> causal features -> full 27-policy reward vector, using
    the same small BurstGPT slice as test_local_e2e_smoke's own smoke test.
    Small (5 windows), deterministic, CPU-only, no network/GPU."""
    import argparse

    smoke = _load_smoke_module()
    args = argparse.Namespace(
        trace_path="data/raw/burstgpt/BurstGPT_1.csv", input_format="burstgpt_csv",
        max_requests=90, window_size=15, min_partial_window=15, seed=seed, time_scale=1.0,
    )
    requests, _ = smoke.load_requests(args)
    requests = sorted(requests, key=lambda r: r.arrival_time)
    windows = make_windows(requests, trace_id="burstgpt_suitability_smoke", window_size=15, min_partial=15, keep_partial=False)
    split_labels = smoke.chronological_split_labels(len(windows), 0.5, 0.25)

    gpu_configs = [GPUConfig(0, max_active_sequences=8, max_batch_tokens=512, max_kv_tokens=4096)]
    service_model = ServiceModel(
        enable_prefill_modeling=True, decode_first=True, enable_decode_prefill_contention=True,
        step_token_budget=512, max_prefill_chunk_tokens=512,
    )

    window_rows, policy_rows = [], []
    reqs_list = list(requests)
    for window, split in zip(windows, split_labels):
        prefix = reqs_list[: window.start_request_index]
        features = extract_selector_v2_features(
            window_requests=window.requests, window_start_time=window.start_time, prefix_requests=prefix,
            gpu_configs=gpu_configs, topology_class="monolithic", step_token_budget=service_model.step_token_budget,
        )
        wr = {"window_idx": window.window_id, "split": split}
        wr.update({f"feat_{k}": v for k, v in features.items()})
        window_rows.append(wr)
        for policy in POLICY_LIBRARY_V2_NAMES:
            outcome = smoke.run_policy_library_v2_candidate_on_window(
                policy, window.requests, gpu_configs, service_model,
                workload_tag=f"suitability_smoke_w{window.window_id}", seed=seed, drain_steps=3000,
            )
            row = {"window_idx": window.window_id, "policy_name": policy}
            row.update(outcome.to_row_dict(prefix="metric"))
            policy_rows.append(row)

    return build_long_format_rows(
        window_rows, policy_rows, deployable_policies=POLICY_LIBRARY_V2_NAMES,
        source="burstgpt_suitability_smoke", trace_family="burstgpt_suitability_smoke", seed=seed,
    )


@pytest.mark.filterwarnings("ignore")
def test_tiny_state_policy_suitability_e2e_smoke():
    """real-trace requests -> causal features -> full 27-policy reward
    vector -> advanced selector -> selected policy -> ANWG. Proves the
    joint suitability infrastructure interoperates end-to-end with real
    trace data and the full deployable registry -- not a performance claim."""
    pytest.importorskip("sklearn")
    seed = 20260723
    long_rows = _tiny_long_format_dataset(seed)

    # No oracle policy appears; every state got a full 27-policy vector.
    assert set(r["policy_name"] for r in long_rows).isdisjoint(set(ORACLE_POLICY_NAMES))
    rbs_all = group_by_state(long_rows)
    for state_rows in rbs_all.values():
        assert {r["policy_name"] for r in state_rows} == set(POLICY_LIBRARY_V2_NAMES)

    # No leaky causal feature made it into state_features.
    for row in long_rows:
        validate_feature_columns(list(row["state_features"].keys()))

    usable = rows_with_reward(long_rows)
    train_rows = [r for r in usable if r["split"] == "TRAIN"]
    eval_rows = [r for r in usable if r["split"] in ("VALIDATION", "TEST")]
    assert train_rows and eval_rows

    model = JointRewardModel(
        name="smoke_hybrid", encoding="hybrid", all_policies=POLICY_LIBRARY_V2_NAMES,
        n_estimators=40, max_depth=4, random_state=seed,
    ).fit(train_rows)

    eval_rbs = group_by_state(eval_rows)
    selections = joint_select(model, eval_rbs, lam=0.5)
    assert selections, "selector must produce at least one selection"
    for state_id, policy in selections.items():
        assert policy in POLICY_LIBRARY_V2_NAMES
        assert policy not in ORACLE_POLICY_NAMES
        # ANWG is present for the selected policy at this state.
        reward = next(r["reward_anwg"] for r in eval_rbs[state_id] if r["policy_name"] == policy)
        assert reward is not None

    # Deterministic rerun (same seed throughout) gives an identical selection.
    long_rows_2 = _tiny_long_format_dataset(seed)
    usable_2 = rows_with_reward(long_rows_2)
    train_rows_2 = [r for r in usable_2 if r["split"] == "TRAIN"]
    eval_rows_2 = [r for r in usable_2 if r["split"] in ("VALIDATION", "TEST")]
    model_2 = JointRewardModel(
        name="smoke_hybrid", encoding="hybrid", all_policies=POLICY_LIBRARY_V2_NAMES,
        n_estimators=40, max_depth=4, random_state=seed,
    ).fit(train_rows_2)
    selections_2 = joint_select(model_2, group_by_state(eval_rows_2), lam=0.5)
    assert selections == selections_2
