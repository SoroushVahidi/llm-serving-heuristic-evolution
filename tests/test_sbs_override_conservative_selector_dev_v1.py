from pathlib import Path
import importlib.util
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sbs_override_conservative_selector_dev_v1.py"
spec = importlib.util.spec_from_file_location("sbs_override_conservative_selector_dev_v1", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

FRESH_FORBIDDEN_SUBSTRING = module.FRESH_FORBIDDEN_SUBSTRING
feature_columns = module.feature_columns
reject_fresh_confirmatory_path = module.reject_fresh_confirmatory_path


def test_rejects_fresh_confirmatory_training_path():
    bad = Path("/tmp") / FRESH_FORBIDDEN_SUBSTRING / "run_v1" / "state_action_labels.csv"
    with pytest.raises(ValueError):
        reject_fresh_confirmatory_path(bad)


def test_state_action_feature_contract_rejects_target_leakage_name():
    import pandas as pd

    df = pd.DataFrame(
        {
            **{f"state__x_{i}": [0.0] for i in range(94)},
            "state__a_sbs_leak": [0.0],
            **{f"action__x_{i}": [0.0] for i in range(70)},
        }
    )
    with pytest.raises(ValueError):
        feature_columns(df)
