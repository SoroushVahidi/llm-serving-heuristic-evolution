from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_gpu_external_validity_defaults_are_user_relative_or_configurable():
    module = _load_script(ROOT / "scripts" / "run_gpu_external_validity_audit.py")

    assert str(module.DEFAULT_VLLM_VENV).endswith(".venvs/vllm_baseline_pilot")
    assert module.DEFAULT_VLLM_EXECUTABLE.endswith(".venvs/vllm_baseline_pilot/bin/vllm")
    assert str(module.DEFAULT_VLLM_PYTHON).endswith(".venvs/vllm_baseline_pilot/bin/python")

    source = (ROOT / "scripts" / "run_gpu_external_validity_audit.py").read_text()
    assert "/home/" + "soroush/.venvs/vllm_baseline_pilot" not in source


def test_paid_api_launchers_discover_repo_root_from_script_location():
    for rel in ["scripts/_run_cohere_v2_live_pilot.sh", "scripts/_run_gemini_v2_live_pilot.sh"]:
        text = (ROOT / rel).read_text()
        assert "SCRIPT_DIR=" in text
        assert "REPO_ROOT=" in text
        assert "cd \"$REPO_ROOT\"" in text
        assert "/home/" + "soroush/llm-serving-heuristic-evolution" not in text
