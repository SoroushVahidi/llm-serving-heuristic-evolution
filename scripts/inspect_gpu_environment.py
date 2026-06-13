#!/usr/bin/env python3
"""
Inspect and save the GPU/software environment.

Saves results/gpu_calibration/environment.json and docs/gpu_environment.md.
Does NOT expose environment variables, tokens, or secrets.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception as e:
        return f"(error: {e})"


def _gpu_info() -> list[dict]:
    try:
        import torch

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "multi_processor_count": props.multi_processor_count,
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        return gpus
    except Exception:
        return []


def _cuda_version() -> str:
    try:
        import torch
        return torch.version.cuda or "unavailable"
    except Exception:
        return "unavailable"


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:
        return "unavailable"


def _transformers_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unavailable"


def _accelerate_version() -> str:
    try:
        import accelerate
        return accelerate.__version__
    except Exception:
        return "unavailable"


def _vllm_version() -> str | None:
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return None


def _driver_version() -> str:
    raw = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if raw.startswith("(error"):
        return "unavailable"
    return raw.strip().split("\n")[0]


def _disk_free_gb() -> float:
    try:
        usage = shutil.disk_usage("/")
        return round(usage.free / (1024**3), 1)
    except Exception:
        return -1.0


def _ram_stats() -> tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / (1024**3), 1), round(vm.available / (1024**3), 1)
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                lines = f.read()
            total_kb = int([l for l in lines.splitlines() if l.startswith("MemTotal")][0].split()[1])
            avail_kb = int([l for l in lines.splitlines() if l.startswith("MemAvailable")][0].split()[1])
            return round(total_kb / (1024**2), 1), round(avail_kb / (1024**2), 1)
        except Exception:
            return -1.0, -1.0


def _hf_cache_info() -> tuple[str, list[str]]:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_dir = Path(hf_home) / "hub"
    models = []
    if cache_dir.exists():
        for child in cache_dir.iterdir():
            if child.is_dir():
                models.append(child.name)
    return hf_home, sorted(models)


def main() -> None:
    out_dir = ROOT / "results" / "gpu_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    nvidia_smi_out = _run(["nvidia-smi"])
    gpus = _gpu_info()
    ram_total, ram_free = _ram_stats()
    hf_cache_dir, hf_models = _hf_cache_info()

    env = {
        "gpu": gpus,
        "nvidia_smi": nvidia_smi_out,
        "cuda_version": _cuda_version(),
        "driver_version": _driver_version(),
        "torch_version": _torch_version(),
        "python_version": platform.python_version(),
        "transformers_version": _transformers_version(),
        "accelerate_version": _accelerate_version(),
        "vllm_version": _vllm_version(),
        "disk_free_gb": _disk_free_gb(),
        "ram_total_gb": ram_total,
        "ram_free_gb": ram_free,
        "hf_cache_dir": hf_cache_dir,
        "hf_cached_models": hf_models,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    json_path = out_dir / "environment.json"
    with open(json_path, "w") as f:
        json.dump(env, f, indent=2)
    print(f"Saved: {json_path}")

    # Create docs/gpu_environment.md
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "gpu_environment.md"
    lines = [
        "# GPU Environment",
        "",
        f"Generated: {env['timestamp']}",
        "",
        "## Hardware",
        "",
    ]
    for g in gpus:
        lines.append(
            f"- GPU {g['index']}: **{g['name']}** — {g['total_memory_gb']} GB VRAM, "
            f"{g['multi_processor_count']} SMs, compute {g['major']}.{g['minor']}"
        )
    lines += [
        "",
        f"- Driver version: {env['driver_version']}",
        f"- CUDA version: {env['cuda_version']}",
        f"- RAM total: {ram_total} GB, free: {ram_free} GB",
        f"- Disk free: {env['disk_free_gb']} GB",
        "",
        "## Software",
        "",
        f"- Python: {env['python_version']}",
        f"- PyTorch: {env['torch_version']}",
        f"- Transformers: {env['transformers_version']}",
        f"- Accelerate: {env['accelerate_version']}",
        f"- vLLM: {env['vllm_version'] or 'not installed'}",
        "",
        "## HuggingFace Cache",
        "",
        f"- Cache dir: `{hf_cache_dir}`",
        f"- Cached models ({len(hf_models)}):",
    ]
    for m in hf_models:
        lines.append(f"  - {m}")
    lines += ["", "## nvidia-smi", "", "```", nvidia_smi_out, "```", ""]

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {md_path}")

    # Print summary
    print("\n=== GPU Environment Summary ===")
    for g in gpus:
        print(f"  GPU {g['index']}: {g['name']} ({g['total_memory_gb']} GB)")
    print(f"  CUDA: {env['cuda_version']}, Driver: {env['driver_version']}")
    print(f"  PyTorch: {env['torch_version']}")
    print(f"  Transformers: {env['transformers_version']}")
    print(f"  vLLM: {env['vllm_version'] or 'NOT installed'}")
    print(f"  Disk free: {env['disk_free_gb']} GB, RAM free: {ram_free} GB")


if __name__ == "__main__":
    main()
