# FGCS Reproducibility Matrix V1

| Surface | Environment | Main command or source | Inputs | Outputs | GPU | Wulver | Provenance |
|---|---|---|---|---|---|---|---|
| Core simulator/tests | Python >=3.10, `.[dev]` | `python3 -m pytest --collect-only -q` | `src/`, `tests/`, package config | test results | No | No | `pyproject.toml` |
| Manuscript figures | Core environment | `python paper/llm2026/scripts/plot_fgcs_figures.py` | Phase A/B and fresh compact tables | five `fgcs_*.pdf` figures | No | No | figure script and result manifests |
| Manuscript LaTeX | TeX installation plus repository sources | `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from `paper/llm2026/` | `main.tex`, bibliography, figures | `main.pdf` | No | No | `experiments/FGCS_MANUSCRIPT_BUILD_V1.json` |
| Fresh causal confirmation | Frozen experiment environment | Use documented frozen scripts/manifests; do not rerun casually | fresh protocol and compact result artifacts | fresh latency summaries | No for analysis | Historical execution may use Wulver | fresh hash/provenance manifests |
| Real-vLLM validation | Dedicated vLLM environment | See `experiments/real_vllm_pressure_action_validation_v1/PREREGISTRATION_V1.json` | bounded request replay and server configuration | request metrics and summaries | Yes | Historical/optional | infrastructure audit and artifact hashes |

Raw third-party data and large Wulver outputs are intentionally outside the
compact reproduction surface unless their release status is explicitly cleared.
