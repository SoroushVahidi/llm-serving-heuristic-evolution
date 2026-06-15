"""
Offline LLM heuristic candidate generation loop.

LLMs propose candidate heuristics offline in a restricted JSON DSL.
Verified candidates are evaluated deterministically in the simulator;
no LLM is called during request scheduling.

Architecture
------------
1. prompt_templates        — system/user prompts for generation and repair
2. provider_base           — LLMResponse dataclass + LLMProvider protocol
3. providers               — CloudRift / Cohere / Mistral / Mock wrappers
4. candidate_io            — archive save/load and index management
5. repair                  — repair prompts from verifier error codes
6. diversity               — design targets + candidate deduplication
7. generation_loop         — generate → verify → repair → archive orchestration
8. evaluation              — compile + simulate each verified heuristic (single regime)
9. ranking                 — rank by priority_weighted_slo_goodput (single regime)
10. multi_regime_evaluation — evaluate across multiple train/val regimes
11. search_ranking          — rank by validation performance with overfitting detection
"""
from .provider_base import LLMResponse, LLMProvider
from .generation_loop import GenerationConfig, run_generation_loop
from .evaluation import EvaluationConfig, evaluate_candidates
from .ranking import rank_candidates
from .diversity import DESIGN_TARGETS, build_targeted_messages, deduplicate_candidates
from .multi_regime_evaluation import (
    MultiRegimeConfig, RegimeSpec, RegimeResult,
    AggregatedCandidateResult, evaluate_multi_regime, aggregate_regime_results,
    TRAIN_REGIMES, VALIDATION_REGIMES, DEFAULT_REGIMES,
)
from .search_ranking import rank_search_results, save_search_ranking_csv, build_search_summary_md

__all__ = [
    "LLMResponse", "LLMProvider",
    "GenerationConfig", "run_generation_loop",
    "EvaluationConfig", "evaluate_candidates",
    "rank_candidates",
    "DESIGN_TARGETS", "build_targeted_messages", "deduplicate_candidates",
    "MultiRegimeConfig", "RegimeSpec", "RegimeResult",
    "AggregatedCandidateResult", "evaluate_multi_regime", "aggregate_regime_results",
    "TRAIN_REGIMES", "VALIDATION_REGIMES", "DEFAULT_REGIMES",
    "rank_search_results", "save_search_ranking_csv", "build_search_summary_md",
]
