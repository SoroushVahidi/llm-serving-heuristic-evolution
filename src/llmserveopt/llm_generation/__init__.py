"""
Offline LLM heuristic candidate generation loop.

LLMs propose candidate heuristics offline in a restricted JSON DSL.
Verified candidates are evaluated deterministically in the simulator;
no LLM is called during request scheduling.

Architecture
------------
1. prompt_templates  — system/user prompts for generation and repair
2. provider_base     — LLMResponse dataclass + LLMProvider protocol
3. providers         — CloudRift / Cohere / Mistral / Mock wrappers
4. candidate_io      — archive save/load and index management
5. repair            — repair prompts from verifier error codes
6. generation_loop   — generate → verify → repair → archive orchestration
7. evaluation        — compile + simulate each verified heuristic
8. ranking           — rank by priority_weighted_slo_goodput
"""
from .provider_base import LLMResponse, LLMProvider
from .generation_loop import GenerationConfig, run_generation_loop
from .evaluation import EvaluationConfig, evaluate_candidates
from .ranking import rank_candidates

__all__ = [
    "LLMResponse",
    "LLMProvider",
    "GenerationConfig",
    "run_generation_loop",
    "EvaluationConfig",
    "evaluate_candidates",
    "rank_candidates",
]
