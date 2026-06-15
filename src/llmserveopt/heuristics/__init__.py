"""
Heuristic DSL package for deterministic LLM-generated scheduling heuristics.

Design principles
-----------------
* Heuristics are represented as a restricted JSON DSL (no arbitrary Python).
* All variables are online-observable at scheduling decision time.
* actual_output_tokens is explicitly forbidden.
* No randomness at runtime.
* The LLM proposes candidates offline; the verifier checks safety; the simulator
  evaluates fitness.

The selector is the adaptive baseline — heuristics must beat it or best fixed policy.
"""
from .expressions import Expression, evaluate_expression
from .verifier import VerificationResult, verify_heuristic
from .compiler import CompiledHeuristic, compile_heuristic
from .policy import HeuristicPolicy, build_heuristic_policy

__all__ = [
    "Expression",
    "evaluate_expression",
    "VerificationResult",
    "verify_heuristic",
    "CompiledHeuristic",
    "compile_heuristic",
    "HeuristicPolicy",
    "build_heuristic_policy",
]
