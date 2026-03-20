"""Evaluator adapters for the AlphaEvolve framework."""

from __future__ import annotations

from typing import Callable


class FunctionEvaluator:
    """Wrapper that converts a function to an Evaluator protocol."""

    def __init__(self, evaluate_fn: Callable[[str], float]):
        """Initialize with an evaluation function.

        Args:
            evaluate_fn: A function that takes a program string and returns a score.
        """
        self._evaluate_fn = evaluate_fn

    def evaluate(self, program: str) -> float:
        """Evaluate a program.

        Args:
            program: Program code as string.

        Returns:
            Score (higher is better).
        """
        return self._evaluate_fn(program)


# Factory function
def create_evaluator(evaluate_fn: Callable[[str], float]) -> FunctionEvaluator:
    """Create an Evaluator from a function.

    Args:
        evaluate_fn: Function that takes program string, returns score.

    Returns:
        FunctionEvaluator instance.
    """
    return FunctionEvaluator(evaluate_fn)


__all__ = [
    "FunctionEvaluator",
    "create_evaluator",
]
