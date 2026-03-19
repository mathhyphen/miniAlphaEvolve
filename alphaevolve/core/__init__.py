"""Core components of AlphaEvolve."""

from alphaevolve.core.data_structures import (
    Individual,
    Population,
    EvolutionConfig,
    MutationResult,
    EvaluatorResult,
)
from alphaevolve.core.mutation import MutationEngine, MutationEngineConfig
from alphaevolve.core.evaluator import (
    BaseEvaluator,
    UnitTestEvaluator,
    PerformanceEvaluator,
    CorrectnessEvaluator,
    CompositeEvaluator,
    SandboxExecutor,
)
from alphaevolve.core.evolution import Evolution

__all__ = [
    "Individual",
    "Population",
    "EvolutionConfig",
    "MutationResult",
    "EvaluatorResult",
    "MutationEngine",
    "MutationEngineConfig",
    "BaseEvaluator",
    "UnitTestEvaluator",
    "PerformanceEvaluator",
    "CorrectnessEvaluator",
    "CompositeEvaluator",
    "SandboxExecutor",
    "Evolution",
]
