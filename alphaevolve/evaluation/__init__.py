"""Evaluation module for AlphaEvolve - Multi-stage evaluation pipeline."""

from alphaevolve.evaluation.pipeline import EvaluationPipeline, StageResult
from alphaevolve.evaluation.stages import (
    EvaluationStage,
    SyntaxCheckStage,
    BasicTestStage,
    EdgeCaseStage,
    PerformanceStage,
)
from alphaevolve.evaluation.test_generation import (
    TestCaseGenerator,
    TestCase,
    Difficulty,
)

__all__ = [
    "EvaluationPipeline",
    "StageResult",
    "EvaluationStage",
    "SyntaxCheckStage",
    "BasicTestStage",
    "EdgeCaseStage",
    "PerformanceStage",
    "TestCaseGenerator",
    "TestCase",
    "Difficulty",
]
