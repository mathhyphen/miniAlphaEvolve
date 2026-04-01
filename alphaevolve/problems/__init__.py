"""AlphaEvolve Problems - 标准化问题接口。

提供通用的 Problem Protocol 和评估器，用于 AlphaEvolve RL 系统中的问题定义和评估。
"""

from alphaevolve.problems.problem import (
    Problem,
    ProblemMetadata,
    ValidationResult,
    validate_solution,
    get_metadata,
)
from alphaevolve.problems.evaluators import (
    FunctionEvaluator,
    BenchmarkEvaluator,
)

__all__ = [
    "Problem",
    "ProblemMetadata",
    "ValidationResult",
    "validate_solution",
    "get_metadata",
    "FunctionEvaluator",
    "BenchmarkEvaluator",
]
