"""标准化 Problem 接口定义。

提供 Problem Protocol 和相关数据结构，用于 AlphaEvolve RL 系统中的问题定义。
用户只需定义 evaluate(candidate_code) -> score 即可使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProblemMetadata:
    """问题元信息数据结构。"""

    name: str
    description: str
    category: str
    difficulty: str = "unknown"
    tags: tuple[str, ...] = field(default_factory=tuple)
    baseline_score: float = 0.0
    optimal_score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    """解决方案验证结果。"""

    passed: bool
    message: str
    score: Optional[float] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    execution_time: Optional[float] = None


@runtime_checkable
class Problem(Protocol):
    """问题接口 Protocol。

    用户只需实现 evaluate(candidate_code) -> score 方法即可。
    可选实现 validate_solution() 和 get_metadata() 方法。

    Example:
        >>> class MyProblem:
        ...     def evaluate(self, code: str) -> float:
        ...         # 执行代码并返回分数
        ...         return score
        ...
        ...     def validate_solution(self, code: str) -> ValidationResult:
        ...         # 验证解决方案正确性
        ...         return ValidationResult(passed=True, message="OK")
        ...
        ...     def get_metadata(self) -> ProblemMetadata:
        ...         return ProblemMetadata(name="my_problem", description="...")
    """

    def evaluate(self, candidate_code: str) -> float:
        """评估候选代码。

        Args:
            candidate_code: 候选代码字符串。

        Returns:
            分数（越高越好）。
        """
        ...

    def validate_solution(self, candidate_code: str) -> ValidationResult:
        """验证解决方案正确性。

        可选实现。如果不实现，默认返回 passed=True。

        Args:
            candidate_code: 候选代码字符串。

        Returns:
            ValidationResult: 验证结果。
        """
        ...

    def get_metadata(self) -> ProblemMetadata:
        """获取问题元信息。

        可选实现。如果不实现，返回默认元信息。

        Returns:
            ProblemMetadata: 问题元信息。
        """
        ...


def validate_solution(problem: Problem, candidate_code: str) -> ValidationResult:
    """验证候选代码解决方案。

    便捷函数，调用问题的 validate_solution 方法。

    Args:
        problem: Problem 实例。
        candidate_code: 候选代码。

    Returns:
        ValidationResult: 验证结果。
    """
    return problem.validate_solution(candidate_code)


def get_metadata(problem: Problem) -> ProblemMetadata:
    """获取问题元信息。

    便捷函数，调用问题的 get_metadata 方法。

    Args:
        problem: Problem 实例。

    Returns:
        ProblemMetadata: 问题元信息。
    """
    return problem.get_metadata()


def create_problem(
    evaluate_fn: Callable[[str], float],
    *,
    name: str = "anonymous",
    description: str = "",
    category: str = "general",
    difficulty: str = "unknown",
    tags: tuple[str, ...] = (),
    validate_fn: Optional[Callable[[str], ValidationResult]] = None,
) -> "SimpleProblem":
    """创建简单问题实例。

    便捷工厂函数，用于快速创建只定义评估函数的问题。

    Args:
        evaluate_fn: 评估函数，接受代码字符串，返回分数。
        name: 问题名称。
        description: 问题描述。
        category: 问题类别。
        difficulty: 难度级别。
        tags: 问题标签。
        validate_fn: 可选的验证函数。

    Returns:
        SimpleProblem 实例。
    """
    return SimpleProblem(
        evaluate_fn=evaluate_fn,
        metadata=ProblemMetadata(
            name=name,
            description=description,
            category=category,
            difficulty=difficulty,
            tags=tags,
        ),
        validate_fn=validate_fn,
    )


@dataclass(frozen=True)
class SimpleProblem:
    """简单问题实现，支持快速创建问题。"""

    evaluate_fn: Callable[[str], float]
    metadata: ProblemMetadata
    validate_fn: Optional[Callable[[str], ValidationResult]] = None

    def evaluate(self, candidate_code: str) -> float:
        """评估候选代码。"""
        return self.evaluate_fn(candidate_code)

    def validate_solution(self, candidate_code: str) -> ValidationResult:
        """验证解决方案。"""
        if self.validate_fn is not None:
            return self.validate_fn(candidate_code)
        return ValidationResult(passed=True, message="No validation implemented")

    def get_metadata(self) -> ProblemMetadata:
        """获取元信息。"""
        return self.metadata
