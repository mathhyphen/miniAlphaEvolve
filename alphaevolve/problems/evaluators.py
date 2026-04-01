"""问题评估器实现。

提供 FunctionEvaluator 和 BenchmarkEvaluator 两种评估器实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from alphaevolve.problems.problem import Problem, ProblemMetadata, ValidationResult


class FunctionEvaluator:
    """基于 Python 函数的评估器。

    将任意评估函数包装为标准 Problem 接口。
    适用于用户已有现成评估逻辑的场景。
    """

    def __init__(
        self,
        evaluate_fn: Callable[[str], float],
        *,
        validate_fn: Optional[Callable[[str], ValidationResult]] = None,
        metadata: Optional[ProblemMetadata] = None,
    ):
        """初始化评估器。

        Args:
            evaluate_fn: 评估函数，接受代码字符串，返回分数。
            validate_fn: 可选的验证函数。
            metadata: 可选的元信息。
        """
        self._evaluate_fn = evaluate_fn
        self._validate_fn = validate_fn
        self._metadata = metadata or ProblemMetadata(
            name="function_evaluator",
            description="Function-based evaluator",
            category="general",
        )

    def evaluate(self, candidate_code: str) -> float:
        """评估候选代码。

        Args:
            candidate_code: 候选代码字符串。

        Returns:
            分数（越高越好）。
        """
        return self._evaluate_fn(candidate_code)

    def validate_solution(self, candidate_code: str) -> ValidationResult:
        """验证解决方案。"""
        if self._validate_fn is not None:
            return self._validate_fn(candidate_code)
        return ValidationResult(passed=True, message="No validation implemented")

    def get_metadata(self) -> ProblemMetadata:
        """获取元信息。"""
        return self._metadata


@dataclass(frozen=True)
class BenchmarkCase:
    """单个基准测试用例。"""

    case_id: str
    description: str
    args: tuple[Any, ...]
    expected: Any
    validator: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """单个基准测试用例的执行结果。"""

    case_id: str
    success: bool
    passed: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    validation_message: str = ""


class BenchmarkEvaluator:
    """基于已有 Benchmark 的评估器。

    使用一组预定义的测试用例来评估候选代码。
    分数基于通过的测试用例比例和执行时间。
    """

    def __init__(
        self,
        cases: Sequence[BenchmarkCase],
        *,
        function_name: str = "solve",
        baseline_time: Optional[float] = None,
        metadata: Optional[ProblemMetadata] = None,
    ):
        """初始化基准评估器。

        Args:
            cases: 基准测试用例序列。
            function_name: 被评估函数的名称。
            baseline_time: 基准执行时间（秒），用于计算加速比。
            metadata: 可选的元信息。
        """
        self._cases = list(cases)
        self._function_name = function_name
        self._baseline_time = baseline_time
        self._metadata = metadata or ProblemMetadata(
            name="benchmark_evaluator",
            description=f"Benchmark with {len(cases)} cases",
            category="benchmark",
        )

    def evaluate(self, candidate_code: str) -> float:
        """评估候选代码。

        使用所有测试用例执行候选代码，计算综合分数。

        Args:
            candidate_code: 候选代码字符串。

        Returns:
            综合分数（0.0 到 1.0+），基于通过率和加速比。
        """
        results = self._run_cases(candidate_code)

        if not results:
            return 0.0

        # 计算通过率
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / len(results)

        # 如果所有用例都失败，返回 0
        if passed_count == 0:
            return 0.0

        # 计算平均执行时间（仅考虑成功的用例）
        successful_times = [r.execution_time for r in results if r.success]
        if not successful_times:
            return 0.0

        avg_time = sum(successful_times) / len(successful_times)

        # 如果有基准时间，计算加速比
        speedup = 1.0
        if self._baseline_time is not None and avg_time > 0:
            speedup = self._baseline_time / avg_time

        # 综合分数 = 通过率 * min(speedup, 2.0) / 2 + 通过率 / 2
        # 这样通过率占主导，但速度也有一定影响
        normalized_speedup = min(speedup, 2.0) / 2.0
        score = pass_rate * (normalized_speedup + 0.5)

        return score

    def validate_solution(self, candidate_code: str) -> ValidationResult:
        """验证解决方案。

        返回所有测试用例的验证结果摘要。
        """
        results = self._run_cases(candidate_code)

        if not results:
            return ValidationResult(
                passed=False,
                message="No test cases available",
            )

        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)

        all_passed = passed_count == total_count
        message = f"Passed {passed_count}/{total_count} cases"

        # 添加失败用例的简要信息
        failed_cases = [r for r in results if not r.passed]
        if failed_cases:
            case_ids = [r.case_id for r in failed_cases]
            message += f" - Failed cases: {case_ids}"

        return ValidationResult(
            passed=all_passed,
            message=message,
            score=passed_count / total_count,
        )

    def get_metadata(self) -> ProblemMetadata:
        """获取元信息。"""
        return self._metadata

    def _run_cases(self, candidate_code: str) -> List[BenchmarkCaseResult]:
        """运行所有测试用例。"""
        import time

        results: List[BenchmarkCaseResult] = []

        # 创建命名空间
        namespace: Dict[str, Any] = {}
        try:
            exec(candidate_code, namespace)
        except Exception as e:
            # 如果代码无法编译，所有用例都失败
            for case in self._cases:
                results.append(
                    BenchmarkCaseResult(
                        case_id=case.case_id,
                        success=False,
                        passed=False,
                        output=None,
                        error=f"Compilation failed: {str(e)}",
                    )
                )
            return results

        # 获取被评估函数
        solve_fn = namespace.get(self._function_name)
        if solve_fn is None:
            for case in self._cases:
                results.append(
                    BenchmarkCaseResult(
                        case_id=case.case_id,
                        success=False,
                        passed=False,
                        output=None,
                        error=f"Function '{self._function_name}' not found",
                    )
                )
            return results

        # 运行每个测试用例
        for case in self._cases:
            try:
                start = time.perf_counter()
                output = solve_fn(*case.args)
                execution_time = time.perf_counter() - start

                # 验证输出
                passed = self._validate_output(case, output)

                results.append(
                    BenchmarkCaseResult(
                        case_id=case.case_id,
                        success=True,
                        passed=passed,
                        output=output,
                        execution_time=execution_time,
                        validation_message="OK" if passed else "Output mismatch",
                    )
                )
            except Exception as e:
                results.append(
                    BenchmarkCaseResult(
                        case_id=case.case_id,
                        success=False,
                        passed=False,
                        output=None,
                        error=str(e),
                    )
                )

        return results

    def _validate_output(self, case: BenchmarkCase, output: Any) -> bool:
        """验证单个用例的输出。"""
        if case.validator is not None:
            # 使用自定义验证器
            if case.validator == "exact":
                return output == case.expected
            elif case.validator == "approx":
                return abs(float(output) - float(case.expected)) < 1e-9
            elif case.validator == "contains":
                return case.expected in output
            else:
                # 未知验证器，默认通过
                return True

        # 默认精确匹配
        if case.expected is None:
            return True
        return output == case.expected


def create_benchmark_evaluator(
    cases: Sequence[BenchmarkCase],
    *,
    function_name: str = "solve",
    baseline_time: Optional[float] = None,
    name: str = "benchmark",
    description: str = "",
    category: str = "benchmark",
) -> BenchmarkEvaluator:
    """创建基准评估器的便捷工厂函数。

    Args:
        cases: 基准测试用例。
        function_name: 函数名称。
        baseline_time: 基准时间。
        name: 问题名称。
        description: 问题描述。
        category: 问题类别。

    Returns:
        BenchmarkEvaluator 实例。
    """
    metadata = ProblemMetadata(
        name=name,
        description=description,
        category=category,
    )
    return BenchmarkEvaluator(
        cases=cases,
        function_name=function_name,
        baseline_time=baseline_time,
        metadata=metadata,
    )
