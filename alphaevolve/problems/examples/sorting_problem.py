"""排序问题示例。

展示如何使用 Problem 接口实现经典的排序算法问题。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, List

from alphaevolve.problems.problem import (
    Problem,
    ProblemMetadata,
    ValidationResult,
)


@dataclass(frozen=True)
class SortingTestCase:
    """排序测试用例。"""

    case_id: str
    input_data: List[Any]
    expected: List[Any]


@dataclass
class SortingProblem:
    """排序问题实现。

    评估候选排序算法的正确性和性能。
    用户需要实现 sort(data) 函数。

    Attributes:
        test_cases: 测试用例列表。
        baseline_time_ns: 基准执行时间（纳秒）。
    """

    test_cases: List[SortingTestCase] = field(default_factory=list)
    baseline_time_ns: int = 100_000  # 基准时间 100 微秒

    def __init__(
        self,
        num_cases: int = 10,
        max_list_size: int = 100,
        value_range: tuple[int, int] = (-1000, 1000),
        baseline_time_ns: int = 100_000,
    ):
        """初始化排序问题。

        Args:
            num_cases: 自动生成的测试用例数量。
            max_list_size: 列表最大长度。
            value_range: 元素值范围。
            baseline_time_ns: 基准时间（纳秒）。
        """
        self.baseline_time_ns = baseline_time_ns
        self.test_cases = self._generate_test_cases(
            num_cases=num_cases,
            max_list_size=max_list_size,
            value_range=value_range,
        )

    def _generate_test_cases(
        self,
        num_cases: int,
        max_list_size: int,
        value_range: tuple[int, int],
    ) -> List[SortingTestCase]:
        """生成随机测试用例。"""
        cases = []

        # 使用固定种子确保可重复性
        random.seed(42)

        # 添加一些特殊用例
        # 空列表
        cases.append(SortingTestCase(
            case_id="empty",
            input_data=[],
            expected=[],
        ))

        # 单元素
        cases.append(SortingTestCase(
            case_id="single",
            input_data=[1],
            expected=[1],
        ))

        # 已排序
        cases.append(SortingTestCase(
            case_id="already_sorted",
            input_data=[1, 2, 3, 4, 5],
            expected=[1, 2, 3, 4, 5],
        ))

        # 逆序
        cases.append(SortingTestCase(
            case_id="reverse_sorted",
            input_data=[5, 4, 3, 2, 1],
            expected=[1, 2, 3, 4, 5],
        ))

        # 随机用例
        for i in range(num_cases - 4):
            size = random.randint(1, max_list_size)
            data = [random.randint(value_range[0], value_range[1]) for _ in range(size)]
            expected = sorted(data)
            cases.append(SortingTestCase(
                case_id=f"random_{i}",
                input_data=data,
                expected=expected,
            ))

        return cases

    def evaluate(self, candidate_code: str) -> float:
        """评估候选排序算法。

        综合考虑正确性和执行速度。

        Args:
            candidate_code: 候选代码字符串。

        Returns:
            分数（0.0 到 1.0+）。
        """
        namespace = {}
        try:
            exec(candidate_code, namespace)
            sort_fn = namespace.get("sort")
            if sort_fn is None:
                return 0.0
        except Exception:
            return 0.0

        passed = 0
        total_time_ns = 0

        for case in self.test_cases:
            try:
                start = time.perf_counter_ns()
                result = sort_fn(case.input_data.copy())
                elapsed = time.perf_counter_ns() - start
                total_time_ns += elapsed

                if result == case.expected:
                    passed += 1
            except Exception:
                # 执行失败，视为未通过
                pass

        if passed == 0:
            return 0.0

        # 计算通过率
        pass_rate = passed / len(self.test_cases)

        # 如果有执行时间，计算加速比
        speedup = 1.0
        if total_time_ns > 0 and self.baseline_time_ns > 0:
            # 归一化到每个用例的时间
            avg_time_ns = total_time_ns / len(self.test_cases)
            speedup = self.baseline_time_ns / max(avg_time_ns, 1)

        # 综合分数
        # 通过率占主导，速度作为辅助因子
        normalized_speedup = min(max(speedup, 0.0), 2.0) / 2.0
        score = pass_rate * (0.5 + 0.5 * normalized_speedup)

        return score

    def validate_solution(self, candidate_code: str) -> ValidationResult:
        """验证排序解决方案的正确性。

        Args:
            candidate_code: 候选代码字符串。

        Returns:
            ValidationResult: 验证结果。
        """
        namespace = {}
        try:
            exec(candidate_code, namespace)
            sort_fn = namespace.get("sort")
            if sort_fn is None:
                return ValidationResult(
                    passed=False,
                    message="Function 'sort' not found in code",
                )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Code compilation failed: {str(e)}",
            )

        failed_cases = []
        for case in self.test_cases:
            try:
                result = sort_fn(case.input_data.copy())
                if result != case.expected:
                    failed_cases.append(case.case_id)
            except Exception as e:
                failed_cases.append(f"{case.case_id}({str(e)})")

        if failed_cases:
            return ValidationResult(
                passed=False,
                message=f"Failed cases: {failed_cases}",
                expected=str(len(self.test_cases) - len(failed_cases)),
                actual=str(len(failed_cases)),
            )

        return ValidationResult(
            passed=True,
            message=f"All {len(self.test_cases)} cases passed",
        )

    def get_metadata(self) -> ProblemMetadata:
        """获取问题元信息。"""
        return ProblemMetadata(
            name="sorting_problem",
            description="Classic sorting algorithm optimization",
            category="algorithms",
            difficulty="medium",
            tags=("sorting", "algorithm", "optimization"),
            baseline_score=0.5,
        )


# 基准排序代码（冒泡排序）
BASELINE_SORT_CODE = """def sort(data):
    result = data.copy()
    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result
"""

# 优化的排序代码（快速排序）
OPTIMIZED_SORT_CODE = """def sort(data):
    if len(data) <= 1:
        return data.copy()
    pivot = data[0]
    less = [x for x in data[1:] if x <= pivot]
    greater = [x for x in data[1:] if x > pivot]
    return sort(less) + [pivot] + sort(greater)
"""
