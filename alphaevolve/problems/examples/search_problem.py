"""搜索问题示例。

展示如何使用 Problem 接口实现搜索相关的问题，
例如二分搜索、图搜索等。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from alphaevolve.problems.problem import (
    Problem,
    ProblemMetadata,
    ValidationResult,
)


@dataclass(frozen=True)
class SearchTestCase:
    """搜索测试用例。"""

    case_id: str
    sorted_data: List[Any]
    target: Any
    expected_index: int  # -1 表示未找到


@dataclass
class SearchProblem:
    """搜索问题实现。

    评估候选搜索算法的正确性和性能。
    用户需要实现 search(sorted_data, target) 函数。

    Attributes:
        test_cases: 测试用例列表。
        baseline_time_ns: 基准执行时间（纳秒）。
    """

    test_cases: List[SearchTestCase] = field(default_factory=list)
    baseline_time_ns: int = 10_000  # 基准时间 10 微秒

    def __init__(
        self,
        num_cases: int = 20,
        max_list_size: int = 1000,
        value_range: tuple[int, int] = (0, 10000),
        baseline_time_ns: int = 10_000,
    ):
        """初始化搜索问题。

        Args:
            num_cases: 自动生成的测试用例数量。
            max_list_size: 有序列表最大长度。
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
    ) -> List[SearchTestCase]:
        """生成随机测试用例。"""
        cases = []
        random.seed(42)

        # 特殊用例
        # 空列表
        cases.append(SearchTestCase(
            case_id="empty",
            sorted_data=[],
            target=1,
            expected_index=-1,
        ))

        # 单元素 - 找到
        cases.append(SearchTestCase(
            case_id="single_found",
            sorted_data=[5],
            target=5,
            expected_index=0,
        ))

        # 单元素 - 未找到
        cases.append(SearchTestCase(
            case_id="single_not_found",
            sorted_data=[5],
            target=3,
            expected_index=-1,
        ))

        # 重复元素
        cases.append(SearchTestCase(
            case_id="duplicates",
            sorted_data=[1, 1, 2, 2, 3, 3],
            target=2,
            expected_index=2,  # 返回第一个匹配的位置
        ))

        # 随机用例
        for i in range(num_cases - 4):
            size = random.randint(1, max_list_size)
            # 生成随机有序列表
            data = sorted(random.sample(range(value_range[0], value_range[1] + 1), min(size, value_range[1] - value_range[0] + 1)))
            # 随机选择目标
            if data and random.random() < 0.7:
                target = random.choice(data)
                expected = data.index(target)
            else:
                # 生成一个不在列表中的值
                all_values = set(range(value_range[0], value_range[1] + 1))
                target = random.choice(list(all_values - set(data)))
                expected = -1

            cases.append(SearchTestCase(
                case_id=f"random_{i}",
                sorted_data=data,
                target=target,
                expected_index=expected,
            ))

        return cases

    def evaluate(self, candidate_code: str) -> float:
        """评估候选搜索算法。

        Args:
            candidate_code: 候选代码字符串。

        Returns:
            分数（0.0 到 1.0+）。
        """
        namespace = {}
        try:
            exec(candidate_code, namespace)
            search_fn = namespace.get("search")
            if search_fn is None:
                return 0.0
        except Exception:
            return 0.0

        passed = 0
        total_time_ns = 0

        for case in self.test_cases:
            try:
                start = time.perf_counter_ns()
                result = search_fn(case.sorted_data, case.target)
                elapsed = time.perf_counter_ns() - start
                total_time_ns += elapsed

                if result == case.expected_index:
                    passed += 1
            except Exception:
                pass

        if passed == 0:
            return 0.0

        pass_rate = passed / len(self.test_cases)

        speedup = 1.0
        if total_time_ns > 0 and self.baseline_time_ns > 0:
            avg_time_ns = total_time_ns / len(self.test_cases)
            speedup = self.baseline_time_ns / max(avg_time_ns, 1)

        normalized_speedup = min(max(speedup, 0.0), 2.0) / 2.0
        score = pass_rate * (0.5 + 0.5 * normalized_speedup)

        return score

    def validate_solution(self, candidate_code: str) -> ValidationResult:
        """验证搜索解决方案的正确性。"""
        namespace = {}
        try:
            exec(candidate_code, namespace)
            search_fn = namespace.get("search")
            if search_fn is None:
                return ValidationResult(
                    passed=False,
                    message="Function 'search' not found in code",
                )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Code compilation failed: {str(e)}",
            )

        failed_cases = []
        for case in self.test_cases:
            try:
                result = search_fn(case.sorted_data, case.target)
                if result != case.expected_index:
                    failed_cases.append(
                        f"{case.case_id}(got {result}, expected {case.expected_index})"
                    )
            except Exception as e:
                failed_cases.append(f"{case.case_id}(error: {str(e)})")

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
            name="search_problem",
            description="Binary search and search algorithm optimization",
            category="algorithms",
            difficulty="easy",
            tags=("search", "binary-search", "algorithm"),
            baseline_score=0.6,
        )


# 基准搜索代码（线性搜索）
BASELINE_SEARCH_CODE = """def search(sorted_data, target):
    for i, item in enumerate(sorted_data):
        if item == target:
            return i
        if item > target:
            break
    return -1
"""

# 优化的搜索代码（二分搜索）
OPTIMIZED_SEARCH_CODE = """def search(sorted_data, target):
    left, right = 0, len(sorted_data) - 1
    while left <= right:
        mid = (left + right) // 2
        if sorted_data[mid] == target:
            return mid
        elif sorted_data[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
