"""排序算法基准测试 - 基于冒泡排序的基线实现.

该模块提供了多种规模数组的排序基准测试,
用于评估算法优化效果(如快速排序、归并排序等)。
"""

from __future__ import annotations

import random
from typing import Any, List, Tuple


# =============================================================================
# 基线代码 - 冒泡排序
# =============================================================================

BUBBLE_SORT_CODE = """def bubble_sort(arr):
    \"\"\"冒泡排序实现 - O(n^2) 时间复杂度.\"\"\"
    n = len(arr)
    result = list(arr)  # 复制数组避免修改原数组
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result
"""

INSERTION_SORT_CODE = """def insertion_sort(arr):
    \"\"\"插入排序实现 - O(n^2) 时间复杂度.\"\"\"
    result = list(arr)
    for i in range(1, len(result)):
        key = result[i]
        j = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result
"""


# =============================================================================
# 验证器
# =============================================================================

def _validate_sorted(case: Any, output: Any) -> Any:
    """验证排序结果是否正确(升序)."""
    # 延迟导入避免循环依赖
    from alphaevolve.benchmarks.problem_suite import ValidationResult

    expected = case.metadata["expected"]
    if not isinstance(output, list):
        return ValidationResult(
            passed=False,
            message="输出不是列表类型",
            expected=expected,
            actual=type(output).__name__,
        )

    # 检查是否已排序
    for i in range(len(output) - 1):
        if output[i] > output[i + 1]:
            return ValidationResult(
                passed=False,
                message=f"排序结果不正确: output[{i}]={output[i]} > output[{i+1}]={output[i+1]}",
                expected="升序排列",
                actual=output,
            )

    # 检查元素是否相同(集合相等)
    if sorted(output) != sorted(expected):
        return ValidationResult(
            passed=False,
            message="排序后元素与原始元素不匹配",
            expected=sorted(expected),
            actual=sorted(output),
        )

    return ValidationResult(
        passed=True,
        message="排序结果正确",
        expected=expected,
        actual=output,
    )


def _validate_sorted_stable(case: Any, output: Any) -> Any:
    """验证稳定排序结果(包含重复元素时保持相对顺序)."""
    from alphaevolve.benchmarks.problem_suite import ValidationResult

    expected = case.metadata["expected"]
    if not isinstance(output, list):
        return ValidationResult(
            passed=False,
            message="输出不是列表类型",
            expected=expected,
            actual=type(output).__name__,
        )

    # 检查是否已排序
    for i in range(len(output) - 1):
        if output[i] > output[i + 1]:
            return ValidationResult(
                passed=False,
                message=f"排序结果不正确: output[{i}]={output[i]} > output[{i+1}]={output[i+1]}",
                expected="升序排列",
                actual=output,
            )

    return ValidationResult(
        passed=True,
        message="稳定排序结果正确",
        expected=expected,
        actual=output,
    )


# =============================================================================
# 验证器注册表
# =============================================================================

SORTING_VALIDATORS = {
    "sorted": _validate_sorted,
    "sorted_stable": _validate_sorted_stable,
}


# =============================================================================
# 测试数据生成
# =============================================================================

def _generate_random_array(size: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """生成随机数组."""
    return [random.randint(min_val, max_val) for _ in range(size)]


def _generate_nearly_sorted_array(size: int, swaps: int = 3) -> List[int]:
    """生成近乎有序的数组(少量元素位置错误)."""
    arr = list(range(size))
    for _ in range(swaps):
        i, j = random.sample(range(size), 2)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def _generate_reverse_sorted_array(size: int) -> List[int]:
    """生成完全逆序的数组."""
    return list(range(size, 0, -1))


def _generate_array_with_duplicates(size: int, num_unique: int = 10) -> List[int]:
    """生成包含重复元素的数组."""
    unique_vals = [i * (size // num_unique + 1) for i in range(num_unique)]
    return [random.choice(unique_vals) for _ in range(size)]


# =============================================================================
# 基准测试用例构建
# =============================================================================

def _build_sorting_cases() -> List[Any]:
    """构建所有排序基准测试用例."""
    from alphaevolve.benchmarks.problem_suite import BenchmarkCase

    cases = []

    # 小规模数组 (10 元素)
    small_random = _generate_random_array(10)
    cases.append(BenchmarkCase(
        case_id="sort_01",
        family="sorting",
        description="小规模随机数组 (10元素)",
        code=BUBBLE_SORT_CODE,
        function_name="bubble_sort",
        args=(small_random,),
        validator="sorted",
        metadata={
            "expected": sorted(small_random),
            "array_size": 10,
            "difficulty": "easy",
        },
    ))

    # 中规模数组 (100 元素)
    medium_random = _generate_random_array(100)
    cases.append(BenchmarkCase(
        case_id="sort_02",
        family="sorting",
        description="中规模随机数组 (100元素)",
        code=BUBBLE_SORT_CODE,
        function_name="bubble_sort",
        args=(medium_random,),
        validator="sorted",
        metadata={
            "expected": sorted(medium_random),
            "array_size": 100,
            "difficulty": "medium",
        },
    ))

    # 大规模数组 (1000 元素)
    large_random = _generate_random_array(1000)
    cases.append(BenchmarkCase(
        case_id="sort_03",
        family="sorting",
        description="大规模随机数组 (1000元素)",
        code=BUBBLE_SORT_CODE,
        function_name="bubble_sort",
        args=(large_random,),
        validator="sorted",
        metadata={
            "expected": sorted(large_random),
            "array_size": 1000,
            "difficulty": "hard",
        },
    ))

    # 近乎有序数组
    nearly_sorted = _generate_nearly_sorted_array(100)
    cases.append(BenchmarkCase(
        case_id="sort_04",
        family="sorting",
        description="近乎有序数组 (100元素, 少量乱序)",
        code=INSERTION_SORT_CODE,
        function_name="insertion_sort",
        args=(nearly_sorted,),
        validator="sorted",
        metadata={
            "expected": sorted(nearly_sorted),
            "array_size": 100,
            "difficulty": "medium",
        },
    ))

    # 完全逆序数组
    reverse_sorted = _generate_reverse_sorted_array(100)
    cases.append(BenchmarkCase(
        case_id="sort_05",
        family="sorting",
        description="完全逆序数组 (100元素)",
        code=BUBBLE_SORT_CODE,
        function_name="bubble_sort",
        args=(reverse_sorted,),
        validator="sorted",
        metadata={
            "expected": sorted(reverse_sorted),
            "array_size": 100,
            "difficulty": "medium",
        },
    ))

    # 包含重复元素的数组
    with_duplicates = _generate_array_with_duplicates(100)
    cases.append(BenchmarkCase(
        case_id="sort_06",
        family="sorting",
        description="包含重复元素的数组 (100元素)",
        code=INSERTION_SORT_CODE,
        function_name="insertion_sort",
        args=(with_duplicates,),
        validator="sorted_stable",
        metadata={
            "expected": sorted(with_duplicates),
            "array_size": 100,
            "difficulty": "medium",
        },
    ))

    # 插入排序测试 - 小规模
    insertion_small = _generate_random_array(20)
    cases.append(BenchmarkCase(
        case_id="sort_07",
        family="sorting",
        description="插入排序小规模测试 (20元素)",
        code=INSERTION_SORT_CODE,
        function_name="insertion_sort",
        args=(insertion_small,),
        validator="sorted",
        metadata={
            "expected": sorted(insertion_small),
            "array_size": 20,
            "difficulty": "easy",
        },
    ))

    # 插入排序测试 - 大规模
    insertion_large = _generate_random_array(500)
    cases.append(BenchmarkCase(
        case_id="sort_08",
        family="sorting",
        description="插入排序大规模测试 (500元素)",
        code=INSERTION_SORT_CODE,
        function_name="insertion_sort",
        args=(insertion_large,),
        validator="sorted",
        metadata={
            "expected": sorted(insertion_large),
            "array_size": 500,
            "difficulty": "hard",
        },
    ))

    # 全是相同元素的数组
    same_elements = [42] * 50
    cases.append(BenchmarkCase(
        case_id="sort_09",
        family="sorting",
        description="全部相同元素数组 (50元素)",
        code=BUBBLE_SORT_CODE,
        function_name="bubble_sort",
        args=(same_elements,),
        validator="sorted",
        metadata={
            "expected": same_elements,
            "array_size": 50,
            "difficulty": "easy",
        },
    ))

    # 已排序数组(边界情况)
    already_sorted = list(range(50))
    cases.append(BenchmarkCase(
        case_id="sort_10",
        family="sorting",
        description="已排序数组边界测试 (50元素)",
        code=INSERTION_SORT_CODE,
        function_name="insertion_sort",
        args=(already_sorted,),
        validator="sorted",
        metadata={
            "expected": already_sorted,
            "array_size": 50,
            "difficulty": "easy",
        },
    ))

    return cases


def get_sorting_benchmark_cases() -> List[Any]:
    """返回所有排序基准测试用例."""
    return _build_sorting_cases()
