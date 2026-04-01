"""二分搜索基准测试 - 基于线性搜索的基线实现.

该模块提供了有序数组查找的基准测试,
用于评估算法优化效果(如二分搜索、插值搜索等)。
"""

from __future__ import annotations

import random
from typing import Any, List, Optional, Tuple


# =============================================================================
# 基线代码 - 线性搜索
# =============================================================================

LINEAR_SEARCH_CODE = """def linear_search(arr, target):
    \"\"\"线性搜索实现 - O(n) 时间复杂度.\"\"\"
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
"""

BINARY_SEARCH_CODE = """def binary_search(arr, target):
    \"\"\"二分搜索实现 - O(log n) 时间复杂度.\"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

EXPONENTIAL_SEARCH_CODE = """def exponential_search(arr, target):
    \"\"\"指数搜索实现 - O(log n) 时间复杂度.\"\"\"
    if len(arr) == 0:
        return -1
    if arr[0] == target:
        return 0

    # 找到目标所在的上界
    i = 1
    while i < len(arr) and arr[i] < target:
        i *= 2

    # 在找到的范围内进行二分搜索
    left, right = i // 2, min(i, len(arr) - 1)
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""


# =============================================================================
# 验证器
# =============================================================================

def _validate_search_index(case: Any, output: Any) -> Any:
    """验证搜索返回的索引是否正确."""
    from alphaevolve.benchmarks.problem_suite import ValidationResult

    expected = case.metadata["expected"]
    if not isinstance(output, (int, type(None))):
        return ValidationResult(
            passed=False,
            message=f"输出类型错误: 期望 int 或 None, 得到 {type(output).__name__}",
            expected=expected,
            actual=type(output).__name__,
        )

    target = case.metadata["target"]
    arr = case.metadata["arr"]

    # 验证索引有效性
    if output != -1 and output is not None:
        if output < 0 or output >= len(arr):
            return ValidationResult(
                passed=False,
                message=f"返回的索引超出范围: {output}",
                expected=f"0 <= index < {len(arr)} 或 -1",
                actual=output,
            )
        if arr[output] != target:
            return ValidationResult(
                passed=False,
                message=f"索引 {output} 处的元素不是目标值: arr[{output}]={arr[output]} != target={target}",
                expected=target,
                actual=arr[output],
            )

    # 验证正确性
    if expected == -1:
        passed = output == -1 or output is None
    else:
        passed = output == expected

    return ValidationResult(
        passed=passed,
        message="搜索索引正确" if passed else f"搜索索引错误: 期望 {expected}, 得到 {output}",
        expected=expected,
        actual=output,
    )


def _validate_search_found(case: Any, output: Any) -> Any:
    """验证搜索是否找到目标(不验证具体索引)."""
    from alphaevolve.benchmarks.problem_suite import ValidationResult

    expected_found = case.metadata["expected_found"]

    if output == -1 or output is None:
        found = False
    else:
        found = True

    passed = found == expected_found
    return ValidationResult(
        passed=passed,
        message="搜索结果正确" if passed else f"搜索结果错误: 期望{'找到' if expected_found else '未找到'}, 得到 {'找到' if found else '未找到'}",
        expected=expected_found,
        actual=found,
    )


# =============================================================================
# 验证器注册表
# =============================================================================

SEARCH_VALIDATORS = {
    "search_index": _validate_search_index,
    "search_found": _validate_search_found,
}


# =============================================================================
# 测试数据生成
# =============================================================================

def _generate_sorted_array(size: int, min_val: int = 0, max_val: int = 10000) -> List[int]:
    """生成有序数组."""
    arr = sorted(random.sample(range(min_val, max_val), size))
    return arr


# =============================================================================
# 基准测试用例构建
# =============================================================================

def _build_search_cases() -> List[Any]:
    """构建所有搜索基准测试用例."""
    from alphaevolve.benchmarks.problem_suite import BenchmarkCase

    cases = []

    # 小规模数组 (20 元素) - 线性搜索基线
    small_arr = _generate_sorted_array(20)
    small_target = random.choice(small_arr)
    small_not_target = small_arr[-1] + 1  # 确保不存在
    cases.append(BenchmarkCase(
        case_id="search_01",
        family="search",
        description="线性搜索小规模 - 目标存在 (20元素)",
        code=LINEAR_SEARCH_CODE,
        function_name="linear_search",
        args=(small_arr, small_target),
        validator="search_index",
        metadata={
            "expected": small_arr.index(small_target),
            "target": small_target,
            "arr": small_arr,
            "difficulty": "easy",
        },
    ))

    cases.append(BenchmarkCase(
        case_id="search_02",
        family="search",
        description="线性搜索小规模 - 目标不存在 (20元素)",
        code=LINEAR_SEARCH_CODE,
        function_name="linear_search",
        args=(small_arr, small_not_target),
        validator="search_index",
        metadata={
            "expected": -1,
            "target": small_not_target,
            "arr": small_arr,
            "difficulty": "easy",
        },
    ))

    # 中规模数组 (100 元素) - 线性搜索基线
    medium_arr = _generate_sorted_array(100)
    medium_target = random.choice(medium_arr)
    medium_not_target = medium_arr[-1] + 1
    cases.append(BenchmarkCase(
        case_id="search_03",
        family="search",
        description="线性搜索中规模 - 目标存在 (100元素)",
        code=LINEAR_SEARCH_CODE,
        function_name="linear_search",
        args=(medium_arr, medium_target),
        validator="search_index",
        metadata={
            "expected": medium_arr.index(medium_target),
            "target": medium_target,
            "arr": medium_arr,
            "difficulty": "medium",
        },
    ))

    # 二分搜索测试 - 小规模
    cases.append(BenchmarkCase(
        case_id="search_04",
        family="search",
        description="二分搜索小规模 - 目标存在 (20元素)",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(small_arr, small_target),
        validator="search_index",
        metadata={
            "expected": small_arr.index(small_target),
            "target": small_target,
            "arr": small_arr,
            "difficulty": "easy",
        },
    ))

    cases.append(BenchmarkCase(
        case_id="search_05",
        family="search",
        description="二分搜索小规模 - 目标不存在 (20元素)",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(small_arr, small_not_target),
        validator="search_index",
        metadata={
            "expected": -1,
            "target": small_not_target,
            "arr": small_arr,
            "difficulty": "easy",
        },
    ))

    # 二分搜索测试 - 中规模
    cases.append(BenchmarkCase(
        case_id="search_06",
        family="search",
        description="二分搜索中规模 - 目标存在 (100元素)",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(medium_arr, medium_target),
        validator="search_index",
        metadata={
            "expected": medium_arr.index(medium_target),
            "target": medium_target,
            "arr": medium_arr,
            "difficulty": "medium",
        },
    ))

    cases.append(BenchmarkCase(
        case_id="search_07",
        family="search",
        description="二分搜索中规模 - 目标不存在 (100元素)",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(medium_arr, medium_not_target),
        validator="search_index",
        metadata={
            "expected": -1,
            "target": medium_not_target,
            "arr": medium_arr,
            "difficulty": "medium",
        },
    ))

    # 二分搜索测试 - 大规模 (1000 元素)
    large_arr = _generate_sorted_array(1000)
    large_target = random.choice(large_arr)
    large_not_target = large_arr[-1] + 1
    cases.append(BenchmarkCase(
        case_id="search_08",
        family="search",
        description="二分搜索大规模 - 目标存在 (1000元素)",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(large_arr, large_target),
        validator="search_index",
        metadata={
            "expected": large_arr.index(large_target),
            "target": large_target,
            "arr": large_arr,
            "difficulty": "hard",
        },
    ))

    cases.append(BenchmarkCase(
        case_id="search_09",
        family="search",
        description="二分搜索大规模 - 目标不存在 (1000元素)",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(large_arr, large_not_target),
        validator="search_index",
        metadata={
            "expected": -1,
            "target": large_not_target,
            "arr": large_arr,
            "difficulty": "hard",
        },
    ))

    # 指数搜索测试
    cases.append(BenchmarkCase(
        case_id="search_10",
        family="search",
        description="指数搜索中规模 - 目标存在 (100元素)",
        code=EXPONENTIAL_SEARCH_CODE,
        function_name="exponential_search",
        args=(medium_arr, medium_target),
        validator="search_index",
        metadata={
            "expected": medium_arr.index(medium_target),
            "target": medium_target,
            "arr": medium_arr,
            "difficulty": "medium",
        },
    ))

    # 二分搜索边界测试 - 查找第一个元素
    boundary_arr = list(range(1, 101))  # 1-100
    cases.append(BenchmarkCase(
        case_id="search_11",
        family="search",
        description="二分搜索边界 - 查找最小元素",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(boundary_arr, 1),
        validator="search_index",
        metadata={
            "expected": 0,
            "target": 1,
            "arr": boundary_arr,
            "difficulty": "easy",
        },
    ))

    # 二分搜索边界测试 - 查找最后一个元素
    cases.append(BenchmarkCase(
        case_id="search_12",
        family="search",
        description="二分搜索边界 - 查找最大元素",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(boundary_arr, 100),
        validator="search_index",
        metadata={
            "expected": 99,
            "target": 100,
            "arr": boundary_arr,
            "difficulty": "easy",
        },
    ))

    # 单元素数组
    single_arr = [42]
    cases.append(BenchmarkCase(
        case_id="search_13",
        family="search",
        description="单元素数组 - 目标存在",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(single_arr, 42),
        validator="search_index",
        metadata={
            "expected": 0,
            "target": 42,
            "arr": single_arr,
            "difficulty": "easy",
        },
    ))

    cases.append(BenchmarkCase(
        case_id="search_14",
        family="search",
        description="单元素数组 - 目标不存在",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(single_arr, 99),
        validator="search_index",
        metadata={
            "expected": -1,
            "target": 99,
            "arr": single_arr,
            "difficulty": "easy",
        },
    ))

    # 空数组
    empty_arr = []
    cases.append(BenchmarkCase(
        case_id="search_15",
        family="search",
        description="空数组测试",
        code=BINARY_SEARCH_CODE,
        function_name="binary_search",
        args=(empty_arr, 1),
        validator="search_index",
        metadata={
            "expected": -1,
            "target": 1,
            "arr": empty_arr,
            "difficulty": "easy",
        },
    ))

    return cases


def get_search_benchmark_cases() -> List[Any]:
    """返回所有搜索基准测试用例."""
    return _build_search_cases()
