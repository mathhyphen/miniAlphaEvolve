"""Deterministic benchmark suite for repository-level evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from alphaevolve.sandbox.executor import ExecutionConfig, SandboxExecutor

# 导入新增的基准测试模块
from alphaevolve.benchmarks.sorting_benchmark import (
    SORTING_VALIDATORS as SORTING_VALIDATORS,
    get_sorting_benchmark_cases as _get_sorting_cases,
)
from alphaevolve.benchmarks.search_benchmark import (
    SEARCH_VALIDATORS as SEARCH_VALIDATORS,
    get_search_benchmark_cases as _get_search_cases,
)
from alphaevolve.benchmarks.graph_benchmark import (
    GRAPH_VALIDATORS as GRAPH_VALIDATORS,
    get_graph_benchmark_cases as _get_graph_cases,
)

MST_BASELINE_CODE = """def mst(edges):
    if not edges:
        return 0.0

    max_node = max(max(left, right) for left, right, _ in edges)
    parent = list(range(max_node + 1))
    rank = [0] * (max_node + 1)

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1
        return True

    total = 0.0
    for left, right, weight in sorted(edges, key=lambda edge: edge[2]):
        if union(left, right):
            total += weight
    return total
"""

STEINER_BASELINE_CODE = """def steiner_tree(terminals):
    import math

    if len(terminals) < 2:
        return 0.0

    remaining = set(range(1, len(terminals)))
    in_tree = {0}
    total = 0.0

    while remaining:
        best_index = None
        best_distance = float("inf")
        for source in in_tree:
            sx, sy = terminals[source]
            for target in remaining:
                tx, ty = terminals[target]
                distance = math.hypot(sx - tx, sy - ty)
                if distance < best_distance:
                    best_distance = distance
                    best_index = target

        in_tree.add(best_index)
        remaining.remove(best_index)
        total += best_distance

    return total
"""


@dataclass(frozen=True)
class BenchmarkCase:
    """A single deterministic benchmark case."""

    case_id: str
    family: str
    description: str
    code: str
    function_name: str
    args: Tuple[Any, ...]
    validator: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome for a benchmark case."""

    passed: bool
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Execution result for a benchmark case."""

    case_id: str
    family: str
    description: str
    success: bool
    passed: bool
    output: Any
    error: Optional[str]
    execution_time: float
    peak_memory_mb: float
    validation_message: str
    expected: Optional[Any] = None


def _mst_weight(edges: Sequence[Tuple[int, int, float]]) -> float:
    if not edges:
        return 0.0

    parent = {}
    rank = {}

    def find(node: int) -> int:
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1
        return True

    for left, right, _ in edges:
        parent.setdefault(left, left)
        parent.setdefault(right, right)
        rank.setdefault(left, 0)
        rank.setdefault(right, 0)

    total = 0.0
    for left, right, weight in sorted(edges, key=lambda edge: edge[2]):
        if union(left, right):
            total += weight
    return total


def _euclidean_mst_length(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0

    remaining = set(range(1, len(points)))
    in_tree = {0}
    total = 0.0

    while remaining:
        best_node = None
        best_dist = float("inf")
        for source in in_tree:
            sx, sy = points[source]
            for target in remaining:
                tx, ty = points[target]
                dist = math.hypot(sx - tx, sy - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_node = target

        assert best_node is not None
        remaining.remove(best_node)
        in_tree.add(best_node)
        total += best_dist

    return total


def _validate_mst_exact(case: BenchmarkCase, output: Any) -> ValidationResult:
    expected = case.metadata["expected"]
    if not isinstance(output, (int, float)):
        return ValidationResult(
            passed=False,
            message="Output is not numeric.",
            expected=expected,
            actual=output,
        )

    passed = abs(float(output) - float(expected)) <= 1e-9
    return ValidationResult(
        passed=passed,
        message="Exact MST weight match." if passed else "MST weight does not match reference.",
        expected=expected,
        actual=output,
    )


def _validate_steiner_not_worse_than_mst(case: BenchmarkCase, output: Any) -> ValidationResult:
    mst_length = case.metadata["mst_length"]
    if not isinstance(output, (int, float)):
        return ValidationResult(
            passed=False,
            message="Output is not numeric.",
            expected=f"0 <= output <= {mst_length}",
            actual=output,
        )

    numeric_output = float(output)
    passed = 0.0 <= numeric_output <= mst_length + 1e-9
    return ValidationResult(
        passed=passed,
        message=(
            "Output is not worse than the MST upper bound."
            if passed
            else "Output exceeds the MST upper bound, so it is not a valid Steiner-tree-length style objective."
        ),
        expected=f"0 <= output <= {mst_length}",
        actual=numeric_output,
    )


VALIDATORS = {
    "mst_exact": _validate_mst_exact,
    "steiner_not_worse_than_mst": _validate_steiner_not_worse_than_mst,
    # 排序验证器
    "sorted": SORTING_VALIDATORS["sorted"],
    "sorted_stable": SORTING_VALIDATORS["sorted_stable"],
    # 搜索验证器
    "search_index": SEARCH_VALIDATORS["search_index"],
    "search_found": SEARCH_VALIDATORS["search_found"],
    # 图遍历验证器
    "traversal": GRAPH_VALIDATORS["traversal"],
    "bfs_order": GRAPH_VALIDATORS["bfs_order"],
    "dfs_order": GRAPH_VALIDATORS["dfs_order"],
}


def _build_mst_cases() -> List[BenchmarkCase]:
    graphs = [
        ("mst_01", "triangle", [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]),
        ("mst_02", "chain_4", [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0), (0, 3, 10.0)]),
        ("mst_03", "square_with_diagonal", [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 5.0)]),
        ("mst_04", "star", [(0, 1, 1.0), (0, 2, 2.0), (0, 3, 3.0), (1, 2, 10.0), (1, 3, 10.0)]),
        ("mst_05", "dense_5", [(0, 1, 4.0), (0, 2, 2.0), (0, 3, 7.0), (0, 4, 3.0), (1, 2, 1.0), (1, 3, 5.0), (1, 4, 8.0), (2, 3, 6.0), (2, 4, 9.0), (3, 4, 2.0)]),
        ("mst_06", "duplicate_weights", [(0, 1, 2.0), (1, 2, 2.0), (2, 3, 2.0), (0, 3, 2.0), (0, 2, 5.0)]),
        ("mst_07", "floating_weights", [(0, 1, 1.5), (1, 2, 2.5), (0, 2, 10.5), (2, 3, 0.5), (1, 3, 1.0)]),
        ("mst_08", "disconnected_by_expensive_bridge", [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (2, 3, 50.0), (0, 4, 60.0)]),
        ("mst_09", "six_nodes_sparse", [(0, 1, 3.0), (1, 2, 1.0), (2, 3, 4.0), (3, 4, 2.0), (4, 5, 6.0), (0, 5, 20.0), (1, 4, 5.0)]),
        ("mst_10", "cycle_with_shortcuts", [(0, 1, 3.0), (1, 2, 3.0), (2, 3, 3.0), (3, 4, 3.0), (4, 0, 3.0), (0, 2, 1.0), (1, 3, 1.0), (2, 4, 1.0)]),
    ]

    cases = []
    for case_id, description, edges in graphs:
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                family="mst",
                description=description,
                code=MST_BASELINE_CODE,
                function_name="mst",
                args=(edges,),
                validator="mst_exact",
                metadata={"expected": _mst_weight(edges)},
            )
        )
    return cases


def _build_steiner_cases() -> List[BenchmarkCase]:
    point_sets = [
        ("steiner_01", "equilateral_triangle", [(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)]),
        ("steiner_02", "unit_square", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
        ("steiner_03", "collinear_3", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]),
        ("steiner_04", "collinear_4", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (4.0, 0.0)]),
        ("steiner_05", "rectangle", [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]),
        ("steiner_06", "pentagon_like", [(0.0, 0.0), (1.2, 0.1), (1.8, 0.9), (0.9, 1.7), (-0.1, 0.8)]),
        ("steiner_07", "cluster_plus_outlier", [(0.0, 0.0), (0.2, 0.1), (0.1, 0.3), (3.0, 3.0)]),
        ("steiner_08", "wide_triangle", [(0.0, 0.0), (3.0, 0.0), (1.5, 2.6)]),
        ("steiner_09", "zigzag_5", [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0), (3.0, 0.5), (4.0, 0.0)]),
        ("steiner_10", "asymmetric_6", [(0.0, 0.0), (2.0, 0.2), (1.0, 1.5), (3.2, 1.4), (4.0, 0.0), (2.2, 2.4)]),
    ]

    cases = []
    for case_id, description, points in point_sets:
        mst_length = _euclidean_mst_length(points)
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                family="steiner",
                description=description,
                code=STEINER_BASELINE_CODE,
                function_name="steiner_tree",
                args=(points,),
                validator="steiner_not_worse_than_mst",
                metadata={"mst_length": mst_length},
            )
        )
    return cases


def build_20_problem_suite() -> List[BenchmarkCase]:
    """Return the fixed 20-case benchmark suite."""
    return _build_mst_cases() + _build_steiner_cases()


def filter_cases(
    cases: Sequence[BenchmarkCase],
    case_ids: Optional[Iterable[str]] = None,
    family: Optional[str] = None,
) -> List[BenchmarkCase]:
    selected = list(cases)
    if family:
        selected = [case for case in selected if case.family == family]
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case.case_id in wanted]
    return selected


def get_all_benchmarks() -> List[BenchmarkCase]:
    """返回所有基准测试用例.

    包含:
    - MST (Minimum Spanning Tree) 问题
    - Steiner Tree 问题
    - 排序算法问题
    - 搜索算法问题
    - 图遍历问题

    Returns:
        所有基准测试用例列表
    """
    all_cases: List[BenchmarkCase] = []
    all_cases.extend(build_20_problem_suite())  # MST + Steiner
    all_cases.extend(_get_sorting_cases())
    all_cases.extend(_get_search_cases())
    all_cases.extend(_get_graph_cases())
    return all_cases


def filter_by_difficulty(
    cases: Sequence[BenchmarkCase],
    difficulty: str,
) -> List[BenchmarkCase]:
    """按难度级别筛选基准测试用例.

    Args:
        cases: 基准测试用例序列
        difficulty: 难度级别 ("easy", "medium", "hard")

    Returns:
        符合指定难度级别的用例列表

    Raises:
        ValueError: 当难度级别无效时
    """
    valid_difficulties = {"easy", "medium", "hard"}
    if difficulty not in valid_difficulties:
        raise ValueError(f"无效的难度级别: {difficulty}. 可选值: {valid_difficulties}")

    return [
        case for case in cases
        if case.metadata.get("difficulty") == difficulty
    ]


def run_problem_suite(
    cases: Sequence[BenchmarkCase],
    executor: Optional[SandboxExecutor] = None,
    config: Optional[ExecutionConfig] = None,
) -> List[BenchmarkResult]:
    """Execute benchmark cases and validate outputs."""
    sandbox = executor or SandboxExecutor()
    exec_config = config or ExecutionConfig(timeout_seconds=5.0, memory_limit_mb=256.0)
    results: List[BenchmarkResult] = []

    for case in cases:
        execution = sandbox.execute(
            code=case.code,
            function=case.function_name,
            args=case.args,
            config=exec_config,
        )
        if execution.success:
            validation = VALIDATORS[case.validator](case, execution.output)
        else:
            validation = ValidationResult(
                passed=False,
                message="Execution failed before validation.",
                expected=case.metadata,
                actual=execution.error,
            )

        results.append(
            BenchmarkResult(
                case_id=case.case_id,
                family=case.family,
                description=case.description,
                success=execution.success,
                passed=execution.success and validation.passed,
                output=execution.output,
                error=execution.error,
                execution_time=execution.execution_time,
                peak_memory_mb=execution.peak_memory_mb,
                validation_message=validation.message,
                expected=validation.expected,
            )
        )

    return results


def summarize_results(results: Sequence[BenchmarkResult]) -> Dict[str, Any]:
    """Build an aggregate summary from benchmark results."""
    summary: Dict[str, Any] = {
        "total_cases": len(results),
        "passed_cases": sum(1 for result in results if result.passed),
        "successful_executions": sum(1 for result in results if result.success),
        "families": {},
    }

    family_names = sorted({result.family for result in results})
    for family in family_names:
        family_results = [result for result in results if result.family == family]
        summary["families"][family] = {
            "total_cases": len(family_results),
            "passed_cases": sum(1 for result in family_results if result.passed),
            "successful_executions": sum(1 for result in family_results if result.success),
        }

    return summary


def results_to_json(results: Sequence[BenchmarkResult]) -> str:
    """Serialize benchmark results to JSON."""
    return json.dumps([asdict(result) for result in results], indent=2)
