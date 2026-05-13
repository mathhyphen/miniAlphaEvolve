"""图遍历基准测试 - 基于BFS和DFS的基线实现.

该模块提供了图遍历的基准测试,
用于评估算法优化效果(如启发式搜索、A*等)。
"""

from __future__ import annotations

import collections
import random
from typing import Any, Dict, List, Set, Tuple


# =============================================================================
# 基线代码 - BFS 和 DFS
# =============================================================================

BFS_CODE = """def bfs(graph, start):
    \"\"\"广度优先搜索实现 - 使用队列.\"\"\"
    visited = set()
    queue = collections.deque([start])
    result = []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            # 假设 graph[node] 是相邻节点列表
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    return result
"""

DFS_CODE = """def dfs(graph, start):
    \"\"\"深度优先搜索实现 - 使用栈.\"\"\"
    visited = set()
    stack = [start]
    result = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            # 假设 graph[node] 是相邻节点列表
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
    return result
"""

DFS_RECURSIVE_CODE = """def dfs_recursive(graph, node, visited=None):
    \"\"\"深度优先搜索递归实现.\"\"\"
    if visited is None:
        visited = set()
    visited.add(node)
    result = [node]
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    return result
"""


# =============================================================================
# 验证器
# =============================================================================

def _validate_traversal_order(case: Any, output: Any) -> Any:
    """验证遍历结果是否包含所有节点且无重复."""
    from alphaevolve.benchmarks.problem_suite import ValidationResult

    expected_nodes = case.metadata["expected_nodes"]
    start_node = case.metadata["start"]

    if not isinstance(output, list):
        return ValidationResult(
            passed=False,
            message=f"输出不是列表类型, 得到 {type(output).__name__}",
            expected=f"包含 {len(expected_nodes)} 个节点的列表",
            actual=type(output).__name__,
        )

    output_set = set(output)

    # 检查是否包含所有节点
    if output_set != expected_nodes:
        missing = expected_nodes - output_set
        extra = output_set - expected_nodes
        msg = []
        if missing:
            msg.append(f"缺少节点: {missing}")
        if extra:
            msg.append(f"多余节点: {extra}")
        return ValidationResult(
            passed=False,
            message="; ".join(msg),
            expected=sorted(expected_nodes),
            actual=sorted(output_set),
        )

    # 检查第一个节点是否是起始节点
    if output and output[0] != start_node:
        return ValidationResult(
            passed=False,
            message=f"遍历起点错误: 期望 {start_node}, 得到 {output[0]}",
            expected=start_node,
            actual=output[0],
        )

    # 检查是否有重复节点
    if len(output) != len(output_set):
        return ValidationResult(
            passed=False,
            message="遍历结果包含重复节点",
            expected="无重复节点",
            actual=output,
        )

    return ValidationResult(
        passed=True,
        message=f"遍历结果正确, 包含 {len(expected_nodes)} 个节点",
        expected=sorted(expected_nodes),
        actual=output,
    )


def _validate_bfs_order(case: Any, output: Any) -> Any:
    """验证 BFS 遍历顺序(按层访问)."""
    from alphaevolve.benchmarks.problem_suite import ValidationResult

    expected_nodes = case.metadata["expected_nodes"]
    levels = case.metadata["levels"]  # Dict[node, level]
    start_node = case.metadata["start"]

    # 先做基本验证
    basic_validation = _validate_traversal_order(case, output)
    if not basic_validation.passed:
        return basic_validation

    # 验证层次顺序
    node_to_level = {}
    for node in output:
        if node in levels:
            node_to_level[node] = levels[node]

    # BFS 应该按层访问
    for i in range(len(output) - 1):
        curr, next_node = output[i], output[i + 1]
        curr_level = node_to_level.get(curr, -1)
        next_level = node_to_level.get(next_node, -1)

        # 当前节点的层数不应大于下一个节点
        if curr_level > next_level + 1:
            return ValidationResult(
                passed=False,
                message=f"BFS层次顺序错误: 节点 {curr}(层{curr_level}) 在节点 {next_node}(层{next_level}) 之前",
                expected="按层递增顺序",
                actual=output,
            )

    return ValidationResult(
        passed=True,
        message="BFS 遍历顺序正确",
        expected=sorted(expected_nodes),
        actual=output,
    )


def _validate_dfs_order(case: Any, output: Any) -> Any:
    """验证 DFS 遍历顺序(先深入后回溯)."""
    # DFS 验证与基本遍历验证相同
    return _validate_traversal_order(case, output)


# =============================================================================
# 验证器注册表
# =============================================================================

GRAPH_VALIDATORS = {
    "traversal": _validate_traversal_order,
    "bfs_order": _validate_bfs_order,
    "dfs_order": _validate_dfs_order,
}


# =============================================================================
# 测试数据生成
# =============================================================================

def _build_graph_chain(n: int) -> Tuple[Dict[int, List[int]], Set[int]]:
    """构建链状图: 0-1-2-3-...-n."""
    graph = {i: [i + 1] for i in range(n - 1)}
    graph[n] = []
    return graph, set(range(n + 1))


def _build_graph_star(center: int, leaves: List[int]) -> Tuple[Dict[int, List[int]], Set[int]]:
    """构建星形图: 中心节点连接所有叶节点."""
    graph = {center: list(leaves)}
    for leaf in leaves:
        graph[leaf] = [center]
    return graph, set([center] + list(leaves))


def _build_graph_binary_tree(depth: int) -> Tuple[Dict[int, List[int]], Set[int]]:
    """构建二叉树."""
    graph = {}
    nodes = set()
    node_id = 0
    frontier = [node_id]
    nodes.add(node_id)

    for _ in range(depth):
        new_frontier = []
        for parent in frontier:
            left = parent * 2 + 1
            right = parent * 2 + 2
            graph[parent] = [left, right]
            nodes.add(left)
            nodes.add(right)
            new_frontier.extend([left, right])
        frontier = new_frontier

    # 最后一个 frontier 没有子节点
    for node in frontier:
        graph[node] = []
    return graph, nodes


def _build_graph_grid(rows: int, cols: int) -> Tuple[Dict[int, List[int]], Set[int]]:
    """构建网格图."""
    graph = {}
    nodes = set()
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            nodes.add(node)
            neighbors = []
            if r > 0:
                neighbors.append((r - 1) * cols + c)
            if r < rows - 1:
                neighbors.append((r + 1) * cols + c)
            if c > 0:
                neighbors.append(r * cols + c - 1)
            if c < cols - 1:
                neighbors.append(r * cols + c + 1)
            graph[node] = neighbors
    return graph, nodes


def _build_graph_random(n: int, edge_prob: float = 0.3, seed: int = 42) -> Tuple[Dict[int, List[int]], Set[int]]:
    """构建随机图."""
    random.seed(seed)
    graph = {i: [] for i in range(n)}
    nodes = set(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < edge_prob:
                graph[i].append(j)
                graph[j].append(i)

    return graph, nodes


# =============================================================================
# 基准测试用例构建
# =============================================================================

def _build_graph_cases() -> List[Any]:
    """构建所有图遍历基准测试用例."""
    from alphaevolve.benchmarks.problem_suite import BenchmarkCase

    cases = []

    # BFS 测试用例
    # 1. 简单链状图
    chain_graph, chain_nodes = _build_graph_chain(10)
    cases.append(BenchmarkCase(
        case_id="graph_bfs_01",
        family="graph",
        description="BFS 链状图 (10节点)",
        code=BFS_CODE,
        function_name="bfs",
        args=(chain_graph, 0),
        validator="bfs_order",
        metadata={
            "expected_nodes": chain_nodes,
            "start": 0,
            "levels": {i: i for i in range(11)},
            "graph_type": "chain",
        },
    ))

    # 2. 星形图
    star_graph, star_nodes = _build_graph_star(0, [1, 2, 3, 4, 5])
    cases.append(BenchmarkCase(
        case_id="graph_bfs_02",
        family="graph",
        description="BFS 星形图 (中心+5叶节点)",
        code=BFS_CODE,
        function_name="bfs",
        args=(star_graph, 0),
        validator="bfs_order",
        metadata={
            "expected_nodes": star_nodes,
            "start": 0,
            "levels": {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
            "graph_type": "star",
        },
    ))

    # 3. 二叉树
    btree_graph, btree_nodes = _build_graph_binary_tree(3)  # 深度3, 约15节点
    cases.append(BenchmarkCase(
        case_id="graph_bfs_03",
        family="graph",
        description="BFS 二叉树 (深度3)",
        code=BFS_CODE,
        function_name="bfs",
        args=(btree_graph, 0),
        validator="bfs_order",
        metadata={
            "expected_nodes": btree_nodes,
            "start": 0,
            "levels": {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3},
            "graph_type": "binary_tree",
        },
    ))

    # 4. 网格图
    grid_graph, grid_nodes = _build_graph_grid(3, 4)  # 3x4网格
    grid_levels = {}
    for r in range(3):
        for c in range(4):
            grid_levels[r * 4 + c] = r  # 行作为层
    cases.append(BenchmarkCase(
        case_id="graph_bfs_04",
        family="graph",
        description="BFS 网格图 (3x4)",
        code=BFS_CODE,
        function_name="bfs",
        args=(grid_graph, 0),
        validator="bfs_order",
        metadata={
            "expected_nodes": grid_nodes,
            "start": 0,
            "levels": grid_levels,
            "graph_type": "grid",
        },
    ))

    # 5. 随机图
    random_graph, random_nodes = _build_graph_random(20, 0.2, seed=123)
    cases.append(BenchmarkCase(
        case_id="graph_bfs_05",
        family="graph",
        description="BFS 随机图 (20节点)",
        code=BFS_CODE,
        function_name="bfs",
        args=(random_graph, 0),
        validator="traversal",
        metadata={
            "expected_nodes": random_nodes,
            "start": 0,
            "graph_type": "random",
        },
    ))

    # DFS 测试用例
    # 6. DFS 链状图
    cases.append(BenchmarkCase(
        case_id="graph_dfs_01",
        family="graph",
        description="DFS 链状图 (10节点)",
        code=DFS_CODE,
        function_name="dfs",
        args=(chain_graph, 0),
        validator="dfs_order",
        metadata={
            "expected_nodes": chain_nodes,
            "start": 0,
            "graph_type": "chain",
        },
    ))

    # 7. DFS 星形图
    cases.append(BenchmarkCase(
        case_id="graph_dfs_02",
        family="graph",
        description="DFS 星形图 (中心+5叶节点)",
        code=DFS_CODE,
        function_name="dfs",
        args=(star_graph, 0),
        validator="dfs_order",
        metadata={
            "expected_nodes": star_nodes,
            "start": 0,
            "graph_type": "star",
        },
    ))

    # 8. DFS 二叉树
    cases.append(BenchmarkCase(
        case_id="graph_dfs_03",
        family="graph",
        description="DFS 二叉树 (深度3)",
        code=DFS_CODE,
        function_name="dfs",
        args=(btree_graph, 0),
        validator="dfs_order",
        metadata={
            "expected_nodes": btree_nodes,
            "start": 0,
            "graph_type": "binary_tree",
        },
    ))

    # 9. DFS 网格图
    cases.append(BenchmarkCase(
        case_id="graph_dfs_04",
        family="graph",
        description="DFS 网格图 (3x4)",
        code=DFS_CODE,
        function_name="dfs",
        args=(grid_graph, 0),
        validator="dfs_order",
        metadata={
            "expected_nodes": grid_nodes,
            "start": 0,
            "graph_type": "grid",
        },
    ))

    # 10. DFS 递归实现
    cases.append(BenchmarkCase(
        case_id="graph_dfs_05",
        family="graph",
        description="DFS 递归实现 链状图 (10节点)",
        code=DFS_RECURSIVE_CODE,
        function_name="dfs_recursive",
        args=(chain_graph, 0),
        validator="dfs_order",
        metadata={
            "expected_nodes": chain_nodes,
            "start": 0,
            "graph_type": "chain",
        },
    ))

    # 11. 大规模 BFS
    large_chain_graph, large_chain_nodes = _build_graph_chain(100)
    cases.append(BenchmarkCase(
        case_id="graph_bfs_06",
        family="graph",
        description="BFS 大规模链状图 (100节点)",
        code=BFS_CODE,
        function_name="bfs",
        args=(large_chain_graph, 0),
        validator="bfs_order",
        metadata={
            "expected_nodes": large_chain_nodes,
            "start": 0,
            "levels": {i: i for i in range(101)},
            "graph_type": "chain",
            "difficulty": "hard",
        },
    ))

    # 12. 大规模 DFS
    cases.append(BenchmarkCase(
        case_id="graph_dfs_06",
        family="graph",
        description="DFS 大规模链状图 (100节点)",
        code=DFS_CODE,
        function_name="dfs",
        args=(large_chain_graph, 0),
        validator="dfs_order",
        metadata={
            "expected_nodes": large_chain_nodes,
            "start": 0,
            "graph_type": "chain",
            "difficulty": "hard",
        },
    ))

    # 13. 从中间节点开始 BFS
    cases.append(BenchmarkCase(
        case_id="graph_bfs_07",
        family="graph",
        description="BFS 从中间节点开始 (链状图)",
        code=BFS_CODE,
        function_name="bfs",
        args=(chain_graph, 5),
        validator="traversal",
        metadata={
            "expected_nodes": chain_nodes,
            "start": 5,
            "graph_type": "chain",
        },
    ))

    # 14. 完全图 (所有节点互连)
    complete_graph = {i: [j for j in range(10) if j != i] for i in range(10)}
    complete_nodes = set(range(10))
    cases.append(BenchmarkCase(
        case_id="graph_bfs_08",
        family="graph",
        description="BFS 完全图 (10节点)",
        code=BFS_CODE,
        function_name="bfs",
        args=(complete_graph, 0),
        validator="traversal",
        metadata={
            "expected_nodes": complete_nodes,
            "start": 0,
            "graph_type": "complete",
        },
    ))

    # 15. 环形图
    ring_graph = {i: [(i - 1) % 8, (i + 1) % 8] for i in range(8)}
    ring_nodes = set(range(8))
    cases.append(BenchmarkCase(
        case_id="graph_bfs_09",
        family="graph",
        description="BFS 环形图 (8节点)",
        code=BFS_CODE,
        function_name="bfs",
        args=(ring_graph, 0),
        validator="traversal",
        metadata={
            "expected_nodes": ring_nodes,
            "start": 0,
            "graph_type": "ring",
        },
    ))

    return cases


def get_graph_benchmark_cases() -> List[Any]:
    """返回所有图遍历基准测试用例."""
    return _build_graph_cases()
