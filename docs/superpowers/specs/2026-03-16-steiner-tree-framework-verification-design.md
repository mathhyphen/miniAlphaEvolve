# Steiner Tree 框架验证研究设计文档

**日期**: 2026-03-16
**状态**: 已批准
**研究类型**: 探索性研究 (1-2 周)

---

## 1. 研究目标

证明 AlphaEvolve 框架可以"重新发现"已知的 Steiner Tree 优化技术，验证其在几何优化问题上的适用性。

### 1.1 具体成功标准

- 重新发现 Fermat 点启发式（3 点最优解）
- 重新发现 120° 角条件（Steiner 点的几何特征）
- 在标准测试用例上达到已知最优解的 95%+ 性能

---

## 2. 设计决策汇总

| 设计维度 | 决策 | 理由 |
|----------|------|------|
| 问题变体 | 欧几里得平面 Steiner Tree | 直接关联 Gilbert-Pollak 猜想 |
| 种子代码 | Fermat 点启发式 | 起点高，快速验证框架 |
| LLM 策略 | 仅用 Anthropic Claude | 成本低，验证现有能力 |
| 评估策略 | A+B+C 混合 | 全面验证 |
| 成功标准 | 重新发现已知技术 | 务实的探索性目标 |
| 时间框架 | 1-2 周 | 探索性研究 |

---

## 3. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AlphaEvolve 框架                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐                                       │
│  │   种子代码        │                                       │
│  │  (Fermat 启发式)  │                                       │
│  └────────┬─────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Evolution 主循环                            ││
│  │  ┌───────────┐  ┌───────────┐  ┌─────────────────────┐ ││
│  │  │ 选择父代   │→ │ LLM 变异   │  │ 评估器 (A+B+C)      │ ││
│  │  │ (锦标赛)   │  │ (5 策略)   │  │ - 已知最优解验证    │ ││
│  │  └───────────┘  └───────────┘  │ - GeoSteiner Oracle │ ││
│  │                                  │ - MST 基线对比      │ ││
│  │                                  └─────────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
│           │                                                 │
│           ▼                                                 │
│  ┌──────────────────┐                                       │
│  │  MAP-Elites 档案  │                                       │
│  │  特征：点数、拓扑、启发式类型                               │
│  └──────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件设计

### 4.1 种子代码结构

**文件**: `steiner_search/seeds/fermat_baseline.py`

种子代码包含：
- Fermat 点计算（3 点最优解）
- MST 基线实现
- Steiner 比率计算框架

LLM 的进化任务：
- 将 Fermat 点启发式推广到 n 点情况
- 发现多 Steiner 点协同优化策略
- 识别并应用 120° 角条件

### 4.2 评估器设计

**文件**: `steiner_search/evaluator.py`

#### CompositeSteinerEvaluator - A+B+C 混合评估器

| 组件 | 权重 | 描述 |
|------|------|------|
| **A: 已知最优解验证** | 50% | 使用文献中标准测试用例验证正确性 |
| **B: GeoSteiner Oracle** | 20% | 调用外部精确求解器验证最优性 |
| **C: MST 基线对比** | 30% | 计算相对于 MST 的改进百分比 |

#### 标准测试用例

```python
KNOWN_OPTIMAL = [
    # (名称，点集，最优比率，容差)
    ("equilateral_triangle", [(0,0), (1,0), (0.5, √3/2)], √3/2, 1e-6),
    ("square", [(0,0), (1,0), (1,1), (0,1)], 0.91068..., 1e-4),
    ("regular_pentagon", [...], ...),
    ("regular_hexagon", [...], ...),
]
```

### 4.3 MAP-Elites 特征维度

**文件**: `steiner_search/features.py`

| 特征维度 | 用途 | Bin 数量 |
|----------|------|----------|
| `NumSteinerPointsFeature` | Steiner 点数量 | 10 |
| `HeuristicTypeFeature` | 启发式类型（Fermat/质心/Voronoi/梯度） | 6 |
| `TopologyComplexityFeature` | 圈复杂度 | 8 |

---

## 5. 目录结构

```
D:/apps/AlphaEvolve/
├── steiner_search/
│   ├── __init__.py
│   ├── evaluator.py           # A+B+C 复合评估器
│   ├── features.py            # MAP-Elites 特征维度
│   ├── test_cases.py          # 标准测试用例生成
│   ├── geosteiner_wrapper.py  # GeoSteiner 接口（可选）
│   ├── seeds/
│   │   └── fermat_baseline.py # 种子代码
│   └── run_experiment.py      # 主实验脚本
├── docs/
│   └── superpowers/specs/
│       └── 2026-03-16-steiner-tree-framework-verification-design.md
└── outputs/
    └── steiner_experiments/   # 实验输出
```

---

## 6. 实验流程

### 第 1 周：基础设施搭建

| 时间 | 任务 | 交付物 |
|------|------|--------|
| 第 1 天 | 实现测试用例生成器 | `test_cases.py` |
| 第 2 天 | 实现复合评估器 | `evaluator.py` |
| 第 3 天 | 实现特征维度 | `features.py` |
| 第 4 天 | 种子代码验证 | 基线运行结果 |
| 第 5 天 | 初步进化实验 | 10 代初步结果 |

### 第 2 周：实验与分析

| 时间 | 任务 | 交付物 |
|------|------|--------|
| 第 6-7 天 | 完整进化实验 | 50-100 代结果 |
| 第 8 天 | 结果分析 | 进化轨迹、档案可视化 |
| 第 9 天 | 重复实验验证 | 可复现性验证 |
| 第 10 天 | 总结报告 | 研究笔记、结论 |

---

## 7. 成功指标

### 7.1 主要指标

- **Fermat 点重新发现率**: 进化出的代码是否正确处理 3 点情况
- **120° 角条件识别**: 是否发现 Steiner 点的角度特征
- **最优解达成率**: 在标准测试用例上达到已知最优的 95%+

### 7.2 次要指标

- **进化收敛速度**: 达到目标性能所需的代数
- **档案多样性**: MAP-Elites 档案覆盖的 cell 数量
- **代码质量**: 进化出的代码的可读性和简洁性

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|----------|
| LLM 无法理解几何概念 | 中 | 高 | 提供详细的变异提示模板 |
| 进化收敛到局部最优 | 中 | 中 | 使用 MAP-Elites 保持多样性 |
| GeoSteiner 安装困难 | 高 | 低 | 作为可选组件，不依赖 |
| 计算时间过长 | 低 | 中 | 减少种群大小，并行评估 |

---

## 9. 关键文件清单

| 文件路径 | 用途 | 优先级 |
|---------|------|--------|
| `steiner_search/evaluator.py` | 复合评估器实现 | P0 |
| `steiner_search/features.py` | 特征维度定义 | P0 |
| `steiner_search/test_cases.py` | 测试用例生成 | P0 |
| `steiner_search/seeds/fermat_baseline.py` | 种子代码 | P0 |
| `steiner_search/run_experiment.py` | 实验主脚本 | P1 |
| `steiner_search/geosteiner_wrapper.py` | GeoSteiner 接口 | P2 (可选) |

---

## 10. 参考文献

1. Gilbert, E. N., & Pollak, H. O. (1968). "Steiner Minimal Trees". SIAM Journal on Applied Mathematics.
2. Du, D.-Z., & Hwang, F. K. (1990). "A Proof of the Gilbert-Pollak Conjecture". Algorithmica.
3. Warme, D. M., Winter, P., & Zachariasen, M. (2000). "GeoSteiner Algorithms".
4. DeepMind FunSearch Nature Paper (2024).

---

## 11. 附录：种子代码示例

```python
"""
Steiner Tree 比率搜索 - Fermat 点基线种子代码

目标：让进化算法发现将 Fermat 点启发式推广到 n 点情况的策略
"""

import math
from typing import List, Tuple

Point = Tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def mst_length(points: List[Point]) -> float:
    """
    Compute Minimum Spanning Tree length using Prim's algorithm.

    Args:
        points: List of (x, y) coordinates

    Returns:
        Total length of MST
    """
    n = len(points)
    if n <= 1:
        return 0.0

    in_mst = [False] * n
    min_dist = [float('inf')] * n
    min_dist[0] = 0.0
    total = 0.0

    for _ in range(n):
        u = -1
        for i in range(n):
            if not in_mst[i] and (u == -1 or min_dist[i] < min_dist[u]):
                u = i

        if min_dist[u] == float('inf'):
            break

        in_mst[u] = True
        total += min_dist[u]

        for v in range(n):
            if not in_mst[v]:
                d = distance(points[u], points[v])
                if d < min_dist[v]:
                    min_dist[v] = d

    return total


def triangle_angles(p1: Point, p2: Point, p3: Point) -> Tuple[float, float, float]:
    """Compute angles of triangle in degrees."""
    a = distance(p2, p3)
    b = distance(p1, p3)
    c = distance(p1, p2)

    def angle(opposite, adj1, adj2):
        if adj1 * adj2 == 0:
            return 180.0
        cos_val = (adj1**2 + adj2**2 - opposite**2) / (2 * adj1 * adj2)
        return math.degrees(math.acos(max(-1, min(1, cos_val))))

    return (angle(a, b, c), angle(b, a, c), angle(c, a, b))


def fermat_point(p1: Point, p2: Point, p3: Point) -> Point:
    """
    Compute Fermat point (Torricelli point) for a triangle.

    The Fermat point minimizes the sum of distances to the three vertices.

    - If all angles < 120°: Fermat point is interior (120° from each vertex)
    - If any angle >= 120°: Fermat point is the vertex with the obtuse angle

    Returns:
        (x, y) coordinates of Fermat point
    """
    angles = triangle_angles(p1, p2, p3)

    # Check for obtuse angle (>= 120°)
    if angles[0] >= 120:
        return p1
    if angles[1] >= 120:
        return p2
    if angles[2] >= 120:
        return p3

    # For acute triangles, use Torricelli construction
    # Simplified: use isogonic center approximation
    # TODO: Evolution should discover better methods

    # Centroid as fallback (not optimal but reasonable)
    return ((p1[0] + p2[0] + p3[0]) / 3,
            (p1[1] + p2[1] + p3[1]) / 3)


def steiner_tree_length(terminals: List[Point]) -> float:
    """
    Approximate Steiner tree length.

    Current implementation:
    - For 3 points: Uses Fermat point
    - For n > 3 points: Falls back to MST

    This is the code that evolution should improve!

    Args:
        terminals: List of terminal points

    Returns:
        Approximate Steiner tree length
    """
    n = len(terminals)

    if n == 3:
        fp = fermat_point(*terminals)
        return sum(distance(fp, p) for p in terminals)
    else:
        # TODO: Evolution should discover better heuristics
        return mst_length(terminals)


def find_steiner_ratio(terminals: List[Point]) -> float:
    """
    Compute Steiner ratio = L_SMT / L_MST.

    Args:
        terminals: List of terminal points

    Returns:
        Steiner ratio (lower is better, optimal is sqrt(3)/2 ≈ 0.866)
    """
    l_mst = mst_length(terminals)
    if l_mst < 1e-10:
        return 1.0

    l_smt = steiner_tree_length(terminals)
    return l_smt / l_mst
```

---

**文档状态**: 已批准，准备进入实现计划阶段
