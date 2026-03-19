# AlphaEvolve 使用指南

## 快速开始：如何用 AlphaEvolve 优化算法

### 第一步：准备你的"种子代码"

种子代码是你想要优化的算法的初始实现。它可以是低效的、朴素的版本。

```python
# 例子：排序算法的种子代码
SEED_CODE = """
def sort_list(items):
    # 这是冒泡排序 - 效率较低
    result = list(items)
    n = len(result)
    for i in range(n):
        for j in range(i + 1, n):
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]
    return result
"""
```

### 第二步：定义测试用例

测试用例用来验证进化后的代码是否正确。

```python
test_cases = [
    # (输入参数), 期望输出
    (([3, 1, 2],), [1, 2, 3]),
    (([5, 4, 3, 2, 1],), [1, 2, 3, 4, 5]),
    (([1],), [1]),
    (([]), []),
]
```

### 第三步：配置并运行进化

```python
from alphaevolve import (
    Evolution,
    EvolutionConfig,
    MutationEngine,
    UnitTestEvaluator,
)

# 配置进化参数
config = EvolutionConfig(
    population_size=10,      # 种群大小
    max_generations=5,        # 最大进化代数
    elitism_count=2,          # 保留的精英个体数
    mutation_rate=0.7,        # 变异率
    seed=42,                  # 随机种子
)

# 创建评估器
evaluator = UnitTestEvaluator(
    test_cases=test_cases,
    function_name="sort_list",  # 你的函数名
    partial_credit=True,        # 部分正确也得部分分数
)

# 创建进化实例
evolution = Evolution(
    config=config,
    mutation_engine=MutationEngine(),
    evaluator=evaluator,
    seed_code=SEED_CODE,
)

# 运行进化
best = evolution.evolve()

print(f"最优适应度：{best.fitness:.2f}")
print(f"进化出的代码:\n{best.code}")
```

---

## 实际例子：最小生成树 (MST) 算法优化

### 问题：Kruskal 算法能被优化吗？

Kruskal 算法的时间复杂度已经是 O(E log V)，但**实现细节**会影响实际运行速度：

- 排序可以用内置的 `sorted()` 而不是手动实现
- 并查集可以用路径压缩优化
- 可以用迭代代替递归避免栈溢出

### 完整的 MST 优化代码

```python
from alphaevolve import Evolution, EvolutionConfig, MutationEngine, UnitTestEvaluator

# 种子代码：低效的 Kruskal 实现
SEED_CODE = """
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, i)

def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        else:
            parent[root_y] = root_x
            if rank[root_x] == rank[root_y]:
                rank[root_x] += 1

def mst_kruskal(vertices, edges):
    # 手动实现选择排序 - 低效
    sorted_edges = []
    for i in range(len(edges)):
        min_idx = i
        for j in range(i + 1, len(edges)):
            if edges[j][2] < edges[min_idx][2]:
                min_idx = j
        sorted_edges.append(edges[min_idx])

    parent = list(range(vertices))
    rank = [0] * vertices

    mst_edges = []
    mst_weight = 0

    for edge in sorted_edges:
        u, v, w = edge
        root_u = find(parent, u)
        root_v = find(parent, v)
        if root_u != root_v:
            mst_edges.append(edge)
            mst_weight += w
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            else:
                parent[root_v] = root_u
                if rank[root_u] == rank[root_v]:
                    rank[root_u] += 1

    return mst_edges, mst_weight
"""

# 测试用例
test_cases = [
    # (vertices, edges), (expected_mst_edges, expected_weight)
    ((3, [(0, 1, 1), (1, 2, 2), (0, 2, 3)]), 3),  # 三角形
    ((4, [(0, 1, 1), (1, 2, 2), (2, 3, 3)]), 6),  # 链式
    ((2, [(0, 1, 10)]), 10),                        # 单边
]

# 运行进化
config = EvolutionConfig(population_size=15, max_generations=8)
evolution = Evolution(
    config=config,
    mutation_engine=MutationEngine(),
    evaluator=UnitTestEvaluator(
        test_cases=[((v, e), w) for v, e, w in test_cases],
        function_name="mst_kruskal",
    ),
    seed_code=SEED_CODE,
)

best = evolution.evolve()
print(f"最优代码:\n{best.code}")
```

### 可能进化出的优化

经过几代进化，你可能会看到：

```python
# 优化 1: 使用内置 sorted() 代替手动排序
sorted_edges = sorted(edges, key=lambda x: x[2])

# 优化 2: 路径压缩的 find
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])  # 路径压缩
    return parent[i]

# 优化 3: 简化的 union
def union(parent, rank, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x == root_y:
        return False
    if rank[root_x] < rank[root_y]:
        parent[root_x] = root_y
    else:
        parent[root_y] = root_x
        if rank[root_x] == rank[root_y]:
            rank[root_x] += 1
    return True
```

---

## 进阶用法

### 1. 使用 LLM 进行智能变异

```python
from alphaevolve import (
    LLMEnsemble, AnthropicClient, LLMConfig,
    RoutingStrategy
)
import os

# 需要设置 ANTHROPIC_API_KEY
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

llm_config = LLMConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
)

ensemble = LLMEnsemble(routing_strategy=RoutingStrategy.PRIORITY_BASED)
ensemble.register_client("main", AnthropicClient(llm_config))

engine = MutationEngine()
engine.set_llm_ensemble(ensemble)

# 然后用这个 engine 进行进化
```

### 2. 使用 MAP-Elites 保持多样性

```python
from alphaevolve import ProgramArchive, ArchiveConfig, CodeComplexityFeature

archive = ProgramArchive(ArchiveConfig(
    feature_dimensions=[CodeComplexityFeature(n_bins=10)],
    bins_per_dimension=10,
))

# 在进化过程中添加个体
for individual in population:
    archive.add(individual)

# 查看档案统计
stats = archive.get_stats()
print(f"填充率：{stats.fill_rate:.2%}")
print(f"独特解数量：{stats.unique_solutions}")
```

### 3. 多阶段评估流水线

```python
from alphaevolve import (
    EvaluationPipeline,
    SyntaxCheckStage,
    BasicTestStage,
    EdgeCaseStage,
)

pipeline = EvaluationPipeline(stages=[
    SyntaxCheckStage(),  # 先检查语法
    BasicTestStage(       # 再跑基本测试
        test_cases=simple_tests,
        function_name="func",
    ),
    EdgeCaseStage(        # 最后跑边界测试
        edge_cases=edge_tests,
        function_name="func",
    ),
], early_terminate=True)  # 失败就提前终止

result = pipeline.evaluate(code)
print(f"通过：{result.passed}, 分数：{result.fitness:.2f}")
```

---

## 运行内置示例

```bash
# 排序算法进化
python -m alphaevolve.examples.sorting_demo

# 最小生成树算法进化
python -m alphaevolve.examples.mst_demo

# MAP-Elites 档案演示
python -m alphaevolve.examples.sorting_demo --demo-archive

# 评估流水线演示
python -m alphaevolve.examples.sorting_demo --demo-pipeline
```

---

## 适合优化的问题类型

| 问题类型 | 例子 | 可优化空间 |
|---------|------|-----------|
| ✅ 排序/搜索 | 快速排序、二分搜索 | 常数因子、代码大小 |
| ✅ 图算法 | MST、最短路径 | 数据结构选择、实现技巧 |
| ✅ 数值计算 | 矩阵乘法、点积 | 减少运算次数 |
| ✅ 动态规划 | 背包问题、LCS | 状态压缩、空间优化 |
| ✅ 调度问题 | 任务分配、资源调度 | 启发式策略 |
| ❌ 数学证明 | 质数定理、微积分 | 需要形式推理 |
| ❌ 业务逻辑 | 用户认证、支付流程 | 不需要进化优化 |

---

## 关键提示

1. **种子代码不一定要对** - 即使种子代码有 bug，进化也可能找到正确的解

2. **测试用例要全面** - 测试用例越好，进化出的代码质量越高

3. **种群大小很重要** - 太小的种群容易陷入局部最优

4. **给进化足够时间** - 复杂问题可能需要 20+ 代

5. **性能优化需要性能评估器** - 只用正确性测试不会自动优化性能

---

## 参考项目

- `alphaevolve/examples/sorting_demo.py` - 排序算法进化
- `alphaevolve/examples/mst_demo.py` - 最小生成树算法进化
- `alphaevolve/tests/test_comprehensive.py` - 测试用例参考
