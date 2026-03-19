"""
AlphaEvolve: MST Algorithm - Intensive Evolution Run

This script runs an extended evolution (1000 generations) to optimize
the MST (Kruskal) algorithm.
"""

import logging
import time
import json
from pathlib import Path
from alphaevolve import (
    EvolutionConfig,
    MutationEngine,
    UnitTestEvaluator,
    Evolution,
)
from alphaevolve.core.data_structures import EvaluatorResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mst_evolution.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# 当前最优的 MST 算法实现（作为参考基准）
# ============================================================
OPTIMAL_REFERENCE = """
def find(parent, i):
    # 路径压缩的并查集 find 操作
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    # 按秩合并的并查集 union 操作
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

def mst_kruskal(vertices, edges):
    '''
    Kruskal 算法最优实现：
    - 使用内置 sorted() O(E log E)
    - 并查集带路径压缩和按秩合并
    - 提前终止（找到 V-1 条边就停止）
    '''
    # 使用内置排序 - 最优
    sorted_edges = sorted(edges, key=lambda x: x[2])

    parent = list(range(vertices))
    rank = [0] * vertices

    mst_weight = 0
    edges_count = 0

    for edge in sorted_edges:
        u, v, w = edge
        if union(parent, rank, u, v):
            mst_weight += w
            edges_count += 1
            if edges_count == vertices - 1:
                break  # 提前终止

    return mst_weight
"""


# ============================================================
# 种子代码：低效版本（供进化优化）
# ============================================================
SEED_CODE = """
def find(parent, i):
    # 没有路径压缩 - 低效
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
        return True
    return False

def mst_kruskal(vertices, edges):
    # 手动实现冒泡排序 - 非常低效 O(n^2)
    sorted_edges = list(edges)
    n = len(sorted_edges)
    for i in range(n):
        for j in range(0, n - i - 1):
            if sorted_edges[j][2] > sorted_edges[j + 1][2]:
                sorted_edges[j], sorted_edges[j + 1] = sorted_edges[j + 1], sorted_edges[j]

    parent = list(range(vertices))
    rank = [0] * vertices

    mst_weight = 0
    edges_count = 0

    for edge in sorted_edges:
        u, v, w = edge
        if union(parent, rank, u, v):
            mst_weight += w
            edges_count += 1
            # 没有提前终止 - 低效

    return mst_weight
"""


def create_test_cases():
    """创建多层次的测试用例"""
    return [
        # 基础测试
        ((3, [(0, 1, 1), (1, 2, 2), (0, 2, 3)]), 3),
        ((4, [(0, 1, 1), (1, 2, 2), (2, 3, 3)]), 6),
        ((2, [(0, 1, 10)]), 10),

        # 带环的图
        ((4, [(0, 1, 1), (1, 2, 2), (2, 3, 3), (0, 2, 10)]), 6),

        # 完全图 K4
        ((4, [(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 4), (1, 3, 5), (2, 3, 6)]), 6),

        # 更复杂的图
        ((5, [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 4, 4), (0, 4, 10), (1, 3, 5)]), 10),

        # 边权重相同
        ((3, [(0, 1, 5), (1, 2, 5), (0, 2, 5)]), 10),

        # 较大图
        ((10, [
            (0, 1, 4), (0, 7, 8), (1, 2, 8), (1, 7, 11),
            (2, 3, 7), (2, 8, 2), (2, 5, 4), (3, 4, 9),
            (3, 5, 14), (4, 5, 10), (5, 6, 2), (6, 7, 1),
            (6, 8, 6), (7, 8, 7)
        ]), 37),
    ]


class MSTEvaluator:
    """MST 专用评估器：同时考虑正确性和性能。"""

    def __init__(self, test_cases, baseline_time=0.1):
        self.test_cases = test_cases
        self.baseline_time = baseline_time  # 基准时间（秒）

    def evaluate(self, code):
        """评估代码：正确性 + 性能。"""
        import time as time_module

        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return EvaluatorResult(
                fitness=0.0,
                passed=False,
                metrics={"error": 1.0},
                feedback=f"Execution error: {e}",
                execution_time=0.0
            )

        if "mst_kruskal" not in namespace:
            return EvaluatorResult(
                fitness=0.0,
                passed=False,
                metrics={"missing_function": 1.0},
                feedback="Function 'mst_kruskal' not found",
                execution_time=0.0
            )

        func = namespace["mst_kruskal"]

        # 1. 正确性测试
        passed = 0
        for test_input, expected in self.test_cases:
            try:
                result = func(*test_input)
                if result == expected:
                    passed += 1
            except:
                pass

        correctness_score = passed / len(self.test_cases)

        # 2. 性能测试 - 使用更大的图
        performance_score = 0.0
        if correctness_score == 1.0:
            # 创建更大的测试图
            large_graph = (
                20,
                [(i, i+1, i+1) for i in range(19)] +
                [(0, 19, 100)] +
                [(i, j, abs(i-j)*2) for i in range(20) for j in range(i+2, min(i+5, 20))]
            )

            # 多次运行取平均
            times = []
            for _ in range(5):
                start = time_module.perf_counter()
                try:
                    func(large_graph[0], large_graph[1])
                except:
                    pass
                times.append(time_module.perf_counter() - start)

            avg_time = min(times) if times else float('inf')

            # 性能分数：比基准快则高分
            if avg_time > 0:
                performance_score = min(1.0, self.baseline_time / avg_time)
            else:
                performance_score = 1.0

        # 综合分数：正确性为主，性能为辅
        # 如果正确性不是 100%，性能再好也没用
        # 如果正确性 100%，性能决定最终分数
        if correctness_score < 1.0:
            fitness = correctness_score * 100  # 最多 100 分
        else:
            # 正确性满分后，性能加分（最多 50 分额外加分）
            fitness = 100 + performance_score * 50

        return EvaluatorResult(
            fitness=fitness,
            passed=correctness_score == 1.0,
            metrics={
                "correctness": correctness_score,
                "performance": performance_score,
                "passed_tests": passed,
                "total_tests": len(self.test_cases)
            },
            feedback=f"Correct: {passed}/{len(self.test_cases)}, Performance: {performance_score:.2f}",
            execution_time=0.0
        )

    @property
    def name(self):
        return "MSTEvaluator"


def save_evolution_history(history, output_dir="mst_evolution_output"):
    """保存进化历史到文件"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 保存完整历史
    history_file = Path(output_dir) / "evolution_history.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info(f"进化历史已保存到：{history_file}")

    # 保存每一代的最优代码
    code_dir = Path(output_dir) / "best_codes"
    code_dir.mkdir(exist_ok=True)

    for gen_data in history:
        gen = gen_data['generation']
        code = gen_data.get('best_code', '')
        if code:
            code_file = code_dir / f"gen_{gen:04d}.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(f"# Generation {gen}\n")
                f.write(f"# Fitness: {gen_data['best_fitness']:.2f}\n\n")
                f.write(code)

    logger.info(f"每代最优代码已保存到：{code_dir}")


def run_evolution(generations=1000, population_size=30, save_interval=50):
    """
    运行 MST 算法进化

    Args:
        generations: 进化代数
        population_size: 种群大小
        save_interval: 保存间隔
    """
    logger.info("=" * 80)
    logger.info("AlphaEvolve: MST Algorithm - Intensive Evolution")
    logger.info("=" * 80)

    start_time = time.time()

    # 创建测试用例
    test_cases = create_test_cases()
    logger.info(f"测试用例数量：{len(test_cases)}")

    # 验证种子代码
    logger.info("\n种子代码预览:")
    logger.info("-" * 40)
    logger.info(SEED_CODE[:300] + "...")

    # 配置进化
    config = EvolutionConfig(
        population_size=population_size,
        max_generations=generations,
        elitism_count=max(3, population_size // 10),
        mutation_rate=0.8,
        seed=42,
    )

    # 创建评估器 - 使用新的 MST 专用评估器（包含性能评分）
    evaluator = MSTEvaluator(
        test_cases=test_cases,
        baseline_time=0.05  # 基准时间 50ms
    )

    # 创建变异引擎
    engine = MutationEngine()

    # 创建进化实例
    evolution = Evolution(
        config=config,
        mutation_engine=engine,
        evaluator=evaluator,
        seed_code=SEED_CODE,
    )

    logger.info("\n" + "=" * 80)
    logger.info("进化配置:")
    logger.info(f"  种群大小：{config.population_size}")
    logger.info(f"  最大代数：{config.max_generations}")
    logger.info(f"  精英数量：{config.elitism_count}")
    logger.info(f"  变异率：{config.mutation_rate}")
    logger.info("=" * 80)

    # 运行进化并记录历史
    logger.info("\n开始进化...\n")

    # 首先初始化种群
    evolution.initialize()

    history = []
    best_ever_fitness = 0
    best_ever_code = SEED_CODE

    # 记录初始状态
    best_ind = evolution.population.get_best(1)[0] if evolution.population.get_best(1) else None
    if best_ind:
        logger.info(f"初始种群 - Best={best_ind.fitness:.2f}, Pop={evolution.population.size()}")
        best_ever_fitness = best_ind.fitness
        best_ever_code = best_ind.code

    # 使用 evolve() 方法运行完整进化过程
    # 该方法会 internally 调用 _run_generation() 并记录历史
    for gen in range(1, generations + 1):
        evolution.generation = gen

        # 检查超时
        elapsed = time.time() - start_time
        if elapsed > config.timeout_seconds:
            logger.info(f"Timeout reached after {elapsed:.1f}s")
            break

        # 运行一代
        evolution._run_generation()

        # 获取当前代的结果
        best_ind = evolution.population.get_best(1)[0] if evolution.population.get_best(1) else None
        avg_fitness = evolution.population.avg_fitness_history[-1] if evolution.population.avg_fitness_history else 0.0
        pop_size = evolution.population.size()

        if best_ind:
            best_fitness = best_ind.fitness
            best_code = best_ind.code

            # 记录历史
            history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'population_size': pop_size,
                'best_code': best_code,
            })

            # 更新最优
            if best_fitness > best_ever_fitness:
                best_ever_fitness = best_fitness
                best_ever_code = best_code

            # 定期输出进度
            if gen % 10 == 0 or gen == 1:
                elapsed = time.time() - start_time
                eta = (elapsed / gen) * (generations - gen) / 60 if gen < generations else 0
                logger.info(
                    f"[Gen {gen:4d}/{generations}] "
                    f"Best={best_fitness:.2f} | "
                    f"Avg={avg_fitness:.2f} | "
                    f"Pop={pop_size} | "
                    f"Elapsed={elapsed:.1f}s | "
                    f"ETA≈{eta:.1f}min"
                )

            # 定期保存
            if gen % save_interval == 0:
                save_evolution_history(history, f"mst_evolution_output/checkpoint_{gen}")

        # 检查收敛
        if evolution.population.is_converged():
            logger.info(f"Converged at generation {gen}")
            # 收敛时不立即退出，继续观察

    # 完成
    total_time = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info("进化完成!")
    logger.info("=" * 80)
    logger.info(f"总耗时：{total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    logger.info(f"最优适应度：{best_ever_fitness:.2f}")
    logger.info(f"最优代码:\n{best_ever_code}")

    # 保存最终结果
    output_dir = "mst_evolution_output/final"
    save_evolution_history(history, output_dir)

    # 保存最终报告
    report_file = Path(output_dir) / "report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("MST 算法进化报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总代数：{generations}\n")
        f.write(f"种群大小：{population_size}\n")
        f.write(f"总耗时：{total_time:.1f}秒\n\n")
        f.write(f"最优适应度：{best_ever_fitness:.2f}\n\n")
        f.write("最优代码:\n")
        f.write("-" * 40 + "\n")
        f.write(best_ever_code)

    logger.info(f"\n最终报告已保存到：{report_file}")

    # 与参考实现对比
    logger.info("\n" + "=" * 80)
    logger.info("参考最优实现:")
    logger.info("=" * 80)
    logger.info(OPTIMAL_REFERENCE)

    return best_ever_code, best_ever_fitness, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MST Algorithm Evolution")
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="进化代数 (默认：100)"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=20,
        help="种群大小 (默认：20)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=25,
        help="保存间隔 (默认：25)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式 (50 代，小种群)"
    )

    args = parser.parse_args()

    if args.quick:
        run_evolution(generations=50, population_size=15, save_interval=10)
    else:
        run_evolution(
            generations=args.generations,
            population_size=args.population,
            save_interval=args.save_interval
        )
