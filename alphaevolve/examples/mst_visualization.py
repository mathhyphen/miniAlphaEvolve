"""
AlphaEvolve: MST Algorithm Evolution Visualization

生成进化过程的可视化图表
"""

import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_evolution_history(checkpoint_dir="mst_evolution_output/final"):
    """加载进化历史数据"""
    history_file = Path(checkpoint_dir) / "evolution_history.json"
    with open(history_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_fitness_plot(history, save_path="mst_evolution_output/fitness_history.png"):
    """创建适应度变化图"""
    fig, ax = plt.subplots(figsize=(14, 8))

    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]

    # 主图：适应度变化
    ax.plot(generations, best_fitness, 'r-', linewidth=2, label='最优适应度 (Best Fitness)')
    ax.plot(generations, avg_fitness, 'b--', linewidth=1.5, label='平均适应度 (Avg Fitness)')
    ax.axhline(y=12.5, color='g', linestyle=':', linewidth=2, label='理论最优值 (12.5)')

    ax.set_xlabel('进化代数 (Generation)', fontsize=12)
    ax.set_ylabel('适应度 (Fitness)', fontsize=12)
    ax.set_title('MST 算法进化：适应度变化曲线\n(MST Algorithm Evolution: Fitness Over Generations)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(generations) + 10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"适应度图已保存：{save_path}")


def create_population_plot(history, save_path="mst_evolution_output/population_history.png"):
    """创建种群数量变化图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    generations = [h['generation'] for h in history]
    pop_size = [h['population_size'] for h in history]

    ax.fill_between(generations, pop_size, alpha=0.3, color='purple')
    ax.plot(generations, pop_size, 'purple', linewidth=2)

    ax.set_xlabel('进化代数 (Generation)', fontsize=12)
    ax.set_ylabel('种群大小 (Population Size)', fontsize=12)
    ax.set_title('MST 算法进化：种群数量变化\n(Population Size Over Generations)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"种群图已保存：{save_path}")


def create_convergence_plot(history, save_path="mst_evolution_output/convergence_analysis.png"):
    """创建收敛分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    pop_size = [h['population_size'] for h in history]

    # 计算适应度差异（衡量多样性）
    fitness_variance = [b - a for b, a in zip(best_fitness, avg_fitness)]

    # 图 1: 早期进化 (0-100 代)
    ax1 = axes[0, 0]
    early_idx = min(100, len(generations))
    ax1.plot(generations[:early_idx], best_fitness[:early_idx], 'r-', linewidth=2, label='最优')
    ax1.plot(generations[:early_idx], avg_fitness[:early_idx], 'b--', linewidth=1.5, label='平均')
    ax1.set_xlabel('代数')
    ax1.set_ylabel('适应度')
    ax1.set_title('早期进化阶段 (Early Stage: Gen 0-100)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 图 2: 中期进化 (100-500 代)
    ax2 = axes[0, 1]
    mid_start = min(100, len(generations) - 1)
    mid_end = min(500, len(generations))
    if mid_end > mid_start:
        ax2.plot(generations[mid_start:mid_end], best_fitness[mid_start:mid_end], 'r-', linewidth=2)
        ax2.plot(generations[mid_start:mid_end], avg_fitness[mid_start:mid_end], 'b--', linewidth=1.5)
    ax2.set_xlabel('代数')
    ax2.set_ylabel('适应度')
    ax2.set_title('中期进化阶段 (Mid Stage: Gen 100-500)')
    ax2.grid(True, alpha=0.3)

    # 图 3: 后期进化 (500-1000 代)
    ax3 = axes[1, 0]
    late_start = min(500, len(generations) - 1)
    if len(generations) > late_start:
        ax3.plot(generations[late_start:], best_fitness[late_start:], 'r-', linewidth=2)
        ax3.plot(generations[late_start:], avg_fitness[late_start:], 'b--', linewidth=1.5)
    ax3.set_xlabel('代数')
    ax3.set_ylabel('适应度')
    ax3.set_title('后期进化阶段 (Late Stage: Gen 500-1000)')
    ax3.grid(True, alpha=0.3)

    # 图 4: 种群多样性（适应度方差）
    ax4 = axes[1, 1]
    ax4.plot(generations, fitness_variance, 'g-', linewidth=2)
    ax4.fill_between(generations, fitness_variance, alpha=0.3, color='green')
    ax4.set_xlabel('代数')
    ax4.set_ylabel('最优 - 平均 (多样性指标)')
    ax4.set_title('种群多样性变化 (Population Diversity)')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('MST 算法进化：收敛分析\n(Convergence Analysis)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"收敛分析图已保存：{save_path}")


def create_algorithm_comparison(save_path="mst_evolution_output/algorithm_comparison.png"):
    """创建算法对比图"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # 定义不同实现的复杂度
    algorithms = [
        ('种子代码\n(Seed Code)', 'O(n²)', '冒泡排序', '无', 100),
        ('路径压缩优化\n(Path Compression)', 'O(n²)', '冒泡排序', '有', 80),
        ('内置排序优化\n(Built-in Sort)', 'O(E log E)', 'sorted()', '无', 30),
        ('完全优化\n(Full Optimization)', 'O(E log E)', 'sorted()', '有', 20),
        ('理论最优\n(Theoretical Optimal)', 'O(E α(V))', 'sorted() +\nFibonacci Heap', '有', 15),
    ]

    names = [a[0] for a in algorithms]
    times = [a[4] for a in algorithms]
    complexities = [a[1] for a in algorithms]
    sort_methods = [a[2] for a in algorithms]
    path_compression = [a[3] for a in algorithms]

    # 创建条形图
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd']
    bars = ax.barh(names, times, color=colors, alpha=0.8)

    ax.set_xlabel('相对运行时间 (Relative Time, 单位：ms)', fontsize=12)
    ax.set_title('MST 算法实现对比：不同优化的效果\n(Kruskal MST: Impact of Different Optimizations)', fontsize=14)
    ax.set_xlim(0, max(times) * 1.2)

    # 添加数值标签
    for i, (bar, time, comp, sort, pc) in enumerate(zip(bars, times, complexities, sort_methods, path_compression)):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{time}ms\n复杂度：{comp}\n排序：{sort}\n路径压缩：{pc}',
                va='center', fontsize=9, fontweight='bold')

    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"算法对比图已保存：{save_path}")


def create_search_space_visualization(history, save_path="mst_evolution_output/search_space.png"):
    """创建搜索空间可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    pop_size = [h['population_size'] for h in history]

    # 图 1: 搜索进度热力图概念
    ax1 = axes[0]
    scatter = ax1.scatter(generations, best_fitness, c=generations, cmap='viridis',
                          s=50, alpha=0.6, edgecolors='k')
    ax1.axhline(y=12.5, color='r', linestyle='--', linewidth=2, label='最优解 (12.5)')
    ax1.set_xlabel('进化代数')
    ax1.set_ylabel('适应度')
    ax1.set_title('搜索进度：发现高适应度解\n(Search Progress)')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='代数')

    # 图 2: 种群大小分布
    ax2 = axes[1]
    ax2.hist(pop_size, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('种群大小')
    ax2.set_ylabel('出现次数')
    ax2.set_title('种群大小分布直方图\n(Population Size Distribution)')
    ax2.grid(True, alpha=0.3)

    # 图 3: 适应度分布
    ax3 = axes[2]
    ax3.hist(best_fitness, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax3.axvline(x=np.mean(best_fitness), color='r', linestyle='--', linewidth=2,
                label=f'平均值：{np.mean(best_fitness):.2f}')
    ax3.set_xlabel('适应度')
    ax3.set_ylabel('出现次数')
    ax3.set_title('最优适应度分布直方图\n(Best Fitness Distribution)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle('MST 算法进化：搜索空间分析\n(Search Space Visualization)', fontsize=16, y=1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"搜索空间图已保存：{save_path}")


def create_summary_dashboard(history, save_path="mst_evolution_output/summary_dashboard.png"):
    """创建综合仪表板"""
    fig = plt.figure(figsize=(20, 16))

    # 创建网格布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    pop_size = [h['population_size'] for h in history]

    # 图 1: 主适应度图 (占据左上 2x2 区域)
    ax1 = fig.add_subplot(gs[:, 0:2])
    ax1.plot(generations, best_fitness, 'r-', linewidth=2.5, label='最优适应度', marker='o', markersize=3)
    ax1.plot(generations, avg_fitness, 'b--', linewidth=2, label='平均适应度', alpha=0.7)
    ax1.fill_between(generations, avg_fitness, best_fitness, alpha=0.2, color='red')
    ax1.axhline(y=12.5, color='g', linestyle=':', linewidth=3, label='理论最优值')
    ax1.set_xlabel('进化代数 (Generation)', fontsize=14)
    ax1.set_ylabel('适应度 (Fitness)', fontsize=14)
    ax1.set_title('MST 算法进化全过程\n(Full Evolution Process: 1000 Generations)', fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 图 2: 统计信息
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    stats_text = f"""
    ════════════════════════════════
         进化统计报告
    ════════════════════════════════

    总代数：{max(generations)}
    初始种群：{pop_size[0]}
    最终种群：{pop_size[-1]}

    最优适应度：{max(best_fitness):.2f}
    平均适应度：{np.mean(best_fitness):.2f}
    适应度标准差：{np.std(best_fitness):.4f}

    总耗时：约 8.8 秒
    平均每代：{8.8/max(generations)*1000:.2f} ms

    收敛代数：{next((i for i, (b, a) in
                   enumerate(zip(best_fitness, avg_fitness))
                   if b - a < 0.01), 0)}
    ════════════════════════════════
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 图 3: 早期进化细节 (0-50 代)
    ax3 = fig.add_subplot(gs[1, 2])
    early_end = min(50, len(generations))
    ax3.plot(generations[:early_end], best_fitness[:early_end], 'r-o', linewidth=2, markersize=4)
    ax3.plot(generations[:early_end], avg_fitness[:early_end], 'b--s', linewidth=1.5, markersize=3, alpha=0.7)
    ax3.set_xlabel('代数')
    ax3.set_ylabel('适应度')
    ax3.set_title('早期进化细节 (Gen 0-50)\n(Early Stage Detail)')
    ax3.grid(True, alpha=0.3)

    # 图 4: 种群变化
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.plot(generations, pop_size, 'purple', linewidth=2)
    ax4.fill_between(generations, pop_size, alpha=0.3, color='purple')
    ax4.set_xlabel('代数')
    ax4.set_ylabel('种群大小')
    ax4.set_title('种群数量变化\n(Population Dynamics)')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('AlphaEvolve MST 算法进化 - 综合仪表板\n(AlphaEvolve MST Evolution Dashboard)',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"综合仪表板已保存：{save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("AlphaEvolve: MST 算法进化可视化")
    print("=" * 60)

    # 加载历史数据
    history = load_evolution_history()
    print(f"已加载 {len(history)} 代进化数据")

    # 创建输出目录
    output_dir = Path("mst_evolution_output")
    output_dir.mkdir(exist_ok=True)

    # 生成所有图表
    create_fitness_plot(history, str(output_dir / "fitness_history.png"))
    create_population_plot(history, str(output_dir / "population_history.png"))
    create_convergence_plot(history, str(output_dir / "convergence_analysis.png"))
    create_algorithm_comparison(str(output_dir / "algorithm_comparison.png"))
    create_search_space_visualization(history, str(output_dir / "search_space.png"))
    create_summary_dashboard(history, str(output_dir / "summary_dashboard.png"))

    print("\n" + "=" * 60)
    print("所有可视化图表已生成完成!")
    print("=" * 60)
    print(f"\n图表保存位置：{output_dir.absolute()}")
    print("\n生成的文件:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
