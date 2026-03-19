"""Example: Evolving a better sorting algorithm with AlphaEvolve."""

import logging
import time
from typing import List

from alphaevolve.core.data_structures import EvolutionConfig
from alphaevolve.core.mutation import MutationEngine
from alphaevolve.core.evaluator import (
    UnitTestEvaluator,
    PerformanceEvaluator,
    CompositeEvaluator,
)
from alphaevolve.core.evolution import Evolution

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Seed implementation - naive bubble sort
def bubble_sort(arr: List[int]) -> List[int]:
    """Naive bubble sort implementation."""
    n = len(arr)
    result = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result


SEED_CODE = '''
def bubble_sort(arr):
    """Naive bubble sort implementation."""
    n = len(arr)
    result = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result
'''


def benchmark_sort(func):
    """Benchmark a sorting function."""
    import random

    # Generate test data
    sizes = [100, 500, 1000]
    total_time = 0.0

    for size in sizes:
        data = [random.randint(0, 10000) for _ in range(size)]
        start = time.time()
        result = func(data)
        elapsed = time.time() - start
        total_time += elapsed

        # Verify sorted
        if result != sorted(data):
            return float('inf')  # Failed correctness

    return total_time


def create_evaluator():
    """Create evaluator for sorting task."""
    # Correctness tests
    test_cases = [
        ([3, 1, 4, 1, 5], [1, 1, 3, 4, 5]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([1], [1]),
        ([], []),
        ([2, 2, 2], [2, 2, 2]),
        ([1, 2, 3], [1, 2, 3]),
    ]

    correctness_eval = UnitTestEvaluator(
        test_cases=test_cases,
        function_name="bubble_sort",
        partial_credit=True
    )

    # Baseline for performance
    baseline_time = benchmark_sort(bubble_sort)
    logger.info(f"Baseline time: {baseline_time:.4f}s")

    performance_eval = PerformanceEvaluator(
        benchmark_func=benchmark_sort,
        baseline_time=baseline_time
    )

    # Combined evaluator
    return CompositeEvaluator([
        (correctness_eval, 0.7),  # 70% weight on correctness
        (performance_eval, 0.3),  # 30% weight on performance
    ])


def main():
    """Run AlphaEvolve to improve sorting algorithm."""
    logger.info("=" * 60)
    logger.info("AlphaEvolve: Sorting Algorithm Discovery")
    logger.info("=" * 60)

    # Configuration
    config = EvolutionConfig(
        population_size=20,
        max_generations=10,
        mutation_rate=0.7,
        selection_pressure=3,
        elitism_count=2,
        seed=42
    )

    # Create components
    mutation_engine = MutationEngine()
    evaluator = create_evaluator()

    # Create evolution
    evolution = Evolution(
        config=config,
        mutation_engine=mutation_engine,
        evaluator=evaluator,
        seed_code=SEED_CODE,
        seed_name="bubble_sort"
    )

    # Run evolution
    logger.info("\nStarting evolution...")
    best = evolution.evolve()

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("EVOLUTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nBest Fitness: {best.fitness:.2f}")
    logger.info(f"Generation: {best.generation}")
    logger.info(f"\nBest Solution:\n{best.code}")

    # Show progression
    logger.info("\n" + "=" * 60)
    logger.info("EVOLUTION PROGRESS")
    logger.info("=" * 60)
    for record in evolution.get_history():
        logger.info(
            f"Gen {record['generation']:3d}: "
            f"Best={record['best_fitness']:.2f}, "
            f"Avg={record['avg_fitness']:.2f}, "
            f"Diversity={record['diversity']:.2f}"
        )

    # Test final solution
    logger.info("\n" + "=" * 60)
    logger.info("FINAL VALIDATION")
    logger.info("=" * 60)

    # Execute best code
    namespace = {}
    exec(best.code, namespace)
    evolved_sort = namespace['bubble_sort']

    test_data = [64, 34, 25, 12, 22, 11, 90]
    result = evolved_sort(test_data)
    logger.info(f"Input:  {test_data}")
    logger.info(f"Output: {result}")
    logger.info(f"Expected: {sorted(test_data)}")
    logger.info(f"Correct: {result == sorted(test_data)}")

    return best


if __name__ == "__main__":
    main()
