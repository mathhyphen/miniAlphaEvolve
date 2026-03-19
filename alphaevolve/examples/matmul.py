"""Example: Discovering optimized matrix multiplication."""

import logging
import time
import random
from typing import List

from alphaevolve.core.data_structures import EvolutionConfig
from alphaevolve.core.mutation import MutationEngine
from alphaevolve.core.evaluator import (
    UnitTestEvaluator,
    PerformanceEvaluator,
    CompositeEvaluator,
)
from alphaevolve.core.evolution import Evolution

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Naive O(n³) matrix multiplication
NAIVE_MATMUL = '''
def matmul(A, B):
    """Naive matrix multiplication O(n³)."""
    n = len(A)
    m = len(B[0])
    p = len(B)

    # Initialize result matrix
    C = [[0 for _ in range(m)] for _ in range(n)]

    # Triple nested loop
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]

    return C
'''


def reference_matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Reference implementation using numpy if available."""
    try:
        import numpy as np
        return np.matmul(A, B).tolist()
    except ImportError:
        # Fallback naive implementation
        n = len(A)
        m = len(B[0])
        p = len(B)
        C = [[0.0 for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C[i][j] += A[i][k] * B[k][j]
        return C


def create_test_matrices(size: int = 10) -> tuple:
    """Create random test matrices."""
    A = [[random.random() for _ in range(size)] for _ in range(size)]
    B = [[random.random() for _ in range(size)] for _ in range(size)]
    return A, B


def benchmark_matmul(func):
    """Benchmark matrix multiplication."""
    total_time = 0.0

    # Test different sizes
    for size in [10, 20, 30]:
        A, B = create_test_matrices(size)

        start = time.time()
        result = func(A, B)
        elapsed = time.time() - start

        # Verify with reference
        expected = reference_matmul(A, B)
        if not matrices_equal(result, expected):
            return float('inf')

        total_time += elapsed

    return total_time


def matrices_equal(A, B, tolerance=1e-6):
    """Check if two matrices are equal within tolerance."""
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        return False
    for i in range(len(A)):
        for j in range(len(A[0])):
            if abs(A[i][j] - B[i][j]) > tolerance:
                return False
    return True


def create_evaluator():
    """Create evaluator for matrix multiplication."""
    # Test cases
    test_cases = [
        # 2x2 matrices
        (
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[19, 22], [43, 50]]
        ),
        # 3x3 identity
        (
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ),
        # 1x1
        (
            [[[5]], [[7]]],
            [[35]]
        ),
    ]

    # Correctness tests (flattened for evaluator)
    def test_matmul_wrapper(args):
        A, B = args
        namespace = {}
        exec(NAIVE_MATMUL, namespace)
        matmul = namespace['matmul']
        return matmul(A, B)

    # Use custom evaluation
    correctness_eval = UnitTestEvaluator(
        test_cases=[
            ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[19, 22], [43, 50]]),
            ([[[1, 0], [0, 1]], [[5, 6], [7, 8]]], [[5, 6], [7, 8]]),
        ],
        function_name="matmul",
        partial_credit=False
    )

    # Performance baseline
    namespace = {}
    exec(NAIVE_MATMUL, namespace)
    naive_func = namespace['matmul']
    baseline = benchmark_matmul(naive_func)
    logger.info(f"Baseline time: {baseline:.4f}s")

    performance_eval = PerformanceEvaluator(
        benchmark_func=benchmark_matmul,
        baseline_time=baseline
    )

    return CompositeEvaluator([
        (correctness_eval, 0.6),
        (performance_eval, 0.4),
    ])


def main():
    """Run matrix multiplication evolution."""
    logger.info("=" * 60)
    logger.info("AlphaEvolve: Matrix Multiplication Discovery")
    logger.info("=" * 60)

    config = EvolutionConfig(
        population_size=15,
        max_generations=15,
        mutation_rate=0.6,
        selection_pressure=3,
        elitism_count=2,
        seed=42,
        timeout_seconds=300
    )

    mutation_engine = MutationEngine()
    evaluator = create_evaluator()

    evolution = Evolution(
        config=config,
        mutation_engine=mutation_engine,
        evaluator=evaluator,
        seed_code=NAIVE_MATMUL,
        seed_name="matmul"
    )

    logger.info("\nStarting evolution...")
    best = evolution.evolve()

    logger.info("\n" + "=" * 60)
    logger.info("BEST SOLUTION FOUND")
    logger.info("=" * 60)
    logger.info(f"Fitness: {best.fitness:.2f}")
    logger.info(f"\n{best.code}")

    # Validate
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION")
    logger.info("=" * 60)

    namespace = {}
    exec(best.code, namespace)
    matmul = namespace['matmul']

    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    result = matmul(A, B)
    expected = [[19, 22], [43, 50]]

    logger.info(f"A = {A}")
    logger.info(f"B = {B}")
    logger.info(f"Result = {result}")
    logger.info(f"Expected = {expected}")
    logger.info(f"Correct: {result == expected}")

    return best


if __name__ == "__main__":
    main()
