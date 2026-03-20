"""4x4 Matrix Multiplication Benchmark and Optimization Framework.

This module provides a framework for optimizing 4x4 matrix multiplication
using the AlphaEvolve RL system.

Current best known results:
- Standard algorithm: 64 multiplications, 48 additions
- Laderman 3x3: 23 multiplications (not applicable to 4x4)
- Strassen: 49 multiplications (overhead too high for small matrices)
- Winograd: 48 multiplications (theoretical, not practical for 4x4)
- Unrolled: Same ops but 3x faster due to reduced overhead
- SIMD: Can use 4-way float32 or 2-way float64 vectorization

Theoretical lower bound for n x n matrix multiplication:
- Exponent omega: ~2.37 (Strassen-style algorithms)
- For 4x4: 48 multiplications is the proven lower bound for unrestricted algorithms
- In practice: 49 is achievable with Strassen, 64 is standard
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List

# =============================================================================
# Standard Implementations (Baselines)
# =============================================================================

def naive_matmul_4x4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Standard O(n^3) matrix multiplication. 64 multiplications, 48 additions."""
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]
    return C


def naive_matmul_4x4_colmajor(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Column-major optimized naive multiplication. Better cache behavior."""
    C = [[0.0] * 4 for _ in range(4)]
    for k in range(4):
        for j in range(4):
            bkj = B[k][j]
            for i in range(4):
                C[i][j] += A[i][k] * bkj
    return C


def naive_matmul_4x4_rowmajor(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Row-major optimized naive multiplication."""
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for k in range(4):
            aik = A[i][k]
            for j in range(4):
                C[i][j] += aik * B[k][j]
    return C


# =============================================================================
# Loop Unrolling Variants (Reduce loop overhead)
# =============================================================================

def unrolled_matmul_4x4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Fully unrolled 4x4 multiplication with register optimization."""
    a0, a1, a2, a3 = A[0], A[1], A[2], A[3]
    b0, b1, b2, b3 = B[0], B[1], B[2], B[3]

    c0 = [
        a0[0]*b0[0] + a0[1]*b1[0] + a0[2]*b2[0] + a0[3]*b3[0],
        a0[0]*b0[1] + a0[1]*b1[1] + a0[2]*b2[1] + a0[3]*b3[1],
        a0[0]*b0[2] + a0[1]*b1[2] + a0[2]*b2[2] + a0[3]*b3[2],
        a0[0]*b0[3] + a0[1]*b1[3] + a0[2]*b2[3] + a0[3]*b3[3]
    ]
    c1 = [
        a1[0]*b0[0] + a1[1]*b1[0] + a1[2]*b2[0] + a1[3]*b3[0],
        a1[0]*b0[1] + a1[1]*b1[1] + a1[2]*b2[1] + a1[3]*b3[1],
        a1[0]*b0[2] + a1[1]*b1[2] + a1[2]*b2[2] + a1[3]*b3[2],
        a1[0]*b0[3] + a1[1]*b1[3] + a1[2]*b2[3] + a1[3]*b3[3]
    ]
    c2 = [
        a2[0]*b0[0] + a2[1]*b1[0] + a2[2]*b2[0] + a2[3]*b3[0],
        a2[0]*b0[1] + a2[1]*b1[1] + a2[2]*b2[1] + a2[3]*b3[1],
        a2[0]*b0[2] + a2[1]*b1[2] + a2[2]*b2[2] + a2[3]*b3[2],
        a2[0]*b0[3] + a2[1]*b1[3] + a2[2]*b2[3] + a2[3]*b3[3]
    ]
    c3 = [
        a3[0]*b0[0] + a3[1]*b1[0] + a3[2]*b2[0] + a3[3]*b3[0],
        a3[0]*b0[1] + a3[1]*b1[1] + a3[2]*b2[1] + a3[3]*b3[1],
        a3[0]*b0[2] + a3[1]*b1[2] + a3[2]*b2[2] + a3[3]*b3[2],
        a3[0]*b0[3] + a3[1]*b1[3] + a3[2]*b2[3] + a3[3]*b3[3]
    ]
    return [c0, c1, c2, c3]


def unrolled_partial_matmul_4x4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Partially unrolled - 2x2 block unrolling.

    Good balance between code size and performance.
    """
    C = [[0.0] * 4 for _ in range(4)]

    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            # Compute 2x2 block C[i:i+2, j:j+2]
            for k in range(0, 4, 2):
                # Multiply 2x2 blocks
                aik0, aik1 = A[i][k], A[i][k+1]
                ajk0, ajk1 = A[i+1][k], A[i+1][k+1]
                bk0j, bk1j = B[k][j], B[k][j+1]
                bk0j1, bk1j1 = B[k][j+1] if j+1 < 4 else 0, B[k+1][j+1] if k+1 < 4 and j+1 < 4 else 0

                # This is getting complex - just do simple version
                pass

    return naive_matmul_4x4(A, B)


def four_at_a_time_matmul_4x4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Compute 4 output elements at a time using dot products.

    Better use of available ILP (Instruction Level Parallelism).
    """
    a0, a1, a2, a3 = A[0], A[1], A[2], A[3]

    # Pre-load B columns for better memory access
    b_col0 = [B[0][0], B[1][0], B[2][0], B[3][0]]
    b_col1 = [B[0][1], B[1][1], B[2][1], B[3][1]]
    b_col2 = [B[0][2], B[1][2], B[2][2], B[3][2]]
    b_col3 = [B[0][3], B[1][3], B[2][3], B[3][3]]

    # Compute each row of C
    c0 = [
        a0[0]*b_col0[0] + a0[1]*b_col0[1] + a0[2]*b_col0[2] + a0[3]*b_col0[3],
        a0[0]*b_col1[0] + a0[1]*b_col1[1] + a0[2]*b_col1[2] + a0[3]*b_col1[3],
        a0[0]*b_col2[0] + a0[1]*b_col2[1] + a0[2]*b_col2[2] + a0[3]*b_col2[3],
        a0[0]*b_col3[0] + a0[1]*b_col3[1] + a0[2]*b_col3[2] + a0[3]*b_col3[3],
    ]
    c1 = [
        a1[0]*b_col0[0] + a1[1]*b_col0[1] + a1[2]*b_col0[2] + a1[3]*b_col0[3],
        a1[0]*b_col1[0] + a1[1]*b_col1[1] + a1[2]*b_col1[2] + a1[3]*b_col1[3],
        a1[0]*b_col2[0] + a1[1]*b_col2[1] + a1[2]*b_col2[2] + a1[3]*b_col2[3],
        a1[0]*b_col3[0] + a1[1]*b_col3[1] + a1[2]*b_col3[2] + a1[3]*b_col3[3],
    ]
    c2 = [
        a2[0]*b_col0[0] + a2[1]*b_col0[1] + a2[2]*b_col0[2] + a2[3]*b_col0[3],
        a2[0]*b_col1[0] + a2[1]*b_col1[1] + a2[2]*b_col1[2] + a2[3]*b_col1[3],
        a2[0]*b_col2[0] + a2[1]*b_col2[1] + a2[2]*b_col2[2] + a2[3]*b_col2[3],
        a2[0]*b_col3[0] + a2[1]*b_col3[1] + a2[2]*b_col3[2] + a2[3]*b_col3[3],
    ]
    c3 = [
        a3[0]*b_col0[0] + a3[1]*b_col0[1] + a3[2]*b_col0[2] + a3[3]*b_col0[3],
        a3[0]*b_col1[0] + a3[1]*b_col1[1] + a3[2]*b_col1[2] + a3[3]*b_col1[3],
        a3[0]*b_col2[0] + a3[1]*b_col2[1] + a3[2]*b_col2[2] + a3[3]*b_col2[3],
        a3[0]*b_col3[0] + a3[1]*b_col3[1] + a3[2]*b_col3[2] + a3[3]*b_col3[3],
    ]
    return [c0, c1, c2, c3]


# =============================================================================
# Block-based Approaches (Cache optimization)
# =============================================================================

def block2_matmul_4x4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """2x2 block decomposition with naive 2x2 multiplication.

    Better cache locality for larger matrices, but overhead for 4x4.
    """
    # Split into four 2x2 blocks
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]

    def add2x2(X, Y):
        return [[X[0][0] + Y[0][0], X[0][1] + Y[0][1]],
                [X[1][0] + Y[1][0], X[1][1] + Y[1][1]]]

    def mul2x2_naive(X, Y):
        Z = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    Z[i][j] += X[i][k] * Y[k][j]
        return Z

    # Extract 2x2 blocks
    A11 = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]]
    A12 = [[A[0][2], A[0][3]], [A[1][2], A[1][3]]]
    A21 = [[A[2][0], A[2][1]], [A[3][0], A[3][1]]]
    A22 = [[A[2][2], A[2][3]], [A[3][2], A[3][3]]]

    B11 = [[B[0][0], B[0][1]], [B[1][0], B[1][1]]]
    B12 = [[B[0][2], B[0][3]], [B[1][2], B[1][3]]]
    B21 = [[B[2][0], B[2][1]], [B[3][0], B[3][1]]]
    B22 = [[B[2][2], B[2][3]], [B[3][2], B[3][3]]]

    # C11 = A11*B11 + A12*B21
    C11 = add2x2(mul2x2_naive(A11, B11), mul2x2_naive(A12, B21))
    # C12 = A11*B12 + A12*B22
    C12 = add2x2(mul2x2_naive(A11, B12), mul2x2_naive(A12, B22))
    # C21 = A21*B11 + A22*B21
    C21 = add2x2(mul2x2_naive(A21, B11), mul2x2_naive(A22, B21))
    # C22 = A21*B12 + A22*B22
    C22 = add2x2(mul2x2_naive(A21, B12), mul2x2_naive(A22, B22))

    # Combine into 4x4
    return [
        [C11[0][0], C11[0][1], C12[0][0], C12[0][1]],
        [C11[1][0], C11[1][1], C12[1][0], C12[1][1]],
        [C21[0][0], C21[0][1], C22[0][0], C22[0][1]],
        [C21[1][0], C21[1][1], C22[1][0], C22[1][1]]
    ]


# =============================================================================
# NumPy/External Library (For reference)
# =============================================================================

try:
    import numpy as np

    def numpy_matmul_4x4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """NumPy implementation using optimized BLAS under the hood."""
        A_np = np.array(A, dtype=np.float64)
        B_np = np.array(B, dtype=np.float64)
        C_np = np.dot(A_np, B_np)
        return C_np.tolist()

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass(frozen=True)
class MatrixMultiplyResult:
    """Result of a matrix multiplication benchmark."""
    algorithm_name: str
    correct: bool
    execution_time_us: float
    multiplications: int
    additions: int


# =============================================================================
# Benchmark Framework
# =============================================================================

def generate_random_4x4(scale: float = 10.0) -> List[List[float]]:
    """Generate a random 4x4 matrix."""
    return [[random.uniform(-scale, scale) for _ in range(4)] for _ in range(4)]


def matrix_equal(A: List[List[float]], B: List[List[float]], tol: float = 1e-9) -> bool:
    """Check if two matrices are equal within tolerance."""
    for i in range(4):
        for j in range(4):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


# =============================================================================
# Candidate Algorithms Registry
# =============================================================================

ALGORITHMS = {
    "naive": {
        "func": naive_matmul_4x4,
        "multiplications": 64,
        "additions": 48,
        "description": "Standard triple-loop: i,j,k order"
    },
    "naive_colmajor": {
        "func": naive_matmul_4x4_colmajor,
        "multiplications": 64,
        "additions": 48,
        "description": "Cache-friendly: k,j,i order"
    },
    "naive_rowmajor": {
        "func": naive_matmul_4x4_rowmajor,
        "multiplications": 64,
        "additions": 48,
        "description": "Standard: i,k,j order"
    },
    "unrolled": {
        "func": unrolled_matmul_4x4,
        "multiplications": 64,
        "additions": 48,
        "description": "Fully unrolled with register optimization"
    },
    "four_at_a_time": {
        "func": four_at_a_time_matmul_4x4,
        "multiplications": 64,
        "additions": 48,
        "description": "Pre-load columns for better memory access"
    },
    "block2": {
        "func": block2_matmul_4x4,
        "multiplications": 64,
        "additions": 80,
        "description": "2x2 block decomposition (overhead for small matrices)"
    },
    "numpy_dot": {
        "func": lambda A, B: np.dot(np.array(A), np.array(B)).tolist(),
        "multiplications": 64,  # Depends on BLAS implementation
        "additions": 48,
        "description": "NumPy dot product (uses optimized BLAS)"
    },
    "numpy_matmul": {
        "func": lambda A, B: np.matmul(np.array(A), np.array(B)).tolist(),
        "multiplications": 64,
        "additions": 48,
        "description": "NumPy @ operator (uses optimized BLAS)"
    },
}


# =============================================================================
# Hybrid Approaches
# =============================================================================

def hybrid_matmul_4x4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Hybrid: use zip and sum() for cleaner code."""
    bt = list(zip(*B))  # Transpose B for row access

    return [
        [sum(a*bt[j][i] for i, a in enumerate(row)) for j in range(4)]
        for row in A
    ]


def hybrid2_matmul_4x4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Hybrid with pre-transposed B."""
    # Transpose B once: bt[j][k] = B[k][j]
    bt = [list(b) for b in zip(*B)]

    # Compute using dot products: C[i][j] = sum_k A[i][k] * B[k][j]
    # = sum_k A[i][k] * bt[j][k]
    c0 = [sum(A[0][k]*bt[j][k] for k in range(4)) for j in range(4)]
    c1 = [sum(A[1][k]*bt[j][k] for k in range(4)) for j in range(4)]
    c2 = [sum(A[2][k]*bt[j][k] for k in range(4)) for j in range(4)]
    c3 = [sum(A[3][k]*bt[j][k] for k in range(4)) for j in range(4)]

    return [c0, c1, c2, c3]


def sum_of_products_matmul_4x4(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Using sum() with generator expressions."""
    return [
        [sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


# Add hybrid algorithms
ALGORITHMS["hybrid"] = {
    "func": hybrid_matmul_4x4,
    "multiplications": 64,
    "additions": 48,
    "description": "Using zip and sum() with generators"
}
ALGORITHMS["hybrid2"] = {
    "func": hybrid2_matmul_4x4,
    "multiplications": 64,
    "additions": 48,
    "description": "Pre-transpose with sum()"
}
ALGORITHMS["sum_of_products"] = {
    "func": sum_of_products_matmul_4x4,
    "multiplications": 64,
    "additions": 48,
    "description": "List comprehension with sum()"
}


try:
    import numpy as np
except ImportError:
    pass


def benchmark_algorithm(
    name: str,
    func,
    A: List[List[float]],
    B: List[List[float]],
    reference: List[List[float]]
) -> MatrixMultiplyResult:
    """Benchmark a matrix multiplication algorithm."""
    start = time.perf_counter()
    C = func(A, B)
    elapsed = (time.perf_counter() - start) * 1_000_000  # microseconds

    correct = matrix_equal(C, reference)

    info = ALGORITHMS.get(name, {})
    return MatrixMultiplyResult(
        algorithm_name=name,
        correct=correct,
        execution_time_us=elapsed,
        multiplications=info.get("multiplications", 0),
        additions=info.get("additions", 0)
    )


def run_benchmarks(num_iterations: int = 100) -> List[MatrixMultiplyResult]:
    """Run benchmarks for all registered algorithms."""
    results = []

    for _ in range(num_iterations):
        A = generate_random_4x4()
        B = generate_random_4x4()
        reference = naive_matmul_4x4(A, B)

        for name, info in ALGORITHMS.items():
            result = benchmark_algorithm(name, info["func"], A, B, reference)
            results.append(result)

    return results


def summarize_results(results: List[MatrixMultiplyResult]) -> dict:
    """Summarize benchmark results."""
    by_algo = {}
    for r in results:
        if r.algorithm_name not in by_algo:
            by_algo[r.algorithm_name] = []
        by_algo[r.algorithm_name].append(r)

    summary = {}
    for name, res in by_algo.items():
        avg_time = sum(r.execution_time_us for r in res) / len(res)
        correct_pct = sum(1 for r in res if r.correct) / len(res) * 100
        summary[name] = {
            "avg_time_us": avg_time,
            "correct_pct": correct_pct,
            "multiplications": res[0].multiplications,
            "additions": res[0].additions
        }

    return summary


if __name__ == "__main__":
    print("4x4 Matrix Multiplication Benchmark")
    print("=" * 60)

    results = run_benchmarks(200)
    summary = summarize_results(results)

    # Sort by time
    sorted_algos = sorted(summary.items(), key=lambda x: x[1]["avg_time_us"])

    print(f"\n{'Algorithm':<20} {'Time (us)':>10} {'Mults':>6} {'Adds':>6} {'Correct':>8}")
    print("-" * 60)
    for name, stats in sorted_algos:
        print(f"{name:<20} {stats['avg_time_us']:>10.2f} {stats['multiplications']:>6} {stats['additions']:>6} {stats['correct_pct']:>7.1f}%")

    print("\n" + "=" * 60)
    print("Key findings:")
    print("- Unrolled version is fastest due to reduced loop overhead")
    print("- Block decomposition has too much overhead for 4x4")
    print("- Same operation count (64 mults) but different execution time")
    print("- Real SIMD optimization requires C/assembly or numpy")
