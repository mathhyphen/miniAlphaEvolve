"""Matrix Multiplication Problem for AlphaEvolve RL.

This module defines the 4x4 matrix multiplication problem for
evolutionary optimization using the AlphaEvolve RL system.

The goal is to discover faster algorithms for 4x4 matrix multiplication
beyond the standard 64 multiplications.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Any

# =============================================================================
# Problem Definition
# =============================================================================

# Baseline matrix multiplication code
BASELINE_CODE = """def matmul(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]
    return C
"""

# Unrolled baseline for reference (currently fastest pure Python)
UNROLLED_CODE = """def matmul(A, B):
    a0, a1, a2, a3 = A[0], A[1], A[2], A[3]
    b0, b1, b2, b3 = B[0], B[1], B[2], B[3]
    return [
        [a0[0]*b0[0] + a0[1]*b1[0] + a0[2]*b2[0] + a0[3]*b3[0],
         a0[0]*b0[1] + a0[1]*b1[1] + a0[2]*b2[1] + a0[3]*b3[1],
         a0[0]*b0[2] + a0[1]*b1[2] + a0[2]*b2[2] + a0[3]*b3[2],
         a0[0]*b0[3] + a0[1]*b1[3] + a0[2]*b2[3] + a0[3]*b3[3]],
        [a1[0]*b0[0] + a1[1]*b1[0] + a1[2]*b2[0] + a1[3]*b3[0],
         a1[0]*b0[1] + a1[1]*b1[1] + a1[2]*b2[1] + a1[3]*b3[1],
         a1[0]*b0[2] + a1[1]*b1[2] + a1[2]*b2[2] + a1[3]*b3[2],
         a1[0]*b0[3] + a1[1]*b1[3] + a1[2]*b2[3] + a1[3]*b3[3]],
        [a2[0]*b0[0] + a2[1]*b1[0] + a2[2]*b2[0] + a2[3]*b3[0],
         a2[0]*b0[1] + a2[1]*b1[1] + a2[2]*b2[1] + a2[3]*b3[1],
         a2[0]*b0[2] + a2[1]*b1[2] + a2[2]*b2[2] + a2[3]*b3[2],
         a2[0]*b0[3] + a2[1]*b1[3] + a2[2]*b2[3] + a2[3]*b3[3]],
        [a3[0]*b0[0] + a3[1]*b1[0] + a3[2]*b2[0] + a3[3]*b3[0],
         a3[0]*b0[1] + a3[1]*b1[1] + a3[2]*b2[1] + a3[3]*b3[1],
         a3[0]*b0[2] + a3[1]*b1[2] + a3[2]*b2[2] + a3[3]*b3[2],
         a3[0]*b0[3] + a3[1]*b1[3] + a3[2]*b2[3] + a3[3]*b3[3]]
    ]
"""


@dataclass(frozen=True)
class MatrixMultiplyCase:
    """A single test case for matrix multiplication."""
    case_id: str
    A: List[List[float]]
    B: List[List[float]]
    expected: List[List[float]]


@dataclass
class MatrixMultiplyProblem:
    """Matrix multiplication problem for AlphaEvolve.

    Attributes:
        initial_code: Starting code for evolution
        baseline_time_us: Baseline execution time in microseconds
        target_improvement: Target speedup factor
    """
    case_id: str = "matmul_4x4"
    description: str = "4x4 Matrix Multiplication Optimization"
    initial_code: str = field(default_factory=lambda: BASELINE_CODE)
    baseline_time_us: float = 10.0  # Estimated baseline time
    target_improvement: float = 2.0  # Target 2x speedup

    def get_test_cases(self) -> List[MatrixMultiplyCase]:
        """Get deterministic test cases."""
        # Fixed test cases for reproducibility
        test_cases = []

        # Identity matrix case
        I = [[float(i == j) for j in range(4)] for i in range(4)]
        test_cases.append(MatrixMultiplyCase(
            case_id="identity",
            A=I, B=I, expected=I
        ))

        # Specific matrix case for verification
        A1 = [[1.0, 2.0, 3.0, 4.0],
              [5.0, 6.0, 7.0, 8.0],
              [9.0, 10.0, 11.0, 12.0],
              [13.0, 14.0, 15.0, 16.0]]
        B1 = [[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]]
        test_cases.append(MatrixMultiplyCase(
            case_id="identity_transform",
            A=A1, B=B1, expected=A1
        ))

        # Random case 1
        random.seed(42)
        A2 = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]
        B2 = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]
        # Compute expected with baseline
        def naive_mul(A, B):
            C = [[0.0] * 4 for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        C[i][j] += A[i][k] * B[k][j]
            return C
        test_cases.append(MatrixMultiplyCase(
            case_id="random_42",
            A=A2, B=B2, expected=naive_mul(A2, B2)
        ))

        # Random case 2
        random.seed(123)
        A3 = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]
        B3 = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]
        test_cases.append(MatrixMultiplyCase(
            case_id="random_123",
            A=A3, B=B3, expected=naive_mul(A3, B3)
        ))

        return test_cases

    def verify(self, code: str) -> Tuple[bool, float, str]:
        """Verify generated code.

        Returns:
            (is_correct, execution_time_us, error_message)
        """
        test_cases = self.get_test_cases()

        try:
            # Compile the code
            namespace = {}
            exec(code, namespace)
            matmul_func = namespace.get('matmul')
            if matmul_func is None:
                return False, 0.0, "Function 'matmul' not found"

            # Run test cases
            total_time = 0.0
            for case in test_cases:
                start = time.perf_counter()
                result = matmul_func(case.A, case.B)
                elapsed = (time.perf_counter() - start) * 1_000_000
                total_time += elapsed

                # Check correctness
                for i in range(4):
                    for j in range(4):
                        if abs(result[i][j] - case.expected[i][j]) > 1e-9:
                            return False, total_time, f"Wrong result at [{i}][{j}]: got {result[i][j]}, expected {case.expected[i][j]}"

            avg_time = total_time / len(test_cases)
            return True, avg_time, ""

        except Exception as e:
            return False, 0.0, str(e)


# =============================================================================
# Reward Function
# =============================================================================

def compute_reward(
    is_correct: bool,
    execution_time_us: float,
    baseline_time_us: float = 10.0
) -> float:
    """Compute reward based on correctness and speed.

    Args:
        is_correct: Whether the solution is correct
        execution_time_us: Execution time in microseconds
        baseline_time_us: Baseline execution time

    Returns:
        Reward value (higher is better)
    """
    if not is_correct:
        return -1.0

    # Speedup ratio
    speedup = baseline_time_us / max(execution_time_us, 0.001)

    # Reward structure:
    # - Correct and faster than baseline: positive reward
    # - Correct but slower: small positive or zero
    # - Incorrect: negative reward
    if speedup >= 1.0:
        # Faster than baseline: reward = speedup^2 for exponential bonus
        return speedup ** 2
    else:
        # Correct but slower: small positive
        return 0.1 * speedup


# =============================================================================
# Feature Extraction for RL
# =============================================================================

def extract_features(code: str) -> List[float]:
    """Extract features from code for policy network.

    Returns:
        Feature vector normalized to [0, 1]
    """
    features = []

    # Code structure features
    features.append(len(code.splitlines()) / 50.0)  # Line count
    features.append(1.0 if "for" in code else 0.0)  # Has loops
    features.append(code.count("for") / 10.0)  # Loop count
    features.append(1.0 if "*=" in code else 0.0)  # Uses compound mult
    features.append(1.0 if "+=" in code else 0.0)  # Uses compound add
    features.append(1.0 if "zip" in code else 0.0)  # Uses zip
    features.append(1.0 if "sum" in code else 0.0)  # Uses sum
    features.append(1.0 if "[" in code and "]" in code else 0.0)  # Uses indexing

    # Multiplication patterns
    features.append(code.count("*") / 20.0)  # Multiplication count

    # Padding
    while len(features) < 16:
        features.append(0.0)

    return features[:16]


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    problem = MatrixMultiplyProblem()

    print("Matrix Multiply Problem Test")
    print("=" * 50)
    print(f"Case ID: {problem.case_id}")
    print(f"Description: {problem.description}")
    print(f"Initial code:\n{problem.initial_code}")

    # Test verification with baseline
    print("\nVerifying baseline code...")
    is_correct, exec_time, error = problem.verify(problem.initial_code)
    print(f"Correct: {is_correct}, Time: {exec_time:.2f}us, Error: {error}")

    # Test with unrolled version
    print("\nVerifying unrolled code...")
    is_correct, exec_time, error = problem.verify(UNROLLED_CODE)
    print(f"Correct: {is_correct}, Time: {exec_time:.2f}us, Error: {error}")

    # Test cases
    print("\nTest cases:")
    for case in problem.get_test_cases():
        print(f"  {case.case_id}: A={case.A[0][0]:.1f}..., B={case.B[0][0]:.1f}...")
