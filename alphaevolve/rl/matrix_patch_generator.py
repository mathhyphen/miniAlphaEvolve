"""Matrix Multiplication Patch Generator for AlphaEvolve RL.

This module extends the base PatchGenerator with matrix-specific
mutation operations for optimizing 4x4 matrix multiplication.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch

from .patch_generator import Patch, PatchType


class MatrixOp(Enum):
    """Matrix-specific optimization operations."""
    # Loop transformations
    REORDER_LOOPS = "reorder_loops"  # Change loop order i,j,k -> i,k,j etc
    UNROLL_LOOP = "unroll_loop"      # Unroll a specific loop
    FUSE_LOOPS = "fuse_loops"        # Fuse adjacent loops

    # Memory access patterns
    PRELOAD_ROW = "preload_row"      # Pre-load A[i][:] into variable
    PRELOAD_COL = "preload_col"       # Pre-load B[:][j] into variable
    TRANSPOSE_B = "transpose_b"      # Use B.T for better cache access

    # Register optimization
    EXTRACT_DIAGONAL = "extract_diagonal"
    USE_SYMMETRY = "use_symmetry"

    # Algorithm substitution
    REPLACE_WITH_STRASSEN = "replace_with_strassen"
    REPLACE_WITH_WINOGRAD = "replace_with_winograd"

    # Expression optimization
    FACTOR_MULT = "factor_multiply"
    CANCEL_SUBS = "cancel_subtract"
    HoIST_INVARIANT = "hoist_invariant"


# Matrix multiplication loop order transformations
LOOP_ORDERINGS = [
    ("i", "j", "k"),  # Standard row-major
    ("i", "k", "j"),  # Better cache for A
    ("j", "i", "k"),  # Transpose access
    ("j", "k", "i"),  #
    ("k", "i", "j"),  # Column-major for A
    ("k", "j", "i"),  # Column-major for both
]


@dataclass
class MatrixPatch:
    """A patch specifically for matrix multiplication optimization."""
    op: MatrixOp
    description: str
    confidence: float = 0.5
    # For loop reordering
    new_loop_order: Optional[Tuple[str, str, str]] = None
    # For unrolling
    unroll_target: Optional[str] = None  # "i", "j", or "k"
    # For preloading
    preload_var: Optional[str] = None  # "a_row", "b_col"
    # For replacement
    replacement_code: Optional[str] = None


class MatrixPatchGenerator:
    """Generates matrix-specific optimization patches.

    This extends the base PatchGenerator with operations specifically
    designed for matrix multiplication optimization.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        num_actions: int = 64,
    ):
        """Initialize matrix patch generator.

        Args:
            temperature: Exploration temperature (higher = more random)
            num_actions: Number of discrete actions to choose from
        """
        self.temperature = temperature
        self.num_actions = num_actions
        self.device = torch.device("cpu")

    def generate_random_patch(self) -> MatrixPatch:
        """Generate a random matrix optimization patch."""
        op = random.choice(list(MatrixOp))
        return self._create_patch(op)

    def _create_patch(self, op: MatrixOp) -> MatrixPatch:
        """Create a patch for the given operation."""
        if op == MatrixOp.REORDER_LOOPS:
            new_order = random.choice(LOOP_ORDERINGS)
            return MatrixPatch(
                op=op,
                description=f"Reorder loops to {new_order}",
                new_loop_order=new_order,
                confidence=0.7,
            )

        elif op == MatrixOp.UNROLL_LOOP:
            target = random.choice(["i", "j", "k"])
            return MatrixPatch(
                op=op,
                description=f"Unroll loop {target}",
                unroll_target=target,
                confidence=0.6,
            )

        elif op == MatrixOp.PRELOAD_ROW:
            return MatrixPatch(
                op=op,
                description="Preload A[i][:] row into variable",
                preload_var="a_row",
                confidence=0.8,
            )

        elif op == MatrixOp.PRELOAD_COL:
            return MatrixPatch(
                op=op,
                description="Preload B[:][j] column into variable",
                preload_var="b_col",
                confidence=0.8,
            )

        elif op == MatrixOp.REPLACE_WITH_STRASSEN:
            return MatrixPatch(
                op=op,
                description="Replace with Strassen 2x2 block algorithm",
                replacement_code=self._get_strassen_code(),
                confidence=0.3,  # Low confidence - complex change
            )

        elif op == MatrixOp.TRANSPOSE_B:
            return MatrixPatch(
                op=op,
                description="Use transposed B for better access",
                confidence=0.5,
            )

        else:
            return MatrixPatch(
                op=op,
                description=f"Apply {op.value}",
                confidence=0.4,
            )

    def _get_strassen_code(self) -> str:
        """Get Strassen 4x4 implementation."""
        return '''def matmul(A, B):
    # 4x4 Strassen algorithm using 2x2 blocks
    def add2x2(X, Y):
        return [[X[0][0]+Y[0][0], X[0][1]+Y[0][1]], X[1][0]+Y[1][0], X[1][1]+Y[1][1]],
                [X[0][0]+Y[0][0], X[0][1]+Y[0][1]], X[1][0]+Y[1][0], X[1][1]+Y[1][1]]]

    def sub2x2(X, Y):
        return [[X[0][0]-Y[0][0], X[0][1]-Y[0][1]], [X[1][0]-Y[1][0], X[1][1]-Y[1][1]]]]

    def mul2x2_naive(X, Y):
        return [[X[0][0]*Y[0][0]+X[0][1]*Y[1][0], X[0][0]*Y[0][1]+X[0][1]*Y[1][1]],
                 [X[1][0]*Y[0][0]+X[1][1]*Y[1][0], X[1][0]*Y[0][1]+X[1][1]*Y[1][1]]]]

    # Split into 2x2 blocks
    A00 = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]]
    A01 = [[A[0][2], A[0][3]], [A[1][2], A[1][3]]]
    A10 = [[A[2][0], A[2][1]], [A[3][0], A[3][1]]]
    A11 = [[A[2][2], A[2][3]], [A[3][2], A[3][3]]]

    B00 = [[B[0][0], B[0][1]], [B[1][0], B[1][1]]]
    B01 = [[B[0][2], B[0][3]], [B[1][2], B[1][3]]]
    B10 = [[B[2][0], B[2][1]], [B[3][0], B[3][1]]]
    B11 = [[B[2][2], B[2][3]], [B[3][2], B[3][3]]]

    # Strassen for 2x2 (simplified - using naive for blocks)
    C00 = [[0,0],[0,0]]
    C01 = [[0,0],[0,0]]
    C10 = [[0,0],[0,0]]
    C11 = [[0,0],[0,0]]

    for i in range(2):
        for j in range(2):
            for k in range(2):
                C00[i][j] += A00[i][k] * B00[k][j]
                C01[i][j] += A01[i][k] * B01[k][j]
                C10[i][j] += A10[i][k] * B10[k][j]
                C11[i][j] += A11[i][k] * B11[k][j]

    # Combine blocks
    return [[C00[0][0], C00[0][1], C01[0][0], C01[0][1]],
            [C00[1][0], C00[1][1], C01[1][0], C01[1][1]],
            [C10[0][0], C10[0][1], C11[0][0], C11[0][1]],
            [C10[1][0], C10[1][1], C11[1][0], C11[1][1]]]
'''

    def apply_patch(self, code: str, patch: MatrixPatch) -> str:
        """Apply a matrix optimization patch to code."""
        if patch.op == MatrixOp.REORDER_LOOPS and patch.new_loop_order:
            return self._apply_loop_reorder(code, patch.new_loop_order)

        elif patch.op == MatrixOp.PRELOAD_ROW:
            return self._apply_preload_row(code)

        elif patch.op == MatrixOp.PRELOAD_COL:
            return self._apply_preload_col(code)

        elif patch.op == MatrixOp.REPLACE_WITH_STRASSEN and patch.replacement_code:
            return patch.replacement_code

        elif patch.op == MatrixOp.UNROLL_LOOP and patch.unroll_target:
            return self._apply_unroll(code, patch.unroll_target)

        return code

    def _apply_loop_reorder(self, code: str, new_order: Tuple[str, str, str]) -> str:
        """Apply loop reordering to matrix multiplication."""
        # Simple case: change i,j,k to i,k,j
        if new_order == ("i", "k", "j"):
            # Better for row-major A access
            old = '''for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]'''
            new = '''for i in range(4):
        for k in range(4):
            aik = A[i][k]
            for j in range(4):
                C[i][j] += aik * B[k][j]'''
            return code.replace(old, new)

        elif new_order == ("k", "j", "i"):
            # Column-major
            old = '''for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]'''
            new = '''for k in range(4):
        for j in range(4):
            for i in range(4):
                C[i][j] += A[i][k] * B[k][j]'''
            return code.replace(old, new)

        return code

    def _apply_preload_row(self, code: str) -> str:
        """Preload A[i][:] row."""
        old = '''for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]'''
        new = '''for i in range(4):
        a0, a1, a2, a3 = A[i][0], A[i][1], A[i][2], A[i][3]
        for j in range(4):
            C[i][j] = a0 * B[0][j] + a1 * B[1][j] + a2 * B[2][j] + a3 * B[3][j]'''
        return code.replace(old, new)

    def _apply_preload_col(self, code: str) -> str:
        """Preload B[:][j] column."""
        old = '''for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]'''
        new = '''for j in range(4):
        b0, b1, b2, b3 = B[0][j], B[1][j], B[2][j], B[3][j]
        for i in range(4):
            C[i][j] = A[i][0] * b0 + A[i][1] * b1 + A[i][2] * b2 + A[i][3] * b3'''
        return code.replace(old, new)

    def _apply_unroll(self, code: str, target: str) -> str:
        """Fully unroll a specific loop."""
        if target == "k":
            old = '''for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]'''
            new = '''for i in range(4):
        for j in range(4):
            C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] + A[i][3] * B[3][j]'''
            return code.replace(old, new)

        return code

    def generate_patches(self, num_patches: int = 5) -> List[MatrixPatch]:
        """Generate multiple patches for exploration."""
        patches = []
        for _ in range(num_patches):
            patch = self.generate_random_patch()
            patches.append(patch)
        return patches


# =============================================================================
# Evolution Strategy for Matrix Multiplication
# =============================================================================

class MatrixMultiplyEvolution:
    """Evolutionary optimization for matrix multiplication.

    Uses AlphaEvolve-style RL with matrix-specific mutations.
    """

    def __init__(
        self,
        baseline_code: str,
        baseline_time_us: float = 10.0,
        population_size: int = 20,
        elite_keep: int = 3,
    ):
        self.baseline_code = baseline_code
        self.baseline_time_us = baseline_time_us
        self.population_size = population_size
        self.elite_keep = elite_keep

        self.patch_gen = MatrixPatchGenerator()
        self.population: List[dict] = []
        self.best_code = baseline_code
        self.best_time = baseline_time_us
        self.best_speedup = 1.0

    def _evaluate(self, code: str) -> dict:
        """Evaluate a candidate algorithm."""
        try:
            namespace = {}
            exec(code, namespace)
            matmul = namespace.get('matmul')
            if matmul is None:
                return {'code': code, 'correct': False, 'time_us': float('inf')}

            # Run test cases
            import random
            random.seed(42)

            total_time = 0.0
            for _ in range(5):
                A = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]
                B = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]

                # Expected result with baseline
                def naive(A, B):
                    C = [[0.0] * 4 for _ in range(4)]
                    for i in range(4):
                        for j in range(4):
                            for k in range(4):
                                C[i][j] += A[i][k] * B[k][j]
                    return C

                expected = naive(A, B)

                import time
                start = time.perf_counter()
                result = matmul(A, B)
                elapsed = (time.perf_counter() - start) * 1_000_000

                # Verify correctness
                for i in range(4):
                    for j in range(4):
                        if abs(result[i][j] - expected[i][j]) > 1e-9:
                            return {'code': code, 'correct': False, 'time_us': elapsed}

                total_time += elapsed

            avg_time = total_time / 5
            speedup = self.baseline_time_us / avg_time if avg_time > 0 else 0

            return {
                'code': code,
                'correct': True,
                'time_us': avg_time,
                'speedup': speedup,
            }

        except Exception as e:
            return {'code': code, 'correct': False, 'time_us': float('inf'), 'error': str(e)}

    def _mutate(self, code: str) -> str:
        """Apply random mutation."""
        patches = self.patch_gen.generate_patches(3)

        for patch in patches:
            mutated = self.patch_gen.apply_patch(code, patch)
            if mutated != code:
                try:
                    compile(mutated, '<string>', 'exec')
                    return mutated
                except:
                    pass

        return code

    def _select_parent(self) -> str:
        """Tournament selection."""
        tournament = random.sample(self.population, min(3, len(self.population)))
        winner = max(tournament, key=lambda x: x.get('speedup', 0))
        return winner['code']

    def initialize(self) -> None:
        """Initialize population."""
        # Start with baseline
        self.population.append(self._evaluate(self.baseline_code))

        # Add known good variants
        variants = [
            self._get_unrolled_code(),
            self._get_rowmajor_code(),
        ]

        for code in variants:
            result = self._evaluate(code)
            self.population.append(result)

        # Random mutations
        while len(self.population) < self.population_size:
            mutated = self._mutate(self.baseline_code)
            result = self._evaluate(mutated)
            self.population.append(result)

        # Update best
        for ind in self.population:
            if ind['correct'] and ind.get('speedup', 0) > self.best_speedup:
                self.best_speedup = ind['speedup']
                self.best_code = ind['code']
                self.best_time = ind['time_us']

    def _get_unrolled_code(self) -> str:
        return '''def matmul(A, B):
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
         a3[0]*b0[3] + a3[1]*b1[3] + a3[2]*b2[3] + a3[3]*b3[3]]]'''

    def _get_rowmajor_code(self) -> str:
        return '''def matmul(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for k in range(4):
            aik = A[i][k]
            for j in range(4):
                C[i][j] += aik * B[k][j]
    return C'''

    def evolve(self, num_iterations: int = 50) -> dict:
        """Run evolution."""
        print("Initializing population...")
        self.initialize()

        print(f"Initial best: {self.best_speedup:.2f}x speedup")

        for iteration in range(num_iterations):
            new_population = []

            # Keep elite
            sorted_pop = sorted(
                [p for p in self.population if p.get('correct', False)],
                key=lambda x: x.get('speedup', 0),
                reverse=True
            )
            elite = sorted_pop[:self.elite_keep]
            new_population.extend(elite)

            # Generate new individuals
            while len(new_population) < self.population_size:
                parent = self._select_parent()
                mutated = self._mutate(parent)
                result = self._evaluate(mutated)
                new_population.append(result)

                if result['correct'] and result.get('speedup', 0) > self.best_speedup:
                    self.best_speedup = result['speedup']
                    self.best_code = result['code']
                    self.best_time = result['time_us']
                    print(f"Iter {iteration}: New best! {self.best_speedup:.2f}x")

            self.population = new_population

            if (iteration + 1) % 10 == 0:
                print(f"Iter {iteration + 1}: Best={self.best_speedup:.2f}x")

        return {
            'best_code': self.best_code,
            'best_speedup': self.best_speedup,
            'best_time_us': self.best_time,
        }


if __name__ == "__main__":
    baseline = '''def matmul(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]
    return C'''

    evolver = MatrixMultiplyEvolution(baseline, baseline_time_us=10.0)
    result = evolver.evolve(100)

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best speedup: {result['best_speedup']:.2f}x")
    print(f"Best time: {result['best_time_us']:.2f}us")
    print(f"\nBest code:\n{result['best_code']}")
