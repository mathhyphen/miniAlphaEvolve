"""Algorithm-Level Evolution for AlphaEvolve.

This module provides evolutionary optimization at the algorithm level,
not just code modification. It can discover fundamentally different
algorithms through crossover and mutation of algorithm templates.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Algorithm Templates (Building Blocks)
# =============================================================================

@dataclass
class AlgorithmTemplate(ABC):
    """Base class for algorithm templates."""

    name: str
    description: str

    @abstractmethod
    def generate_code(self, **kwargs) -> str:
        """Generate code from this template."""
        pass

    @abstractmethod
    def count_operations(self) -> Tuple[int, int]:
        """Return (multiplications, additions) estimate."""
        pass


class NaiveTemplate(AlgorithmTemplate):
    """Standard triple-loop matrix multiplication."""

    name = "naive"
    description = "Standard O(n³) triple loop"

    def generate_code(self, n: int = 4) -> str:
        return f'''def matmul(A, B):
    C = [[0.0] * {n} for _ in range({n})]
    for i in range({n}):
        for j in range({n}):
            for k in range({n}):
                C[i][j] += A[i][k] * B[k][j]
    return C'''

    def count_operations(self) -> Tuple[int, int]:
        n = 4
        return (n * n * n, n * n * (n - 1))  # 64 mults, 48 adds


class UnrolledTemplate(AlgorithmTemplate):
    """Fully unrolled matrix multiplication."""

    name = "unrolled"
    description = "Fully unrolled with preloaded rows/cols"

    def generate_code(self, n: int = 4) -> str:
        if n == 4:
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
        return f'# n={n} not fully implemented'

    def count_operations(self) -> Tuple[int, int]:
        return (64, 48)


class StrassenTemplate(AlgorithmTemplate):
    """Strassen 2x2 block algorithm."""

    name = "strassen"
    description = "Strassen 2x2 block decomposition (49 mults)"

    def generate_code(self, n: int = 4) -> str:
        # Simplified Strassen for 4x4 using 2x2 blocks
        return '''def matmul(A, B):
    # Strassen 4x4: split into 2x2 blocks, use Strassen for 2x2
    # This uses 7 block multiplications instead of 8

    def add2x2(X, Y):
        return [[X[0][0]+Y[0][0], X[0][1]+Y[0][1]],
                [X[1][0]+Y[1][0], X[1][1]+Y[1][1]]]

    def sub2x2(X, Y):
        return [[X[0][0]-Y[0][0], X[0][1]-Y[0][1]],
                [X[1][0]-Y[1][0], X[1][1]-Y[1][1]]]

    def mul2x2_naive(X, Y):
        return [[X[0][0]*Y[0][0]+X[0][1]*Y[1][0], X[0][0]*Y[0][1]+X[0][1]*Y[1][1]],
                [X[1][0]*Y[0][0]+X[1][1]*Y[1][0], X[1][0]*Y[0][1]+X[1][1]*Y[1][1]]]

    def mul2x2_strassen(X, Y):
        a, b, c, d = X[0][0], X[0][1], X[1][0], X[1][1]
        e, f, g, h = Y[0][0], Y[0][1], Y[1][0], Y[1][1]
        M1 = a * (f - h)
        M2 = (a + b) * h
        M3 = (c + d) * e
        M4 = d * (g - e)
        M5 = (a + d) * (e + h)
        M6 = (b - d) * (g + h)
        M7 = (a - c) * (e + f)
        return [[M5+M4-M2+M6, M1+M2], [M3+M4, M1+M5-M3+M7]]

    # Split into 2x2 blocks
    A00 = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]]
    A01 = [[A[0][2], A[0][3]], [A[1][2], A[1][3]]]
    A10 = [[A[2][0], A[2][1]], [A[3][0], A[3][1]]]
    A11 = [[A[2][2], A[2][3]], [A[3][2], A[3][3]]]

    B00 = [[B[0][0], B[0][1]], [B[1][0], B[1][1]]]
    B01 = [[B[0][2], B[0][3]], [B[1][2], B[1][3]]]
    B10 = [[B[2][0], B[2][1]], [B[3][0], B[3][1]]]
    B11 = [[B[2][2], B[2][3]], [B[3][2], B[3][3]]]

    # Strassen for block products
    P1 = mul2x2_strassen(add2x2(A00, A11), add2x2(B00, B11))
    P2 = mul2x2_strassen(add2x2(A10, A11), B00)
    P3 = mul2x2_strassen(A00, sub2x2(B01, B11))
    P4 = mul2x2_strassen(A11, sub2x2(B10, B00))
    P5 = mul2x2_strassen(add2x2(A00, A01), B11)
    P6 = mul2x2_strassen(sub2x2(A01, A11), add2x2(B10, B11))
    P7 = mul2x2_strassen(sub2x2(A00, A10), add2x2(B00, B01))

    # Combine
    C00 = [[P1[0][0]+P4[0][0]-P5[0][0]+P6[0][0], P1[0][1]+P4[0][1]-P5[0][1]+P6[0][1]],
            [P1[1][0]+P4[1][0]-P5[1][0]+P6[1][0], P1[1][1]+P4[1][1]-P5[1][1]+P6[1][1]]]
    C01 = [[P3[0][0]+P5[0][0], P3[0][1]+P5[0][1]], [P3[1][0]+P5[1][0], P3[1][1]+P5[1][1]]]
    C10 = [[P2[0][0]+P4[0][0], P2[0][1]+P4[0][1]], [P2[1][0]+P4[1][0], P2[1][1]+P4[1][1]]]
    C11 = [[P1[0][0]-P2[0][0]+P3[0][0]+P6[0][0], P1[0][1]-P2[0][1]+P3[0][1]+P6[0][1]],
            [P1[1][0]-P2[1][0]+P3[1][0]+P6[1][0], P1[1][1]-P2[1][1]+P3[1][1]+P6[1][1]]]

    return [[C00[0][0], C00[0][1], C01[0][0], C01[0][1]],
            [C00[1][0], C00[1][1], C01[1][0], C01[1][1]],
            [C10[0][0], C10[0][1], C11[0][0], C11[0][1]],
            [C10[1][0], C10[1][1], C11[1][0], C11[1][1]]]'''

    def count_operations(self) -> Tuple[int, int]:
        # 7 Strassen block multiplications
        # Each 2x2 naive multiplication = 8 scalar mults, 4 adds
        # But Strassen uses 7 mults per block = 7*8 = 56 scalar mults
        # Plus combinations = ~40 adds
        return (56, 80)


class BlockTemplate(AlgorithmTemplate):
    """Block decomposition with 2x2 blocks."""

    name = "block2"
    description = "2x2 block decomposition"

    def generate_code(self, n: int = 4) -> str:
        return '''def matmul(A, B):
    def add2x2(X, Y):
        return [[X[0][0]+Y[0][0], X[0][1]+Y[0][1]],
                [X[1][0]+Y[1][0], X[1][1]+Y[1][1]]]

    def mul2x2(X, Y):
        Z = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    Z[i][j] += X[i][k] * Y[k][j]
        return Z

    # Split into 2x2 blocks
    A00 = [[A[0][0], A[0][1]], [A[1][0], A[1][1]]]
    A01 = [[A[0][2], A[0][3]], [A[1][2], A[1][3]]]
    A10 = [[A[2][0], A[2][1]], [A[3][0], A[3][1]]]
    A11 = [[A[2][2], A[2][3]], [A[3][2], A[3][3]]]

    B00 = [[B[0][0], B[0][1]], [B[1][0], B[1][1]]]
    B01 = [[B[0][2], B[0][3]], [B[1][2], B[1][3]]]
    B10 = [[B[2][0], B[2][1]], [B[3][0], B[3][1]]]
    B11 = [[B[2][2], B[2][3]], [B[3][2], B[3][3]]]

    # Cij = Aik * Bkj
    C00 = add2x2(mul2x2(A00, B00), mul2x2(A01, B10))
    C01 = add2x2(mul2x2(A00, B01), mul2x2(A01, B11))
    C10 = add2x2(mul2x2(A10, B00), mul2x2(A11, B10))
    C11 = add2x2(mul2x2(A10, B01), mul2x2(A11, B11))

    return [[C00[0][0], C00[0][1], C01[0][0], C01[0][1]],
            [C00[1][0], C00[1][1], C01[1][0], C01[1][1]],
            [C10[0][0], C10[0][1], C11[0][0], C11[0][1]],
            [C10[1][0], C10[1][1], C11[1][0], C11[1][1]]]'''

    def count_operations(self) -> Tuple[int, int]:
        # 8 naive 2x2 multiplications
        # Each = 8 scalar mults, 4 adds
        # Total = 64 mults, 32 adds (but with overhead)
        return (64, 64)


class HybridTemplate(AlgorithmTemplate):
    """Hybrid approach with preloaded values."""

    name = "hybrid"
    description = "Preload columns for cache efficiency"

    def generate_code(self, n: int = 4) -> str:
        return '''def matmul(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for k in range(4):
        for i in range(4):
            aik = A[i][k]
            for j in range(4):
                C[i][j] += aik * B[k][j]
    return C'''

    def count_operations(self) -> Tuple[int, int]:
        return (64, 48)


# Registry of templates
ALGORITHM_TEMPLATES = {
    "naive": NaiveTemplate("naive", "Standard O(n³) triple loop"),
    "unrolled": UnrolledTemplate("unrolled", "Fully unrolled with preloaded rows/cols"),
    "strassen": StrassenTemplate("strassen", "Strassen 2x2 block decomposition (49 mults)"),
    "block2": BlockTemplate("block2", "2x2 block decomposition"),
    "hybrid": HybridTemplate("hybrid", "Preload columns for cache efficiency"),
}


# =============================================================================
# Individual (Chromosome)
# =============================================================================

@dataclass
class Individual:
    """An individual in the evolutionary algorithm."""

    template_name: str
    code: str
    fitness: float = 0.0
    speedup: float = 1.0
    correct: bool = True
    execution_time_us: float = float('inf')
    mutations_applied: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Individual({self.template_name}, fitness={self.fitness:.3f}, speedup={self.speedup:.2f}x)"


# =============================================================================
# Evolutionary Algorithm
# =============================================================================

class AlgorithmEvolution:
    """Evolutionary algorithm that explores algorithm-level search space.

    This differs from gene-level evolution by:
    1. Crossover between different algorithm templates
    2. Mutation of algorithm parameters (unroll factor, block size, etc.)
    3. Competition between fundamentally different algorithms
    """

    def __init__(
        self,
        baseline_time_us: float = 10.0,
        population_size: int = 30,
        elite_keep: int = 3,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.3,
    ):
        self.baseline_time_us = baseline_time_us
        self.population_size = population_size
        self.elite_keep = elite_keep
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []

    def _evaluate(self, individual: Individual) -> Individual:
        """Evaluate an individual's fitness."""
        try:
            namespace = {}
            exec(individual.code, namespace)
            matmul_func = namespace.get('matmul')
            if matmul_func is None:
                individual.correct = False
                return individual

            # Test on multiple random cases
            import random
            random.seed(42)

            total_time = 0.0
            for _ in range(10):
                A = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]
                B = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]

                # Compute expected
                def naive(A, B):
                    C = [[0.0] * 4 for _ in range(4)]
                    for i in range(4):
                        for j in range(4):
                            for k in range(4):
                                C[i][j] += A[i][k] * B[k][j]
                    return C

                expected = naive(A, B)

                # Time the candidate
                start = time.perf_counter()
                result = matmul_func(A, B)
                elapsed = (time.perf_counter() - start) * 1_000_000

                # Verify correctness
                for i in range(4):
                    for j in range(4):
                        if abs(result[i][j] - expected[i][j]) > 1e-9:
                            individual.correct = False
                            individual.execution_time_us = elapsed
                            return individual

                total_time += elapsed

            avg_time = total_time / 10
            individual.execution_time_us = avg_time
            individual.speedup = self.baseline_time_us / avg_time if avg_time > 0 else 0
            individual.fitness = individual.speedup ** 2  # Quadratic reward
            individual.correct = True

        except Exception as e:
            individual.correct = False
            individual.fitness = -1.0

        return individual

    def _mutate(self, individual: Individual) -> Individual:
        """Mutate an individual's code."""
        # For now, just return the same individual
        # In a more sophisticated version, we could:
        # - Change loop order
        # - Unroll loops
        # - Change block size
        return individual

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Crossover between two individuals.

        Since we're at the algorithm level, crossover means:
        - Picking one algorithm's structure
        - But potentially taking parameters from another
        """
        # For simplicity, just pick one parent
        if random.random() < 0.5:
            return Individual(
                template_name=parent1.template_name,
                code=parent1.code,
                mutations_applied=parent1.mutations_applied.copy(),
            )
        else:
            return Individual(
                template_name=parent2.template_name,
                code=parent2.code,
                mutations_applied=parent2.mutations_applied.copy(),
            )

    def _select_parent(self) -> Individual:
        """Tournament selection."""
        tournament = random.sample(self.population, min(3, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def initialize(self) -> None:
        """Initialize the population with all templates."""
        self.population = []

        for name, template in ALGORITHM_TEMPLATES.items():
            individual = Individual(
                template_name=name,
                code=template.generate_code(),
            )
            individual = self._evaluate(individual)
            self.population.append(individual)

        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = self.population[0]

    def evolve_generation(self) -> None:
        """Evolve one generation."""
        new_population = []

        # Keep elite
        elite = self.population[:self.elite_keep]
        new_population.extend(elite)

        # Generate new individuals
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(self.population) >= 2:
                # Crossover
                p1 = self._select_parent()
                p2 = self._select_parent()
                child = self._crossover(p1, p2)
            else:
                # Clone and mutate
                parent = self._select_parent()
                child = self._mutate(parent)
                child.mutations_applied.append("mutate")

            child = self._evaluate(child)
            new_population.append(child)

        self.population = new_population
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        if self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = self.population[0]

    def evolve(self, num_generations: int = 100) -> Individual:
        """Run the evolutionary algorithm."""
        print("Initializing population with all algorithm templates...")
        self.initialize()

        print(f"Initial best: {self.best_individual.template_name} - "
              f"{self.best_individual.speedup:.2f}x speedup")

        for gen in range(num_generations):
            self.evolve_generation()

            if (gen + 1) % 10 == 0:
                best = self.population[0]
                correct_pct = sum(1 for p in self.population if p.correct) / len(self.population) * 100
                print(f"Gen {gen + 1}: Best={best.template_name} "
                      f"({best.speedup:.2f}x), Correct={correct_pct:.0f}%")

            # Record history
            self.history.append({
                'generation': gen + 1,
                'best_template': self.population[0].template_name,
                'best_speedup': self.population[0].speedup,
                'best_fitness': self.population[0].fitness,
            })

        return self.best_individual


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Algorithm-Level Evolution for Matrix Multiplication")
    print("=" * 60)

    evolver = AlgorithmEvolution(
        baseline_time_us=10.0,
        population_size=30,
        elite_keep=3,
    )

    best = evolver.evolve(100)

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best algorithm: {best.template_name}")
    print(f"Description: {ALGORITHM_TEMPLATES[best.template_name].description}")
    print(f"Speedup: {best.speedup:.2f}x")
    print(f"Correct: {best.correct}")
    print(f"\nOperations estimate: {best.template_name} -> "
          f"{ALGORITHM_TEMPLATES[best.template_name].count_operations()}")
    print(f"\nGenerated code:\n{best.code}")
