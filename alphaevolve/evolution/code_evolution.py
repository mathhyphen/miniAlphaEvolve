"""Code-Level Evolution with Genetic Programming for AlphaEvolve.

This module provides gene-level evolution where the chromosomes are
actual code snippets that can be mutated and crossed over.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# =============================================================================
# Code Chromosome
# =============================================================================

@dataclass
class CodeChromosome:
    """A chromosome represented as code."""
    code: str
    template_type: str  # "naive", "unrolled", "strassen", etc.
    fitness: float = 0.0
    speedup: float = 1.0
    correct: bool = True
    execution_time_us: float = float('inf')

    def __repr__(self) -> str:
        return f"CodeChromosome({self.template_type}, fitness={self.fitness:.3f})"


# =============================================================================
# Mutation Operators (Code-Level)
# =============================================================================

class MutationOperators:
    """Collection of mutation operators for code evolution."""

    @staticmethod
    def unroll_loops(code: str) -> str:
        """Unroll innermost loop."""
        patterns = [
            # Unroll k-loop in naive multiplication
            (
                r'(for i in range\(4\):\s*\n\s*for j in range\(4\):\s*\n\s*for k in range\(4\):\s*\n\s*C\[i\]\[j\] \+= A\[i\]\[k\] \* B\[k\]\[j\])',
                '''for i in range(4):
    for j in range(4):
        C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j] + A[i][3]*B[3][j]'''
            ),
            # Unroll j-loop
            (
                r'(for i in range\(4\):\s*\n\s*for j in range\(4\):)',
                '''for i in range(4):
    row = A[i]
    for j in range(4):'''
            ),
        ]

        for pattern, replacement in patterns:
            new_code = re.sub(pattern, replacement, code, flags=re.DOTALL)
            if new_code != code:
                return new_code

        return code

    @staticmethod
    def reorder_loops(code: str) -> str:
        """Reorder loops for better cache behavior."""
        # i,k,j ordering (better for row-major A access)
        old = '''for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]'''

        new = '''for i in range(4):
        for k in range(4):
            aik = A[i][k]
            for j in range(4):
                C[i][j] += aik * B[k][j]'''

        if old in code:
            return code.replace(old, new)

        return code

    @staticmethod
    def preload_values(code: str) -> str:
        """Preload values into registers."""
        old = '''for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]'''

        new = '''for i in range(4):
        a0, a1, a2, a3 = A[i][0], A[i][1], A[i][2], A[i][3]
        for j in range(4):
            C[i][j] = a0*B[0][j] + a1*B[1][j] + a2*B[2][j] + a3*B[3][j]'''

        if old in code:
            return code.replace(old, new)

        return code

    @staticmethod
    def full_unroll(code: str) -> str:
        """Fully unroll all loops."""
        old = '''def matmul(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]
    return C'''

        new = '''def matmul(A, B):
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

        if old in code:
            return new

        return code

    @staticmethod
    def transpose_b(code: str) -> str:
        """Use transposed B for better memory access."""
        old = '''for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]'''

        new = '''bt = list(zip(*B))
for i in range(4):
    for j in range(4):
        C[i][j] = sum(A[i][k] * bt[j][k] for k in range(4))'''

        if old in code:
            return code.replace(old, new)

        return code


# =============================================================================
# Templates
# =============================================================================

TEMPLATES = {
    "naive": '''def matmul(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]
    return C''',

    "rowmajor": '''def matmul(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for k in range(4):
            aik = A[i][k]
            for j in range(4):
                C[i][j] += aik * B[k][j]
    return C''',

    "colmajor": '''def matmul(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for k in range(4):
        for j in range(4):
            bkj = B[k][j]
            for i in range(4):
                C[i][j] += A[i][k] * bkj
    return C''',

    "preload": '''def matmul(A, B):
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        a0, a1, a2, a3 = A[i][0], A[i][1], A[i][2], A[i][3]
        for j in range(4):
            C[i][j] = a0*B[0][j] + a1*B[1][j] + a2*B[2][j] + a3*B[3][j]
    return C''',

    "unrolled": '''def matmul(A, B):
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
         a3[0]*b0[3] + a3[1]*b1[3] + a3[2]*b2[3] + a3[3]*b3[3]]]''',
}


# =============================================================================
# Evolution Engine
# =============================================================================

class CodeEvolution:
    """Genetic Programming-style evolution of code.

    Key features:
    1. Chromosomes are actual code strings
    2. Mutation operators modify code structure
    3. Crossover can combine code from different parents
    4. Fitness evaluation runs the code and measures performance
    """

    def __init__(
        self,
        baseline_time_us: float = 10.0,
        population_size: int = 50,
        elite_keep: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.4,
    ):
        self.baseline_time_us = baseline_time_us
        self.population_size = population_size
        self.elite_keep = elite_keep
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population: List[CodeChromosome] = []
        self.best: Optional[CodeChromosome] = None
        self.history: List[dict] = []

        self.mutation_ops = MutationOperators()

    def _evaluate(self, chromosome: CodeChromosome) -> CodeChromosome:
        """Evaluate a chromosome's fitness."""
        try:
            namespace = {}
            exec(chromosome.code, namespace)
            matmul_func = namespace.get('matmul')
            if matmul_func is None:
                chromosome.correct = False
                return chromosome

            import random
            random.seed(42)

            total_time = 0.0
            for _ in range(20):  # More test cases
                A = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]
                B = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]

                def naive(A, B):
                    C = [[0.0] * 4 for _ in range(4)]
                    for i in range(4):
                        for j in range(4):
                            for k in range(4):
                                C[i][j] += A[i][k] * B[k][j]
                    return C

                expected = naive(A, B)

                start = time.perf_counter()
                result = matmul_func(A, B)
                elapsed = (time.perf_counter() - start) * 1_000_000

                for i in range(4):
                    for j in range(4):
                        if abs(result[i][j] - expected[i][j]) > 1e-9:
                            chromosome.correct = False
                            chromosome.execution_time_us = elapsed
                            return chromosome

                total_time += elapsed

            avg_time = total_time / 20
            chromosome.execution_time_us = avg_time
            chromosome.speedup = self.baseline_time_us / avg_time if avg_time > 0 else 0
            chromosome.fitness = chromosome.speedup ** 2
            chromosome.correct = True

        except Exception as e:
            chromosome.correct = False
            chromosome.fitness = -1.0

        return chromosome

    def _mutate(self, chromosome: CodeChromosome) -> CodeChromosome:
        """Apply random mutation to chromosome."""
        mutations = [
            ("unroll", self.mutation_ops.unroll_loops),
            ("reorder", self.mutation_ops.reorder_loops),
            ("preload", self.mutation_ops.preload_values),
            ("full_unroll", self.mutation_ops.full_unroll),
            ("transpose", self.mutation_ops.transpose_b),
        ]

        # Try multiple mutations
        for _ in range(3):
            name, op = random.choice(mutations)
            new_code = op(chromosome.code)
            if new_code != chromosome.code:
                try:
                    compile(new_code, '<string>', 'exec')
                    chromosome.code = new_code
                    chromosome.template_type = name
                    break
                except:
                    pass

        return chromosome

    def _crossover(self, p1: CodeChromosome, p2: CodeChromosome) -> CodeChromosome:
        """Crossover between two chromosomes.

        Since code structure matters, we do a simple blend:
        take the better template but apply some mutations from the other.
        """
        # For now, just return a clone of the better parent
        return CodeChromosome(
            code=p1.code,
            template_type=p1.template_type,
        )

    def _select_parent(self) -> CodeChromosome:
        """Tournament selection."""
        tournament = random.sample(self.population, min(5, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)

    def initialize(self) -> None:
        """Initialize population from templates."""
        self.population = []

        for name, code in TEMPLATES.items():
            chrom = CodeChromosome(code=code, template_type=name)
            chrom = self._evaluate(chrom)
            self.population.append(chrom)

        # Add some mutations of templates
        while len(self.population) < self.population_size:
            template_name = random.choice(list(TEMPLATES.keys()))
            chrom = CodeChromosome(code=TEMPLATES[template_name], template_type=template_name)
            chrom = self._mutate(chrom)
            chrom = self._evaluate(chrom)
            self.population.append(chrom)

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best = self.population[0]

    def evolve_generation(self) -> None:
        """Evolve one generation."""
        new_pop = []

        # Keep elite
        elite = self.population[:self.elite_keep]
        new_pop.extend([CodeChromosome(code=c.code, template_type=c.template_type,
                                        fitness=c.fitness, speedup=c.speedup,
                                        correct=c.correct, execution_time_us=c.execution_time_us)
                       for c in elite])

        while len(new_pop) < self.population_size:
            if random.random() < self.crossover_rate:
                p1 = self._select_parent()
                p2 = self._select_parent()
                child = self._crossover(p1, p2)
            else:
                parent = self._select_parent()
                child = CodeChromosome(code=parent.code, template_type=parent.template_type)

            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            child = self._evaluate(child)
            new_pop.append(child)

        self.population = new_pop
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        if self.population[0].fitness > (self.best.fitness if self.best else 0):
            self.best = self.population[0]

    def evolve(self, num_generations: int = 200) -> CodeChromosome:
        """Run the evolution."""
        print("Initializing population...")
        self.initialize()

        print(f"Initial best: {self.best.template_type} - {self.best.speedup:.2f}x speedup")

        for gen in range(num_generations):
            self.evolve_generation()

            if (gen + 1) % 20 == 0:
                best = self.population[0]
                correct_pct = sum(1 for p in self.population if p.correct) / len(self.population) * 100
                print(f"Gen {gen + 1}: Best={best.template_type} ({best.speedup:.2f}x), "
                      f"Correct={correct_pct:.0f}%, Best ever={self.best.speedup:.2f}x")

            self.history.append({
                'generation': gen + 1,
                'best_speedup': self.population[0].speedup,
                'best_type': self.population[0].template_type,
            })

        return self.best


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Code-Level Evolution for Matrix Multiplication")
    print("=" * 60)

    evolver = CodeEvolution(
        baseline_time_us=10.0,
        population_size=50,
        elite_keep=5,
        mutation_rate=0.4,
    )

    best = evolver.evolve(200)

    print("\n" + "=" * 60)
    print("EVOLUTION COMPLETE")
    print("=" * 60)
    print(f"Best algorithm: {best.template_type}")
    print(f"Speedup: {best.speedup:.2f}x")
    print(f"Fitness: {best.fitness:.3f}")
    print(f"\nGenerated code:\n{best.code}")
