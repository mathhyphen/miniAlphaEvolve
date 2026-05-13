"""Run AlphaEvolve to optimize 4x4 matrix multiplication.

Usage:
    python -m run.optimize_matrix_multiply
    python -m run.optimize_matrix_multiply --iterations 100
    python -m run.optimize_matrix_multiply --method random_search
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaevolve.problems.matrix_multiply_problem import (
    MatrixMultiplyProblem,
    compute_reward,
    extract_features,
    UNROLLED_CODE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for matrix multiply optimization."""
    num_iterations: int = 100
    population_size: int = 20
    elite_keep: int = 3
    mutation_probability: float = 0.2
    crossover_probability: float = 0.3
    baseline_time_us: float = 10.0


class MatrixMultiplyOptimizer:
    """Optimizer for matrix multiplication algorithms."""

    # Known optimization patterns
    OPTIMIZATION_PATTERNS = {
        'unroll_outer': [
            ('for i in range(4):\n        for j in range(4):',
             'for i in range(4):\n    row_i = A[i]\n    for j in range(4):'),
        ],
        'preload_b': [
            ('for i in range(4):\n        for j in range(4):\n            for k in range(4):\n                C[i][j] += A[i][k] * B[k][j]',
             'for i in range(4):\n        for k in range(4):\n            aik = A[i][k]\n            for j in range(4):\n                C[i][j] += aik * B[k][j]'),
        ],
        'preload_a': [
            ('for i in range(4):\n        for j in range(4):\n            for k in range(4):\n                C[i][j] += A[i][k] * B[k][j]',
             'for j in range(4):\n        for k in range(4):\n            bkj = B[k][j]\n            for i in range(4):\n                C[i][j] += A[i][k] * bkj'),
        ],
        'register_opt': [
            ('for i in range(4):\n        for j in range(4):\n            for k in range(4):\n                C[i][j] += A[i][k] * B[k][j]',
             'a0, a1, a2, a3 = A[0], A[1], A[2], A[3]\nb0, b1, b2, b3 = B[0], B[1], B[2], B[3]\nC = [\n    [a0[0]*b0[0] + a0[1]*b1[0] + a0[2]*b2[0] + a0[3]*b3[0],\n     a0[0]*b0[1] + a0[1]*b1[1] + a0[2]*b2[1] + a0[3]*b3[1],\n     a0[0]*b0[2] + a0[1]*b1[2] + a0[2]*b2[2] + a0[3]*b3[2],\n     a0[0]*b0[3] + a0[1]*b1[3] + a0[2]*b2[3] + a0[3]*b3[3]],\n    [a1[0]*b0[0] + a1[1]*b1[0] + a1[2]*b2[0] + a1[3]*b3[0],\n     a1[0]*b0[1] + a1[1]*b1[1] + a1[2]*b2[1] + a1[3]*b3[1],\n     a1[0]*b0[2] + a1[1]*b1[2] + a1[2]*b2[2] + a1[3]*b3[2],\n     a1[0]*b0[3] + a1[1]*b1[3] + a1[2]*b2[3] + a1[3]*b3[3]],\n    [a2[0]*b0[0] + a2[1]*b1[0] + a2[2]*b2[0] + a2[3]*b3[0],\n     a2[0]*b0[1] + a2[1]*b1[1] + a2[2]*b2[1] + a2[3]*b3[1],\n     a2[0]*b0[2] + a2[1]*b1[2] + a2[2]*b2[2] + a2[3]*b3[2],\n     a2[0]*b0[3] + a2[1]*b1[3] + a2[2]*b2[3] + a2[3]*b3[3]],\n    [a3[0]*b0[0] + a3[1]*b1[0] + a3[2]*b2[0] + a3[3]*b3[0],\n     a3[0]*b0[1] + a3[1]*b1[1] + a3[2]*b2[1] + a3[3]*b3[1],\n     a3[0]*b0[2] + a3[1]*b1[2] + a3[2]*b2[2] + a3[3]*b3[2],\n     a3[0]*b0[3] + a3[1]*b1[3] + a3[2]*b2[3] + a3[3]*b3[3]]\n]'),
        ],
        'column_major': [
            ('for i in range(4):\n        for j in range(4):\n            for k in range(4):\n                C[i][j] += A[i][k] * B[k][j]',
             'for k in range(4):\n        for j in range(4):\n            for i in range(4):\n                C[i][j] += A[i][k] * B[k][j]'),
        ],
    }

    def __init__(
        self,
        problem: MatrixMultiplyProblem,
        config: Optional[OptimizationConfig] = None
    ):
        self.problem = problem
        self.config = config or OptimizationConfig()

        self.population: List[Dict[str, Any]] = []
        self.best_code = problem.initial_code
        self.best_reward = -float('inf')
        self.best_speedup = 1.0
        self.history: List[Dict[str, Any]] = []

        # Known good solutions
        self.known_solutions = {
            'naive': self.problem.initial_code,
            'unrolled': UNROLLED_CODE,
        }

        # Count optimization attempts
        self.attempts = 0
        self.successful_mutations = 0

    def _evaluate(self, code: str) -> Dict[str, Any]:
        """Evaluate a candidate algorithm."""
        self.attempts += 1
        is_correct, exec_time, error = self.problem.verify(code)
        reward = compute_reward(is_correct, exec_time, self.config.baseline_time_us)

        return {
            'code': code,
            'is_correct': is_correct,
            'execution_time_us': exec_time,
            'reward': reward,
            'speedup': self.config.baseline_time_us / max(exec_time, 0.001) if is_correct else 0.0,
            'error': error,
        }

    def _apply_pattern(self, code: str, pattern_name: str) -> str:
        """Apply a known optimization pattern."""
        if pattern_name not in self.OPTIMIZATION_PATTERNS:
            return code

        # Pattern is a list of (old, new) tuples
        old, new = self.OPTIMIZATION_PATTERNS[pattern_name][0]
        return code.replace(old, new)

    def _mutate(self, code: str) -> str:
        """Apply random mutation to code."""
        patterns = list(self.OPTIMIZATION_PATTERNS.keys())

        # Try random patterns
        for _ in range(3):
            pattern = random.choice(patterns)
            mutated = self._apply_pattern(code, pattern)
            if mutated != code:
                try:
                    compile(mutated, '<string>', 'exec')
                    self.successful_mutations += 1
                    return mutated
                except SyntaxError:
                    pass

        return code

    def _crossover(self, parent1: str, parent2: str) -> str:
        """Perform crossover between two parent codes."""
        # Simplified: just pick the better one
        return parent1 if random.random() < 0.5 else parent2

    def _select_parent(self) -> str:
        """Select parent using tournament selection."""
        tournament_size = 3
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        winner = max(tournament, key=lambda x: x['reward'])
        return winner['code']

    def _initialize_population(self) -> None:
        """Initialize population with known good solutions and variants."""
        # Add known solutions
        for name, code in self.known_solutions.items():
            result = self._evaluate(code)
            self.population.append(result)
            if result['reward'] > self.best_reward:
                self.best_reward = result['reward']
                self.best_code = code
                self.best_speedup = result['speedup']

        # Generate variants
        while len(self.population) < self.config.population_size:
            base = random.choice(list(self.known_solutions.values()))
            mutated = self._mutate(base)
            result = self._evaluate(mutated)
            self.population.append(result)

            if result['reward'] > self.best_reward:
                self.best_reward = result['reward']
                self.best_code = result['code']
                self.best_speedup = result['speedup']

    def _evolve_population(self) -> None:
        """Evolve the population for one generation."""
        new_population = []

        # Keep elite
        sorted_pop = sorted(self.population, key=lambda x: x['reward'], reverse=True)
        elite = sorted_pop[:self.config.elite_keep]
        new_population.extend(elite)

        # Generate new individuals
        while len(new_population) < self.config.population_size:
            if self.population and random.random() < self.config.crossover_probability:
                # Crossover
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child_code = self._crossover(parent1, parent2)
            else:
                # Mutation
                if self.population:
                    parent_code = self._select_parent()
                else:
                    parent_code = self.problem.initial_code
                child_code = self._mutate(parent_code)

            result = self._evaluate(child_code)
            new_population.append(result)

            if result['reward'] > self.best_reward:
                self.best_reward = result['reward']
                self.best_code = result['code']
                self.best_speedup = result['speedup']

        self.population = new_population

    def optimize(self) -> Dict[str, Any]:
        """Run the optimization."""
        logger.info("Initializing population...")
        self._initialize_population()

        logger.info(f"Starting evolution for {self.config.num_iterations} iterations")
        logger.info(f"Initial best: {self.best_speedup:.2f}x speedup ({self.best_reward:.3f} reward)")

        for iteration in range(self.config.num_iterations):
            self._evolve_population()

            if (iteration + 1) % 10 == 0:
                correct_count = sum(1 for p in self.population if p['is_correct'])
                avg_speedup = sum(p['speedup'] for p in self.population if p['is_correct']) / max(correct_count, 1)
                logger.info(f"Iter {iteration + 1}: Best={self.best_speedup:.2f}x, "
                          f"Avg={avg_speedup:.2f}x, "
                          f"Correct={correct_count}/{len(self.population)}, "
                          f"Mutations={self.successful_mutations}/{self.attempts}")

                self.history.append({
                    'iteration': iteration + 1,
                    'best_speedup': self.best_speedup,
                    'avg_speedup': avg_speedup,
                    'correct_count': correct_count
                })

        return {
            'best_code': self.best_code,
            'best_reward': self.best_reward,
            'best_speedup': self.best_speedup,
            'total_attempts': self.attempts,
            'successful_mutations': self.successful_mutations,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optimize 4x4 matrix multiplication"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=100,
        help="Number of iterations"
    )
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=20,
        help="Population size"
    )
    parser.add_argument(
        "--baseline-time",
        type=float,
        default=10.0,
        help="Baseline execution time in microseconds"
    )
    parser.add_argument(
        "--use-unrolled",
        action="store_true",
        help="Start from unrolled baseline"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create problem
    problem = MatrixMultiplyProblem()
    if args.use_unrolled:
        problem.initial_code = UNROLLED_CODE

    config = OptimizationConfig(
        num_iterations=args.iterations,
        population_size=args.population,
        baseline_time_us=args.baseline_time,
    )

    # Run optimization
    optimizer = MatrixMultiplyOptimizer(problem, config)
    results = optimizer.optimize()

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best speedup: {results['best_speedup']:.2f}x")
    print(f"Total attempts: {results['total_attempts']}")
    print(f"Successful mutations: {results['successful_mutations']}")
    print(f"Mutation success rate: {results['successful_mutations']/max(results['total_attempts'],1)*100:.1f}%")
    print(f"\nBest code:\n{results['best_code']}")

    # Save results
    output_dir = Path(f"outputs/matmul_opt_{datetime.now():%Y%m%d_%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "best_code.py", 'w') as f:
        f.write(results['best_code'])

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
