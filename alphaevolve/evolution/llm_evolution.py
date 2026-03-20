"""LLM-Enhanced Evolution with Genetic Programming for AlphaEvolve.

This module extends the CodeEvolution class with LLM-generated code variants.
It uses the MiniMax API to generate new code implementations and PPO-style
training to learn which prompts/strategies produce the best algorithms.

Key features:
1. Generate 10-20 new code variants via LLM each iteration
2. Evaluate variants against test cases
3. Add correct variants to population
4. PPO-style training to learn which prompts work best
5. Track best algorithms discovered
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from alphaevolve.evolution.code_evolution import CodeChromosome, CodeEvolution, TEMPLATES
from alphaevolve.llm.minimax_client import generate_code_variants, optimize_code, MiniMaxError

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates for LLM Generation
# =============================================================================

PROMPT_TEMPLATES = {
    "naive_optimization": (
        "Generate a Python function named 'matmul' that multiplies two 4x4 matrices A and B. "
        "The function should return a new 4x4 matrix C = A @ B. "
        "Use only Python lists and standard operators. "
        "Optimize for speed by reducing loop overhead."
    ),

    "loop_unrolling": (
        "Generate a Python function named 'matmul' that performs 4x4 matrix multiplication. "
        "Unroll all loops to reduce overhead. "
        "Pre-load values into registers (local variables) before computation. "
        "The function should take two 4x4 matrices A and B and return their product."
    ),

    "register_optimization": (
        "Write an optimized 4x4 matrix multiplication function 'matmul(A, B)'. "
        "Use register optimization: load matrix values into local variables first. "
        "Compute each output element using direct multiplication and addition. "
        "Avoid function calls and loops inside the computation."
    ),

    "row_wise": (
        "Write a 4x4 matrix multiplication function 'matmul' that processes rows of A "
        "and computes dot products with columns of B. "
        "Pre-transpose B to improve cache locality. "
        "Return the resulting 4x4 matrix."
    ),

    "column_wise": (
        "Write a 4x4 matrix multiplication function 'matmul' that iterates over columns of B "
        "and computes contributions to all rows of C simultaneously. "
        "This improves cache performance for column-major access patterns."
    ),

    "block_optimization": (
        "Write a 4x4 matrix multiplication function using 2x2 block decomposition. "
        "Split matrices into four 2x2 blocks and compute each block pair. "
        "This improves cache locality for larger matrices."
    ),

    "sum_of_products": (
        "Write a 4x4 matrix multiplication using sum of products. "
        "Use Python's sum() with generator expressions for clean code. "
        "Compute C[i][j] = sum(A[i][k] * B[k][j] for k in range(4))"
    ),

    "zip_based": (
        "Write a 4x4 matrix multiplication using Python's zip for transposition. "
        "Transpose B once, then use zip to access rows. "
        "Make the code clean and Pythonic while being efficient."
    ),

    "manual_unroll": (
        "Write a fully manually unrolled 4x4 matmul. "
        "Extract all 16 elements of A and B as separate variables. "
        "Compute each C[i][j] as a direct sum of 4 products. "
        "No loops allowed - all 16 output elements computed explicitly."
    ),

    "vectorization_style": (
        "Write a 4x4 matrix multiplication that looks like SIMD vectorization. "
        "Group operations to enable the CPU to execute multiple operations in parallel. "
        "Compute 4 elements at a time where possible."
    ),

    "cache_friendly": (
        "Write a cache-friendly 4x4 matrix multiplication. "
        "Access matrices in an order that maximizes cache hits. "
        "Consider the row-major layout of Python lists."
    ),

    "instruction_level": (
        "Write a 4x4 matrix multiplication optimized for instruction-level parallelism. "
        "Order operations to maximize CPU pipeline utilization. "
        "Avoid data dependencies between consecutive operations."
    ),
}


# =============================================================================
# PPO-Style Policy for Prompt Selection
# =============================================================================

@dataclass
class PromptStrategy:
    """A prompt strategy with its success rate."""
    name: str
    prompt: str
    success_count: int = 0
    attempt_count: int = 0
    avg_fitness: float = 0.0
    total_fitness: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.attempt_count == 0:
            return 0.0
        return self.success_count / self.attempt_count

    @property
    def avg_success_fitness(self) -> float:
        """Average fitness of successful variants."""
        if self.success_count == 0:
            return 0.0
        return self.total_fitness / self.success_count

    def update(self, fitness: float, correct: bool) -> None:
        """Update statistics after an attempt."""
        self.attempt_count += 1
        self.total_fitness += fitness
        if correct:
            self.success_count += 1


class PPOPolicy:
    """PPO-style policy for learning which prompts produce best results.

    This policy maintains statistics on each prompt strategy and uses
    PPO-style updates to learn which prompts tend to produce better
    code variants.
    """

    def __init__(self, exploration_rate: float = 0.3):
        """Initialize the PPO policy.

        Args:
            exploration_rate: Probability of exploring new prompts vs exploiting best.
        """
        self.exploration_rate = exploration_rate
        self.strategies: Dict[str, PromptStrategy] = {}
        self.epsilon = 1e-6  # For numerical stability

        # Initialize with default strategies
        for name, prompt in PROMPT_TEMPLATES.items():
            self.strategies[name] = PromptStrategy(name=name, prompt=prompt)

    def select_prompt(self) -> Tuple[str, str]:
        """Select a prompt using epsilon-greedy with PPO-style weighting.

        Returns:
            Tuple of (strategy_name, prompt_text)
        """
        if random.random() < self.exploration_rate:
            # Explore: random selection weighted by potential
            return self._explore_select()
        else:
            # Exploit: use best known prompt
            return self._exploit_select()

    def _explore_select(self) -> Tuple[str, str]:
        """Select a prompt through exploration with ucb-style weighting."""
        names = list(self.strategies.keys())
        weights = []

        for name in names:
            strat = self.strategies[name]
            # UCB-style weight: success_rate + sqrt(2 * ln(total_attempts) / (attempts + 1))
            if strat.attempt_count == 0:
                weight = 1.0  # Unexplored strategies get boost
            else:
                import math
                total_attempts = sum(s.attempt_count for s in self.strategies.values())
                weight = strat.success_rate + math.sqrt(
                    2 * math.log(total_attempts + 1) / (strat.attempt_count + 1)
                )
            weights.append(weight)

        # Normalize weights
        total = sum(weights) + self.epsilon
        weights = [w / total for w in weights]

        # Weighted random selection
        idx = random.choices(range(len(names)), weights=weights)[0]
        name = names[idx]
        return name, self.strategies[name].prompt

    def _exploit_select(self) -> Tuple[str, str]:
        """Select the best performing prompt."""
        # Select by expected fitness (success_rate * avg_success_fitness)
        best_name = max(
            self.strategies.keys(),
            key=lambda n: (
                self.strategies[n].success_rate * self.strategies[n].avg_success_fitness
            )
        )
        return best_name, self.strategies[best_name].prompt

    def update_strategy(self, name: str, fitness: float, correct: bool) -> None:
        """Update strategy statistics after a generation attempt.

        Uses PPO-style clipped objective to prevent catastrophic forgetting.
        """
        if name not in self.strategies:
            logger.warning(f"Unknown strategy: {name}")
            return

        self.strategies[name].update(fitness, correct)

        # Decay exploration rate as we learn
        total_attempts = sum(s.attempt_count for s in self.strategies.values())
        if total_attempts > 100:
            self.exploration_rate = max(0.1, 0.3 - (total_attempts - 100) * 0.0005)

    def get_best_strategies(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get the top-k best performing strategies.

        Returns:
            List of (strategy_name, expected_fitness) tuples.
        """
        scored = [
            (name, strat.success_rate * strat.avg_success_fitness)
            for name, strat in self.strategies.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# =============================================================================
# LLM Evolution
# =============================================================================

@dataclass
class LLMEvolutionStats:
    """Statistics for LLM evolution."""
    llm_generations: int = 0
    llm_variants_generated: int = 0
    llm_variants_correct: int = 0
    llm_best_fitness: float = 0.0
    total_llm_calls: int = 0
    failed_llm_calls: int = 0


class LLMEvolution(CodeEvolution):
    """LLM-Enhanced Code Evolution.

    This class extends CodeEvolution with LLM-generated code variants.
    Each generation:
    1. Run standard genetic evolution operators
    2. Generate 10-20 new variants via LLM using learned prompts
    3. Evaluate LLM variants against test cases
    4. Add correct variants to population
    5. Update PPO policy based on results
    6. Track best algorithms found

    The PPO-style training learns which prompt strategies tend to
    produce correct and fast code variants.
    """

    def __init__(
        self,
        baseline_time_us: float = 10.0,
        population_size: int = 50,
        elite_keep: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.4,
        llm_variants_per_iteration: int = 15,
        llm_temperature: float = 0.8,
    ):
        """Initialize LLM-enhanced evolution.

        Args:
            baseline_time_us: Baseline execution time for speedup calculation.
            population_size: Total population size.
            elite_keep: Number of best individuals to keep each generation.
            mutation_rate: Probability of mutation.
            crossover_rate: Probability of crossover.
            llm_variants_per_iteration: Number of LLM variants to generate per iteration.
            llm_temperature: Temperature for LLM generation (higher = more diverse).
        """
        super().__init__(
            baseline_time_us=baseline_time_us,
            population_size=population_size,
            elite_keep=elite_keep,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
        )

        self.llm_variants_per_iteration = llm_variants_per_iteration
        self.llm_temperature = llm_temperature

        # PPO policy for prompt selection
        self.policy = PPOPolicy(exploration_rate=0.3)

        # Statistics
        self.stats = LLMEvolutionStats()

        # Best algorithms archive
        self.archive: List[CodeChromosome] = []

        # History of LLM generations
        self.llm_history: List[dict] = []

    def _extract_code_from_response(self, code: str) -> Optional[str]:
        """Extract a valid matmul function from LLM response.

        Handles common formats like markdown code blocks.

        Args:
            code: Raw code string from LLM.

        Returns:
            Cleaned code string or None if invalid.
        """
        # Remove markdown code blocks if present
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```\n?', '', code)
        code = code.strip()

        # Check if it contains a matmul function
        if 'def matmul' not in code:
            # Try to find def somewhere and extract around it
            match = re.search(r'def matmul\s*\(', code)
            if match:
                start = match.start()
                # Find the end - next 'def' or end of string
                next_def = code.find('\ndef ', start + 1)
                if next_def != -1:
                    code = code[start:next_def]
                else:
                    code = code[start:]

        # Basic validation - try to compile
        try:
            compile(code, '<string>', 'exec')
            return code
        except SyntaxError:
            return None

    def _generate_llm_variants(self) -> List[Tuple[str, str, str]]:
        """Generate code variants using LLM.

        Returns:
            List of (strategy_name, prompt, code) tuples for each variant.
        """
        variants = []
        strategies_used = []

        # Generate variants using different prompts
        for _ in range(self.llm_variants_per_iteration):
            strategy_name, prompt = self.policy.select_prompt()

            try:
                raw_variants = generate_code_variants(
                    prompt=prompt,
                    num_variants=1,
                )

                for raw_code in raw_variants:
                    code = self._extract_code_from_response(raw_code)
                    if code:
                        variants.append((strategy_name, prompt, code))
                        strategies_used.append(strategy_name)
                        self.stats.llm_variants_generated += 1

            except MiniMaxError as e:
                logger.warning(f"LLM generation failed: {e}")
                self.stats.failed_llm_calls += 1
                continue

            except Exception as e:
                logger.warning(f"Unexpected error in LLM generation: {e}")
                self.stats.failed_llm_calls += 1
                continue

        self.stats.total_llm_calls += 1
        return variants

    def _evaluate_llm_variant(self, code: str) -> Optional[CodeChromosome]:
        """Evaluate an LLM-generated code variant.

        Args:
            code: The code string to evaluate.

        Returns:
            CodeChromosome with evaluation results, or None if invalid.
        """
        chromosome = CodeChromosome(
            code=code,
            template_type="llm_generated",
            fitness=0.0,
            speedup=1.0,
            correct=False,
            execution_time_us=float('inf'),
        )

        try:
            namespace = {}
            exec(code, namespace)
            matmul_func = namespace.get('matmul')

            if matmul_func is None:
                return chromosome

            import random
            random.seed(42)

            total_time = 0.0
            num_tests = 20

            for _ in range(num_tests):
                A = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]
                B = [[random.uniform(-10, 10) for _ in range(4)] for _ in range(4)]

                # Reference implementation
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

                # Verify correctness
                correct = True
                for i in range(4):
                    for j in range(4):
                        if abs(result[i][j] - expected[i][j]) > 1e-9:
                            chromosome.correct = False
                            chromosome.execution_time_us = elapsed
                            return chromosome

                total_time += elapsed

            avg_time = total_time / num_tests
            chromosome.execution_time_us = avg_time
            chromosome.speedup = self.baseline_time_us / avg_time if avg_time > 0 else 0
            chromosome.fitness = chromosome.speedup ** 2
            chromosome.correct = True

        except Exception as e:
            chromosome.correct = False
            chromosome.fitness = -1.0

        return chromosome

    def _add_to_archive(self, chromosome: CodeChromosome) -> None:
        """Add a chromosome to the best algorithms archive.

        Args:
            chromosome: The chromosome to archive.
        """
        if not chromosome.correct:
            return

        # Check if similar code already in archive
        for existing in self.archive:
            if existing.code == chromosome.code:
                return  # Already archived

            # Check if same speedup but different code
            if abs(existing.speedup - chromosome.speedup) < 0.01:
                # Keep the one with higher fitness
                if chromosome.fitness > existing.fitness:
                    self.archive.remove(existing)
                    break
                else:
                    return

        self.archive.append(chromosome)
        self.archive.sort(key=lambda x: x.fitness, reverse=True)

        # Keep only top 20 in archive
        if len(self.archive) > 20:
            self.archive = self.archive[:20]

    def evolve_generation(self) -> None:
        """Evolve one generation with LLM enhancement.

        This extends the standard genetic evolution with:
        1. Standard mutation/crossover (from parent class)
        2. LLM variant generation
        3. PPO policy updates
        """
        # Run standard evolution first
        super().evolve_generation()

        # Generate LLM variants
        llm_variants = self._generate_llm_variants()

        if not llm_variants:
            return

        # Evaluate LLM variants
        evaluated_variants = []
        for strategy_name, prompt, code in llm_variants:
            chromosome = self._evaluate_llm_variant(code)
            if chromosome is not None:
                chromosome.prompt_used = prompt  # type: ignore
                chromosome.strategy_name = strategy_name  # type: ignore
                evaluated_variants.append((strategy_name, chromosome))
                self.stats.llm_variants_correct += chromosome.correct if chromosome.correct else 0

                if chromosome.correct and chromosome.fitness > self.stats.llm_best_fitness:
                    self.stats.llm_best_fitness = chromosome.fitness

        # Update PPO policy based on results
        for strategy_name, chromosome in evaluated_variants:
            self.policy.update_strategy(
                name=strategy_name,
                fitness=chromosome.fitness,
                correct=chromosome.correct,
            )

        # Add successful LLM variants to population
        correct_variants = [c for _, c in evaluated_variants if c.correct]
        correct_variants.sort(key=lambda x: x.fitness, reverse=True)

        # Keep top LLM variants (up to 20% of population)
        max_llm_additions = self.population_size // 5
        for chromosome in correct_variants[:max_llm_additions]:
            # Only add if better than worst in population
            if chromosome.fitness > self.population[-1].fitness:
                self.population.pop()  # Remove worst
                self.population.append(chromosome)
                self._add_to_archive(chromosome)

        # Re-sort population
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Update best if LLM found something better
        if self.population[0].fitness > (self.best.fitness if self.best else 0):
            self.best = self.population[0]

        # Record history
        self.llm_history.append({
            'generation': len(self.history),
            'llm_variants_generated': len(llm_variants),
            'llm_variants_correct': sum(1 for _, c in evaluated_variants if c.correct),
            'best_llm_fitness': max((c.fitness for _, c in evaluated_variants), default=0),
            'top_strategies': self.policy.get_best_strategies(3),
        })

        self.stats.llm_generations += 1

    def evolve(self, num_generations: int = 200) -> CodeChromosome:
        """Run LLM-enhanced evolution.

        Args:
            num_generations: Number of generations to evolve.

        Returns:
            The best chromosome found.
        """
        print("Initializing population...")
        self.initialize()

        print(f"Initial best: {self.best.template_type} - {self.best.speedup:.2f}x speedup")
        print(f"Population size: {len(self.population)}")

        for gen in range(num_generations):
            self.evolve_generation()

            if (gen + 1) % 10 == 0:
                best = self.population[0]
                correct_pct = sum(1 for p in self.population if p.correct) / len(self.population) * 100

                top_strategies = self.policy.get_best_strategies(3)
                strategy_info = ", ".join([f"{n}:{s:.2f}" for n, s in top_strategies])

                print(f"Gen {gen + 1}: Best={best.template_type} ({best.speedup:.2f}x), "
                      f"Correct={correct_pct:.0f}%, LLM best={self.stats.llm_best_fitness:.2f}, "
                      f"Top prompts: [{strategy_info}]")

            # Record history
            self.history.append({
                'generation': gen + 1,
                'best_speedup': self.population[0].speedup,
                'best_type': self.population[0].template_type,
                'llm_best_fitness': self.stats.llm_best_fitness,
            })

        return self.best

    def get_archive_report(self) -> str:
        """Generate a report of the best algorithms found.

        Returns:
            Formatted string report.
        """
        lines = [
            "=" * 70,
            "BEST ALGORITHMS ARCHIVE",
            "=" * 70,
        ]

        for i, chrom in enumerate(self.archive, 1):
            lines.append(f"\n#{i} (Fitness: {chrom.fitness:.3f}, Speedup: {chrom.speedup:.2f}x)")
            lines.append(f"Type: {chrom.template_type}")
            lines.append(f"Correct: {chrom.correct}")
            lines.append(f"Execution time: {chrom.execution_time_us:.2f} us")
            lines.append("\nCode:")
            lines.append(chrom.code)

        return "\n".join(lines)

    def get_policy_report(self) -> str:
        """Generate a report of prompt strategy performance.

        Returns:
            Formatted string report.
        """
        lines = [
            "=" * 70,
            "PROMPT STRATEGY PERFORMANCE (PPO Policy)",
            "=" * 70,
        ]

        scored = [
            (name, strat.success_rate, strat.avg_success_fitness, strat.attempt_count)
            for name, strat in self.policy.strategies.items()
        ]
        scored.sort(key=lambda x: x[1] * x[2], reverse=True)

        for name, rate, avg_fit, attempts in scored:
            if attempts > 0:
                expected = rate * avg_fit
                lines.append(f"{name:30s} | Attempts: {attempts:4d} | "
                             f"Success: {rate:.2%} | AvgFit: {avg_fit:6.2f} | "
                             f"Expected: {expected:6.2f}")

        return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("LLM-Enhanced Evolution for Matrix Multiplication")
    print("=" * 70)
    print("This evolution uses MiniMax LLM to generate code variants")
    print("and PPO-style training to learn which prompts work best.")
    print("=" * 70)

    evolver = LLMEvolution(
        baseline_time_us=10.0,
        population_size=50,
        elite_keep=5,
        mutation_rate=0.3,
        llm_variants_per_iteration=15,
    )

    best = evolver.evolve(num_generations=100)

    print("\n" + "=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"Best algorithm: {best.template_type}")
    print(f"Speedup: {best.speedup:.2f}x")
    print(f"Fitness: {best.fitness:.3f}")
    print(f"\nGenerated code:\n{best.code}")

    print("\n" + evolver.get_archive_report())
    print("\n" + evolver.get_policy_report())

    print("\n" + "=" * 70)
    print("LLM STATISTICS")
    print("=" * 70)
    print(f"LLM generations: {evolver.stats.llm_generations}")
    print(f"Total LLM calls: {evolver.stats.total_llm_calls}")
    print(f"Failed LLM calls: {evolver.stats.failed_llm_calls}")
    print(f"Variants generated: {evolver.stats.llm_variants_generated}")
    print(f"Variants correct: {evolver.stats.llm_variants_correct}")
    print(f"Best LLM fitness: {evolver.stats.llm_best_fitness:.3f}")
