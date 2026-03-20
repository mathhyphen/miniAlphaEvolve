"""Test script for LLM-enhanced AlphaEvolve on 4x4 matrix multiplication.

This script runs the LLM evolution for a specified number of generations
and reports the best algorithm found along with its speedup.
"""

import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alphaevolve.evolution.llm_evolution import LLMEvolution


def run_test(num_generations: int = 4) -> dict:
    """Run the LLM-enhanced evolution test.

    Args:
        num_generations: Number of generations to evolve.

    Returns:
        Dictionary with results and statistics.
    """
    print("=" * 70)
    print("LLM-Enhanced AlphaEvolve Test - 4x4 Matrix Multiplication")
    print("=" * 70)

    # Initialize evolution
    evolver = LLMEvolution(
        baseline_time_us=10.0,
        population_size=25,
        elite_keep=3,
        mutation_rate=0.3,
        crossover_rate=0.4,
        llm_variants_per_iteration=2,
        llm_temperature=0.8,
    )

    print(f"Running for {num_generations} generations with {evolver.llm_variants_per_iteration} LLM variants per generation...")
    print()

    start_time = time.time()

    # Run evolution
    best = evolver.evolve(num_generations=num_generations)

    elapsed = time.time() - start_time

    # Collect results
    results = {
        "best_algorithm": best.template_type,
        "best_speedup": best.speedup,
        "best_fitness": best.fitness,
        "best_code": best.code,
        "correct": best.correct,
        "execution_time_us": best.execution_time_us,
        "generations": num_generations,
        "total_time_seconds": elapsed,
        "llm_stats": {
            "llm_generations": evolver.stats.llm_generations,
            "total_llm_calls": evolver.stats.total_llm_calls,
            "failed_llm_calls": evolver.stats.failed_llm_calls,
            "variants_generated": evolver.stats.llm_variants_generated,
            "variants_correct": evolver.stats.llm_variants_correct,
            "best_llm_fitness": evolver.stats.llm_best_fitness,
        },
    }

    return results


def print_results(results: dict) -> None:
    """Print formatted results."""
    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Generations run: {results['generations']}")
    print(f"Total time: {results['total_time_seconds']:.2f} seconds")
    print()
    print("-" * 70)
    print("BEST ALGORITHM FOUND")
    print("-" * 70)
    print(f"Name: {results['best_algorithm']}")
    print(f"Speedup: {results['best_speedup']:.4f}x")
    print(f"Fitness: {results['best_fitness']:.4f}")
    print(f"Correct: {results['correct']}")
    print(f"Execution time: {results['execution_time_us']:.4f} us")
    print()
    print("-" * 70)
    print("CODE")
    print("-" * 70)
    print(results['best_code'])
    print()
    print("-" * 70)
    print("LLM STATISTICS")
    print("-" * 70)
    stats = results['llm_stats']
    print(f"LLM generations: {stats['llm_generations']}")
    print(f"Total LLM calls: {stats['total_llm_calls']}")
    print(f"Failed LLM calls: {stats['failed_llm_calls']}")
    print(f"Variants generated: {stats['variants_generated']}")
    print(f"Variants correct: {stats['variants_correct']}")
    if stats['variants_generated'] > 0:
        pct = stats['variants_correct'] / stats['variants_generated'] * 100
        print(f"Success rate: {pct:.1f}%")
    print(f"Best LLM fitness: {stats['best_llm_fitness']:.4f}")


if __name__ == "__main__":
    # Check API key
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("ERROR: MINIMAX_API_KEY environment variable not set!")
        print("Please set it before running this test.")
        sys.exit(1)

    # Run test (using 4 generations to keep runtime reasonable)
    results = run_test(num_generations=4)
    print_results(results)
