"""AlphaEvolve CLI - End-to-end LLM-driven program evolution.

Usage:
    python -m run.alphaevolve --problem matrix_multiply --generations 20
    python -m run.alphaevolve --problem steiner_tree --generations 50
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from alphaevolve.llm_evolution import (
    ProgramCandidate,
    ProgramDatabase,
    SimplePromptSampler,
    run_evolution_loop,
    create_minimax_proposer,
    create_evaluator,
)


# =============================================================================
# Problem Definitions
# =============================================================================

PROBLEMS = {
    "matrix_multiply": {
        "initial_code": '''def matmul(A, B):
    """4x4 matrix multiplication."""
    C = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]
    return C''',
        "evaluator": "matrix_multiply",
    },
    "steiner_tree": {
        "initial_code": '''def steiner_tree(terminals):
    """Euclidean Steiner Tree approximation."""
    if len(terminals) < 2:
        return 0.0
    import math
    total = 0.0
    for i in range(len(terminals)):
        for j in range(i + 1, len(terminals)):
            x1, y1 = terminals[i]
            x2, y2 = terminals[j]
            total += math.hypot(x2 - x1, y2 - y1)
    return total''',
        "evaluator": "steiner_tree",
    },
}


# =============================================================================
# Evaluators
# =============================================================================

def evaluate_matrix_multiply(program: str) -> float:
    """Evaluate matrix multiplication program.

    Returns higher-is-better score.
    Speedup relative to baseline is used as score.
    """
    import random
    import time

    try:
        namespace = {}
        exec(program, namespace)
        matmul = namespace.get("matmul")
        if matmul is None:
            return -1.0

        baseline_time = 10.0  # microseconds

        # Test correctness and speed
        total_time = 0.0
        for _ in range(20):
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

            # Time candidate
            start = time.perf_counter()
            result = matmul(A, B)
            elapsed = (time.perf_counter() - start) * 1_000_000

            # Verify correctness
            for i in range(4):
                for j in range(4):
                    if abs(result[i][j] - expected[i][j]) > 1e-9:
                        return -1.0

            total_time += elapsed

        avg_time = total_time / 20
        speedup = baseline_time / max(avg_time, 0.001)

        return speedup ** 2  # Quadratic reward

    except Exception:
        return -1.0


def evaluate_steiner_tree(program: str) -> float:
    """Evaluate Steiner tree program."""
    import math
    import random

    try:
        namespace = {}
        exec(program, namespace)
        steiner = namespace.get("steiner_tree")
        if steiner is None:
            return -1.0

        # Test on random terminal sets
        total_error = 0.0
        for _ in range(10):
            terminals = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(random.randint(3, 6))]

            result = steiner(terminals)

            if result < 0:
                return -1.0

            # Simple upper bound: MST length
            def mst_length(points):
                if len(points) < 2:
                    return 0.0
                total = 0.0
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        total += math.hypot(x2 - x1, y2 - y1)
                return total

            mst = mst_length(terminals)
            if result > mst:
                return -1.0

            total_error += (mst - result) / mst

        avg_error = total_error / 10
        return 1.0 + avg_error  # Higher is better

    except Exception:
        return -1.0


EVALUATORS = {
    "matrix_multiply": evaluate_matrix_multiply,
    "steiner_tree": evaluate_steiner_tree,
}


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="AlphaEvolve: LLM-driven program evolution")
    parser.add_argument(
        "--problem",
        choices=list(PROBLEMS.keys()),
        default="matrix_multiply",
        help="Problem to solve",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of evolution generations",
    )
    parser.add_argument(
        "--archive-size",
        type=int,
        default=16,
        help="Maximum archive size",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature for generation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    problem_config = PROBLEMS[args.problem]
    evaluator = EVALUATORS[args.problem]

    print(f"AlphaEvolve - {args.problem}")
    print("=" * 50)
    print(f"Generations: {args.generations}")
    print(f"Archive size: {args.archive_size}")
    print(f"Temperature: {args.temperature}")
    print()

    # Initialize LLM proposer
    try:
        proposer = create_minimax_proposer(temperature=args.temperature)
        print("MiniMax LLM proposer initialized")
    except ValueError as e:
        print(f"Error: {e}")
        print("Set MINIMAX_API_KEY environment variable")
        return 1

    # Create prompt sampler
    sampler = SimplePromptSampler()

    # Run evolution
    print(f"\nRunning evolution...")
    start_time = time.time()

    try:
        result = run_evolution_loop(
            initial_program=problem_config["initial_code"],
            proposer=proposer,
            evaluator=create_evaluator(evaluator),
            generations=args.generations,
            archive_size=args.archive_size,
        )
    except Exception as e:
        print(f"Evolution failed: {e}")
        return 1

    elapsed = time.time() - start_time

    # Report results
    print(f"\nEvolution complete in {elapsed:.1f}s")
    print(f"\nBest program (score={result.best.score:.4f}):")
    print("-" * 50)
    print(result.best.program)
    print("-" * 50)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/alphaevolve_{args.problem}_{datetime.now():%Y%m%d_%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best program
    (output_dir / "best_program.py").write_text(result.best.program)

    # Save full results
    results_data = {
        "problem": args.problem,
        "generations": args.generations,
        "elapsed_seconds": elapsed,
        "best_score": result.best.score,
        "best_program": result.best.program,
        "history": [
            {
                "generation": step.generation,
                "parent_id": step.parent_id,
                "score": step.candidate.score,
            }
            for step in result.history
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(results_data, indent=2))

    print(f"\nResults saved to {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
