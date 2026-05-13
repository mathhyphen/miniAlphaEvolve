"""Enhanced sorting algorithm evolution test with more mutation strategies."""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from alphaevolve.llm_evolution import (
    ProgramCandidate,
    ProgramDatabase,
)


# =============================================================================
# Baseline Sorting Algorithm
# =============================================================================

BASELINE_SORT = '''def sort(arr):
    """Bubble sort - baseline algorithm."""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr'''


# =============================================================================
# Evaluator
# =============================================================================

def evaluate_sorting(program: str) -> float:
    """Evaluate sorting algorithm.

    Returns higher-is-better score based on speedup over baseline.
    """
    try:
        namespace = {}
        exec(program, namespace)
        sort_fn = namespace.get("sort")
        if sort_fn is None:
            return -1.0

        # Baseline time (microseconds for small arrays)
        baseline_time = 500.0

        # Test correctness and speed on various array sizes
        total_time = 0.0
        sizes = [10, 50, 100, 500, 1000]

        for size in sizes:
            for _ in range(10):  # 10 tests per size
                # Generate random array
                arr = [random.uniform(-1000, 1000) for _ in range(size)]
                expected = sorted(arr)

                # Time candidate
                start = time.perf_counter()
                result = sort_fn(arr.copy())
                elapsed = (time.perf_counter() - start) * 1_000_000

                # Verify correctness
                if result != expected:
                    return -1.0

                total_time += elapsed

        avg_time = total_time / 50
        speedup = baseline_time / max(avg_time, 0.001)

        return speedup ** 2

    except Exception as e:
        print(f"Evaluation error: {e}")
        return -1.0


# =============================================================================
# Mutation Strategies
# =============================================================================

MUTATIONS = [
    # Bubble sort optimizations
    ("n_minus_1", "for i in range(n):", "for i in range(n - 1):"),
    ("n_minus_2", "for i in range(n):", "for i in range(n - 2):"),
    ("early_exit_swap", "for i in range(n):\n        for j in range(0, n - i - 1):", "swapped = False\n    for i in range(n):\n        swapped = False\n        for j in range(0, n - i - 1):"),
    ("track_swaps", "    for j in range(0, n - i - 1):", "    for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1] or arr[j] == arr[j + 1]:"),
    ("bidirectional", "for j in range(0, n - i - 1):", "for j in range(0, n - i - 1):\n        if j > 0 and arr[j] < arr[j - 1]:\n            arr[j], arr[j - 1] = arr[j - 1], arr[j]"),
    # Comparison changes
    ("use_getitem", "arr[j]", "arr[j] if j < len(arr) else arr[-1]"),
    # Insertion sort hybrid
    ("insertion_inner", "    for j in range(0, n - i - 1):", "    key = arr[j + 1]\n    k = j\n    while k >= 0 and arr[k] > key:\n        arr[k + 1] = arr[k]\n        k -= 1\n    arr[k + 1] = key\nfor j in range(0, 0):"),
]


# =============================================================================
# Evolution with Enhanced Mutations
# =============================================================================

def run_evolution(initial_program: str, generations: int = 20) -> dict:
    """Run evolution with enhanced mutation strategies."""

    archive = ProgramDatabase(max_size=32)

    # Baseline candidate
    initial_candidate = ProgramCandidate(
        program=initial_program,
        score=evaluate_sorting(initial_program),
        generation=0,
        candidate_id="gen-0",
    )
    archive.add(initial_candidate)

    history = []
    best_ever = initial_candidate.score

    print(f"Initial score: {initial_candidate.score:.4f}")
    print("=" * 60)

    for generation in range(1, generations + 1):
        parent = archive.best()
        best_child = None
        best_child_score = parent.score

        # Try each mutation
        for mut_name, old, new in MUTATIONS:
            mutated = parent.program.replace(old, new, 1)
            if mutated != parent.program:  # Mutation applied
                score = evaluate_sorting(mutated)
                if score > best_child_score:
                    best_child_score = score
                    best_child = mutated
                    improvement = (score / parent.score - 1) * 100
                    print(f"Gen {generation:2d}: [{mut_name}] Score: {score:.4f} ({improvement:+.1f}%)")

        if best_child is not None:
            child_candidate = ProgramCandidate(
                program=best_child,
                score=best_child_score,
                generation=generation,
                candidate_id=f"gen-{generation}",
                parent_id=parent.candidate_id,
            )
            archive.add(child_candidate)
            history.append({
                "generation": generation,
                "score": best_child_score,
                "improvement": best_child_score - parent.score,
            })
            best_ever = max(best_ever, best_child_score)
        else:
            history.append({
                "generation": generation,
                "score": parent.score,
                "improvement": 0,
            })
            if generation % 5 == 0:
                print(f"Gen {generation:2d}: No improvement (best: {parent.score:.4f})")

    return {
        "archive": archive,
        "history": history,
        "best": archive.best(),
    }


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("AlphaEvolve Enhanced Sorting Evolution Test")
    print("=" * 60)
    print()
    print("Testing various mutation strategies on bubble sort...")
    print()

    # Test baseline
    baseline_score = evaluate_sorting(BASELINE_SORT)
    print(f"Baseline bubble sort score: {baseline_score:.4f}")
    print()

    # Run evolution
    result = run_evolution(BASELINE_SORT, generations=20)

    print()
    print("=" * 60)
    print(f"Evolution complete!")
    print(f"Best score: {result['best'].score:.4f}")
    print(f"Improvement: {(result['best'].score / baseline_score - 1) * 100:.1f}%")
    print()
    print("Best program found:")
    print("-" * 60)
    print(result['best'].program)
    print("-" * 60)

    # Save results
    output_dir = Path(f"outputs/sorting_evolution_{datetime.now():%Y%m%d_%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "baseline_score": baseline_score,
        "best_score": result['best'].score,
        "improvement_percent": (result['best'].score / baseline_score - 1) * 100,
        "best_program": result['best'].program,
        "history": result['history'],
    }
    (output_dir / "results.json").write_text(json.dumps(results_data, indent=2))
    (output_dir / "best_program.py").write_text(result['best'].program)

    print(f"\nResults saved to {output_dir}")

    # Summary
    improvements = [h for h in result['history'] if h['improvement'] > 0]
    print(f"\nSummary: {len(improvements)} improvements found in {len(result['history'])} generations")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
