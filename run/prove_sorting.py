"""Bounded proof demo for sorting algorithms.

This is a proof-assistant style workflow, not a full formal proof.
It checks a candidate sorting implementation against:

- concrete examples,
- exhaustive arrays up to a bounded size,
- contract-style postconditions.

Usage:
    python -m run.prove_sorting
    python -m run.prove_sorting --candidate path/to/candidate.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from alphaevolve.proof import (
    ExampleCase,
    ProofHarness,
    PropertyCase,
    is_sorted,
    preserves_multiset,
)
from alphaevolve.sandbox.verifier import Contract


DEFAULT_CANDIDATE = """def sort_list(items):
    result = list(items)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]
    return result
"""


def bounded_arrays(max_length: int, values: Iterable[int]) -> List[List[int]]:
    """Enumerate all arrays over a bounded finite domain."""

    value_list = list(values)
    all_arrays: List[List[int]] = [[]]
    frontier: List[List[int]] = [[]]

    for _ in range(max_length):
        next_frontier: List[List[int]] = []
        for array in frontier:
            for value in value_list:
                next_frontier.append(array + [value])
        all_arrays.extend(next_frontier)
        frontier = next_frontier

    return all_arrays


def build_sorting_harness() -> ProofHarness:
    """Build a bounded proof harness for sorting algorithms."""

    harness = ProofHarness("sort_list")

    harness.add_example_obligation(
        name="sorting-examples",
        description="Canonical sorting examples must match expected outputs.",
        cases=[
            ExampleCase(args=([3, 1, 2],), expected=[1, 2, 3], label="basic"),
            ExampleCase(args=([],), expected=[], label="empty"),
            ExampleCase(args=([2, 2, 1],), expected=[1, 2, 2], label="duplicates"),
        ],
    )

    bounded_cases = [
        PropertyCase(args=(array,), label=f"n={len(array)}:{index}")
        for index, array in enumerate(bounded_arrays(max_length=4, values=range(3)))
    ]

    harness.add_property_obligation(
        name="sorted-output",
        description="Output must be non-decreasing.",
        cases=bounded_cases,
        predicate=lambda result, args, kwargs: (
            is_sorted(result),
            "output is sorted" if is_sorted(result) else "output is not sorted",
            {"output": result},
        ),
    )

    harness.add_property_obligation(
        name="multiset-preservation",
        description="Sorting must preserve the input multiset.",
        cases=bounded_cases,
        predicate=lambda result, args, kwargs: (
            preserves_multiset(args[0], result),
            "multiset preserved" if preserves_multiset(args[0], result) else "multiset changed",
            {"input": args[0], "output": result},
        ),
    )

    harness.add_contract_obligation(
        name="sorting-contract",
        description="Basic postconditions for sorting.",
        contract=Contract(
            name="sorting-contract",
            description="Postconditions for sort_list",
            preconditions=[lambda items: isinstance(items, list)],
            postconditions=[
                lambda result, items: isinstance(result, list),
                lambda result, items: len(result) == len(items),
            ],
        ),
        cases=bounded_cases,
    )

    return harness


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded proof harness for sorting algorithms")
    parser.add_argument("--candidate", type=str, default=None, help="Path to candidate Python file")
    args = parser.parse_args()

    if args.candidate:
        candidate_code = Path(args.candidate).read_text(encoding="utf-8")
    else:
        candidate_code = DEFAULT_CANDIDATE

    harness = build_sorting_harness()
    summary = harness.verify(candidate_code)

    print("Sorting Proof Summary")
    print("=" * 60)
    print(f"Passed: {summary.passed}")
    print(f"Score: {summary.score:.3f}")
    print(f"Checks: {summary.passed_checks}/{summary.total_checks}")
    print(f"Execution time: {summary.execution_time_ms:.1f} ms")

    if summary.counterexamples:
        print("\nCounterexamples:")
        for counterexample in summary.counterexamples[:5]:
            print(
                f"- [{counterexample.obligation_kind}] {counterexample.obligation_name} "
                f"{counterexample.case_label}: {counterexample.message}"
            )
            if counterexample.error:
                print(f"  error={counterexample.error}")
            else:
                print(f"  args={counterexample.args}, output={counterexample.output}")
    else:
        print("\nNo counterexamples found in the bounded domain.")

    if summary.passed:
        print("\nThis is bounded verification evidence, not a universal formal proof.")

    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
