"""Built-in machine-checkable tasks for the product workbench."""

from __future__ import annotations

from .models import EvaluationCase, TaskSpec


def _matrix(rows: int, columns: int, *, offset: int = 0) -> list[list[int]]:
    return [
        [((row * columns + column + offset) % 11) - 5 for column in range(columns)]
        for row in range(rows)
    ]


def _matrix_product(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [sum(a_row[index] * b[index][column] for index in range(len(b))) for column in range(len(b[0]))]
        for a_row in a
    ]


SORT_NUMBERS_INITIAL = """def solve(values):
    return list(values)
"""

FIND_FIRST_INITIAL = """def solve(values, target):
    return -1
"""

BINARY_SEARCH_INITIAL = """def solve(values, target):
    return -1
"""

TWO_SUM_INITIAL = """def solve(values, target):
    return []
"""

VALID_PARENTHESES_INITIAL = """def solve(text):
    return False
"""

FIBONACCI_INITIAL = """def solve(n):
    return 0
"""

GCD_INITIAL = """def solve(a, b):
    return 0
"""

IS_PRIME_INITIAL = """def solve(n):
    return False
"""

SIEVE_PRIMES_INITIAL = """def solve(n):
    return []
"""

FACTORIAL_INITIAL = """def solve(n):
    return 0
"""

REVERSE_STRING_INITIAL = """def solve(text):
    return text
"""

PALINDROME_INITIAL = """def solve(text):
    return False
"""

MERGE_INTERVALS_INITIAL = """def solve(intervals):
    return intervals
"""

MAX_SUBARRAY_INITIAL = """def solve(values):
    return 0
"""

LCS_INITIAL = """def solve(left, right):
    return 0
"""

EDIT_DISTANCE_INITIAL = """def solve(left, right):
    return 0
"""

KNAPSACK_INITIAL = """def solve(weights, values, capacity):
    return 0
"""

BFS_INITIAL = """def solve(graph, start):
    return []
"""

DIJKSTRA_INITIAL = """def solve(graph, start):
    return {}
"""

MATRIX_MULTIPLY_INITIAL = """def solve(a, b):
    if not a or not b:
        return []

    rows = len(a)
    shared = len(b)
    columns = len(b[0])
    result = []
    for row_index in range(rows):
        row = []
        for column_index in range(columns):
            total = 0
            for shared_index in range(shared):
                total += a[row_index][shared_index] * b[shared_index][column_index]
            row.append(total)
        result.append(row)
    return result
"""


_BUILTIN_TASKS: dict[str, TaskSpec] = {
    "sort_numbers": TaskSpec(
        task_id="sort_numbers",
        title="Sort Numbers",
        objective="Return a new list containing the input numbers in ascending order.",
        initial_program=SORT_NUMBERS_INITIAL,
        function_name="solve",
        constraints=(
            "Do not mutate the input list.",
            "Return a Python list.",
            "Handle empty lists and duplicate values.",
        ),
        tags=("algorithm-discovery", "sorting", "python"),
        cases=(
            EvaluationCase("sort_empty", ([],), [], "Empty input remains empty."),
            EvaluationCase("sort_three", ([3, 1, 2],), [1, 2, 3], "Unsorted values become ascending."),
            EvaluationCase("sort_duplicates", ([5, 1, 5, 2],), [1, 2, 5, 5], "Duplicates are preserved."),
            EvaluationCase("sort_negative", ([0, -4, 7, -1],), [-4, -1, 0, 7], "Negative values are ordered."),
        ),
    ),
    "find_first_index": TaskSpec(
        task_id="find_first_index",
        title="Find First Index",
        objective="Return the first index of target in values, or -1 when the target is absent.",
        initial_program=FIND_FIRST_INITIAL,
        function_name="solve",
        constraints=(
            "Return an integer index.",
            "Prefer the first matching index.",
            "Do not raise when the target is missing.",
        ),
        tags=("algorithm-discovery", "search", "python"),
        cases=(
            EvaluationCase("first_present", ([4, 9, 4], 4), 0, "Return the first matching index."),
            EvaluationCase("middle_present", ([1, 2, 3], 2), 1, "Find a middle value."),
            EvaluationCase("absent", ([1, 2, 3], 7), -1, "Return -1 for missing values."),
            EvaluationCase("empty", ([], 1), -1, "Empty inputs are safe."),
        ),
    ),
    "binary_search": TaskSpec(
        task_id="binary_search",
        title="Binary Search",
        objective="Return the index of target in a sorted list, or -1 if absent.",
        initial_program=BINARY_SEARCH_INITIAL,
        function_name="solve",
        constraints=(
            "Input values are sorted in ascending order.",
            "Return an integer index.",
            "Return -1 when the target is absent.",
        ),
        tags=("algorithm-discovery", "search", "classic"),
        cases=(
            EvaluationCase("middle", ([1, 3, 5, 7, 9], 5), 2),
            EvaluationCase("left_edge", ([1, 3, 5, 7, 9], 1), 0),
            EvaluationCase("right_edge", ([1, 3, 5, 7, 9], 9), 4),
            EvaluationCase("absent", ([1, 3, 5, 7, 9], 4), -1),
        ),
    ),
    "two_sum_indices": TaskSpec(
        task_id="two_sum_indices",
        title="Two Sum Indices",
        objective="Return the indices of two distinct values whose sum equals target.",
        initial_program=TWO_SUM_INITIAL,
        function_name="solve",
        constraints=(
            "Return a two-item list of indices.",
            "Do not reuse the same element twice.",
            "Return an empty list if no pair exists.",
        ),
        tags=("algorithm-discovery", "hashing", "python"),
        cases=(
            EvaluationCase("classic", ([2, 7, 11, 15], 9), [0, 1], "Classic two-sum example."),
            EvaluationCase("later_pair", ([3, 2, 4], 6), [1, 2], "Pair can appear after index 0."),
            EvaluationCase("duplicate_values", ([3, 3], 6), [0, 1], "Distinct indices can share a value."),
            EvaluationCase("no_pair", ([1, 2, 5], 20), [], "Return empty list when no pair exists."),
        ),
    ),
    "valid_parentheses": TaskSpec(
        task_id="valid_parentheses",
        title="Valid Parentheses",
        objective="Return True when bracket characters are balanced and correctly nested.",
        initial_program=VALID_PARENTHESES_INITIAL,
        function_name="solve",
        constraints=(
            "Support (), [], and {} brackets.",
            "Ignore no characters; inputs contain only brackets.",
            "Empty strings are valid.",
        ),
        tags=("algorithm-discovery", "stack", "classic"),
        cases=(
            EvaluationCase("empty", ("",), True),
            EvaluationCase("simple", ("()[]{}",), True),
            EvaluationCase("nested", ("({[]})",), True),
            EvaluationCase("wrong_order", ("([)]",), False),
            EvaluationCase("unclosed", ("(()",), False),
        ),
    ),
    "fibonacci": TaskSpec(
        task_id="fibonacci",
        title="Fibonacci",
        objective="Return the nth Fibonacci number with F(0)=0 and F(1)=1.",
        initial_program=FIBONACCI_INITIAL,
        function_name="solve",
        constraints=("Return an integer.", "Handle n=0.", "Use zero-based indexing."),
        tags=("algorithm-discovery", "dynamic-programming", "classic"),
        cases=(
            EvaluationCase("zero", (0,), 0),
            EvaluationCase("one", (1,), 1),
            EvaluationCase("seven", (7,), 13),
            EvaluationCase("ten", (10,), 55),
        ),
    ),
    "gcd": TaskSpec(
        task_id="gcd",
        title="Greatest Common Divisor",
        objective="Return the greatest common divisor of two integers.",
        initial_program=GCD_INITIAL,
        function_name="solve",
        constraints=("Return a non-negative integer.", "Handle zero inputs.", "Use exact integer arithmetic."),
        tags=("algorithm-discovery", "number-theory", "classic"),
        cases=(
            EvaluationCase("common", (54, 24), 6),
            EvaluationCase("coprime", (17, 13), 1),
            EvaluationCase("zero_left", (0, 9), 9),
            EvaluationCase("negative", (-24, 18), 6),
        ),
    ),
    "is_prime": TaskSpec(
        task_id="is_prime",
        title="Prime Check",
        objective="Return True when n is prime and False otherwise.",
        initial_program=IS_PRIME_INITIAL,
        function_name="solve",
        constraints=("Return a boolean.", "Numbers below 2 are not prime.", "Use exact integer logic."),
        tags=("algorithm-discovery", "number-theory", "classic"),
        cases=(
            EvaluationCase("one", (1,), False),
            EvaluationCase("two", (2,), True),
            EvaluationCase("composite", (49,), False),
            EvaluationCase("prime", (97,), True),
        ),
    ),
    "sieve_primes": TaskSpec(
        task_id="sieve_primes",
        title="Sieve of Eratosthenes",
        objective="Return all prime numbers less than or equal to n.",
        initial_program=SIEVE_PRIMES_INITIAL,
        function_name="solve",
        constraints=("Return a list of integers.", "Include n if n is prime.", "Return [] for n < 2."),
        tags=("algorithm-discovery", "number-theory", "classic"),
        cases=(
            EvaluationCase("below_two", (1,), []),
            EvaluationCase("ten", (10,), [2, 3, 5, 7]),
            EvaluationCase("two", (2,), [2]),
            EvaluationCase("twenty", (20,), [2, 3, 5, 7, 11, 13, 17, 19]),
        ),
    ),
    "factorial": TaskSpec(
        task_id="factorial",
        title="Factorial",
        objective="Return n factorial for non-negative integer n.",
        initial_program=FACTORIAL_INITIAL,
        function_name="solve",
        constraints=("Return an integer.", "0! is 1.", "Inputs are non-negative."),
        tags=("algorithm-discovery", "math", "classic"),
        cases=(
            EvaluationCase("zero", (0,), 1),
            EvaluationCase("one", (1,), 1),
            EvaluationCase("five", (5,), 120),
            EvaluationCase("seven", (7,), 5040),
        ),
    ),
    "reverse_string": TaskSpec(
        task_id="reverse_string",
        title="Reverse String",
        objective="Return the input string reversed.",
        initial_program=REVERSE_STRING_INITIAL,
        function_name="solve",
        constraints=("Return a string.", "Preserve whitespace.", "Handle empty strings."),
        tags=("algorithm-discovery", "strings", "classic"),
        cases=(
            EvaluationCase("empty", ("",), ""),
            EvaluationCase("word", ("alpha",), "ahpla"),
            EvaluationCase("spaces", ("a b",), "b a"),
            EvaluationCase("palindrome", ("level",), "level"),
        ),
    ),
    "palindrome_check": TaskSpec(
        task_id="palindrome_check",
        title="Palindrome Check",
        objective="Return True if the input string reads the same forward and backward.",
        initial_program=PALINDROME_INITIAL,
        function_name="solve",
        constraints=("Return a boolean.", "Use exact characters.", "Case-sensitive comparison is expected."),
        tags=("algorithm-discovery", "strings", "classic"),
        cases=(
            EvaluationCase("empty", ("",), True),
            EvaluationCase("odd", ("level",), True),
            EvaluationCase("even", ("abba",), True),
            EvaluationCase("not_palindrome", ("alpha",), False),
        ),
    ),
    "merge_intervals": TaskSpec(
        task_id="merge_intervals",
        title="Merge Intervals",
        objective="Merge overlapping intervals and return sorted non-overlapping intervals.",
        initial_program=MERGE_INTERVALS_INITIAL,
        function_name="solve",
        constraints=("Return a list of [start, end] lists.", "Merge touching intervals.", "Sort by start."),
        tags=("algorithm-discovery", "intervals", "classic"),
        cases=(
            EvaluationCase("empty", ([],), []),
            EvaluationCase("overlap", ([[1, 3], [2, 6], [8, 10]],), [[1, 6], [8, 10]]),
            EvaluationCase("touching", ([[1, 4], [4, 5]],), [[1, 5]]),
            EvaluationCase("unsorted", ([[5, 7], [1, 2], [2, 3]],), [[1, 3], [5, 7]]),
        ),
    ),
    "max_subarray_sum": TaskSpec(
        task_id="max_subarray_sum",
        title="Maximum Subarray Sum",
        objective="Return the maximum sum over all contiguous non-empty subarrays.",
        initial_program=MAX_SUBARRAY_INITIAL,
        function_name="solve",
        constraints=("Return an integer.", "The subarray must be non-empty.", "Handle all-negative arrays."),
        tags=("algorithm-discovery", "dynamic-programming", "classic"),
        cases=(
            EvaluationCase("mixed", ([-2, 1, -3, 4, -1, 2, 1, -5, 4],), 6),
            EvaluationCase("all_negative", ([-8, -3, -6],), -3),
            EvaluationCase("single", ([5],), 5),
            EvaluationCase("all_positive", ([1, 2, 3],), 6),
        ),
    ),
    "longest_common_subsequence": TaskSpec(
        task_id="longest_common_subsequence",
        title="Longest Common Subsequence",
        objective="Return the length of the longest common subsequence of two strings.",
        initial_program=LCS_INITIAL,
        function_name="solve",
        constraints=("Return an integer length.", "Subsequence characters need not be contiguous.", "Handle empty strings."),
        tags=("algorithm-discovery", "dynamic-programming", "classic"),
        cases=(
            EvaluationCase("classic", ("abcde", "ace"), 3),
            EvaluationCase("same", ("abc", "abc"), 3),
            EvaluationCase("none", ("abc", "def"), 0),
            EvaluationCase("empty", ("", "abc"), 0),
        ),
    ),
    "edit_distance": TaskSpec(
        task_id="edit_distance",
        title="Edit Distance",
        objective="Return the Levenshtein distance between two strings.",
        initial_program=EDIT_DISTANCE_INITIAL,
        function_name="solve",
        constraints=("Return an integer.", "Allowed operations are insert, delete, and replace.", "Each operation costs 1."),
        tags=("algorithm-discovery", "dynamic-programming", "classic"),
        cases=(
            EvaluationCase("kitten_sitting", ("kitten", "sitting"), 3),
            EvaluationCase("horse_ros", ("horse", "ros"), 3),
            EvaluationCase("empty", ("", "abc"), 3),
            EvaluationCase("same", ("abc", "abc"), 0),
        ),
    ),
    "knapsack_01": TaskSpec(
        task_id="knapsack_01",
        title="0/1 Knapsack",
        objective="Return the maximum value achievable without exceeding capacity.",
        initial_program=KNAPSACK_INITIAL,
        function_name="solve",
        constraints=("Each item can be used at most once.", "Return an integer.", "Inputs are parallel weights and values lists."),
        tags=("algorithm-discovery", "dynamic-programming", "classic"),
        cases=(
            EvaluationCase("classic", ([2, 3, 4, 5], [3, 4, 5, 6], 5), 7),
            EvaluationCase("capacity_zero", ([1, 2], [10, 20], 0), 0),
            EvaluationCase("single_fit", ([5], [9], 5), 9),
            EvaluationCase("single_too_heavy", ([6], [9], 5), 0),
        ),
    ),
    "bfs_order": TaskSpec(
        task_id="bfs_order",
        title="Breadth-First Search",
        objective="Return nodes in breadth-first order from the start node.",
        initial_program=BFS_INITIAL,
        function_name="solve",
        constraints=("Graph is an adjacency dictionary.", "Visit neighbors in listed order.", "Return a list of visited nodes."),
        tags=("algorithm-discovery", "graphs", "classic"),
        cases=(
            EvaluationCase("tree", ({"A": ["B", "C"], "B": ["D"], "C": [], "D": []}, "A"), ["A", "B", "C", "D"]),
            EvaluationCase("cycle", ({1: [2, 3], 2: [1, 4], 3: [], 4: []}, 1), [1, 2, 3, 4]),
            EvaluationCase("missing_start", ({"A": ["B"]}, "Z"), ["Z"]),
        ),
    ),
    "dijkstra_shortest_path": TaskSpec(
        task_id="dijkstra_shortest_path",
        title="Dijkstra Shortest Paths",
        objective="Return shortest path distances from the start node in a weighted graph.",
        initial_program=DIJKSTRA_INITIAL,
        function_name="solve",
        constraints=("Graph maps node to (neighbor, weight) pairs.", "Weights are non-negative.", "Return a distance dictionary."),
        tags=("algorithm-discovery", "graphs", "classic"),
        cases=(
            EvaluationCase(
                "weighted",
                ({"A": [("B", 1), ("C", 4)], "B": [("C", 2), ("D", 5)], "C": [("D", 1)], "D": []}, "A"),
                {"A": 0, "B": 1, "C": 3, "D": 4},
            ),
            EvaluationCase("single", ({"A": []}, "A"), {"A": 0}),
            EvaluationCase("unreachable_omitted", ({"A": [("B", 2)], "B": [], "C": []}, "A"), {"A": 0, "B": 2}),
        ),
    ),
    "matrix_multiply": TaskSpec(
        task_id="matrix_multiply",
        title="Matrix Multiplication Optimization",
        objective=(
            "Implement solve(a, b) for matrix multiplication. Preserve exact "
            "integer results while making the pure-Python implementation faster."
        ),
        initial_program=MATRIX_MULTIPLY_INITIAL,
        baseline_program=MATRIX_MULTIPLY_INITIAL,
        benchmark_repetitions=20,
        function_name="solve",
        metric="correctness_speed",
        constraints=(
            "Return a nested Python list.",
            "Do not import numpy or external libraries.",
            "Support rectangular matrices with compatible dimensions.",
            "Correctness is mandatory before speed is rewarded.",
        ),
        tags=("algorithm-discovery", "matrix-multiply", "optimization", "python"),
        cases=(
            EvaluationCase(
                "identity_2x2",
                ([[1, 2], [3, 4]], [[1, 0], [0, 1]]),
                [[1, 2], [3, 4]],
                "Multiplication by identity preserves the matrix.",
            ),
            EvaluationCase(
                "rectangular_2x3_3x2",
                ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]),
                [[58, 64], [139, 154]],
                "Rectangular matrices are supported.",
            ),
            EvaluationCase(
                "negative_values",
                ([[2, -1], [0, 3]], [[4, 5], [-2, 1]]),
                [[10, 9], [-6, 3]],
                "Negative values are handled exactly.",
            ),
            EvaluationCase(
                "larger_4x4",
                (
                    [[1, 2, 3, 4], [2, 0, 1, 3], [5, 1, 2, 0], [0, 2, 4, 1]],
                    [[3, 1, 0, 2], [1, 4, 2, 0], [0, 5, 1, 3], [2, 0, 4, 1]],
                ),
                [[13, 24, 23, 15], [12, 7, 13, 10], [16, 19, 4, 16], [4, 28, 12, 13]],
                "A larger deterministic case makes performance differences visible.",
            ),
            EvaluationCase(
                "benchmark_12x12",
                (_matrix(12, 12, offset=1), _matrix(12, 12, offset=4)),
                _matrix_product(_matrix(12, 12, offset=1), _matrix(12, 12, offset=4)),
                "A repeated 12x12 case provides a stronger speed signal.",
            ),
        ),
    ),
}


def list_builtin_tasks() -> tuple[TaskSpec, ...]:
    """Return all built-in tasks in stable product order."""
    return tuple(_BUILTIN_TASKS.values())


def get_builtin_task(task_id: str) -> TaskSpec:
    """Return a built-in task by id."""
    try:
        return _BUILTIN_TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task_id}") from exc
