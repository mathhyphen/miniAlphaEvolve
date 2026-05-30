from __future__ import annotations

import pytest

from alphaevolve.product import (
    AlphaEvolveWorkbench,
    EvaluationCase,
    Proposal,
    TaskSpec,
    get_builtin_task,
    list_builtin_tasks,
)
from alphaevolve.product.proposers import HeuristicProductProposer, _extract_program


CLASSIC_ALGORITHM_TASK_IDS = (
    "sort_numbers",
    "find_first_index",
    "binary_search",
    "two_sum_indices",
    "valid_parentheses",
    "fibonacci",
    "gcd",
    "is_prime",
    "sieve_primes",
    "factorial",
    "reverse_string",
    "palindrome_check",
    "merge_intervals",
    "max_subarray_sum",
    "longest_common_subsequence",
    "edit_distance",
    "knapsack_01",
    "bfs_order",
    "dijkstra_shortest_path",
    "matrix_multiply",
)


def test_builtin_task_evaluator_scores_initial_and_improved_programs() -> None:
    task = get_builtin_task("sort_numbers")

    initial_report = task.create_evaluator().evaluate(task.initial_program)
    improved_report = task.create_evaluator().evaluate(
        "def solve(values):\n"
        "    return sorted(values)\n"
    )

    assert initial_report.score < 1.0
    assert initial_report.passed_cases < initial_report.total_cases
    assert improved_report.score == pytest.approx(1.0)
    assert improved_report.failures == ()


def test_correctness_speed_metric_rewards_faster_correct_programs() -> None:
    slow_program = (
        "def solve(value):\n"
        "    import time\n"
        "    time.sleep(0.02)\n"
        "    return value * 2\n"
    )
    fast_program = "def solve(value):\n    return value * 2\n"
    task = TaskSpec(
        task_id="speed_double",
        title="Speed Double",
        objective="Return twice the input as quickly as possible.",
        initial_program=slow_program,
        baseline_program=slow_program,
        function_name="solve",
        metric="correctness_speed",
        cases=(
            EvaluationCase("positive", (4,), 8),
            EvaluationCase("zero", (0,), 0),
        ),
    )

    evaluator = task.create_evaluator()
    slow_report = evaluator.evaluate(slow_program)
    fast_report = evaluator.evaluate(fast_program)

    assert slow_report.passed_cases == 2
    assert fast_report.passed_cases == 2
    assert fast_report.score > slow_report.score
    assert fast_report.metrics["speedup"] > 1.0


def test_matrix_multiply_builtin_task_has_correctness_speed_contract() -> None:
    task = get_builtin_task("matrix_multiply")

    baseline_report = task.create_evaluator().evaluate(task.initial_program)
    optimized_report = task.create_evaluator().evaluate(
        "def solve(a, b):\n"
        "    columns = list(zip(*b))\n"
        "    return [[sum(left * right for left, right in zip(row, column)) for column in columns] for row in a]\n"
    )

    assert baseline_report.passed_cases == baseline_report.total_cases
    assert optimized_report.passed_cases == optimized_report.total_cases
    assert baseline_report.metrics["metric"] == "correctness_speed"


def test_builtin_suite_contains_20_classic_algorithm_contracts() -> None:
    tasks = {task.task_id: task for task in list_builtin_tasks()}

    assert len(CLASSIC_ALGORITHM_TASK_IDS) == 20
    assert set(CLASSIC_ALGORITHM_TASK_IDS).issubset(tasks)
    for task_id in CLASSIC_ALGORITHM_TASK_IDS:
        task = tasks[task_id]
        assert task.objective
        assert task.initial_program.startswith("def solve")
        assert len(task.cases) >= 2


def test_20_classic_algorithms_evolve_to_passing_candidates() -> None:
    workbench = AlphaEvolveWorkbench(proposer=HeuristicProductProposer())
    failures: list[str] = []

    for task_id in CLASSIC_ALGORITHM_TASK_IDS:
        run = workbench.start_run(task_id=task_id, generations=1, archive_size=4)
        assert run.best_candidate is not None
        report = run.best_candidate.evaluation
        if report.passed_cases != report.total_cases:
            failures.append(
                f"{task_id}: passed {report.passed_cases}/{report.total_cases}; "
                f"failures={report.failures}"
            )

    assert failures == []


def test_workbench_run_records_archive_lineage_prompts_and_best_program() -> None:
    workbench = AlphaEvolveWorkbench()

    run = workbench.start_run(
        task_id="sort_numbers",
        generations=3,
        archive_size=4,
    )

    assert run.status == "completed"
    assert run.current_generation == 3
    assert run.best_candidate is not None
    assert run.best_candidate.score == pytest.approx(1.0)
    assert "sorted" in run.best_candidate.program
    assert len(run.history) == 4
    assert len(run.archive) <= 4

    first_child = run.history[1]
    assert first_child.parent_id == run.history[0].candidate_id
    assert "Task objective:" in first_child.prompt
    assert "Archive inspirations:" in first_child.prompt
    assert first_child.evaluation.total_cases == 4


def test_workbench_rejects_unknown_tasks_and_excessive_generations() -> None:
    workbench = AlphaEvolveWorkbench()

    with pytest.raises(ValueError, match="Unknown task"):
        workbench.start_run(task_id="missing", generations=1)

    with pytest.raises(ValueError, match="generations"):
        workbench.start_run(task_id="sort_numbers", generations=101)


def test_workbench_can_retrieve_candidate_and_export_best_program() -> None:
    workbench = AlphaEvolveWorkbench()
    run = workbench.start_run(task_id="find_first_index", generations=2)

    assert run.best_candidate is not None
    candidate = workbench.get_candidate(run.run_id, run.best_candidate.candidate_id)
    exported = workbench.export_best_program(run.run_id)

    assert candidate.candidate_id == run.best_candidate.candidate_id
    assert exported == run.best_candidate.program
    assert "def solve" in exported


def test_workbench_runs_user_defined_task_with_evaluator_cases_in_prompt() -> None:
    class CustomProposer:
        def propose(self, task, parent, archive, prompt):
            del task, parent, archive
            assert "Evaluator cases:" in prompt
            assert "double_positive" in prompt
            return Proposal(
                text="set:def solve(value):\n    return value * 2\n",
                source="test-proposer",
            )

    task = TaskSpec(
        task_id="custom_double",
        title="Double a Number",
        objective="Return twice the numeric input.",
        initial_program="def solve(value):\n    return value\n",
        function_name="solve",
        cases=(
            EvaluationCase("double_positive", (3,), 6),
            EvaluationCase("double_zero", (0,), 0),
        ),
    )
    workbench = AlphaEvolveWorkbench(proposer=CustomProposer())

    run = workbench.start_task_run(task=task, generations=1)

    assert run.best_candidate is not None
    assert run.best_candidate.score == 1.0
    assert run.task.task_id == "custom_double"


def test_product_proposer_extracts_set_program_with_imports() -> None:
    raw_output = (
        "set:from collections import deque\n\n"
        "def solve(graph, start):\n"
        "    queue = deque([start])\n"
        "    return list(queue)\n"
    )

    extracted = _extract_program(raw_output)

    assert extracted.startswith("set:from collections import deque\n\n")
    assert "queue = deque([start])" in extracted
