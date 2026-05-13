"""Tests for the deterministic benchmark suite."""

from alphaevolve.benchmarks.problem_suite import (
    BenchmarkCase,
    build_20_problem_suite,
    filter_cases,
    run_problem_suite,
    summarize_results,
)
from alphaevolve.sandbox.executor import ExecutionConfig


def test_build_20_problem_suite_has_expected_shape():
    cases = build_20_problem_suite()
    assert len(cases) == 20
    assert sum(1 for case in cases if case.family == "mst") == 10
    assert sum(1 for case in cases if case.family == "steiner") == 10


def test_filter_cases_can_select_specific_cases():
    cases = build_20_problem_suite()
    selected = filter_cases(cases, case_ids=["mst_01", "steiner_01"])
    assert [case.case_id for case in selected] == ["mst_01", "steiner_01"]


def test_run_problem_suite_can_pass_for_valid_case():
    case = BenchmarkCase(
        case_id="custom_mst",
        family="mst",
        description="custom passing case",
        code="def mst(edges):\n    return 3.0\n",
        function_name="mst",
        args=([("ignored", "ignored", 0.0)],),
        validator="mst_exact",
        metadata={"expected": 3.0},
    )
    results = run_problem_suite(
        [case],
        config=ExecutionConfig(timeout_seconds=2.0, memory_limit_mb=128.0),
    )

    assert len(results) == 1
    assert results[0].passed is True


def test_run_problem_suite_passes_reference_cases():
    cases = filter_cases(build_20_problem_suite(), case_ids=["mst_01", "steiner_01"])
    results = run_problem_suite(
        cases,
        config=ExecutionConfig(timeout_seconds=2.0, memory_limit_mb=128.0),
    )

    assert len(results) == 2
    assert results[0].case_id == "mst_01"
    assert results[0].success is True
    assert results[0].passed is True
    assert results[1].case_id == "steiner_01"
    assert results[1].success is True
    assert results[1].passed is True

    summary = summarize_results(results)
    assert summary["passed_cases"] == 2


def test_run_problem_suite_passes_all_20_cases():
    results = run_problem_suite(
        build_20_problem_suite(),
        config=ExecutionConfig(timeout_seconds=2.0, memory_limit_mb=128.0),
    )

    summary = summarize_results(results)
    assert summary["total_cases"] == 20
    assert summary["successful_executions"] == 20
    assert summary["passed_cases"] == 20
