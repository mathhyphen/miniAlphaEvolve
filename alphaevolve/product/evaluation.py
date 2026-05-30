"""Evaluator adapters for product-grade AlphaEvolve tasks."""

from __future__ import annotations

import math
from typing import Any

from alphaevolve.sandbox.executor import ExecutionConfig, SandboxExecutor

from .models import CaseFailure, EvaluationReport, TaskSpec


class PythonFunctionEvaluator:
    """Evaluate Python candidate programs against structured test cases."""

    def __init__(
        self,
        task: TaskSpec,
        *,
        executor: SandboxExecutor | None = None,
        config: ExecutionConfig | None = None,
    ) -> None:
        self._task = task
        self._executor = executor or SandboxExecutor()
        self._config = config or ExecutionConfig(
            timeout_seconds=1.5,
            memory_limit_mb=128.0,
            max_output_size=100_000,
        )
        self._baseline_execution_time: float | None = None

    def evaluate(self, program: str) -> EvaluationReport:
        passed_cases, failures, execution_time, peak_memory_mb = self._run_cases(program)
        total_cases = len(self._task.cases)
        pass_rate = passed_cases / total_cases if total_cases else 0.0
        score = pass_rate
        speedup = None

        if self._task.metric == "correctness_speed" and pass_rate == 1.0:
            baseline_time = self._get_baseline_execution_time()
            if baseline_time is not None and execution_time > 0:
                speedup = baseline_time / execution_time
                score = 1.0 + min(max(speedup, 0.0), 5.0) / 10.0

        metrics = {
            "pass_rate": pass_rate,
            "failed_cases": len(failures),
            "metric": self._task.metric,
        }
        if speedup is not None:
            metrics = {
                **metrics,
                "speedup": speedup,
                "baseline_execution_time": self._baseline_execution_time,
            }

        return EvaluationReport(
            score=score,
            passed_cases=passed_cases,
            total_cases=total_cases,
            failures=tuple(failures),
            execution_time=execution_time,
            peak_memory_mb=peak_memory_mb,
            metrics=metrics,
        )

    def _run_cases(self, program: str) -> tuple[int, list[CaseFailure], float, float]:
        failures: list[CaseFailure] = []
        passed_cases = 0
        execution_time = 0.0
        peak_memory_mb = 0.0

        for case in self._task.cases:
            executable_program, function_name = self._benchmark_program(program)
            result = self._executor.execute(
                code=executable_program,
                function=function_name,
                args=case.args,
                config=self._config,
            )
            execution_time += result.execution_time
            peak_memory_mb = max(peak_memory_mb, result.peak_memory_mb)

            if not result.success:
                failures.append(
                    CaseFailure(
                        case_id=case.case_id,
                        message=result.error or "Execution failed",
                        expected=repr(case.expected),
                        actual="<execution-error>",
                    )
                )
                continue

            if self._matches(case.validator, result.output, case.expected):
                passed_cases += 1
                continue

            failures.append(
                CaseFailure(
                    case_id=case.case_id,
                    message="Output mismatch",
                    expected=repr(case.expected),
                    actual=repr(result.output),
                )
            )

        return passed_cases, failures, execution_time, peak_memory_mb

    def _get_baseline_execution_time(self) -> float | None:
        if self._task.baseline_program is None:
            return None
        if self._baseline_execution_time is None:
            passed_cases, failures, execution_time, _ = self._run_cases(self._task.baseline_program)
            if failures or passed_cases != len(self._task.cases):
                return None
            self._baseline_execution_time = execution_time
        return self._baseline_execution_time

    def _benchmark_program(self, program: str) -> tuple[str, str]:
        repetitions = self._task.benchmark_repetitions
        if self._task.metric != "correctness_speed" or repetitions <= 1:
            return program, self._task.function_name

        benchmark_function = "__alphaevolve_benchmark__"
        wrapped_program = (
            f"{program}\n\n"
            f"_alphaevolve_candidate = {self._task.function_name}\n"
            f"def {benchmark_function}(*args):\n"
            f"    result = None\n"
            f"    for _ in range({repetitions}):\n"
            f"        result = _alphaevolve_candidate(*args)\n"
            f"    return result\n"
        )
        return wrapped_program, benchmark_function

    def _matches(self, validator: str, actual: Any, expected: Any) -> bool:
        if validator == "approx":
            try:
                return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9)
            except (TypeError, ValueError):
                return False
        if validator == "contains":
            try:
                return expected in actual
            except TypeError:
                return False
        return actual == expected
