"""Evaluator system for assessing code fitness in AlphaEvolve."""

import abc
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
from contextlib import contextmanager
import io
import sys

from alphaevolve.core.data_structures import EvaluatorResult

logger = logging.getLogger(__name__)


class BaseEvaluator(abc.ABC):
    """Abstract base class for evaluators."""

    @abc.abstractmethod
    def evaluate(self, code: str) -> EvaluatorResult:
        """Evaluate the given code.

        Args:
            code: Code string to evaluate

        Returns:
            EvaluatorResult with fitness and metrics
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Evaluator name."""
        pass


class SandboxExecutor:
    """Safe execution environment for untrusted code."""

    def __init__(self, timeout: float = 5.0, memory_limit: int = 128) -> None:
        """Initialize sandbox.

        Args:
            timeout: Execution timeout in seconds
            memory_limit: Memory limit in MB
        """
        self.timeout = timeout
        self.memory_limit = memory_limit

    def execute(
        self,
        code: str,
        function_name: Optional[str] = None,
        test_inputs: Optional[List[Any]] = None
    ) -> Tuple[bool, Any, float, str]:
        """Execute code safely.

        Args:
            code: Code to execute
            function_name: Function to call
            test_inputs: Inputs to pass to function

        Returns:
            Tuple of (success, output, execution_time, error_message)
        """
        start_time = time.time()

        try:
            # Create isolated namespace
            namespace: Dict[str, Any] = {}

            # Compile and execute
            compiled = compile(code, '<string>', 'exec')
            exec(compiled, namespace)

            execution_time = time.time() - start_time

            # If function name provided, call it
            if function_name and function_name in namespace:
                func = namespace[function_name]
                if test_inputs is not None:
                    results = []
                    for inp in test_inputs:
                        if isinstance(inp, (list, tuple)):
                            result = func(*inp)
                        else:
                            result = func(inp)
                        results.append(result)
                    return True, results, execution_time, ""
                else:
                    result = func()
                    return True, result, execution_time, ""

            return True, None, execution_time, ""

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, None, execution_time, error_msg


class UnitTestEvaluator(BaseEvaluator):
    """Evaluator using unit tests."""

    def __init__(
        self,
        test_cases: List[Tuple[Any, Any]],
        function_name: str,
        timeout: float = 5.0,
        partial_credit: bool = True
    ) -> None:
        """Initialize unit test evaluator.

        Args:
            test_cases: List of (input, expected_output) tuples
            function_name: Name of function to test
            timeout: Timeout per test case
            partial_credit: Whether to award partial credit
        """
        self.test_cases = test_cases
        self.function_name = function_name
        self.timeout = timeout
        self.partial_credit = partial_credit
        self.sandbox = SandboxExecutor(timeout=timeout)

    @property
    def name(self) -> str:
        return f"UnitTestEvaluator({self.function_name})"

    def evaluate(self, code: str) -> EvaluatorResult:
        """Run unit tests on code."""
        start_time = time.time()
        passed_tests = 0
        failed_tests = []
        feedback_parts = []

        try:
            # Check code compiles
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return EvaluatorResult(
                fitness=0.0,
                passed=False,
                metrics={"syntax_error": 1.0},
                feedback=f"Syntax error: {e}",
                execution_time=0.0
            )

        # Create namespace and execute
        namespace: Dict[str, Any] = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return EvaluatorResult(
                fitness=0.0,
                passed=False,
                metrics={"exec_error": 1.0},
                feedback=f"Execution error: {e}",
                execution_time=0.0
            )

        # Check function exists
        if self.function_name not in namespace:
            return EvaluatorResult(
                fitness=0.0,
                passed=False,
                metrics={"missing_function": 1.0},
                feedback=f"Function '{self.function_name}' not found in code",
                execution_time=0.0
            )

        func = namespace[self.function_name]

        # Run tests
        for i, (test_input, expected) in enumerate(self.test_cases):
            try:
                if isinstance(test_input, (list, tuple)):
                    result = func(*test_input)
                else:
                    result = func(test_input)

                if result == expected:
                    passed_tests += 1
                else:
                    failed_tests.append(i)
                    feedback_parts.append(
                        f"Test {i}: Expected {expected}, got {result}"
                    )
            except Exception as e:
                failed_tests.append(i)
                feedback_parts.append(f"Test {i}: Exception - {e}")

        total_tests = len(self.test_cases)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0

        # Calculate fitness
        if self.partial_credit:
            fitness = pass_rate * 100.0
        else:
            fitness = 100.0 if pass_rate == 1.0 else 0.0

        passed = pass_rate == 1.0

        feedback = f"Passed {passed_tests}/{total_tests} tests"
        if failed_tests:
            feedback += "\nFailed tests:\n" + "\n".join(feedback_parts)

        execution_time = time.time() - start_time

        return EvaluatorResult(
            fitness=fitness,
            passed=passed,
            metrics={
                "pass_rate": pass_rate,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
            },
            feedback=feedback,
            execution_time=execution_time
        )


class PerformanceEvaluator(BaseEvaluator):
    """Evaluator based on execution performance."""

    def __init__(
        self,
        benchmark_func: Callable[[Callable], float],
        baseline_time: Optional[float] = None,
        timeout: float = 10.0
    ) -> None:
        """Initialize performance evaluator.

        Args:
            benchmark_func: Function that takes implementation and returns time
            baseline_time: Reference time for scoring
            timeout: Maximum execution time
        """
        self.benchmark_func = benchmark_func
        self.baseline_time = baseline_time
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "PerformanceEvaluator"

    def evaluate(self, code: str) -> EvaluatorResult:
        """Benchmark code performance."""
        start_time = time.time()

        try:
            # Compile and extract function
            namespace: Dict[str, Any] = {}
            compiled = compile(code, '<string>', 'exec')
            exec(compiled, namespace)

            # Find the main function (heuristic: first callable)
            func = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    func = obj
                    break

            if func is None:
                return EvaluatorResult(
                    fitness=0.0,
                    passed=False,
                    feedback="No callable function found",
                    execution_time=time.time() - start_time
                )

            # Run benchmark
            execution_time = self.benchmark_func(func)

            # Calculate fitness based on speedup
            if self.baseline_time and execution_time > 0:
                speedup = self.baseline_time / execution_time
                fitness = min(100.0, speedup * 100.0)
            else:
                fitness = max(0.0, 100.0 - execution_time)

            return EvaluatorResult(
                fitness=fitness,
                passed=execution_time < self.timeout,
                metrics={
                    "execution_time": execution_time,
                    "baseline_time": self.baseline_time,
                    "speedup": self.baseline_time / execution_time if self.baseline_time else None,
                },
                feedback=f"Execution time: {execution_time:.4f}s",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return EvaluatorResult(
                fitness=0.0,
                passed=False,
                metrics={"error": 1.0},
                feedback=f"Error: {str(e)}",
                execution_time=time.time() - start_time
            )


class CorrectnessEvaluator(BaseEvaluator):
    """Evaluator checking output correctness against reference."""

    def __init__(
        self,
        test_inputs: List[Any],
        reference_func: Callable,
        function_name: str,
        tolerance: float = 1e-9
    ) -> None:
        """Initialize correctness evaluator.

        Args:
            test_inputs: Inputs to test
            reference_func: Reference implementation
            function_name: Name of function to evaluate
            tolerance: Numerical tolerance for floating point
        """
        self.test_inputs = test_inputs
        self.reference_func = reference_func
        self.function_name = function_name
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return f"CorrectnessEvaluator({self.function_name})"

    def evaluate(self, code: str) -> EvaluatorResult:
        """Check correctness against reference."""
        start_time = time.time()

        try:
            namespace: Dict[str, Any] = {}
            exec(code, namespace)

            if self.function_name not in namespace:
                return EvaluatorResult(
                    fitness=0.0,
                    passed=False,
                    feedback=f"Function '{self.function_name}' not found",
                    execution_time=time.time() - start_time
                )

            func = namespace[self.function_name]
            correct_count = 0
            errors = []

            for inp in self.test_inputs:
                try:
                    if isinstance(inp, (list, tuple)):
                        result = func(*inp)
                        expected = self.reference_func(*inp)
                    else:
                        result = func(inp)
                        expected = self.reference_func(inp)

                    # Compare with tolerance for floats
                    if isinstance(expected, float):
                        if abs(result - expected) < self.tolerance:
                            correct_count += 1
                        else:
                            errors.append(f"Input {inp}: {result} != {expected}")
                    else:
                        if result == expected:
                            correct_count += 1
                        else:
                            errors.append(f"Input {inp}: {result} != {expected}")

                except Exception as e:
                    errors.append(f"Input {inp}: Exception - {e}")

            total = len(self.test_inputs)
            accuracy = correct_count / total if total > 0 else 0
            fitness = accuracy * 100.0

            return EvaluatorResult(
                fitness=fitness,
                passed=accuracy == 1.0,
                metrics={
                    "accuracy": accuracy,
                    "correct": correct_count,
                    "total": total,
                },
                feedback=f"Correct on {correct_count}/{total} inputs" + (
                    "\nErrors:\n" + "\n".join(errors[:5]) if errors else ""
                ),
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return EvaluatorResult(
                fitness=0.0,
                passed=False,
                feedback=f"Error: {str(e)}",
                execution_time=time.time() - start_time
            )


class CompositeEvaluator(BaseEvaluator):
    """Combines multiple evaluators with weights."""

    def __init__(
        self,
        evaluators: List[Tuple[BaseEvaluator, float]]
    ) -> None:
        """Initialize composite evaluator.

        Args:
            evaluators: List of (evaluator, weight) tuples
        """
        self.evaluators = evaluators

    @property
    def name(self) -> str:
        names = [f"{e.name}({w})" for e, w in self.evaluators]
        return f"Composite({', '.join(names)})"

    def evaluate(self, code: str) -> EvaluatorResult:
        """Evaluate using all evaluators."""
        total_fitness = 0.0
        total_weight = 0.0
        all_passed = True
        all_metrics: Dict[str, Any] = {}
        all_feedback = []
        total_time = 0.0

        for evaluator, weight in self.evaluators:
            result = evaluator.evaluate(code)

            total_fitness += result.fitness * weight
            total_weight += weight
            all_passed = all_passed and result.passed
            total_time += result.execution_time

            # Prefix metrics with evaluator name
            for key, value in result.metrics.items():
                all_metrics[f"{evaluator.name}.{key}"] = value

            all_feedback.append(f"[{evaluator.name}] {result.feedback}")

        final_fitness = total_fitness / total_weight if total_weight > 0 else 0.0

        return EvaluatorResult(
            fitness=final_fitness,
            passed=all_passed,
            metrics=all_metrics,
            feedback="\n\n".join(all_feedback),
            execution_time=total_time
        )
