"""Multi-stage evaluation pipeline with early termination."""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from alphaevolve.core.data_structures import EvaluatorResult

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result of a single evaluation stage.

    Args:
        stage_name: Name of the stage
        passed: Whether stage passed
        score: Stage score (0-100)
        feedback: Textual feedback
        execution_time: Time to run stage
        metrics: Additional metrics
    """
    stage_name: str
    passed: bool
    score: float = 0.0
    feedback: str = ""
    execution_time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)


class EvaluationStage:
    """Base class for evaluation stages.

    Each stage performs a specific type of evaluation
    and returns a StageResult.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return self.__class__.__name__

    def evaluate(self, code: str, context: Optional[Dict] = None) -> StageResult:
        """Evaluate code at this stage.

        Args:
            code: Code to evaluate
            context: Optional context from previous stages

        Returns:
            StageResult
        """
        raise NotImplementedError


class SyntaxCheckStage(EvaluationStage):
    """Stage 1: Check if code is syntactically valid."""

    def evaluate(self, code: str, context: Optional[Dict] = None) -> StageResult:
        """Check Python syntax."""
        start_time = time.time()

        try:
            compile(code, '<string>', 'exec')
            return StageResult(
                stage_name=self.name,
                passed=True,
                score=100.0,
                feedback="Syntax valid",
                execution_time=time.time() - start_time,
            )
        except SyntaxError as e:
            return StageResult(
                stage_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Syntax error: {e}",
                execution_time=time.time() - start_time,
            )


class BasicTestStage(EvaluationStage):
    """Stage 2: Run basic unit tests."""

    def __init__(
        self,
        test_cases: List[tuple],
        function_name: str,
    ) -> None:
        """Initialize basic test stage.

        Args:
            test_cases: List of (input, expected) tuples
            function_name: Function to test
        """
        self.test_cases = test_cases
        self.function_name = function_name

    def evaluate(self, code: str, context: Optional[Dict] = None) -> StageResult:
        """Run basic tests."""
        start_time = time.time()

        try:
            namespace: Dict[str, Any] = {}
            exec(code, namespace)

            if self.function_name not in namespace:
                return StageResult(
                    stage_name=self.name,
                    passed=False,
                    score=0.0,
                    feedback=f"Function '{self.function_name}' not found",
                    execution_time=time.time() - start_time,
                )

            func = namespace[self.function_name]
            passed = 0
            failed = 0

            for test_input, expected in self.test_cases:
                try:
                    if isinstance(test_input, (list, tuple)):
                        result = func(*test_input)
                    else:
                        result = func(test_input)

                    if result == expected:
                        passed += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1

            total = passed + failed
            score = (passed / total * 100) if total > 0 else 0

            return StageResult(
                stage_name=self.name,
                passed=failed == 0,
                score=score,
                feedback=f"Passed {passed}/{total} basic tests",
                execution_time=time.time() - start_time,
                metrics={"passed": passed, "failed": failed, "total": total},
            )

        except Exception as e:
            return StageResult(
                stage_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Execution error: {e}",
                execution_time=time.time() - start_time,
            )


class EdgeCaseStage(EvaluationStage):
    """Stage 3: Test edge cases and boundary conditions."""

    def __init__(
        self,
        edge_cases: List[tuple],
        function_name: str,
    ) -> None:
        """Initialize edge case stage.

        Args:
            edge_cases: List of (input, expected) tuples
            function_name: Function to test
        """
        self.edge_cases = edge_cases
        self.function_name = function_name

    def evaluate(self, code: str, context: Optional[Dict] = None) -> StageResult:
        """Run edge case tests."""
        start_time = time.time()

        try:
            namespace: Dict[str, Any] = {}
            exec(code, namespace)

            if self.function_name not in namespace:
                return StageResult(
                    stage_name=self.name,
                    passed=False,
                    score=0.0,
                    feedback=f"Function '{self.function_name}' not found",
                    execution_time=time.time() - start_time,
                )

            func = namespace[self.function_name]
            passed = 0

            for test_input, expected in self.edge_cases:
                try:
                    if isinstance(test_input, (list, tuple)):
                        result = func(*test_input)
                    else:
                        result = func(test_input)

                    if result == expected:
                        passed += 1
                except Exception:
                    pass  # Edge cases often fail

            total = len(self.edge_cases)
            score = (passed / total * 100) if total > 0 else 0

            # Pass if at least 80% edge cases handled
            passed_stage = score >= 80

            return StageResult(
                stage_name=self.name,
                passed=passed_stage,
                score=score,
                feedback=f"Handled {passed}/{total} edge cases",
                execution_time=time.time() - start_time,
                metrics={"passed": passed, "total": total},
            )

        except Exception as e:
            return StageResult(
                stage_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Execution error: {e}",
                execution_time=time.time() - start_time,
            )


class PerformanceStage(EvaluationStage):
    """Stage 4: Benchmark performance."""

    def __init__(
        self,
        benchmark_inputs: List[Any],
        function_name: str,
        baseline_time: Optional[float] = None,
        iterations: int = 5,
    ) -> None:
        """Initialize performance stage.

        Args:
            benchmark_inputs: Inputs for benchmarking
            function_name: Function to benchmark
            baseline_time: Reference time for comparison
            iterations: Number of iterations per input
        """
        self.benchmark_inputs = benchmark_inputs
        self.function_name = function_name
        self.baseline_time = baseline_time
        self.iterations = iterations

    def evaluate(self, code: str, context: Optional[Dict] = None) -> StageResult:
        """Run performance benchmark."""
        start_time = time.time()

        try:
            namespace: Dict[str, Any] = {}
            exec(code, namespace)

            if self.function_name not in namespace:
                return StageResult(
                    stage_name=self.name,
                    passed=False,
                    score=0.0,
                    feedback=f"Function '{self.function_name}' not found",
                    execution_time=time.time() - start_time,
                )

            func = namespace[self.function_name]
            times = []

            for inp in self.benchmark_inputs:
                for _ in range(self.iterations):
                    iter_start = time.time()
                    if isinstance(inp, (list, tuple)):
                        func(*inp)
                    else:
                        func(inp)
                    times.append(time.time() - iter_start)

            avg_time = sum(times) / len(times) if times else 0

            # Calculate score based on baseline
            if self.baseline_time and self.baseline_time > 0:
                speedup = self.baseline_time / avg_time if avg_time > 0 else 1.0
                score = min(100.0, speedup * 50)  # 2x speedup = 100 score
            else:
                # No baseline: score based on absolute time
                score = max(0, 100 - avg_time * 10)

            return StageResult(
                stage_name=self.name,
                passed=True,
                score=score,
                feedback=f"Average execution time: {avg_time*1000:.2f}ms",
                execution_time=time.time() - start_time,
                metrics={"avg_time": avg_time, "iterations": len(times)},
            )

        except Exception as e:
            return StageResult(
                stage_name=self.name,
                passed=False,
                score=0.0,
                feedback=f"Benchmark error: {e}",
                execution_time=time.time() - start_time,
            )
