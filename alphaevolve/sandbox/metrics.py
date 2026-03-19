"""Metrics tracking for algorithm performance measurement.

This module provides utilities for tracking latency, memory usage,
and throughput metrics during algorithm execution.
"""

import asyncio
import dataclasses
import logging
import time
import typing as T
from collections import deque
from contextlib import contextmanager
from dataclasses import field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LatencyMetric:
    """Latency measurement for a single execution."""
    operation: str
    duration_ms: float
    timestamp: float


@dataclasses.dataclass(frozen=True)
class MemoryMetric:
    """Memory usage measurement."""
    operation: str
    memory_mb: float
    peak_memory_mb: float
    timestamp: float


@dataclasses.dataclass(frozen=True)
class ThroughputMetric:
    """Throughput measurement for batch operations."""
    operation: str
    items_processed: int
    duration_ms: float
    items_per_second: float


@dataclasses.dataclass
class MetricsSummary:
    """Summary statistics for a set of metrics."""
    operation: str
    count: int
    total_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class MetricsCollector:
    """Collect and aggregate performance metrics.

    The collector maintains rolling windows of metrics and computes
    summary statistics.

    Example:
        collector = MetricsCollector(window_size=1000)
        collector.record_latency("graph_solver", 45.2)
        collector.record_memory("graph_solver", peak=128.5)
        summary = collector.get_summary("graph_solver")
    """

    def __init__(self, window_size: int = 10000) -> None:
        """Initialize the metrics collector.

        Args:
            window_size: Maximum number of samples to retain in rolling window.
        """
        self._window_size = window_size
        self._latencies: T.Dict[str, deque[float]] = {}
        self._memory_snapshots: T.Dict[str, deque[float]] = {}
        self._peak_memories: T.Dict[str, float] = {}
        self._throughput_data: T.Dict[str, deque[T.Tuple[int, float]]] = {}
        self._lock = Lock()
        self._start_times: T.Dict[str, float] = {}

    def record_latency(self, operation: str, duration_ms: float) -> None:
        """Record a latency measurement.

        Args:
            operation: Name of the operation being measured.
            duration_ms: Duration in milliseconds.
        """
        with self._lock:
            if operation not in self._latencies:
                self._latencies[operation] = deque(maxlen=self._window_size)
            self._latencies[operation].append(duration_ms)

        logger.debug(f"Latency {operation}: {duration_ms:.2f}ms")

    @contextmanager
    def measure_latency(self, operation: str) -> T.Iterator[None]:
        """Context manager to measure operation latency.

        Args:
            operation: Name of the operation being measured.

        Example:
            with collector.measure_latency("algorithm_run"):
                result = algorithm.execute()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.record_latency(operation, duration_ms)

    def record_memory(
        self, operation: str, memory_mb: float, peak_memory_mb: float
    ) -> None:
        """Record memory usage for an operation.

        Args:
            operation: Name of the operation being measured.
            memory_mb: Current memory usage in MB.
            peak_memory_mb: Peak memory usage in MB.
        """
        with self._lock:
            if operation not in self._memory_snapshots:
                self._memory_snapshots[operation] = deque(maxlen=self._window_size)
            self._memory_snapshots[operation].append(memory_mb)

            current_peak = self._peak_memories.get(operation, 0.0)
            if peak_memory_mb > current_peak:
                self._peak_memories[operation] = peak_memory_mb

    def record_throughput(
        self, operation: str, items_processed: int, duration_ms: float
    ) -> None:
        """Record throughput measurement.

        Args:
            operation: Name of the operation being measured.
            items_processed: Number of items processed.
            duration_ms: Duration in milliseconds.
        """
        items_per_second = (items_processed / duration_ms) * 1000 if duration_ms > 0 else 0

        with self._lock:
            if operation not in self._throughput_data:
                self._throughput_data[operation] = deque(maxlen=self._window_size)
            self._throughput_data[operation].append((items_processed, duration_ms))

        logger.debug(
            f"Throughput {operation}: {items_per_second:.2f} items/s "
            f"({items_processed} items in {duration_ms:.2f}ms)"
        )

    def start_operation(self, operation: str) -> None:
        """Mark the start of an operation for timing.

        Args:
            operation: Name of the operation.
        """
        with self._lock:
            self._start_times[operation] = time.perf_counter()

    def end_operation(self, operation: str) -> float:
        """Mark the end of an operation and return duration.

        Args:
            operation: Name of the operation.

        Returns:
            Duration in milliseconds.
        """
        with self._lock:
            start_time = self._start_times.pop(operation, None)
            if start_time is None:
                logger.warning(f"Operation {operation} was not started")
                return 0.0

        duration_ms = (time.perf_counter() - start_time) * 1000
        self.record_latency(operation, duration_ms)
        return duration_ms

    def get_latencies(self, operation: str) -> T.List[float]:
        """Get all latency measurements for an operation.

        Args:
            operation: Name of the operation.

        Returns:
            List of latency measurements in milliseconds.
        """
        with self._lock:
            if operation not in self._latencies:
                return []
            return list(self._latencies[operation])

    def get_summary(self, operation: str) -> T.Optional[MetricsSummary]:
        """Get summary statistics for an operation.

        Args:
            operation: Name of the operation.

        Returns:
            MetricsSummary with statistics, or None if no data.
        """
        with self._lock:
            if operation not in self._latencies:
                return None

            latencies = list(self._latencies[operation])
            if not latencies:
                return None

        return self._compute_summary(operation, latencies)

    def _compute_summary(
        self, operation: str, latencies: T.List[float]
    ) -> MetricsSummary:
        """Compute summary statistics from latency values."""
        import statistics

        count = len(latencies)
        total_ms = sum(latencies)
        mean_ms = statistics.mean(latencies)
        min_ms = min(latencies)
        max_ms = max(latencies)
        std_ms = statistics.stdev(latencies) if count > 1 else 0.0

        sorted_latencies = sorted(latencies)
        p50_idx = int(count * 0.50)
        p95_idx = int(count * 0.95)
        p99_idx = int(count * 0.99)

        return MetricsSummary(
            operation=operation,
            count=count,
            total_ms=total_ms,
            mean_ms=mean_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            std_ms=std_ms,
            p50_ms=sorted_latencies[p50_idx] if sorted_latencies else 0.0,
            p95_ms=sorted_latencies[p95_idx] if sorted_latencies else 0.0,
            p99_ms=sorted_latencies[p99_idx] if sorted_latencies else 0.0,
        )

    def get_peak_memory(self, operation: str) -> float:
        """Get peak memory usage for an operation.

        Args:
            operation: Name of the operation.

        Returns:
            Peak memory in MB.
        """
        with self._lock:
            return self._peak_memories.get(operation, 0.0)

    def get_throughput_summary(self, operation: str) -> T.Optional[ThroughputMetric]:
        """Get aggregate throughput statistics for an operation.

        Args:
            operation: Name of the operation.

        Returns:
            Aggregate ThroughputMetric or None if no data.
        """
        with self._lock:
            if operation not in self._throughput_data:
                return None

            data = list(self._throughput_data[operation])
            if not data:
                return None

        total_items = sum(items for items, _ in data)
        total_duration = sum(duration for _, duration in data)
        avg_throughput = (total_items / total_duration * 1000) if total_duration > 0 else 0.0

        return ThroughputMetric(
            operation=operation,
            items_processed=total_items,
            duration_ms=total_duration,
            items_per_second=avg_throughput,
        )

    def reset(self, operation: T.Optional[str] = None) -> None:
        """Reset metrics for an operation or all operations.

        Args:
            operation: Optional operation name. If None, resets all.
        """
        with self._lock:
            if operation is None:
                self._latencies.clear()
                self._memory_snapshots.clear()
                self._peak_memories.clear()
                self._throughput_data.clear()
                self._start_times.clear()
            else:
                self._latencies.pop(operation, None)
                self._memory_snapshots.pop(operation, None)
                self._peak_memories.pop(operation, None)
                self._throughput_data.pop(operation, None)
                self._start_times.pop(operation, None)


class PerformanceTracker:
    """Track performance metrics for Steiner tree algorithm evaluation.

    This specialized tracker captures algorithm-specific metrics
    for graph-based algorithms.
    """

    def __init__(self, collector: T.Optional[MetricsCollector] = None) -> None:
        self._collector = collector or MetricsCollector()

    @property
    def collector(self) -> MetricsCollector:
        """Get the underlying metrics collector."""
        return self._collector

    def track_graph_operation(
        self, operation: str, num_nodes: int, num_edges: int
    ) -> T.ContextManager[None]:
        """Track a graph operation with size information.

        Args:
            operation: Name of the operation.
            num_nodes: Number of nodes in the graph.
            num_edges: Number of edges in the graph.

        Returns:
            Context manager for timing.
        """
        label = f"{operation}_n{num_nodes}_e{num_edges}"
        return self._collector.measure_latency(label)

    def record_graph_solution(
        self,
        operation: str,
        num_nodes: int,
        num_edges: int,
        solution_cost: float,
        execution_time_ms: float,
        peak_memory_mb: float,
    ) -> None:
        """Record a complete graph solution measurement.

        Args:
            operation: Name of the algorithm.
            num_nodes: Number of nodes in the graph.
            num_edges: Number of edges in the graph.
            solution_cost: Cost of the computed solution.
            execution_time_ms: Execution time in milliseconds.
            peak_memory_mb: Peak memory usage in MB.
        """
        self._collector.record_latency(f"{operation}_solution", execution_time_ms)
        self._collector.record_memory(
            f"{operation}_solution", peak_memory_mb, peak_memory_mb
        )

    def benchmark_algorithm(
        self,
        code: str,
        function: str,
        test_cases: T.List[T.Dict[str, T.Any]],
        timeout_seconds: float = 30.0,
    ) -> T.Dict[str, T.Any]:
        """Benchmark an algorithm across multiple test cases.

        Args:
            code: Python code containing the algorithm.
            function: Name of the algorithm function.
            test_cases: List of test case dictionaries.
            timeout_seconds: Maximum time per test case.

        Returns:
            Dictionary with benchmark results and statistics.
        """
        from .executor import SandboxExecutor, ExecutionConfig, ExecutionResult

        executor = SandboxExecutor()
        results: T.List[ExecutionResult] = []
        execution_times: T.List[float] = []
        peak_memories: T.List[float] = []

        for i, test_case in enumerate(test_cases):
            args = test_case.get("args", ())
            kwargs = test_case.get("kwargs", {})

            config = ExecutionConfig(timeout_seconds=timeout_seconds)
            result = executor.execute(code, function, args, kwargs, config)

            results.append(result)

            if result.success:
                execution_times.append(result.execution_time * 1000)
                peak_memories.append(result.peak_memory_mb)

                self._collector.record_latency(f"benchmark_{i}", result.execution_time * 1000)
                self._collector.record_memory(
                    f"benchmark_{i}", result.peak_memory_mb, result.peak_memory_mb
                )

        success_count = sum(1 for r in results if r.success)
        failure_count = len(results) - success_count

        summary: T.Dict[str, T.Any] = {
            "total_cases": len(test_cases),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / len(test_cases) if test_cases else 0.0,
            "results": [
                {
                    "case_id": i,
                    "success": r.success,
                    "execution_time_ms": r.execution_time * 1000,
                    "peak_memory_mb": r.peak_memory_mb,
                    "error": r.error,
                }
                for i, r in enumerate(results)
            ],
        }

        if execution_times:
            import statistics
            summary["timing"] = {
                "mean_ms": statistics.mean(execution_times),
                "median_ms": statistics.median(execution_times),
                "min_ms": min(execution_times),
                "max_ms": max(execution_times),
                "std_ms": statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0,
            }

        if peak_memories:
            import statistics
            summary["memory"] = {
                "mean_mb": statistics.mean(peak_memories),
                "peak_mb": max(peak_memories),
            }

        return summary
