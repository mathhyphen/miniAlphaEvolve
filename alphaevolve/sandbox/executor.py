"""Sandbox executor for safe code execution with resource limits.

This module provides a secure execution environment for algorithm code
with support for timeout, memory limits, and isolation.
"""

import asyncio
import concurrent.futures
import dataclasses
import logging
import os
import sys
import threading
import time
import typing as T
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    """Result of a sandboxed execution."""
    success: bool
    output: T.Any
    error: T.Optional[str]
    execution_time: float
    peak_memory_mb: float
    exit_code: T.Optional[int]


@dataclasses.dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for sandboxed execution."""
    timeout_seconds: float = 30.0
    memory_limit_mb: float = 512.0
    max_output_size: int = 10_000_000  # 10MB
    max_iterations: int = 10_000_000
    enable_network: bool = False
    working_directory: T.Optional[Path] = None


class ExecutionError(Exception):
    """Raised when execution fails."""
    pass


class TimeoutError(ExecutionError):
    """Raised when execution exceeds timeout."""
    pass


class MemoryLimitError(ExecutionError):
    """Raised when execution exceeds memory limit."""
    pass


class ResourceTracker:
    """Track resource usage during execution."""

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._peak_memory: int = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start tracking resources."""
        self._start_time = _get_time()
        self._peak_memory = 0

    def update_peak(self, current_memory: int) -> None:
        """Update peak memory usage."""
        with self._lock:
            if current_memory > self._peak_memory:
                self._peak_memory = current_memory

    @property
    def execution_time(self) -> float:
        """Get elapsed execution time."""
        return _get_time() - self._start_time

    @property
    def peak_memory_mb(self) -> float:
        """Get peak memory in MB."""
        return self._peak_memory / (1024 * 1024)


def _get_time() -> float:
    """Get current time in seconds."""
    return time.perf_counter()


def _set_memory_limit(limit_mb: float) -> None:
    """Set memory limit for the current process (Unix only)."""
    if sys.platform != "win32" and resource is not None:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit_bytes = int(limit_mb * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))


def _set_timeout(timeout_seconds: float) -> None:
    """Set CPU timeout for the current process (Unix only)."""
    if sys.platform != "win32" and resource is not None:
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout_seconds), hard))


class SandboxExecutor:
    """Execute code in a sandboxed environment with resource limits.

    The executor provides isolation and resource tracking for untrusted
    or potentially expensive algorithm code.

    Example:
        executor = SandboxExecutor()
        result = executor.execute(
            code="def solve(n): return n * 2",
            function="solve",
            args=(42,),
            config=ExecutionConfig(timeout_seconds=5.0)
        )
        print(result.output)  # 84
    """

    def __init__(self) -> None:
        self._tracker = ResourceTracker()
        self._current_result: T.Optional[ExecutionResult] = None

    def execute(
        self,
        code: str,
        function: str,
        args: T.Tuple[T.Any, ...] = (),
        kwargs: T.Optional[T.Dict[str, T.Any]] = None,
        config: T.Optional[ExecutionConfig] = None,
    ) -> ExecutionResult:
        """Execute a function in a sandboxed environment.

        Args:
            code: Python code containing the function definition.
            function: Name of the function to execute.
            args: Positional arguments to pass to the function.
            kwargs: Keyword arguments to pass to the function.
            config: Execution configuration with resource limits.

        Returns:
            ExecutionResult with output, metrics, and any error.
        """
        if config is None:
            config = ExecutionConfig()
        if kwargs is None:
            kwargs = {}

        self._tracker.start()
        self._current_result = None

        logger.debug(
            f"Executing {function} with timeout={config.timeout_seconds}s, "
            f"memory_limit={config.memory_limit_mb}MB"
        )

        try:
            result = self._execute_in_subprocess(
                code, function, args, kwargs, config
            )
            self._current_result = result
            return result
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=self._tracker.execution_time,
                peak_memory_mb=self._tracker.peak_memory_mb,
                exit_code=-1,
            )

    def _execute_in_subprocess(
        self,
        code: str,
        function: str,
        args: T.Tuple[T.Any, ...],
        kwargs: T.Dict[str, T.Any],
        config: ExecutionConfig,
    ) -> ExecutionResult:
        """Execute code in a separate process for isolation."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _run_in_subprocess,
                code,
                function,
                args,
                kwargs,
                config,
            )
            try:
                result = future.result(timeout=config.timeout_seconds + 1.0)
                return result
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(
                    f"Execution exceeded timeout of {config.timeout_seconds}s"
                )
            except Exception as e:
                raise ExecutionError(f"Execution error: {e}") from e


def _run_in_subprocess(
    code: str,
    function: str,
    args: T.Tuple[T.Any, ...],
    kwargs: T.Dict[str, T.Any],
    config: ExecutionConfig,
) -> ExecutionResult:
    """Run execution in a subprocess with resource limits."""
    start_time = _get_time()
    peak_memory = 0
    error: T.Optional[str] = None
    exit_code: T.Optional[int] = None

    # Set resource limits
    if sys.platform != "win32":
        _set_memory_limit(config.memory_limit_mb)
        _set_timeout(config.timeout_seconds)

    # Memory tracking thread
    def track_memory() -> None:
        nonlocal peak_memory
        while True:
            try:
                if psutil is None:
                    return
                process = psutil.Process(os.getpid())
                memory = process.memory_info().rss
                if memory > peak_memory:
                    peak_memory = memory
                threading.Event().wait(0.1)
            except Exception:
                break

    memory_thread = None
    if psutil is not None:
        memory_thread = threading.Thread(target=track_memory, daemon=True)
        memory_thread.start()

    try:
        # Compile and execute the code
        namespace: T.Dict[str, T.Any] = {
            "__name__": "__sandbox__",
            "__builtins__": __builtins__,
        }

        exec(code, namespace)

        if function not in namespace:
            raise ExecutionError(f"Function '{function}' not found in code")

        func = namespace[function]
        output = func(*args, **kwargs)
        success = True
        error = None
        exit_code = 0

    except TimeoutError:
        success = False
        error = f"Execution timed out after {config.timeout_seconds}s"
        exit_code = -1
        output = None
    except MemoryError:
        success = False
        error = f"Execution exceeded memory limit of {config.memory_limit_mb}MB"
        exit_code = -1
        output = None
    except Exception as e:
        success = False
        error = f"{type(e).__name__}: {e}"
        exit_code = -1
        output = None
    finally:
        if memory_thread:
            memory_thread.join(timeout=0.1)

    execution_time = _get_time() - start_time
    peak_memory_mb = peak_memory / (1024 * 1024) if peak_memory > 0 else 0.0

    return ExecutionResult(
        success=success,
        output=output,
        error=error,
        execution_time=execution_time,
        peak_memory_mb=peak_memory_mb,
        exit_code=exit_code,
    )


class AsyncSandboxExecutor:
    """Async wrapper for sandbox execution."""

    def __init__(self, executor: T.Optional[SandboxExecutor] = None) -> None:
        self._executor = executor or SandboxExecutor()

    async def execute_async(
        self,
        code: str,
        function: str,
        args: T.Tuple[T.Any, ...] = (),
        kwargs: T.Optional[T.Dict[str, T.Any]] = None,
        config: T.Optional[ExecutionConfig] = None,
    ) -> ExecutionResult:
        """Execute function asynchronously in sandbox.

        Args:
            code: Python code containing the function definition.
            function: Name of the function to execute.
            args: Positional arguments to pass to the function.
            kwargs: Keyword arguments to pass to the function.
            config: Execution configuration with resource limits.

        Returns:
            ExecutionResult with output, metrics, and any error.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._executor.execute(code, function, args, kwargs, config),
        )
