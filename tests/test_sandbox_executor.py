"""Smoke tests for the sandbox executor."""

from alphaevolve.sandbox.executor import ExecutionConfig, SandboxExecutor


def test_sandbox_executor_executes_simple_function():
    """The executor should import cleanly and run a simple function."""
    executor = SandboxExecutor()
    result = executor.execute(
        code="""
def solve(x):
    return x * 2
""",
        function="solve",
        args=(21,),
        config=ExecutionConfig(timeout_seconds=1.0),
    )

    assert result.success is True
    assert result.output == 42
    assert result.error is None
