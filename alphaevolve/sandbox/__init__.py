"""Sandbox execution environment for AlphaEvolve.

This package provides safe code execution, performance metrics,
and formal verification for algorithm evaluation.
"""

from .executor import (
    ExecutionConfig,
    ExecutionResult,
    ExecutionError,
    TimeoutError,
    MemoryLimitError,
    SandboxExecutor,
    AsyncSandboxExecutor,
)

from .metrics import (
    LatencyMetric,
    MemoryMetric,
    ThroughputMetric,
    MetricsSummary,
    MetricsCollector,
    PerformanceTracker,
)

from .verifier import (
    Contract,
    VerificationResult,
    VerificationSummary,
    ContractError,
    PreconditionError,
    PostconditionError,
    InvariantError,
    ContractVerifier,
    precondition,
    postcondition,
    invariant,
    create_steiner_tree_contract,
    verify_steiner_tree_properties,
)

__all__ = [
    # Executor
    "ExecutionConfig",
    "ExecutionResult",
    "ExecutionError",
    "TimeoutError",
    "MemoryLimitError",
    "SandboxExecutor",
    "AsyncSandboxExecutor",
    # Metrics
    "LatencyMetric",
    "MemoryMetric",
    "ThroughputMetric",
    "MetricsSummary",
    "MetricsCollector",
    "PerformanceTracker",
    # Verifier
    "Contract",
    "VerificationResult",
    "VerificationSummary",
    "ContractError",
    "PreconditionError",
    "PostconditionError",
    "InvariantError",
    "ContractVerifier",
    "precondition",
    "postcondition",
    "invariant",
    "create_steiner_tree_contract",
    "verify_steiner_tree_properties",
]
