"""
AlphaEvolve - Google DeepMind's AlphaEvolve-style Algorithm Discovery Workbench

This package provides an evolutionary coding agent that uses Large Language Models (LLMs)
to iteratively improve algorithms for automatically evaluable tasks.

Core Product API:
    AlphaEvolveWorkbench - The canonical task -> prompt -> propose -> evaluate -> archive loop
    TaskSpec - Task specification with evaluator cases
    PythonFunctionEvaluator - Standard Python function evaluator

Quick Start:
    from alphaevolve import AlphaEvolveWorkbench, get_builtin_task

    workbench = AlphaEvolveWorkbench()
    task = get_builtin_task("sort_numbers")
    run = workbench.start_task_run(task=task, generations=8, archive_size=8)
    best_program = workbench.export_best_program(run.run_id)

For more details, see:
    - docs/alphaevolve_product_gap_report.md - Product gap analysis vs Google DeepMind AlphaEvolve
    - docs/alphadev_architecture.md - System architecture documentation
"""

from alphaevolve.product import (
    AlphaEvolveWorkbench,
    CandidateRecord,
    CaseFailure,
    EvaluationCase,
    EvaluationReport,
    Proposal,
    PythonFunctionEvaluator,
    RunSnapshot,
    TaskSpec,
    apply_proposal,
    get_builtin_task,
    list_builtin_tasks,
)

from alphaevolve.sandbox import SandboxExecutor, ExecutionConfig, Contract
from alphaevolve.problems import Problem, FunctionEvaluator, BenchmarkEvaluator

__all__ = [
    # Product API (AlphaEvolve core loop)
    "AlphaEvolveWorkbench",
    "CandidateRecord",
    "CaseFailure",
    "EvaluationCase",
    "EvaluationReport",
    "Proposal",
    "PythonFunctionEvaluator",
    "RunSnapshot",
    "TaskSpec",
    "apply_proposal",
    "get_builtin_task",
    "list_builtin_tasks",
    # Sandbox
    "SandboxExecutor",
    "ExecutionConfig",
    "Contract",
    # Problems
    "Problem",
    "FunctionEvaluator",
    "BenchmarkEvaluator",
]
