"""Product-grade AlphaEvolve workbench API."""

from .evaluation import PythonFunctionEvaluator
from .models import (
    CandidateRecord,
    CaseFailure,
    EvaluationCase,
    EvaluationReport,
    Proposal,
    RunSnapshot,
    TaskSpec,
)
from .tasks import get_builtin_task, list_builtin_tasks
from .workbench import AlphaEvolveWorkbench, apply_proposal

__all__ = [
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
]
