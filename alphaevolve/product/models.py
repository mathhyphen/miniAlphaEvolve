"""Canonical product models for the AlphaEvolve workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    """A machine-checkable evaluator case."""

    case_id: str
    args: tuple[Any, ...]
    expected: Any
    description: str = ""
    validator: str = "exact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "args": list(self.args),
            "expected": self.expected,
            "description": self.description,
            "validator": self.validator,
        }


@dataclass(frozen=True)
class CaseFailure:
    """A concise diagnostic for a failed evaluator case."""

    case_id: str
    message: str
    expected: str
    actual: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Structured evaluator output for one candidate."""

    score: float
    passed_cases: int
    total_cases: int
    failures: tuple[CaseFailure, ...] = ()
    execution_time: float = 0.0
    peak_memory_mb: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed_cases": self.passed_cases,
            "total_cases": self.total_cases,
            "failures": [failure.to_dict() for failure in self.failures],
            "execution_time": self.execution_time,
            "peak_memory_mb": self.peak_memory_mb,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class TaskSpec:
    """The product-level contract for an AlphaEvolve task."""

    task_id: str
    title: str
    objective: str
    initial_program: str
    cases: tuple[EvaluationCase, ...]
    function_name: str = "solve"
    language: str = "python"
    metric: str = "pass_rate"
    constraints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    baseline_program: str | None = None
    benchmark_repetitions: int = 1

    def create_evaluator(self):
        from .evaluation import PythonFunctionEvaluator

        return PythonFunctionEvaluator(self)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "objective": self.objective,
            "initial_program": self.initial_program,
            "function_name": self.function_name,
            "language": self.language,
            "metric": self.metric,
            "constraints": list(self.constraints),
            "tags": list(self.tags),
            "cases": [case.to_dict() for case in self.cases],
            "baseline_program": self.baseline_program,
            "benchmark_repetitions": self.benchmark_repetitions,
        }


@dataclass(frozen=True)
class Proposal:
    """A proposal emitted by a model or local fallback proposer."""

    text: str
    source: str
    raw_output: str = ""


@dataclass(frozen=True)
class CandidateRecord:
    """A candidate program plus lineage, prompt, and evidence."""

    candidate_id: str
    generation: int
    program: str
    evaluation: EvaluationReport
    parent_id: str | None = None
    prompt: str = ""
    proposal: str = ""
    proposal_source: str = ""
    kept: bool = True
    created_at: str = ""

    @property
    def score(self) -> float:
        return self.evaluation.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "program": self.program,
            "score": self.score,
            "evaluation": self.evaluation.to_dict(),
            "prompt": self.prompt,
            "proposal": self.proposal,
            "proposal_source": self.proposal_source,
            "kept": self.kept,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class RunSnapshot:
    """Immutable snapshot of one workbench run."""

    run_id: str
    task: TaskSpec
    status: str
    current_generation: int
    max_generations: int
    archive: tuple[CandidateRecord, ...]
    history: tuple[CandidateRecord, ...]
    created_at: str
    updated_at: str

    @property
    def best_candidate(self) -> CandidateRecord | None:
        if not self.archive:
            return None
        return self.archive[0]

    def to_dict(self) -> dict[str, Any]:
        best = self.best_candidate
        return {
            "run_id": self.run_id,
            "task": self.task.to_public_dict(),
            "status": self.status,
            "current_generation": self.current_generation,
            "max_generations": self.max_generations,
            "best_candidate": best.to_dict() if best else None,
            "archive": [candidate.to_dict() for candidate in self.archive],
            "history": [candidate.to_dict() for candidate in self.history],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
