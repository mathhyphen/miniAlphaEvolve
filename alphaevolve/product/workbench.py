"""AlphaEvolve-style product controller and in-memory run store."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Sequence
from uuid import uuid4

from .models import CandidateRecord, RunSnapshot, TaskSpec
from .proposers import ProductPromptSampler, ProductProposer, create_default_proposer
from .tasks import get_builtin_task, list_builtin_tasks


class AlphaEvolveWorkbench:
    """Run the canonical task -> prompt -> propose -> evaluate -> archive loop."""

    def __init__(
        self,
        *,
        proposer: ProductProposer | None = None,
        sampler: ProductPromptSampler | None = None,
        max_generations: int = 100,
    ) -> None:
        self._proposer = proposer or create_default_proposer()
        self._sampler = sampler or ProductPromptSampler()
        self._max_generations = max_generations
        self._runs: dict[str, RunSnapshot] = {}

    def list_tasks(self):
        return list_builtin_tasks()

    def get_task(self, task_id: str):
        return get_builtin_task(task_id)

    def start_run(
        self,
        *,
        task_id: str,
        generations: int = 8,
        archive_size: int = 8,
    ) -> RunSnapshot:
        task = get_builtin_task(task_id)
        return self.start_task_run(
            task=task,
            generations=generations,
            archive_size=archive_size,
        )

    def start_task_run(
        self,
        *,
        task: TaskSpec,
        generations: int = 8,
        archive_size: int = 8,
    ) -> RunSnapshot:
        if generations < 1 or generations > self._max_generations:
            raise ValueError(f"generations must be between 1 and {self._max_generations}")
        if archive_size < 1 or archive_size > 64:
            raise ValueError("archive_size must be between 1 and 64")
        if not task.cases:
            raise ValueError("task must include at least one evaluator case")

        evaluator = task.create_evaluator()
        now = _utc_now()
        run_id = f"run-{uuid4().hex[:12]}"

        initial_report = evaluator.evaluate(task.initial_program)
        initial_candidate = CandidateRecord(
            candidate_id=f"{run_id}-gen-0",
            generation=0,
            parent_id=None,
            program=task.initial_program,
            evaluation=initial_report,
            prompt="",
            proposal="",
            proposal_source="seed",
            kept=True,
            created_at=now,
        )

        archive: tuple[CandidateRecord, ...] = (initial_candidate,)
        history: tuple[CandidateRecord, ...] = (initial_candidate,)

        for generation in range(1, generations + 1):
            parent = archive[0]
            prompt = self._sampler.build_prompt(task, parent, archive)
            proposal = self._proposer.propose(task, parent, archive, prompt)
            child_program = apply_proposal(parent.program, proposal.text)
            evaluation = evaluator.evaluate(child_program)
            child = CandidateRecord(
                candidate_id=f"{run_id}-gen-{generation}",
                generation=generation,
                parent_id=parent.candidate_id,
                program=child_program,
                evaluation=evaluation,
                prompt=prompt,
                proposal=proposal.text,
                proposal_source=proposal.source,
                kept=True,
                created_at=_utc_now(),
            )

            new_archive = _add_to_archive(archive, child, archive_size)
            kept = any(candidate.candidate_id == child.candidate_id for candidate in new_archive)
            recorded_child = child if kept else replace(child, kept=False)
            archive = new_archive
            history = (*history, recorded_child)

        snapshot = RunSnapshot(
            run_id=run_id,
            task=task,
            status="completed",
            current_generation=generations,
            max_generations=generations,
            archive=archive,
            history=history,
            created_at=now,
            updated_at=_utc_now(),
        )
        self._runs = {**self._runs, run_id: snapshot}
        return snapshot

    def get_run(self, run_id: str) -> RunSnapshot:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise ValueError(f"Unknown run: {run_id}") from exc

    def get_candidate(self, run_id: str, candidate_id: str) -> CandidateRecord:
        run = self.get_run(run_id)
        for candidate in run.history:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise ValueError(f"Unknown candidate: {candidate_id}")

    def export_best_program(self, run_id: str) -> str:
        run = self.get_run(run_id)
        if run.best_candidate is None:
            raise ValueError(f"Run has no candidates: {run_id}")
        return run.best_candidate.program


def apply_proposal(parent_program: str, proposal: str) -> str:
    """Apply a small diff language used by both model and fallback proposers."""
    if proposal.startswith("set:"):
        return proposal.removeprefix("set:").lstrip("\n")
    if proposal.startswith("append:"):
        return parent_program + proposal.removeprefix("append:")
    if proposal.startswith("replace:"):
        replacement = proposal.removeprefix("replace:")
        old, separator, new = replacement.partition("=>")
        if separator:
            return parent_program.replace(old, new)
    return proposal


def _add_to_archive(
    archive: Sequence[CandidateRecord],
    candidate: CandidateRecord,
    max_size: int,
) -> tuple[CandidateRecord, ...]:
    by_program: dict[str, CandidateRecord] = {item.program: item for item in archive}
    existing = by_program.get(candidate.program)
    if existing is None or _candidate_sort_key(candidate) < _candidate_sort_key(existing):
        by_program = {**by_program, candidate.program: candidate}
    ordered = sorted(by_program.values(), key=_candidate_sort_key)
    return tuple(ordered[:max_size])


def _candidate_sort_key(candidate: CandidateRecord) -> tuple[float, float, int]:
    return (
        -candidate.score,
        candidate.evaluation.execution_time,
        candidate.generation,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
