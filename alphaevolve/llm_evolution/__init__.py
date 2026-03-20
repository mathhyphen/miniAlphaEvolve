"""Minimal AlphaEvolve-style controller loop for proposal/evaluate/archive search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProgramCandidate:
    """A single program candidate and its evaluation score."""

    program: str
    score: float
    generation: int = 0
    candidate_id: str = ""
    parent_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class DiffProposer(Protocol):
    """Propose a textual diff for a parent program."""

    def propose(self, parent: ProgramCandidate, archive: "ProgramDatabase") -> str:
        """Return a lightweight diff instruction or a full child program."""


class Evaluator(Protocol):
    """Score a child program."""

    def evaluate(self, program: str) -> float:
        """Return a higher-is-better score."""


def _apply_diff(parent_program: str, proposal: str) -> str:
    """Apply a tiny diff language or fall back to treating the proposal as a full program."""
    if proposal.startswith("append:"):
        return parent_program + proposal.removeprefix("append:")

    if proposal.startswith("replace:"):
        replacement = proposal.removeprefix("replace:")
        old, separator, new = replacement.partition("=>")
        if separator:
            return parent_program.replace(old, new)

    if proposal.startswith("set:"):
        return proposal.removeprefix("set:")

    return proposal


@dataclass
class ProgramDatabase:
    """Score-ordered archive of the best discovered programs."""

    max_size: int = 16
    _candidates: list[ProgramCandidate] = field(default_factory=list)

    def add(self, candidate: ProgramCandidate) -> None:
        """Insert a candidate, deduplicating by program text and keeping the best score."""
        for index, existing in enumerate(self._candidates):
            if existing.program == candidate.program:
                if candidate.score > existing.score:
                    self._candidates[index] = candidate
                    self._sort_and_trim()
                return

        self._candidates.append(candidate)
        self._sort_and_trim()

    def best(self) -> ProgramCandidate:
        if not self._candidates:
            raise ValueError("program database is empty")
        return self._candidates[0]

    def top(self, limit: int | None = None) -> list[ProgramCandidate]:
        if limit is None:
            return list(self._candidates)
        return list(self._candidates[:limit])

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self):
        return iter(self._candidates)

    def _sort_and_trim(self) -> None:
        self._candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        del self._candidates[self.max_size :]


@dataclass(frozen=True)
class EvolutionStep:
    """One proposal/evaluate/archive step."""

    generation: int
    parent_id: str
    candidate: ProgramCandidate
    proposal: str


@dataclass(frozen=True)
class EvolutionResult:
    """Summary of the controller loop."""

    archive: ProgramDatabase
    history: list[EvolutionStep]

    @property
    def best(self) -> ProgramCandidate:
        return self.archive.best()


def run_evolution_loop(
    initial_program: str,
    proposer: DiffProposer,
    evaluator: Evaluator,
    *,
    generations: int = 8,
    archive_size: int = 16,
) -> EvolutionResult:
    """Run a minimal AlphaEvolve-style proposal/evaluate/archive loop."""
    archive = ProgramDatabase(max_size=archive_size)
    initial_candidate = ProgramCandidate(
        program=initial_program,
        score=evaluator.evaluate(initial_program),
        generation=0,
        candidate_id="gen-0",
    )
    archive.add(initial_candidate)

    history: list[EvolutionStep] = []
    for generation in range(1, generations + 1):
        parent = archive.best()
        proposal = proposer.propose(parent, archive)
        child_program = _apply_diff(parent.program, proposal)
        child_candidate = ProgramCandidate(
            program=child_program,
            score=evaluator.evaluate(child_program),
            generation=generation,
            candidate_id=f"gen-{generation}",
            parent_id=parent.candidate_id,
            metadata={"proposal": proposal},
        )
        archive.add(child_candidate)
        history.append(
            EvolutionStep(
                generation=generation,
                parent_id=parent.candidate_id,
                candidate=child_candidate,
                proposal=proposal,
            )
        )

    return EvolutionResult(archive=archive, history=history)


__all__ = [
    "DiffProposer",
    "Evaluator",
    "EvolutionResult",
    "EvolutionStep",
    "ProgramCandidate",
    "ProgramDatabase",
    "run_evolution_loop",
]
