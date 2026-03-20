"""Tests for the minimal AlphaEvolve-style controller scaffold."""

from __future__ import annotations

from alphaevolve.llm_evolution import (
    ProgramCandidate,
    ProgramDatabase,
    run_evolution_loop,
)


def test_program_database_keeps_best_score_per_program() -> None:
    archive = ProgramDatabase(max_size=2)
    archive.add(ProgramCandidate(program="alpha", score=1.0, candidate_id="a"))
    archive.add(ProgramCandidate(program="beta", score=3.0, candidate_id="b"))
    archive.add(ProgramCandidate(program="alpha", score=2.0, candidate_id="a2"))

    assert len(archive) == 2
    assert archive.best().program == "beta"
    assert archive.top()[1].program == "alpha"
    assert archive.top()[1].score == 2.0


def test_run_evolution_loop_applies_diff_and_archives_results() -> None:
    class ToyProposer:
        def __init__(self) -> None:
            self.calls = 0

        def propose(self, parent: ProgramCandidate, archive: ProgramDatabase) -> str:
            del archive
            self.calls += 1
            if self.calls == 1:
                return "append:!"
            return f"replace:{parent.program}=>{parent.program}?"

    class LengthEvaluator:
        def evaluate(self, program: str) -> float:
            return float(len(program))

    result = run_evolution_loop(
        "alpha",
        ToyProposer(),
        LengthEvaluator(),
        generations=3,
        archive_size=4,
    )

    assert len(result.history) == 3
    assert result.history[0].candidate.program == "alpha!"
    assert result.history[0].candidate.parent_id == "gen-0"
    assert result.best.program.endswith("??")
    assert result.best.score == float(len(result.best.program))


def test_run_evolution_loop_uses_best_archive_parent() -> None:
    class RecordingProposer:
        def __init__(self) -> None:
            self.parents: list[str] = []

        def propose(self, parent: ProgramCandidate, archive: ProgramDatabase) -> str:
            del archive
            self.parents.append(parent.program)
            return "append:x"

    class ScoreEvaluator:
        def evaluate(self, program: str) -> float:
            return float(program.count("x"))

    result = run_evolution_loop(
        "base",
        RecordingProposer(),
        ScoreEvaluator(),
        generations=2,
        archive_size=2,
    )

    assert result.history[0].parent_id == "gen-0"
    assert result.history[1].parent_id == "gen-1"
    assert result.best.program == "basexx"
