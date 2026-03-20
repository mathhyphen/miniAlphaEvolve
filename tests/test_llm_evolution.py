"""Tests for the minimal AlphaEvolve-style controller scaffold."""

from __future__ import annotations

from alphaevolve.llm_evolution import (
    PromptDrivenDiffProposer,
    ProgramCandidate,
    ProgramDatabase,
    SimplePromptSampler,
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


def test_program_database_returns_archive_inspirations_excluding_parent() -> None:
    archive = ProgramDatabase(max_size=4)
    alpha = ProgramCandidate(program="alpha", score=1.0, candidate_id="a")
    beta = ProgramCandidate(program="beta", score=3.0, candidate_id="b")
    gamma = ProgramCandidate(program="gamma", score=2.0, candidate_id="c")
    archive.add(alpha)
    archive.add(beta)
    archive.add(gamma)

    inspirations = archive.inspirations_for(beta, limit=2)

    assert [candidate.program for candidate in inspirations] == ["gamma", "alpha"]


def test_simple_prompt_sampler_renders_parent_and_inspirations() -> None:
    archive = ProgramDatabase(max_size=3)
    parent = ProgramCandidate(program="parent()", score=1.5, candidate_id="p")
    inspiration = ProgramCandidate(program="helper()", score=2.5, candidate_id="i")
    archive.add(inspiration)
    archive.add(parent)

    prompt = SimplePromptSampler().build_prompt(parent, [inspiration], archive)

    assert "Parent candidate:" in prompt
    assert "Archive inspirations:" in prompt
    assert "parent()" in prompt
    assert "helper()" in prompt
    assert "score=1.500000" in prompt


def test_prompt_driven_diff_proposer_uses_sampler_and_model() -> None:
    archive = ProgramDatabase(max_size=3)
    parent = ProgramCandidate(program="seed", score=1.0, candidate_id="gen-0")
    archive.add(parent)
    archive.add(ProgramCandidate(program="seed+", score=2.0, candidate_id="gen-1"))

    class RecordingModel:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "append:!"

    model = RecordingModel()
    proposer = PromptDrivenDiffProposer(
        sampler=SimplePromptSampler(),
        model=model,
        inspiration_limit=1,
    )

    proposal = proposer.propose(parent, archive)

    assert proposal == "append:!"
    assert len(model.prompts) == 1
    assert "seed+" in model.prompts[0]
    assert "seed" in model.prompts[0]


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


def test_run_evolution_loop_can_use_prompt_driven_proposer() -> None:
    class SuffixModel:
        def generate(self, prompt: str) -> str:
            if "Archive inspirations" in prompt:
                return "append:y"
            return "append:x"

    class ScoreEvaluator:
        def evaluate(self, program: str) -> float:
            return float(len(program))

    result = run_evolution_loop(
        "base",
        PromptDrivenDiffProposer(
            sampler=SimplePromptSampler(),
            model=SuffixModel(),
            inspiration_limit=1,
        ),
        ScoreEvaluator(),
        generations=2,
        archive_size=4,
    )

    assert len(result.history) == 2
    assert result.best.program in {"baseyy", "basexy", "baseyx"}
    assert result.best.score == float(len(result.best.program))
