# AlphaEvolve Research Report

Date: 2026-03-21

## Scope

This report answers a specific question for this repository:

- what Google DeepMind's AlphaEvolve actually is,
- how the current committed code in this repository compares to it,
- and what concrete engineering steps are required to move this project materially closer to that architecture.

This report intentionally ignores the current workspace's unrelated uncommitted experiment files.

## Primary Sources

- Google DeepMind blog, May 14, 2025:
  <https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/>
- Google DeepMind white paper PDF:
  <https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf>

The official blog states that AlphaEvolve combines large language models with automated evaluators, and that its prompt sampler assembles prompts, models generate new programs, evaluators score them, and the resulting programs are stored in a programs database used for future prompts.

## What DeepMind AlphaEvolve is

The official architecture is a general-purpose evolutionary coding system for problems with automatically verifiable solutions.

At the highest level, DeepMind's system is:

1. user supplies an initial program, evaluation code, and optional background context,
2. a prompt sampler builds rich prompts from a program database,
3. an LLM ensemble proposes code modifications,
4. the system applies those diffs to produce child programs,
5. automated evaluators execute and score the children,
6. promising programs are written back into the program database.

The official white paper also makes four details explicit:

- the program database is the central memory of the system,
- prompts may include previous solutions, evaluation results, code snippets, equations, or PDFs,
- the API supports `EVOLVE-BLOCK` markers so only parts of a larger codebase are evolved,
- AlphaEvolve is designed to evolve different abstractions, not only final answers but also search procedures.

## What this repository currently is

The committed code in this repository is now a combination of four different lines:

1. RL/PPO-driven code patching:
   - [`training/evolution_loop.py`](/D:/apps/AlphaEvolve/training/evolution_loop.py)
   - [`alphaevolve/rl/patch_generator.py`](/D:/apps/AlphaEvolve/alphaevolve/rl/patch_generator.py)
   - [`alphaevolve/rl/policy_network.py`](/D:/apps/AlphaEvolve/alphaevolve/rl/policy_network.py)

2. Deterministic benchmark and sandbox execution:
   - [`alphaevolve/benchmarks/problem_suite.py`](/D:/apps/AlphaEvolve/alphaevolve/benchmarks/problem_suite.py)
   - [`alphaevolve/sandbox/executor.py`](/D:/apps/AlphaEvolve/alphaevolve/sandbox/executor.py)
   - [`run/problem_suite.py`](/D:/apps/AlphaEvolve/run/problem_suite.py)

3. Research-oriented Steiner-ratio search:
   - [`alphaevolve/research/steiner_ratio_search.py`](/D:/apps/AlphaEvolve/alphaevolve/research/steiner_ratio_search.py)
   - [`run/steiner_ratio_search.py`](/D:/apps/AlphaEvolve/run/steiner_ratio_search.py)
   - [`run/steiner_ratio_sweep.py`](/D:/apps/AlphaEvolve/run/steiner_ratio_sweep.py)

4. Minimal AlphaEvolve-style scaffold:
   - [`alphaevolve/llm_evolution/__init__.py`](/D:/apps/AlphaEvolve/alphaevolve/llm_evolution/__init__.py)
   - [`tests/test_llm_evolution.py`](/D:/apps/AlphaEvolve/tests/test_llm_evolution.py)

## Current similarity to DeepMind AlphaEvolve

The repository is now closer than before, but it is still not the same core system.

### What now aligns

- There is now a code-centric `proposal -> evaluate -> archive` controller loop.
- There is now a score-ordered `ProgramDatabase`.
- There is now archive-backed inspiration sampling plus a `PromptSampler`.
- There is now a prompt-driven diff proposer interface.
- The repository already had automated evaluation and reproducible reporting.

### What still does not align

- The main shipped loop of the repository is still not LLM-first; RL and research loops remain the dominant concrete workflows.
- The new LLM scaffold is model-agnostic infrastructure, not a full running AlphaEvolve stack.
- There is still no `EVOLVE-BLOCK` integration into real user programs.
- There is still no rich prompt assembly that includes evaluation traces, background documents, or co-evolved prompt strategies.
- There is still no distributed controller / evaluator pool resembling the official design.

## Priority Engineering Gaps

If the goal is to move this repository materially closer to DeepMind AlphaEvolve, these are the top five gaps in order.

### 1. No real LLM-backed proposal engine

Current state:

- [`alphaevolve/llm_evolution/__init__.py`](/D:/apps/AlphaEvolve/alphaevolve/llm_evolution/__init__.py) now defines `PromptDrivenDiffProposer`, but only against a generic `TextGenerator` protocol.

Gap:

- there is no concrete model backend wired into the controller loop.

Why this is first:

- In DeepMind AlphaEvolve, LLM-generated diffs are the main creative operator.
- Without a real proposer, the architecture exists only as scaffolding.

### 2. No `EVOLVE-BLOCK` program API

Current state:

- the repository does not expose a way to mark mutable regions inside a larger codebase and evolve only those regions.

Gap:

- DeepMind's system is designed around evolving marked code blocks while leaving the surrounding scaffold intact.

Why this matters:

- It is the practical bridge from toy string programs to real codebase optimization.

### 3. Prompt construction is still too thin

Current state:

- [`alphaevolve/llm_evolution/__init__.py`](/D:/apps/AlphaEvolve/alphaevolve/llm_evolution/__init__.py) has a minimal `SimplePromptSampler`.

Gap:

- prompts do not yet include rendered evaluation results, rich historical context, domain notes, or alternate prompt templates.

Why this matters:

- The white paper emphasizes that prompt richness is a central part of the search system, not a cosmetic wrapper.

### 4. No general evaluator contract shared across loops

Current state:

- benchmarking, RL training, Steiner research, and LLM scaffold each use separate evaluation shapes.

Relevant files:

- [`alphaevolve/benchmarks/problem_suite.py`](/D:/apps/AlphaEvolve/alphaevolve/benchmarks/problem_suite.py)
- [`training/evolution_loop.py`](/D:/apps/AlphaEvolve/training/evolution_loop.py)
- [`alphaevolve/research/steiner_ratio_search.py`](/D:/apps/AlphaEvolve/alphaevolve/research/steiner_ratio_search.py)
- [`alphaevolve/llm_evolution/__init__.py`](/D:/apps/AlphaEvolve/alphaevolve/llm_evolution/__init__.py)

Gap:

- there is no unified evaluator payload for correctness, metrics, metadata, and artifacts.

Why this matters:

- A shared evaluator contract is what would let a future AlphaEvolve controller operate across multiple problem families without custom glue every time.

### 5. No archive/database as the global system of record

Current state:

- the new `ProgramDatabase` is local to the minimal scaffold,
- the research and RL lines have their own state stores and histories.

Gap:

- there is no single archive abstraction that can store programs, scores, feedback, prompts, ancestry, and artifacts in one place.

Why this matters:

- In the official design, the database is not just a leaderboard; it is the memory that drives future prompt construction and exploration.

## Recommended Direction

The right next move is not to replace the whole repository. It is to add one narrow but real vertical slice:

1. take the new LLM scaffold,
2. add a concrete model backend,
3. add `EVOLVE-BLOCK` parsing and diff application,
4. define a shared evaluator result schema,
5. run one end-to-end problem through that path.

That would produce the first genuinely AlphaEvolve-like workflow in this repository.

## Bottom Line

This repository is no longer purely RL-centric, and it now contains the beginnings of an AlphaEvolve-style controller. But it is still not a faithful implementation of DeepMind AlphaEvolve.

The main reason is simple:

- DeepMind AlphaEvolve is centered on LLM-generated diffs operating over a program database with rich prompt construction.
- This repository still centers on RL and research-specific loops, with the LLM architecture only partially in place.

That is the exact gap the new `plan.md` should address.
