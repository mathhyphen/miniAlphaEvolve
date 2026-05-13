# AlphaEvolve Research Report

Date: 2026-03-21

## Purpose

This report compares the current state of this repository with Google DeepMind's official AlphaEvolve architecture and identifies the most important gaps for future work.

## Primary Sources

- Google DeepMind blog:
  <https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/>
- Google DeepMind AlphaEvolve white paper:
  <https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf>

## Official AlphaEvolve Summary

From the official materials, AlphaEvolve is best understood as an LLM-guided evolutionary controller for problems with machine-checkable evaluation.

Its defining loop is:

1. A human specifies the problem, evaluation code, and an initial program.
2. A prompt sampler builds rich prompts from a program database and prior results.
3. One or more LLMs propose code modifications or new program variants.
4. The system applies those changes to produce child programs.
5. Evaluators run the children and score them automatically.
6. Strong candidates are stored back into the program database and used to guide future proposals.

Important traits described by DeepMind:

- The search object is usually code.
- The proposal operator is LLM generation, not a fixed RL action space.
- The archive is central rather than incidental.
- Evaluation is automatic and problem-specific.
- The system can evolve different abstractions, including raw constructions, constructor functions, and search procedures.
- The codebase can expose evolve-marked regions rather than requiring every problem to fit one narrow internal representation.

## Current Repository Summary

This repository is now a hybrid of several lines of work:

### 1. RL/PPO patch learning

- [`training/evolution_loop.py`](/D:/apps/AlphaEvolve/training/evolution_loop.py)
- [`alphaevolve/rl/patch_generator.py`](/D:/apps/AlphaEvolve/alphaevolve/rl/patch_generator.py)
- [`alphaevolve/rl/policy_network.py`](/D:/apps/AlphaEvolve/alphaevolve/rl/policy_network.py)

This line learns patch actions through a policy network. It is useful research infrastructure, but it is not the same as AlphaEvolve's LLM-first mutation strategy.

### 2. Deterministic evaluation / execution infrastructure

- [`alphaevolve/sandbox/executor.py`](/D:/apps/AlphaEvolve/alphaevolve/sandbox/executor.py)
- [`alphaevolve/benchmarks/problem_suite.py`](/D:/apps/AlphaEvolve/alphaevolve/benchmarks/problem_suite.py)
- [`run/problem_suite.py`](/D:/apps/AlphaEvolve/run/problem_suite.py)

This line gives the repository a solid machine-checkable evaluation base, which *is* aligned with AlphaEvolve's problem setting.

### 3. Research search loops for Steiner ratio

- [`alphaevolve/research/steiner_ratio_search.py`](/D:/apps/AlphaEvolve/alphaevolve/research/steiner_ratio_search.py)
- [`run/steiner_ratio_search.py`](/D:/apps/AlphaEvolve/run/steiner_ratio_search.py)
- [`run/steiner_ratio_sweep.py`](/D:/apps/AlphaEvolve/run/steiner_ratio_sweep.py)

This line is a genuine discovery workflow with automatic evaluation, archive-like reporting, and iterative search. However, it evolves point sets directly, not code via LLM-generated diffs.

### 4. New minimal LLM-evolution scaffold

- [`alphaevolve/llm_evolution/__init__.py`](/D:/apps/AlphaEvolve/alphaevolve/llm_evolution/__init__.py)
- [`tests/test_llm_evolution.py`](/D:/apps/AlphaEvolve/tests/test_llm_evolution.py)

This is the repository's first code-centric AlphaEvolve-style controller skeleton. It now includes:

- `ProgramCandidate`
- `ProgramDatabase`
- archive-backed inspiration sampling
- `PromptSampler`
- `PromptDrivenDiffProposer`
- `run_evolution_loop()`

This is the clearest architectural bridge toward the DeepMind system.

## Comparison: Same Idea or Not?

### Short answer

Not originally. The repository's earlier center of gravity was RL patch learning plus benchmark infrastructure, not DeepMind-style LLM-driven code evolution.

### More precise answer

The repository is now partially aligned at the architectural level, but it is still not a full AlphaEvolve-style system.

What matches now:

- automated evaluation,
- archive/database thinking,
- iterative improvement loops,
- problem-specific discovery workflows,
- a minimal prompt-driven proposal/evaluate/archive controller.

What still does not match:

- no real LLM ensemble is wired into the controller,
- no evolve-block API exists yet,
- no rich prompt construction from detailed evaluator feedback,
- no distributed evaluator pool,
- no unified top-level controller that treats code evolution as the main workflow of the project.

## Main Gaps

The highest-priority remaining gaps are:

1. **Real model backend integration**
   The repository has model-agnostic interfaces, but no actual LLM-backed proposer connected to the new controller loop.

2. **Evolve-block program representation**
   DeepMind AlphaEvolve supports evolving parts of a larger program. This repository still lacks a dedicated representation for marked mutable blocks and diff application within those blocks.

3. **Archive richness**
   The current `ProgramDatabase` stores candidates and scores, but it does not yet store enough structured feedback to support strong prompt construction from prior trials.

4. **Unified controller surface**
   The project still feels like several parallel research tracks rather than one coherent AlphaEvolve-style entrypoint.

5. **Evaluator/controller scaling**
   The current loops are local and simple. They do not yet resemble the evaluator-pool and controller orchestration described by DeepMind.

## Recommended Direction

The right next step is not to delete the RL or research work. It is to make the new LLM-evolution scaffold the primary architectural spine for code-centric discovery.

That means:

1. Keep the current benchmark and sandbox layers as evaluation infrastructure.
2. Keep the Steiner-ratio research line as a domain-specific discovery workflow.
3. Keep PPO/RL as an optional sidecar research path, not the main AlphaEvolve analog.
4. Promote the new `alphaevolve.llm_evolution` package into a more realistic controller:
   - real model backend,
   - evolve-block parsing,
   - archive-backed prompt sampling,
   - richer evaluator feedback,
   - one dedicated CLI for code evolution experiments.

## Bottom Line

This repository now contains enough pieces to move toward an AlphaEvolve-like architecture, but it should still be described as **AlphaEvolve-inspired** rather than a faithful implementation.

The strongest claim that is currently justified is:

- it now supports machine-checkable evolutionary discovery workflows,
- and it now has a minimal prompt-driven LLM-evolution scaffold that can be extended into a much closer AlphaEvolve analog.
