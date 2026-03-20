# Plan

Date: 2026-03-21

## Goal

Move this repository from "AlphaEvolve-inspired collection of research components" toward a more coherent AlphaEvolve-style system centered on LLM-driven program evolution.

## Ground Truth From Research

Based on Google DeepMind's official AlphaEvolve materials, the defining architecture is:

1. problem definition + initial program + automatic evaluator
2. program database / archive
3. prompt sampler built from past trials and inspirations
4. LLM-generated code diffs
5. automatic execution and scoring
6. archive update and repeat

## Current State

Already present:

- sandboxed execution and deterministic evaluation
- benchmark infrastructure
- research workflows with archive-like reporting
- minimal LLM-evolution scaffold with prompt-driven proposal interfaces

Still missing:

- real LLM backend integration for the new scaffold
- evolve-block parsing and block-scoped diff application
- richer archive metadata and prompt construction
- one unified CLI for code evolution experiments
- stronger evaluator/controller orchestration

## Execution Plan

### Phase 1: Make the LLM scaffold usable

1. Add a real model backend adapter under `alphaevolve/llm_evolution/`.
2. Add prompt templates that include parent code, archive inspirations, scores, and evaluator feedback.
3. Add a small end-to-end CLI that runs the new controller loop on a toy benchmark.

Success criteria:

- one command can run a prompt-driven proposal/evaluate/archive loop,
- the controller uses a real model interface,
- and results are persisted as reproducible artifacts.

### Phase 2: Add evolve-block semantics

1. Introduce a block marker format such as `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END`.
2. Parse mutable blocks from a larger code file.
3. Restrict diff application to those blocks.
4. Evaluate the reconstructed full program.

Success criteria:

- the system can evolve part of a larger program without replacing the whole file,
- and tests verify block-scoped mutation behavior.

### Phase 3: Enrich the archive

1. Extend `ProgramDatabase` to store:
   - scores,
   - proposal history,
   - evaluator diagnostics,
   - parent-child lineage,
   - optional prompt metadata.
2. Add archive sampling strategies for:
   - best-so-far,
   - diverse inspirations,
   - recent successful variants.

Success criteria:

- prompt construction can draw on more than just top scores,
- and the archive supports lineage-aware analysis.

### Phase 4: Unify the code-evolution surface

1. Add a dedicated CLI for AlphaEvolve-style code evolution.
2. Keep RL/PPO and research search loops separate from this main path.
3. Document the intended boundaries between:
   - benchmark infrastructure,
   - code evolution,
   - domain-specific research experiments.

Success criteria:

- a new contributor can identify the main AlphaEvolve-style workflow immediately,
- and the repository reads as one system rather than several unrelated prototypes.

### Phase 5: Scale evaluation

1. Add evaluator-pool abstractions.
2. Support multiple evaluation metrics and richer feedback.
3. Add artifact logging for every generation.
4. Add long-run experiment reports.

Success criteria:

- larger searches can be run reproducibly,
- and prompt/evaluator/archive interactions are inspectable after the run.

## Non-Goals For Now

- replacing the Steiner-ratio research line
- deleting the PPO/RL code
- claiming full parity with Google DeepMind AlphaEvolve before a real LLM-backed controller exists

## Immediate Next Step

Implement Phase 1 first: connect the new prompt-driven scaffold to a real model backend and add one toy end-to-end CLI demo.
