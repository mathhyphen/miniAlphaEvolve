# AlphaEvolve Alignment Notes

Date: 2026-03-21

## Reference

This comparison is based on Google DeepMind's official AlphaEvolve materials:

- Blog: <https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/>
- White paper PDF: <https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf>

## What DeepMind AlphaEvolve is

According to the white paper, AlphaEvolve is an LLM-guided evolutionary coding agent for problems with automatically verifiable solutions. The core loop is:

1. Human provides a problem definition, an initial program, and evaluation code.
2. A prompt sampler builds rich prompts from a program database and past results.
3. One or more LLMs propose code diffs.
4. The diffs are applied to produce child programs.
5. Evaluators execute the children and score them.
6. Promising programs are added back to the database, closing the evolutionary loop.

Important properties:

- the search object is usually code,
- proposals come from LLMs rather than a learned RL policy over a fixed mutation space,
- evaluation is automatic and problem-specific,
- the archive/database is a first-class component,
- the system can evolve different abstractions, from raw constructions to search procedures.

## Where this repository differed

Before the recent research-line additions, this repository was not actually following the same main idea.

It had useful pieces, but the center of gravity was different:

- [`training/evolution_loop.py`](/D:/apps/AlphaEvolve/training/evolution_loop.py) focused on PPO-style policy learning for patch actions.
- [`alphaevolve/rl/patch_generator.py`](/D:/apps/AlphaEvolve/alphaevolve/rl/patch_generator.py) searched in a fixed action space rather than using LLM-generated diffs.
- [`run/pipeline.py`](/D:/apps/AlphaEvolve/run/pipeline.py) was a legacy mixed CLI for benchmark tasks and code execution.
- The newer Steiner-ratio work under [`alphaevolve/research/steiner_ratio_search.py`](/D:/apps/AlphaEvolve/alphaevolve/research/steiner_ratio_search.py) evolves point sets directly, which is valid for discovery research but is not itself the same architecture as DeepMind AlphaEvolve.

So the short answer was:

- **No, the project was not really the same overall approach.**
- It was closer to a combination of:
  - RL-driven patch optimization,
  - deterministic benchmarking,
  - and separate evolutionary search experiments.

## What has now been aligned

The recent changes improved the repository in the direction of AlphaEvolve-style research workflows, even though it is still not a full reproduction.

What is now closer:

- there is a clear proposal/evaluate/archive style research workflow in the Steiner-ratio line,
- there are dedicated runners for structured experiments and sweeps,
- evaluation is automatic and machine-checkable,
- reports and archives make iterative discovery reproducible.

Relevant files:

- [`alphaevolve/research/steiner_ratio_search.py`](/D:/apps/AlphaEvolve/alphaevolve/research/steiner_ratio_search.py)
- [`run/steiner_ratio_search.py`](/D:/apps/AlphaEvolve/run/steiner_ratio_search.py)
- [`run/steiner_ratio_sweep.py`](/D:/apps/AlphaEvolve/run/steiner_ratio_sweep.py)

## What is still missing for real AlphaEvolve parity

This repository still does **not** implement the most distinctive DeepMind AlphaEvolve ingredient:

- an LLM ensemble that proposes substantial code diffs using prompts built from an evolving program database.

The biggest remaining gaps are:

1. No prompt sampler that constructs rich context from prior trials, inspirations, and feedback.
2. No LLM-based diff generation loop over evolve-marked code blocks.
3. No general program database that stores code candidates plus evaluation metadata as the central search memory.
4. No controller loop whose main mutation operator is "ask model for an improved diff, then execute it."

## Recommended next step

If the goal is to move materially closer to DeepMind AlphaEvolve, the next architectural milestone should be:

- add a lightweight LLM-diff evolutionary loop that sits beside the current RL code, not inside it.

That loop should look like:

1. sample parent program(s) from an archive,
2. build a prompt with code, scores, and prior feedback,
3. ask an LLM for a diff,
4. apply the diff,
5. run the evaluator,
6. store the child and its metrics back into the archive.

Only after that would this repository deserve to be called "AlphaEvolve-like" in the same primary sense as DeepMind's system.
