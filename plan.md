# AlphaEvolve Parity Plan

Date: 2026-03-21

## Goal

Move this repository from a mixed RL / benchmark / research prototype toward a genuinely AlphaEvolve-like architecture for automatically verifiable coding tasks.

## Constraints

- Do not break existing benchmark and research workflows.
- Do not overwrite the user's uncommitted experiment files.
- Keep new work adjacent to, not inside, the current RL path unless integration is justified.

## Phase 1: Make the LLM scaffold executable

Status: partially started

Tasks:

1. Add a concrete model backend for [`alphaevolve/llm_evolution/__init__.py`](/D:/apps/AlphaEvolve/alphaevolve/llm_evolution/__init__.py).
2. Add a small demo runner that executes the LLM loop on one automatically verifiable task.
3. Persist archive entries, ancestry, prompts, and scores to disk.

Exit criteria:

- one end-to-end problem runs through `prompt -> diff -> evaluate -> archive`
- results are reproducible from a CLI command

## Phase 2: Add EVOLVE-BLOCK support

Tasks:

1. Implement parsing for `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END`.
2. Restrict diff application to marked regions.
3. Preserve the surrounding code scaffold exactly.

Exit criteria:

- a larger program can expose only selected regions to evolution
- diff failures are isolated and recoverable

## Phase 3: Unify evaluator contracts

Tasks:

1. Define a shared evaluator result schema containing correctness, scalar metrics, logs, and metadata.
2. Adapt benchmark and LLM-evolution paths to emit that schema.
3. Keep sandbox execution as the underlying execution boundary where practical.

Exit criteria:

- benchmark and LLM loops can share evaluator outputs
- archive entries store more than a single scalar score

## Phase 4: Strengthen prompt construction

Tasks:

1. Extend the prompt sampler to include rendered evaluation results.
2. Add optional domain context such as equations, examples, and code snippets.
3. Add prompt templates with stochastic formatting for diversity.

Exit criteria:

- prompts are built from parent code, archive inspirations, and structured feedback
- prompt diversity is configurable

## Phase 5: Converge the archive

Tasks:

1. Introduce a global archive abstraction for program candidates.
2. Store prompts, proposals, ancestry, and evaluation artifacts together.
3. Define parent/inspiration sampling policies over that archive.

Exit criteria:

- future search loops use one archive abstraction instead of per-module ad hoc stores
- archive-backed prompt construction is the default path

## Phase 6: Run one canonical AlphaEvolve-style demo

Candidate domains:

- a deterministic algorithmic benchmark case
- matrix multiplication micro-optimization
- one constrained code-generation task with objective evaluation

Exit criteria:

- the repository demonstrates one full LLM-diff evolutionary run
- the run uses a real backend, EVOLVE-BLOCK support, archive-backed prompting, and automatic evaluation

## Near-Term Priority

If only one next coding step is taken, it should be:

- implement Phase 1 plus the smallest usable subset of Phase 2

Reason:

- until the repository can evolve a marked code block with a real model backend, it is still only architecturally adjacent to AlphaEvolve rather than operationally similar.
