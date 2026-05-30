# Autoresearch Rebuild

This repository now contains a generic autoresearch-style backbone for
algorithm and broader research loops under `alphaevolve/autoresearch/`.

## Design

The design borrows the key constraints from Karpathy's March 2026
`autoresearch` and broadens them using the ecosystem map collected in
`yibie/awesome-autoresearch`:

- a human-authored `program.md`,
- an immutable evaluator in `prepare.py`,
- one mutable file for the agent (`candidate.py`),
- a compact `memory.md` for reusable discoveries,
- one primary metric,
- a fixed per-run budget,
- keep/discard decisions recorded in `results.tsv`.

## Why this version is broader

The original repo is specialized for short LLM training runs with `train.py`
and `val_bpb`. This rebuild generalizes the loop to:

- optimization tasks can return a scalar objective,
- proof-oriented tasks can return bounded proof scores and counterexamples,
- evaluation and red-teaming tasks can score attack or defense success,
- retrieval and trading tasks can use domain-specific metrics under the same loop,
- the runtime stays small and file-based so agents can still reason about it.

## Main Files

- `alphaevolve/autoresearch/runtime.py`: runtime, result types, TSV logging.
- `scripts/init_autoresearch_exp.py`: create new experiment directories.
- `scripts/run_autoresearch.py`: run one evaluation and produce a keep/discard decision.
- `workspace/examples/sorting_proof/`: proof-oriented example experiment.
- `workspace/examples/toy_law_discovery/`: scientific-research style example experiment.

## Example Commands

```bash
python -m scripts.run_autoresearch workspace/examples/sorting_proof
python -m scripts.run_autoresearch workspace/examples/toy_law_discovery
python -m scripts.init_autoresearch_exp --name my_algorithm_lab
python -m scripts.init_autoresearch_exp --name my_conjecture --kind proof_search
python -m scripts.init_autoresearch_exp --name my_law_search --kind scientific_research
```

## Intended Workflow

1. Create an experiment directory.
2. Write the human intent in `program.md`.
3. Freeze `prepare.py`.
4. Let the agent iterate on `candidate.py`.
5. Use `results.tsv` plus `artifacts/best_result.json` as the experiment ledger.

## Supported Experiment Families

Inspired by `awesome-autoresearch`, the intended first-class families are:

- `proof`
- `software_optimization`
- `scientific_research`
- `evaluation_red_teaming`
- `retrieval_or_rag`
- `finance_or_trading`
