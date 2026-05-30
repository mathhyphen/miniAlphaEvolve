# Proof Mode

This repository now includes a proof-oriented layer under `alphaevolve/proof/`.

It is designed for proof assistance, not for pretending bounded testing is a
full theorem proof. The current workflow is:

1. write a candidate algorithm as a Python function,
2. define bounded cases, properties, and contracts,
3. run the proof harness,
4. inspect the returned counterexamples and proof certificate.

## What It Supports

- example obligations: exact input/output checks,
- bounded property obligations: exhaustive or curated finite domains,
- contract obligations: preconditions, postconditions, invariants,
- proof-driven scoring for AlphaEvolve loops.

## Main Entry Points

- `alphaevolve.proof.ProofHarness`
- `alphaevolve.proof.ProofDrivenEvaluator`
- `python -m run.prove_sorting`

## Example

```bash
python -m run.prove_sorting
python -m run.prove_sorting --candidate path/to/candidate.py
```

The sorting demo proves correctness only over a bounded domain:

- arrays of length up to `4`
- values in `0..2`

If all checks pass, that is strong bounded evidence. It is not a universal
proof over all inputs.

## How To Adapt It

For a new algorithm, create a new harness that defines:

- the target function name,
- canonical examples,
- bounded domains or counterexample generators,
- postconditions and invariants that capture correctness.

If your end goal is a real machine-checked proof, the next step is to add a
formal backend such as Lean, Coq, or an SMT-based verifier. This proof mode is
best used as the bridge between LLM search and formalization.
