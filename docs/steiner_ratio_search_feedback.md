# Steiner Ratio Search Feedback

Date: 2026-03-20

## What this experiment does

The new `run.steiner_ratio_search` runner treats point sets as the search object and
minimizes `SMT(P) / MST(P)` directly.

- `3` terminals use an exact evaluator via the geometric median.
- `4` terminals use a valid upper bound built from the best of:
  - the MST itself
  - the three full Steiner topologies with two Steiner points optimized by alternating geometric-median updates

For `4` terminals, any candidate with upper bound below `sqrt(3)/2` would already be a certified counterexample, because the true SMT can only be shorter.

## Commands used

```bash
python -m run.steiner_ratio_search --runs 2 --population-size 56 --generations 90
python -m run.steiner_ratio_search --terminals 4 --runs 4 --population-size 72 --generations 140
python -m run.steiner_ratio_search --terminals 4 --runs 4 --population-size 72 --generations 140 --min-separation 0.05
```

## Current search results

### Three-terminal sanity check

The `3`-terminal search repeatedly recovers the equilateral triangle and hits the
conjectured value exactly:

- best ratio: `0.866025403784`

That confirms the experiment is at least discovering the known extremizer.

### Unconstrained four-terminal search

Four independent runs with `population_size=72`, `generations=140`, `local_trials=3` produced:

- seed `0`: best ratio `0.867605067960`
- seed `1`: best ratio `0.868448420799`
- seed `2`: best ratio `0.866510194304`
- seed `3`: best ratio `0.867209236385`

Observed pattern:

- the best run stayed above the conjectured lower bound by about `4.85e-4`
- every strong run contained a very small terminal pair
- minimum pairwise distances were approximately `0.0084`, `0.0097`, `0.0022`, and `0.0089`

This is strong evidence that the optimizer is mostly exploiting a near-collision mechanism, effectively degenerating toward the classical `3`-terminal extremizer.

### Constrained four-terminal search

Adding `--min-separation 0.05` rules out the near-collision family after normalization.

Four constrained runs with the same search budget produced:

- seed `0`: best ratio `0.870616001277`
- seed `1`: best ratio `0.871156605773`
- seed `2`: best ratio `0.870398218682`
- seed `3`: best ratio `0.870405308513`

Observed pattern:

- the best constrained run is now about `0.00437` above the conjectured lower bound
- minimum pairwise distances stayed near the requested floor: about `0.0516` to `0.0614`
- once collapse is forbidden, the search no longer gets nearly as close to `sqrt(3)/2`

That makes the interpretation much clearer: the near-bound unconstrained `4`-terminal candidates are not a stable new extremal family. They are largely a degeneration effect.

## Interpretation

- No counterexample was found.
- The search is strong enough to recover the known `3`-terminal extremizer exactly.
- For `4` terminals, the dominant low-ratio mechanism is terminal clustering.
- Minimum-separation constraints are essential if the goal is genuine structure discovery rather than rediscovering the triangle case in disguise.

## Recommended next steps

1. Sweep `min_terminal_separation` over a grid such as `0.02, 0.05, 0.1, 0.15`.
2. Add a convex-position-only mode to separate true `4`-point geometry from degenerate triangle-like families.
3. Cluster the best constrained candidates by hull size, pairing topology, and aspect ratio.
4. Extend the framework to `5` terminals only after the constrained `4`-terminal picture stabilizes.
