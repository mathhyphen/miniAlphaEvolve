# Steiner Ratio Search Feedback

Date: 2026-03-20

## What this experiment does

The new Steiner-ratio tooling treats point sets as the search object and minimizes
`SMT(P) / MST(P)` directly.

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
python -m run.steiner_ratio_sweep --runs 3 --population-size 56 --generations 90
python -m run.steiner_ratio_sweep --runs 2 --population-size 48 --generations 80 --require-full-hull
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

### Min-separation sweep

A direct sweep over `min_terminal_separation in {0, 0.02, 0.05, 0.1, 0.15}` with
`runs=3`, `population_size=56`, `generations=90` produced:

- `0.00`: best ratio `0.868204966936`, mean gap `0.002524840107`, hull-3 fraction `1.000`
- `0.02`: best ratio `0.868677187750`, mean gap `0.003337497697`, hull-3 fraction `0.333`
- `0.05`: best ratio `0.871022807368`, mean gap `0.005070066015`, boundary-hugging fraction `0.667`
- `0.10`: best ratio `0.873884746816`, mean gap `0.008378782964`, boundary-hugging fraction `1.000`
- `0.15`: best ratio `0.878884946028`, mean gap `0.012921322757`, boundary-hugging fraction `1.000`

This is the cleanest signal for clustering:

- best ratios rise steadily as the allowed collision scale is increased,
- once the floor reaches `0.05`, the best candidates start hugging that floor,
- by `0.10` and `0.15`, the search is fully constrained by the separation limit.

So the low-ratio regime appears to be controlled mainly by terminal clustering. The scan does not support the existence of a robust new `4`-terminal extremizer below or even very near the conjectured bound.

### Full-hull sweep

To isolate genuinely nondegenerate `4`-point geometry, I also ran the sweep with
`--require-full-hull`, which forces every accepted candidate to satisfy `hull_size == 4`.

Results:

- `0.00`: best ratio `0.868263196273`, mean gap `0.003602057429`, full-hull fraction `1.000`
- `0.02`: best ratio `0.870537259024`, mean gap `0.004571752020`, full-hull fraction `1.000`
- `0.05`: best ratio `0.871225379880`, mean gap `0.005741278943`, full-hull fraction `1.000`
- `0.10`: best ratio `0.875216539012`, mean gap `0.009806039083`, full-hull fraction `1.000`
- `0.15`: best ratio `0.877651855250`, mean gap `0.012356583142`, full-hull fraction `1.000`

This adds a stronger structural conclusion:

- even without a positive separation floor, once hull-3 degenerations are forbidden the best ratio already sits noticeably above `sqrt(3)/2`,
- the low-ratio mechanism is therefore not just “any four points”, but specifically four points collapsing into a triangle-like hull,
- and genuinely four-extreme-point candidates still do not show evidence of a counterexample family.

## Interpretation

- No counterexample was found.
- The search is strong enough to recover the known `3`-terminal extremizer exactly.
- For `4` terminals, the dominant low-ratio mechanism is terminal clustering.
- Minimum-separation constraints are essential if the goal is genuine structure discovery rather than rediscovering the triangle case in disguise.
- The sweep result strengthens that claim because the best ratio moves away from the conjectured bound in a controlled, monotone way as clustering is forbidden.
- The full-hull sweep strengthens it again: even before adding a positive separation floor, forbidding hull-3 candidates already raises the best observed ratio.

## Recommended next steps

1. Increase the full-hull sweep budget so each separation bucket gets at least `5-10` seeds.
2. Add a stricter nondegenerate mode that enforces both `hull_size == 4` and a positive separation floor such as `0.05` or `0.1`.
3. Cluster the best full-hull candidates by pairing topology and aspect ratio.
4. Extend the framework to `5` terminals only after the constrained `4`-terminal picture stabilizes.
