# Steiner Ratio Experiment

This repository now includes a research-only search loop for Euclidean Steiner-ratio experiments.

The current scope is intentionally narrow:

- `3` terminals: exact evaluation via the geometric median, which should recover the classical equilateral-triangle extremizer near `sqrt(3)/2`.
- `4` terminals: upper-bound search over the three full Steiner topologies, with alternating geometric-median updates for the two Steiner points and an MST fallback.

The goal is not to prove the conjecture directly. The goal is to:

- search for low-ratio candidate point sets,
- cluster the geometric motifs that appear near the lower bound,
- and surface possible counterexample candidates for deeper verification.

Run the experiment with:

```bash
python -m run.steiner_ratio_search
python -m run.steiner_ratio_search --terminals 4 --runs 3 --min-separation 0.1
python -m run.steiner_ratio_sweep
python -m run.steiner_ratio_sweep --require-full-hull
```

Artifacts are written to `outputs/steiner_ratio_search_<timestamp>/`:

- `config.json`
- `results.json`
- `history.json`
- `best_candidates.json`
- `counterexample_candidates.json`
- `report.md`

Sweep artifacts are written to `outputs/steiner_ratio_sweep_<timestamp>/`:

- `summary.json`
- `sweep_summary.json`
- `report.md`
- `runs/results.json`
- `runs/report.md`

The current evaluator contract is conservative:

- For `3` terminals it returns the exact ratio.
- For `4` terminals it returns the best ratio found among the three full topologies or the MST fallback, so it is an *upper bound* on the true Steiner ratio, not a proof-grade lower-bound certificate.

That means:

- if the reported `4`-terminal upper bound is still above `sqrt(3)/2`, it is only evidence and needs further improvement;
- if the reported `4`-terminal upper bound ever drops below `sqrt(3)/2`, that point set is already a certified counterexample, because the true SMT can only be shorter.

For extremal-structure discovery, `--min-separation` is important. Without it, the search tends to collapse two terminals together and imitate the `3`-terminal extremizer instead of revealing a genuinely new `4`-terminal family.

The dedicated sweep runner is useful when the question is structural rather than adversarial: it lets you see how the best ratio changes as terminal collisions are ruled out.

`--require-full-hull` is the stronger nondegeneracy mode for `4` terminals:

- it enforces `hull_size == 4` during the search rather than only reporting it afterward,
- it removes triangle-like hull-3 best candidates from the search space,
- and it is the right mode when the question is specifically about genuinely four-point extremal structure.
