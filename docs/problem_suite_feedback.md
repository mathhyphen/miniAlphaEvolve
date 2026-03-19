# 20-Problem Suite Feedback

## Commands

```bash
python -m run.problem_suite
pytest -q
```

## Benchmark outcome

- Initial run before the baseline fixes: `0/20` passed.
- Post-fix run: `20/20` passed.
- Family breakdown after the fixes:
  - `mst`: `10/10` passed
  - `steiner`: `10/10` passed

The latest local benchmark artifacts were written to:

- `outputs/problem_suite_20260320_010727/results.json`
- `outputs/problem_suite_20260320_010727/report.md`

## What failed before

### MST family

All 10 MST cases failed before validation with:

```text
NameError: name 'pu' is not defined
```

The one-line baseline implementation relied on assignment-expression state inside a generator expression, which was not valid in practice for these cases.

### Steiner family

All 10 Steiner cases executed successfully, but every one failed validation because the baseline returned the sum of all pairwise terminal distances. That value is not a valid Steiner-tree-style objective and frequently exceeds the MST upper bound used by the benchmark validator.

## What changed

- Replaced the MST baseline with a straightforward Kruskal implementation.
- Replaced the Steiner baseline with a Euclidean MST upper-bound implementation that is valid for the suite's objective check.
- Updated the deterministic benchmark tests to assert the fixed behavior and the full `20/20` pass rate.
- Mirrored the same baseline fixes into `run/pipeline.py` so the CLI registry and the benchmark suite do not drift apart.

## Project feedback

- The new deterministic 20-case suite is useful as a regression harness for the repository's baseline problem families.
- The sandbox execution path is now stable enough to support this style of batch validation.
- The full RL/evolution path is still a separate concern:
  - it still depends on `torch`
  - it still needs stronger task-level correctness scoring before it can serve as a trustworthy benchmark driver

