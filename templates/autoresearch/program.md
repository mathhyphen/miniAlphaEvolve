# Program: <name>

## Purpose
This experiment follows an autoresearch-style loop for algorithm work.

The human owns:
- `program.md`
- `memory.md`

The immutable evaluation harness is:
- `prepare.py`

The single mutable file for the agent is:
- `candidate.py`

## Loop Rules
1. Only edit `candidate.py`.
2. Do not manually edit `results.tsv`.
3. Keep changes only if the latest run improves the primary metric and stays within budget.
4. Prefer simple diffs over clever but fragile changes.
5. If a change is fundamentally broken, discard it quickly instead of patching endlessly.

## What To Optimize
Read `prepare.py` to learn:
- the experiment kind,
- the primary metric,
- whether lower or higher is better,
- the evaluation budget,
- the proof or objective obligations.

Check `memory.md` before proposing large changes. Prefer compounding reusable
insights over rediscovering the same failures.

## Commands
Run one evaluation:

```bash
python -m scripts.run_autoresearch .
```

Initialize a new experiment:

```bash
python -m scripts.init_autoresearch_exp --name my_experiment
```

## Notes
This experiment template is generic on purpose. Adapt `prepare.py` to your
target domain while keeping it immutable once the loop starts.

Suggested experiment kinds inspired by the wider autoresearch ecosystem:
- `proof`
- `software_optimization`
- `scientific_research`
- `evaluation_red_teaming`
- `retrieval_or_rag`
- `finance_or_trading`
