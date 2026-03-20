"""CLI for min-separation sweeps in Steiner-ratio experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from alphaevolve.research import (
    STEINER_RATIO_CONJECTURE,
    run_min_separation_sweep,
    write_min_separation_sweep_report,
)


def _default_output_dir() -> Path:
    path = Path(f"outputs/steiner_ratio_sweep_{datetime.now():%Y%m%d_%H%M%S}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep min-separation constraints for 4-terminal Steiner-ratio search."
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        action="append",
        dest="separations",
        help="Minimum separation value to include in the sweep. Defaults to 0, 0.02, 0.05, 0.1, 0.15.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of seeds per separation bucket.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument("--population-size", type=int, default=64, help="Population size per run.")
    parser.add_argument("--generations", type=int, default=100, help="Generations per run.")
    parser.add_argument("--elite-count", type=int, default=12, help="Elite survivors per generation.")
    parser.add_argument("--mutation-sigma", type=float, default=0.16, help="Initial mutation scale.")
    parser.add_argument("--mutation-decay", type=float, default=0.994, help="Per-generation mutation decay.")
    parser.add_argument("--crossover-rate", type=float, default=0.35, help="Fraction of offspring from crossover.")
    parser.add_argument(
        "--random-injection-rate",
        type=float,
        default=0.15,
        help="Fraction of each generation replaced with fresh random samples.",
    )
    parser.add_argument("--local-trials", type=int, default=3, help="Mutation attempts per offspring.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for reports.")
    args = parser.parse_args()

    separations = args.separations or [0.0, 0.02, 0.05, 0.1, 0.15]
    sweep = run_min_separation_sweep(
        separations=separations,
        runs_per_separation=args.runs,
        num_terminals=4,
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        mutation_sigma=args.mutation_sigma,
        mutation_decay=args.mutation_decay,
        crossover_rate=args.crossover_rate,
        random_injection_rate=args.random_injection_rate,
        local_trials=args.local_trials,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    write_min_separation_sweep_report(sweep, output_dir)

    summary = {
        "conjectured_lower_bound": STEINER_RATIO_CONJECTURE,
        "rows": [
            {
                "min_terminal_separation": row.min_terminal_separation,
                "best_ratio": row.best_ratio,
                "mean_gap_to_conjecture": row.mean_gap_to_conjecture,
                "boundary_hugging_fraction": row.boundary_hugging_fraction,
                "hull3_fraction": row.hull3_fraction,
            }
            for row in sweep.rows
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved sweep reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
