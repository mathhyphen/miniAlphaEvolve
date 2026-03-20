"""CLI for Steiner-ratio counterexample search and extremal-structure discovery.

Usage:
    python -m run.steiner_ratio_search
    python -m run.steiner_ratio_search --terminals 4 --generations 400 --population-size 128
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from alphaevolve.research import (
    STEINER_RATIO_CONJECTURE,
    SearchConfig,
    SearchOutcome,
    evolutionary_search,
    write_outcome_report,
)


def _default_output_dir() -> Path:
    path = Path(f"outputs/steiner_ratio_search_{datetime.now():%Y%m%d_%H%M%S}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_configs(args: argparse.Namespace) -> List[SearchConfig]:
    configs: List[SearchConfig] = []
    for run_index in range(args.runs):
        for terminal_index, terminals in enumerate(args.terminals):
            configs.append(
                SearchConfig(
                    num_terminals=terminals,
                    population_size=args.population_size,
                    generations=args.generations,
                    elite_count=args.elite_count,
                    mutation_sigma=args.mutation_sigma,
                    mutation_decay=args.mutation_decay,
                    crossover_rate=args.crossover_rate,
                    random_injection_rate=args.random_injection_rate,
                    local_trials=args.local_trials,
                    min_terminal_separation=args.min_separation,
                    seed=args.seed + run_index * len(args.terminals) + terminal_index,
                )
            )
    return configs


def _label_for_config(config: SearchConfig) -> str:
    if config.num_terminals == 3:
        return f"three_terminal_exact_search_seed_{config.seed}"
    if config.num_terminals == 4:
        return f"four_terminal_upper_bound_search_seed_{config.seed}"
    return f"{config.num_terminals}_terminal_search"


def _summary_payload(outcomes: Iterable[SearchOutcome]) -> dict:
    outcome_list = list(outcomes)
    return {
        "conjectured_lower_bound": STEINER_RATIO_CONJECTURE,
        "searches": [
            {
                "label": outcome.label,
                "num_terminals": outcome.config.num_terminals,
                "best_ratio": outcome.best.ratio,
                "gap_above_lower_bound": outcome.best.ratio - STEINER_RATIO_CONJECTURE,
                "best_topology": outcome.best.topology,
                "stored_candidates": len(outcome.top_candidates),
                "minimum_pairwise_distance": outcome.best.metadata.get("minimum_pairwise_distance"),
            }
            for outcome in outcome_list
        ],
        "best_by_terminal_count": {
            str(num_terminals): {
                "label": min(
                    (
                        outcome
                        for outcome in outcome_list
                        if outcome.config.num_terminals == num_terminals
                    ),
                    key=lambda outcome: outcome.best.ratio,
                ).label,
                "best_ratio": min(
                    outcome.best.ratio
                    for outcome in outcome_list
                    if outcome.config.num_terminals == num_terminals
                ),
            }
            for num_terminals in sorted({outcome.config.num_terminals for outcome in outcome_list})
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run evolutionary point-set searches for the Euclidean Steiner ratio conjecture."
    )
    parser.add_argument(
        "--terminals",
        type=int,
        action="append",
        choices=[3, 4],
        help="Terminal count to search. Can be passed multiple times. Defaults to both 3 and 4.",
    )
    parser.add_argument("--population-size", type=int, default=72, help="Population size per search.")
    parser.add_argument("--generations", type=int, default=120, help="Number of generations per search.")
    parser.add_argument("--runs", type=int, default=2, help="Number of seeds to run for each terminal count.")
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
    parser.add_argument(
        "--min-separation",
        type=float,
        default=0.0,
        help="Minimum allowed pairwise terminal distance after normalization.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for reports.")
    args = parser.parse_args()

    if args.terminals is None:
        args.terminals = [3, 4]

    configs = _build_configs(args)
    outcomes = [evolutionary_search(config, label=_label_for_config(config)) for config in configs]

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    write_outcome_report(outcomes, output_dir)

    summary = _summary_payload(outcomes)
    print(json.dumps(summary, indent=2))
    print(
        "Interpretation: 4-terminal results are upper bounds from restricted full-topology search, "
        "so they are evidence for candidate structures, not proofs of a counterexample."
    )
    print(f"Saved reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
