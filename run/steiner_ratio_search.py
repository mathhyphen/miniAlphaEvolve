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
    summarize_separation_sweep,
    write_outcome_report,
)


def _default_output_dir() -> Path:
    path = Path(f"outputs/steiner_ratio_search_{datetime.now():%Y%m%d_%H%M%S}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_configs(args: argparse.Namespace) -> List[SearchConfig]:
    configs: List[SearchConfig] = []
    min_separation_values = args.min_separation_grid or [args.min_separation]
    for run_index in range(args.runs):
        for separation_index, min_separation in enumerate(min_separation_values):
            for terminal_index, terminals in enumerate(args.terminals):
                seed_offset = run_index * len(args.terminals) * len(min_separation_values)
                seed_offset += separation_index * len(args.terminals)
                seed_offset += terminal_index
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
                        min_terminal_separation=min_separation,
                        require_full_hull=args.require_full_hull,
                        seed=args.seed + seed_offset,
                    )
                )
    return configs


def _label_for_config(config: SearchConfig) -> str:
    separation_tag = str(config.min_terminal_separation).replace(".", "p")
    hull_tag = "fullhull" if config.require_full_hull else "anyhull"
    if config.num_terminals == 3:
        return f"three_terminal_exact_search_{hull_tag}_sep_{separation_tag}_seed_{config.seed}"
    if config.num_terminals == 4:
        return f"four_terminal_upper_bound_search_{hull_tag}_sep_{separation_tag}_seed_{config.seed}"
    return f"{config.num_terminals}_terminal_search"


def _summary_payload(outcomes: Iterable[SearchOutcome]) -> dict:
    outcome_list = list(outcomes)
    sweep_summaries = summarize_separation_sweep(outcome_list)
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
                "hull_size": outcome.best.metadata.get("hull_size"),
                "require_full_hull": outcome.config.require_full_hull,
            }
            for outcome in outcome_list
        ],
        "separation_sweep": [
            {
                "num_terminals": summary.num_terminals,
                "min_terminal_separation": summary.min_terminal_separation,
                "runs": summary.runs,
                "best_ratio": summary.best_ratio,
                "mean_best_ratio": summary.mean_best_ratio,
                "worst_best_ratio": summary.worst_best_ratio,
                "best_gap": summary.best_gap,
                "mean_gap": summary.mean_gap,
                "mean_minimum_pairwise_distance": summary.mean_minimum_pairwise_distance,
                "best_candidate_hull_sizes": list(summary.best_candidate_hull_sizes),
                "require_full_hull": all(
                    outcome.config.require_full_hull
                    for outcome in outcome_list
                    if outcome.config.num_terminals == summary.num_terminals
                    and outcome.config.min_terminal_separation == summary.min_terminal_separation
                ),
            }
            for summary in sweep_summaries
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


def _write_sweep_report(summary: dict, output_dir: Path) -> None:
    sweep_rows = summary["separation_sweep"]
    (output_dir / "separation_sweep.json").write_text(
        json.dumps(sweep_rows, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Steiner Ratio Separation Sweep",
        "",
        f"- Conjectured lower bound: `{summary['conjectured_lower_bound']:.12f}`",
        "",
    ]
    for row in sweep_rows:
        lines.extend(
            [
                f"## {row['num_terminals']}-terminal, min separation {row['min_terminal_separation']:.6f}",
                "",
                f"- Runs: `{row['runs']}`",
                f"- Best ratio: `{row['best_ratio']:.12f}`",
                f"- Mean best ratio: `{row['mean_best_ratio']:.12f}`",
                f"- Worst best ratio: `{row['worst_best_ratio']:.12f}`",
                f"- Best gap: `{row['best_gap']:.12f}`",
                f"- Mean gap: `{row['mean_gap']:.12f}`",
                f"- Mean minimum pairwise distance: `{row['mean_minimum_pairwise_distance']:.12f}`",
                f"- Hull sizes of best runs: `{row['best_candidate_hull_sizes']}`",
                f"- Require full hull: `{row['require_full_hull']}`",
                "",
            ]
        )

    (output_dir / "separation_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument(
        "--min-separation-grid",
        type=float,
        nargs="+",
        default=None,
        help="Run a sweep over multiple minimum-separation values.",
    )
    parser.add_argument(
        "--require-full-hull",
        action="store_true",
        help="Require every accepted point set to have all terminals on the convex hull.",
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
    _write_sweep_report(summary, output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(
        "Interpretation: 4-terminal results are upper bounds from restricted full-topology search, "
        "so they are evidence for candidate structures, not proofs of a counterexample."
    )
    print(f"Saved reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
