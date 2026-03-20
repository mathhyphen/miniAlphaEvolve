from __future__ import annotations

import json
import math

from alphaevolve.research import (
    STEINER_RATIO_CONJECTURE,
    SearchConfig,
    evaluate_four_terminal_upper_bound,
    evaluate_three_terminal_ratio,
    evolutionary_search,
    run_min_separation_sweep,
    summarize_separation_sweep,
    write_min_separation_sweep_report,
)
from run.steiner_ratio_search import _summary_payload, _write_sweep_report


def test_three_terminal_equilateral_hits_conjectured_ratio() -> None:
    record = evaluate_three_terminal_ratio(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, math.sqrt(3.0) / 2.0),
        )
    )

    assert math.isclose(record.ratio, STEINER_RATIO_CONJECTURE, rel_tol=1e-6, abs_tol=1e-6)
    assert record.topology == "triangle_geometric_median"


def test_four_terminal_square_matches_known_full_steiner_ratio() -> None:
    record = evaluate_four_terminal_upper_bound(
        (
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        )
    )

    expected_ratio = (1.0 + math.sqrt(3.0)) / 3.0
    assert math.isclose(record.ratio, expected_ratio, rel_tol=1e-4, abs_tol=1e-4)
    assert record.ratio > STEINER_RATIO_CONJECTURE
    assert record.topology != "mst"


def test_evolutionary_search_tracks_history_and_candidates() -> None:
    outcome = evolutionary_search(
        SearchConfig(
            num_terminals=3,
            population_size=12,
            generations=5,
            elite_count=3,
            local_trials=2,
            seed=7,
        ),
        label="triangle_smoke",
    )

    assert outcome.label == "triangle_smoke"
    assert len(outcome.generation_history) == 5
    assert len(outcome.median_history) == 5
    assert outcome.top_candidates
    assert outcome.best.ratio <= STEINER_RATIO_CONJECTURE + 1e-6
    assert outcome.best.points == outcome.top_candidates[0].points


def test_constrained_search_respects_minimum_separation() -> None:
    outcome = evolutionary_search(
        SearchConfig(
            num_terminals=4,
            population_size=24,
            generations=12,
            elite_count=4,
            local_trials=2,
            min_terminal_separation=0.1,
            seed=5,
        ),
        label="four_terminal_constrained",
    )

    assert outcome.best.metadata["minimum_pairwise_distance"] >= 0.1 - 1e-9
    assert outcome.best.ratio > STEINER_RATIO_CONJECTURE


def test_evolutionary_search_is_deterministic_for_same_seed() -> None:
    config = SearchConfig(
        num_terminals=4,
        population_size=20,
        generations=6,
        elite_count=4,
        local_trials=2,
        seed=11,
    )

    first = evolutionary_search(config, label="first")
    second = evolutionary_search(config, label="second")

    assert first.best.points == second.best.points
    assert first.best.ratio == second.best.ratio
    assert first.generation_history == second.generation_history


def test_write_outcome_report_emits_expected_files(tmp_path) -> None:
    outcome = evolutionary_search(
        SearchConfig(
            num_terminals=4,
            population_size=16,
            generations=4,
            elite_count=4,
            local_trials=2,
            seed=2,
        ),
        label="report_case",
    )

    from alphaevolve.research import write_outcome_report

    write_outcome_report([outcome], tmp_path)

    expected_files = {
        "best_candidates.json",
        "config.json",
        "counterexample_candidates.json",
        "history.json",
        "report.md",
        "results.json",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})

    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert results["outcomes"][0]["label"] == "report_case"


def test_search_respects_minimum_terminal_separation() -> None:
    outcome = evolutionary_search(
        SearchConfig(
            num_terminals=4,
            population_size=18,
            generations=5,
            elite_count=4,
            local_trials=2,
            min_terminal_separation=0.08,
            seed=5,
        ),
        label="separated",
    )

    assert outcome.best.metadata["minimum_pairwise_distance"] >= 0.08 - 1e-9


def test_summarize_separation_sweep_groups_by_threshold() -> None:
    outcomes = [
        evolutionary_search(
            SearchConfig(
                num_terminals=4,
                population_size=16,
                generations=4,
                elite_count=4,
                local_trials=2,
                min_terminal_separation=0.0,
                seed=0,
            ),
            label="free",
        ),
        evolutionary_search(
            SearchConfig(
                num_terminals=4,
                population_size=16,
                generations=4,
                elite_count=4,
                local_trials=2,
                min_terminal_separation=0.1,
                seed=1,
            ),
            label="constrained",
        ),
    ]

    summary = summarize_separation_sweep(outcomes)

    assert [row.min_terminal_separation for row in summary] == [0.0, 0.1]
    assert summary[1].best_ratio >= summary[0].best_ratio
    assert summary[1].mean_minimum_pairwise_distance >= 0.1 - 1e-9


def test_cli_summary_and_sweep_report_include_grid_rows(tmp_path) -> None:
    outcomes = [
        evolutionary_search(
            SearchConfig(
                num_terminals=4,
                population_size=14,
                generations=4,
                elite_count=4,
                local_trials=2,
                min_terminal_separation=0.0,
                seed=2,
            ),
            label="sep0",
        ),
        evolutionary_search(
            SearchConfig(
                num_terminals=4,
                population_size=14,
                generations=4,
                elite_count=4,
                local_trials=2,
                min_terminal_separation=0.05,
                seed=3,
            ),
            label="sep005",
        ),
    ]

    summary = _summary_payload(outcomes)
    _write_sweep_report(summary, tmp_path)

    rows = json.loads((tmp_path / "separation_sweep.json").read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert rows[0]["min_terminal_separation"] == 0.0
    assert rows[1]["min_terminal_separation"] == 0.05
    assert "4-terminal, min separation 0.050000" in (tmp_path / "separation_sweep.md").read_text(
        encoding="utf-8"
    )


def test_run_min_separation_sweep_returns_one_row_per_bucket() -> None:
    sweep = run_min_separation_sweep(
        separations=[0.0, 0.05, 0.1],
        runs_per_separation=2,
        num_terminals=4,
        population_size=16,
        generations=4,
        elite_count=4,
        mutation_sigma=0.16,
        mutation_decay=0.994,
        crossover_rate=0.35,
        random_injection_rate=0.15,
        local_trials=2,
        seed=3,
    )

    assert [row.min_terminal_separation for row in sweep.rows] == [0.0, 0.05, 0.1]
    assert len(sweep.run_outcomes) == 6
    assert sweep.rows[1].mean_minimum_pairwise_distance >= 0.05 - 1e-9


def test_write_min_separation_sweep_report_emits_expected_files(tmp_path) -> None:
    sweep = run_min_separation_sweep(
        separations=[0.0, 0.05],
        runs_per_separation=1,
        num_terminals=4,
        population_size=14,
        generations=3,
        elite_count=4,
        mutation_sigma=0.16,
        mutation_decay=0.994,
        crossover_rate=0.35,
        random_injection_rate=0.15,
        local_trials=2,
        seed=1,
    )

    write_min_separation_sweep_report(sweep, tmp_path)

    assert (tmp_path / "sweep_summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "runs" / "results.json").exists()
