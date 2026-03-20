"""Evolutionary search utilities for Steiner-ratio experiments."""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

STEINER_RATIO_CONJECTURE = math.sqrt(3.0) / 2.0


@dataclass(frozen=True)
class CandidateRecord:
    """A evaluated point-set candidate."""

    points: Tuple[Point, ...]
    ratio: float
    steiner_length: float
    mst_length: float
    topology: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for evolutionary point-set search."""

    num_terminals: int
    population_size: int = 96
    generations: int = 240
    elite_count: int = 12
    mutation_sigma: float = 0.16
    mutation_decay: float = 0.994
    crossover_rate: float = 0.35
    random_injection_rate: float = 0.15
    local_trials: int = 3
    min_terminal_separation: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class SearchOutcome:
    """Summary of one search run."""

    label: str
    config: SearchConfig
    best: CandidateRecord
    top_candidates: List[CandidateRecord]
    generation_history: List[float]
    median_history: List[float]


@dataclass(frozen=True)
class SeparationSweepRow:
    """Aggregate statistics for one min-separation bucket."""

    min_terminal_separation: float
    best_ratio: float
    median_best_ratio: float
    mean_gap_to_conjecture: float
    mean_minimum_pairwise_distance: float
    boundary_hugging_fraction: float
    hull3_fraction: float
    best_label: str
    best_topology: str


@dataclass(frozen=True)
class SeparationSweepOutcome:
    """Summary of a full min-separation sweep."""

    rows: List[SeparationSweepRow]
    run_outcomes: List[SearchOutcome]


@dataclass(frozen=True)
class SeparationSweepSummary:
    """Aggregate search outcomes by minimum terminal separation."""

    num_terminals: int
    min_terminal_separation: float
    runs: int
    best_ratio: float
    mean_best_ratio: float
    worst_best_ratio: float
    best_gap: float
    mean_gap: float
    mean_minimum_pairwise_distance: float
    best_candidate_hull_sizes: Tuple[int, ...]


def _distance(left: Point, right: Point) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _centroid(points: Sequence[Point]) -> Point:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _translate(points: Sequence[Point], dx: float, dy: float) -> Tuple[Point, ...]:
    return tuple((point[0] + dx, point[1] + dy) for point in points)


def _diameter(points: Sequence[Point]) -> float:
    return max(
        _distance(points[left], points[right])
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )


def _normalize_points(points: Sequence[Point]) -> Tuple[Point, ...]:
    """Remove trivial translation/scale freedom."""
    center = _centroid(points)
    translated = _translate(points, -center[0], -center[1])
    diameter = _diameter(translated)
    if diameter <= 1e-9:
        return tuple(sorted((0.0, 0.0) for _ in translated))
    scaled = tuple((point[0] / diameter, point[1] / diameter) for point in translated)
    return tuple(sorted(scaled))


def _prim_mst_length(points: Sequence[Point]) -> float:
    """Compute the Euclidean MST length."""
    if len(points) < 2:
        return 0.0

    visited = {0}
    total = 0.0
    while len(visited) < len(points):
        best_distance = float("inf")
        best_index = None
        for source in visited:
            for target in range(len(points)):
                if target in visited:
                    continue
                distance = _distance(points[source], points[target])
                if distance < best_distance:
                    best_distance = distance
                    best_index = target
        assert best_index is not None
        visited.add(best_index)
        total += best_distance
    return total


def _geometric_median(
    points: Sequence[Point],
    initial_guess: Optional[Point] = None,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> Point:
    """Approximate the geometric median using Weiszfeld iterations."""
    if not points:
        raise ValueError("geometric median requires at least one point")

    current = initial_guess or _centroid(points)
    for _ in range(max_iterations):
        numerator_x = 0.0
        numerator_y = 0.0
        denominator = 0.0

        for point in points:
            distance = _distance(current, point)
            if distance <= tolerance:
                return point
            weight = 1.0 / distance
            numerator_x += point[0] * weight
            numerator_y += point[1] * weight
            denominator += weight

        next_point = (numerator_x / denominator, numerator_y / denominator)
        if _distance(current, next_point) <= tolerance:
            return next_point
        current = next_point

    return current


def _pairings_for_four_terminals() -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    return [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]


def _optimize_full_four_terminal_topology(
    points: Sequence[Point],
    pairing: Tuple[Tuple[int, int], Tuple[int, int]],
    max_iterations: int = 80,
    tolerance: float = 1e-10,
) -> Tuple[float, Tuple[Point, Point]]:
    """Optimize the two Steiner points for a fixed full topology."""
    pair_a, pair_b = pairing
    point_a = points[pair_a[0]]
    point_b = points[pair_a[1]]
    point_c = points[pair_b[0]]
    point_d = points[pair_b[1]]

    steiner_a = ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)
    steiner_b = ((point_c[0] + point_d[0]) / 2.0, (point_c[1] + point_d[1]) / 2.0)

    for _ in range(max_iterations):
        next_steiner_a = _geometric_median((point_a, point_b, steiner_b), initial_guess=steiner_a)
        next_steiner_b = _geometric_median((point_c, point_d, next_steiner_a), initial_guess=steiner_b)
        step_size = max(
            _distance(steiner_a, next_steiner_a),
            _distance(steiner_b, next_steiner_b),
        )
        steiner_a, steiner_b = next_steiner_a, next_steiner_b
        if step_size <= tolerance:
            break

    length = (
        _distance(steiner_a, point_a)
        + _distance(steiner_a, point_b)
        + _distance(steiner_a, steiner_b)
        + _distance(steiner_b, point_c)
        + _distance(steiner_b, point_d)
    )
    return length, (steiner_a, steiner_b)


def evaluate_three_terminal_ratio(points: Sequence[Point]) -> CandidateRecord:
    """Exact Steiner ratio for a 3-terminal set."""
    normalized = _normalize_points(points)
    mst_length = _prim_mst_length(normalized)
    steiner_point = _geometric_median(normalized)
    steiner_length = sum(_distance(point, steiner_point) for point in normalized)
    return CandidateRecord(
        points=normalized,
        ratio=steiner_length / mst_length if mst_length > 0 else 1.0,
        steiner_length=steiner_length,
        mst_length=mst_length,
        topology="triangle_geometric_median",
        metadata={"steiner_point": steiner_point},
    )


def evaluate_four_terminal_upper_bound(points: Sequence[Point]) -> CandidateRecord:
    """Upper-bound the Steiner ratio for a 4-terminal point set."""
    normalized = _normalize_points(points)
    mst_length = _prim_mst_length(normalized)
    best_length = mst_length
    best_topology = "mst"
    best_steiner_points: Tuple[Point, Point] | Tuple[()] = ()

    for pairing in _pairings_for_four_terminals():
        length, steiner_points = _optimize_full_four_terminal_topology(normalized, pairing)
        if length < best_length:
            best_length = length
            best_topology = f"{pairing[0][0]}{pairing[0][1]}|{pairing[1][0]}{pairing[1][1]}"
            best_steiner_points = steiner_points

    return CandidateRecord(
        points=normalized,
        ratio=best_length / mst_length if mst_length > 0 else 1.0,
        steiner_length=best_length,
        mst_length=mst_length,
        topology=best_topology,
        metadata={"steiner_points": best_steiner_points},
    )


def _convex_hull_size(points: Sequence[Point]) -> int:
    """Compute hull size using monotonic chain."""
    sorted_points = sorted(set(points))
    if len(sorted_points) <= 1:
        return len(sorted_points)

    def cross(origin: Point, left: Point, right: Point) -> float:
        return (
            (left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0])
        )

    lower: List[Point] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Point] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return len(lower[:-1] + upper[:-1])


def _candidate_metadata(points: Sequence[Point]) -> Dict[str, Any]:
    distances = sorted(
        _distance(points[left], points[right])
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    aspect_ratio = width / height if height > 1e-9 else float("inf")
    return {
        "pairwise_distances": distances,
        "minimum_pairwise_distance": distances[0] if distances else 0.0,
        "hull_size": _convex_hull_size(points),
        "bounding_box_aspect_ratio": aspect_ratio,
    }


def _with_geometry_metadata(record: CandidateRecord) -> CandidateRecord:
    metadata = dict(record.metadata)
    metadata.update(_candidate_metadata(record.points))
    return CandidateRecord(
        points=record.points,
        ratio=record.ratio,
        steiner_length=record.steiner_length,
        mst_length=record.mst_length,
        topology=record.topology,
        metadata=metadata,
    )


def _minimum_pairwise_distance(points: Sequence[Point]) -> float:
    return min(
        _distance(points[left], points[right])
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )


def _meets_minimum_separation(points: Sequence[Point], min_terminal_separation: float) -> bool:
    if min_terminal_separation <= 0.0:
        return True
    return _minimum_pairwise_distance(points) + 1e-12 >= min_terminal_separation


def _random_point_set(
    num_terminals: int,
    rng: random.Random,
    min_terminal_separation: float = 0.0,
    max_attempts: int = 128,
) -> Tuple[Point, ...]:
    for _ in range(max_attempts):
        candidate = _normalize_points(
            tuple((rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)) for _ in range(num_terminals))
        )
        if _meets_minimum_separation(candidate, min_terminal_separation):
            return candidate
    raise RuntimeError("Failed to sample a point set satisfying the minimum separation constraint.")


def _structured_seed_population(num_terminals: int) -> Iterable[Tuple[Point, ...]]:
    if num_terminals == 3:
        yield _normalize_points(((0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2.0)))
        yield _normalize_points(((0.0, 0.0), (1.0, 0.0), (0.55, 0.75)))
        return

    if num_terminals == 4:
        yield _normalize_points(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
        yield _normalize_points(((-1.2, 0.0), (0.0, -0.6), (1.2, 0.0), (0.0, 0.6)))
        yield _normalize_points(((-1.0, 0.0), (1.0, 0.0), (0.1, 0.9), (0.2, -0.8)))
        return

    raise ValueError("This experiment currently supports only 3 or 4 terminals.")


def _mutate(
    points: Sequence[Point],
    rng: random.Random,
    sigma: float,
    min_terminal_separation: float = 0.0,
    max_attempts: int = 24,
) -> Tuple[Point, ...]:
    for _ in range(max_attempts):
        mutated = [
            (point[0] + rng.gauss(0.0, sigma), point[1] + rng.gauss(0.0, sigma))
            for point in points
        ]
        candidate = _normalize_points(mutated)
        if _meets_minimum_separation(candidate, min_terminal_separation):
            return candidate
    return _normalize_points(points)


def _crossover(
    left: Sequence[Point],
    right: Sequence[Point],
    rng: random.Random,
    sigma: float,
    min_terminal_separation: float = 0.0,
    max_attempts: int = 24,
) -> Tuple[Point, ...]:
    for _ in range(max_attempts):
        blended = []
        for point_left, point_right in zip(left, right):
            weight = rng.uniform(0.25, 0.75)
            blended.append(
                (
                    point_left[0] * weight + point_right[0] * (1.0 - weight) + rng.gauss(0.0, sigma * 0.35),
                    point_left[1] * weight + point_right[1] * (1.0 - weight) + rng.gauss(0.0, sigma * 0.35),
                )
            )
        candidate = _normalize_points(blended)
        if _meets_minimum_separation(candidate, min_terminal_separation):
            return candidate
    return _normalize_points(left)


def _evaluator_for_terminals(num_terminals: int) -> Callable[[Sequence[Point]], CandidateRecord]:
    if num_terminals == 3:
        return evaluate_three_terminal_ratio
    if num_terminals == 4:
        return evaluate_four_terminal_upper_bound
    raise ValueError("This experiment currently supports only 3 or 4 terminals.")


def _update_global_top_records(
    records: Sequence[CandidateRecord],
    registry: Dict[Tuple[Point, ...], CandidateRecord],
    limit: int = 10,
) -> List[CandidateRecord]:
    for record in records:
        existing = registry.get(record.points)
        if existing is None or record.ratio < existing.ratio:
            registry[record.points] = record
    return sorted(registry.values(), key=lambda record: record.ratio)[:limit]


def evolutionary_search(config: SearchConfig, label: str) -> SearchOutcome:
    """Run a point-set evolutionary search for low Steiner ratios."""
    rng = random.Random(config.seed)
    evaluator = _evaluator_for_terminals(config.num_terminals)

    population: List[Tuple[Point, ...]] = [
        candidate
        for candidate in _structured_seed_population(config.num_terminals)
        if _meets_minimum_separation(candidate, config.min_terminal_separation)
    ]
    while len(population) < config.population_size:
        population.append(
            _random_point_set(
                config.num_terminals,
                rng,
                min_terminal_separation=config.min_terminal_separation,
            )
        )

    sigma = config.mutation_sigma
    history: List[float] = []
    median_history: List[float] = []
    best_record: Optional[CandidateRecord] = None
    top_records: List[CandidateRecord] = []
    registry: Dict[Tuple[Point, ...], CandidateRecord] = {}

    for _ in range(config.generations):
        evaluated = [_with_geometry_metadata(evaluator(candidate)) for candidate in population]
        evaluated.sort(key=lambda record: record.ratio)
        history.append(evaluated[0].ratio)
        median_history.append(evaluated[len(evaluated) // 2].ratio)

        if best_record is None or evaluated[0].ratio < best_record.ratio:
            best_record = evaluated[0]

        top_records = _update_global_top_records(evaluated[: min(20, len(evaluated))], registry)
        elite_points = [record.points for record in evaluated[: config.elite_count]]

        next_population = list(elite_points)
        while len(next_population) < config.population_size:
            roll = rng.random()
            if roll < config.random_injection_rate:
                next_population.append(
                    _random_point_set(
                        config.num_terminals,
                        rng,
                        min_terminal_separation=config.min_terminal_separation,
                    )
                )
            elif roll < config.random_injection_rate + config.crossover_rate and len(elite_points) >= 2:
                parent_left, parent_right = rng.sample(elite_points, 2)
                next_population.append(
                    _crossover(
                        parent_left,
                        parent_right,
                        rng,
                        sigma,
                        min_terminal_separation=config.min_terminal_separation,
                    )
                )
            else:
                parent = rng.choice(elite_points)
                child = _mutate(
                    parent,
                    rng,
                    sigma,
                    min_terminal_separation=config.min_terminal_separation,
                )
                child_ratio = evaluator(child).ratio
                for _ in range(config.local_trials - 1):
                    candidate = _mutate(
                        parent,
                        rng,
                        sigma * 0.6,
                        min_terminal_separation=config.min_terminal_separation,
                    )
                    candidate_ratio = evaluator(candidate).ratio
                    if candidate_ratio < child_ratio:
                        child = candidate
                        child_ratio = candidate_ratio
                next_population.append(child)

        sigma *= config.mutation_decay
        population = next_population

    assert best_record is not None
    return SearchOutcome(
        label=label,
        config=config,
        best=best_record,
        top_candidates=top_records,
        generation_history=history,
        median_history=median_history,
    )


def outcome_to_jsonable(outcome: SearchOutcome) -> Dict[str, Any]:
    """Convert an outcome to a JSON-safe dictionary."""
    return asdict(outcome)


def summarize_separation_sweep(outcomes: Sequence[SearchOutcome]) -> List[SeparationSweepSummary]:
    """Aggregate outcomes by terminal count and minimum separation."""
    grouped: Dict[Tuple[int, float], List[SearchOutcome]] = {}
    for outcome in outcomes:
        key = (outcome.config.num_terminals, outcome.config.min_terminal_separation)
        grouped.setdefault(key, []).append(outcome)

    summaries: List[SeparationSweepSummary] = []
    for (num_terminals, min_terminal_separation), bucket in sorted(grouped.items()):
        best_ratios = [outcome.best.ratio for outcome in bucket]
        min_distances = [
            float(outcome.best.metadata.get("minimum_pairwise_distance", 0.0))
            for outcome in bucket
        ]
        hull_sizes = tuple(
            int(outcome.best.metadata.get("hull_size", 0))
            for outcome in sorted(bucket, key=lambda item: item.best.ratio)
        )
        summaries.append(
            SeparationSweepSummary(
                num_terminals=num_terminals,
                min_terminal_separation=min_terminal_separation,
                runs=len(bucket),
                best_ratio=min(best_ratios),
                mean_best_ratio=sum(best_ratios) / len(best_ratios),
                worst_best_ratio=max(best_ratios),
                best_gap=min(best_ratios) - STEINER_RATIO_CONJECTURE,
                mean_gap=(sum(best_ratios) / len(best_ratios)) - STEINER_RATIO_CONJECTURE,
                mean_minimum_pairwise_distance=sum(min_distances) / len(min_distances),
                best_candidate_hull_sizes=hull_sizes,
            )
        )

    return summaries


def write_outcome_report(outcomes: Sequence[SearchOutcome], output_dir: Path) -> None:
    """Persist JSON and Markdown summaries for search outcomes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        outcome.label: {
            "num_terminals": outcome.config.num_terminals,
            "population_size": outcome.config.population_size,
            "generations": outcome.config.generations,
            "elite_count": outcome.config.elite_count,
            "mutation_sigma": outcome.config.mutation_sigma,
            "mutation_decay": outcome.config.mutation_decay,
            "crossover_rate": outcome.config.crossover_rate,
            "random_injection_rate": outcome.config.random_injection_rate,
            "local_trials": outcome.config.local_trials,
            "min_terminal_separation": outcome.config.min_terminal_separation,
            "seed": outcome.config.seed,
        }
        for outcome in outcomes
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    payload = {"outcomes": [outcome_to_jsonable(outcome) for outcome in outcomes]}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(
        json.dumps(
            {
                outcome.label: {
                    "best_ratio": outcome.generation_history,
                    "median_ratio": outcome.median_history,
                }
                for outcome in outcomes
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "best_candidates.json").write_text(
        json.dumps(
            {
                outcome.label: [asdict(candidate) for candidate in outcome.top_candidates]
                for outcome in outcomes
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    counterexample_candidates = {
        outcome.label: [
            asdict(candidate)
            for candidate in outcome.top_candidates
            if candidate.ratio < STEINER_RATIO_CONJECTURE
        ]
        for outcome in outcomes
    }
    (output_dir / "counterexample_candidates.json").write_text(
        json.dumps(counterexample_candidates, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Steiner Ratio Search Report",
        "",
        f"- Conjectured lower bound: `{STEINER_RATIO_CONJECTURE:.12f}`",
        "",
    ]
    for outcome in outcomes:
        gap = outcome.best.ratio - STEINER_RATIO_CONJECTURE
        lines.extend(
            [
                f"## {outcome.label}",
                "",
                f"- Best ratio found: `{outcome.best.ratio:.12f}`",
                f"- Gap above conjectured lower bound: `{gap:.12f}`",
                f"- Topology: `{outcome.best.topology}`",
                f"- Points: `{list(outcome.best.points)}`",
                f"- Minimum terminal separation: `{outcome.config.min_terminal_separation:.6f}`",
                f"- Final median ratio: `{outcome.median_history[-1]:.12f}`",
                f"- Candidate count stored: `{len(outcome.top_candidates)}`",
                "",
                "### Top Candidates",
                "",
            ]
        )
        for index, candidate in enumerate(outcome.top_candidates[:5], start=1):
            candidate_gap = candidate.ratio - STEINER_RATIO_CONJECTURE
            lines.append(
                f"- #{index}: ratio `{candidate.ratio:.12f}` (gap `{candidate_gap:.12f}`), "
                f"topology `{candidate.topology}`, hull `{candidate.metadata['hull_size']}`, "
                f"aspect `{candidate.metadata['bounding_box_aspect_ratio']:.6f}`, "
                f"min-pair-distance `{candidate.metadata['minimum_pairwise_distance']:.6f}`"
            )
        lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_min_separation_sweep(
    *,
    separations: Sequence[float],
    runs_per_separation: int,
    num_terminals: int,
    population_size: int,
    generations: int,
    elite_count: int,
    mutation_sigma: float,
    mutation_decay: float,
    crossover_rate: float,
    random_injection_rate: float,
    local_trials: int,
    seed: int = 0,
) -> SeparationSweepOutcome:
    """Run the same search budget across a grid of minimum-separation constraints."""
    if num_terminals != 4:
        raise ValueError("Min-separation sweep is currently intended for 4-terminal searches.")

    rows: List[SeparationSweepRow] = []
    run_outcomes: List[SearchOutcome] = []

    for separation_index, separation in enumerate(separations):
        bucket_outcomes: List[SearchOutcome] = []
        for run_index in range(runs_per_separation):
            run_seed = seed + separation_index * runs_per_separation + run_index
            config = SearchConfig(
                num_terminals=num_terminals,
                population_size=population_size,
                generations=generations,
                elite_count=elite_count,
                mutation_sigma=mutation_sigma,
                mutation_decay=mutation_decay,
                crossover_rate=crossover_rate,
                random_injection_rate=random_injection_rate,
                local_trials=local_trials,
                min_terminal_separation=separation,
                seed=run_seed,
            )
            label = f"four_terminal_sep_{separation:.3f}_seed_{run_seed}"
            outcome = evolutionary_search(config, label=label)
            bucket_outcomes.append(outcome)
            run_outcomes.append(outcome)

        best_outcome = min(bucket_outcomes, key=lambda outcome: outcome.best.ratio)
        best_ratios = [outcome.best.ratio for outcome in bucket_outcomes]
        gaps = [ratio - STEINER_RATIO_CONJECTURE for ratio in best_ratios]
        min_pairwise = [
            float(outcome.best.metadata["minimum_pairwise_distance"])
            for outcome in bucket_outcomes
        ]
        hull3_fraction = sum(
            1 for outcome in bucket_outcomes if int(outcome.best.metadata["hull_size"]) == 3
        ) / len(bucket_outcomes)
        boundary_hugging_fraction = sum(
            1
            for distance in min_pairwise
            if distance <= separation + max(0.01, separation * 0.15)
        ) / len(bucket_outcomes)

        rows.append(
            SeparationSweepRow(
                min_terminal_separation=separation,
                best_ratio=min(best_ratios),
                median_best_ratio=statistics.median(best_ratios),
                mean_gap_to_conjecture=statistics.fmean(gaps),
                mean_minimum_pairwise_distance=statistics.fmean(min_pairwise),
                boundary_hugging_fraction=boundary_hugging_fraction,
                hull3_fraction=hull3_fraction,
                best_label=best_outcome.label,
                best_topology=best_outcome.best.topology,
            )
        )

    return SeparationSweepOutcome(rows=rows, run_outcomes=run_outcomes)


def write_min_separation_sweep_report(
    sweep: SeparationSweepOutcome,
    output_dir: Path,
) -> None:
    """Persist JSON and Markdown summaries for a min-separation sweep."""
    output_dir.mkdir(parents=True, exist_ok=True)

    write_outcome_report(sweep.run_outcomes, output_dir / "runs")
    (output_dir / "sweep_summary.json").write_text(
        json.dumps([asdict(row) for row in sweep.rows], indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Steiner Ratio Min-Separation Sweep",
        "",
        f"- Conjectured lower bound: `{STEINER_RATIO_CONJECTURE:.12f}`",
        "",
        "| min separation | best ratio | median best ratio | mean gap | mean min pair | boundary-hugging frac | hull=3 frac | best topology |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in sweep.rows:
        lines.append(
            "| "
            f"{row.min_terminal_separation:.3f} | "
            f"{row.best_ratio:.12f} | "
            f"{row.median_best_ratio:.12f} | "
            f"{row.mean_gap_to_conjecture:.12f} | "
            f"{row.mean_minimum_pairwise_distance:.6f} | "
            f"{row.boundary_hugging_fraction:.3f} | "
            f"{row.hull3_fraction:.3f} | "
            f"{row.best_topology} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Hints",
            "",
            "- High boundary-hugging fraction means the search is pushing terminals against the minimum-separation floor.",
            "- High hull=3 fraction suggests many best candidates still place one terminal effectively inside a triangle-like hull.",
            "- If best ratios rise as separation increases, the low-ratio regime is likely driven by terminal clustering rather than a stable 4-terminal extremizer.",
            "",
        ]
    )

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
