"""Population-Wide Archive (PWA) for AlphaEvolve.

The PWA is a quality-diversity archive that differs from MAP-Elites:
- Fitness = latency improvement (not correctness score)
- Diversity = algorithmic approaches (not feature dimensions)
- Stores code snapshots at different performance levels
- Enables discovering novel algorithms

For Steiner Tree research:
- Archive solutions by latency improvement over MST baseline
- Track different algorithmic strategies (Fermat, centroid, geometric, etc.)
- Enable counterexample discovery (cases where algorithms fail)
"""

import ast
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AlgorithmicStrategy(Enum):
    """Enumeration of algorithmic strategy types for diversity tracking.

    Unlike MAP-Elites which uses feature dimensions, PWA tracks solutions
    by their underlying algorithmic approach.
    """
    UNKNOWN = "unknown"
    FERMAT_BASED = "fermat"
    CENTROID_BASED = "centroid"
    GEOMETRIC = "geometric"
    OPTIMIZATION = "optimization"
    HYBRID = "hybrid"
    HEURISTIC = "heuristic"
    EXACT = "exact"


@dataclass(frozen=True)
class LatencyImprovement:
    """Represents latency improvement of a solution.

    Args:
        improvement_ratio: Ratio of improvement (0.0 to 1.0)
            0.0 = no improvement (same as baseline)
            1.0 = theoretical maximum improvement
        absolute_improvement: Absolute latency improvement in ms
        baseline_latency: Baseline latency in ms
        achieved_latency: Achieved latency in ms
    """
    improvement_ratio: float
    absolute_improvement: float
    baseline_latency: float
    achieved_latency: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.improvement_ratio <= 1.0:
            object.__setattr__(
                self, 'improvement_ratio',
                max(-1.0, min(1.0, self.improvement_ratio))
            )


@dataclass
class CodeSnapshot:
    """A code snapshot stored in the archive.

    Represents a solution at a specific performance level with its
    algorithmic characteristics.

    Args:
        id: Unique identifier for this snapshot
        code: The actual code string
        strategy: The algorithmic strategy used
        latency: Latency metric when evaluated
        improvement: Latency improvement metrics
        generation: Generation when created
        parent_ids: Parent snapshot IDs (for lineage tracking)
        metadata: Additional metadata
        created_at: Timestamp
    """
    id: str
    code: str
    strategy: AlgorithmicStrategy
    latency: float
    improvement: LatencyImprovement
    generation: int
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        code: str,
        strategy: AlgorithmicStrategy,
        latency: float,
        improvement: LatencyImprovement,
        generation: int,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CodeSnapshot":
        """Factory method to create a new snapshot.

        Args:
            code: Source code
            strategy: Detected algorithmic strategy
            latency: Achieved latency in ms
            improvement: Improvement metrics
            generation: Evolution generation
            parent_ids: Parent snapshot IDs
            metadata: Additional metadata

        Returns:
            New CodeSnapshot instance
        """
        content = f"{code}{time.time()}"
        snapshot_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(
            id=snapshot_id,
            code=code,
            strategy=strategy,
            latency=latency,
            improvement=improvement,
            generation=generation,
            parent_ids=parent_ids or [],
            metadata=metadata or {},
        )


@dataclass
class PerformanceTier:
    """A performance tier in the archive.

    Each tier represents a range of latency improvement levels.
    Solutions are stored at different tiers for archival purposes.

    Args:
        name: Tier name (e.g., "excellent", "good", "moderate")
        min_improvement: Minimum improvement ratio (inclusive)
        max_improvement: Maximum improvement ratio (exclusive)
    """
    name: str
    min_improvement: float
    max_improvement: float

    def contains(self, improvement_ratio: float) -> bool:
        """Check if improvement falls within this tier."""
        return self.min_improvement <= improvement_ratio < self.max_improvement

    @classmethod
    def get_default_tiers(cls) -> List["PerformanceTier"]:
        """Get default performance tiers for latency improvement.

        Returns:
            List of performance tiers from worst to best
        """
        return [
            cls("counterexample", -1.0, 0.01),     # No/worse improvement
            cls("baseline", 0.01, 0.1),            # Below MST improvement
            cls("moderate", 0.1, 0.3),             # Moderate improvement
            cls("good", 0.3, 0.6),                 # Good improvement
            cls("excellent", 0.6, 0.85),           # Near optimal
            cls("optimal", 0.85, 1.0),             # Theoretical optimal
        ]


@dataclass
class StrategyCell:
    """A cell in the PWA archive representing a strategy-improvemen tuple.

    Unlike MAP-Elites cells (feature-based), PWA cells are defined by:
    - Algorithmic strategy type
    - Performance tier

    Args:
        strategy: The algorithmic strategy
        tier: The performance tier
        snapshot: The best snapshot in this cell
        snapshot_ids: All snapshot IDs in this cell
        added_count: Number of snapshots added
        replaced_count: Number of replacements
    """
    strategy: AlgorithmicStrategy
    tier: PerformanceTier
    snapshot: Optional[CodeSnapshot] = None
    snapshot_ids: List[str] = field(default_factory=list)
    added_count: int = 0
    replaced_count: int = 0


@dataclass
class PWArchiveConfig:
    """Configuration for Population-Wide Archive.

    Args:
        max_snapshots_per_cell: Maximum snapshots per strategy-tier cell
        max_total_snapshots: Maximum total snapshots in archive
        elitism: If True, only replace on strictly better improvement
        enable_counterexamples: If True, store counterexamples (solutions that fail)
        performance_tiers: Custom performance tiers
        baseline_latency: Reference baseline latency for improvement calculation
    """
    max_snapshots_per_cell: int = 10
    max_total_snapshots: Optional[int] = 1000
    elitism: bool = True
    enable_counterexamples: bool = True
    performance_tiers: List[PerformanceTier] = field(
        default_factory=PerformanceTier.get_default_tiers
    )
    baseline_latency: float = 100.0  # ms

    def __post_init__(self) -> None:
        if self.max_snapshots_per_cell < 1:
            raise ValueError("max_snapshots_per_cell must be at least 1")


@dataclass
class PWAStats:
    """Statistics about Population-Wide Archive state."""
    total_snapshots: int
    occupied_cells: int
    total_cells: int
    fill_rate: float
    best_improvement: float
    strategies_used: Dict[str, int]
    tier_distribution: Dict[str, int]
    counterexample_count: int
    total_added: int
    total_rejected: int


class StrategyDetector:
    """Detects algorithmic strategies from code.

    Uses AST analysis and keyword matching to identify the underlying
    algorithmic approach used in a solution.
    """

    def __init__(self) -> None:
        """Initialize strategy detector."""
        self._fermat_keywords = [
            'fermat', 'torricelli', '120', 'angle', 'equilateral',
            'triangular', 'three_point'
        ]
        self._centroid_keywords = [
            'centroid', 'center', 'average', 'mean', 'barycenter'
        ]
        self._geometric_keywords = [
            'voronoi', 'delaunay', 'triangulation', 'convex', 'circumcenter'
        ]
        self._optimization_keywords = [
            'gradient', 'iterative', 'optimize', 'minimize', 'maximize',
            'convex', 'concave', ' descent'
        ]
        self._hybrid_keywords = [
            'combine', 'hybrid', 'ensemble', 'multiple', 'two_stage'
        ]
        self._heuristic_keywords = [
            'heuristic', 'approximate', 'greedy', 'local_search'
        ]

    def detect(self, code: str) -> AlgorithmicStrategy:
        """Detect algorithmic strategy from code.

        Args:
            code: Source code string

        Returns:
            Detected AlgorithmicStrategy
        """
        code_lower = code.lower()

        scores: Dict[AlgorithmicStrategy, int] = {
            AlgorithmicStrategy.FERMAT_BASED: 0,
            AlgorithmicStrategy.CENTROID_BASED: 0,
            AlgorithmicStrategy.GEOMETRIC: 0,
            AlgorithmicStrategy.OPTIMIZATION: 0,
            AlgorithmicStrategy.HYBRID: 0,
            AlgorithmicStrategy.HEURISTIC: 0,
            AlgorithmicStrategy.EXACT: 0,
        }

        # Count keyword matches
        scores[AlgorithmicStrategy.FERMAT_BASED] = sum(
            1 for kw in self._fermat_keywords if kw in code_lower
        )
        scores[AlgorithmicStrategy.CENTROID_BASED] = sum(
            1 for kw in self._centroid_keywords if kw in code_lower
        )
        scores[AlgorithmicStrategy.GEOMETRIC] = sum(
            1 for kw in self._geometric_keywords if kw in code_lower
        )
        scores[AlgorithmicStrategy.OPTIMIZATION] = sum(
            1 for kw in self._optimization_keywords if kw in code_lower
        )
        scores[AlgorithmicStrategy.HYBRID] = sum(
            1 for kw in self._hybrid_keywords if kw in code_lower
        )
        scores[AlgorithmicStrategy.HEURISTIC] = sum(
            1 for kw in self._heuristic_keywords if kw in code_lower
        )

        # AST-based detection
        try:
            tree = ast.parse(code)
            self._detect_from_ast(tree, scores)
        except SyntaxError:
            pass

        # Return best matching strategy
        if max(scores.values()) == 0:
            return AlgorithmicStrategy.UNKNOWN

        return max(scores.keys(), key=lambda s: scores[s])

    def _detect_from_ast(
        self,
        tree: ast.AST,
        scores: Dict[AlgorithmicStrategy, int]
    ) -> None:
        """Detect strategy from AST structure.

        Args:
            tree: Parsed AST
            scores: Score dictionary to update
        """
        # Check for recursion (often used in exact/optimization approaches)
        func_defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for func_name in func_defs:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == func_name:
                        scores[AlgorithmicStrategy.EXACT] += 1
                        break

        # Check for loops (heuristic/iterative approaches)
        loop_count = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, (ast.For, ast.While))
        )
        if loop_count > 3:
            scores[AlgorithmicStrategy.HEURISTIC] += loop_count - 3


class LatencyCalculator:
    """Calculates latency improvement metrics.

    Computes improvement ratios and absolute improvements for solutions.
    """

    def __init__(self, baseline_latency: float) -> None:
        """Initialize latency calculator.

        Args:
            baseline_latency: Reference baseline latency in ms
        """
        self._baseline = baseline_latency

    def calculate(
        self,
        achieved_latency: float,
        baseline_latency: Optional[float] = None
    ) -> LatencyImprovement:
        """Calculate latency improvement.

        Args:
            achieved_latency: Achieved latency in ms
            baseline_latency: Override baseline (optional)

        Returns:
            LatencyImprovement metrics
        """
        baseline = baseline_latency or self._baseline

        if baseline <= 0:
            return LatencyImprovement(
                improvement_ratio=0.0,
                absolute_improvement=0.0,
                baseline_latency=baseline,
                achieved_latency=achieved_latency,
            )

        improvement_ratio = 1.0 - achieved_latency / baseline

        absolute_improvement = baseline - achieved_latency

        return LatencyImprovement(
            improvement_ratio=max(-1.0, min(1.0, improvement_ratio)),
            absolute_improvement=absolute_improvement,
            baseline_latency=baseline,
            achieved_latency=achieved_latency,
        )


class PopulationWideArchive:
    """Population-Wide Archive for algorithmic diversity search.

    The PWA maintains an archive of solutions organized by:
    1. Algorithmic strategy (not features)
    2. Performance tier (improvement level)

    Key differences from MAP-Elites:
    - Fitness衡量 = latency improvement (not correctness)
    - Diversity = algorithmic approaches (not behavioral features)
    - Stores multiple snapshots per cell (not just elite)
    - Designed for algorithm discovery (not just optimization)

    Example:
        >>> config = PWArchiveConfig(baseline_latency=100.0)
        >>> archive = PopulationWideArchive(config)
        >>> # Add solution
        >>> archive.add(snapshot)
        >>> # Query by strategy
        >>> fermat_solutions = archive.get_by_strategy(AlgorithmicStrategy.FERMAT_BASED)
    """

    def __init__(self, config: PWArchiveConfig) -> None:
        """Initialize Population-Wide Archive.

        Args:
            config: Archive configuration
        """
        self._config = config
        self._strategy_detector = StrategyDetector()
        self._latency_calculator = LatencyCalculator(config.baseline_latency)

        # Archive structure: (strategy, tier_name) -> StrategyCell
        self._cells: Dict[Tuple[AlgorithmicStrategy, str], StrategyCell] = {}

        # Global snapshot storage: snapshot_id -> CodeSnapshot
        self._snapshots: Dict[str, CodeSnapshot] = {}

        # Statistics
        self._total_added = 0
        self._total_rejected = 0
        self._total_replaced = 0

        # Initialize cells for all strategy-tier combinations
        self._initialize_cells()

    def _initialize_cells(self) -> None:
        """Initialize all strategy-tier cells."""
        for strategy in AlgorithmicStrategy:
            for tier in self._config.performance_tiers:
                cell = StrategyCell(strategy=strategy, tier=tier)
                self._cells[(strategy, tier.name)] = cell

    @property
    def config(self) -> PWArchiveConfig:
        """Get archive configuration."""
        return self._config

    def add(
        self,
        code: str,
        latency: float,
        generation: int,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        strategy: Optional[AlgorithmicStrategy] = None,
    ) -> Optional[CodeSnapshot]:
        """Add a solution to the archive.

        Args:
            code: Source code
            latency: Achieved latency in ms
            generation: Evolution generation
            parent_ids: Parent snapshot IDs
            metadata: Additional metadata
            strategy: Force specific strategy (auto-detect if None)

        Returns:
            Created CodeSnapshot or None if rejected
        """
        # Detect or use provided strategy
        detected_strategy = strategy or self._strategy_detector.detect(code)

        # Calculate improvement
        improvement = self._latency_calculator.calculate(latency)

        # Determine performance tier
        tier = self._get_tier(improvement.improvement_ratio)
        if tier is None:
            logger.warning(f"No tier found for improvement {improvement.improvement_ratio}")
            self._total_rejected += 1
            return None

        # Check if counterexample should be stored
        is_counterexample = improvement.improvement_ratio < 0.01
        if is_counterexample and not self._config.enable_counterexamples:
            self._total_rejected += 1
            return None

        # Create snapshot
        snapshot = CodeSnapshot.create(
            code=code,
            strategy=detected_strategy,
            latency=latency,
            improvement=improvement,
            generation=generation,
            parent_ids=parent_ids,
            metadata=metadata,
        )

        # Add to archive
        return self._add_to_cell(snapshot, detected_strategy, tier)

    def _add_to_cell(
        self,
        snapshot: CodeSnapshot,
        strategy: AlgorithmicStrategy,
        tier: PerformanceTier,
    ) -> Optional[CodeSnapshot]:
        """Add snapshot to appropriate cell.

        Args:
            snapshot: Snapshot to add
            strategy: Detected strategy
            tier: Performance tier

        Returns:
            Snapshot if added, None if rejected
        """
        cell_key = (strategy, tier.name)
        cell = self._cells.get(cell_key)

        if cell is None:
            cell = StrategyCell(strategy=strategy, tier=tier)
            self._cells[cell_key] = cell

        # Check if should add
        should_add = self._should_add_to_cell(cell, snapshot)

        if should_add:
            # Add snapshot
            self._snapshots[snapshot.id] = snapshot
            cell.snapshot_ids.append(snapshot.id)
            cell.added_count += 1

            # Update best if improved
            if self._is_better(snapshot, cell.snapshot):
                cell.snapshot = snapshot

            self._total_added += 1

            # Handle capacity
            self._prune_cell_if_needed(cell, cell_key)

            return snapshot
        else:
            self._total_rejected += 1
            return None

    def _should_add_to_cell(
        self,
        cell: StrategyCell,
        snapshot: CodeSnapshot,
    ) -> bool:
        """Determine if snapshot should be added to cell.

        Args:
            cell: Target cell
            snapshot: Snapshot to add

        Returns:
            True if should add
        """
        # Empty cell always accepts
        if cell.snapshot is None:
            return True

        # Check elitism
        if self._config.elitism:
            # For counterexamples, we want diverse examples
            if snapshot.improvement.improvement_ratio < 0.01:
                return len(cell.snapshot_ids) < self._config.max_snapshots_per_cell
            return self._is_better(snapshot, cell.snapshot)

        return True

    def _is_better(
        self,
        snapshot: CodeSnapshot,
        other: Optional[CodeSnapshot],
    ) -> bool:
        """Check if snapshot is better than other.

        Args:
            snapshot: Candidate snapshot
            other: Comparison snapshot

        Returns:
            True if snapshot is better
        """
        if other is None:
            return True

        # Higher improvement ratio is better
        return snapshot.improvement.improvement_ratio > other.improvement.improvement_ratio

    def _prune_cell_if_needed(
        self,
        cell: StrategyCell,
        cell_key: Tuple[AlgorithmicStrategy, str]
    ) -> None:
        """Prune cell if over capacity.

        Args:
            cell: Cell to prune
            cell_key: Cell key for removal
        """
        while len(cell.snapshot_ids) > self._config.max_snapshots_per_cell:
            # Remove oldest/weakest snapshot
            oldest_id = cell.snapshot_ids.pop(0)
            self._remove_snapshot_from_cell(cell, oldest_id)
            cell.replaced_count += 1
            self._total_replaced += 1

        # Global capacity check
        if (self._config.max_total_snapshots and
                len(self._snapshots) > self._config.max_total_snapshots):
            self._prune_global()

    def _prune_global(self) -> None:
        """Prune archive to stay within global capacity."""
        if not self._config.max_total_snapshots:
            return

        # Remove snapshots from weakest cells
        while len(self._snapshots) > self._config.max_total_snapshots:
            # Find cell with lowest improvement
            worst_cell_key = None
            worst_improvement = float('inf')

            for key, cell in self._cells.items():
                if cell.snapshot and cell.snapshot.improvement.improvement_ratio < worst_improvement:
                    worst_improvement = cell.snapshot.improvement.improvement_ratio
                    worst_cell_key = key

            if worst_cell_key is None:
                break

            cell = self._cells[worst_cell_key]
            if cell.snapshot_ids:
                removed_id = cell.snapshot_ids.pop(0)
                self._remove_snapshot_from_cell(cell, removed_id)
                self._total_replaced += 1

    def _remove_snapshot_from_cell(
        self,
        cell: StrategyCell,
        snapshot_id: str,
    ) -> None:
        """Remove a snapshot and refresh the cell elite pointer."""
        self._snapshots.pop(snapshot_id, None)
        self._refresh_cell_snapshot(cell)

    def _refresh_cell_snapshot(self, cell: StrategyCell) -> None:
        """Recompute the current best snapshot for a cell."""
        live_snapshots = [
            self._snapshots[snapshot_id]
            for snapshot_id in cell.snapshot_ids
            if snapshot_id in self._snapshots
        ]
        if not live_snapshots:
            cell.snapshot = None
            return

        cell.snapshot = max(
            live_snapshots,
            key=lambda snapshot: snapshot.improvement.improvement_ratio,
        )

    def _get_tier(self, improvement_ratio: float) -> Optional[PerformanceTier]:
        """Get performance tier for improvement ratio.

        Args:
            improvement_ratio: Improvement ratio

        Returns:
            Matching tier or None
        """
        for tier in self._config.performance_tiers:
            if tier.contains(improvement_ratio):
                return tier
        return None

    def get_by_strategy(
        self,
        strategy: AlgorithmicStrategy
    ) -> List[CodeSnapshot]:
        """Get all snapshots for a strategy.

        Args:
            strategy: Algorithmic strategy

        Returns:
            List of snapshots for that strategy
        """
        results = []
        for tier in self._config.performance_tiers:
            cell = self._cells.get((strategy, tier.name))
            if cell:
                for snapshot_id in cell.snapshot_ids:
                    if snapshot_id in self._snapshots:
                        results.append(self._snapshots[snapshot_id])
        return results

    def get_by_tier(
        self,
        tier_name: str
    ) -> List[CodeSnapshot]:
        """Get all snapshots in a performance tier.

        Args:
            tier_name: Name of the tier

        Returns:
            List of snapshots in that tier
        """
        results = []
        for strategy in AlgorithmicStrategy:
            cell = self._cells.get((strategy, tier_name))
            if cell:
                for snapshot_id in cell.snapshot_ids:
                    if snapshot_id in self._snapshots:
                        results.append(self._snapshots[snapshot_id])
        return results

    def get_by_strategy_and_tier(
        self,
        strategy: AlgorithmicStrategy,
        tier_name: str
    ) -> List[CodeSnapshot]:
        """Get snapshots for a specific strategy-tier combination.

        Args:
            strategy: Algorithmic strategy
            tier_name: Performance tier name

        Returns:
            List of snapshots
        """
        cell = self._cells.get((strategy, tier_name))
        if cell is None:
            return []

        return [
            self._snapshots[sid]
            for sid in cell.snapshot_ids
            if sid in self._snapshots
        ]

    def get_counterexamples(
        self,
        strategy: Optional[AlgorithmicStrategy] = None
    ) -> List[CodeSnapshot]:
        """Get counterexamples (solutions with minimal/negative improvement).

        Counterexamples are valuable for:
        - Understanding algorithm limitations
        - Discovering edge cases
        - Guiding mutation directions

        Args:
            strategy: Filter by strategy (None for all)

        Returns:
            List of counterexample snapshots
        """
        counterexamples = self.get_by_tier("counterexample")
        if strategy is not None:
            counterexamples = [s for s in counterexamples if s.strategy == strategy]
        return counterexamples

    def get_best_by_strategy(
        self,
        strategy: AlgorithmicStrategy
    ) -> Optional[CodeSnapshot]:
        """Get best snapshot for a strategy.

        Args:
            strategy: Algorithmic strategy

        Returns:
            Best snapshot or None
        """
        best = None
        for tier in self._config.performance_tiers:
            cell = self._cells.get((strategy, tier.name))
            if cell and cell.snapshot:
                if best is None or self._is_better(cell.snapshot, best):
                    best = cell.snapshot
        return best

    def get_best_overall(self) -> Optional[CodeSnapshot]:
        """Get best snapshot across all strategies.

        Returns:
            Best snapshot or None
        """
        best = None
        for cell in self._cells.values():
            if cell.snapshot:
                if best is None or self._is_better(cell.snapshot, best):
                    best = cell.snapshot
        return best

    def sample(
        self,
        strategy: Optional[AlgorithmicStrategy] = None,
        prefer_diverse: bool = True,
    ) -> Optional[CodeSnapshot]:
        """Sample a snapshot for mutation.

        Args:
            strategy: Prefer this strategy (None for any)
            prefer_diverse: If True, prefer under-represented strategies

        Returns:
            Sampled snapshot or None
        """
        if not self._snapshots:
            return None

        candidates = list(self._snapshots.values())

        # Filter by strategy if specified
        if strategy is not None:
            candidates = [s for s in candidates if s.strategy == strategy]

        if not candidates:
            return None

        if prefer_diverse:
            # Prefer strategies with fewer snapshots
            strategy_counts: Dict[AlgorithmicStrategy, int] = {}
            for s in candidates:
                strategy_counts[s.strategy] = strategy_counts.get(s.strategy, 0) + 1

            min_count = min(strategy_counts.values())
            candidates = [
                s for s in candidates
                if strategy_counts[s.strategy] == min_count
            ]

        return random.choice(candidates)

    def sample_neighbors(
        self,
        snapshot: CodeSnapshot,
        radius: int = 1,
    ) -> List[CodeSnapshot]:
        """Sample neighboring snapshots (similar strategy/improvement).

        Args:
            snapshot: Center snapshot
            radius: Neighborhood radius

        Returns:
            List of neighboring snapshots
        """
        neighbors = []

        # Get tier index
        current_tier = self._get_tier(snapshot.improvement.improvement_ratio)
        if current_tier is None:
            return neighbors

        tier_index = None
        for i, tier in enumerate(self._config.performance_tiers):
            if tier.name == current_tier.name:
                tier_index = i
                break

        # Find adjacent tiers
        if tier_index is not None:
            for offset in range(-radius, radius + 1):
                idx = tier_index + offset
                if 0 <= idx < len(self._config.performance_tiers):
                    tier = self._config.performance_tiers[idx]
                    cell = self._cells.get((snapshot.strategy, tier.name))
                    if cell:
                        for sid in cell.snapshot_ids:
                            if sid in self._snapshots and sid != snapshot.id:
                                neighbors.append(self._snapshots[sid])

        return neighbors

    def get_stats(self) -> PWAStats:
        """Get archive statistics.

        Returns:
            PWAStats object with archive metrics
        """
        occupied_cells = sum(
            1 for cell in self._cells.values() if cell.snapshot is not None
        )
        total_cells = len(self._cells)

        # Strategy distribution
        strategies_used: Dict[str, int] = {}
        for snapshot in self._snapshots.values():
            strategies_used[snapshot.strategy.value] = (
                strategies_used.get(snapshot.strategy.value, 0) + 1
            )

        # Tier distribution
        tier_distribution: Dict[str, int] = {}
        for cell in self._cells.values():
            if cell.snapshot:
                tier_distribution[cell.tier.name] = (
                    tier_distribution.get(cell.tier.name, 0) + 1
                )

        # Best improvement
        best = self.get_best_overall()
        best_improvement = best.improvement.improvement_ratio if best else 0.0

        return PWAStats(
            total_snapshots=len(self._snapshots),
            occupied_cells=occupied_cells,
            total_cells=total_cells,
            fill_rate=occupied_cells / total_cells if total_cells > 0 else 0.0,
            best_improvement=best_improvement,
            strategies_used=strategies_used,
            tier_distribution=tier_distribution,
            counterexample_count=len(self.get_counterexamples()),
            total_added=self._total_added,
            total_rejected=self._total_rejected,
        )

    def get_all_snapshots(self) -> List[CodeSnapshot]:
        """Get all snapshots in archive.

        Returns:
            List of all snapshots
        """
        return list(self._snapshots.values())

    def clear(self) -> None:
        """Clear all archive contents."""
        self._snapshots.clear()
        self._cells.clear()
        self._total_added = 0
        self._total_rejected = 0
        self._total_replaced = 0
        self._initialize_cells()

    def export(self) -> Dict[str, Any]:
        """Export archive to dictionary.

        Returns:
            Serializable dictionary
        """
        return {
            "config": {
                "max_snapshots_per_cell": self._config.max_snapshots_per_cell,
                "max_total_snapshots": self._config.max_total_snapshots,
                "elitism": self._config.elitism,
                "enable_counterexamples": self._config.enable_counterexamples,
                "baseline_latency": self._config.baseline_latency,
                "tiers": [
                    {"name": t.name, "min": t.min_improvement, "max": t.max_improvement}
                    for t in self._config.performance_tiers
                ],
            },
            "snapshots": [
                {
                    "id": s.id,
                    "code": s.code,
                    "strategy": s.strategy.value,
                    "latency": s.latency,
                    "improvement": {
                        "ratio": s.improvement.improvement_ratio,
                        "absolute": s.improvement.absolute_improvement,
                        "baseline": s.improvement.baseline_latency,
                        "achieved": s.improvement.achieved_latency,
                    },
                    "generation": s.generation,
                    "parent_ids": s.parent_ids,
                    "metadata": s.metadata,
                    "created_at": s.created_at,
                }
                for s in self._snapshots.values()
            ],
            "stats": {
                "total_added": self._total_added,
                "total_rejected": self._total_rejected,
                "total_replaced": self._total_replaced,
            },
        }

    def save(self, path: str) -> None:
        """Save archive to file.

        Args:
            path: File path to save to
        """
        data = self.export()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"PWA archive saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PopulationWideArchive":
        """Load archive from file.

        Args:
            path: File path to load from

        Returns:
            Loaded PopulationWideArchive
        """
        with open(path, 'r') as f:
            data = json.load(f)

        # Reconstruct config
        tiers = [
            PerformanceTier(t["name"], t["min"], t["max"])
            for t in data["config"]["tiers"]
        ]
        config = PWArchiveConfig(
            max_snapshots_per_cell=data["config"]["max_snapshots_per_cell"],
            max_total_snapshots=data["config"]["max_total_snapshots"],
            elitism=data["config"]["elitism"],
            enable_counterexamples=data["config"]["enable_counterexamples"],
            baseline_latency=data["config"]["baseline_latency"],
            performance_tiers=tiers,
        )

        archive = cls(config)

        # Reconstruct snapshots
        for snap_data in data["snapshots"]:
            strategy = AlgorithmicStrategy(snap_data["strategy"])
            improvement = LatencyImprovement(
                improvement_ratio=snap_data["improvement"]["ratio"],
                absolute_improvement=snap_data["improvement"]["absolute"],
                baseline_latency=snap_data["improvement"]["baseline"],
                achieved_latency=snap_data["improvement"]["achieved"],
            )
            snapshot = CodeSnapshot(
                id=snap_data["id"],
                code=snap_data["code"],
                strategy=strategy,
                latency=snap_data["latency"],
                improvement=improvement,
                generation=snap_data["generation"],
                parent_ids=snap_data["parent_ids"],
                metadata=snap_data["metadata"],
                created_at=snap_data["created_at"],
            )

            # Re-add to cells
            tier = archive._get_tier(improvement.improvement_ratio)
            if tier:
                archive._snapshots[snapshot.id] = snapshot
                cell = archive._cells.get((strategy, tier.name))
                if cell:
                    if snapshot.id not in cell.snapshot_ids:
                        cell.snapshot_ids.append(snapshot.id)
                    cell.added_count += 1
                    if cell.snapshot is None or archive._is_better(snapshot, cell.snapshot):
                        cell.snapshot = snapshot

        archive._total_added = data["stats"]["total_added"]
        archive._total_rejected = data["stats"]["total_rejected"]
        archive._total_replaced = data["stats"].get("total_replaced", 0)

        logger.info(f"PWA archive loaded from {path}")
        return archive


class PWARetrieval:
    """Retrieval utilities for Population-Wide Archive.

    Provides methods to query the archive for specific needs
    like algorithm discovery and counterexample analysis.
    """

    def __init__(self, archive: PopulationWideArchive) -> None:
        """Initialize retrieval utilities.

        Args:
            archive: The archive to query
        """
        self._archive = archive

    def find_counterexamples_for_strategy(
        self,
        target_strategy: AlgorithmicStrategy,
    ) -> List[CodeSnapshot]:
        """Find counterexamples that could guide improvement of a strategy.

        Args:
            target_strategy: Strategy to improve

        Returns:
            Counterexamples from other strategies that might inform the target
        """
        counterexamples = self._archive.get_counterexamples()
        return [
            c for c in counterexamples
            if c.strategy != target_strategy
        ]

    def find_missing_improvements(
        self,
    ) -> List[Tuple[AlgorithmicStrategy, str]]:
        """Find strategy-tier combinations with no solutions.

        Useful for identifying gaps in the archive where
        new algorithms should be discovered.

        Returns:
            List of (strategy, tier_name) combinations with no solutions
        """
        missing = []
        for strategy in AlgorithmicStrategy:
            for tier in self._archive.config.performance_tiers:
                cell = self._archive._cells.get((strategy, tier.name))
                if cell is None or cell.snapshot is None:
                    missing.append((strategy, tier.name))
        return missing

    def get_algorithm_diversity_score(self) -> float:
        """Calculate diversity score based on strategy coverage.

        Returns:
            Diversity score (0.0 to 1.0)
        """
        if not self._archive._snapshots:
            return 0.0

        strategies_with_solutions = set(
            s.strategy for s in self._archive._snapshots.values()
        )

        # Each strategy adds to diversity
        strategy_score = len(strategies_with_solutions) / len(AlgorithmicStrategy)

        # Check tier coverage
        tiers_with_solutions = set()
        for cell in self._archive._cells.values():
            if cell.snapshot:
                tiers_with_solutions.add(cell.tier.name)

        tier_score = len(tiers_with_solutions) / len(self._archive.config.performance_tiers)

        return (strategy_score + tier_score) / 2.0

    def get_improvement_frontier(
        self,
    ) -> Dict[AlgorithmicStrategy, CodeSnapshot]:
        """Get best solution for each strategy.

        Returns:
            Dictionary mapping strategy to best snapshot
        """
        frontier = {}
        for strategy in AlgorithmicStrategy:
            best = self._archive.get_best_by_strategy(strategy)
            if best:
                frontier[strategy] = best
        return frontier
