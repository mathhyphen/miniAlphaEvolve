"""Steiner Tree-specific Population-Wide Archive extensions.

This module provides specialized PWA functionality for Steiner Tree research:
- Custom latency metrics for Steiner ratio improvement
- Counterexample discovery for algorithm limitations
- Integration with Steiner test cases
- Visualization data generation
"""

import ast
import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from alphaevolve.archive.pwa import (
    AlgorithmicStrategy,
    CodeSnapshot,
    LatencyImprovement,
    PerformanceTier,
    PopulationWideArchive,
    PWArchiveConfig,
    PWAStats,
    PWARetrieval,
)

logger = logging.getLogger(__name__)


# Steiner Tree specific algorithmic strategies (extends base strategies)
class SteinerStrategy(Enum):
    """Extended strategy set for Steiner Tree algorithms.

    These are additional strategies beyond the base AlgorithmicStrategy
    that are specific to Steiner Tree problems.
    """
    GEOMETRIC_FERMAT = "geometric_fermat"   # Fermat point with geometric extensions
    VORONOI_BASED = "voronoi"              # Voronoi diagram based
    GRAPH_CONSTRUCT = "graph_construct"    # Graph construction based
    FLATTENING = "flattening"              # Point flattening technique
    CLUSTERING = "clustering"              # Terminal clustering
    DYNAMIC = "dynamic"                    # Dynamic programming
    MINKOWSKI = "minkowski"                # Minkowski sum based


# All strategies including Steiner-specific ones
STEINER_STRATEGIES = [
    AlgorithmicStrategy.UNKNOWN,
    AlgorithmicStrategy.FERMAT_BASED,
    AlgorithmicStrategy.CENTROID_BASED,
    AlgorithmicStrategy.GEOMETRIC,
    AlgorithmicStrategy.OPTIMIZATION,
    AlgorithmicStrategy.HYBRID,
    AlgorithmicStrategy.HEURISTIC,
    AlgorithmicStrategy.EXACT,
]


@dataclass(frozen=True)
class SteinerImprovement(LatencyImprovement):
    """Steiner Tree specific improvement metrics.

    Adds Steiner-ratio specific measurements to LatencyImprovement.

    Note: Must be frozen because parent LatencyImprovement is frozen.
    """
    steiner_ratio: float = 0.0
    optimal_ratio: float = 0.866  # sqrt(3)/2 theoretical optimum
    mst_ratio: float = 1.0        # MST ratio (baseline)


class SteinerLatencyCalculator:
    """Calculator for Steiner Tree latency improvement metrics.

    Computes improvement based on Steiner ratio rather than raw latency.
    """

    # Theoretical optimal ratio for Euclidean Steiner Tree
    OPTIMAL_RATIO = math.sqrt(3) / 2  # ≈ 0.866

    def __init__(self, baseline_mst_ratio: float = 1.0) -> None:
        """Initialize calculator.

        Args:
            baseline_mst_ratio: MST ratio (should be 1.0)
        """
        self._baseline = baseline_mst_ratio

    def calculate(
        self,
        steiner_ratio: float,
        optimal_ratio: Optional[float] = None,
    ) -> SteinerImprovement:
        """Calculate improvement from Steiner ratio.

        Args:
            steiner_ratio: Achieved Steiner ratio (between 0 and 1)
            optimal_ratio: Known optimal ratio for this problem (default: theoretical optimum)

        Returns:
            SteinerImprovement with detailed metrics
        """
        optimal = optimal_ratio or self.OPTIMAL_RATIO

        # Ensure ratio is valid
        steiner_ratio = max(0.0, min(1.0, steiner_ratio))

        # Calculate improvement over MST (baseline = 1.0)
        # Higher ratio = better (closer to optimal)
        if self._baseline > 0:
            improvement_ratio = 1.0 - steiner_ratio / self._baseline
        else:
            improvement_ratio = 0.0

        # Calculate how close to optimal
        # optimal_ratio represents optimal, steiner_ratio should be >= optimal
        if optimal > 0:
            optimality_ratio = steiner_ratio / optimal
        else:
            optimality_ratio = 0.0

        return SteinerImprovement(
            improvement_ratio=max(-1.0, min(1.0, improvement_ratio)),
            absolute_improvement=self._baseline - steiner_ratio,
            baseline_latency=self._baseline,
            achieved_latency=steiner_ratio,
            steiner_ratio=steiner_ratio,
            optimal_ratio=optimal,
            mst_ratio=self._baseline,
        )


class SteinerStrategyDetector:
    """Specialized strategy detector for Steiner Tree algorithms.

    Detects algorithmic approaches specific to Steiner Tree problems.
    """

    def __init__(self) -> None:
        """Initialize detector."""
        self._steiner_keywords = {
            'steiner', 'steiner_tree', 'steiner_ratio',
        }
        self._fermat_keywords = [
            'fermat', 'torricelli', '120_degrees', 'three_point',
            'equilateral_triangle', 'angle_property'
        ]
        self._voronoi_keywords = [
            'voronoi', 'voronoi_diagram', 'delaunay', 'duality'
        ]
        self._mst_keywords = [
            'mst', 'minimum_spanning', 'prim', 'kruskal', 'spanning_tree'
        ]
        self._clustering_keywords = [
            'cluster', 'group', 'partition', 'kmeans', 'hierarchical'
        ]
        self._geometric_keywords = [
            'geometric', 'convex', 'circumcenter', 'perpendicular_bisector'
        ]
        self._flattening_keywords = [
            'flatten', 'projection', '2d', 'transform'
        ]
        self._dynamic_keywords = [
            'dynamic', 'dp', 'memoize', 'bitmask', 'subset'
        ]
        self._graph_keywords = [
            'graph', 'node', 'edge', 'adjacency', 'neighbor'
        ]

    def detect(self, code: str) -> AlgorithmicStrategy:
        """Detect Steiner Tree algorithm strategy.

        Args:
            code: Source code

        Returns:
            AlgorithmicStrategy
        """
        code_lower = code.lower()

        # Count keyword matches for each strategy
        fermat_count = sum(1 for kw in self._fermat_keywords if kw in code_lower)
        voronoi_count = sum(1 for kw in self._voronoi_keywords if kw in code_lower)
        mst_count = sum(1 for kw in self._mst_keywords if kw in code_lower)
        cluster_count = sum(1 for kw in self._clustering_keywords if kw in code_lower)
        geo_count = sum(1 for kw in self._geometric_keywords if kw in code_lower)
        flatten_count = sum(1 for kw in self._flattening_keywords if kw in code_lower)
        dp_count = sum(1 for kw in self._dynamic_keywords if kw in code_lower)
        graph_count = sum(1 for kw in self._graph_keywords if kw in code_lower)

        # AST-based detection
        try:
            tree = ast.parse(code)
            struct_score = self._detect_structure(tree)
        except SyntaxError:
            struct_score = {}

        # Combine scores - map to AlgorithmicStrategy
        # Note: Steiner-specific strategies map to base strategies
        scores: Dict[AlgorithmicStrategy, int] = {
            AlgorithmicStrategy.FERMAT_BASED: fermat_count + struct_score.get('fermat', 0),
            AlgorithmicStrategy.GEOMETRIC: (
                voronoi_count +
                geo_count +
                flatten_count +
                graph_count +
                struct_score.get('voronoi', 0) +
                struct_score.get('geometric', 0)
            ),
            AlgorithmicStrategy.CENTROID_BASED: cluster_count + struct_score.get('centroid', 0),
            AlgorithmicStrategy.OPTIMIZATION: dp_count + struct_score.get('optimization', 0),
            AlgorithmicStrategy.HEURISTIC: mst_count + struct_score.get('heuristic', 0),
            AlgorithmicStrategy.EXACT: struct_score.get('exact', 0),
        }

        # If MST is primary, might just be baseline
        if mst_count > 0 and max(scores.values()) == mst_count:
            return AlgorithmicStrategy.HEURISTIC

        # Return best matching strategy
        if max(scores.values()) == 0:
            return AlgorithmicStrategy.UNKNOWN

        return max(scores.keys(), key=lambda s: scores[s])

    def _detect_structure(self, tree: ast.AST) -> Dict[str, int]:
        """Detect strategy from AST structure.

        Args:
            tree: Parsed AST

        Returns:
            Dictionary of detected patterns
        """
        scores: Dict[str, int] = {}

        # Check for recursive functions (exact/optimization)
        func_defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for func_name in func_defs:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == func_name:
                        scores['exact'] = scores.get('exact', 0) + 1
                        scores['optimization'] = scores.get('optimization', 0) + 1

        # Check for loops with termination conditions (heuristic)
        loop_count = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, (ast.For, ast.While))
        )
        if loop_count > 2:
            scores['heuristic'] = scores.get('heuristic', 0) + loop_count

        return scores


class SteinerPWA(PopulationWideArchive):
    """Population-Wide Archive specialized for Steiner Tree research.

    Extends PopulationWideArchive with Steiner-specific functionality:
    - Steiner ratio-based improvement calculation
    - Counterexample discovery for algorithm analysis
    - Test case integration
    """

    def __init__(
        self,
        config: PWArchiveConfig,
        steiner_calculator: Optional[SteinerLatencyCalculator] = None,
        strategy_detector: Optional[SteinerStrategyDetector] = None,
    ) -> None:
        """Initialize Steiner PWA.

        Args:
            config: Archive configuration
            steiner_calculator: Custom Steiner latency calculator
            strategy_detector: Custom strategy detector
        """
        super().__init__(config)

        self._steiner_calculator = steiner_calculator or SteinerLatencyCalculator()
        self._strategy_detector = strategy_detector or SteinerStrategyDetector()

    def add_solution(
        self,
        code: str,
        steiner_ratio: float,
        generation: int,
        optimal_ratio: Optional[float] = None,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[CodeSnapshot]:
        """Add a Steiner Tree solution to the archive.

        Args:
            code: Source code
            steiner_ratio: Achieved Steiner ratio
            generation: Evolution generation
            optimal_ratio: Known optimal ratio (for scoring)
            parent_ids: Parent snapshot IDs
            metadata: Additional metadata

        Returns:
            Created CodeSnapshot or None if rejected
        """
        # Detect strategy
        strategy = self._strategy_detector.detect(code)

        # Calculate improvement using Steiner calculator
        improvement = self._steiner_calculator.calculate(steiner_ratio, optimal_ratio)

        # Determine performance tier
        tier = self._get_tier(improvement.improvement_ratio)
        if tier is None:
            logger.warning(f"No tier for improvement {improvement.improvement_ratio}")
            return None

        # Check counterexample storage
        is_counterexample = improvement.improvement_ratio < 0.01
        if is_counterexample and not self._config.enable_counterexamples:
            return None

        # Create snapshot
        snapshot = CodeSnapshot(
            id=self._generate_id(code),
            code=code,
            strategy=strategy,
            latency=steiner_ratio,
            improvement=improvement,
            generation=generation,
            parent_ids=parent_ids or [],
            metadata=metadata or {"type": "steiner"},
        )

        return self._add_to_cell(snapshot, strategy, tier)

    def _generate_id(self, code: str) -> str:
        """Generate unique ID for snapshot."""
        import time
        content = f"{code}{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_counterexamples_by_type(
        self,
    ) -> Dict[str, List[CodeSnapshot]]:
        """Get counterexamples grouped by strategy.

        Returns:
            Dictionary mapping strategy to counterexample list
        """
        counterexamples = self.get_counterexamples()
        grouped: Dict[str, List[CodeSnapshot]] = {}

        for snapshot in counterexamples:
            strategy_name = snapshot.strategy.value
            if strategy_name not in grouped:
                grouped[strategy_name] = []
            grouped[strategy_name].append(snapshot)

        return grouped

    def find_potential_hybrids(
        self,
    ) -> List[Tuple[CodeSnapshot, CodeSnapshot]]:
        """Find pairs of solutions that could be combined into hybrids.

        Returns:
            List of (solution1, solution2) pairs with different strategies
        """
        pairs = []

        # Get best from each strategy
        best_by_strategy: Dict[AlgorithmicStrategy, CodeSnapshot] = {}
        for snapshot in self.get_all_snapshots():
            current = best_by_strategy.get(snapshot.strategy)
            if current is None or snapshot.improvement.improvement_ratio > current.improvement.improvement_ratio:
                best_by_strategy[snapshot.strategy] = snapshot

        strategies = list(best_by_strategy.keys())

        # Create pairs of different strategies
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1:]:
                snapshot1 = best_by_strategy[s1]
                snapshot2 = best_by_strategy[s2]
                # Prefer pairs where both are reasonably good
                if (snapshot1.improvement.improvement_ratio > 0.1 and
                        snapshot2.improvement.improvement_ratio > 0.1):
                    pairs.append((snapshot1, snapshot2))

        return pairs


class SteinerPWARetrieval(PWARetrieval):
    """Specialized retrieval for Steiner Tree PWA."""

    def find_critical_test_cases(
        self,
        test_cases: List[Any],
    ) -> List[Any]:
        """Find test cases that discriminate between algorithms.

        Args:
            test_cases: List of test cases

        Returns:
            Most discriminating test cases
        """
        # This would require running solutions on test cases
        # Placeholder for integration with test case system
        return test_cases[:5]  # Return first 5 as placeholder

    def generate_diversity_report(self) -> Dict[str, Any]:
        """Generate a diversity report for the archive.

        Returns:
            Dictionary with diversity metrics
        """
        frontier = self.get_improvement_frontier()

        strategies_covered = len(frontier)
        total_strategies = len(AlgorithmicStrategy)

        # Calculate average improvement per strategy
        improvement_by_strategy = {
            strat.value: snap.improvement.improvement_ratio
            for strat, snap in frontier.items()
        }

        return {
            "strategies_covered": strategies_covered,
            "total_strategies": total_strategies,
            "coverage_ratio": strategies_covered / total_strategies,
            "improvement_by_strategy": improvement_by_strategy,
            "diversity_score": self.get_algorithm_diversity_score(),
            "missing_strategies": [
                s.value for s in AlgorithmicStrategy
                if s not in frontier
            ],
        }


def create_steiner_pwa(
    baseline_latency: float = 100.0,
    max_snapshots: int = 500,
    enable_counterexamples: bool = True,
) -> Tuple[SteinerPWA, SteinerPWARetrieval]:
    """Create a configured Steiner PWA.

    Args:
        baseline_latency: Baseline latency for improvement calculation
        max_snapshots: Maximum snapshots to store
        enable_counterexamples: Whether to store counterexamples

    Returns:
        Tuple of (SteinerPWA, SteinerPWARetrieval)
    """
    config = PWArchiveConfig(
        max_snapshots_per_cell=10,
        max_total_snapshots=max_snapshots,
        elitism=True,
        enable_counterexamples=enable_counterexamples,
        baseline_latency=baseline_latency,
        performance_tiers=PerformanceTier.get_default_tiers(),
    )

    archive = SteinerPWA(config)
    retrieval = SteinerPWARetrieval(archive)

    return archive, retrieval
