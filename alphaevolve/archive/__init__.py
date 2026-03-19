"""Population-Wide Archive module for AlphaEvolve-redesign.

This module provides the PWA archive system that differs from MAP-Elites:
- Fitness = latency improvement (not correctness)
- Diversity = algorithmic approaches (not features)
- Stores code snapshots at different performance levels
- Enables discovering novel algorithms

Steiner Tree Extensions:
- Specialized latency calculators for Steiner ratio
- Counterexample discovery for algorithm analysis
- Integration with test cases
"""

from alphaevolve.archive.pwa import (
    AlgorithmicStrategy,
    LatencyImprovement,
    CodeSnapshot,
    PerformanceTier,
    StrategyCell,
    PWArchiveConfig,
    PWAStats,
    StrategyDetector,
    LatencyCalculator,
    PopulationWideArchive,
    PWARetrieval,
)

# Steiner-specific extensions
from alphaevolve.archive.steiner_pwa import (
    SteinerImprovement,
    SteinerLatencyCalculator,
    SteinerStrategyDetector,
    SteinerStrategy,
    SteinerPWA,
    SteinerPWARetrieval,
    create_steiner_pwa,
    STEINER_STRATEGIES,
)

__all__ = [
    # Core PWA classes
    "PopulationWideArchive",
    "PWArchiveConfig",
    "PWAStats",
    "PWARetrieval",

    # Snapshot and tier classes
    "CodeSnapshot",
    "PerformanceTier",
    "StrategyCell",
    "LatencyImprovement",

    # Strategy enumeration
    "AlgorithmicStrategy",

    # Utility classes
    "StrategyDetector",
    "LatencyCalculator",

    # Steiner Tree extensions
    "SteinerImprovement",
    "SteinerLatencyCalculator",
    "SteinerStrategyDetector",
    "SteinerStrategy",
    "SteinerPWA",
    "SteinerPWARetrieval",
    "create_steiner_pwa",
    "STEINER_STRATEGIES",
]
