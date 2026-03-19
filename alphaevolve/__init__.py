"""AlphaEvolve Redesign - Evolutionary algorithm framework with LLM agents.

This package provides the Population-Wide Archive (PWA) system for
evolutionary algorithm discovery. Key differences from standard MAP-Elites:

- Fitness = latency improvement (not correctness score)
- Diversity = algorithmic approaches (not feature dimensions)
- Stores code snapshots at different performance levels
- Enables discovering novel algorithms

Example:
    >>> from alphaevolve.archive import PopulationWideArchive, PWArchiveConfig
    >>> config = PWArchiveConfig(baseline_latency=100.0)
    >>> archive = PopulationWideArchive(config)
    >>> # Add a solution
    >>> archive.add(code, latency=85.0, generation=1)
    >>> # Query by strategy
    >>> from alphaevolve.archive import AlgorithmicStrategy
    >>> fermat_solutions = archive.get_by_strategy(AlgorithmicStrategy.FERMAT_BASED)
"""

__version__ = "0.1.0"
__author__ = "alphaevolve-redesign Team"

from alphaevolve.archive import (
    # Core PWA
    PopulationWideArchive,
    PWArchiveConfig,
    PWAStats,
    PWARetrieval,
    CodeSnapshot,
    PerformanceTier,
    StrategyCell,
    LatencyImprovement,
    AlgorithmicStrategy,
    StrategyDetector,
    LatencyCalculator,
    # Steiner extensions
    SteinerPWA,
    SteinerImprovement,
    SteinerLatencyCalculator,
    SteinerStrategyDetector,
    SteinerPWARetrieval,
    create_steiner_pwa,
    STEINER_STRATEGIES,
)

__all__ = [
    # Version
    "__version__",
    "__author__",

    # Core PWA
    "PopulationWideArchive",
    "PWArchiveConfig",
    "PWAStats",
    "PWARetrieval",
    "CodeSnapshot",
    "PerformanceTier",
    "StrategyCell",
    "LatencyImprovement",
    "AlgorithmicStrategy",
    "StrategyDetector",
    "LatencyCalculator",

    # Steiner extensions
    "SteinerPWA",
    "SteinerImprovement",
    "SteinerLatencyCalculator",
    "SteinerStrategyDetector",
    "SteinerPWARetrieval",
    "create_steiner_pwa",
    "STEINER_STRATEGIES",
]
