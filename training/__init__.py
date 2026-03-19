"""AlphaEvolve training module.

This module provides the RL-based training loop for evolving
code patches targeting graph theory problems.
"""

from .evolution_loop import (
    EvolutionConfig,
    EvolutionLoop,
    EpisodeResult,
)

__all__ = [
    "EvolutionConfig",
    "EvolutionLoop",
    "EpisodeResult",
]
