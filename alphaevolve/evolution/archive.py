"""MAP-Elites archive for quality-diversity search.

The MAP-Elites algorithm maintains an archive of diverse, high-performing solutions.
Each solution is placed in a cell based on its behavioral characteristics (features).
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import random

from alphaevolve.core.data_structures import Individual
from alphaevolve.evolution.features import FeatureDimension

logger = logging.getLogger(__name__)


@dataclass
class ArchiveConfig:
    """Configuration for MAP-Elites archive.

    Args:
        feature_dimensions: List of feature extractors
        bins_per_dimension: Number of bins for each feature
        max_archive_size: Maximum archive capacity (None for unlimited)
        elitism: Whether to replace only on strictly better fitness
    """
    feature_dimensions: List[FeatureDimension] = field(default_factory=list)
    bins_per_dimension: int = 10
    max_archive_size: Optional[int] = None
    elitism: bool = True

    def __post_init__(self) -> None:
        if self.bins_per_dimension < 2:
            raise ValueError("bins_per_dimension must be at least 2")


@dataclass
class ArchiveStats:
    """Statistics about archive state."""
    fill_rate: float  # Percentage of cells filled
    unique_solutions: int  # Number of unique solutions
    best_fitness: float  # Best fitness in archive
    avg_fitness: float  # Average fitness
    diversity: float  # Feature space coverage


class ProgramArchive:
    """MAP-Elites archive for program storage.

    The archive is a multi-dimensional grid where each cell contains
    the best program for that behavioral niche.

    Features:
    - Quality-diversity maintenance
    - Feature-based retrieval
    - Neighbor-based sampling
    - Statistics tracking
    """

    def __init__(self, config: ArchiveConfig) -> None:
        """Initialize archive.

        Args:
            config: Archive configuration
        """
        self.config = config
        self._features = config.feature_dimensions
        self._bins_per_dim = config.bins_per_dimension
        self._max_size = config.max_archive_size

        # Archive storage: cell_id -> Individual
        self._archive: Dict[Tuple[int, ...], Individual] = {}

        # Statistics tracking
        self._total_added = 0
        self._total_rejected = 0
        self._total_replaced = 0

        # Feature space dimensions
        self._ndims = len(self._features)

    def add(self, individual: Individual) -> bool:
        """Add individual to archive.

        Uses MAP-Elites selection:
        1. Extract feature values
        2. Map to cell ID
        3. Add if cell empty OR fitness better than current

        Args:
            individual: Individual to add

        Returns:
            True if added, False if rejected
        """
        # Extract feature vector
        feature_vector = self._extract_features(individual.code)
        if feature_vector is None:
            self._total_rejected += 1
            return False

        # Convert to cell ID
        cell_id = self._feature_to_cell(feature_vector)

        # Check if should add
        should_add = self._should_add(cell_id, individual)

        if should_add:
            self._archive[cell_id] = individual
            individual.metadata["cell_id"] = cell_id
            individual.metadata["feature_vector"] = feature_vector
            self._total_added += 1

            # Handle size limit
            if self._max_size and len(self._archive) > self._max_size:
                self._evict_worst()

            return True
        else:
            self._total_rejected += 1
            return False

    def get(self, cell_id: Tuple[int, ...]) -> Optional[Individual]:
        """Get individual by cell ID.

        Args:
            cell_id: Cell identifier

        Returns:
            Individual or None if cell empty
        """
        return self._archive.get(cell_id)

    def get_random(self) -> Optional[Individual]:
        """Get random individual from archive.

        Returns:
            Random individual or None if archive empty
        """
        if not self._archive:
            return None
        return random.choice(list(self._archive.values()))

    def get_neighbors(
        self,
        individual: Individual,
        radius: int = 1,
    ) -> List[Individual]:
        """Get neighboring individuals in feature space.

        Args:
            individual: Center individual
            radius: Neighborhood radius in cells

        Returns:
            List of neighboring individuals
        """
        cell_id = individual.metadata.get("cell_id")
        if cell_id is None:
            # Extract features if not available
            feature_vector = self._extract_features(individual.code)
            if feature_vector is None:
                return []
            cell_id = self._feature_to_cell(feature_vector)

        neighbors = []
        for offset in self._generate_offsets(radius):
            neighbor_cell = tuple(
                max(0, min(self._bins_per_dim - 1, cell_id[i] + offset[i]))
                for i in range(self._ndims)
            )
            if neighbor_cell in self._archive and neighbor_cell != cell_id:
                neighbors.append(self._archive[neighbor_cell])

        return neighbors

    def sample(self, strategy: str = "random") -> Optional[Individual]:
        """Sample individual for mutation.

        Args:
            strategy: Sampling strategy ("random", "elite", "diverse")

        Returns:
            Selected individual or None
        """
        if not self._archive:
            return None

        if strategy == "elite":
            return self.get_best()
        elif strategy == "diverse":
            return self._sample_diverse()
        else:  # random
            return self.get_random()

    def get_best(self) -> Optional[Individual]:
        """Get individual with highest fitness.

        Returns:
            Best individual or None if archive empty
        """
        if not self._archive:
            return None
        return max(self._archive.values(), key=lambda ind: ind.fitness or 0)

    def get_stats(self) -> ArchiveStats:
        """Get archive statistics.

        Returns:
            ArchiveStats object
        """
        total_cells = self._bins_per_dim ** self._ndims if self._ndims > 0 else 0
        fill_rate = len(self._archive) / total_cells if total_cells > 0 else 0

        individuals = list(self._archive.values())
        if individuals:
            fitnesses = [ind.fitness or 0 for ind in individuals]
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
        else:
            best_fitness = 0.0
            avg_fitness = 0.0

        return ArchiveStats(
            fill_rate=fill_rate,
            unique_solutions=len(self._archive),
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            diversity=fill_rate,  # Use fill rate as diversity proxy
        )

    def clear(self) -> None:
        """Clear all archive contents."""
        self._archive.clear()
        self._total_added = 0
        self._total_rejected = 0
        self._total_replaced = 0

    def export(self) -> Dict[str, Any]:
        """Export archive to dictionary.

        Returns:
            Serializable dictionary
        """
        return {
            "config": {
                "bins_per_dimension": self._bins_per_dim,
                "feature_count": self._ndims,
                "max_size": self._max_size,
            },
            "individuals": [
                {
                    "id": ind.id,
                    "code": ind.code,
                    "fitness": ind.fitness,
                    "cell_id": ind.metadata.get("cell_id"),
                    "feature_vector": ind.metadata.get("feature_vector"),
                }
                for ind in self._archive.values()
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
        logger.info(f"Archive saved to {path}")

    @classmethod
    def load(cls, path: str, features: List[FeatureDimension]) -> "ProgramArchive":
        """Load archive from file.

        Args:
            path: File path to load from
            features: Feature dimensions (must match saved archive)

        Returns:
            Loaded archive
        """
        with open(path, 'r') as f:
            data = json.load(f)

        config = ArchiveConfig(
            feature_dimensions=features,
            bins_per_dimension=data["config"]["bins_per_dimension"],
            max_size=data["config"]["max_size"],
        )

        archive = cls(config)
        archive._total_added = data["stats"]["total_added"]
        archive._total_rejected = data["stats"]["total_rejected"]

        for ind_data in data["individuals"]:
            cell_id = tuple(ind_data["cell_id"]) if ind_data.get("cell_id") else None
            if cell_id is None:
                continue  # Skip individuals without valid cell_id

            ind = Individual(
                id=ind_data["id"],
                code=ind_data["code"],
                fitness=ind_data["fitness"],
            )
            ind.metadata["cell_id"] = cell_id
            ind.metadata["feature_vector"] = ind_data.get("feature_vector")
            archive._archive[cell_id] = ind

        logger.info(f"Archive loaded from {path}")
        return archive

    def _extract_features(self, code: str) -> Optional[List[float]]:
        """Extract feature values from code.

        Args:
            code: Source code

        Returns:
            List of feature values or None on error
        """
        try:
            return [f.extract(code) for f in self._features]
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def _feature_to_cell(
        self,
        feature_vector: List[float],
    ) -> Tuple[int, ...]:
        """Convert feature vector to cell ID.

        Args:
            feature_vector: Raw feature values

        Returns:
            Discretized cell ID tuple
        """
        cell_id = tuple(
            f.discretize(v)
            for f, v in zip(self._features, feature_vector)
        )
        return cell_id

    def _should_add(
        self,
        cell_id: Tuple[int, ...],
        individual: Individual,
    ) -> bool:
        """Check if individual should be added to cell.

        Args:
            cell_id: Target cell
            individual: Individual to add

        Returns:
            True if should add
        """
        existing = self._archive.get(cell_id)

        if existing is None:
            return True

        if self.config.elitism:
            return (individual.fitness or 0) > (existing.fitness or 0)
        else:
            return True

    def _evict_worst(self) -> None:
        """Remove worst individual when archive is full."""
        if not self._archive:
            return

        # Find and remove lowest fitness
        worst_cell = min(
            self._archive.keys(),
            key=lambda c: self._archive[c].fitness or 0
        )
        del self._archive[worst_cell]

    def _generate_offsets(
        self,
        radius: int,
    ) -> List[Tuple[int, ...]]:
        """Generate all offset tuples within radius.

        Args:
            radius: Maximum offset in each dimension

        Returns:
            List of offset tuples
        """
        from itertools import product

        offsets = range(-radius, radius + 1)
        return list(product(offsets, repeat=self._ndims))

    def _sample_diverse(self) -> Optional[Individual]:
        """Sample individual favoring under-explored regions."""
        if not self._archive:
            return None

        # Simple implementation: prefer cells with fewer neighbors
        # This encourages diversity
        cells = list(self._archive.keys())

        # Score by local density (fewer neighbors = more diverse)
        scored = []
        for cell in cells:
            neighbors = sum(
                1 for c in cells
                if c != cell and all(abs(c[i] - cell[i]) <= 1 for i in range(self._ndims))
            )
            scored.append((neighbors, cell))

        # Select from least crowded
        min_neighbors = min(s[0] for s in scored)
        candidates = [self._archive[s[1]] for s in scored if s[0] == min_neighbors]

        return random.choice(candidates) if candidates else self.get_random()
