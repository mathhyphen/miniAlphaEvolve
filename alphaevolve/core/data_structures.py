"""Core data structures for AlphaEvolve."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import hashlib
import time


@dataclass(frozen=True)
class EvolutionConfig:
    """Configuration for evolutionary process.

    Args:
        population_size: Number of individuals in population
        mutation_rate: Probability of mutation (0.0-1.0)
        selection_pressure: Tournament size for selection
        max_generations: Maximum number of generations
        elitism_count: Number of best individuals to preserve
        diversity_threshold: Minimum diversity to maintain
        timeout_seconds: Maximum time for evolution
        seed: Random seed for reproducibility
    """
    population_size: int = 50
    mutation_rate: float = 0.7
    selection_pressure: int = 3
    max_generations: int = 100
    elitism_count: int = 2
    diversity_threshold: float = 0.1
    timeout_seconds: float = 3600.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("Population size must be at least 2")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")


@dataclass(frozen=True)
class MutationResult:
    """Result of a mutation operation.

    Args:
        code: The mutated code
        success: Whether mutation was successful
        explanation: Description of changes made
        confidence: LLM confidence in mutation (0.0-1.0)
        parent_ids: IDs of parent individuals
        mutation_type: Type of mutation applied
    """
    code: str
    success: bool
    explanation: str
    confidence: float = 0.5
    parent_ids: List[str] = field(default_factory=list)
    mutation_type: str = "unknown"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(self, 'confidence', max(0.0, min(1.0, self.confidence)))


@dataclass(frozen=True)
class EvaluatorResult:
    """Result of fitness evaluation.

    Args:
        fitness: Overall fitness score (higher is better)
        passed: Whether solution meets requirements
        metrics: Detailed metrics dict
        feedback: Textual feedback for improvement
        execution_time: Time to evaluate
        memory_usage: Memory consumed
    """
    fitness: float
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    execution_time: float = 0.0
    memory_usage: float = 0.0

    def __post_init__(self) -> None:
        if self.fitness < 0:
            raise ValueError("Fitness cannot be negative")


@dataclass(frozen=False)
class Individual:
    """An individual solution in the population.

    Args:
        id: Unique identifier (auto-generated if not provided)
        code: The code solution
        fitness: Fitness score (None if not evaluated)
        generation: Generation number created
        parent_ids: IDs of parent individuals
        metadata: Additional metadata
        created_at: Timestamp
    """
    id: str = field(default="")
    code: str = ""
    fitness: Optional[float] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique ID based on code content and timestamp."""
        content = f"{self.code}{self.created_at}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def update_fitness(self, fitness: float) -> None:
        """Update fitness score."""
        self.fitness = fitness

    def is_evaluated(self) -> bool:
        """Check if individual has been evaluated."""
        return self.fitness is not None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.id == other.id


class Population:
    """Manages a population of individuals."""

    def __init__(self, config: EvolutionConfig) -> None:
        """Initialize population with configuration.

        Args:
            config: Evolution configuration
        """
        self.config = config
        self.individuals: List[Individual] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

    def add(self, individual: Individual) -> None:
        """Add individual to population."""
        self.individuals.append(individual)

    def add_batch(self, individuals: List[Individual]) -> None:
        """Add multiple individuals."""
        self.individuals.extend(individuals)

    def get_best(self, count: int = 1) -> List[Individual]:
        """Get top individuals by fitness.

        Args:
            count: Number of individuals to return

        Returns:
            List of best individuals
        """
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]
        if not evaluated:
            return []
        sorted_individuals = sorted(evaluated, key=lambda x: x.fitness or 0, reverse=True)
        return sorted_individuals[:count]

    def get_elites(self) -> List[Individual]:
        """Get elite individuals for preservation."""
        return self.get_best(self.config.elitism_count)

    def tournament_select(self) -> Individual:
        """Select individual using tournament selection.

        Returns:
            Selected individual
        """
        import random
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]
        if not evaluated:
            if not self.individuals:
                raise ValueError("Population is empty")
            return random.choice(self.individuals)

        tournament_size = min(self.config.selection_pressure, len(evaluated))
        tournament = random.sample(evaluated, tournament_size)
        return max(tournament, key=lambda x: x.fitness or 0)

    def roulette_select(self) -> Individual:
        """Select individual using roulette wheel selection.

        Returns:
            Selected individual
        """
        import random
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]
        if not evaluated:
            if not self.individuals:
                raise ValueError("Population is empty")
            return random.choice(self.individuals)

        total_fitness = sum(ind.fitness or 0 for ind in evaluated)
        if total_fitness == 0:
            return random.choice(evaluated)

        pick = random.uniform(0, total_fitness)
        current = 0.0
        for ind in evaluated:
            current += ind.fitness or 0
            if current >= pick:
                return ind
        return evaluated[-1]

    def prune(self) -> None:
        """Remove excess individuals to maintain population size."""
        if len(self.individuals) <= self.config.population_size:
            return

        # Keep elites and remove lowest fitness individuals
        elites = set(self.get_elites())
        non_elites = [ind for ind in self.individuals if ind not in elites]

        evaluated = [ind for ind in non_elites if ind.is_evaluated()]
        unevaluated = [ind for ind in non_elites if not ind.is_evaluated()]

        # Sort evaluated by fitness (ascending for removal)
        evaluated.sort(key=lambda x: x.fitness or 0)

        # Calculate how many to keep
        slots_available = self.config.population_size - len(elites) - len(unevaluated)
        keep_evaluated = evaluated[:max(0, slots_available)]

        self.individuals = list(elites) + unevaluated + keep_evaluated

    def calculate_diversity(self) -> float:
        """Calculate population diversity based on code similarity.

        Returns:
            Diversity score (0.0-1.0)
        """
        if len(self.individuals) < 2:
            return 0.0

        # Use hash of code to measure diversity
        unique_hashes = set(hashlib.sha256(ind.code.encode()).hexdigest()[:8]
                           for ind in self.individuals)
        return len(unique_hashes) / len(self.individuals)

    def update_statistics(self) -> None:
        """Update fitness statistics for current generation."""
        evaluated = [ind for ind in self.individuals if ind.is_evaluated()]
        if evaluated:
            fitnesses = [ind.fitness or 0 for ind in evaluated]
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        else:
            self.best_fitness_history.append(0.0)
            self.avg_fitness_history.append(0.0)

    def is_converged(self, threshold: float = 0.01, window: int = 10) -> bool:
        """Check if population has converged.

        Args:
            threshold: Fitness change threshold
            window: Number of generations to check

        Returns:
            True if converged
        """
        if len(self.best_fitness_history) < window:
            return False

        recent = self.best_fitness_history[-window:]
        if max(recent) == 0:
            return False

        relative_change = (max(recent) - min(recent)) / max(recent)
        return relative_change < threshold

    def size(self) -> int:
        """Get current population size."""
        return len(self.individuals)

    def evaluated_count(self) -> int:
        """Get number of evaluated individuals."""
        return len([ind for ind in self.individuals if ind.is_evaluated()])
