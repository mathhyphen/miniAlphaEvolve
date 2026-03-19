"""Evolutionary algorithm implementation for AlphaEvolve."""

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import json

from alphaevolve.core.data_structures import (
    EvolutionConfig,
    Individual,
    Population,
    MutationResult,
)
from alphaevolve.core.mutation import MutationEngine
from alphaevolve.core.evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class Evolution:
    """Main evolutionary algorithm orchestrator."""

    def __init__(
        self,
        config: EvolutionConfig,
        mutation_engine: MutationEngine,
        evaluator: BaseEvaluator,
        seed_code: str,
        seed_name: str = "solution"
    ) -> None:
        """Initialize evolution process.

        Args:
            config: Evolution configuration
            mutation_engine: Engine for generating mutations
            evaluator: Evaluator for fitness assessment
            seed_code: Initial seed solution
            seed_name: Name of the function in seed code
        """
        self.config = config
        self.mutation_engine = mutation_engine
        self.evaluator = evaluator
        self.seed_code = seed_code
        self.seed_name = seed_name

        self.population = Population(config)
        self.generation = 0
        self.start_time = time.time()
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict[str, Any]] = []

        # Set random seed if provided
        if config.seed is not None:
            random.seed(config.seed)

    def initialize(self) -> None:
        """Initialize population with seed and random variations."""
        logger.info("Initializing population with seed solution")

        # Add seed individual
        seed_individual = Individual(
            code=self.seed_code,
            generation=0
        )

        # Evaluate seed
        result = self.evaluator.evaluate(self.seed_code)
        seed_individual.update_fitness(result.fitness)
        self.population.add(seed_individual)
        self.best_individual = seed_individual

        logger.info(f"Seed fitness: {result.fitness:.2f}")

        # Create initial variations through mutation
        variations_needed = self.config.population_size // 2
        logger.info(f"Creating {variations_needed} initial variations")

        for i in range(variations_needed):
            mutation_result = self.mutation_engine.mutate(
                code=self.seed_code,
                feedback=result.feedback,
                parent_ids=[seed_individual.id]
            )

            if mutation_result.success:
                individual = Individual(
                    code=mutation_result.code,
                    generation=0,
                    parent_ids=[seed_individual.id],
                    metadata={"mutation_type": mutation_result.mutation_type}
                )

                # Evaluate
                eval_result = self.evaluator.evaluate(mutation_result.code)
                individual.update_fitness(eval_result.fitness)
                self.population.add(individual)

                # Update best
                if self.best_individual is None or individual.fitness > self.best_individual.fitness:
                    self.best_individual = individual

        logger.info(f"Population initialized with {self.population.size()} individuals")

    def evolve(self) -> Individual:
        """Run the evolutionary process.

        Returns:
            Best individual found
        """
        if self.population.size() == 0:
            self.initialize()

        logger.info("Starting evolution")
        logger.info(f"Config: pop_size={self.config.population_size}, "
                   f"generations={self.config.max_generations}")

        for generation in range(1, self.config.max_generations + 1):
            self.generation = generation

            # Check timeout
            elapsed = time.time() - self.start_time
            if elapsed > self.config.timeout_seconds:
                logger.info(f"Timeout reached after {elapsed:.1f}s")
                break

            # Run one generation
            self._run_generation()

            # Log progress
            best = self.population.get_best(1)[0] if self.population.get_best(1) else None
            if best:
                logger.info(f"Gen {generation}: Best fitness={best.fitness:.2f}, "
                           f"Pop size={self.population.size()}, "
                           f"Evaluated={self.population.evaluated_count()}")

            # Check convergence
            if self.population.is_converged():
                logger.info(f"Converged at generation {generation}")
                break

            # Check diversity
            diversity = self.population.calculate_diversity()
            if diversity < self.config.diversity_threshold:
                logger.warning(f"Low diversity detected: {diversity:.2f}")

        # Return best individual
        best_final = self.population.get_best(1)
        if best_final:
            self.best_individual = best_final[0]
            logger.info(f"Evolution complete. Best fitness: {self.best_individual.fitness:.2f}")
            return self.best_individual
        else:
            raise RuntimeError("No valid solution found")

    def _run_generation(self) -> None:
        """Execute one generation of evolution."""
        offspring: List[Individual] = []

        # Calculate number of offspring to generate
        target_size = self.config.population_size
        current_size = self.population.size()
        num_offspring = max(1, target_size - current_size + self.config.elitism_count)

        # Generate offspring
        for _ in range(num_offspring):
            # Select parent
            try:
                parent = self.population.tournament_select()
            except ValueError:
                break

            # Get feedback from last evaluation
            feedback = ""
            if parent.metadata.get("last_feedback"):
                feedback = parent.metadata["last_feedback"]

            # Mutate
            mutation_result = self.mutation_engine.mutate(
                code=parent.code,
                feedback=feedback,
                parent_ids=[parent.id]
            )

            if mutation_result.success:
                # Create new individual
                individual = Individual(
                    code=mutation_result.code,
                    generation=self.generation,
                    parent_ids=[parent.id],
                    metadata={
                        "mutation_type": mutation_result.mutation_type,
                        "mutation_confidence": mutation_result.confidence
                    }
                )

                # Evaluate
                eval_result = self.evaluator.evaluate(mutation_result.code)
                individual.update_fitness(eval_result.fitness)
                individual.metadata["last_feedback"] = eval_result.feedback

                offspring.append(individual)

        # Add offspring to population
        self.population.add_batch(offspring)

        # Update statistics
        self.population.update_statistics()

        # Prune to maintain size
        self.population.prune()

        # Record history
        best = self.population.get_best(1)[0] if self.population.get_best(1) else None
        self.history.append({
            "generation": self.generation,
            "population_size": self.population.size(),
            "best_fitness": best.fitness if best else 0.0,
            "avg_fitness": self.population.avg_fitness_history[-1] if self.population.avg_fitness_history else 0.0,
            "diversity": self.population.calculate_diversity()
        })

    def get_history(self) -> List[Dict[str, Any]]:
        """Get evolution history.

        Returns:
            List of generation statistics
        """
        return self.history

    def save_checkpoint(self, path: str) -> None:
        """Save evolution state to checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "generation": self.generation,
            "config": {
                "population_size": self.config.population_size,
                "mutation_rate": self.config.mutation_rate,
                "max_generations": self.config.max_generations,
            },
            "best_individual": {
                "id": self.best_individual.id if self.best_individual else None,
                "code": self.best_individual.code if self.best_individual else None,
                "fitness": self.best_individual.fitness if self.best_individual else None,
            },
            "seed_code": self.seed_code,
            "seed_name": self.seed_name,
            "history": self.history,
            "population": [
                {
                    "id": ind.id,
                    "code": ind.code,
                    "fitness": ind.fitness,
                    "generation": ind.generation
                }
                for ind in self.population.individuals
            ]
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.info(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        mutation_engine: MutationEngine,
        evaluator: BaseEvaluator
    ) -> "Evolution":
        """Load evolution from checkpoint.

        Args:
            path: Path to checkpoint file
            mutation_engine: Mutation engine instance
            evaluator: Evaluator instance

        Returns:
            Evolution instance
        """
        with open(path, 'r') as f:
            checkpoint = json.load(f)

        config = EvolutionConfig(**checkpoint["config"])

        # Create evolution with dummy seed
        evolution = cls(
            config=config,
            mutation_engine=mutation_engine,
            evaluator=evaluator,
            seed_code=""
        )

        # Restore state
        evolution.generation = checkpoint["generation"]
        evolution.history = checkpoint["history"]

        # Restore population
        for ind_data in checkpoint["population"]:
            ind = Individual(
                id=ind_data["id"],
                code=ind_data["code"],
                fitness=ind_data["fitness"],
                generation=ind_data["generation"]
            )
            evolution.population.add(ind)

        # Restore seed info
        evolution.seed_code = checkpoint.get("seed_code", "")
        evolution.seed_name = checkpoint.get("seed_name", "solution")

        # Restore best individual
        best_data = checkpoint["best_individual"]
        if best_data["id"]:
            evolution.best_individual = Individual(
                id=best_data["id"],
                code=best_data["code"],
                fitness=best_data["fitness"],
                generation=checkpoint["generation"]
            )

        logger.info(f"Checkpoint loaded from {path}")
        return evolution
