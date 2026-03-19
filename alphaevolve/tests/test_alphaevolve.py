"""Tests for AlphaEvolve core components."""

import pytest
from alphaevolve.core.data_structures import (
    EvolutionConfig,
    Individual,
    Population,
    MutationResult,
    EvaluatorResult,
)
from alphaevolve.core.mutation import MutationEngine
from alphaevolve.core.evaluator import UnitTestEvaluator


class TestEvolutionConfig:
    """Test EvolutionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EvolutionConfig()
        assert config.population_size == 50
        assert config.mutation_rate == 0.7
        assert config.max_generations == 100

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            EvolutionConfig(population_size=1)

        with pytest.raises(ValueError):
            EvolutionConfig(mutation_rate=1.5)

    def test_custom_values(self):
        """Test custom configuration."""
        config = EvolutionConfig(
            population_size=20,
            mutation_rate=0.5,
            seed=42
        )
        assert config.population_size == 20
        assert config.seed == 42


class TestIndividual:
    """Test Individual class."""

    def test_creation(self):
        """Test individual creation."""
        code = "def foo(): return 42"
        ind = Individual(code=code, generation=0)
        assert ind.code == code
        assert ind.fitness is None
        assert ind.generation == 0

    def test_evaluation(self):
        """Test fitness update."""
        ind = Individual(code="def test(): pass", generation=0)
        assert not ind.is_evaluated()

        ind.update_fitness(85.0)
        assert ind.is_evaluated()
        assert ind.fitness == 85.0

    def test_parent_tracking(self):
        """Test parent ID tracking."""
        parent = Individual(code="def parent(): pass", generation=0)
        child = Individual(
            code="def child(): pass",
            generation=1,
            parent_ids=[parent.id]
        )
        assert child.parent_ids == [parent.id]


class TestPopulation:
    """Test Population class."""

    def test_initialization(self):
        """Test population initialization."""
        config = EvolutionConfig(population_size=10)
        pop = Population(config)
        assert pop.size() == 0

    def test_add_individual(self):
        """Test adding individuals."""
        config = EvolutionConfig()
        pop = Population(config)
        ind = Individual(code="def test(): pass", generation=0)
        pop.add(ind)
        assert pop.size() == 1

    def test_selection(self):
        """Test tournament selection."""
        config = EvolutionConfig(population_size=10)
        pop = Population(config)

        # Add evaluated individuals
        for i in range(5):
            ind = Individual(code=f"def f{i}(): pass", generation=0)
            ind.update_fitness(float(i * 10))
            pop.add(ind)

        selected = pop.tournament_select()
        assert selected is not None

    def test_elites(self):
        """Test elite selection."""
        config = EvolutionConfig(elitism_count=2)
        pop = Population(config)

        for i in range(5):
            ind = Individual(code=f"def f{i}(): pass", generation=0)
            ind.update_fitness(float(i))
            pop.add(ind)

        elites = pop.get_elites()
        assert len(elites) == 2
        assert elites[0].fitness == 4.0
        assert elites[1].fitness == 3.0

    def test_diversity(self):
        """Test diversity calculation."""
        config = EvolutionConfig()
        pop = Population(config)

        # Identical individuals = low diversity
        for _ in range(3):
            ind = Individual(code="def same(): pass", generation=0)
            pop.add(ind)

        diversity = pop.calculate_diversity()
        assert diversity == 1 / 3  # 1 unique out of 3


class TestMutationEngine:
    """Test MutationEngine."""

    def test_initialization(self):
        """Test engine creation."""
        engine = MutationEngine()
        assert engine is not None

    def test_mutation(self):
        """Test code mutation."""
        engine = MutationEngine()
        code = "def add(a, b): return a + b"

        result = engine.mutate(code)
        assert isinstance(result, MutationResult)
        assert result.code is not None

    def test_batch_mutation(self):
        """Test batch mutation."""
        engine = MutationEngine()
        code = "def test(): pass"

        results = engine.mutate_batch(code, count=3)
        assert len(results) == 3


class TestUnitTestEvaluator:
    """Test UnitTestEvaluator."""

    def test_correct_solution(self):
        """Test with correct solution."""
        test_cases = [
            ((2, 3), 5),
            ((0, 0), 0),
            ((-1, 1), 0),
        ]

        evaluator = UnitTestEvaluator(
            test_cases=test_cases,
            function_name="add"
        )

        code = """
def add(a, b):
    return a + b
"""
        result = evaluator.evaluate(code)
        assert result.passed is True
        assert result.fitness == 100.0

    def test_incorrect_solution(self):
        """Test with incorrect solution."""
        test_cases = [
            ((2, 3), 5),
            ((1, 1), 2),
        ]

        evaluator = UnitTestEvaluator(
            test_cases=test_cases,
            function_name="add"
        )

        code = """
def add(a, b):
    return a * b  # Wrong!
"""
        result = evaluator.evaluate(code)
        assert result.passed is False
        assert result.fitness < 100.0


class TestMutationResult:
    """Test MutationResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = MutationResult(
            code="def test(): pass",
            success=True,
            explanation="Added optimization",
            confidence=0.9
        )
        assert result.success is True
        assert result.confidence == 0.9


class TestEvaluatorResult:
    """Test EvaluatorResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = EvaluatorResult(
            fitness=85.0,
            passed=True,
            metrics={"accuracy": 0.85}
        )
        assert result.fitness == 85.0
        assert result.passed is True

    def test_negative_fitness(self):
        """Test negative fitness validation."""
        with pytest.raises(ValueError):
            EvaluatorResult(fitness=-1.0, passed=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
