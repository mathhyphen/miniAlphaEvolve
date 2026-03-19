"""Comprehensive tests for AlphaEvolve components."""

import pytest
import unittest
from typing import List, Tuple

# Test imports from all modules
from alphaevolve.core.data_structures import (
    Individual,
    Population,
    EvolutionConfig,
    MutationResult,
    EvaluatorResult,
)
from alphaevolve.core.evolution import Evolution
from alphaevolve.core.mutation import MutationEngine, MutationEngineConfig
from alphaevolve.core.evaluator import UnitTestEvaluator


class TestDataStructures(unittest.TestCase):
    """Test core data structures."""

    def test_individual_creation(self) -> None:
        """Test Individual creation."""
        code = "def add(a, b): return a + b"
        ind = Individual(code=code, generation=0)

        self.assertEqual(ind.code, code)
        self.assertIsNone(ind.fitness)
        self.assertEqual(ind.generation, 0)
        self.assertTrue(len(ind.id) > 0)

    def test_individual_update_fitness(self) -> None:
        """Test fitness update."""
        ind = Individual(code="test")
        ind.update_fitness(85.0)

        self.assertEqual(ind.fitness, 85.0)
        self.assertTrue(ind.is_evaluated())

    def test_evolution_config_validation(self) -> None:
        """Test EvolutionConfig validation."""
        # Valid config
        config = EvolutionConfig(population_size=10)
        self.assertEqual(config.population_size, 10)

        # Invalid: population too small
        with self.assertRaises(ValueError):
            EvolutionConfig(population_size=1)

        # Invalid: mutation rate out of range
        with self.assertRaises(ValueError):
            EvolutionConfig(mutation_rate=1.5)

    def test_population_operations(self) -> None:
        """Test Population operations."""
        config = EvolutionConfig(population_size=10, selection_pressure=3)
        pop = Population(config)

        # Add individuals
        for i in range(5):
            ind = Individual(code=f"code_{i}", generation=i)
            ind.update_fitness(float(i * 10))
            pop.add(ind)

        self.assertEqual(pop.size(), 5)
        self.assertEqual(pop.evaluated_count(), 5)

        # Get best
        best = pop.get_best(1)
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0].fitness, 40.0)

        # Tournament selection
        selected = pop.tournament_select()
        self.assertIsNotNone(selected)

        # Diversity calculation
        diversity = pop.calculate_diversity()
        self.assertGreater(diversity, 0)


class TestMutationEngine(unittest.TestCase):
    """Test MutationEngine."""

    def test_fallback_mutation(self) -> None:
        """Test fallback mutation (no LLM)."""
        engine = MutationEngine()

        code = """def add(a, b):
    for i in range(len(a)):
        pass
    return a + b"""

        result = engine.mutate(
            code=code,
            mutation_type="improve_performance",
            feedback="Optimize for speed",
        )

        # Fallback should still produce output
        self.assertIsInstance(result.code, str)
        self.assertIsInstance(result.success, bool)
        self.assertEqual(result.mutation_type, "improve_performance")

    def test_batch_mutation(self) -> None:
        """Test batch mutation."""
        engine = MutationEngine()
        code = "def test(): return 1"

        results = engine.mutate_batch(code, count=3)

        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], MutationResult)

    def test_set_llm_ensemble(self) -> None:
        """Test setting LLM ensemble."""
        engine = MutationEngine()

        # Test that method exists
        self.assertTrue(hasattr(engine, 'set_llm_ensemble'))


class TestEvaluator(unittest.TestCase):
    """Test evaluators."""

    def test_unit_test_evaluator(self) -> None:
        """Test UnitTestEvaluator."""
        test_cases: List[Tuple[tuple, int]] = [
            ((2, 3), 5),
            ((0, 0), 0),
            ((-1, 1), 0),
        ]

        evaluator = UnitTestEvaluator(
            test_cases=test_cases,
            function_name="add",
        )

        # Test correct implementation
        correct_code = "def add(a, b): return a + b"
        result = evaluator.evaluate(correct_code)

        self.assertEqual(result.fitness, 100.0)
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.execution_time, 0)

        # Test incorrect implementation
        incorrect_code = "def add(a, b): return a * b"
        result = evaluator.evaluate(incorrect_code)

        self.assertLess(result.fitness, 100.0)

    def test_syntax_error_handling(self) -> None:
        """Test syntax error handling."""
        test_cases = [((1, 2), 3)]
        evaluator = UnitTestEvaluator(
            test_cases=test_cases,
            function_name="func",
        )

        # Invalid syntax
        bad_code = "def func(a, b): return a + b:"  # Extra colon
        result = evaluator.evaluate(bad_code)

        self.assertEqual(result.fitness, 0.0)
        self.assertFalse(result.passed)
        self.assertIn("syntax", result.feedback.lower())


class TestLLMComponents(unittest.TestCase):
    """Test LLM integration components."""

    def test_llm_config(self) -> None:
        """Test LLMConfig creation."""
        from alphaevolve.llm.config import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=4096,
        )

        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.temperature, 0.7)

        # Test validation
        with self.assertRaises(ValueError):
            LLMConfig(provider="test", model="test", temperature=3.0)

    def test_llm_response(self) -> None:
        """Test LLMResponse dataclass."""
        from alphaevolve.llm.client import LLMResponse

        response = LLMResponse(
            content="test content",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

        self.assertEqual(response.content, "test content")
        self.assertEqual(response.total_tokens, 30)


class TestArchiveComponents(unittest.TestCase):
    """Test MAP-Elites archive components."""

    def test_feature_dimension(self) -> None:
        """Test feature dimension extraction."""
        from alphaevolve.evolution.features import CodeComplexityFeature

        feature = CodeComplexityFeature(n_bins=10)

        # Simple code
        simple_code = "def add(a, b): return a + b"
        complexity = feature.extract(simple_code)
        self.assertIsInstance(complexity, (int, float))

        # Complex code
        complex_code = """
def process(items):
    for item in items:
        if item > 0:
            for i in range(item):
                if i % 2 == 0:
                    print(i)
                elif i % 3 == 0:
                    print(i * 3)
        else:
            continue
"""
        complexity_high = feature.extract(complex_code)
        self.assertGreater(complexity_high, complexity)

    def test_discretize(self) -> None:
        """Test feature discretization."""
        from alphaevolve.evolution.features import CodeComplexityFeature

        feature = CodeComplexityFeature(n_bins=10, max_complexity=50)

        # Test bin boundaries
        bin_0 = feature.discretize(0)
        bin_high = feature.discretize(45)

        self.assertEqual(bin_0, 0)
        self.assertGreater(bin_high, bin_0)

    def test_archive_config(self) -> None:
        """Test ArchiveConfig creation."""
        from alphaevolve.evolution.archive import ArchiveConfig
        from alphaevolve.evolution.features import CodeComplexityFeature

        features = [CodeComplexityFeature()]
        config = ArchiveConfig(
            feature_dimensions=features,
            bins_per_dimension=10,
        )

        self.assertEqual(config.bins_per_dimension, 10)
        self.assertEqual(len(config.feature_dimensions), 1)


class TestEvaluationPipeline(unittest.TestCase):
    """Test evaluation pipeline."""

    def test_syntax_check_stage(self) -> None:
        """Test syntax check stage."""
        from alphaevolve.evaluation.stages import SyntaxCheckStage

        stage = SyntaxCheckStage()

        # Valid code
        result = stage.evaluate("def test(): return 1")
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 100.0)

        # Invalid code
        result = stage.evaluate("def test(: return 1")  # Missing )
        self.assertFalse(result.passed)

    def test_basic_test_stage(self) -> None:
        """Test basic test stage."""
        from alphaevolve.evaluation.stages import BasicTestStage

        stage = BasicTestStage(
            test_cases=[((2, 3), 5), ((0, 0), 0)],
            function_name="add",
        )

        # Correct implementation
        result = stage.evaluate("def add(a, b): return a + b")
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 100.0)

        # Incorrect implementation
        result = stage.evaluate("def add(a, b): return a * b")
        self.assertFalse(result.passed)
        self.assertLess(result.score, 100.0)

    def test_evaluation_pipeline(self) -> None:
        """Test full evaluation pipeline."""
        from alphaevolve.evaluation.pipeline import EvaluationPipeline
        from alphaevolve.evaluation.stages import (
            SyntaxCheckStage,
            BasicTestStage,
        )

        pipeline = EvaluationPipeline(
            stages=[
                SyntaxCheckStage(),
                BasicTestStage(
                    test_cases=[((1, 1), 2)],
                    function_name="add",
                ),
            ],
            early_terminate=True,
        )

        # Good code
        result = pipeline.evaluate("def add(a, b): return a + b")
        self.assertTrue(result.passed)
        self.assertGreater(result.fitness, 0)

        # Bad syntax - should early terminate
        result = pipeline.evaluate("def add(a, b): return")
        self.assertFalse(result.passed)


class TestTestCaseGenerator(unittest.TestCase):
    """Test test case generator."""

    def test_generate_basic_tests(self) -> None:
        """Test basic test generation."""
        from alphaevolve.evaluation.test_generation import TestCaseGenerator, Difficulty

        generator = TestCaseGenerator(seed=42)
        tests = generator.generate("add(a, b)", Difficulty.BASIC, count=5)

        self.assertEqual(len(tests), 5)
        for test in tests:
            self.assertEqual(test.difficulty, Difficulty.BASIC)
            self.assertEqual(len(test.inputs), 2)

    def test_generate_edge_tests(self) -> None:
        """Test edge case test generation."""
        from alphaevolve.evaluation.test_generation import TestCaseGenerator, Difficulty

        generator = TestCaseGenerator(seed=42)
        tests = generator.generate("divide(a, b)", Difficulty.EDGE, count=5)

        # Note: Generator adds an extra "all zeros" test, so expect at least 5
        self.assertGreaterEqual(len(tests), 5)
        for test in tests:
            self.assertEqual(test.difficulty, Difficulty.EDGE)

    def test_difficulty_levels(self) -> None:
        """Test different difficulty levels."""
        from alphaevolve.evaluation.test_generation import TestCaseGenerator, Difficulty

        generator = TestCaseGenerator(seed=42)

        basic = generator.generate("func(a, b)", Difficulty.BASIC, count=3)
        stress = generator.generate("func(a, b)", Difficulty.STRESS, count=3)

        # Stress tests should have larger inputs
        self.assertEqual(len(basic), 3)
        self.assertEqual(len(stress), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for full evolution pipeline."""

    def test_full_evolution_run(self) -> None:
        """Test a complete evolution run."""
        # Seed code: naive sorting
        seed_code = """
def sort_list(items):
    result = list(items)
    n = len(result)
    for i in range(n):
        for j in range(i + 1, n):
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]
    return result
"""

        # Test cases
        test_cases = [
            (([3, 1, 2],), [1, 2, 3]),
            (([5, 4, 3, 2, 1],), [1, 2, 3, 4, 5]),
            (([1],), [1]),
            (([]), []),
        ]

        # Configure evolution
        config = EvolutionConfig(
            population_size=6,
            max_generations=3,
            elitism_count=1,
            seed=42,
        )

        # Create components
        evaluator = UnitTestEvaluator(
            test_cases=test_cases,
            function_name="sort_list",
        )

        engine = MutationEngine()

        # Run evolution
        evolution = Evolution(
            config=config,
            mutation_engine=engine,
            evaluator=evaluator,
            seed_code=seed_code,
        )

        best = evolution.evolve()

        # Verify result
        self.assertIsNotNone(best)
        self.assertIsNotNone(best.fitness)
        self.assertGreaterEqual(best.fitness, 0)

        # Check history
        history = evolution.get_history()
        self.assertGreater(len(history), 0)


if __name__ == "__main__":
    unittest.main()
