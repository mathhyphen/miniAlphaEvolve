"""
AlphaEvolve Example: Sorting Algorithm Evolution

This example demonstrates how AlphaEvolve can evolve a sorting algorithm
from a naive bubble sort implementation to a more efficient solution.
"""

import logging
from alphaevolve import (
    EvolutionConfig,
    MutationEngine,
    UnitTestEvaluator,
    Evolution,
    LLMConfig,
    LLMEnsemble,
    Priority,
    RoutingStrategy,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Seed code: Naive bubble sort
SEED_CODE = """
def sort_list(items):
    \"\"\"Sort a list of numbers.\"\"\"
    result = list(items)
    n = len(result)
    for i in range(n):
        for j in range(i + 1, n):
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]
    return result
"""


def create_test_cases():
    """Create test cases for sorting."""
    return [
        # Basic cases
        (([3, 1, 2],), [1, 2, 3]),
        (([5, 4, 3, 2, 1],), [1, 2, 3, 4, 5]),
        (([1],), [1]),
        (([]), []),
        # Negative numbers
        (([-1, -3, 2],), [-3, -1, 2]),
        # Duplicates
        (([3, 1, 3, 1, 2],), [1, 1, 2, 3, 3]),
    ]


def run_evolution(use_llm: bool = False):
    """Run the evolution process.

    Args:
        use_llm: If True, use LLM ensemble for mutations
    """
    logger.info("=" * 60)
    logger.info("AlphaEvolve: Sorting Algorithm Evolution")
    logger.info("=" * 60)

    # Configure evolution
    config = EvolutionConfig(
        population_size=10,
        max_generations=5,
        elitism_count=2,
        mutation_rate=0.7,
        seed=42,
    )

    # Create evaluator
    test_cases = create_test_cases()
    evaluator = UnitTestEvaluator(
        test_cases=test_cases,
        function_name="sort_list",
        partial_credit=True,
    )

    # Create mutation engine
    engine = MutationEngine()

    # Optionally configure LLM ensemble
    if use_llm:
        logger.info("Configuring LLM ensemble...")
        try:
            llm_config = LLMConfig(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                temperature=0.7,
            )
            ensemble = LLMEnsemble(routing_strategy=RoutingStrategy.PRIORITY_BASED)
            # Note: This requires ANTHROPIC_API_KEY to be set
            # ensemble.register_client("main", AnthropicClient(llm_config))
            # engine.set_llm_ensemble(ensemble)
            logger.info("LLM ensemble configured (requires API key)")
        except Exception as e:
            logger.warning(f"LLM setup failed, using fallback: {e}")

    # Create evolution instance
    evolution = Evolution(
        config=config,
        mutation_engine=engine,
        evaluator=evaluator,
        seed_code=SEED_CODE,
    )

    # Run evolution
    logger.info("Starting evolution...")
    logger.info(f"Population size: {config.population_size}")
    logger.info(f"Max generations: {config.max_generations}")
    logger.info(f"Elitism count: {config.elitism_count}")

    best = evolution.evolve()

    # Report results
    logger.info("=" * 60)
    logger.info("Evolution Complete!")
    logger.info("=" * 60)
    logger.info(f"Best fitness: {best.fitness:.2f}")
    logger.info(f"Best code:\n{best.code}")

    # Show evolution history
    history = evolution.get_history()
    if history:
        logger.info("\nEvolution History:")
        logger.info("-" * 40)
        for gen in history[-3:]:  # Show last 3 generations
            logger.info(
                f"Gen {gen['generation']}: "
                f"Best={gen['best_fitness']:.2f}, "
                f"Avg={gen['avg_fitness']:.2f}, "
                f"Pop={gen['population_size']}"
            )

    return best


def demo_with_archive():
    """Demonstrate MAP-Elites archive usage."""
    from alphaevolve import (
        ProgramArchive,
        ArchiveConfig,
        CodeComplexityFeature,
        Individual,
    )

    logger.info("\n" + "=" * 60)
    logger.info("MAP-Elites Archive Demo")
    logger.info("=" * 60)

    # Create archive with complexity feature
    config = ArchiveConfig(
        feature_dimensions=[CodeComplexityFeature(n_bins=10)],
        bins_per_dimension=10,
    )
    archive = ProgramArchive(config)

    # Add some individuals
    codes = [
        "def f(x): return x",  # Simple
        "def f(x):\n    if x > 0:\n        return x\n    return -x",  # Medium
        "def f(x):\n    if x > 0:\n        if x > 10:\n            return x * 2\n        return x\n    return -x",  # Complex
    ]

    for i, code in enumerate(codes):
        ind = Individual(code=code, generation=i)
        ind.update_fitness(float((i + 1) * 20))
        added = archive.add(ind)
        logger.info(f"Individual {i+1}: Added={added}, Fitness={ind.fitness}")

    # Show stats
    stats = archive.get_stats()
    logger.info(f"\nArchive Statistics:")
    logger.info(f"  Fill rate: {stats.fill_rate:.2%}")
    logger.info(f"  Unique solutions: {stats.unique_solutions}")
    logger.info(f"  Best fitness: {stats.best_fitness:.2f}")
    logger.info(f"  Avg fitness: {stats.avg_fitness:.2f}")

    return archive


def demo_evaluation_pipeline():
    """Demonstrate multi-stage evaluation pipeline."""
    from alphaevolve import (
        EvaluationPipeline,
        SyntaxCheckStage,
        BasicTestStage,
        EdgeCaseStage,
        TestCaseGenerator,
        Difficulty,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Pipeline Demo")
    logger.info("=" * 60)

    # Generate test cases
    test_gen = TestCaseGenerator(seed=42)
    basic_tests = test_gen.generate("func(a, b)", Difficulty.BASIC, count=3)
    edge_tests = test_gen.generate("func(a, b)", Difficulty.EDGE, count=3)

    # Create pipeline
    pipeline = EvaluationPipeline(
        stages=[
            SyntaxCheckStage(),
            BasicTestStage(
                test_cases=[t.to_tuple() for t in basic_tests],
                function_name="func",
            ),
            EdgeCaseStage(
                edge_cases=[t.to_tuple() for t in edge_tests],
                function_name="func",
            ),
        ],
        early_terminate=True,
    )

    # Test with good code
    logger.info("Testing good code...")
    good_code = "def func(a, b): return a + b"
    result = pipeline.evaluate(good_code)
    logger.info(f"  Passed: {result.passed}")
    logger.info(f"  Fitness: {result.fitness:.2f}")
    logger.info(f"  Feedback: {result.feedback[:100]}...")

    # Test with bad code
    logger.info("\nTesting bad code...")
    bad_code = "def func(a, b): return"  # Returns None
    result = pipeline.evaluate(bad_code)
    logger.info(f"  Passed: {result.passed}")
    logger.info(f"  Fitness: {result.fitness:.2f}")

    return pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaEvolve Sorting Example")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM ensemble for mutations (requires API key)"
    )
    parser.add_argument(
        "--demo-archive",
        action="store_true",
        help="Run MAP-Elites archive demo"
    )
    parser.add_argument(
        "--demo-pipeline",
        action="store_true",
        help="Run evaluation pipeline demo"
    )

    args = parser.parse_args()

    if args.demo_archive:
        demo_with_archive()
    elif args.demo_pipeline:
        demo_evaluation_pipeline()
    else:
        run_evolution(use_llm=args.use_llm)
