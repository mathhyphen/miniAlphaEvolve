"""AlphaEvolve: An evolutionary coding agent for algorithm discovery.

AlphaEvolve is an evolutionary framework inspired by Google DeepMind's
AlphaDev that uses Large Language Models (LLMs) to iteratively improve
algorithms through quality-diversity search.
"""

__version__ = "0.2.0"
__author__ = "AlphaEvolve Team"

# Core data structures
from alphaevolve.core.data_structures import (
    Individual,
    Population,
    EvolutionConfig,
    MutationResult,
    EvaluatorResult,
)

# Evolution engine
from alphaevolve.core.evolution import Evolution

# Mutation engine
from alphaevolve.core.mutation import MutationEngine, MutationEngineConfig

# Evaluators
from alphaevolve.core.evaluator import (
    BaseEvaluator,
    UnitTestEvaluator,
    PerformanceEvaluator,
    CompositeEvaluator,
    CorrectnessEvaluator,
)

# LLM integration
from alphaevolve.llm.config import LLMConfig
from alphaevolve.llm.client import BaseLLMClient, LLMResponse
from alphaevolve.llm.ensemble import LLMEnsemble, Priority, RoutingStrategy
from alphaevolve.llm.providers.anthropic import AnthropicClient

# Evolution archive (MAP-Elites)
from alphaevolve.evolution.archive import ProgramArchive, ArchiveConfig, ArchiveStats
from alphaevolve.evolution.features import (
    FeatureDimension,
    CodeComplexityFeature,
    PerformanceFeature,
    CodeSizeFeature,
)

# Evaluation pipeline
from alphaevolve.evaluation.pipeline import EvaluationPipeline
from alphaevolve.evaluation.stages import (
    EvaluationStage,
    SyntaxCheckStage,
    BasicTestStage,
    EdgeCaseStage,
    PerformanceStage,
)
from alphaevolve.evaluation.test_generation import (
    TestCaseGenerator,
    TestCase,
    Difficulty,
)

# Configuration system (Hydra is optional)
from alphaevolve.config.config import (
    AlphaEvolveConfig,
    ExperimentConfig,
    EvolutionConfig as EvoConfig,
    MutationConfig,
    EvaluationConfig,
    ArchiveConfig,
    LoggingConfig,
    CheckpointConfig,
    HYDRA_AVAILABLE,
)

# load_config is only available when Hydra is installed
if HYDRA_AVAILABLE:
    from alphaevolve.config.config import load_config

# Utilities
from alphaevolve.utils import (
    ExperimentLogger,
    CheckpointManager,
    create_logger,
    resume_from_checkpoint,
)

__all__ = [
    # Version
    "__version__",
    "__author__",

    # Core data structures
    "Individual",
    "Population",
    "EvolutionConfig",
    "MutationResult",
    "EvaluatorResult",

    # Evolution engine
    "Evolution",

    # Mutation engine
    "MutationEngine",
    "MutationEngineConfig",

    # Evaluators
    "BaseEvaluator",
    "UnitTestEvaluator",
    "PerformanceEvaluator",
    "CompositeEvaluator",
    "CorrectnessEvaluator",

    # LLM integration
    "LLMConfig",
    "BaseLLMClient",
    "LLMResponse",
    "LLMEnsemble",
    "Priority",
    "RoutingStrategy",
    "AnthropicClient",

    # Evolution archive (MAP-Elites)
    "ProgramArchive",
    "ArchiveConfig",
    "ArchiveStats",
    "FeatureDimension",
    "CodeComplexityFeature",
    "PerformanceFeature",
    "CodeSizeFeature",

    # Evaluation pipeline
    "EvaluationPipeline",
    "EvaluationStage",
    "SyntaxCheckStage",
    "BasicTestStage",
    "EdgeCaseStage",
    "PerformanceStage",

    # Test generation
    "TestCaseGenerator",
    "TestCase",
    "Difficulty",

    # Configuration system
    "AlphaEvolveConfig",
    "ExperimentConfig",
    "EvoConfig",
    "MutationConfig",
    "EvaluationConfig",
    "ArchiveConfig",
    "LoggingConfig",
    "CheckpointConfig",
    "HYDRA_AVAILABLE",

    # Utilities
    "ExperimentLogger",
    "CheckpointManager",
    "create_logger",
    "resume_from_checkpoint",
]

# Add load_config only when Hydra is available
if HYDRA_AVAILABLE:
    __all__.append("load_config")
