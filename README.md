# AlphaEvolve

AlphaEvolve is an evolutionary coding agent for algorithm discovery, inspired by Google DeepMind's AlphaDev. It uses Large Language Models (LLMs) to iteratively improve algorithms through quality-diversity search.

## Features

- **Evolutionary Algorithm Framework**: Population-based search with selection, mutation, and elitism
- **LLM-Guided Mutations**: Intelligent code modifications using Claude or other LLM providers
- **MAP-Elites Archive**: Quality-diversity search maintaining diverse high-performing solutions
- **Multi-Stage Evaluation Pipeline**: Progressive evaluation with early termination
- **Test Case Generation**: Automatic test case generation with multiple difficulty levels
- **Hydra Configuration**: Flexible YAML-based experiment configuration
- **Experiment Logging**: Structured logging with automatic output management
- **Checkpoint Management**: Save and resume evolution experiments

## Installation

```bash
pip install alphaevolve
```

For full functionality including Hydra configuration:

```bash
pip install alphaevolve[hydra]
```

## Quick Start

```python
from alphaevolve import (
    Evolution,
    EvolutionConfig,
    MutationEngine,
    UnitTestEvaluator,
)

# Seed code
SEED_CODE = """
def sort_list(items):
    result = list(items)
    n = len(result)
    for i in range(n):
        for j in range(i + 1, n):
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]
    return result
"""

# Configure evolution
config = EvolutionConfig(
    population_size=10,
    max_generations=5,
    elitism_count=2,
    mutation_rate=0.7,
)

# Create evaluator
evaluator = UnitTestEvaluator(
    test_cases=[
        (([3, 1, 2],), [1, 2, 3]),
        (([5, 4, 3, 2, 1],), [1, 2, 3, 4, 5]),
    ],
    function_name="sort_list",
)

# Run evolution
evolution = Evolution(
    config=config,
    mutation_engine=MutationEngine(),
    evaluator=evaluator,
    seed_code=SEED_CODE,
)

best = evolution.evolve()
print(f"Best fitness: {best.fitness}")
print(f"Best code: {best.code}")
```

## Project Structure

```
alphaevolve/
├── core/                      # Core evolutionary components
│   ├── data_structures.py     # Individual, Population, Config classes
│   ├── evolution.py           # Evolution engine
│   ├── mutation.py            # Mutation engine with LLM integration
│   └── evaluator.py           # Base evaluator and implementations
├── llm/                       # LLM integration
│   ├── config.py              # LLMConfig dataclass
│   ├── client.py              # BaseLLMClient abstract class
│   ├── ensemble.py            # LLMEnsemble for multi-model routing
│   └── providers/
│       └── anthropic.py       # Anthropic Claude client
├── evolution/                 # Advanced evolution features
│   ├── archive.py             # MAP-Elites ProgramArchive
│   └── features.py            # Feature dimensions for archive
├── evaluation/                # Evaluation pipeline
│   ├── pipeline.py            # Multi-stage evaluation pipeline
│   ├── stages.py              # Evaluation stage implementations
│   └── test_generation/
│       └── generator_module.py # Test case generator
├── config/                    # Configuration system
│   ├── config.py              # Dataclass configurations
│   └── experiment.yaml        # Default experiment config
├── utils/                     # Utilities
│   ├── logger.py              # ExperimentLogger
│   └── checkpoint.py          # CheckpointManager
├── tests/                     # Test suite
│   ├── test_alphaevolve.py    # Core tests
│   ├── test_comprehensive.py  # Comprehensive tests
│   └── test_config_and_utils.py # Config and utils tests
└── examples/                  # Example scripts
    └── sorting_demo.py        # Sorting algorithm evolution demo
```

## Core Components

### Evolution Engine

The `Evolution` class manages the evolutionary search process:

```python
evolution = Evolution(
    config=EvolutionConfig(population_size=10, max_generations=5),
    mutation_engine=MutationEngine(),
    evaluator=UnitTestEvaluator(test_cases, function_name="func"),
    seed_code="def func(x): return x",
)
best = evolution.evolve()
```

### LLM Ensemble

Use multiple LLM providers with intelligent routing:

```python
from alphaevolve import LLMEnsemble, AnthropicClient, LLMConfig, RoutingStrategy

ensemble = LLMEnsemble(routing_strategy=RoutingStrategy.PRIORITY_BASED)
ensemble.register_client("claude", AnthropicClient(LLMConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
)))
```

### MAP-Elites Archive

Maintain a diverse archive of high-performing solutions:

```python
from alphaevolve import ProgramArchive, ArchiveConfig, CodeComplexityFeature

archive = ProgramArchive(ArchiveConfig(
    feature_dimensions=[CodeComplexityFeature(n_bins=10)],
    bins_per_dimension=10,
))
```

### Evaluation Pipeline

Multi-stage evaluation with early termination:

```python
from alphaevolve import EvaluationPipeline, SyntaxCheckStage, BasicTestStage

pipeline = EvaluationPipeline(stages=[
    SyntaxCheckStage(),
    BasicTestStage(test_cases, function_name="func"),
], early_terminate=True)
```

### Test Case Generator

Generate test cases at multiple difficulty levels:

```python
from alphaevolve import TestCaseGenerator, Difficulty

generator = TestCaseGenerator(seed=42)
basic_tests = generator.generate("func(a, b)", Difficulty.BASIC, count=5)
edge_tests = generator.generate("func(a, b)", Difficulty.EDGE, count=5)
```

## Configuration

### Programmatic Configuration

```python
from alphaevolve import AlphaEvolveConfig

config = AlphaEvolveConfig(
    experiment={"name": "my_experiment", "seed": 42},
    evolution={"population_size": 20, "max_generations": 10},
    mutation={"use_llm": True, "temperature": 0.7},
)
```

### YAML Configuration (requires Hydra)

```yaml
# config/experiment.yaml
experiment:
  name: "sorting_evolution"
  seed: 42

evolution:
  population_size: 10
  max_generations: 5
  elitism_count: 2

mutation:
  use_llm: true
  llm_provider: "anthropic"
  temperature: 0.7
```

```python
from alphaevolve import load_config

config = load_config(config_path="config/experiment.yaml")
```

## Experiment Logging

```python
from alphaevolve import create_logger, ExperimentLogger

logger = create_logger(
    experiment_name="sorting_evolution",
    config=config,
)

# Log evolution progress
logger.log_evolution_step(
    generation=5,
    best_fitness=85.0,
    avg_fitness=72.0,
)

# Save checkpoint
logger.log_checkpoint(
    best_code=best.code,
    best_fitness=best.fitness,
    generation=5,
)

# Log final results
logger.log_final_results(
    best_code=best.code,
    best_fitness=best.fitness,
    total_generations=5,
    history=history,
)
```

## Checkpoint Management

```python
from alphaevolve import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="outputs/checkpoints",
    save_interval=2,
    max_checkpoints=3,
)

# Save checkpoint
manager.save(
    generation=5,
    best_code=best.code,
    best_fitness=best.fitness,
    population=population,
)

# Load checkpoint
checkpoint = manager.load()
if checkpoint:
    print(f"Resuming from generation {checkpoint.generation}")
```

## Running Examples

```bash
# Run sorting evolution demo
python -m alphaevolve.examples.sorting_demo

# Run with LLM (requires ANTHROPIC_API_KEY)
python -m alphaevolve.examples.sorting_demo --use-llm

# Run MAP-Elites archive demo
python -m alphaevolve.examples.sorting_demo --demo-archive

# Run evaluation pipeline demo
python -m alphaevolve.examples.sorting_demo --demo-pipeline
```

## Running Tests

```bash
# Run all tests
python -m pytest alphaevolve/tests/ -v

# Run specific test file
python -m pytest alphaevolve/tests/test_comprehensive.py -v

# Run with coverage
python -m pytest alphaevolve/tests/ --cov=alphaevolve
```

## Test Results

All **54 tests** pass successfully:

- **13 tests**: Core functionality (`test_alphaevolve.py`)
- **21 tests**: Comprehensive tests (`test_comprehensive.py`)
- **14 tests**: Configuration and utilities (`test_config_and_utils.py`)

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Evolution` | `alphaevolve.core.evolution` | Main evolution engine |
| `EvolutionConfig` | `alphaevolve.core.data_structures` | Evolution hyperparameters |
| `Individual` | `alphaevolve.core.data_structures` | Single program candidate |
| `Population` | `alphaevolve.core.data_structures` | Population management |
| `MutationEngine` | `alphaevolve.core.mutation` | Code mutation engine |
| `UnitTestEvaluator` | `alphaevolve.core.evaluator` | Unit test evaluation |

### LLM Integration

| Class | Module | Description |
|-------|--------|-------------|
| `LLMConfig` | `alphaevolve.llm.config` | LLM configuration |
| `BaseLLMClient` | `alphaevolve.llm.client` | Abstract LLM client |
| `LLMEnsemble` | `alphaevolve.llm.ensemble` | Multi-model ensemble |
| `AnthropicClient` | `alphaevolve.llm.providers.anthropic` | Claude API client |

### Archive (MAP-Elites)

| Class | Module | Description |
|-------|--------|-------------|
| `ProgramArchive` | `alphaevolve.evolution.archive` | MAP-Elites archive |
| `ArchiveConfig` | `alphaevolve.evolution.archive` | Archive configuration |
| `CodeComplexityFeature` | `alphaevolve.evolution.features` | Cyclomatic complexity |
| `CodeSizeFeature` | `alphaevolve.evolution.features` | Lines of code feature |

### Evaluation

| Class | Module | Description |
|-------|--------|-------------|
| `EvaluationPipeline` | `alphaevolve.evaluation.pipeline` | Multi-stage pipeline |
| `SyntaxCheckStage` | `alphaevolve.evaluation.stages` | Syntax validation |
| `BasicTestStage` | `alphaevolve.evaluation.stages` | Basic unit tests |
| `EdgeCaseStage` | `alphaevolve.evaluation.stages` | Edge case testing |
| `TestCaseGenerator` | `alphaevolve.evaluation.test_generation` | Test case generation |

### Configuration & Utilities

| Class | Module | Description |
|-------|--------|-------------|
| `AlphaEvolveConfig` | `alphaevolve.config` | Root configuration |
| `ExperimentLogger` | `alphaevolve.utils` | Experiment logging |
| `CheckpointManager` | `alphaevolve.utils` | Checkpoint management |

## License

MIT License

## Author

AlphaEvolve Team
