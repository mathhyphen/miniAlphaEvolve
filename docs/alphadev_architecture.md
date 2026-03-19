# AlphaDev Architecture Document

## Based on "AlphaEvolve: A coding agent for scientific and algorithmic discovery" (arXiv 2506.13131)

---

## 1. System Overview

### 1.1 High-Level Architecture

AlphaDev is an evolutionary coding agent that uses Large Language Models (LLMs) to iteratively improve algorithms. The system combines genetic programming with LLM-guided mutations to discover optimized code solutions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AlphaDev System                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Prompt     │    │   LLM        │    │   Diff       │                   │
│  │   Sampler    │───▶│   Ensemble   │───▶│   Parser     │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                    │                                               │
│         │                    ▼                                                │
│         │         ┌────────────────────┐                                     │
│         │         │  Program Database  │                                     │
│         │         │  (MAP-Elites       │                                     │
│         │         │   Archive)         │                                     │
│         │         └────────┬───────────┘                                     │
│         │                  │                                                │
│         ▼                  ▼                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Evolution   │◀──▶│  Evaluation  │◀──▶│   Test Case  │                   │
│  │    Loop      │    │   Pipeline   │    │   Generator  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                                              │                     │
│         └──────────────────────────────────────────────┘                     │
│                            (Feedback Loop)                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Design Principles

1. **Multi-Objective Evolution**: Optimizes for correctness, performance, and simplicity simultaneously
2. **LLM-Guided Mutations**: Uses LLM prompts rather than traditional genetic operators
3. **Quality-Diversity Trade-off**: MAP-Elites inspired archive maintains diverse solutions
4. **Progressive Evaluation**: Test cases of increasing difficulty for efficient filtering
5. **Language Agnostic**: Supports any programming language through pluggable components

### 1.3 System Capabilities

| Capability | Description | Status |
|------------|-------------|--------|
| Code Evolution | Evolve entire code files (hundreds of lines) | ✅ Core Implemented |
| Multi-Language | Support for any programming language | ⚠️ Partial |
| Multi-Objective | Optimize multiple metrics simultaneously | ✅ Implemented |
| Parallel Evaluation | Concurrent fitness assessment | 🔲 Planned |
| Diff Generation | SEARCH/REPLACE block based mutations | 🔲 Planned |
| Quality Diversity | MAP-Elites inspired archive | 🔲 Planned |

---

## 2. Component Specifications

### 2.1 Prompt Sampler

**Responsibilities:**
- Generate context-rich prompts for LLM code mutations
- Incorporate feedback from evaluation into prompts
- Support multiple mutation strategies
- Manage prompt templates and versioning

**Interface:**
```python
class PromptSampler:
    def sample(
        self,
        code: str,
        feedback: EvaluationFeedback,
        mutation_strategy: MutationStrategy,
        history: List[Individual]
    ) -> Prompt:
        """Generate mutation prompt."""
        pass

    def register_template(self, name: str, template: str) -> None:
        """Register custom prompt template."""
        pass
```

### 2.2 LLM Ensemble

**Responsibilities:**
- Manage multiple LLM backends (Claude, Gemini, etc.)
- Route requests based on complexity and priority
- Handle rate limiting and quota management
- Implement retry logic with exponential backoff

**Interface:**
```python
class LLMEnsemble:
    def generate(
        self,
        prompt: Prompt,
        priority: Priority = Priority.NORMAL
    ) -> LLMResponse:
        """Generate code mutation."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics per model."""
        pass
```

### 2.3 Program Database (MAP-Elites Archive)

**Responsibilities:**
- Store evolved programs with behavioral descriptors
- Maintain quality-diversity trade-off
- Support efficient retrieval by feature dimensions

**Interface:**
```python
class ProgramDatabase:
    def add(self, individual: Individual) -> bool:
        """Add individual to archive."""
        pass

    def get_neighbors(self, individual: Individual, radius: int = 1) -> List[Individual]:
        """Get neighboring individuals in feature space."""
        pass

    def sample(self, strategy: SamplingStrategy) -> Individual:
        """Sample individual for mutation."""
        pass
```

### 2.4 Test Case Generator

**Responsibilities:**
- Generate test cases of increasing difficulty
- Create edge case scenarios
- Support property-based testing

**Difficulty Levels:**
1. **Basic**: Simple inputs, happy path
2. **Standard**: Typical use cases
3. **Edge**: Boundary conditions, empty inputs
4. **Stress**: Large inputs, performance testing
5. **Adversarial**: Inputs designed to break implementations

---

## 3. Module Structure

```
alphaevolve/
├── __init__.py
├── __version__.py
│
├── core/
│   ├── __init__.py
│   ├── data_structures.py      # Individual, Population, Results
│   ├── evolution.py            # EvolutionLoop orchestration
│   ├── config.py               # Configuration system
│   └── state.py                # Checkpoint/restore
│
├── llm/
│   ├── __init__.py
│   ├── ensemble.py             # LLMEnsemble
│   ├── client.py               # Base LLM interface
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── anthropic.py        # Claude API
│   │   └── google.py           # Gemini API
│   └── prompt/
│       ├── __init__.py
│       ├── sampler.py          # PromptSampler
│       └── templates.py        # Default templates
│
├── evolution/
│   ├── __init__.py
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── tournament.py
│   │   └── roulette.py
│   ├── archive.py              # MAP-Elites archive
│   └── features.py             # Feature dimensions
│
├── evaluation/
│   ├── __init__.py
│   ├── pipeline.py             # EvaluationPipeline
│   ├── stages.py               # Evaluation stages
│   ├── sandbox.py              # Safe execution
│   └── test_generation/
│       ├── __init__.py
│       └── generator.py        # TestCaseGenerator
│
├── utils/
│   ├── __init__.py
│   ├── logging.py
│   └── serialization.py
│
├── run/
│   ├── __init__.py
│   └── pipeline/
│       ├── evolve.py
│       └── evaluate.py
│
├── examples/
│   ├── __init__.py
│   ├── sorting.py
│   └── matmul.py
│
└── tests/
    ├── __init__.py
    └── test_*.py
```

---

## 4. Design Patterns

### 4.1 Factory Pattern

```python
LLM_FACTORY: Dict[str, Type[LLMClient]] = {}

def register_llm(name: str):
    def decorator(cls):
        LLM_FACTORY[name] = cls
        return cls
    return decorator

def LLMFactory(provider: str, config: Dict) -> LLMClient:
    client_class = LLM_FACTORY.get(provider)
    if not client_class:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return client_class(config)
```

### 4.2 Registry Pattern

```python
EVALUATION_STAGES: Dict[str, Type[EvaluationStage]] = {}

def register_stage(name: str):
    def decorator(cls):
        EVALUATION_STAGES[name] = cls
        return cls
    return decorator
```

### 4.3 Strategy Pattern

```python
class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population: Population, count: int) -> List[Individual]:
        pass
```

---

## 5. Configuration Requirements

### 5.1 Evolution Configuration

```python
@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.7
    elitism_count: int = 2
    diversity_threshold: float = 0.1
    timeout_seconds: float = 3600.0
    seed: Optional[int] = None
```

### 5.2 LLM Configuration

```python
@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: Optional[str] = None
```

---

## 6. Extension Points

### 6.1 Custom Evaluators

```python
from alphaevolve.evaluation import BaseEvaluator, EvaluatorResult

class MyEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "MyEvaluator"

    def evaluate(self, code: str) -> EvaluatorResult:
        # Custom evaluation logic
        pass
```

### 6.2 Custom Mutation Strategies

```python
from alphaevolve.llm.prompt import PromptSampler

class CustomSampler(PromptSampler):
    def sample(self, code, feedback, strategy, history):
        # Custom prompt generation
        pass
```

---

## 7. Implementation Roadmap

### Phase 1: Core Framework (Current Status)

**Implemented:**
- ✅ Basic data structures (Individual, Population)
- ✅ Simple evolutionary loop
- ✅ Unit test evaluator
- ✅ Performance evaluator
- ✅ Basic mutation engine (fallback mode)

### Phase 2: LLM Integration

**To Implement:**
- 🔲 LLM client abstraction
- 🔲 Anthropic/Google API integration
- 🔲 LLM ensemble with routing
- 🔲 Prompt template system

### Phase 3: Advanced Evolution

**To Implement:**
- 🔲 MAP-Elites archive
- 🔲 Feature dimension extractors
- 🔲 Multiple selection strategies

### Phase 4: Evaluation Pipeline

**To Implement:**
- 🔲 Multi-stage evaluation
- 🔲 Sandboxed execution improvements
- 🔲 Automatic test generation

### Phase 5: Production Features

**To Implement:**
- 🔲 Hydra configuration
- 🔲 Checkpoint/resume improvements
- 🔲 Distributed evolution support
