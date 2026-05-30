# AlphaEvolve

AlphaEvolve is an **algorithm discovery workbench** inspired by Google DeepMind's AlphaEvolve. It uses Large Language Models (LLMs) to iteratively improve algorithms for automatically evaluable tasks through an evolutionary search loop.

> This implementation focuses on the canonical AlphaEvolve product loop: **task → prompt → propose → evaluate → archive**.

## Features

- **AlphaEvolve Workbench**: The canonical task → prompt → propose → evaluate → archive loop
- **20 Built-in Algorithm Tasks**: Classic algorithm problems (sorting, search, graphs, dynamic programming, etc.)
- **LLM-Powered Proposers**: Uses MiniMax (or any compatible LLM) for intelligent code mutations
- **Heuristic Fallback Proposer**: Deterministic proposer for testing without API keys
- **Automatic Evaluation**: Built-in Python function evaluator with correctness and speed metrics
- **Archive System**: Maintains best candidates across generations
- **REST API**: FastAPI-based API for programmatic access
- **Web UI**: React-based frontend for visual interaction

## Architecture

The product is structured around one clear center — the `AlphaEvolveWorkbench`:

```
TaskSpec + EvaluationCases
        ↓
AlphaEvolveWorkbench.start_task_run()
        ↓
┌─────────────────────────────────────────────────────────┐
│  For each generation:                                    │
│    1. PromptSampler builds context from task + archive   │
│    2. Proposer (LLM or Heuristic) generates proposal      │
│    3. Evaluator scores the candidate                    │
│    4. Archive updated with best candidates               │
└─────────────────────────────────────────────────────────┘
        ↓
RunSnapshot (history + archive + best program)
```

## Installation

```bash
pip install -e .
```

For development with additional dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from alphaevolve import AlphaEvolveWorkbench, get_builtin_task

# Use heuristic proposer (no API key needed)
workbench = AlphaEvolveWorkbench()

# Or use MiniMax LLM proposer (requires MINIMAX_API_KEY)
# from alphaevolve.product.proposers import create_minimax_proposer
# workbench = AlphaEvolveWorkbench(proposer=create_minimax_proposer())

# List available tasks
from alphaevolve import list_builtin_tasks
for task in list_builtin_tasks():
    print(f"{task.task_id}: {task.title}")

# Run evolution on a task
run = workbench.start_run(
    task_id="sort_numbers",
    generations=8,
    archive_size=8
)

# Get the best program
best_program = workbench.export_best_program(run.run_id)
print(best_program)
```

### REST API

```bash
# Start the API server
cd alphaevolve_webui/backend
uvicorn main:app --reload

# List tasks
curl http://localhost:8000/api/tasks

# Run evolution
curl -X POST http://localhost:8000/api/runs \
  -H "Content-Type: application/json" \
  -d '{"task_id": "sort_numbers", "generations": 4, "archive_size": 4}'

# Export best program
curl http://localhost:8000/api/runs/{run_id}/export
```

### Web UI

```bash
cd alphaevolve_webui/frontend
npm install
npm run dev
```

Then open http://localhost:3000 in your browser.

## Project Structure

```
alphaevolve/
├── __init__.py          # Public API exports
├── product/             # AlphaEvolve Workbench (CORE PRODUCT)
│   ├── workbench.py     # AlphaEvolveWorkbench - main product controller
│   ├── models.py        # TaskSpec, CandidateRecord, RunSnapshot
│   ├── evaluation.py    # PythonFunctionEvaluator
│   ├── proposers.py    # ProductProposer, HeuristicProductProposer
│   └── tasks.py         # 20 built-in algorithm tasks
├── llm/                 # LLM integration
│   └── minimax_client.py # MiniMax API client
├── sandbox/             # Safe code execution
│   ├── executor.py      # SandboxExecutor
│   ├── metrics.py       # Performance metrics
│   └── verifier.py      # Contract verification
├── problems/            # Problem interface definitions
│   ├── problem.py       # Problem Protocol
│   └── evaluators.py    # Evaluator implementations
└── benchmarks/          # Benchmark suites
    └── problem_suite.py  # MST/Steiner problem suite

alphaevolve_webui/
├── backend/             # FastAPI backend
│   ├── main.py         # App factory
│   └── product_api.py   # Product REST endpoints
└── frontend/            # React + Vite frontend
    └── src/
        ├── components/  # UI components
        └── api/         # API client

tests/                   # Test suite
```

## Built-in Tasks

| Task ID | Title | Category |
|---------|-------|----------|
| `sort_numbers` | Sort Numbers | Sorting |
| `find_first_index` | Find First Index | Search |
| `binary_search` | Binary Search | Search |
| `two_sum_indices` | Two Sum Indices | Hashing |
| `valid_parentheses` | Valid Parentheses | Stack |
| `fibonacci` | Fibonacci | DP |
| `gcd` | Greatest Common Divisor | Math |
| `is_prime` | Prime Check | Math |
| `sieve_primes` | Sieve of Eratosthenes | Math |
| `factorial` | Factorial | Math |
| `reverse_string` | Reverse String | Strings |
| `palindrome_check` | Palindrome Check | Strings |
| `merge_intervals` | Merge Intervals | Intervals |
| `max_subarray_sum` | Maximum Subarray Sum | DP |
| `longest_common_subsequence` | Longest Common Subsequence | DP |
| `edit_distance` | Edit Distance | DP |
| `knapsack_01` | 0/1 Knapsack | DP |
| `bfs_order` | Breadth-First Search | Graphs |
| `dijkstra_shortest_path` | Dijkstra Shortest Paths | Graphs |
| `matrix_multiply` | Matrix Multiplication | Matrix |

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `AlphaEvolveWorkbench` | Main product controller for the evolution loop |
| `TaskSpec` | Task specification with objective, initial program, and evaluation cases |
| `EvaluationCase` | A single test case (input args, expected output) |
| `PythonFunctionEvaluator` | Evaluator that runs Python functions against test cases |
| `RunSnapshot` | Complete run state including history and archive |
| `CandidateRecord` | A single candidate with program, evaluation, and lineage |

### Proposers

| Class | Description |
|-------|-------------|
| `HeuristicProductProposer` | Deterministic fallback (no API key needed) |
| `MiniMaxProductProposer` | LLM-powered proposer using MiniMax API |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MINIMAX_API_KEY` | MiniMax API key for LLM proposer | Required for LLM mode |
| `MINIMAX_MODEL` | MiniMax model name | `MiniMax-M2.7-highspeed` |
| `ALPHAEVOLVE_CORS_ORIGINS` | CORS allowed origins | `http://localhost:3000` |

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=alphaevolve

# Run specific test file
python -m pytest tests/test_alphaevolve_product.py -v
```

## Documentation

- [Product Gap Report](docs/alphaevolve_product_gap_report.md) - Analysis of differences from Google DeepMind AlphaEvolve
- [Architecture Document](docs/alphadev_architecture.md) - System architecture details
- [Alignment Report](docs/alphaevolve_alignment.md) - Implementation alignment status

## License

MIT License

## Acknowledgments

Inspired by Google DeepMind's AlphaEvolve research. See:
- [AlphaEvolve Blog Post](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [AlphaEvolve Paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)
