# Repository Guidelines

## Project Structure & Module Organization

```
alphaevolve/                   # Core Python package
├── product/                   # AlphaEvolve Workbench (CORE PRODUCT)
│   ├── workbench.py          # AlphaEvolveWorkbench - main controller
│   ├── models.py             # TaskSpec, CandidateRecord, RunSnapshot
│   ├── evaluation.py         # PythonFunctionEvaluator
│   ├── proposers.py          # ProductProposer, HeuristicProductProposer
│   └── tasks.py              # 20 built-in algorithm tasks
├── llm/                      # LLM integration
│   └── minimax_client.py     # MiniMax API client
├── sandbox/                  # Safe code execution
│   ├── executor.py          # SandboxExecutor
│   ├── metrics.py            # Performance metrics
│   └── verifier.py           # Contract verification
├── problems/                 # Problem interface definitions
│   ├── problem.py            # Problem Protocol
│   └── evaluators.py         # Evaluator implementations
└── benchmarks/                # Benchmark suites
    └── problem_suite.py       # MST/Steiner problem suite

alphaevolve_webui/
├── backend/                  # FastAPI backend
│   ├── main.py              # App factory
│   └── product_api.py       # Product REST endpoints
└── frontend/                 # React + Vite frontend
    └── src/
        ├── components/        # UI components
        └── api/              # API client

tests/                        # Test suite
```

## Build, Test and Development Commands

Set up Python dependencies:

```bash
pip install -e .
```

Run tests:

```bash
pytest -q tests
pytest --cov=alphaevolve tests
```

Start the API server:

```bash
cd alphaevolve_webui/backend
uvicorn main:app --reload
```

Start the Web UI:

```bash
cd alphaevolve_webui/frontend
npm install
npm run dev
```

Build the Web UI:

```bash
cd alphaevolve_webui/frontend
npm run build
```

## Coding Style & Naming Conventions

Use 4-space indentation in Python and keep APIs typed where practical.

- **Python**: `snake_case` files, `PascalCase` classes, `snake_case` functions
- **Frontend**: React components in `PascalCase` under `src/components/`, helper functions in `camelCase`

## Testing Guidelines

Pytest is the primary test runner. Add new tests as `tests/test_<feature>.py`.

For web UI changes, rebuild the frontend before running integration tests.

## Commit & Pull Request Guidelines

Use Conventional Commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, etc.

PRs should include:
- Clear summary of changes
- Commands run to verify
- Screenshots for UI changes

## Security & Configuration Tips

- Keep secrets in `.env`; do not hardcode API keys
- LLM-related scripts expect `MINIMAX_API_KEY` and optionally `MINIMAX_MODEL`
- Treat `outputs/`, checkpoints, and generated artifacts as disposable
