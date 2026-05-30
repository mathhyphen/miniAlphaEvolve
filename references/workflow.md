# Workflow Reference

## Lifecycle

Use the repository as a two-agent research workflow:

1. Ask Codex for a research brief and experiment plan.
2. Save the plan in the workspace using the documentation templates.
3. Convert the approved plan into an implementation handoff.
4. Give the handoff to Claude Code for code changes.
5. Create or update `intent.yaml`.
6. Validate the intent before running anything.
7. Train or evaluate with logging enabled.
8. Run `python -m scripts.summarize <name>` to generate the report.

Run CLI modules from the repository root with `python -m scripts.<module>`.

## Agent Responsibilities

### Codex

- research framing
- experiment design
- hypothesis writing
- validation planning
- report drafting
- implementation handoff creation

### Claude Code

- implementation
- script wiring
- training loop changes
- logging integration
- tests and bug fixes
- refactors

## Directory Layout

Experiments are searched in:

- `workspace/`
- `.claude/worktrees/`
- `~/.claude/worktrees/`

Each experiment should look like:

```text
<experiment>/
|- intent.yaml
|- logs/
|  `- metrics.jsonl
|- findings/
|  `- report.md|report.json|report.html
`- checkpoints/
```

## Script Selection

- Use `new_exp.py` when creating a new experiment scaffold.
- Use `validate_intent.py` before training starts.
- Use `summarize.py` after logging exists.
- Use `list_exp.py` and `show_exp.py` for discovery.
- Use `compare_exp.py` and `export_csv.py` for analysis.
- Use `rm_exp.py` only when cleanup is requested.

## Documentation Templates

- `templates/experiment_plan.md` records the proposed study.
- `templates/validation_checklist.md` records the checks to run.
- `templates/implementation_handoff.md` records what Claude Code should build.
- `templates/research_report.md` records the final research summary.

## Status Expectations

- `I` means initialized only.
- `M` means metrics exist.
- `R` means at least one report artifact exists.

## Logging Expectations

- `LogHook.log_epoch()` writes flat metrics with `epoch` and `phase`.
- Nested `{"metrics": {...}}` payloads are also supported.
- Reports summarize numeric metrics and separate them by phase when present.
