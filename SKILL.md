---
name: unix-research-workflow
description: Intent-driven research workflow for local ML projects. Use when Codex is helping with research briefs, experiment plans, validation criteria, report drafts, or implementation handoffs, and when Claude Code should be used for code implementation tasks.
---

# Unix Research Workflow

Use this skill to keep research work and implementation work separated.

## Role Split

- Codex owns research thinking: problem framing, experiment planning, hypothesis design, validation criteria, report drafting, and handoff notes.
- Claude Code owns implementation: code edits, training loops, configuration wiring, tests, bug fixes, and refactors.

If a request mixes both, split it into two steps:

1. Ask Codex for the research plan and validation checklist.
2. Hand the result to Claude Code for implementation.

## Recommended Flow

1. Create or locate the experiment workspace.
2. Draft the plan using the templates in `templates/`.
3. Validate the intent with `python -m scripts.validate_intent <path>`.
4. Implement the approved plan with Claude Code.
5. Log metrics and generate the report with `python -m scripts.summarize <name>`.

## Useful Files

- `templates/intent.yaml` for the experiment contract
- `templates/research_report.md` for Codex output
- `templates/experiment_plan.md` for the proposed study
- `templates/validation_checklist.md` for verification steps
- `templates/implementation_handoff.md` for Claude Code
- `.claude/agents/implementation-engineer.md` for Claude Code implementation work
- `.claude/agents/workflow-validator.md` for Claude Code verification work
- `references/agent-collaboration.md` for the Codex-to-Claude handoff contract

## When To Use The Repository Scripts

Prefer the repository scripts over handwritten one-off commands:

- `scripts/new_exp.py` for experiment scaffolding
- `scripts/validate_intent.py` for intent checks
- `scripts/summarize.py` for report generation
- `scripts/list_exp.py` for discovery
- `scripts/show_exp.py` for quick inspection
- `scripts/compare_exp.py` for comparisons
- `scripts/export_csv.py` for tabular export
- `scripts/rm_exp.py` for cleanup

## What Not To Do

- Do not ask Claude Code to invent the research plan from scratch when a Codex plan already exists.
- Do not use Codex as the implementation driver for large code changes.
- Do not skip validation before training or evaluation.
