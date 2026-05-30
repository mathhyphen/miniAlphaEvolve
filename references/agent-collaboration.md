# Agent Collaboration Reference

## Role Split

This repository is designed around a two-agent workflow:

- Codex handles research work: experiment framing, plans, validation criteria, report drafts, and implementation handoffs.
- Claude Code handles implementation work: source changes, config wiring, tests, refactors, and execution checks.

Inside Claude Code, the project enables two subagents:

- `implementation-engineer`
- `workflow-validator`

`.claude/settings.json` makes `implementation-engineer` the default Claude Code agent for this repository.

## Handoff Contract

Before Claude Code implements anything, the handoff should say:

- what is being built or changed
- which files or modules are likely involved
- what must stay unchanged
- which validations should pass at the end

When the handoff belongs to an experiment, store it under `findings/`, for example:

- `workspace/<experiment>/findings/implementation_handoff.md`
- `workspace/<experiment>/findings/plan.md`

## Recommended Loop

1. Codex drafts the experiment plan and validation checklist.
2. Codex prepares the implementation handoff.
3. Claude Code implements with `implementation-engineer`.
4. Claude Code verifies with `workflow-validator`.
5. Codex interprets the results and writes the final report.
