# Research Routing Protocol

This repository uses a two-agent split:

- Codex handles research work: planning, hypothesis design, validation criteria, report drafting, and implementation handoff.
- Claude Code handles implementation work: code changes, script updates, integrations, tests, and refactors.

## Routing Rules

Use Codex when the task is any of the following:

- research brief or literature-style reasoning
- experiment plan or ablation plan
- validation checklist or success criteria
- report outline or result interpretation
- implementation handoff for another agent

Use Claude Code when the task is any of the following:

- code implementation
- training loop changes
- config wiring
- logging integration
- test updates
- bug fixes or refactors

## Hand Off Rules

1. Codex produces the plan, validation checklist, and implementation handoff.
2. Claude Code implements the approved handoff with the `implementation-engineer` project subagent.
3. Claude Code verifies the result with the `workflow-validator` project subagent when the task includes code or workflow changes.
4. If implementation changes the research assumptions, return to Codex for a plan update.

## Required Artifacts

- `templates/experiment_plan.md`
- `templates/validation_checklist.md`
- `templates/implementation_handoff.md`
- `templates/research_report.md`

## Suggested Sequence

1. Draft the research plan with Codex.
2. Review the validation checklist.
3. Hand the implementation work to Claude Code.
4. Update `intent.yaml` if the experiment scope changes.
5. Validate before execution.
6. Summarize results after logging.

## Guardrails

- Do not mix open-ended research drafting with implementation in the same pass when the task can be split.
- Do not start training without a validated intent.
- Do not treat code generation and research planning as the same responsibility.
