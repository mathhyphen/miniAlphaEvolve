#!/usr/bin/env python3
"""Validate intent.yaml files for research experiments."""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from scripts.utils import resolve_workspace_path

VALID_DIRECTIONS = {"lower_is_better", "higher_is_better"}
VALID_AGGREGATIONS = {"best", "last", "mean", "min", "max"}
GENERATION_KEYWORDS = {"generation", "synthesis", "translation"}


def load_intent_data(
    intent_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[List[str]]]:
    """Load and parse intent.yaml data."""
    if not intent_path.exists():
        return None, [f"File not found: {intent_path}"]

    try:
        content = intent_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return None, [f"Unable to read intent file: {e}"]

    try:
        data: Dict[str, Any] = yaml.safe_load(content)
    except yaml.YAMLError as e:
        return None, [f"Invalid YAML syntax: {e}"]

    if data is None:
        return None, ["Empty YAML file"]

    if not isinstance(data, dict):
        return None, ["Intent YAML must be a mapping at the top level"]

    return data, None


def _is_non_empty_string(value: Any) -> bool:
    """Return True when value is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def _get_mapping(data: Dict[str, Any], key: str, errors: List[str]) -> Dict[str, Any]:
    """Fetch a required mapping or record a validation error."""
    value = data.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} field is required and must be a mapping")
        return {}
    return value


def _is_generation_task(task_type: str) -> bool:
    """Return True when task type implies source-target generation."""
    lowered = task_type.lower()
    return any(keyword in lowered for keyword in GENERATION_KEYWORDS)


def validate_intent_data(data: Dict[str, Any]) -> Optional[List[str]]:
    """Validate parsed intent data."""
    errors: List[str] = []

    experiment = data.get("experiment", "")
    if not _is_non_empty_string(experiment):
        errors.append("experiment field is required and must be a string")

    branch = data.get("branch", "")
    if not branch or not re.match(r"^expl/[a-zA-Z0-9_-]+$", branch):
        errors.append("Branch pattern must be 'expl/<name>'")

    objective = data.get("objective", "")
    if not objective or not isinstance(objective, str):
        errors.append("Objective field is required and must be a string")
    elif len(objective.strip()) < 20:
        errors.append("Objective must be at least 20 characters")

    hypothesis = data.get("hypothesis", "")
    if not hypothesis or not isinstance(hypothesis, str):
        errors.append("Hypothesis field is required and must be a string")
    elif len(hypothesis.strip()) < 20:
        errors.append("Hypothesis must be at least 20 characters")

    collaboration = data.get("collaboration")
    if collaboration is not None and not isinstance(collaboration, dict):
        errors.append("collaboration field must be a mapping when provided")
    elif isinstance(collaboration, dict):
        for key in ("research_agent", "implementation_agent"):
            value = collaboration.get(key)
            if value is not None and not _is_non_empty_string(value):
                errors.append(f"collaboration.{key} must be a string when provided")
        handoff_artifacts = collaboration.get("handoff_artifacts")
        if handoff_artifacts is not None and (
            not isinstance(handoff_artifacts, list)
            or not all(_is_non_empty_string(item) for item in handoff_artifacts)
        ):
            errors.append(
                "collaboration.handoff_artifacts must be a list of strings when provided"
            )

    dataset = _get_mapping(data, "dataset", errors)
    if dataset:
        if not _is_non_empty_string(dataset.get("name")):
            errors.append("dataset.name is required and must be a string")
        split = dataset.get("split")
        if split is not None and not _is_non_empty_string(split):
            errors.append("dataset.split must be a string when provided")

    task = _get_mapping(data, "task", errors)
    task_type = ""
    if task:
        task_type = task.get("type", "")
        if not _is_non_empty_string(task_type):
            errors.append("task.type is required and must be a string")

    model = _get_mapping(data, "model", errors)
    if model:
        if not _is_non_empty_string(model.get("family")):
            errors.append("model.family is required and must be a string")
        backbone = model.get("backbone")
        if backbone is not None and not _is_non_empty_string(backbone):
            errors.append("model.backbone must be a string when provided")

    preprocessing = data.get("preprocessing")
    if preprocessing is not None and not isinstance(preprocessing, dict):
        errors.append("preprocessing field must be a mapping when provided")

    reproducibility = _get_mapping(data, "reproducibility", errors)
    if reproducibility:
        if not isinstance(reproducibility.get("seed"), int):
            errors.append("reproducibility.seed is required and must be an integer")
        for key in ("data_version", "config_path"):
            value = reproducibility.get(key)
            if value is not None and not _is_non_empty_string(value):
                errors.append(
                    f"reproducibility.{key} must be a string when provided"
                )

    research_plan = data.get("research_plan")
    if research_plan is not None and not isinstance(research_plan, dict):
        errors.append("research_plan field must be a mapping when provided")
    elif isinstance(research_plan, dict):
        for key in ("question", "motivation", "planned_approach"):
            value = research_plan.get(key)
            if value is not None and not _is_non_empty_string(value):
                errors.append(f"research_plan.{key} must be a string when provided")
        ablations = research_plan.get("ablations")
        if ablations is not None and (
            not isinstance(ablations, list)
            or not all(_is_non_empty_string(item) for item in ablations)
        ):
            errors.append(
                "research_plan.ablations must be a list of strings when provided"
            )

    success_criteria = data.get("success_criteria")
    if not isinstance(success_criteria, dict):
        errors.append("success_criteria field is required and must be a mapping")
    else:
        metrics = success_criteria.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            errors.append("success_criteria.metrics must be a non-empty list")
        else:
            for index, metric in enumerate(metrics, start=1):
                prefix = f"success_criteria.metrics[{index}]"
                if not isinstance(metric, dict):
                    errors.append(f"{prefix} must be a mapping")
                    continue

                name = metric.get("name")
                if not _is_non_empty_string(name):
                    errors.append(f"{prefix}.name is required and must be a string")

                threshold = metric.get("threshold")
                if not isinstance(threshold, (int, float)):
                    errors.append(f"{prefix}.threshold is required and must be numeric")

                direction = metric.get("direction")
                if direction not in VALID_DIRECTIONS:
                    valid = ", ".join(sorted(VALID_DIRECTIONS))
                    errors.append(f"{prefix}.direction must be one of: {valid}")

                aggregation = metric.get("aggregation", "best")
                if aggregation not in VALID_AGGREGATIONS:
                    valid = ", ".join(sorted(VALID_AGGREGATIONS))
                    errors.append(f"{prefix}.aggregation must be one of: {valid}")

    validation_plan = data.get("validation_plan")
    if validation_plan is not None and not isinstance(validation_plan, dict):
        errors.append("validation_plan field must be a mapping when provided")
    elif isinstance(validation_plan, dict):
        for key in ("checks", "artifacts"):
            value = validation_plan.get(key)
            if value is not None and (
                not isinstance(value, list)
                or not all(_is_non_empty_string(item) for item in value)
            ):
                errors.append(
                    f"validation_plan.{key} must be a list of strings when provided"
                )

    implementation_handoff = data.get("implementation_handoff")
    if implementation_handoff is not None and not isinstance(
        implementation_handoff, dict
    ):
        errors.append("implementation_handoff field must be a mapping when provided")
    elif isinstance(implementation_handoff, dict):
        for key in ("owner", "summary"):
            value = implementation_handoff.get(key)
            if value is not None and not _is_non_empty_string(value):
                errors.append(
                    f"implementation_handoff.{key} must be a string when provided"
                )
        for key in ("code_scope", "deliverables"):
            value = implementation_handoff.get(key)
            if value is not None and (
                not isinstance(value, list)
                or not all(_is_non_empty_string(item) for item in value)
            ):
                errors.append(
                    f"implementation_handoff.{key} must be a list of strings when provided"
                )

    return errors if errors else None


def collect_intent_warnings(data: Dict[str, Any]) -> List[str]:
    """Collect non-fatal recommendations for stronger reproducibility."""
    warnings: List[str] = []

    dataset = data.get("dataset", {}) if isinstance(data.get("dataset"), dict) else {}
    task = data.get("task", {}) if isinstance(data.get("task"), dict) else {}
    model = data.get("model", {}) if isinstance(data.get("model"), dict) else {}
    preprocessing = (
        data.get("preprocessing", {})
        if isinstance(data.get("preprocessing"), dict)
        else {}
    )
    constraints = (
        data.get("constraints", {})
        if isinstance(data.get("constraints"), dict)
        else {}
    )

    if not _is_non_empty_string(dataset.get("split")):
        warnings.append("dataset.split is recommended for reproducible evaluation")
    if not _is_non_empty_string(model.get("backbone")):
        warnings.append("model.backbone is recommended for architecture traceability")
    reproducibility = (
        data.get("reproducibility", {})
        if isinstance(data.get("reproducibility"), dict)
        else {}
    )
    collaboration = (
        data.get("collaboration", {})
        if isinstance(data.get("collaboration"), dict)
        else {}
    )
    research_plan = (
        data.get("research_plan", {})
        if isinstance(data.get("research_plan"), dict)
        else {}
    )
    validation_plan = (
        data.get("validation_plan", {})
        if isinstance(data.get("validation_plan"), dict)
        else {}
    )
    implementation_handoff = (
        data.get("implementation_handoff", {})
        if isinstance(data.get("implementation_handoff"), dict)
        else {}
    )
    if not _is_non_empty_string(reproducibility.get("data_version")):
        warnings.append("reproducibility.data_version is recommended for dataset traceability")
    if not _is_non_empty_string(reproducibility.get("config_path")):
        warnings.append(
            "reproducibility.config_path is recommended for config traceability"
        )
    if not _is_non_empty_string(constraints.get("gpu")):
        warnings.append("constraints.gpu is recommended for compute budgeting")
    if not _is_non_empty_string(collaboration.get("research_agent")):
        warnings.append(
            "collaboration.research_agent is recommended to record who owns planning and reporting"
        )
    elif "codex" not in collaboration["research_agent"].lower():
        warnings.append(
            "collaboration.research_agent should point to Codex for this workflow"
        )
    if not _is_non_empty_string(collaboration.get("implementation_agent")):
        warnings.append(
            "collaboration.implementation_agent is recommended to record who owns code implementation"
        )
    elif "claude" not in collaboration["implementation_agent"].lower():
        warnings.append(
            "collaboration.implementation_agent should point to Claude Code for this workflow"
        )
    if not _is_non_empty_string(research_plan.get("question")):
        warnings.append(
            "research_plan.question is recommended so Codex can frame the scientific question"
        )
    if not _is_non_empty_string(research_plan.get("planned_approach")):
        warnings.append(
            "research_plan.planned_approach is recommended for reproducible experiment planning"
        )
    ablations = research_plan.get("ablations")
    if not isinstance(ablations, list) or not ablations:
        warnings.append(
            "research_plan.ablations is recommended to capture comparison points"
        )
    checks = validation_plan.get("checks")
    if not isinstance(checks, list) or not checks:
        warnings.append(
            "validation_plan.checks is recommended to capture pre-run and post-run validation"
        )
    deliverables = implementation_handoff.get("deliverables")
    if not isinstance(deliverables, list) or not deliverables:
        warnings.append(
            "implementation_handoff.deliverables is recommended for Claude Code handoff clarity"
        )
    code_scope = implementation_handoff.get("code_scope")
    if not isinstance(code_scope, list) or not code_scope:
        warnings.append(
            "implementation_handoff.code_scope is recommended to bound implementation work"
        )

    task_type = task.get("type", "") if isinstance(task.get("type"), str) else ""
    if _is_generation_task(task_type):
        if not _is_non_empty_string(task.get("source_modality")):
            warnings.append("task.source_modality is recommended for generation tasks")
        if not _is_non_empty_string(task.get("target_modality")):
            warnings.append("task.target_modality is recommended for generation tasks")
        if not _is_non_empty_string(dataset.get("pairing")):
            warnings.append("dataset.pairing is recommended for generation tasks")
        if "spacing_mm" not in preprocessing:
            warnings.append(
                "preprocessing.spacing_mm is recommended for medical imaging generation tasks"
            )
        if not _is_non_empty_string(preprocessing.get("intensity_normalization")):
            warnings.append(
                "preprocessing.intensity_normalization is recommended for medical imaging generation tasks"
            )

    return warnings


def validate_intent(intent_path: Path) -> Optional[List[str]]:
    """Validate an intent.yaml file."""
    data, load_errors = load_intent_data(intent_path)
    if load_errors:
        return load_errors
    return validate_intent_data(data)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate intent.yaml file for research experiment"
    )
    parser.add_argument(
        "intent_file",
        type=Path,
        help="Path to intent.yaml file",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s 1.0.0",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        intent_file = resolve_workspace_path(args.intent_file)
    except ValueError as e:
        print(f"Error: Invalid path - {e}")
        sys.exit(1)

    data, load_errors = load_intent_data(intent_file)
    if load_errors:
        print("Validation FAILED:")
        for error in load_errors:
            print(f"  - {error}")
        sys.exit(1)

    errors = validate_intent_data(data)
    if errors:
        print("Validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    warnings = collect_intent_warnings(data)
    if warnings:
        print("Validation WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    print("Intent is valid.")
    sys.exit(0)


if __name__ == "__main__":
    main()
