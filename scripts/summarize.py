#!/usr/bin/env python3
"""Generate summary reports from experiment metrics with multiple format support."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from scripts.report_factory import ReportFactory
from scripts.utils import (
    find_experiment_path,
    format_error,
    validate_experiment_name,
)

logger = logging.getLogger(__name__)
CONTROL_KEYS = {"_timestamp", "epoch", "metrics", "phase", "seed", "step", "type"}
DEFAULT_AGGREGATION = "best"


def configure_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )


def find_experiment(name: str) -> Optional[Path]:
    """Find experiment directory across all workspaces."""
    return find_experiment_path(name)


def load_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    """Load metrics from JSONL file."""
    metrics: List[Dict[str, Any]] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    return metrics


def load_intent(intent_path: Path) -> Dict[str, Any]:
    """Load intent.yaml when available."""
    if not intent_path.exists():
        return {}

    try:
        content = intent_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}

    return data if isinstance(data, dict) else {}


def compute_metric_stats(metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metric statistics for each metric key."""
    if not metrics:
        return {}

    stats: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        phase = metric.get("phase")
        prefix = f"{phase}." if isinstance(phase, str) and phase else ""

        if isinstance(metric.get("metrics"), dict):
            metric_values = metric["metrics"]
        else:
            metric_values = {
                key: value
                for key, value in metric.items()
                if key not in CONTROL_KEYS
            }

        for key, value in metric_values.items():
            if not isinstance(value, (int, float)):
                continue

            summary_key = f"{prefix}{key}"
            numeric_value = float(value)
            current = stats.get(summary_key)
            if current is None:
                stats[summary_key] = {
                    "count": 1,
                    "sum": numeric_value,
                    "min": numeric_value,
                    "max": numeric_value,
                    "last": numeric_value,
                }
                continue

            current["count"] += 1
            current["sum"] += numeric_value
            current["min"] = min(current["min"], numeric_value)
            current["max"] = max(current["max"], numeric_value)
            current["last"] = numeric_value

    return stats


def compute_summary(metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute summary statistics for each metric key."""
    stats = compute_metric_stats(metrics)
    summary: Dict[str, float] = {}
    for key in sorted(stats):
        current = stats[key]
        mean = current["sum"] / current["count"]
        summary[f"{key}.last"] = current["last"]
        summary[f"{key}.mean"] = mean
        summary[f"{key}.min"] = current["min"]
        summary[f"{key}.max"] = current["max"]
    return summary


def _metric_candidates(name: str) -> List[str]:
    """Return likely metric key variants for success criteria lookup."""
    candidates = [name]
    for prefix in ("train_", "val_", "test_"):
        if name.startswith(prefix):
            candidates.append(name.replace("_", ".", 1))
            break
    return list(dict.fromkeys(candidates))


def _find_metric_series(
    name: str,
    stats: Dict[str, Dict[str, float]],
) -> Optional[tuple[str, Dict[str, float]]]:
    """Resolve a success-criteria name to an observed metric series."""
    for candidate in _metric_candidates(name):
        if candidate in stats:
            return candidate, stats[candidate]

    suffix_matches = [key for key in stats if key.endswith(f".{name}")]
    if suffix_matches:
        priority = {"val": 0, "test": 1, "train": 2}
        suffix_matches.sort(
            key=lambda key: (priority.get(key.split(".", 1)[0], 99), key)
        )
        chosen = suffix_matches[0]
        return chosen, stats[chosen]

    return None


def _resolve_aggregation(criterion: Dict[str, Any]) -> str:
    """Map criterion aggregation to a supported statistic key."""
    aggregation = criterion.get("aggregation", DEFAULT_AGGREGATION)
    if aggregation == "best":
        return "min" if criterion.get("direction") == "lower_is_better" else "max"
    return aggregation


def evaluate_success_criteria(
    intent: Dict[str, Any],
    stats: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Evaluate success criteria from intent against collected metrics."""
    success_criteria = intent.get("success_criteria", {})
    metrics = success_criteria.get("metrics", [])
    results: List[Dict[str, Any]] = []

    if not isinstance(metrics, list) or not metrics:
        return {"passed": None, "metrics": results}

    overall_passed = True
    for criterion in metrics:
        name = criterion.get("name", "unknown")
        threshold = criterion.get("threshold")
        direction = criterion.get("direction")
        resolved = _find_metric_series(name, stats) if isinstance(name, str) else None

        if (
            resolved is None
            or not isinstance(threshold, (int, float))
            or direction not in {"lower_is_better", "higher_is_better"}
        ):
            results.append({
                "name": name,
                "threshold": threshold,
                "direction": direction,
                "aggregation": criterion.get("aggregation", DEFAULT_AGGREGATION),
                "observed": None,
                "source_key": None,
                "passed": False,
            })
            overall_passed = False
            continue

        metric_key, series = resolved
        aggregation = _resolve_aggregation(criterion)
        observed = series.get(aggregation)
        if observed is None:
            results.append({
                "name": name,
                "threshold": float(threshold),
                "direction": direction,
                "aggregation": aggregation,
                "observed": None,
                "source_key": None,
                "passed": False,
            })
            overall_passed = False
            continue

        passed = (
            observed <= float(threshold)
            if direction == "lower_is_better"
            else observed >= float(threshold)
        )
        results.append({
            "name": name,
            "threshold": float(threshold),
            "direction": direction,
            "aggregation": aggregation,
            "observed": observed,
            "source_key": f"{metric_key}.{aggregation}",
            "passed": passed,
        })
        overall_passed = overall_passed and passed

    return {"passed": overall_passed, "metrics": results}


def _format_metadata_value(value: Any) -> Any:
    """Normalize metadata values for report rendering."""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


def extract_report_metadata(
    metrics: List[Dict[str, Any]],
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    """Build report metadata from intent and logged context."""
    metadata: Dict[str, Any] = {}

    if intent:
        dataset = intent.get("dataset", {})
        task = intent.get("task", {})
        model = intent.get("model", {})
        reproducibility = intent.get("reproducibility", {})
        constraints = intent.get("constraints", {})

        if isinstance(dataset, dict):
            metadata["dataset_name"] = dataset.get("name", "unknown")
            metadata["dataset_split"] = dataset.get("split", "unknown")
            if dataset.get("pairing"):
                metadata["dataset_pairing"] = dataset["pairing"]
        if isinstance(task, dict):
            metadata["task_type"] = task.get("type", "unknown")
            if task.get("source_modality"):
                metadata["source_modality"] = task["source_modality"]
            if task.get("target_modality"):
                metadata["target_modality"] = task["target_modality"]
        if isinstance(model, dict):
            metadata["model_family"] = model.get("family", "unknown")
            if model.get("backbone"):
                metadata["model_backbone"] = model["backbone"]
        if isinstance(reproducibility, dict):
            if reproducibility.get("seed") is not None:
                metadata["seed"] = reproducibility["seed"]
            if reproducibility.get("data_version"):
                metadata["data_version"] = reproducibility["data_version"]
            config_value = reproducibility.get("config_path")
            if config_value is None:
                config_value = reproducibility.get("config")
            if config_value is not None:
                metadata["config_path"] = config_value
        if isinstance(constraints, dict) and constraints.get("gpu"):
            metadata["gpu_budget"] = constraints["gpu"]

    for metric in metrics:
        metric_type = metric.get("type")
        if metric_type == "environment":
            for key in (
                "python_version",
                "platform",
                "hostname",
                "cwd",
                "command",
                "argv",
                "git_commit",
                "git_branch",
                "git_dirty",
            ):
                if key in metric and key not in metadata:
                    metadata[key] = metric[key]
        elif metric_type in {"run_context", "reproducibility"}:
            for key, value in metric.items():
                if key in {"type", "_timestamp"}:
                    continue
                metadata[key] = value

    return {
        key: _format_metadata_value(value)
        for key, value in sorted(metadata.items())
        if value not in (None, "", [], {})
    }


def extract_research_context(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Build research-planning context from intent.yaml."""
    context: Dict[str, Any] = {}

    collaboration = (
        intent.get("collaboration", {})
        if isinstance(intent.get("collaboration"), dict)
        else {}
    )
    research_plan = (
        intent.get("research_plan", {})
        if isinstance(intent.get("research_plan"), dict)
        else {}
    )
    validation_plan = (
        intent.get("validation_plan", {})
        if isinstance(intent.get("validation_plan"), dict)
        else {}
    )
    implementation_handoff = (
        intent.get("implementation_handoff", {})
        if isinstance(intent.get("implementation_handoff"), dict)
        else {}
    )

    if collaboration.get("research_agent"):
        context["research_agent"] = collaboration["research_agent"]
    if collaboration.get("implementation_agent"):
        context["implementation_agent"] = collaboration["implementation_agent"]
    if collaboration.get("handoff_artifacts"):
        context["handoff_artifacts"] = collaboration["handoff_artifacts"]

    if research_plan.get("question"):
        context["research_question"] = research_plan["question"]
    if research_plan.get("motivation"):
        context["research_motivation"] = research_plan["motivation"]
    if research_plan.get("planned_approach"):
        context["planned_approach"] = research_plan["planned_approach"]
    if research_plan.get("ablations"):
        context["planned_ablations"] = research_plan["ablations"]

    if validation_plan.get("checks"):
        context["validation_checks"] = validation_plan["checks"]
    if validation_plan.get("artifacts"):
        context["validation_artifacts"] = validation_plan["artifacts"]

    if implementation_handoff.get("owner"):
        context["implementation_owner"] = implementation_handoff["owner"]
    if implementation_handoff.get("summary"):
        context["implementation_summary"] = implementation_handoff["summary"]
    if implementation_handoff.get("code_scope"):
        context["implementation_scope"] = implementation_handoff["code_scope"]
    if implementation_handoff.get("deliverables"):
        context["implementation_deliverables"] = implementation_handoff["deliverables"]

    return {
        key: value
        for key, value in sorted(context.items())
        if value not in (None, "", [], {})
    }


def summarize(
    name: str,
    format_name: str = "markdown",
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Generate summary report for an experiment."""
    exp_dir = find_experiment(name)

    if exp_dir is None:
        logger.error(
            format_error(
                f"Experiment '{name}' not found",
                "Check experiment name or create it first with: python -m scripts.new_exp --name " + name,
            )
        )
        return None

    metrics_path = exp_dir / "logs" / "metrics.jsonl"
    if output_path is None:
        formatter = ReportFactory.get_format(format_name)
        output_path = exp_dir / "findings" / f"report{formatter.extension()}"

    if not metrics_path.exists():
        logger.error(
            format_error(
                f"No metrics.jsonl found for '{name}'",
                "Run metrics logging first or check experiment name",
            )
        )
        return None

    metrics = load_metrics(metrics_path)
    if not metrics:
        logger.error("No data in metrics file.")
        return None

    intent = load_intent(exp_dir / "intent.yaml")
    stats = compute_metric_stats(metrics)
    summary = compute_summary(metrics)
    metadata = extract_report_metadata(metrics, intent)
    research_context = extract_research_context(intent)
    success_criteria = evaluate_success_criteria(intent, stats)
    if success_criteria.get("passed") is not None:
        metadata["success_criteria_passed"] = (
            "yes" if success_criteria["passed"] else "no"
        )

    ReportFactory.generate_report(
        format_name=format_name,
        experiment_name=name,
        summary=summary,
        output_path=output_path,
        metadata=metadata,
        success_criteria=success_criteria,
        research_context=research_context,
    )

    logger.info(f"Report generated at {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate summary report from experiment metrics"
    )
    parser.add_argument(
        "name",
        type=validate_experiment_name,
        help="Experiment name",
    )
    parser.add_argument(
        "--format",
        "-F",
        choices=ReportFactory.list_formats(),
        default="markdown",
        help="Report output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Custom output path for report",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version="%(prog)s 1.1.0",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    configure_logging()

    if sys.version_info < (3, 9):
        logger.error("Error: Python 3.9 or later is required")
        return 1

    args = parse_args()
    result = summarize(args.name, args.format, args.output)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
