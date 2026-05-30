#!/usr/bin/env python3
"""Report generation factory supporting multiple output formats."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


def _format_value(value: Any) -> str:
    """Format report values consistently."""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _format_context_value(value: Any) -> str:
    """Format structured context fields for reports."""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _format_value(value)


class ReportFormat(ABC):
    """Abstract base class for report formats."""

    @abstractmethod
    def generate(
        self,
        name: str,
        summary: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate report content."""

    @abstractmethod
    def extension(self) -> str:
        """Return file extension."""


class MarkdownReport(ReportFormat):
    """Markdown format report."""

    def generate(
        self,
        name: str,
        summary: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines: List[str] = [
            f"# Experiment Report: {name}",
            "",
            "## Summary Metrics",
            "",
        ]

        for key, value in sorted(summary.items()):
            lines.append(f"- **{key}**: {value:.4f}")

        if success_criteria and success_criteria.get("metrics"):
            lines.extend(["", "## Success Criteria", ""])
            overall = success_criteria.get("passed")
            if overall is not None:
                lines.append(f"- **overall**: {'PASS' if overall else 'FAIL'}")
            for metric in success_criteria["metrics"]:
                status = "PASS" if metric.get("passed") else "FAIL"
                observed = metric.get("observed")
                if observed is None:
                    lines.append(f"- **{metric['name']}**: {status} (metric not found in logs)")
                    continue
                operator = "<=" if metric.get("direction") == "lower_is_better" else ">="
                lines.append(
                    "- **{name}**: {status} ({observed} {operator} {threshold}, source `{source}`)".format(
                        name=metric["name"],
                        status=status,
                        observed=_format_value(observed),
                        operator=operator,
                        threshold=_format_value(metric.get("threshold")),
                        source=metric.get("source_key", "unknown"),
                    )
                )

        if metadata:
            lines.extend(["", "## Metadata", ""])
            for key, value in sorted(metadata.items()):
                lines.append(f"- **{key}**: {value}")

        if research_context:
            lines.extend(["", "## Research Context", ""])
            for key, value in sorted(research_context.items()):
                lines.append(f"- **{key}**: {_format_context_value(value)}")

        lines.append("")
        return "\n".join(lines)

    def extension(self) -> str:
        return ".md"


class JSONReport(ReportFormat):
    """JSON format report."""

    def generate(
        self,
        name: str,
        summary: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        report_data: Dict[str, Any] = {
            "experiment": name,
            "summary": summary,
        }
        if metadata:
            report_data["metadata"] = metadata
        if success_criteria:
            report_data["success_criteria"] = success_criteria
        if research_context:
            report_data["research_context"] = research_context
        return json.dumps(report_data, indent=2)

    def extension(self) -> str:
        return ".json"


class HTMLReport(ReportFormat):
    """HTML format report."""

    def generate(
        self,
        name: str,
        summary: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines: List[str] = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"  <title>Experiment Report: {name}</title>",
            "  <style>",
            "    body { font-family: sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; }",
            "    h1 { color: #333; }",
            "    table { border-collapse: collapse; width: 100%; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "    th { background-color: #4CAF50; color: white; }",
            "  </style>",
            "</head>",
            "<body>",
            f"  <h1>Experiment Report: {name}</h1>",
            "  <h2>Summary Metrics</h2>",
            "  <table>",
            "    <tr><th>Metric</th><th>Value</th></tr>",
        ]

        for key, value in sorted(summary.items()):
            lines.append(f"    <tr><td>{key}</td><td>{value:.4f}</td></tr>")

        lines.append("  </table>")

        if success_criteria and success_criteria.get("metrics"):
            lines.extend([
                "  <h2>Success Criteria</h2>",
                "  <table>",
                "    <tr><th>Metric</th><th>Status</th><th>Observed</th><th>Threshold</th><th>Source</th></tr>",
            ])
            for metric in success_criteria["metrics"]:
                status = "PASS" if metric.get("passed") else "FAIL"
                observed = "missing"
                if metric.get("observed") is not None:
                    observed = _format_value(metric["observed"])
                threshold = _format_value(metric.get("threshold"))
                source = metric.get("source_key", "unknown")
                lines.append(
                    f"    <tr><td>{metric['name']}</td><td>{status}</td><td>{observed}</td><td>{threshold}</td><td>{source}</td></tr>"
                )
            lines.append("  </table>")

        if metadata:
            lines.extend([
                "  <h2>Metadata</h2>",
                "  <table>",
                "    <tr><th>Key</th><th>Value</th></tr>",
            ])
            for key, value in sorted(metadata.items()):
                lines.append(f"    <tr><td>{key}</td><td>{value}</td></tr>")
            lines.append("  </table>")

        if research_context:
            lines.extend([
                "  <h2>Research Context</h2>",
                "  <table>",
                "    <tr><th>Key</th><th>Value</th></tr>",
            ])
            for key, value in sorted(research_context.items()):
                lines.append(
                    f"    <tr><td>{key}</td><td>{_format_context_value(value)}</td></tr>"
                )
            lines.append("  </table>")

        lines.extend(["</body>", "</html>"])
        return "\n".join(lines)

    def extension(self) -> str:
        return ".html"


class ReportFactory:
    """Factory for creating report generators."""

    _formats: Dict[str, ReportFormat] = {
        "markdown": MarkdownReport(),
        "md": MarkdownReport(),
        "json": JSONReport(),
        "html": HTMLReport(),
    }

    @classmethod
    def get_format(cls, name: str) -> ReportFormat:
        """Get report format by name."""
        name_lower = name.lower()
        if name_lower not in cls._formats:
            available = ", ".join(sorted(set(cls._formats.keys())))
            raise ValueError(f"Unknown report format: '{name}'. Available: {available}")
        return cls._formats[name_lower]

    @classmethod
    def list_formats(cls) -> List[str]:
        """List available format names."""
        return sorted(set(cls._formats.keys()))

    @classmethod
    def register_format(cls, name: str, formatter: ReportFormat) -> None:
        """Register a new report format."""
        cls._formats[name.lower()] = formatter

    @classmethod
    def generate_report(
        cls,
        format_name: str,
        experiment_name: str,
        summary: Dict[str, float],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Generate and save a report."""
        formatter = cls.get_format(format_name)
        content = formatter.generate(
            experiment_name,
            summary,
            metadata,
            success_criteria,
            research_context,
        )

        if not str(output_path).endswith(formatter.extension()):
            output_path = output_path.with_suffix(formatter.extension())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        return output_path
