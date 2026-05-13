"""CLI for the deterministic 20-problem benchmark suite.

Usage:
    python -m run.problem_suite
    python -m run.problem_suite --case mst_01 --case steiner_01
    python -m run.problem_suite --family mst
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from alphaevolve.benchmarks.problem_suite import (
    BenchmarkResult,
    build_20_problem_suite,
    filter_cases,
    run_problem_suite,
    summarize_results,
)
from alphaevolve.sandbox.executor import ExecutionConfig


def _default_output_dir() -> Path:
    path = Path(f"outputs/problem_suite_{datetime.now():%Y%m%d_%H%M%S}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _render_markdown(results: Sequence[BenchmarkResult]) -> str:
    summary = summarize_results(results)
    lines = [
        "# 20-Problem Evaluation Report",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- Passed cases: {summary['passed_cases']}",
        f"- Successful executions: {summary['successful_executions']}",
        "",
        "## Family Summary",
        "",
    ]

    for family, family_summary in summary["families"].items():
        lines.append(
            f"- `{family}`: {family_summary['passed_cases']}/{family_summary['total_cases']} passed"
        )

    lines.extend(["", "## Case Results", ""])
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"- `{result.case_id}` [{result.family}] {status}: {result.validation_message}"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic 20-problem evaluation suite.")
    parser.add_argument("--case", dest="cases", action="append", default=None, help="Case ID to run. Can be provided multiple times.")
    parser.add_argument("--family", choices=["mst", "steiner"], default=None, help="Filter by problem family.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-case execution timeout in seconds.")
    parser.add_argument("--memory-limit-mb", type=float, default=256.0, help="Per-case memory limit in MB.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for JSON and Markdown reports.")
    args = parser.parse_args()

    all_cases = build_20_problem_suite()
    selected_cases = filter_cases(all_cases, case_ids=args.cases, family=args.family)
    if not selected_cases:
        raise SystemExit("No benchmark cases selected.")

    results = run_problem_suite(
        selected_cases,
        config=ExecutionConfig(
            timeout_seconds=args.timeout,
            memory_limit_mb=args.memory_limit_mb,
        ),
    )

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "results.json"
    md_path = output_dir / "report.md"

    json_path.write_text(json.dumps([result.__dict__ for result in results], indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(results), encoding="utf-8")

    summary = summarize_results(results)
    print(json.dumps(summary, indent=2))
    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
