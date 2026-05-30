#!/usr/bin/env python3
"""Run one autoresearch-style experiment iteration."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaevolve.autoresearch import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one autoresearch-style experiment")
    parser.add_argument("experiment", type=str, help="Path to the experiment directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = run_experiment(Path(args.experiment))

    print("Autoresearch Run")
    print("=" * 60)
    print(f"Run ID: {record.run_id}")
    print(f"Experiment kind: {record.experiment_kind}")
    print(f"Metric: {record.metric_name} ({record.direction})")
    print(f"Primary metric: {record.primary_metric:.6f}")
    print(f"Passed: {record.passed}")
    print(f"Budget OK: {record.budget_ok} ({record.elapsed_seconds:.3f}s / {record.budget_seconds:.3f}s)")
    print(f"Keep: {record.keep}")
    print(f"Summary: {record.summary}")

    if record.keep:
        print("\nDecision: keep this candidate.")
        return 0

    print("\nDecision: do not keep this candidate.")
    return 2 if record.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
