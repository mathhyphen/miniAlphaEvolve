"""Immutable evaluator for experiment <name>.

Edit this file before the autonomous loop starts. After that, keep it fixed.
"""

from __future__ import annotations

from alphaevolve.autoresearch import objective_evaluation


EXPERIMENT_NAME = "<name>"
BUDGET_SECONDS = 30.0
EXPERIMENT_KIND = "generic"
PRIMARY_METRIC = "placeholder_score"


def evaluate_candidate(candidate_module):
    """Evaluate the mutable candidate.

    Replace this placeholder with your real metric or proof harness.
    """

    if not hasattr(candidate_module, "solve"):
        return objective_evaluation(
            metric_name=PRIMARY_METRIC,
            primary_metric=float("-inf"),
            direction="maximize",
            passed=False,
            summary="candidate.py must define solve",
            failed_checks=1,
        )

    output = candidate_module.solve(3)
    if output == 9:
        return objective_evaluation(
            metric_name=PRIMARY_METRIC,
            primary_metric=1.0,
            direction="maximize",
            passed=True,
            summary="placeholder objective passed",
        )

    return objective_evaluation(
        metric_name=PRIMARY_METRIC,
        primary_metric=0.0,
        direction="maximize",
        passed=False,
        summary="placeholder objective failed",
        failed_checks=1,
        details={"expected": 9, "actual": output},
    )
