"""Immutable scientific-research evaluator for experiment <name>."""

from __future__ import annotations

from alphaevolve.autoresearch import objective_evaluation


EXPERIMENT_NAME = "<name>"
EXPERIMENT_KIND = "scientific_research"
BUDGET_SECONDS = 5.0
PRIMARY_METRIC = "val_mae"

TRAIN_SAMPLES = [
    (-2.0, 17.0),
    (-1.0, 6.0),
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 9.0),
]

VAL_SAMPLES = [
    (-1.5, 10.75),
    (0.5, 0.75),
    (1.5, 4.75),
]


def evaluate_candidate(candidate_module):
    """Evaluate a candidate scientific hypothesis on held-out observations."""

    if not hasattr(candidate_module, "predict"):
        return objective_evaluation(
            metric_name=PRIMARY_METRIC,
            primary_metric=float("inf"),
            direction="minimize",
            passed=False,
            summary="candidate.py must define predict(x)",
            failed_checks=1,
        )

    train_errors = []
    val_errors = []
    for x, y in TRAIN_SAMPLES:
        train_errors.append(abs(candidate_module.predict(x) - y))
    for x, y in VAL_SAMPLES:
        val_errors.append(abs(candidate_module.predict(x) - y))

    train_mae = sum(train_errors) / len(train_errors)
    val_mae = sum(val_errors) / len(val_errors)

    return objective_evaluation(
        metric_name=PRIMARY_METRIC,
        primary_metric=val_mae,
        direction="minimize",
        passed=val_mae <= 0.25,
        summary=f"train_mae={train_mae:.4f}, val_mae={val_mae:.4f}",
        failed_checks=0 if val_mae <= 0.25 else 1,
        details={
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
        },
    )
