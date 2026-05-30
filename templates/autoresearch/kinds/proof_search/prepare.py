"""Immutable proof-search evaluator for experiment <name>."""

from __future__ import annotations

from pathlib import Path

from alphaevolve.autoresearch import proof_evaluation
from run.prove_sorting import build_sorting_harness


EXPERIMENT_NAME = "<name>"
EXPERIMENT_KIND = "proof_search"
BUDGET_SECONDS = 5.0


def evaluate_candidate(candidate_module):
    """Replace the sorting harness with your real conjecture harness."""

    candidate_path = Path(candidate_module.__file__)
    candidate_code = candidate_path.read_text(encoding="utf-8")
    summary = build_sorting_harness().verify(candidate_code)
    return proof_evaluation(summary, metric_name="proof_score")
