"""Tests for Steiner Tree evaluators."""

import math
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from steiner_search.evaluator import (
    mst_length,
    euclidean_distance,
    OptimalVerificationEvaluator,
    MSTBaselineEvaluator,
    CompositeSteinerEvaluator,
)
from steiner_search.test_cases import (
    KNOWN_OPTIMAL_CASES,
    get_test_case,
)


def test_euclidean_distance_basic():
    """Verify Euclidean distance calculation."""
    p1 = (0.0, 0.0)
    p2 = (3.0, 4.0)
    assert abs(euclidean_distance(p1, p2) - 5.0) < 1e-9


def test_euclidean_distance_same_point():
    """Verify distance to same point is zero."""
    p = (1.0, 2.0)
    assert euclidean_distance(p, p) == 0.0


def test_mst_length_basic():
    """Verify MST length calculation."""
    # Right triangle: MST should be sum of two shorter edges
    points = [(0, 0), (1, 0), (0, 1)]
    mst_len = mst_length(points)
    assert mst_len > 0
    # MST should connect (0,0)-(1,0) and (0,0)-(0,1) = 2.0
    assert abs(mst_len - 2.0) < 0.01


def test_mst_length_single_point():
    """Verify MST with single point."""
    assert mst_length([(0, 0)]) == 0.0


def test_mst_length_empty():
    """Verify MST with no points."""
    assert mst_length([]) == 0.0


def test_mst_length_collinear():
    """Verify MST length for collinear points."""
    # Points on a line: MST = sum of consecutive distances
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    mst_len = mst_length(points)
    # Should be 3.0 (connecting consecutive points)
    assert abs(mst_len - 3.0) < 1e-9


def test_mst_length_square():
    """Verify MST length for unit square."""
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    mst_len = mst_length(points)
    # MST for square = 3 edges of length 1 = 3.0
    assert abs(mst_len - 3.0) < 0.01


def test_optimal_verification_evaluator_basic():
    """Verify optimal verification evaluator runs."""
    # Use just the equilateral triangle case
    test_cases = [get_test_case("equilateral_triangle_3")]

    evaluator = OptimalVerificationEvaluator(
        test_cases=test_cases,
        weight=0.625
    )

    # Create a simple code that returns the optimal ratio
    code = '''
import math

def find_steiner_ratio(points):
    """Return optimal ratio for equilateral triangle."""
    return math.sqrt(3) / 2
'''

    result = evaluator.evaluate(code)
    assert result.fitness >= 0
    assert result.passed == True


def test_optimal_verification_evaluator_partial_credit():
    """Verify partial credit is awarded."""
    test_cases = [get_test_case("equilateral_triangle_3")]

    evaluator = OptimalVerificationEvaluator(
        test_cases=test_cases,
        weight=0.625
    )

    # Return a suboptimal ratio
    code = '''
def find_steiner_ratio(points):
    return 0.9  # Close to optimal 0.866
'''

    result = evaluator.evaluate(code)
    # Should get some credit for being close
    assert result.fitness > 0


def test_optimal_verification_evaluator_execution_failure():
    """Verify handling of execution failure."""
    test_cases = [get_test_case("equilateral_triangle_3")]

    evaluator = OptimalVerificationEvaluator(
        test_cases=test_cases,
        weight=0.625
    )

    # Code without required function
    code = '''
def wrong_function(points):
    return 0.5
'''

    result = evaluator.evaluate(code)
    assert result.fitness == 0.0
    assert result.passed == False


def test_mst_baseline_evaluator_basic():
    """Verify MST baseline evaluator runs."""
    test_cases = [get_test_case("equilateral_triangle_3")]

    evaluator = MSTBaselineEvaluator(
        test_cases=test_cases,
        weight=0.375
    )

    # Code that achieves optimal ratio
    code = '''
import math

def find_steiner_ratio(points):
    return math.sqrt(3) / 2
'''

    result = evaluator.evaluate(code)
    assert result.fitness >= 0
    # Should show improvement over MST (ratio < 1.0)
    assert "IMPROVED" in result.feedback or result.fitness > 0


def test_mst_baseline_evaluator_no_improvement():
    """Verify handling when no improvement over MST."""
    test_cases = [get_test_case("equilateral_triangle_3")]

    evaluator = MSTBaselineEvaluator(
        test_cases=test_cases,
        weight=0.375
    )

    # Code that returns ratio >= 1.0 (no improvement)
    code = '''
def find_steiner_ratio(points):
    return 1.0  # Same as MST
'''

    result = evaluator.evaluate(code)
    # Should have low or zero fitness
    assert result.fitness < 50.0  # Less than half credit


def test_mst_baseline_evaluator_execution_failure():
    """Verify MST baseline handles execution failure."""
    test_cases = [get_test_case("equilateral_triangle_3")]

    evaluator = MSTBaselineEvaluator(
        test_cases=test_cases,
        weight=0.375
    )

    code = '''
def wrong_function(points):
    return 0.5
'''

    result = evaluator.evaluate(code)
    assert result.fitness == 0.0


def test_composite_evaluator_basic():
    """Verify composite evaluator combines components."""
    evaluator = CompositeSteinerEvaluator(
        test_cases=KNOWN_OPTIMAL_CASES[:2],  # Use first 2 cases
        use_geosteiner=False
    )

    # Code that achieves optimal ratio
    code = '''
import math

def find_steiner_ratio(points):
    return math.sqrt(3) / 2
'''

    result = evaluator.evaluate(code)
    assert result.fitness >= 0
    assert "OptimalVerification" in result.feedback or "MSTBaseline" in result.feedback


def test_composite_evaluator_multiple_test_cases():
    """Verify composite evaluator handles multiple test cases."""
    evaluator = CompositeSteinerEvaluator(
        test_cases=KNOWN_OPTIMAL_CASES[:3],  # Use first 3 cases
        use_geosteiner=False
    )

    # Optimal code
    code = '''
import math

def find_steiner_ratio(points):
    n = len(points)
    if n == 3:
        return math.sqrt(3) / 2
    elif n == 4:
        # Return reasonable approximation
        return 0.91
    return 0.87
'''

    result = evaluator.evaluate(code)
    assert result.fitness > 0


def test_composite_evaluator_weights():
    """Verify composite evaluator uses correct weights."""
    evaluator = CompositeSteinerEvaluator(
        test_cases=KNOWN_OPTIMAL_CASES[:2],
        use_geosteiner=False
    )

    # Verify weights sum to 1.0
    total_weight = sum(w for _, w in evaluator.evaluators)
    assert abs(total_weight - 1.0) < 1e-9


def test_collinear_case():
    """Verify collinear case where MST is optimal."""
    test_cases = [get_test_case("collinear_4")]

    evaluator = OptimalVerificationEvaluator(
        test_cases=test_cases,
        weight=0.625
    )

    # For collinear points, optimal ratio is 1.0
    code = '''
def find_steiner_ratio(points):
    return 1.0
'''

    result = evaluator.evaluate(code)
    assert result.passed == True
    assert result.fitness > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
