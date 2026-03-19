"""Tests for Steiner Tree test cases."""

import math
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from steiner_search.test_cases import (
    KNOWN_OPTIMAL_CASES,
    get_test_case,
    get_test_cases_by_point_count,
    generate_random_case,
)


def test_known_optimal_cases_loaded():
    """Verify at least 6 test cases are loaded."""
    assert len(KNOWN_OPTIMAL_CASES) >= 6


def test_equilateral_triangle_optimal():
    """Verify equilateral triangle optimal ratio is sqrt(3)/2."""
    tc = get_test_case("equilateral_triangle_3")
    expected = math.sqrt(3) / 2
    assert abs(tc.optimal_ratio - expected) < 1e-6


def test_square_optimal():
    """Verify square optimal ratio."""
    tc = get_test_case("square_4")
    # Known optimal for unit square
    assert 0.91 < tc.optimal_ratio < 0.92


def test_get_test_cases_by_point_count():
    """Verify filtering by point count works."""
    cases_3 = get_test_cases_by_point_count(3)
    cases_4 = get_test_cases_by_point_count(4)
    assert len(cases_3) >= 1
    assert len(cases_4) >= 1


def test_generate_random_case():
    """Verify random case generation."""
    tc = generate_random_case(5, seed=42)
    assert len(tc.points) == 5
    assert tc.name == "random_5_42"


def test_get_test_case_not_found():
    """Verify ValueError for unknown test case."""
    try:
        get_test_case("nonexistent_case")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
