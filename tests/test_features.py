"""Tests for MAP-Elites feature dimensions."""

import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from steiner_search.features import (
    NumSteinerPointsFeature,
    HeuristicTypeFeature,
    TopologyComplexityFeature,
    get_steiner_feature_dimensions,
)


def test_num_steiner_points_feature():
    """Verify Steiner point count extraction."""
    feature = NumSteinerPointsFeature(n_bins=10, max_points=20)

    # Code with no Steiner indicators
    code_no_steiner = "def mst(points): return 0"
    count = feature.extract(code_no_steiner)
    assert count >= 0

    # Code with Steiner indicators
    code_with_steiner = """
def add_steiner_point(points):
    steiner_points = []
    fermat = compute_fermat()
    return steiner_points
"""
    count = feature.extract(code_with_steiner)
    assert count > 0


def test_heuristic_type_feature():
    """Verify heuristic type detection."""
    feature = HeuristicTypeFeature(n_bins=10)

    # Fermat-based code
    fermat_code = """
def fermat_point(p1, p2, p3):
    # Torricelli construction with 120 degree angles
    pass
"""
    score = feature.extract(fermat_code)
    assert score == 1.0

    # Centroid-based code
    centroid_code = """
def centroid(points):
    # Average/mean center
    pass
"""
    score = feature.extract(centroid_code)
    assert score == 3.0


def test_topology_complexity_feature():
    """Verify topology complexity extraction."""
    feature = TopologyComplexityFeature(n_bins=8, max_complexity=40)

    # Simple code
    simple_code = "def f(x): return x + 1"
    complexity = feature.extract(simple_code)
    assert complexity >= 1

    # Complex code
    complex_code = """
def complex_function(data):
    for i in range(10):
        if i > 5:
            while True:
                try:
                    pass
                except:
                    break
    return data
"""
    complexity_complex = feature.extract(complex_code)
    complexity_simple = feature.extract(simple_code)
    assert complexity_complex > complexity_simple


def test_get_steiner_feature_dimensions():
    """Verify feature dimension factory."""
    features = get_steiner_feature_dimensions()
    assert len(features) >= 3
    assert all(hasattr(f, 'extract') for f in features)
    assert all(hasattr(f, 'get_bins') for f in features)
