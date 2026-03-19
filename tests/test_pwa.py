"""Tests for Population-Wide Archive (PWA) system."""

import pytest
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from alphaevolve.archive.pwa import (
    AlgorithmicStrategy,
    LatencyImprovement,
    CodeSnapshot,
    PerformanceTier,
    PWArchiveConfig,
    PopulationWideArchive,
    PWARetrieval,
    StrategyDetector,
    LatencyCalculator,
)


class TestLatencyImprovement:
    """Tests for LatencyImprovement dataclass."""

    def test_improvement_creation(self):
        """Test creating a latency improvement."""
        improvement = LatencyImprovement(
            improvement_ratio=0.2,
            absolute_improvement=20.0,
            baseline_latency=100.0,
            achieved_latency=80.0,
        )
        assert improvement.improvement_ratio == 0.2
        assert improvement.absolute_improvement == 20.0
        assert improvement.baseline_latency == 100.0
        assert improvement.achieved_latency == 80.0

    def test_improvement_clamping(self):
        """Test that improvement ratio is clamped to valid range."""
        improvement = LatencyImprovement(
            improvement_ratio=1.5,  # Invalid, should be clamped
            absolute_improvement=20.0,
            baseline_latency=100.0,
            achieved_latency=80.0,
        )
        assert improvement.improvement_ratio == 1.0  # Clamped to max


class TestPerformanceTier:
    """Tests for PerformanceTier."""

    def test_default_tiers(self):
        """Test default tier creation."""
        tiers = PerformanceTier.get_default_tiers()
        assert len(tiers) == 6
        assert tiers[0].name == "counterexample"
        assert tiers[-1].name == "optimal"

    def test_tier_contains(self):
        """Test tier contains method."""
        tier = PerformanceTier("good", 0.3, 0.6)
        assert tier.contains(0.3)
        assert tier.contains(0.5)
        assert not tier.contains(0.6)  # max is exclusive
        assert not tier.contains(0.0)


class TestStrategyDetector:
    """Tests for StrategyDetector."""

    def test_detect_fermat(self):
        """Test detection of Fermat-based strategy."""
        detector = StrategyDetector()
        code = """
def fermat_point(points):
    # Use Fermat point calculation
    return calculate_120_degree_angle(points)
"""
        strategy = detector.detect(code)
        assert strategy == AlgorithmicStrategy.FERMAT_BASED

    def test_detect_centroid(self):
        """Test detection of centroid-based strategy."""
        detector = StrategyDetector()
        code = """
def compute_centroid(points):
    return sum(p.x for p in points) / len(points)
"""
        strategy = detector.detect(code)
        assert strategy == AlgorithmicStrategy.CENTROID_BASED

    def test_detect_unknown(self):
        """Test detection with no keywords."""
        detector = StrategyDetector()
        code = """
def solve(x):
    return x * 2
"""
        strategy = detector.detect(code)
        assert strategy == AlgorithmicStrategy.UNKNOWN


class TestLatencyCalculator:
    """Tests for LatencyCalculator."""

    def test_calculate_improvement(self):
        """Test latency improvement calculation."""
        calculator = LatencyCalculator(baseline_latency=100.0)
        improvement = calculator.calculate(achieved_latency=80.0)
        assert abs(improvement.improvement_ratio - 0.2) < 1e-9
        assert improvement.absolute_improvement == 20.0

    def test_calculate_worse(self):
        """Test when solution is worse than baseline."""
        calculator = LatencyCalculator(baseline_latency=100.0)
        improvement = calculator.calculate(achieved_latency=120.0)
        # Negative improvement (worse than baseline)
        assert improvement.improvement_ratio <= 0


class TestPopulationWideArchive:
    """Tests for PopulationWideArchive."""

    def test_archive_creation(self):
        """Test archive creation with config."""
        config = PWArchiveConfig(baseline_latency=100.0)
        archive = PopulationWideArchive(config)
        assert archive.config.baseline_latency == 100.0

    def test_add_solution(self):
        """Test adding a solution to archive."""
        config = PWArchiveConfig(baseline_latency=100.0)
        archive = PopulationWideArchive(config)

        code = """
def solve_steiner(points):
    return 0.9  # 10% improvement
"""
        snapshot = archive.add(
            code=code,
            latency=90.0,
            generation=1,
        )
        assert snapshot is not None
        assert snapshot.latency == 90.0

    def test_get_by_strategy(self):
        """Test retrieving solutions by strategy."""
        config = PWArchiveConfig(baseline_latency=100.0)
        archive = PopulationWideArchive(config)

        fermat_code = """
def fermat_solver(points):
    return 0.85
"""
        archive.add(code=fermat_code, latency=85.0, generation=1)

        centroid_code = """
def centroid_solver(points):
    return 0.90
"""
        archive.add(code=centroid_code, latency=90.0, generation=1)

        fermat_solutions = archive.get_by_strategy(AlgorithmicStrategy.FERMAT_BASED)
        assert len(fermat_solutions) == 1
        assert fermat_solutions[0].strategy == AlgorithmicStrategy.FERMAT_BASED

    def test_get_best_overall(self):
        """Test getting best solution overall."""
        config = PWArchiveConfig(baseline_latency=100.0)
        archive = PopulationWideArchive(config)

        archive.add(code="code1", latency=90.0, generation=1)
        archive.add(code="code2", latency=80.0, generation=1)
        archive.add(code="code3", latency=70.0, generation=1)

        best = archive.get_best_overall()
        assert best is not None
        assert best.latency == 70.0

    def test_counterexamples(self):
        """Test counterexample storage."""
        config = PWArchiveConfig(
            baseline_latency=100.0,
            enable_counterexamples=True,
        )
        archive = PopulationWideArchive(config)

        # Add poor solution (counterexample)
        archive.add(code="poor_solution", latency=101.0, generation=1)

        counterexamples = archive.get_counterexamples()
        assert len(counterexamples) == 1

    def test_save_load(self):
        """Test archive persistence."""
        config = PWArchiveConfig(baseline_latency=100.0)
        archive = PopulationWideArchive(config)

        archive.add(code="test_code", latency=85.0, generation=1)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            archive.save(temp_path)
            loaded = PopulationWideArchive.load(temp_path)

            assert len(loaded.get_all_snapshots()) == 1
            best = loaded.get_best_overall()
            assert best is not None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_stats(self):
        """Test archive statistics."""
        config = PWArchiveConfig(baseline_latency=100.0)
        archive = PopulationWideArchive(config)

        archive.add(code="code1", latency=90.0, generation=1)
        archive.add(code="code2", latency=80.0, generation=1)

        stats = archive.get_stats()
        assert stats.total_snapshots == 2
        assert stats.total_added == 2


class TestPWARetrieval:
    """Tests for PWARetrieval."""

    def test_find_missing_improvements(self):
        """Test finding missing strategy-tier combinations."""
        config = PWArchiveConfig(baseline_latency=100.0)
        archive = PopulationWideArchive(config)

        # Add only fermat solution
        archive.add(
            code="fermat_code",
            latency=85.0,
            generation=1,
            strategy=AlgorithmicStrategy.FERMAT_BASED,
        )

        retrieval = PWARetrieval(archive)
        missing = retrieval.find_missing_improvements()

        # Most combinations should be missing
        assert len(missing) > 0

    def test_diversity_score(self):
        """Test diversity score calculation."""
        config = PWArchiveConfig(baseline_latency=100.0)
        archive = PopulationWideArchive(config)

        retrieval = PWARetrieval(archive)
        score = retrieval.get_algorithm_diversity_score()
        assert score == 0.0  # Empty archive

        # Add solutions of different strategies
        archive.add(code="c1", latency=85.0, generation=1, strategy=AlgorithmicStrategy.FERMAT_BASED)
        archive.add(code="c2", latency=85.0, generation=1, strategy=AlgorithmicStrategy.CENTROID_BASED)

        score = retrieval.get_algorithm_diversity_score()
        assert score > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
