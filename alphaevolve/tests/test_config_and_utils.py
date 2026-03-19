"""Tests for configuration and utility modules."""

import pytest
import unittest
import json
from pathlib import Path
import tempfile
import shutil

from alphaevolve.config.config import (
    AlphaEvolveConfig,
    ExperimentConfig,
    EvolutionConfig,
    MutationConfig,
    EvaluationConfig,
    ArchiveConfig,
    LoggingConfig,
    CheckpointConfig,
)

from alphaevolve.utils.checkpoint import (
    CheckpointManager,
    CheckpointData,
)

from alphaevolve.core.data_structures import Individual


class TestConfiguration(unittest.TestCase):
    """Test configuration system."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AlphaEvolveConfig()

        self.assertEqual(config.experiment.seed, 42)
        self.assertEqual(config.evolution.population_size, 10)
        self.assertEqual(config.mutation.temperature, 0.7)
        self.assertTrue(config.evaluation.partial_credit)
        self.assertTrue(config.archive.enabled)
        self.assertEqual(config.logging.level, "INFO")
        self.assertTrue(config.checkpoint.enabled)

    def test_experiment_config(self) -> None:
        """Test ExperimentConfig creation."""
        exp_config = ExperimentConfig(
            name="test_experiment",
            seed=123,
            output_dir="custom_outputs",
        )

        self.assertEqual(exp_config.name, "test_experiment")
        self.assertEqual(exp_config.seed, 123)
        self.assertEqual(exp_config.output_dir, "custom_outputs")

    def test_evolution_config(self) -> None:
        """Test EvolutionConfig creation."""
        evo_config = EvolutionConfig(
            population_size=20,
            max_generations=10,
            elitism_count=3,
            mutation_rate=0.8,
        )

        self.assertEqual(evo_config.population_size, 20)
        self.assertEqual(evo_config.max_generations, 10)
        self.assertEqual(evo_config.elitism_count, 3)
        self.assertEqual(evo_config.mutation_rate, 0.8)

    def test_mutation_config(self) -> None:
        """Test MutationConfig creation."""
        mut_config = MutationConfig(
            use_llm=True,
            llm_provider="anthropic",
            llm_model="claude-3-opus-20240229",
            temperature=0.5,
        )

        self.assertTrue(mut_config.use_llm)
        self.assertEqual(mut_config.llm_provider, "anthropic")
        self.assertEqual(mut_config.temperature, 0.5)

    def test_archive_config(self) -> None:
        """Test ArchiveConfig creation."""
        arch_config = ArchiveConfig(
            enabled=True,
            bins_per_dimension=20,
        )

        self.assertTrue(arch_config.enabled)
        self.assertEqual(arch_config.bins_per_dimension, 20)


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint management."""

    def setUp(self) -> None:
        """Create temporary directory for checkpoints."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_checkpoint_manager_init(self) -> None:
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            save_interval=2,
            max_checkpoints=3,
        )

        self.assertTrue(self.checkpoint_dir.exists())
        self.assertEqual(manager.save_interval, 2)
        self.assertEqual(manager.max_checkpoints, 3)

    def test_should_save_interval(self) -> None:
        """Test checkpoint save interval logic."""
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            save_interval=2,
            save_best_only=False,
        )

        # Should save on multiples of interval
        self.assertFalse(manager.should_save(1, 50.0))
        self.assertTrue(manager.should_save(2, 50.0))
        self.assertFalse(manager.should_save(3, 50.0))
        self.assertTrue(manager.should_save(4, 50.0))

    def test_should_save_best_only(self) -> None:
        """Test best-only checkpoint logic."""
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            save_interval=1,
            save_best_only=True,
        )

        # First checkpoint should always save
        self.assertTrue(manager.should_save(1, 50.0))

        # Simulate saving to update best_fitness_seen
        manager._best_fitness_seen = 50.0

        # Worse fitness should not save
        self.assertFalse(manager.should_save(2, 45.0))

        # Better fitness should save
        self.assertTrue(manager.should_save(3, 55.0))

    def test_save_and_load_checkpoint(self) -> None:
        """Test saving and loading checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            save_interval=1,
            max_checkpoints=5,
        )

        # Create test population
        population = [
            Individual(code="def f1(): return 1", generation=0),
            Individual(code="def f2(): return 2", generation=0),
        ]
        population[0].update_fitness(80.0)
        population[1].update_fitness(60.0)

        # Save checkpoint
        best_code = "def f1(): return 1"
        manager.save(
            generation=5,
            best_code=best_code,
            best_fitness=80.0,
            population=population,
            metadata={"test_key": "test_value"},
        )

        # Verify checkpoint file exists
        checkpoint_file = self.checkpoint_dir / "checkpoint_gen_0005.json"
        self.assertTrue(checkpoint_file.exists())

        # Verify latest checkpoint exists
        latest_file = self.checkpoint_dir / "checkpoint_latest.json"
        self.assertTrue(latest_file.exists())

        # Verify best code file exists
        best_file = self.checkpoint_dir / "best_code.py"
        self.assertTrue(best_file.exists())

        # Load checkpoint
        checkpoint = manager.load(checkpoint_file)

        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint.generation, 5)
        self.assertEqual(checkpoint.best_fitness, 80.0)
        self.assertEqual(checkpoint.best_code, best_code)
        self.assertEqual(len(checkpoint.population), 2)
        self.assertEqual(checkpoint.metadata.get("test_key"), "test_value")

    def test_checkpoint_cleanup(self) -> None:
        """Test old checkpoint cleanup."""
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            save_interval=1,
            max_checkpoints=2,
        )

        # Create minimal population for saving
        population = [Individual(code="def f(): pass", generation=0)]
        population[0].update_fitness(50.0)

        # Save multiple checkpoints
        for gen in range(1, 6):
            manager.save(
                generation=gen,
                best_code="def f(): pass",
                best_fitness=float(gen * 10),
                population=population,
            )

        # Should only have max_checkpoints + latest
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_gen_*.json"))
        self.assertEqual(len(checkpoint_files), 2)

    def test_list_checkpoints(self) -> None:
        """Test listing available checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            save_interval=1,
        )

        population = [Individual(code="def f(): pass", generation=0)]
        population[0].update_fitness(50.0)

        # Save checkpoints
        for gen in [1, 3, 5, 7]:
            manager.save(
                generation=gen,
                best_code="def f(): pass",
                best_fitness=50.0,
                population=population,
            )

        checkpoints = manager.list_checkpoints()

        self.assertEqual(len(checkpoints), 4)
        # Should be sorted by generation
        self.assertEqual(checkpoints[0].stem, "checkpoint_gen_0001")
        self.assertEqual(checkpoints[-1].stem, "checkpoint_gen_0007")

    def test_get_latest_generation(self) -> None:
        """Test getting latest generation from checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            save_interval=1,
        )

        # No checkpoints yet
        self.assertEqual(manager.get_latest_generation(), -1)

        # Save a checkpoint
        population = [Individual(code="def f(): pass", generation=0)]
        population[0].update_fitness(50.0)

        manager.save(
            generation=10,
            best_code="def f(): pass",
            best_fitness=50.0,
            population=population,
        )

        self.assertEqual(manager.get_latest_generation(), 10)


class TestCheckpointData(unittest.TestCase):
    """Test CheckpointData dataclass."""

    def test_checkpoint_data_creation(self) -> None:
        """Test CheckpointData creation with default values."""
        data = CheckpointData(
            generation=5,
            best_fitness=85.0,
            best_code="def test(): return 42",
            population=[],
        )

        self.assertEqual(data.generation, 5)
        self.assertEqual(data.best_fitness, 85.0)
        self.assertIsNone(data.archive_state)
        self.assertEqual(data.metadata, {})

    def test_checkpoint_data_with_metadata(self) -> None:
        """Test CheckpointData with custom metadata."""
        metadata = {"custom_key": "custom_value", "generation_time": 123.45}

        data = CheckpointData(
            generation=5,
            best_fitness=85.0,
            best_code="def test(): return 42",
            population=[],
            metadata=metadata,
        )

        self.assertEqual(data.metadata["custom_key"], "custom_value")
        self.assertEqual(data.metadata["generation_time"], 123.45)


if __name__ == "__main__":
    unittest.main()
