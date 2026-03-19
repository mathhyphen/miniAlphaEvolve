"""Checkpoint manager for evolution state persistence."""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """Checkpoint data structure."""
    generation: int
    best_fitness: float
    best_code: str
    population: List[Dict[str, Any]]
    archive_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CheckpointManager:
    """Manager for saving and loading evolution checkpoints.

    Features:
    - Save checkpoints at regular intervals
    - Keep only last N checkpoints
    - Save best solution separately
    - Resume from checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int = 1,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N generations
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Only save when best fitness improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only

        self._best_fitness_seen = float('-inf')
        self._checkpoint_files: List[Path] = []

        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
        logger.info(f"Save interval: {self.save_interval}, Max checkpoints: {self.max_checkpoints}")

    def should_save(self, generation: int, best_fitness: float) -> bool:
        """Check if checkpoint should be saved.

        Args:
            generation: Current generation
            best_fitness: Current best fitness

        Returns:
            True if checkpoint should be saved
        """
        # Check interval
        if generation % self.save_interval != 0:
            return False

        # Check best fitness
        if self.save_best_only:
            if best_fitness <= self._best_fitness_seen:
                return False
            self._best_fitness_seen = best_fitness

        return True

    def save(
        self,
        generation: int,
        best_code: str,
        best_fitness: float,
        population: List[Any],
        archive_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save evolution checkpoint.

        Args:
            generation: Current generation
            best_code: Best code in population
            best_fitness: Fitness of best code
            population: List of Individual objects
            archive_state: Archive state dict (optional)
            metadata: Additional metadata (optional)

        Returns:
            Path to saved checkpoint file
        """
        # Serialize population
        pop_data = []
        for ind in population:
            ind_data = {
                "id": ind.id,
                "code": ind.code,
                "fitness": ind.fitness,
                "generation": ind.generation,
            }
            if hasattr(ind, 'features') and ind.features is not None:
                ind_data["features"] = ind.features
            pop_data.append(ind_data)

        # Create checkpoint data
        checkpoint = CheckpointData(
            generation=generation,
            best_fitness=best_fitness,
            best_code=best_code,
            population=pop_data,
            archive_state=archive_state,
            metadata=metadata or {},
        )

        # Save to file
        filename = f"checkpoint_gen_{generation:04d}.json"
        checkpoint_file = self.checkpoint_dir / filename

        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2)

        # Also save as latest
        latest_file = self.checkpoint_dir / "checkpoint_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2)

        # Save best code separately
        best_file = self.checkpoint_dir / "best_code.py"
        with open(best_file, 'w') as f:
            f.write(best_code)

        # Track checkpoint files
        self._checkpoint_files.append(checkpoint_file)
        self._cleanup_old_checkpoints()

        logger.info(f"Checkpoint saved: {checkpoint_file}")
        return checkpoint_file

    def load(self, checkpoint_path: Optional[Path] = None) -> Optional[CheckpointData]:
        """Load evolution checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (default: latest)

        Returns:
            CheckpointData if checkpoint exists, None otherwise
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.json"

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)

            checkpoint = CheckpointData(**data)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"  Generation: {checkpoint.generation}")
            logger.info(f"  Best fitness: {checkpoint.best_fitness:.2f}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only max_checkpoints."""
        if len(self._checkpoint_files) <= self.max_checkpoints:
            return

        # Sort by generation number
        self._checkpoint_files.sort(key=lambda p: int(p.stem.split('_')[-1]))

        # Remove oldest
        while len(self._checkpoint_files) > self.max_checkpoints:
            old_file = self._checkpoint_files.pop(0)
            if old_file.exists():
                old_file.unlink()
                logger.debug(f"Removed old checkpoint: {old_file}")

    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints.

        Returns:
            List of checkpoint file paths
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_gen_*.json"))
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
        return checkpoints

    def get_latest_generation(self) -> int:
        """Get the latest generation number from checkpoints.

        Returns:
            Latest generation number, or -1 if no checkpoints
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return -1

        # Load latest checkpoint
        latest = checkpoints[-1]
        checkpoint = self.load(latest)
        if checkpoint:
            return checkpoint.generation
        return -1


def resume_from_checkpoint(
    checkpoint_dir: str,
    evolution,
    archive=None,
) -> int:
    """Resume evolution from checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints
        evolution: Evolution instance to resume
        archive: Optional archive to restore

    Returns:
        Generation to resume from, or -1 if no checkpoint found
    """
    manager = CheckpointManager(checkpoint_dir)
    checkpoint = manager.load()

    if checkpoint is None:
        return -1

    # Restore population
    # Note: This requires evolution to have a method to set population
    # You may need to add this to your Evolution class
    logger.info(f"Resuming from generation {checkpoint.generation}")

    # Restore archive if provided
    if archive is not None and checkpoint.archive_state is not None:
        archive.restore_from_dict(checkpoint.archive_state)
        logger.info("Archive restored")

    return checkpoint.generation
