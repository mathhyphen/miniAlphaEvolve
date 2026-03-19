"""Experiment logging and tracking for reproducible research."""

import os
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import shutil

logger = logging.getLogger(__name__)


@dataclass
class ExperimentInfo:
    """Experiment metadata for reproducibility."""
    name: str
    timestamp: str
    seed: int
    config_hash: str
    python_version: str
    git_commit: Optional[str] = None


class ExperimentLogger:
    """Structured experiment logging with automatic output management."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: Optional[str] = None,
        level: str = "INFO",
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ) -> None:
        """Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            output_dir: Base output directory (default: outputs/{experiment_name}_{timestamp})
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Logging format string
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directory
        if output_dir is None:
            output_dir = "outputs"
        self.output_dir = Path(output_dir) / f"{experiment_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(f"alphaevolve.{experiment_name}")
        self.logger.setLevel(getattr(logging, level.upper()))

        # Console handler
        if level.upper() != "CRITICAL":  # Allow CRITICAL to disable console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(console_handler)

        # File handler
        log_file = self.output_dir / "logs" / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)

        self.logger.info(f"Experiment '{experiment_name}' started")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Experiment info
        self.experiment_info = ExperimentInfo(
            name=experiment_name,
            timestamp=self.timestamp,
            seed=0,  # Will be set later
            config_hash="",  # Will be set later
            python_version=self._get_python_version(),
            git_commit=self._get_git_commit(),
        )

    def _get_python_version(self) -> str:
        """Get Python version string."""
        import platform
        return platform.python_version()

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash if available."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None

    def log_config(self, config: Any) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration object (dict, dataclass, or OmegaConf)
        """
        # Convert to dict
        if hasattr(config, '__dataclass_fields__'):
            from dataclasses import asdict
            config_dict = asdict(config)
        elif hasattr(config, 'to_container'):
            # OmegaConf
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = dict(config)

        # Save config to file
        config_file = self.output_dir / "configs" / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Log config hash
        config_hash = hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:12]
        self.experiment_info.config_hash = config_hash

        self.logger.info(f"Configuration saved to {config_file}")
        self.logger.debug(f"Config hash: {config_hash}")

    def log_environment(self) -> Dict[str, Any]:
        """Log environment information for reproducibility.

        Returns:
            Dictionary containing environment information
        """
        import platform

        env_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

        # Try to get torch info
        try:
            import torch
            env_info["torch_version"] = torch.__version__
            env_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env_info["cuda_version"] = torch.version.cuda
                env_info["gpu_model"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        # Try to get numpy info
        try:
            import numpy
            env_info["numpy_version"] = numpy.__version__
        except ImportError:
            pass

        # Save environment info
        env_file = self.output_dir / "configs" / "environment.json"
        with open(env_file, 'w') as f:
            json.dump(env_info, f, indent=2)

        self.logger.info(f"Environment info saved to {env_file}")
        return env_info

    def log_evolution_step(
        self,
        generation: int,
        best_fitness: float,
        avg_fitness: float,
        diversity: Optional[float] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log evolution step metrics.

        Args:
            generation: Current generation number
            best_fitness: Best fitness in population
            avg_fitness: Average fitness in population
            diversity: Population diversity metric
            extra_info: Additional metrics to log
        """
        metrics = {
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "timestamp": datetime.now().isoformat(),
        }
        if diversity is not None:
            metrics["diversity"] = diversity
        if extra_info:
            metrics.update(extra_info)

        # Append to metrics file
        metrics_file = self.output_dir / "metrics.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + "\n")

        # Log to console/file
        self.logger.info(
            f"Gen {generation}: best={best_fitness:.2f}, avg={avg_fitness:.2f}"
        )

    def log_checkpoint(
        self,
        best_code: str,
        best_fitness: float,
        generation: int,
        population: Optional[List[Any]] = None,
    ) -> None:
        """Save evolution checkpoint.

        Args:
            best_code: Best code found so far
            best_fitness: Fitness of best code
            generation: Current generation
            population: Full population (optional)
        """
        checkpoint = {
            "generation": generation,
            "best_fitness": best_fitness,
            "best_code": best_code,
            "timestamp": datetime.now().isoformat(),
        }
        if population is not None:
            # Serialize population if provided
            checkpoint["population"] = [
                {"code": ind.code, "fitness": ind.fitness}
                for ind in population
            ]

        # Save checkpoint
        checkpoint_file = self.output_dir / "checkpoints" / f"checkpoint_gen_{generation}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Also save as latest
        latest_file = self.output_dir / "checkpoints" / "checkpoint_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        self.logger.info(f"Checkpoint saved: {checkpoint_file}")

    def log_final_results(
        self,
        best_code: str,
        best_fitness: float,
        total_generations: int,
        history: List[Dict[str, Any]],
    ) -> None:
        """Log final experiment results.

        Args:
            best_code: Best code found
            best_fitness: Fitness of best code
            total_generations: Total generations run
            history: Evolution history
        """
        results = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "best_fitness": best_fitness,
            "best_code": best_code,
            "total_generations": total_generations,
            "history": history,
        }

        # Save results
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save best code to separate file
        code_file = self.output_dir / "best_code.py"
        with open(code_file, 'w') as f:
            f.write(best_code)

        self.logger.info("=" * 60)
        self.logger.info("Experiment Complete!")
        self.logger.info(f"Best fitness: {best_fitness:.2f}")
        self.logger.info(f"Results saved to {results_file}")
        self.logger.info(f"Best code saved to {code_file}")

    def save_requirements(self) -> None:
        """Save pip freeze output for reproducibility."""
        import subprocess

        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                req_file = self.output_dir / "requirements.txt"
                with open(req_file, 'w') as f:
                    f.write(result.stdout)
                self.logger.info(f"Requirements saved to {req_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save requirements: {e}")

    def get_output_dir(self) -> Path:
        """Get the output directory path.

        Returns:
            Path to experiment output directory
        """
        return self.output_dir


def create_logger(
    experiment_name: str,
    config: Optional[Any] = None,
    output_dir: Optional[str] = None,
) -> ExperimentLogger:
    """Create and initialize experiment logger.

    Args:
        experiment_name: Name of the experiment
        config: Optional configuration to log
        output_dir: Base output directory

    Returns:
        Initialized ExperimentLogger instance
    """
    # Extract logging config if available
    level = "INFO"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if config is not None:
        if hasattr(config, 'logging'):
            level = getattr(config.logging, 'level', 'INFO')
            log_format = getattr(config.logging, 'format', log_format)

    # Create logger
    logger = ExperimentLogger(
        experiment_name=experiment_name,
        output_dir=output_dir,
        level=level,
        log_format=log_format,
    )

    # Log config and environment
    if config is not None:
        logger.log_config(config)
    logger.log_environment()
    logger.save_requirements()

    return logger
