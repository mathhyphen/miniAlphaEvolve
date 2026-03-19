"""Configuration system for AlphaEvolve experiments.

This module provides dataclass-based configuration with optional Hydra support.
Hydra is used for loading YAML configuration files, but the module works
without Hydra for programmatic configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Hydra is optional - only needed for loading YAML configs
try:
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    OmegaConf = None  # type: ignore


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str = "alphaevolve_experiment"
    seed: int = 42
    output_dir: str = "outputs/${experiment.name}"


@dataclass
class EvolutionConfig:
    """Evolution hyperparameters configuration."""
    population_size: int = 10
    max_generations: int = 5
    elitism_count: int = 2
    mutation_rate: float = 0.7
    selection_pressure: int = 3


@dataclass
class MutationConfig:
    """Mutation engine configuration."""
    use_llm: bool = False
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class EvaluationConfig:
    """Evaluation pipeline configuration."""
    function_name: str = "func"
    partial_credit: bool = True
    timeout_per_test: float = 5.0
    early_terminate: bool = True


@dataclass
class FeatureDimensionConfig:
    """Feature dimension configuration for archive."""
    name: str
    n_bins: int = 10
    max_complexity: Optional[int] = None
    max_size: Optional[int] = None


@dataclass
class ArchiveConfig:
    """MAP-Elites archive configuration."""
    enabled: bool = True
    bins_per_dimension: int = 10
    feature_dimensions: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "complexity", "n_bins": 10, "max_complexity": 50},
            {"name": "code_size", "n_bins": 10, "max_size": 100},
        ]
    )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    save_to_file: bool = True
    console_output: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    enabled: bool = True
    save_interval: int = 1
    save_best_only: bool = False
    include_population: bool = True


@dataclass
class AlphaEvolveConfig:
    """Root configuration for AlphaEvolve experiments."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> AlphaEvolveConfig:
    """Load configuration from Hydra config file.

    Args:
        config_path: Path to config file (e.g., "config/experiment.yaml")
        overrides: List of Hydra override strings (e.g., ["evolution.population_size=20"])

    Returns:
        AlphaEvolveConfig instance with loaded configuration

    Raises:
        ImportError: If Hydra is not installed and config_path is provided
    """
    if config_path is None:
        # Return default config - no Hydra needed
        return AlphaEvolveConfig()

    # Hydra is required for loading YAML configs
    if not HYDRA_AVAILABLE:
        raise ImportError(
            "Hydra is required for loading YAML configuration files. "
            "Install with: pip install hydra-core"
        )

    import hydra

    # Load config using Hydra
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name="experiment", overrides=overrides or [])
        return _omegaconf_to_dataclass(cfg)


def _omegaconf_to_dataclass(omega_cfg) -> AlphaEvolveConfig:
    """Convert OmegaConf to dataclass."""
    if OmegaConf is None:
        raise ImportError("OmegaConf is not available")

    # Convert to dict first, then to dataclass
    cfg_dict = OmegaConf.to_container(omega_cfg, resolve=True)

    return AlphaEvolveConfig(
        experiment=ExperimentConfig(**cfg_dict.get('experiment', {})),
        evolution=EvolutionConfig(**cfg_dict.get('evolution', {})),
        mutation=MutationConfig(**cfg_dict.get('mutation', {})),
        evaluation=EvaluationConfig(**cfg_dict.get('evaluation', {})),
        archive=ArchiveConfig(**cfg_dict.get('archive', {})),
        logging=LoggingConfig(**cfg_dict.get('logging', {})),
        checkpoint=CheckpointConfig(**cfg_dict.get('checkpoint', {})),
    )
