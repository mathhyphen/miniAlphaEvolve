"""Configuration module for AlphaEvolve."""

from alphaevolve.config.config import (
    AlphaEvolveConfig,
    ExperimentConfig,
    EvolutionConfig,
    MutationConfig,
    EvaluationConfig,
    ArchiveConfig,
    LoggingConfig,
    CheckpointConfig,
    FeatureDimensionConfig,
    HYDRA_AVAILABLE,
)

# load_config is only available when Hydra is installed
if HYDRA_AVAILABLE:
    from alphaevolve.config.config import load_config
    __all__ = [
        "AlphaEvolveConfig",
        "ExperimentConfig",
        "EvolutionConfig",
        "MutationConfig",
        "EvaluationConfig",
        "ArchiveConfig",
        "LoggingConfig",
        "CheckpointConfig",
        "FeatureDimensionConfig",
        "HYDRA_AVAILABLE",
        "load_config",
    ]
else:
    __all__ = [
        "AlphaEvolveConfig",
        "ExperimentConfig",
        "EvolutionConfig",
        "MutationConfig",
        "EvaluationConfig",
        "ArchiveConfig",
        "LoggingConfig",
        "CheckpointConfig",
        "FeatureDimensionConfig",
        "HYDRA_AVAILABLE",
    ]
