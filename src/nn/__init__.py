"""Neural network modules."""

from .model import (
    ModelConfig,
    ModelOutput,
    PeakWaveEncoder,
    ResidualTCNBlock,
    SpikeEmbeddingModel,
)

__all__ = [
    "ModelConfig",
    "ModelOutput",
    "PeakWaveEncoder",
    "ResidualTCNBlock",
    "SpikeEmbeddingModel",
]
