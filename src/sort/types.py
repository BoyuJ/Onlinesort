from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class InitializedUnit:
    unit_id: int
    center: np.ndarray
    dominant_channel: int
    channel_median: float
    amp_mean: float
    amp_var: float
    first_t0: int
    last_t0: int
    n_events: int
    member_indices: np.ndarray = field(repr=False)
    source_communities: List[int] = field(default_factory=list)


def normalize_embedding_matrix(z: np.ndarray) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embedding must be 2D, got shape={arr.shape}")
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.maximum(norm, 1e-8)).astype(np.float32, copy=False)


__all__ = ["InitializedUnit", "normalize_embedding_matrix"]
