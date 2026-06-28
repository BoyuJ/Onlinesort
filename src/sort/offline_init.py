from __future__ import annotations

# Compatibility module for older imports. The graph-based warmup code was
# pruned; online warmup now uses HDBSCAN labels built in scripts/run_online.py.
from .types import InitializedUnit, normalize_embedding_matrix

__all__ = ["InitializedUnit", "normalize_embedding_matrix"]
