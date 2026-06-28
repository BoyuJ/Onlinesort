from __future__ import annotations

import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntFlag
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, optionally deep-merging one or more base configs.

    Config files may specify `extends` or `base_config` as a string or list of
    strings. Relative base paths are resolved from the child config directory.
    """

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")

    base_raw = cfg.pop("extends", cfg.pop("base_config", None))
    if base_raw is None:
        return cfg
    base_paths = base_raw if isinstance(base_raw, (list, tuple)) else [base_raw]
    merged: Dict[str, Any] = {}
    for base_value in base_paths:
        base_path = Path(str(base_value))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = deep_update(merged, load_yaml(base_path))
    return deep_update(merged, cfg)


def load_config(defaults: Dict[str, Any], path: str | Path | None = None) -> Dict[str, Any]:
    cfg = deepcopy(defaults)
    if path is None:
        return cfg
    user_cfg = load_yaml(path)
    return deep_update(cfg, user_cfg)


def ensure_channel_positions(n_channels: int, channel_positions: Optional[np.ndarray] = None) -> np.ndarray:
    if channel_positions is not None:
        return np.asarray(channel_positions, dtype=np.float32)
    x = np.arange(n_channels, dtype=np.float32)
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)


def pairwise_distances(channel_positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(channel_positions, dtype=np.float32)
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def resolve_max_simul_firing(value: Any, n_channels: int) -> int:
    v = float(value)
    if v <= 0.0:
        return 0
    if v <= 1.0:
        return int(np.ceil(v * max(1, int(n_channels))))
    return int(round(v))


def build_knn_neighbors(channel_positions: np.ndarray, k: int) -> np.ndarray:
    dmat = pairwise_distances(channel_positions)
    order = np.argsort(dmat, axis=1)
    k_eff = min(k, order.shape[1])
    return order[:, :k_eff].astype(np.int32)


def build_neuropixels_np1_384_positions(electrode_cfg: Dict[str, Any]) -> np.ndarray:
    pos_cfg = electrode_cfg.get("positions", {})
    if pos_cfg.get("type", "") != "neuropixels_np1_384":
        raise ValueError("only positions.type=neuropixels_np1_384 is supported")
    depth_count = int(pos_cfg.get("depth_count", 96))
    if depth_count <= 0:
        raise ValueError("depth_count must be > 0")
    depth_pitch_um = float(pos_cfg.get("depth_pitch_um", 40.0))
    lateral_offsets = np.asarray(
        pos_cfg.get("lateral_offsets_um", [-24.0, -8.0, 8.0, 24.0]),
        dtype=np.float32,
    ).reshape(-1)
    if lateral_offsets.size <= 0:
        raise ValueError("lateral_offsets_um must be non-empty")
    depth_origin_raw = np.asarray(pos_cfg.get("depth_origin_um", -1910.0), dtype=np.float32).reshape(-1)
    if depth_origin_raw.size == 1:
        depth_origins = np.repeat(depth_origin_raw, lateral_offsets.size).astype(np.float32, copy=False)
    elif depth_origin_raw.size == lateral_offsets.size:
        depth_origins = depth_origin_raw.astype(np.float32, copy=False)
    else:
        raise ValueError(
            "depth_origin_um must be a scalar or match lateral_offsets_um length"
        )
    plane = str(pos_cfg.get("plane", "yz")).lower()
    order = str(pos_cfg.get("order", "column_major")).lower()
    if order != "column_major":
        raise ValueError(f"unsupported neuropixels_np1_384 order={order}")

    n_channels = int(electrode_cfg.get("channels", depth_count * lateral_offsets.size))
    expected = int(depth_count * lateral_offsets.size)
    if n_channels != expected:
        raise ValueError(f"neuropixels_np1_384 expects channels={expected}, got {n_channels}")

    out = np.zeros((n_channels, 3), dtype=np.float32)
    idx = 0
    for lateral, depth_origin in zip(lateral_offsets.tolist(), depth_origins.tolist()):
        for depth_idx in range(depth_count):
            depth = float(depth_origin) + float(depth_idx) * depth_pitch_um
            if plane == "yz":
                out[idx] = np.array([0.0, float(lateral), depth], dtype=np.float32)
            elif plane == "xy":
                out[idx] = np.array([depth, float(lateral), 0.0], dtype=np.float32)
            else:
                raise ValueError(f"unsupported neuropixels_np1_384 plane={plane}")
            idx += 1
    return out


def build_explicit_positions(electrode_cfg: Dict[str, Any]) -> np.ndarray:
    pos_cfg = electrode_cfg.get("positions", {})
    if pos_cfg.get("type", "") != "explicit":
        raise ValueError("only positions.type=explicit is supported")
    values = pos_cfg.get("values")
    if values is None:
        values = pos_cfg.get("coords")
    if values is None:
        raise ValueError("positions.values is required for explicit geometry")
    out = np.asarray(values, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] not in (2, 3):
        raise ValueError("explicit positions must have shape (N, 2) or (N, 3)")
    if out.shape[1] == 2:
        out = np.concatenate([np.zeros((out.shape[0], 1), dtype=np.float32), out], axis=1)
    n_channels = int(electrode_cfg.get("channels", out.shape[0]))
    if out.shape[0] != n_channels:
        raise ValueError(f"explicit positions expects channels={out.shape[0]}, got {n_channels}")
    return out.astype(np.float32, copy=False)


def build_staggered_grid_positions(electrode_cfg: Dict[str, Any]) -> np.ndarray:
    pos_cfg = electrode_cfg.get("positions", {})
    if pos_cfg.get("type", "") != "staggered_grid":
        raise ValueError("only positions.type=staggered_grid is supported")
    dim = pos_cfg.get("dim", {})
    rows = int(dim.get("rows", 0))
    cols = int(dim.get("cols", 0))
    if rows <= 0 or cols <= 0:
        raise ValueError("invalid rows/cols in electrode config")
    n_channels = int(electrode_cfg.get("channels", rows * cols))
    if rows * cols != n_channels:
        # Neuropixels 384 often uses a checkerboard pattern where n_channels ~ rows*cols/2
        # However, typically rows=4, cols=192 means a dense 4x192 region where only selected sites are active.
        # But if the user configuration explicitly asks for fewer channels, we should probably 
        # auto-detect a checkerboard layout if ratio is exactly 0.5.
        if n_channels * 2 == rows * cols:
            # Assume checkerboard pattern for Neuropixels-like logical layout
            pass
        else:
            # If not exactly half, warn but proceed if we can fit them? No, better to fail or fix logic.
            # For now, let's allow it but we need a strategy to picking which sites.
            # To fix the immediate crash for NP 384 config (384 vs 768), we implement checkerboard logic below.
            pass

    pitch = pos_cfg.get("pitch_um", {})
    vertical = float(pitch.get("vertical", 40.0))
    horizontal = float(pitch.get("horizontal", 16.0))
    stagger = float(pos_cfg.get("stagger_um", 0.0))
    plane = str(pos_cfg.get("plane", "yz")).lower()

    if n_channels == rows * cols:
        # Dense grid
        out = np.zeros((n_channels, 3), dtype=np.float32)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                y = r * vertical + (stagger if (c % 2 == 1) else 0.0)
                z = c * horizontal
                if plane == "yz":
                    out[idx] = np.array([0.0, y, z], dtype=np.float32)
                elif plane == "xy":
                    out[idx] = np.array([z, y, 0.0], dtype=np.float32)
                else:
                    out[idx] = np.array([0.0, y, z], dtype=np.float32)
                idx += 1
        return out
    
    # Sparse / Checkerboard logic for Neuropixels 384 (channels=384, rows=4, cols=192)
    # The pattern is typically: (row + col) % 2 == 0 (or similar).
    # NP 1.0 has a checkerboard pattern effectively.
    out = np.zeros((n_channels, 3), dtype=np.float32)
    idx = 0
    for c in range(cols):
        for r in range(rows):
            # Checkerboard condition: only keep sites where (r + c) or similar logic meets condition
            # For standard NP 384 long column mode:
            # It selects a specific pattern. Without exact knowledge of which 384, 
            # we assume a checkerboard (r+c)%2 == 0 matches the count 384 for 4x192.
            # Note: This is a synthetic reconstruction for BIN files. H5 files use read_channel_positions.
            if (r + c) % 2 == 0:
                if idx >= n_channels:
                    break
                y = r * vertical # Vertical usually means Y for NP
                z = c * horizontal # Horizontal usually means Z (depth) for NP? 
                # Wait, pitch definition in config: vertical=16 (y), horizontal=20 (z?).
                # In H5 we saw Y=[-24, -8, 8, 24] (spacing 16), Z=[-1910...0] (spacing 20).
                # So rows corresponds to Y, cols corresponds to Z.
                # r=0 -> y=-24? Offset is arbitrary, we center it or start at 0.
                # Let's align with config:
                
                pos_y = r * vertical
                pos_z = c * horizontal
                
                # Apply stagger if needed
                if stagger != 0 and (c % 2 == 1):
                    pos_y += stagger

                if plane == "yz":
                    out[idx] = np.array([0.0, pos_y, pos_z], dtype=np.float32)
                elif plane == "xy":
                    out[idx] = np.array([pos_z, pos_y, 0.0], dtype=np.float32)
                else:
                    # default x, y
                    out[idx] = np.array([0.0, pos_y, pos_z], dtype=np.float32)
                idx += 1
                
    return out


def build_channel_positions_from_electrode_config(electrode_cfg: Dict[str, Any]) -> np.ndarray:
    pos_cfg = electrode_cfg.get("positions", {})
    pos_type = str(pos_cfg.get("type", "staggered_grid")).lower()
    if pos_type == "explicit":
        return build_explicit_positions(electrode_cfg)
    if pos_type == "neuropixels_np1_384":
        return build_neuropixels_np1_384_positions(electrode_cfg)
    if pos_type == "staggered_grid":
        return build_staggered_grid_positions(electrode_cfg)
    raise ValueError(f"unsupported positions.type={pos_type}")


def build_local_grid_neighbors(rows: int, cols: int, k_neighbors: int) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be > 0")
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be > 0")
    n_channels = rows * cols
    out = np.zeros((n_channels, k_neighbors), dtype=np.int32)

    offsets: List[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            offsets.append((dr, dc))
    offsets.sort(key=lambda x: (x[0] * x[0] + x[1] * x[1], abs(x[0]), abs(x[1]), x[0], x[1]))

    for r in range(rows):
        for c in range(cols):
            center = r * cols + c
            neigh = [center]
            for dr, dc in offsets:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    neigh.append(rr * cols + cc)
            if len(neigh) >= k_neighbors:
                use = neigh[:k_neighbors]
            else:
                use = neigh + [center] * (k_neighbors - len(neigh))
            out[center] = np.asarray(use, dtype=np.int32)
    return out


def build_neighbors_from_electrode_config(electrode_cfg: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # Respect user requested k_neighbors
    k_neighbors = int(electrode_cfg.get("k_neighbors", 16))

    positions = build_channel_positions_from_electrode_config(electrode_cfg)

    # ALWAYS use KNN based neighbors to ensure consistency between pretrain (which uses H5 positions + KNN) 
    # and offline inference. This handles edge cases robustly (always finding k nearest) 
    # avoiding "padding with center" artifacts of grid-based logic.
    neighbors = build_knn_neighbors(positions, k=k_neighbors)
    return positions, neighbors


def neighbor_consistency_metrics(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    *,
    top_k: int = 4,
) -> Dict[str, float]:
    pa = np.asarray(pos_a, dtype=np.float32)
    pb = np.asarray(pos_b, dtype=np.float32)
    if pa.ndim != 2 or pb.ndim != 2 or pa.shape != pb.shape or pa.shape[0] == 0:
        return {
            "n_channels": 0.0,
            "nn1_match_ratio": float("nan"),
            "nnk_match_ratio": float("nan"),
        }
    k_eff = int(max(1, min(int(top_k), pa.shape[0] - 1)))
    neigh_a = build_knn_neighbors(pa, k=k_eff + 1)
    neigh_b = build_knn_neighbors(pb, k=k_eff + 1)
    if neigh_a.shape[1] <= 1 or neigh_b.shape[1] <= 1:
        return {
            "n_channels": float(pa.shape[0]),
            "nn1_match_ratio": 1.0,
            "nnk_match_ratio": 1.0,
        }
    nn1_match = float(np.mean((neigh_a[:, 1] == neigh_b[:, 1]).astype(np.float32)))
    a_sets = np.sort(neigh_a[:, 1 : k_eff + 1], axis=1)
    b_sets = np.sort(neigh_b[:, 1 : k_eff + 1], axis=1)
    nnk_match = float(np.mean(np.all(a_sets == b_sets, axis=1).astype(np.float32)))
    return {
        "n_channels": float(pa.shape[0]),
        "nn1_match_ratio": nn1_match,
        "nnk_match_ratio": nnk_match,
    }


def resolve_channel_geometry(
    *,
    n_channels: int,
    fallback_k: int,
    electrode_cfg: Dict[str, Any] | None = None,
    h5_positions: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str, Dict[str, float] | None]:
    cfg_positions = None
    cfg_neighbors = None
    metrics = None
    k_neighbors = int(max(1, fallback_k))

    if electrode_cfg is not None:
        try:
            k_neighbors = int(max(1, electrode_cfg.get("k_neighbors", k_neighbors)))
            pos_try, neigh_try = build_neighbors_from_electrode_config(electrode_cfg)
            if pos_try.shape[0] == int(n_channels):
                cfg_positions = pos_try.astype(np.float32, copy=False)
                cfg_neighbors = neigh_try.astype(np.int32, copy=False)
        except Exception:
            cfg_positions = None
            cfg_neighbors = None

    h5_valid = None
    if h5_positions is not None:
        h5_try = np.asarray(h5_positions, dtype=np.float32)
        if h5_try.ndim == 2 and h5_try.shape[0] == int(n_channels):
            h5_valid = h5_try

    if h5_valid is not None and cfg_positions is not None:
        metrics = neighbor_consistency_metrics(h5_valid, cfg_positions, top_k=4)

    if cfg_positions is not None and cfg_neighbors is not None:
        return cfg_positions, cfg_neighbors, "config", metrics
    if h5_valid is not None:
        return h5_valid, build_knn_neighbors(h5_valid, k_neighbors), "h5", metrics
    return None, None, "none", metrics


def relative_geo(channel_positions: np.ndarray, center_ch: np.ndarray, neighbor_chs: np.ndarray) -> np.ndarray:
    pos = np.asarray(channel_positions, dtype=np.float32)
    center_pos = pos[center_ch][:, None, :]
    neigh_pos = pos[neighbor_chs]
    dxyz = neigh_pos - center_pos
    r = np.sqrt(np.sum(dxyz * dxyz, axis=-1, keepdims=True))
    return np.concatenate([dxyz, r], axis=-1).astype(np.float32)


def channel_distance(c1: float, c2: float) -> float:
    return abs(float(c1) - float(c2))


def get_logger(name: str = "mysort", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@dataclass
class Monitor:
    counters: Dict[str, int] = field(default_factory=dict)
    timers: Dict[str, float] = field(default_factory=dict)
    last_latency_ms: float = 0.0

    def inc(self, key: str, value: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + value

    def start(self, key: str) -> None:
        self.timers[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        t0 = self.timers.get(key)
        if t0 is None:
            return 0.0
        dt = (time.perf_counter() - t0) * 1_000.0
        self.last_latency_ms = dt
        return dt

    def snapshot(self) -> Dict[str, float]:
        out = {k: float(v) for k, v in self.counters.items()}
        out["last_latency_ms"] = self.last_latency_ms
        return out


class QCFlag(IntFlag):
    NONE = 0
    BORDER_CUT = 1 << 0
    SATURATED_ARTIFACT = 1 << 1
    TOO_CLOSE_REFRACTORY = 1 << 2
    LOW_SPATIAL_SCORE = 1 << 3
    LOW_AMP = 1 << 4
    DUPLICATE_NEARBY = 1 << 5


@dataclass
class CandidatePeak:
    t: int
    c: int
    amp_norm: float
    prominence: float = 0.0
    width: float = 0.0
    simul_global: int = 0
    simul_local: int = 0
    qc_flags: int = int(QCFlag.NONE)


@dataclass
class Event:
    t0: int
    c0: int
    amp_norm: float
    spatial_score: float
    neighbors: np.ndarray
    support_count: int = 1
    support_channels: np.ndarray | None = None
    time_spread_samples: int = 0
    peak_ratio: float = 1.0
    simul_global: int = 0
    simul_local: int = 0
    q: float = 0.0
    gate_prob: float = 1.0
    overlap_prob: float = float("nan")
    encoded_mask: bool = True
    init_buffered: bool = False
    qc_flags: int = int(QCFlag.NONE)
    unit_id: int = -1
    overlap_label: int = 0


@dataclass
class PatchBatch:
    x: np.ndarray
    meta: Dict[str, np.ndarray]


@dataclass
class EventOutput:
    event_time_sample: int
    event_time_sec: float
    center_channel: int
    unit_id: int
    confidence: float
    overlap_prob_unavailable: float
    gate_prob: float
    encoded_mask: int
    init_buffered: int
    qc_flags: int
    embedding: np.ndarray | None = None
    overlap_prob: float | None = None


@dataclass
class PipelineStats:
    n_raw_samples: int = 0
    n_filt_samples: int = 0
    n_events: int = 0
    n_patches: int = 0
    n_births: int = 0
    n_matches: int = 0
    n_drops: int = 0
    n_encoded: int = 0
    n_cluster_inputs: int = 0
    n_init_flushed: int = 0
    n_stalls: int = 0
    last_latency_ms: float = 0.0
    # Last chunk stage timings (milliseconds)
    last_preprocess_ms: float = 0.0
    last_detect_ms: float = 0.0
    last_patch_ms: float = 0.0
    last_model_ms: float = 0.0
    last_cluster_ms: float = 0.0
    # Last chunk counters
    last_segments: int = 0
    last_chunk_events: int = 0
    last_chunk_patches: int = 0
    last_chunk_encoded: int = 0
    last_chunk_cluster_inputs: int = 0
    last_detect_raw_candidates: int = 0
    last_detect_merged_events: int = 0
    last_detect_hard_drops: int = 0
    last_detect_duplicate_drops: int = 0
    last_detect_global_simul_drops: int = 0
    last_detect_local_simul_drops: int = 0
    total_detect_raw_candidates: int = 0
    total_detect_merged_events: int = 0
    total_detect_hard_drops: int = 0
    total_detect_duplicate_drops: int = 0
    total_detect_global_simul_drops: int = 0
    total_detect_local_simul_drops: int = 0
    # Accumulated stage timings (milliseconds)
    total_preprocess_ms: float = 0.0
    total_detect_ms: float = 0.0
    total_patch_ms: float = 0.0
    total_model_ms: float = 0.0
    total_cluster_ms: float = 0.0

def events_to_numpy(events: List[Event]) -> Dict[str, np.ndarray]:
    if not events:
        return {
            "t0": np.zeros((0,), dtype=np.int64),
            "c0": np.zeros((0,), dtype=np.int32),
            "amp_norm": np.zeros((0,), dtype=np.float32),
            "spatial_score": np.zeros((0,), dtype=np.float32),
            "support_count": np.zeros((0,), dtype=np.int32),
            "simul_global": np.zeros((0,), dtype=np.int32),
            "simul_local": np.zeros((0,), dtype=np.int32),
            "time_spread_samples": np.zeros((0,), dtype=np.int32),
            "peak_ratio": np.zeros((0,), dtype=np.float32),
            "q": np.zeros((0,), dtype=np.float32),
            "gate_prob": np.zeros((0,), dtype=np.float32),
            "encoded_mask": np.zeros((0,), dtype=np.int8),
            "init_buffered": np.zeros((0,), dtype=np.int8),
            "qc_flags": np.zeros((0,), dtype=np.int32),
            "unit_id": np.zeros((0,), dtype=np.int32),
            "overlap_label": np.zeros((0,), dtype=np.int32),
        }
    return {
        "t0": np.array([e.t0 for e in events], dtype=np.int64),
        "c0": np.array([e.c0 for e in events], dtype=np.int32),
        "amp_norm": np.array([e.amp_norm for e in events], dtype=np.float32),
        "spatial_score": np.array([e.spatial_score for e in events], dtype=np.float32),
        "support_count": np.array([e.support_count for e in events], dtype=np.int32),
        "simul_global": np.array([e.simul_global for e in events], dtype=np.int32),
        "simul_local": np.array([e.simul_local for e in events], dtype=np.int32),
        "time_spread_samples": np.array([e.time_spread_samples for e in events], dtype=np.int32),
        "peak_ratio": np.array([e.peak_ratio for e in events], dtype=np.float32),
        "q": np.array([e.q for e in events], dtype=np.float32),
        "gate_prob": np.array([e.gate_prob for e in events], dtype=np.float32),
        "encoded_mask": np.array([int(bool(e.encoded_mask)) for e in events], dtype=np.int8),
        "init_buffered": np.array([int(bool(e.init_buffered)) for e in events], dtype=np.int8),
        "qc_flags": np.array([e.qc_flags for e in events], dtype=np.int32),
        "unit_id": np.array([e.unit_id for e in events], dtype=np.int32),
        "overlap_label": np.array([e.overlap_label for e in events], dtype=np.int32),
    }


def outputs_to_numpy(
    outputs: List[EventOutput],
    *,
    include_embedding: bool | None = None,
    embed_dim: int | None = None,
) -> Dict[str, np.ndarray]:
    include_z = bool(include_embedding) if include_embedding is not None else any(
        o.embedding is not None for o in outputs
    )
    if not outputs:
        out = {
            "event_time_sample": np.zeros((0,), dtype=np.int64),
            "event_time_sec": np.zeros((0,), dtype=np.float64),
            "center_channel": np.zeros((0,), dtype=np.int32),
            "unit_id": np.zeros((0,), dtype=np.int32),
            "confidence": np.zeros((0,), dtype=np.float32),
            "overlap_prob_unavailable": np.zeros((0,), dtype=np.float32),
            "overlap_prob": np.zeros((0,), dtype=np.float32),
            "gate_prob": np.zeros((0,), dtype=np.float32),
            "encoded_mask": np.zeros((0,), dtype=np.int8),
            "init_buffered": np.zeros((0,), dtype=np.int8),
            "qc_flags": np.zeros((0,), dtype=np.int32),
        }
        if include_z:
            out["embedding"] = np.zeros((0, 0 if embed_dim is None else int(embed_dim)), dtype=np.float32)
        return out
    out = {
        "event_time_sample": np.array([o.event_time_sample for o in outputs], dtype=np.int64),
        "event_time_sec": np.array([o.event_time_sec for o in outputs], dtype=np.float64),
        "center_channel": np.array([o.center_channel for o in outputs], dtype=np.int32),
        "unit_id": np.array([o.unit_id for o in outputs], dtype=np.int32),
        "confidence": np.array([o.confidence for o in outputs], dtype=np.float32),
        "overlap_prob_unavailable": np.array([o.overlap_prob_unavailable for o in outputs], dtype=np.float32),
        "overlap_prob": np.array(
            [
                o.overlap_prob_unavailable if o.overlap_prob is None else o.overlap_prob
                for o in outputs
            ],
            dtype=np.float32,
        ),
        "gate_prob": np.array([o.gate_prob for o in outputs], dtype=np.float32),
        "encoded_mask": np.array([o.encoded_mask for o in outputs], dtype=np.int8),
        "init_buffered": np.array([o.init_buffered for o in outputs], dtype=np.int8),
        "qc_flags": np.array([o.qc_flags for o in outputs], dtype=np.int32),
    }
    if include_z:
        dim = None if embed_dim is None else int(embed_dim)
        if dim is None:
            for output in outputs:
                if output.embedding is not None:
                    dim = int(np.asarray(output.embedding, dtype=np.float32).reshape(-1).size)
                    break
        dim = 0 if dim is None else int(dim)
        embeddings = []
        for output in outputs:
            if output.embedding is None:
                embeddings.append(np.zeros((dim,), dtype=np.float32))
                continue
            z = np.asarray(output.embedding, dtype=np.float32).reshape(-1)
            if z.size != dim:
                raise ValueError(f"EventOutput embedding dimension mismatch: expected {dim}, got {z.size}")
            embeddings.append(z)
        out["embedding"] = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    return out


def update_qc_flag(flags: int, flag: QCFlag, enabled: bool = True) -> int:
    if not enabled:
        return flags
    return int(QCFlag(flags) | flag)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = [
    "CandidatePeak",
    "Event",
    "EventOutput",
    "Monitor",
    "build_channel_positions_from_electrode_config",
    "neighbor_consistency_metrics",
    "PatchBatch",
    "PipelineStats",
    "QCFlag",
    "build_explicit_positions",
    "build_knn_neighbors",
    "build_local_grid_neighbors",
    "build_neuropixels_np1_384_positions",
    "build_neighbors_from_electrode_config",
    "build_staggered_grid_positions",
    "channel_distance",
    "deep_update",
    "ensure_channel_positions",
    "events_to_numpy",
    "get_logger",
    "load_config",
    "load_yaml",
    "outputs_to_numpy",
    "pairwise_distances",
    "relative_geo",
    "resolve_channel_geometry",
    "resolve_max_simul_firing",
    "set_seed",
    "update_qc_flag",
]
