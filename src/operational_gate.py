from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import eval as eval_mod
from src.benchmark_io import load_gt_events
from src.dataset import (
    MEArecH5Reader,
    _alignment_window_bounds,
    _normalize_gt_time_reference,
    _pick_peak_channel_and_time,
)
from src.nn.losses import gate_detection_loss
from src.nn.model import SpikeEmbeddingModel
from src.model_io import build_model_config
from src.online.buffer import RingBuffer
from src.stream import (
    EventDetector,
    PatchBuilder,
    PatchConfig,
    RobustNoiseEstimator,
    StreamBandpass,
    StreamCAR,
    WindowScheduler,
    build_detector_config,
)
from src.utils import build_knn_neighbors, load_yaml, pairwise_distances, resolve_channel_geometry, set_seed


LABELING_VERSION = "neighbor_aware_peak_aligned_unit_snr_drift_h5_drift_metadata_anchor_peak_union_missed_gt_v12"
DRIFT_VECTOR_DATASET_CANDIDATES = (
    "drift_vector_idxs",
    "drift_list/0/drift_vector_idxs",
    "gt/drift_vector_idxs",
)
DRIFT_FS_DATASET_CANDIDATES = (
    "drift_fs",
    "drift_list/0/drift_fs",
    "info/recordings/drift_fs",
)


@dataclass(frozen=True)
class GateJob:
    name: str
    input_h5: Path
    split: str
    enabled: bool = True


def resolve_path(path_value: str | Path, root: Path = ROOT) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else root / p


def nested_get(mapping: Dict[str, Any], dotted: str) -> Any:
    cur: Any = mapping
    for part in str(dotted).split("."):
        if not isinstance(cur, dict):
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def build_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gate_operational_alpha4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def resolve_device(requested: str) -> torch.device:
    req = str(requested).lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if req not in ("cuda", "cpu"):
        raise ValueError(f"unsupported device={requested!r}")
    return torch.device(req)


def resolve_car_mode(n_channels: int, raw_mode: str) -> str:
    mode = str(raw_mode or "auto").lower()
    if mode == "auto":
        return "none" if int(n_channels) == 1 else "global"
    if mode == "global" and int(n_channels) == 1:
        return "none"
    if mode not in ("none", "global"):
        raise ValueError(f"unsupported car_mode={raw_mode!r}; expected auto, none, or global")
    return mode


def short_job_name(path: Path) -> str:
    return path.stem.replace("recording_", "")


def build_jobs(cfg: Dict[str, Any]) -> List[GateJob]:
    data_cfg = dict(cfg.get("data", {}) or {})
    source_cfg_path = resolve_path(data_cfg["source_config"])
    source_cfg = load_yaml(source_cfg_path)
    source_key = str(data_cfg.get("source_config_key", "data.mearec_h5_paths"))
    source_paths = [resolve_path(p) for p in nested_get(source_cfg, source_key)]

    val_raw = [dict(x) for x in data_cfg.get("validation", [])]
    test_raw = [dict(x) for x in data_cfg.get("test", [])]
    special_paths = {
        str(resolve_path(item["input_h5"]))
        for item in val_raw + test_raw
        if bool(item.get("enabled", True))
    }

    jobs: List[GateJob] = []
    for path in source_paths:
        if str(path) in special_paths:
            continue
        jobs.append(GateJob(name=short_job_name(path), input_h5=path, split="train"))
    for item in val_raw:
        if bool(item.get("enabled", True)):
            jobs.append(GateJob(name=str(item["name"]), input_h5=resolve_path(item["input_h5"]), split="val"))
    for item in test_raw:
        if bool(item.get("enabled", True)):
            jobs.append(GateJob(name=str(item["name"]), input_h5=resolve_path(item["input_h5"]), split="test"))
    seen: set[str] = set()
    out: List[GateJob] = []
    for job in jobs:
        key = f"{job.split}:{job.input_h5}"
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out


def validate_eval_jobs(jobs: Sequence[GateJob]) -> None:
    blocked: List[str] = []
    for job in jobs:
        if job.split not in ("val", "test"):
            continue
        if "noise20" not in job.input_h5.name.lower():
            continue
        reader = MEArecH5Reader(job.input_h5)
        _, n_channels = reader.get_recordings_shape()
        if int(n_channels) == 32:
            blocked.append(f"{job.split}:{job.name}:{job.input_h5}")
    if blocked:
        raise ValueError(
            "validation/test must not use 32-channel noise20 recordings: "
            + "; ".join(blocked)
        )


def resolve_geometry(
    *,
    n_channels: int,
    h5_positions: np.ndarray | None,
    electrode_config: str | None,
    neighbor_source: str,
    fallback_k: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    n_ch = int(n_channels)
    source_req = str(neighbor_source or "auto").lower()
    electrode_cfg = None
    cfg_path = None
    if electrode_config:
        cfg_path = resolve_path(electrode_config)
        if not cfg_path.exists() and n_ch > 1:
            raise FileNotFoundError(f"electrode_config not found: {cfg_path}")
        if cfg_path.exists():
            try:
                electrode_cfg = load_yaml(cfg_path)
            except Exception as exc:
                if n_ch > 1:
                    raise ValueError(f"failed to load electrode_config {cfg_path}: {exc}") from exc
                logger.warning("failed to load electrode config %s: %s", cfg_path, exc)
    elif n_ch > 1:
        raise ValueError(
            "multi-channel pretrain candidate generation requires offline.electrode_config; "
            "peak_only patches still need a dense electrode neighbor graph"
        )
    positions, neighbors, geometry_source, metrics = resolve_channel_geometry(
        n_channels=n_ch,
        fallback_k=max(1, int(fallback_k)),
        electrode_cfg=electrode_cfg,
        h5_positions=h5_positions,
    )
    if n_ch > 1 and geometry_source != "config":
        raise ValueError(
            f"multi-channel pretrain candidate generation requires electrode_config matching n_channels={n_ch}; "
            f"resolved neighbor_graph source={geometry_source}"
        )
    if metrics is not None:
        logger.info(
            "neighbor_graph_compare source=%s resolved=%s nn1=%.4f nnk=%.4f",
            source_req,
            geometry_source,
            float(metrics.get("nn1_match_ratio", float("nan"))),
            float(metrics.get("nnk_match_ratio", float("nan"))),
        )
    if positions is None:
        positions = np.stack([np.arange(int(n_channels), dtype=np.float32), np.zeros((int(n_channels),), dtype=np.float32)], axis=1)
    if neighbors is None:
        neighbors = np.arange(int(n_channels), dtype=np.int32)[:, None]
    return np.asarray(positions, dtype=np.float32), np.asarray(neighbors, dtype=np.int32)


def compute_gt_overlap_labels(
    gt: eval_mod.EventSet,
    *,
    fs: float,
    n_channels: int,
    strict_overlap_ms: float,
    strict_overlap_um: float,
) -> np.ndarray:
    n = int(gt.times_samples.size)
    out = np.zeros((n,), dtype=np.float32)
    if n <= 1:
        return out
    order = np.argsort(gt.times_samples, kind="mergesort")
    times = gt.times_samples[order]
    labels = gt.labels[order]
    dt_max = max(0, int(round(float(strict_overlap_ms) * float(fs) / 1000.0)))
    positions = gt.unit_positions or {}
    channels = gt.channels

    def spatial_ok(i_sorted: int, j_sorted: int) -> bool:
        if int(labels[i_sorted]) == int(labels[j_sorted]):
            return False
        if int(n_channels) == 1:
            return True
        pi = positions.get(int(labels[i_sorted]))
        pj = positions.get(int(labels[j_sorted]))
        if pi is not None and pj is not None:
            return float(np.linalg.norm(np.asarray(pi, dtype=np.float32) - np.asarray(pj, dtype=np.float32))) <= float(strict_overlap_um)
        if channels is not None:
            ci = int(channels[order[i_sorted]])
            cj = int(channels[order[j_sorted]])
            return ci >= 0 and cj >= 0 and ci == cj
        return True

    for i in range(n):
        j = i + 1
        while j < n and int(times[j] - times[i]) <= dt_max:
            if spatial_ok(i, j):
                out[order[i]] = 1.0
                out[order[j]] = 1.0
            j += 1
    return out


def nearest_gt_diagnostics(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    *,
    tol_samples: int,
    matched_gt_indices: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_times = np.asarray(pred_times, dtype=np.int64)
    if gt_times.size == 0:
        fill = np.full(pred_times.shape, -1, dtype=np.int64)
        return fill.astype(np.int32), fill, np.zeros(pred_times.shape, dtype=np.bool_)
    order = np.argsort(gt_times, kind="mergesort")
    sorted_times = gt_times[order]
    nearest_dt = np.full(pred_times.shape, np.iinfo(np.int32).max, dtype=np.int32)
    nearest_gt = np.full(pred_times.shape, -1, dtype=np.int64)
    pos = np.searchsorted(sorted_times, pred_times, side="left")
    for offset in (0, -1):
        idx = pos + offset
        valid = (idx >= 0) & (idx < sorted_times.size)
        if not np.any(valid):
            continue
        cand_dt = np.abs(sorted_times[idx[valid]] - pred_times[valid]).astype(np.int64)
        replace = cand_dt < nearest_dt[valid].astype(np.int64)
        valid_idx = np.flatnonzero(valid)
        take = valid_idx[replace]
        nearest_dt[take] = cand_dt[replace].astype(np.int32)
        nearest_gt[take] = order[idx[valid][replace]].astype(np.int64)
    near = nearest_dt <= int(tol_samples)
    matched_gt = np.asarray([int(g) in matched_gt_indices for g in nearest_gt.tolist()], dtype=np.bool_)
    duplicate_negative = near & matched_gt
    return nearest_dt, nearest_gt, duplicate_negative


def _read_h5_dataset(h5: h5py.File, candidates: Sequence[str]) -> np.ndarray | None:
    for key_raw in candidates:
        key = str(key_raw).strip("/")
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            return np.asarray(h5[key][:])
    return None


def _read_h5_scalar(h5: h5py.File, candidates: Sequence[str]) -> Any | None:
    for key_raw in candidates:
        key = str(key_raw).strip("/")
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            return h5[key][()]
    return None


def _read_h5_spiketrain_unit_ids(h5: h5py.File) -> List[int]:
    if "spiketrains" not in h5 or not isinstance(h5["spiketrains"], h5py.Group):
        return []
    out: List[int] = []
    for key in h5["spiketrains"].keys():
        try:
            out.append(int(key))
        except ValueError:
            try:
                out.append(int(str(key).split("_")[-1]))
            except Exception:
                continue
    return sorted(out)


def _unit_row_lookup(h5: h5py.File, n_units: int) -> Dict[int, int]:
    if "template_ids" in h5 and isinstance(h5["template_ids"], h5py.Dataset):
        template_ids = np.asarray(h5["template_ids"][:]).reshape(-1)
    else:
        template_ids = np.arange(int(n_units), dtype=np.int64)
    if template_ids.size != int(n_units):
        template_ids = np.arange(int(n_units), dtype=np.int64)
    out: Dict[int, int] = {}
    for row, unit_id in enumerate(template_ids.tolist()):
        out[int(unit_id)] = int(row)
        out.setdefault(int(row), int(row))
    spiketrain_ids = _read_h5_spiketrain_unit_ids(h5)
    if len(spiketrain_ids) == int(n_units):
        for row, unit_id in enumerate(spiketrain_ids):
            out[int(unit_id)] = int(row)
    return out


def _map_gt_unit_rows(labels: np.ndarray, row_lookup: Dict[int, int]) -> np.ndarray:
    return np.asarray([int(row_lookup.get(int(label), -1)) for label in labels.tolist()], dtype=np.int64)


def _channels_from_unit_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    row_lookup: Dict[int, int],
) -> np.ndarray:
    scores = np.asarray(scores)
    rows = _map_gt_unit_rows(labels, row_lookup)
    out = np.full(labels.shape, -1, dtype=np.int32)
    valid = (rows >= 0) & (rows < int(scores.shape[0]))
    if np.any(valid):
        peak_by_unit = np.argmax(scores, axis=1).astype(np.int32, copy=False)
        out[valid] = peak_by_unit[rows[valid]]
    return out


def _fallback_gt_channels(gt: eval_mod.EventSet) -> tuple[np.ndarray, str]:
    fallback = np.full(gt.times_samples.shape, -1, dtype=np.int32)
    if gt.channels is not None:
        ch = np.asarray(gt.channels, dtype=np.int32).reshape(-1)
        if ch.shape == fallback.shape:
            fallback = ch.astype(np.int32, copy=True)
    source = "static_gt_channels" if np.any(fallback >= 0) else "missing"
    return fallback, source


def infer_static_gt_peak_channels(
    input_h5: Path,
    gt: eval_mod.EventSet,
    *,
    fs: float,
    n_channels: int,
) -> tuple[np.ndarray, str]:
    del fs
    fallback, source = _fallback_gt_channels(gt)
    if int(n_channels) == 1 and not np.any(fallback >= 0):
        return np.zeros(gt.times_samples.shape, dtype=np.int32), "single_channel_default"

    try:
        with h5py.File(input_h5, "r") as h5:
            templates_ds = h5["templates"] if "templates" in h5 and isinstance(h5["templates"], h5py.Dataset) else None
            voltage_peaks = _read_h5_dataset(h5, ["voltage_peaks", "gt/voltage_peaks"])
            if voltage_peaks is not None:
                peaks = np.asarray(voltage_peaks, dtype=np.float32)
                if peaks.ndim == 2 and peaks.shape[1] == int(n_channels):
                    return _channels_from_unit_scores(gt.labels, np.abs(peaks), _unit_row_lookup(h5, int(peaks.shape[0]))), "static_voltage_peaks"

            if templates_ds is not None:
                templates = np.asarray(templates_ds[:], dtype=np.float32)
                if templates.ndim == 4:
                    scores = np.max(np.abs(templates), axis=(1, 3))
                    return _channels_from_unit_scores(gt.labels, scores, _unit_row_lookup(h5, int(templates.shape[0]))), "static_templates"
                if templates.ndim == 5:
                    scores = np.max(np.abs(templates), axis=(1, 2, 4))
                    return _channels_from_unit_scores(gt.labels, scores, _unit_row_lookup(h5, int(templates.shape[0]))), "static_templates"
    except Exception:
        return fallback.astype(np.int32, copy=False), source
    return fallback.astype(np.int32, copy=False), source


def infer_drifted_gt_peak_channels(
    input_h5: Path,
    gt: eval_mod.EventSet,
    *,
    fs: float,
    n_channels: int,
) -> tuple[np.ndarray, str]:
    out = np.full(gt.times_samples.shape, -1, dtype=np.int32)
    if int(n_channels) <= 1 or int(gt.times_samples.size) == 0:
        return out, "missing"
    try:
        with h5py.File(input_h5, "r") as h5:
            templates_ds = h5["templates"] if "templates" in h5 and isinstance(h5["templates"], h5py.Dataset) else None
            if templates_ds is None or int(templates_ds.ndim) != 5:
                return out, "missing"
            drift_vector = _read_h5_dataset(
                h5,
                DRIFT_VECTOR_DATASET_CANDIDATES,
            )
            drift_fs_raw = _read_h5_scalar(
                h5,
                DRIFT_FS_DATASET_CANDIDATES,
            )
            drift_steps_raw = _read_h5_scalar(
                h5,
                [
                    "drift_steps",
                    "drift_list/0/drift_steps",
                    "info/recordings/drift_steps",
                ],
            )
            drift_factors_raw = _read_h5_dataset(
                h5,
                [
                    "drift_factors",
                    "drift_list/0/drift_factors",
                    "gt/drift_factors",
                ],
            )
            if drift_vector is None or drift_fs_raw is None:
                return out, "missing"

            templates = np.asarray(templates_ds[:], dtype=np.float32)
            if templates.ndim != 5 or templates.shape[1] <= 0 or templates.shape[3] <= 0:
                return out, "missing"
            amp = np.max(np.abs(templates), axis=(2, 4))
            peak_ch_by_step = np.argmax(amp, axis=2).astype(np.int32, copy=False)
            row_lookup = _unit_row_lookup(h5, int(templates.shape[0]))
            rows = _map_gt_unit_rows(gt.labels, row_lookup)
            drift_vector_arr = np.asarray(drift_vector, dtype=np.int64).reshape(-1)
            drift_fs = float(np.asarray(drift_fs_raw).reshape(()))
            if drift_vector_arr.size <= 0 or drift_fs <= 0.0 or not math.isfinite(drift_fs):
                return out, "missing"
            n_steps = int(peak_ch_by_step.shape[1])
            try:
                drift_steps = int(np.asarray(drift_steps_raw).reshape(())) if drift_steps_raw is not None else n_steps
            except Exception:
                drift_steps = n_steps
            if drift_steps <= 0:
                drift_steps = n_steps
            mid_point_idx = int(drift_steps // 2)

            ti = np.clip(
                (gt.times_samples.astype(np.float64, copy=False) / float(fs) * drift_fs).astype(np.int64),
                0,
                int(drift_vector_arr.size) - 1,
            )
            valid = (rows >= 0) & (rows < int(peak_ch_by_step.shape[0]))
            if not np.any(valid):
                return out, "missing"
            drift_factors = np.ones(rows.shape, dtype=np.float32)
            if drift_factors_raw is not None:
                factor_arr = np.asarray(drift_factors_raw, dtype=np.float32).reshape(-1)
                factor_valid = valid & (rows < int(factor_arr.size))
                if np.any(factor_valid):
                    drift_factors[factor_valid] = factor_arr[rows[factor_valid]]
            drift_displacements = (drift_vector_arr[ti].astype(np.float32, copy=False) * drift_factors).astype(np.int16).astype(np.int64)
            drift_idx = np.clip(drift_displacements + int(mid_point_idx), 0, int(n_steps) - 1).astype(np.int64, copy=False)
            out[valid] = peak_ch_by_step[rows[valid], drift_idx[valid]]
            valid_channels = (out[valid] >= 0) & (out[valid] < int(n_channels))
            if not np.all(valid_channels):
                out[valid] = np.where(valid_channels, out[valid], -1).astype(np.int32, copy=False)
            if not np.all(out >= 0):
                return out.astype(np.int32, copy=False), "missing"
            return out.astype(np.int32, copy=False), "drift_templates"
    except Exception:
        return out, "missing"


def drift_gt_channel_enabled(input_h5: Path, *, n_channels: int, neighbors_lookup: np.ndarray | None) -> bool:
    if int(n_channels) <= 1:
        return False
    if neighbors_lookup is None:
        return False
    neigh = np.asarray(neighbors_lookup)
    if neigh.ndim != 2 or int(neigh.shape[1]) <= 1:
        return False
    try:
        with h5py.File(input_h5, "r") as h5:
            templates_ds = h5["templates"] if "templates" in h5 and isinstance(h5["templates"], h5py.Dataset) else None
            if templates_ds is None or int(templates_ds.ndim) != 5:
                return False
            drift_vector = _read_h5_dataset(h5, DRIFT_VECTOR_DATASET_CANDIDATES)
            if drift_vector is None or np.asarray(drift_vector).reshape(-1).size <= 0:
                return False
            drift_fs_raw = _read_h5_scalar(h5, DRIFT_FS_DATASET_CANDIDATES)
            if drift_fs_raw is None:
                return False
            drift_fs = float(np.asarray(drift_fs_raw).reshape(()))
            return bool(drift_fs > 0.0 and math.isfinite(drift_fs))
    except Exception:
        return False


def infer_gt_peak_channels(
    input_h5: Path,
    gt: eval_mod.EventSet,
    *,
    fs: float,
    n_channels: int,
    use_drift_templates: bool = False,
) -> tuple[np.ndarray, str]:
    template_channels, template_source = infer_static_gt_peak_channels(input_h5, gt, fs=float(fs), n_channels=int(n_channels))
    if bool(use_drift_templates):
        drift_channels, drift_source = infer_drifted_gt_peak_channels(input_h5, gt, fs=float(fs), n_channels=int(n_channels))
        if drift_source == "drift_templates":
            return drift_channels.astype(np.int32, copy=False), drift_source
    return template_channels.astype(np.int32, copy=False), template_source


def build_symmetric_hop_matrix(neighbors_lookup: np.ndarray, *, n_channels: int) -> np.ndarray:
    n = int(n_channels)
    neigh = np.asarray(neighbors_lookup, dtype=np.int32)
    if n <= 0:
        return np.zeros((0, 0), dtype=np.int16)
    adjacency: List[set[int]] = [set([i]) for i in range(n)]
    if neigh.ndim == 2:
        for src in range(min(n, int(neigh.shape[0]))):
            for dst_raw in neigh[src].tolist():
                dst = int(dst_raw)
                if 0 <= dst < n:
                    adjacency[src].add(dst)
                    adjacency[dst].add(src)
    out = np.full((n, n), 32767, dtype=np.int16)
    for src in range(n):
        out[src, src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            next_dist = int(out[src, cur]) + 1
            for dst in adjacency[cur]:
                if out[src, dst] <= next_dist:
                    continue
                out[src, dst] = np.int16(next_dist)
                queue.append(int(dst))
    return out


def gt_neighbor_union_indices(
    cand_idx: np.ndarray,
    *,
    event_channels: np.ndarray,
    hop_matrix: np.ndarray,
    n_channels: int,
    anchor_channel: int,
    peak_aligned_channel: int,
    hops: int,
) -> np.ndarray:
    if cand_idx.size == 0:
        return cand_idx
    clipped_channels = np.clip(event_channels[cand_idx], 0, max(0, int(n_channels) - 1))
    in_union = np.zeros(cand_idx.shape, dtype=np.bool_)
    for raw_ch in (int(anchor_channel), int(peak_aligned_channel)):
        if 0 <= raw_ch < int(n_channels):
            in_union |= hop_matrix[raw_ch, clipped_channels] <= int(hops)
    return cand_idx[in_union]


def _cfg_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _load_unit_snr_map(input_h5: Path) -> Dict[int, float]:
    try:
        raw = eval_mod._unit_snr_from_h5(input_h5)
    except Exception:
        return {}
    out: Dict[int, float] = {}
    for key, value in dict(raw or {}).items():
        try:
            unit = int(key)
            snr = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(snr):
            out[unit] = snr
    return out


def _filter_recording_for_gt_alignment(
    input_h5: Path,
    *,
    fs: float,
    n_channels: int,
    preprocess_cfg: Dict[str, Any],
    car_mode: str,
    chunk_samples: int,
) -> np.ndarray:
    with h5py.File(input_h5, "r") as h5:
        if "recordings" not in h5 or not isinstance(h5["recordings"], h5py.Dataset):
            raise KeyError(f"recordings dataset not found in {input_h5}")
        recordings = h5["recordings"]
        n_samples = int(recordings.shape[0])
        filtered = np.zeros((n_samples, int(n_channels)), dtype=np.float32)
        bandpass = StreamBandpass(
            fs=float(fs),
            low_hz=float(preprocess_cfg.get("bandpass_low", 300.0)),
            high_hz=float(preprocess_cfg.get("bandpass_high", 6000.0)),
            order=3,
            n_channels=int(n_channels),
        )
        car = StreamCAR(mode=car_mode)
        chunk = max(1, int(chunk_samples))
        for t0 in range(0, n_samples, chunk):
            t1 = min(t0 + chunk, n_samples)
            x = np.asarray(recordings[t0:t1], dtype=np.float32)
            filtered[t0:t1] = car.process(bandpass.process(x))
    return filtered


def align_gt_to_filtered_peaks(
    gt: eval_mod.EventSet,
    *,
    input_h5: Path,
    fs: float,
    n_channels: int,
    gt_channels: np.ndarray,
    labeling_cfg: Dict[str, Any],
    preprocess_cfg: Dict[str, Any],
    car_mode: str,
    neighbors_lookup: np.ndarray | None,
) -> tuple[eval_mod.EventSet, Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "enabled": False,
        "total": int(gt.times_samples.size),
        "aligned": 0,
        "channel_changed": 0,
        "boundary_hits": 0,
        "max_abs_shift_samples": 0,
        "median_shift_samples": 0.0,
        "p95_abs_shift_samples": 0.0,
    }
    if not _cfg_bool(labeling_cfg.get("gt_align_to_peak", False), default=False) or int(gt.times_samples.size) == 0:
        return gt, stats

    n_ch = int(n_channels)
    if n_ch <= 0:
        return gt, stats

    radius_ms = float(labeling_cfg.get("gt_align_radius_ms", 0.5))
    time_reference = str(labeling_cfg.get("gt_time_reference", "near_peak"))
    if n_ch == 1:
        radius_ms = float(labeling_cfg.get("single_channel_gt_align_radius_ms", radius_ms))
        time_reference = str(labeling_cfg.get("single_channel_gt_time_reference", time_reference))
    align_r = max(1, int(round(radius_ms * float(fs) / 1000.0)))
    time_ref = _normalize_gt_time_reference(time_reference)
    polarity = str(labeling_cfg.get("gt_align_polarity", "negative"))
    channel_hops = int(labeling_cfg.get("gt_align_channel_hops", labeling_cfg.get("neighbor_hops", 1)))

    chunk_samples = max(1, int(round(float(labeling_cfg.get("gt_align_chunk_sec", 1.0)) * float(fs))))
    filtered = _filter_recording_for_gt_alignment(
        input_h5,
        fs=float(fs),
        n_channels=n_ch,
        preprocess_cfg=preprocess_cfg,
        car_mode=car_mode,
        chunk_samples=chunk_samples,
    )
    if filtered.shape[0] == 0:
        return gt, stats

    subsample_step = max(1, int(labeling_cfg.get("gt_align_stats_subsample_step", 8)))
    sample = filtered[::subsample_step] if filtered.shape[0] > subsample_step else filtered
    ch_med = np.median(sample, axis=0).astype(np.float32, copy=False)

    hop_matrix = None
    if neighbors_lookup is not None and np.asarray(neighbors_lookup).ndim == 2:
        hop_matrix = build_symmetric_hop_matrix(np.asarray(neighbors_lookup, dtype=np.int32), n_channels=n_ch)

    raw_gt_channels = np.asarray(gt_channels, dtype=np.int32).reshape(-1)
    if raw_gt_channels.shape != gt.times_samples.shape:
        raw_gt_channels = np.full(gt.times_samples.shape, -1, dtype=np.int32)

    aligned_times = gt.times_samples.astype(np.int64, copy=True)
    aligned_channels = raw_gt_channels.astype(np.int32, copy=True)
    shifts: List[int] = []
    n_samples_total = int(filtered.shape[0])
    for idx, t_anchor_raw in enumerate(gt.times_samples.astype(np.int64, copy=False).tolist()):
        a_lo, a_hi = _alignment_window_bounds(
            t_anchor=int(t_anchor_raw),
            align_radius_samples=align_r,
            n_samples_total=n_samples_total,
            gt_time_reference=time_ref,
        )
        if a_hi <= a_lo:
            continue
        gt_ch = int(raw_gt_channels[idx]) if idx < int(raw_gt_channels.shape[0]) else -1
        if n_ch == 1:
            channel_idx = np.asarray([0], dtype=np.int32)
        elif hop_matrix is not None and 0 <= gt_ch < n_ch and channel_hops >= 0:
            channel_idx = np.flatnonzero(hop_matrix[gt_ch] <= int(channel_hops)).astype(np.int32, copy=False)
            if channel_idx.size == 0:
                channel_idx = np.asarray([gt_ch], dtype=np.int32)
        elif 0 <= gt_ch < n_ch:
            channel_idx = np.asarray([gt_ch], dtype=np.int32)
        else:
            channel_idx = np.arange(n_ch, dtype=np.int32)

        seg_align = filtered[a_lo:a_hi][:, channel_idx].astype(np.float32, copy=False)
        if seg_align.size == 0:
            continue
        try:
            t_rel, c_rel, _, _ = _pick_peak_channel_and_time(
                seg_align,
                ch_med=ch_med[channel_idx],
                polarity=polarity,
            )
        except (ValueError, IndexError):
            continue
        t_peak = int(a_lo + t_rel)
        c_peak = int(channel_idx[int(c_rel)])
        if t_rel == 0 or t_rel == int(seg_align.shape[0]) - 1:
            stats["boundary_hits"] = int(stats["boundary_hits"]) + 1
        aligned_times[idx] = int(t_peak)
        aligned_channels[idx] = int(c_peak)
        if c_peak != gt_ch:
            stats["channel_changed"] = int(stats["channel_changed"]) + 1
        dt = int(t_peak) - int(t_anchor_raw)
        shifts.append(dt)
        stats["aligned"] = int(stats["aligned"]) + 1

    if shifts:
        shift_arr = np.asarray(shifts, dtype=np.int32)
        stats["max_abs_shift_samples"] = int(np.max(np.abs(shift_arr)))
        stats["median_shift_samples"] = float(np.median(shift_arr))
        stats["p95_abs_shift_samples"] = float(np.percentile(np.abs(shift_arr), 95.0))
    stats["enabled"] = True
    stats["radius_ms"] = float(radius_ms)
    stats["time_reference"] = str(time_ref)
    stats["polarity"] = str(polarity)
    stats["channel_hops"] = int(channel_hops)

    metadata = dict(gt.metadata or {})
    metadata.update(
        {
            "gt_peak_aligned": True,
            "gt_align_radius_ms": float(radius_ms),
            "gt_align_time_reference": str(time_ref),
        }
    )
    return (
        eval_mod.EventSet(
            times_samples=aligned_times.astype(np.int64, copy=False),
            labels=gt.labels,
            channels=aligned_channels.astype(np.int32, copy=False),
            confidences=gt.confidences,
            unit_positions=gt.unit_positions,
            metadata=metadata,
        ),
        stats,
    )


def _default_training_mask(arr: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key in arr:
        return np.asarray(arr[key], dtype=np.bool_).reshape(-1)
    n = int(arr["spike_label"].shape[0])
    return np.ones((n,), dtype=np.bool_)


def _mask_event_arrays(arr: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    n = int(mask.shape[0])
    out: Dict[str, np.ndarray] = {}
    for key, value in arr.items():
        value_arr = np.asarray(value)
        if value_arr.shape[:1] == (n,):
            out[key] = value_arr[mask]
        else:
            out[key] = value_arr
    return out


def build_gate_spatial_stats(arr: Dict[str, np.ndarray]) -> np.ndarray:
    patches = np.asarray(arr["patches"], dtype=np.float32)
    n = int(patches.shape[0])

    def _event_vector(key: str, default: float, dtype: np.dtype) -> np.ndarray:
        raw = arr.get(key)
        if raw is None:
            return np.full((n,), default, dtype=dtype)
        value = np.asarray(raw, dtype=dtype).reshape(-1)
        if value.shape[0] != n:
            return np.full((n,), default, dtype=dtype)
        return value

    n_channels = _event_vector("n_channels", float(max(1, patches.shape[2])), np.float32)
    n_channels = np.maximum(n_channels.astype(np.float32, copy=False), 1.0)
    amp_norm = _event_vector("amp_norm", 0.0, np.float32)
    spatial_score = _event_vector("spatial_score", 0.0, np.float32)
    support_count = _event_vector("support_count", 0.0, np.float32)
    simul_global = _event_vector("simul_global", 0.0, np.float32)
    simul_local = _event_vector("simul_local", 0.0, np.float32)
    return np.stack(
        [
            amp_norm / 30.0,
            spatial_score,
            support_count,
            simul_global / n_channels,
            simul_local,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _selector_len(selector: slice | np.ndarray, n: int) -> int:
    if isinstance(selector, slice):
        return len(range(*selector.indices(int(n))))
    return int(np.asarray(selector).shape[0])


def _select_event_array(
    arr: Dict[str, np.ndarray],
    key: str,
    selector: slice | np.ndarray,
    *,
    n: int,
    shape_tail: Sequence[int],
    default: float | int | bool,
    dtype: np.dtype,
) -> np.ndarray:
    raw = arr.get(key)
    out_shape = (_selector_len(selector, n), *[int(v) for v in shape_tail])
    if raw is None:
        return np.full(out_shape, default, dtype=dtype)
    value = np.asarray(raw, dtype=dtype)
    if value.shape[:1] != (int(n),):
        return np.full(out_shape, default, dtype=dtype)
    selected = value[selector]
    if selected.shape != out_shape:
        return np.full(out_shape, default, dtype=dtype)
    return selected.astype(dtype, copy=False)


def build_gate_meta(
    arr: Dict[str, np.ndarray],
    selector: slice | np.ndarray,
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    patches = np.asarray(arr["patches"], dtype=np.float32)
    n = int(patches.shape[0])
    k = int(patches.shape[2]) if patches.ndim == 3 else 1
    geo = _select_event_array(arr, "geo", selector, n=n, shape_tail=(k, 4), default=0.0, dtype=np.float32)
    channel_mask = _select_event_array(arr, "channel_mask", selector, n=n, shape_tail=(k,), default=True, dtype=np.bool_)
    spatial_stats = arr.get("spatial_stats")
    if spatial_stats is None or np.asarray(spatial_stats).shape[:1] != (n,):
        spatial_stats = build_gate_spatial_stats(arr)
    spatial = np.asarray(spatial_stats, dtype=np.float32)[selector]
    return {
        "geo": torch.from_numpy(geo).to(device),
        "spatial_stats": torch.from_numpy(spatial.astype(np.float32, copy=False)).to(device),
        "channel_mask": torch.from_numpy(channel_mask.astype(np.bool_, copy=False)).to(device),
    }


def label_candidates(
    arrays: Dict[str, np.ndarray],
    *,
    input_h5: Path,
    fs: float,
    n_channels: int,
    labeling_cfg: Dict[str, Any],
    preprocess_cfg: Dict[str, Any] | None = None,
    car_mode: str = "auto",
    neighbors_lookup: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    n = int(arrays["patches"].shape[0])
    gt = load_gt_events(input_h5, format="mearec")
    if neighbors_lookup is None:
        neighbors_lookup = np.arange(int(n_channels), dtype=np.int32)[:, None]
    else:
        neighbors_lookup = np.asarray(neighbors_lookup, dtype=np.int32)

    missed_gt_ignore_enabled = _cfg_bool(labeling_cfg.get("missed_gt_ignore_enabled", True), default=True)
    missed_gt_ignore_ms = float(labeling_cfg.get("missed_gt_ignore_near_gt_ms", 0.5) or 0.0)
    missed_gt_ignore_samples = max(0, int(round(missed_gt_ignore_ms * float(fs) / 1000.0)))
    missed_gt_ignore_hops = max(
        0,
        int(labeling_cfg.get("missed_gt_ignore_neighbor_hops", labeling_cfg.get("neighbor_hops", 1))),
    )
    require_drift_gt_channel_for_missed_ignore = _cfg_bool(
        labeling_cfg.get("require_drift_gt_channel_for_missed_ignore", True),
        default=True,
    )
    drift_enabled = drift_gt_channel_enabled(input_h5, n_channels=int(n_channels), neighbors_lookup=neighbors_lookup)
    gt_template_channels, gt_template_source = infer_static_gt_peak_channels(input_h5, gt, fs=float(fs), n_channels=int(n_channels))
    gt_drift_channels = np.full(gt.times_samples.shape, -1, dtype=np.int32)
    drift_valid = False
    gt_channels = gt_template_channels.astype(np.int32, copy=True)
    gt_channel_source = gt_template_source
    if drift_enabled:
        drift_candidate_channels, drift_source = infer_drifted_gt_peak_channels(input_h5, gt, fs=float(fs), n_channels=int(n_channels))
        gt_drift_channels = drift_candidate_channels.astype(np.int32, copy=True)
        if drift_source == "drift_templates":
            drift_valid = True
            gt_channels = gt_drift_channels.astype(np.int32, copy=True)
            gt_channel_source = drift_source
    if (
        missed_gt_ignore_enabled
        and require_drift_gt_channel_for_missed_ignore
        and drift_enabled
        and not drift_valid
    ):
        raise RuntimeError(
            "missed-GT ignore requires valid drift-aware GT channels for drift recordings; "
            f"could not infer drift channels for {input_h5}"
        )
    drift_gt_channel_changed_count = 0
    if drift_valid:
        both_valid = (
            (gt_template_channels >= 0)
            & (gt_template_channels < int(n_channels))
            & (gt_drift_channels >= 0)
            & (gt_drift_channels < int(n_channels))
        )
        drift_gt_channel_changed_count = int(np.sum(both_valid & (gt_template_channels != gt_drift_channels)))

    gt_anchor_channels = gt_channels.astype(np.int32, copy=True)
    gt_align_stats: Dict[str, Any] = {"enabled": False}
    if _cfg_bool(labeling_cfg.get("gt_align_to_peak", False), default=False):
        gt, gt_align_stats = align_gt_to_filtered_peaks(
            gt,
            input_h5=input_h5,
            fs=float(fs),
            n_channels=int(n_channels),
            gt_channels=gt_channels,
            labeling_cfg=labeling_cfg,
            preprocess_cfg=dict(preprocess_cfg or {}),
            car_mode=car_mode,
            neighbors_lookup=neighbors_lookup,
        )
        if gt.channels is not None and np.asarray(gt.channels).shape == gt.times_samples.shape:
            gt_channels = np.asarray(gt.channels, dtype=np.int32).reshape(-1)
            gt_channel_source = f"{gt_channel_source}+peak_aligned"
    gt_peak_aligned_channels = gt_channels.astype(np.int32, copy=True)
    gt_for_labels = eval_mod.EventSet(
        times_samples=gt.times_samples,
        labels=gt.labels,
        channels=gt_channels if gt_channels.shape == gt.times_samples.shape and np.any(gt_channels >= 0) else gt.channels,
        unit_positions=gt.unit_positions,
        metadata=gt.metadata,
    )
    pred = eval_mod.EventSet(
        times_samples=np.asarray(arrays["event_time_sample"], dtype=np.int64),
        labels=np.zeros((n,), dtype=np.int64),
        channels=np.asarray(arrays["center_channel"], dtype=np.int32),
        metadata={"fs": float(fs), "source": "operational_gate_candidates"},
    )
    event_times = pred.times_samples.astype(np.int64, copy=False)
    event_channels = pred.channels.astype(np.int32, copy=False) if pred.channels is not None else np.full((n,), -1, dtype=np.int32)
    amp_norm = np.asarray(arrays.get("amp_norm", np.ones((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    q_score = np.asarray(arrays.get("q", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if amp_norm.shape[0] != n:
        amp_norm = np.ones((n,), dtype=np.float32)
    if q_score.shape[0] != n:
        q_score = np.zeros((n,), dtype=np.float32)

    unit_snr_min_positive = max(0.0, float(labeling_cfg.get("unit_snr_min_positive", 0.0) or 0.0))
    unit_snr_requested = unit_snr_min_positive > 0.0
    unit_snr_by_unit = _load_unit_snr_map(input_h5) if unit_snr_requested else {}
    unit_snr_filter_enabled = bool(unit_snr_requested and unit_snr_by_unit)
    gt_unit_snr = np.full(gt.labels.shape, np.nan, dtype=np.float32)
    if unit_snr_requested and unit_snr_by_unit:
        for idx, label in enumerate(gt.labels.astype(np.int64, copy=False).tolist()):
            snr = unit_snr_by_unit.get(int(label))
            if snr is not None and math.isfinite(float(snr)):
                gt_unit_snr[int(idx)] = float(snr)
    gt_snr_known = np.isfinite(gt_unit_snr)
    low_snr_gt = unit_snr_filter_enabled & gt_snr_known & (gt_unit_snr < float(unit_snr_min_positive))
    gt_spike_allowed = np.ones(gt.labels.shape, dtype=np.bool_)
    if unit_snr_filter_enabled:
        gt_spike_allowed[low_snr_gt] = False

    gt_units = np.unique(gt.labels.astype(np.int64, copy=False))
    high_snr_units: set[int] = set()
    low_snr_units: set[int] = set()
    missing_snr_units: set[int] = set()
    if unit_snr_requested:
        for unit in gt_units.tolist():
            snr = unit_snr_by_unit.get(int(unit))
            if snr is None or not math.isfinite(float(snr)):
                missing_snr_units.add(int(unit))
            elif float(snr) < float(unit_snr_min_positive):
                low_snr_units.add(int(unit))
            else:
                high_snr_units.add(int(unit))
    high_snr_gt_events = int(np.sum(np.isin(gt.labels, np.asarray(sorted(high_snr_units), dtype=np.int64)))) if high_snr_units else 0
    low_snr_gt_events = int(np.sum(np.isin(gt.labels, np.asarray(sorted(low_snr_units), dtype=np.int64)))) if low_snr_units else 0
    missing_snr_gt_events = int(np.sum(np.isin(gt.labels, np.asarray(sorted(missing_snr_units), dtype=np.int64)))) if missing_snr_units else 0

    match_samples = max(0, int(round(float(labeling_cfg.get("match_ms", 0.2)) * float(fs) / 1000.0)))
    secondary_samples = max(0, int(round(float(labeling_cfg.get("secondary_match_ms", 0.1)) * float(fs) / 1000.0)))
    neighbor_hops = max(0, int(labeling_cfg.get("neighbor_hops", 1)))
    max_positive_per_gt = max(1, int(labeling_cfg.get("max_positive_per_gt", 2)))
    near_higher_peak_ratio = float(labeling_cfg.get("near_higher_peak_ratio", 0.85))
    primary_channel_mode = str(labeling_cfg.get("primary_channel_mode", "exact")).strip().lower()
    allow_neighbor_primary = primary_channel_mode in ("exact_then_neighbor", "neighbor_fallback", "neighbor", "zone")
    hop_matrix = build_symmetric_hop_matrix(neighbors_lookup, n_channels=int(n_channels))

    spike_label = np.zeros((n,), dtype=np.float32)
    matched_gt_index = np.full((n,), -1, dtype=np.int64)
    matched_gt_label = np.full((n,), -1, dtype=np.int64)
    matched_h5_unit_id = np.full((n,), -1, dtype=np.int64)
    matched_gt_channel = np.full((n,), -1, dtype=np.int32)
    label_role = np.zeros((n,), dtype=np.int8)
    dt_samples = np.full((n,), np.iinfo(np.int32).max, dtype=np.int32)
    ignored_neighbor = np.zeros((n,), dtype=np.bool_)
    neighbor_primary = np.zeros((n,), dtype=np.bool_)

    cand_order = np.argsort(event_times, kind="mergesort")
    sorted_event_times = event_times[cand_order]
    gt_order = np.argsort(gt.times_samples, kind="mergesort")
    for gi in gt_order.tolist():
        peak_gt_ch = int(gt_peak_aligned_channels[int(gi)]) if int(gi) < int(gt_peak_aligned_channels.shape[0]) else -1
        anchor_gt_ch = int(gt_anchor_channels[int(gi)]) if int(gi) < int(gt_anchor_channels.shape[0]) else -1
        gt_ch = peak_gt_ch if 0 <= peak_gt_ch < int(n_channels) else anchor_gt_ch
        if gt_ch < 0 or gt_ch >= int(n_channels):
            continue
        gt_time = int(gt.times_samples[int(gi)])
        lo = int(np.searchsorted(sorted_event_times, gt_time - match_samples, side="left"))
        hi = int(np.searchsorted(sorted_event_times, gt_time + match_samples, side="right"))
        if lo >= hi:
            continue
        cand_idx = cand_order[lo:hi]
        zone_idx = gt_neighbor_union_indices(
            cand_idx,
            event_channels=event_channels,
            hop_matrix=hop_matrix,
            n_channels=int(n_channels),
            anchor_channel=anchor_gt_ch,
            peak_aligned_channel=gt_ch,
            hops=int(neighbor_hops),
        )
        if zone_idx.size > 0:
            ignored_neighbor[zone_idx] = True
        if not bool(gt_spike_allowed[int(gi)]):
            continue

        available = spike_label[cand_idx] <= 0.5
        primary_idx = cand_idx[(event_channels[cand_idx] == gt_ch) & available]
        primary_is_neighbor = False
        if primary_idx.size == 0 and allow_neighbor_primary:
            primary_idx = zone_idx[spike_label[zone_idx] <= 0.5]
            primary_is_neighbor = primary_idx.size > 0
        if primary_idx.size == 0:
            continue
        primary_dt = np.abs(event_times[primary_idx] - gt_time)
        primary_order = np.lexsort((primary_idx, -q_score[primary_idx], -amp_norm[primary_idx], primary_dt))
        primary = int(primary_idx[int(primary_order[0])])
        h5_unit_id = int(gt.labels[int(gi)])
        spike_label[primary] = 1.0
        matched_gt_index[primary] = int(gi)
        matched_gt_label[primary] = h5_unit_id
        matched_h5_unit_id[primary] = h5_unit_id
        matched_gt_channel[primary] = int(gt_ch)
        dt_samples[primary] = int(event_times[primary] - gt_time)
        label_role[primary] = 1
        neighbor_primary[primary] = bool(primary_is_neighbor and int(event_channels[primary]) != gt_ch)

        if max_positive_per_gt < 2:
            continue
        secondary_idx = zone_idx[
            (zone_idx != primary)
            & (spike_label[zone_idx] <= 0.5)
            & (event_channels[zone_idx] != int(event_channels[primary]))
            & (amp_norm[zone_idx] >= near_higher_peak_ratio * float(amp_norm[primary]))
            & (np.abs(event_times[zone_idx] - int(event_times[primary])) <= int(secondary_samples))
        ]
        if secondary_idx.size == 0:
            continue
        secondary_dt_primary = np.abs(event_times[secondary_idx] - int(event_times[primary]))
        secondary_dt_gt = np.abs(event_times[secondary_idx] - gt_time)
        secondary_order = np.lexsort((secondary_idx, secondary_dt_gt, secondary_dt_primary, -q_score[secondary_idx], -amp_norm[secondary_idx]))
        secondary = int(secondary_idx[int(secondary_order[0])])
        h5_unit_id = int(gt.labels[int(gi)])
        spike_label[secondary] = 1.0
        matched_gt_index[secondary] = int(gi)
        matched_gt_label[secondary] = h5_unit_id
        matched_h5_unit_id[secondary] = h5_unit_id
        matched_gt_channel[secondary] = int(gt_ch)
        dt_samples[secondary] = int(event_times[secondary] - gt_time)
        label_role[secondary] = 2

    positive_gt_mask = np.zeros(gt.labels.shape, dtype=np.bool_)
    positive_gt_indices = matched_gt_index[matched_gt_index >= 0]
    if positive_gt_indices.size > 0:
        positive_gt_mask[np.unique(positive_gt_indices)] = True
    missed_gt_mask = gt_spike_allowed & (~positive_gt_mask)
    missed_gt_count = int(np.sum(missed_gt_mask))

    near_tol = int(round(float(labeling_cfg.get("near_gt_ms", 0.5)) * float(fs) / 1000.0))
    high_snr_ignore_ms = float(labeling_cfg.get("high_snr_ignore_near_gt_ms", 0.0) or 0.0)
    high_snr_ignore_samples = max(0, int(round(high_snr_ignore_ms * float(fs) / 1000.0)))
    high_snr_ignore_hops = max(0, int(labeling_cfg.get("high_snr_ignore_neighbor_hops", neighbor_hops)))
    high_snr_near_gt = np.zeros((n,), dtype=np.bool_)
    if high_snr_ignore_samples > 0:
        for gi in gt_order.tolist():
            if not bool(gt_spike_allowed[int(gi)]):
                continue
            peak_gt_ch = int(gt_peak_aligned_channels[int(gi)]) if int(gi) < int(gt_peak_aligned_channels.shape[0]) else -1
            anchor_gt_ch = int(gt_anchor_channels[int(gi)]) if int(gi) < int(gt_anchor_channels.shape[0]) else -1
            gt_ch = peak_gt_ch if 0 <= peak_gt_ch < int(n_channels) else anchor_gt_ch
            if gt_ch < 0 or gt_ch >= int(n_channels):
                continue
            gt_time = int(gt.times_samples[int(gi)])
            lo = int(np.searchsorted(sorted_event_times, gt_time - high_snr_ignore_samples, side="left"))
            hi = int(np.searchsorted(sorted_event_times, gt_time + high_snr_ignore_samples, side="right"))
            if lo >= hi:
                continue
            cand_idx = cand_order[lo:hi]
            zone_idx = gt_neighbor_union_indices(
                cand_idx,
                event_channels=event_channels,
                hop_matrix=hop_matrix,
                n_channels=int(n_channels),
                anchor_channel=anchor_gt_ch,
                peak_aligned_channel=gt_ch,
                hops=int(high_snr_ignore_hops),
            )
            high_snr_near_gt[zone_idx] = True

    low_snr_near_gt = np.zeros((n,), dtype=np.bool_)
    if unit_snr_filter_enabled and np.any(low_snr_gt):
        for gi in gt_order.tolist():
            if not bool(low_snr_gt[int(gi)]):
                continue
            peak_gt_ch = int(gt_peak_aligned_channels[int(gi)]) if int(gi) < int(gt_peak_aligned_channels.shape[0]) else -1
            anchor_gt_ch = int(gt_anchor_channels[int(gi)]) if int(gi) < int(gt_anchor_channels.shape[0]) else -1
            gt_ch = peak_gt_ch if 0 <= peak_gt_ch < int(n_channels) else anchor_gt_ch
            if gt_ch < 0 or gt_ch >= int(n_channels):
                continue
            gt_time = int(gt.times_samples[int(gi)])
            lo = int(np.searchsorted(sorted_event_times, gt_time - near_tol, side="left"))
            hi = int(np.searchsorted(sorted_event_times, gt_time + near_tol, side="right"))
            if lo >= hi:
                continue
            cand_idx = cand_order[lo:hi]
            zone_idx = gt_neighbor_union_indices(
                cand_idx,
                event_channels=event_channels,
                hop_matrix=hop_matrix,
                n_channels=int(n_channels),
                anchor_channel=anchor_gt_ch,
                peak_aligned_channel=gt_ch,
                hops=int(neighbor_hops),
            )
            low_snr_near_gt[zone_idx] = True

    missed_gt_near_gt = np.zeros((n,), dtype=np.bool_)
    if missed_gt_ignore_enabled and missed_gt_ignore_samples > 0 and missed_gt_count > 0:
        for gi in gt_order.tolist():
            if not bool(missed_gt_mask[int(gi)]):
                continue
            peak_gt_ch = int(gt_peak_aligned_channels[int(gi)]) if int(gi) < int(gt_peak_aligned_channels.shape[0]) else -1
            anchor_gt_ch = int(gt_anchor_channels[int(gi)]) if int(gi) < int(gt_anchor_channels.shape[0]) else -1
            gt_ch = peak_gt_ch if 0 <= peak_gt_ch < int(n_channels) else anchor_gt_ch
            if gt_ch < 0 or gt_ch >= int(n_channels):
                continue
            gt_time = int(gt.times_samples[int(gi)])
            lo = int(np.searchsorted(sorted_event_times, gt_time - missed_gt_ignore_samples, side="left"))
            hi = int(np.searchsorted(sorted_event_times, gt_time + missed_gt_ignore_samples, side="right"))
            if lo >= hi:
                continue
            cand_idx = cand_order[lo:hi]
            zone_idx = gt_neighbor_union_indices(
                cand_idx,
                event_channels=event_channels,
                hop_matrix=hop_matrix,
                n_channels=int(n_channels),
                anchor_channel=anchor_gt_ch,
                peak_aligned_channel=gt_ch,
                hops=int(missed_gt_ignore_hops),
            )
            missed_gt_near_gt[zone_idx] = True

    ignored_neighbor = ignored_neighbor & (spike_label <= 0.5)
    low_snr_near_gt = low_snr_near_gt & (spike_label <= 0.5)
    high_snr_near_gt = high_snr_near_gt & (spike_label <= 0.5) & (~ignored_neighbor) & (~low_snr_near_gt)
    missed_gt_near_gt = missed_gt_near_gt & (spike_label <= 0.5)
    training_mask = ~(ignored_neighbor | high_snr_near_gt | low_snr_near_gt | missed_gt_near_gt)
    eval_mask = ~(ignored_neighbor | high_snr_near_gt | low_snr_near_gt | missed_gt_near_gt)
    label_role[ignored_neighbor] = 3
    label_role[high_snr_near_gt] = 5
    label_role[low_snr_near_gt & (~ignored_neighbor)] = 4
    missed_gt_only = missed_gt_near_gt & (~ignored_neighbor) & (~high_snr_near_gt) & (~low_snr_near_gt)
    label_role[missed_gt_only] = 6
    missed_gt_ignored_candidates = int(np.sum(missed_gt_near_gt))
    missed_gt_drift_channel_valid = bool((not drift_enabled) or drift_valid)

    gt_overlap = compute_gt_overlap_labels(
        gt_for_labels,
        fs=float(fs),
        n_channels=int(n_channels),
        strict_overlap_ms=float(labeling_cfg.get("strict_overlap_ms", 0.5)),
        strict_overlap_um=float(labeling_cfg.get("strict_overlap_um", 60.0)),
    )
    overlap_label = np.zeros((n,), dtype=np.float32)
    overlap_supervision_mask = (spike_label > 0.5) & training_mask
    matched = matched_gt_index >= 0
    if np.any(matched):
        overlap_label[matched] = gt_overlap[matched_gt_index[matched]]

    nearest_dt, nearest_gt, _duplicate_negative = nearest_gt_diagnostics(
        event_times,
        gt_for_labels.times_samples.astype(np.int64, copy=False),
        tol_samples=near_tol,
        matched_gt_indices={int(v) for v in matched_gt_index[matched].tolist()},
    )
    matched_gt_snr = np.full((n,), np.nan, dtype=np.float32)
    valid_matched_snr = matched_gt_index >= 0
    if np.any(valid_matched_snr):
        matched_gt_snr[valid_matched_snr] = gt_unit_snr[matched_gt_index[valid_matched_snr]]
    nearest_gt_snr = np.full((n,), np.nan, dtype=np.float32)
    valid_nearest_snr = (nearest_gt >= 0) & (nearest_gt < int(gt_unit_snr.shape[0]))
    if np.any(valid_nearest_snr):
        nearest_gt_snr[valid_nearest_snr] = gt_unit_snr[nearest_gt[valid_nearest_snr]]
    diagnostic_gt_index = nearest_gt.astype(np.int64, copy=True)
    valid_matched_diagnostic = matched_gt_index >= 0
    if np.any(valid_matched_diagnostic):
        diagnostic_gt_index[valid_matched_diagnostic] = matched_gt_index[valid_matched_diagnostic]
    gt_template_channel = np.full((n,), -1, dtype=np.int32)
    gt_drift_channel = np.full((n,), -1, dtype=np.int32)
    gt_anchor_channel = np.full((n,), -1, dtype=np.int32)
    gt_peak_aligned_channel = np.full((n,), -1, dtype=np.int32)
    valid_template_channel = (diagnostic_gt_index >= 0) & (diagnostic_gt_index < int(gt_template_channels.shape[0]))
    if np.any(valid_template_channel):
        gt_template_channel[valid_template_channel] = gt_template_channels[diagnostic_gt_index[valid_template_channel]]
    valid_drift_channel = (diagnostic_gt_index >= 0) & (diagnostic_gt_index < int(gt_drift_channels.shape[0]))
    if np.any(valid_drift_channel):
        gt_drift_channel[valid_drift_channel] = gt_drift_channels[diagnostic_gt_index[valid_drift_channel]]
    valid_anchor_channel = (diagnostic_gt_index >= 0) & (diagnostic_gt_index < int(gt_anchor_channels.shape[0]))
    if np.any(valid_anchor_channel):
        gt_anchor_channel[valid_anchor_channel] = gt_anchor_channels[diagnostic_gt_index[valid_anchor_channel]]
    valid_peak_aligned_channel = (diagnostic_gt_index >= 0) & (diagnostic_gt_index < int(gt_peak_aligned_channels.shape[0]))
    if np.any(valid_peak_aligned_channel):
        gt_peak_aligned_channel[valid_peak_aligned_channel] = gt_peak_aligned_channels[diagnostic_gt_index[valid_peak_aligned_channel]]
    gt_anchor_peak_aligned_channel_changed = (
        (gt_anchor_channel >= 0)
        & (gt_anchor_channel < int(n_channels))
        & (gt_peak_aligned_channel >= 0)
        & (gt_peak_aligned_channel < int(n_channels))
        & (gt_anchor_channel != gt_peak_aligned_channel)
    )
    near_gt_negative = (spike_label <= 0.5) & training_mask & (nearest_dt <= near_tol)
    duplicate_negative = ignored_neighbor.astype(np.bool_, copy=False)
    arrays.update(
        {
            "spike_label": spike_label.astype(np.float32, copy=False),
            "overlap_label": overlap_label.astype(np.float32, copy=False),
            "overlap_supervision_mask": overlap_supervision_mask.astype(np.bool_, copy=False),
            "training_mask": training_mask.astype(np.bool_, copy=False),
            "eval_mask": eval_mask.astype(np.bool_, copy=False),
            "ignored_neighbor_event": ignored_neighbor.astype(np.bool_, copy=False),
            "neighbor_primary": neighbor_primary.astype(np.bool_, copy=False),
            "label_role": label_role.astype(np.int8, copy=False),
            "matched_gt_index": matched_gt_index.astype(np.int64, copy=False),
            "matched_gt_label": matched_gt_label.astype(np.int64, copy=False),
            "matched_h5_unit_id": matched_h5_unit_id.astype(np.int64, copy=False),
            "h5_unit_id": matched_h5_unit_id.astype(np.int64, copy=False),
            "local_unit_id": matched_h5_unit_id.astype(np.int64, copy=False),
            "matched_gt_channel": matched_gt_channel.astype(np.int32, copy=False),
            "matched_gt_snr": matched_gt_snr.astype(np.float32, copy=False),
            "dt_samples": dt_samples.astype(np.int32, copy=False),
            "nearest_gt_dt_samples": nearest_dt.astype(np.int32, copy=False),
            "nearest_gt_index": nearest_gt.astype(np.int64, copy=False),
            "nearest_gt_snr": nearest_gt_snr.astype(np.float32, copy=False),
            "near_gt_negative": near_gt_negative.astype(np.bool_, copy=False),
            "high_snr_near_gt": high_snr_near_gt.astype(np.bool_, copy=False),
            "low_snr_near_gt": low_snr_near_gt.astype(np.bool_, copy=False),
            "missed_gt_near_gt": missed_gt_near_gt.astype(np.bool_, copy=False),
            "duplicate_negative": duplicate_negative.astype(np.bool_, copy=False),
            "gt_channel_source": np.asarray([gt_channel_source] * n),
            "gt_template_channel": gt_template_channel.astype(np.int32, copy=False),
            "gt_drift_channel": gt_drift_channel.astype(np.int32, copy=False),
            "gt_anchor_channel": gt_anchor_channel.astype(np.int32, copy=False),
            "gt_peak_aligned_channel": gt_peak_aligned_channel.astype(np.int32, copy=False),
            "gt_anchor_peak_aligned_channel_changed": gt_anchor_peak_aligned_channel_changed.astype(np.bool_, copy=False),
            "drift_gt_channel_enabled": np.full((n,), bool(drift_enabled), dtype=np.bool_),
            "drift_gt_channel_valid": np.full((n,), bool(drift_valid), dtype=np.bool_),
            "drift_gt_channel_changed_count": np.full((n,), int(drift_gt_channel_changed_count), dtype=np.int32),
            "gt_align_enabled": np.full((n,), bool(gt_align_stats.get("enabled", False)), dtype=np.bool_),
            "gt_align_radius_ms": np.full((n,), float(gt_align_stats.get("radius_ms", 0.0)), dtype=np.float32),
            "gt_align_median_shift_samples": np.full((n,), float(gt_align_stats.get("median_shift_samples", 0.0)), dtype=np.float32),
            "gt_align_p95_abs_shift_samples": np.full((n,), float(gt_align_stats.get("p95_abs_shift_samples", 0.0)), dtype=np.float32),
            "gt_align_max_abs_shift_samples": np.full((n,), int(gt_align_stats.get("max_abs_shift_samples", 0)), dtype=np.int32),
            "gt_align_channel_changed": np.full((n,), int(gt_align_stats.get("channel_changed", 0)), dtype=np.int32),
            "unit_snr_min_positive": np.full((n,), float(unit_snr_min_positive), dtype=np.float32),
            "unit_snr_filter_enabled": np.full((n,), bool(unit_snr_filter_enabled), dtype=np.bool_),
            "high_snr_ignore_near_gt_ms": np.full((n,), float(high_snr_ignore_ms), dtype=np.float32),
            "high_snr_ignore_neighbor_hops": np.full((n,), int(high_snr_ignore_hops), dtype=np.int32),
            "missed_gt_ignore_enabled": np.full((n,), bool(missed_gt_ignore_enabled), dtype=np.bool_),
            "missed_gt_ignore_near_gt_ms": np.full((n,), float(missed_gt_ignore_ms), dtype=np.float32),
            "missed_gt_ignore_neighbor_hops": np.full((n,), int(missed_gt_ignore_hops), dtype=np.int32),
            "require_drift_gt_channel_for_missed_ignore": np.full(
                (n,),
                bool(require_drift_gt_channel_for_missed_ignore),
                dtype=np.bool_,
            ),
            "missed_gt_count": np.full((n,), int(missed_gt_count), dtype=np.int32),
            "missed_gt_ignored_candidates": np.full((n,), int(missed_gt_ignored_candidates), dtype=np.int32),
            "missed_gt_drift_channel_valid": np.full((n,), bool(missed_gt_drift_channel_valid), dtype=np.bool_),
            "gt_snr_high_unit_count": np.full((n,), int(len(high_snr_units)), dtype=np.int32),
            "gt_snr_low_unit_count": np.full((n,), int(len(low_snr_units)), dtype=np.int32),
            "gt_snr_missing_unit_count": np.full((n,), int(len(missing_snr_units)), dtype=np.int32),
            "gt_snr_high_event_count": np.full((n,), int(high_snr_gt_events), dtype=np.int32),
            "gt_snr_low_event_count": np.full((n,), int(low_snr_gt_events), dtype=np.int32),
            "gt_snr_missing_event_count": np.full((n,), int(missing_snr_gt_events), dtype=np.int32),
            "labeling_version": np.asarray([LABELING_VERSION] * n),
        }
    )
    return arrays


def generate_candidates_for_job(
    job: GateJob,
    *,
    cfg: Dict[str, Any],
    output_npz: Path,
    max_seconds: float,
    logger: logging.Logger,
) -> Dict[str, Any]:
    reader = MEArecH5Reader(job.input_h5)
    fs = float(reader.get_fs())
    n_samples, n_channels = reader.get_recordings_shape()
    if max_seconds > 0:
        n_samples = min(n_samples, int(round(max_seconds * fs)))
    try:
        h5_positions = reader.read_channel_positions()
    except Exception:
        h5_positions = None

    patch_cfg_raw = dict(cfg.get("patch", {}) or {})
    det_cfg_raw = dict(cfg.get("detection", {}) or {})
    stream_cfg = dict(cfg.get("stream", {}) or {})
    preprocess_cfg = dict(cfg.get("preprocess", {}) or {})
    labeling_cfg = dict(cfg.get("labeling", {}) or {})

    fallback_k = int(max(1, patch_cfg_raw.get("k", 1)))
    patch_peak_only = _cfg_bool(patch_cfg_raw.get("peak_only", False), default=False)
    positions, neighbors = resolve_geometry(
        n_channels=int(n_channels),
        h5_positions=h5_positions,
        electrode_config=cfg.get("offline", {}).get("electrode_config"),
        neighbor_source=str(det_cfg_raw.get("neighbor_source", "auto")),
        fallback_k=fallback_k,
        logger=logger,
    )
    label_neighbor_k = int(max(1, labeling_cfg.get("neighbor_k", 9)))
    label_neighbors = build_knn_neighbors(positions, k=label_neighbor_k)
    car_mode = resolve_car_mode(int(n_channels), str(preprocess_cfg.get("car_mode", "auto")))
    det_cfg = build_detector_config(det_cfg_raw, fs=fs, n_channels=int(n_channels))
    detector = EventDetector(det_cfg, neighbors_lookup=neighbors, channel_dist=pairwise_distances(positions))
    resolved_patch_k = 1 if patch_peak_only else int(neighbors.shape[1])
    patch_cfg = PatchConfig(
        left=int(patch_cfg_raw.get("left", 20)),
        right=int(patch_cfg_raw.get("right", 44)),
        k=resolved_patch_k,
        peak_only=patch_peak_only,
    )
    patch_builder = PatchBuilder(patch_cfg, neighbors_lookup=neighbors, channel_positions=positions)
    bandpass = StreamBandpass(
        fs=fs,
        low_hz=float(preprocess_cfg.get("bandpass_low", 300.0)),
        high_hz=float(preprocess_cfg.get("bandpass_high", 6000.0)),
        order=3,
        n_channels=int(n_channels),
    )
    car = StreamCAR(mode=car_mode)
    noise = RobustNoiseEstimator(
        int(n_channels),
        beta=float(preprocess_cfg.get("noise_beta", 0.05)),
        update_every_windows=int(det_cfg.stats_update_every_windows),
        subsample_step=int(det_cfg.stats_subsample_step),
    )
    scheduler = WindowScheduler(
        fs=fs,
        window_sec=float(stream_cfg.get("window_sec", 1.5)),
        stride_sec=float(stream_cfg.get("stride_sec", 0.5)),
        margin_sec=float(stream_cfg.get("margin_sec", 0.05)),
    )
    chunk_samples = max(1, int(round(float(stream_cfg.get("chunk_sec", 0.5)) * fs)))
    ring_capacity_sec = float(stream_cfg.get("ring_capacity_sec", 4.0))
    min_capacity = float(stream_cfg.get("window_sec", 1.5)) + 2.0 * float(stream_cfg.get("margin_sec", 0.05)) + float(stream_cfg.get("stride_sec", 0.5)) + 0.1
    ring_capacity_sec = max(ring_capacity_sec, min_capacity)
    ring = RingBuffer(int(round(ring_capacity_sec * fs)), int(n_channels))

    patches: List[np.ndarray] = []
    event_time: List[np.ndarray] = []
    center_ch: List[np.ndarray] = []
    amp_norm: List[np.ndarray] = []
    spatial_score: List[np.ndarray] = []
    support_count: List[np.ndarray] = []
    simul_global: List[np.ndarray] = []
    simul_local: List[np.ndarray] = []
    event_q: List[np.ndarray] = []
    qc_flags: List[np.ndarray] = []
    neighbor_chs: List[np.ndarray] = []
    geo_parts: List[np.ndarray] = []
    rel_pos: List[np.ndarray] = []
    channel_mask: List[np.ndarray] = []
    raw_candidates = merged_events = hard_drops = duplicate_drops = 0
    global_simul_drops = local_simul_drops = 0

    t0_read = 0
    for x in reader.iter_recordings(chunk_samples=chunk_samples, start=0, stop=n_samples):
        if x.size == 0:
            continue
        y = car.process(bandpass.process(x.astype(np.float32, copy=False)))
        ring.push(y)
        for seg in scheduler.iter_ready_segments(ring):
            sigma = noise.update(seg.data.x[scheduler.margin : scheduler.margin + scheduler.window])
            events = detector.detect(
                seg.data.x,
                seg.data.t0,
                sigma,
                active_start=int(scheduler.margin),
                active_stop=int(scheduler.margin + scheduler.stride),
            )
            raw_candidates += int(detector.last_stats.get("raw_candidates", 0))
            merged_events += int(detector.last_stats.get("merged_events", 0))
            hard_drops += int(detector.last_stats.get("hard_drops", 0))
            duplicate_drops += int(detector.last_stats.get("duplicate_drops", 0))
            global_simul_drops += int(detector.last_stats.get("global_simul_drops", 0))
            local_simul_drops += int(detector.last_stats.get("local_simul_drops", 0))
            events = scheduler.keep_center_window(events, seg.center_start, seg.center_stop)
            patch_batch, kept_events = patch_builder.build(seg.data.x, seg.data.t0, events)
            if patch_batch.x.shape[0] == 0:
                continue
            x_np = patch_batch.x.astype(np.float32, copy=False)
            if str(preprocess_cfg.get("patch_norm_mode", "peak_channel_sigma")).lower() == "peak_channel_sigma":
                neigh = patch_batch.meta["neighbor_chs"].astype(np.int64, copy=False)
                sigma_neigh = sigma[neigh].astype(np.float32, copy=False)
                offsets = patch_builder.time_offsets.astype(np.int32, copy=False)
                if offsets.shape[0] != x_np.shape[1]:
                    offsets = np.arange(-int(patch_builder.cfg.left), -int(patch_builder.cfg.left) + x_np.shape[1], dtype=np.int32)
                baseline_mask = (offsets >= -20) & (offsets <= -6)
                if not np.any(baseline_mask):
                    baseline_mask[: min(x_np.shape[1], max(1, int(patch_builder.cfg.left)))] = True
                baseline = np.median(x_np[:, baseline_mask, :], axis=1, keepdims=True)
                x_np = (x_np - baseline) / np.maximum(sigma_neigh[:, None, :], 1e-6)
            patches.append(x_np.astype(np.float32, copy=False))
            event_time.append(np.asarray([e.t0 for e in kept_events], dtype=np.int64))
            center_ch.append(np.asarray([e.c0 for e in kept_events], dtype=np.int32))
            amp_norm.append(np.asarray([e.amp_norm for e in kept_events], dtype=np.float32))
            spatial_score.append(np.asarray([e.spatial_score for e in kept_events], dtype=np.float32))
            support_count.append(np.asarray([e.support_count for e in kept_events], dtype=np.int32))
            simul_global.append(np.asarray([e.simul_global for e in kept_events], dtype=np.int32))
            simul_local.append(np.asarray([e.simul_local for e in kept_events], dtype=np.int32))
            event_q.append(np.asarray([e.q for e in kept_events], dtype=np.float32))
            qc_flags.append(np.asarray([e.qc_flags for e in kept_events], dtype=np.int32))
            neighbor_chs.append(patch_batch.meta["neighbor_chs"].astype(np.int32, copy=False))
            geo_parts.append(patch_batch.meta.get("geo", np.zeros((x_np.shape[0], x_np.shape[2], 4), dtype=np.float32)).astype(np.float32, copy=False))
            rel_pos.append(patch_batch.meta.get("rel_pos", np.zeros((x_np.shape[0], x_np.shape[2], 2), dtype=np.float32)).astype(np.float32, copy=False))
            channel_mask.append(patch_batch.meta.get("channel_mask", np.ones((x_np.shape[0], x_np.shape[2]), dtype=np.bool_)).astype(np.bool_, copy=False))
        t0_read += int(x.shape[0])

    arrays: Dict[str, np.ndarray]
    if patches:
        arrays = {
            "patches": np.concatenate(patches, axis=0).astype(np.float32, copy=False),
            "event_time_sample": np.concatenate(event_time, axis=0).astype(np.int64, copy=False),
            "center_channel": np.concatenate(center_ch, axis=0).astype(np.int32, copy=False),
            "amp_norm": np.concatenate(amp_norm, axis=0).astype(np.float32, copy=False),
            "spatial_score": np.concatenate(spatial_score, axis=0).astype(np.float32, copy=False),
            "support_count": np.concatenate(support_count, axis=0).astype(np.int32, copy=False),
            "simul_global": np.concatenate(simul_global, axis=0).astype(np.int32, copy=False),
            "simul_local": np.concatenate(simul_local, axis=0).astype(np.int32, copy=False),
            "q": np.concatenate(event_q, axis=0).astype(np.float32, copy=False),
            "qc_flags": np.concatenate(qc_flags, axis=0).astype(np.int32, copy=False),
            "neighbor_chs": np.concatenate(neighbor_chs, axis=0).astype(np.int32, copy=False),
            "geo": np.concatenate(geo_parts, axis=0).astype(np.float32, copy=False),
            "rel_pos": np.concatenate(rel_pos, axis=0).astype(np.float32, copy=False),
            "channel_mask": np.concatenate(channel_mask, axis=0).astype(np.bool_, copy=False),
        }
    else:
        t_len = int(patch_cfg.left + patch_cfg.right)
        k_len = int(1 if patch_cfg.peak_only else patch_cfg.k)
        arrays = {
            "patches": np.zeros((0, t_len, k_len), dtype=np.float32),
            "event_time_sample": np.zeros((0,), dtype=np.int64),
            "center_channel": np.zeros((0,), dtype=np.int32),
            "amp_norm": np.zeros((0,), dtype=np.float32),
            "spatial_score": np.zeros((0,), dtype=np.float32),
            "support_count": np.zeros((0,), dtype=np.int32),
            "simul_global": np.zeros((0,), dtype=np.int32),
            "simul_local": np.zeros((0,), dtype=np.int32),
            "q": np.zeros((0,), dtype=np.float32),
            "qc_flags": np.zeros((0,), dtype=np.int32),
            "neighbor_chs": np.zeros((0, k_len), dtype=np.int32),
            "geo": np.zeros((0, k_len, 4), dtype=np.float32),
            "rel_pos": np.zeros((0, k_len, 2), dtype=np.float32),
            "channel_mask": np.zeros((0, k_len), dtype=np.bool_),
        }
    arrays = label_candidates(
        arrays,
        input_h5=job.input_h5,
        fs=fs,
        n_channels=int(n_channels),
        labeling_cfg=labeling_cfg,
        preprocess_cfg=preprocess_cfg,
        car_mode=car_mode,
        neighbors_lookup=label_neighbors,
    )
    n = int(arrays["patches"].shape[0])
    arrays.update(
        {
            "recording_name": np.asarray([job.name] * n),
            "split": np.asarray([job.split] * n),
            "input_h5": np.asarray([str(job.input_h5)] * n),
            "n_channels": np.full((n,), int(n_channels), dtype=np.int32),
            "fs": np.full((n,), float(fs), dtype=np.float32),
            "domain_type": np.asarray(["single_channel" if int(n_channels) == 1 else "multi_channel"] * n),
            "car_mode": np.asarray([car_mode] * n),
        }
    )
    arrays["spatial_stats"] = build_gate_spatial_stats(arrays)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    train_mask = _default_training_mask(arrays, "training_mask")
    ignored_neighbor_events = np.asarray(arrays.get("ignored_neighbor_event", np.zeros((n,), dtype=np.bool_)), dtype=np.bool_)
    high_snr_near_gt = np.asarray(arrays.get("high_snr_near_gt", np.zeros((n,), dtype=np.bool_)), dtype=np.bool_)
    low_snr_near_gt = np.asarray(arrays.get("low_snr_near_gt", np.zeros((n,), dtype=np.bool_)), dtype=np.bool_)
    missed_gt_near_gt = np.asarray(arrays.get("missed_gt_near_gt", np.zeros((n,), dtype=np.bool_)), dtype=np.bool_)
    trainable_spikes = (arrays["spike_label"] > 0.5) & train_mask
    trainable_noise = (arrays["spike_label"] <= 0.5) & train_mask
    ignored_total_events = (arrays["spike_label"] <= 0.5) & (~train_mask)
    summary = {
        "job": job.name,
        "split": job.split,
        "input_h5": str(job.input_h5),
        "output_npz": str(output_npz),
        "samples_read": int(t0_read),
        "fs": float(fs),
        "n_channels": int(n_channels),
        "car_mode": car_mode,
        "events": n,
        "spikes": int(np.sum(arrays["spike_label"] > 0.5)),
        "noise": int(np.sum(trainable_noise)),
        "trainable_spikes": int(np.sum(trainable_spikes)),
        "trainable_noise": int(np.sum(trainable_noise)),
        "ignored_neighbor_events": int(np.sum(ignored_neighbor_events)),
        "ignored_total_events": int(np.sum(ignored_total_events)),
        "high_snr_ignored_candidates": int(np.sum(high_snr_near_gt)),
        "low_snr_ignored_candidates": int(np.sum(low_snr_near_gt)),
        "missed_gt_ignored_candidates": int(np.sum(missed_gt_near_gt)),
        "missed_gt_count": int(arrays["missed_gt_count"][0]) if "missed_gt_count" in arrays and arrays["missed_gt_count"].size else 0,
        "neighbor_primary": int(np.sum(arrays["neighbor_primary"])) if "neighbor_primary" in arrays else 0,
        "overlap_positive": int(np.sum(arrays["overlap_label"] > 0.5)),
        "near_gt_negative": int(np.sum(arrays["near_gt_negative"])),
        "duplicate_negative": int(np.sum(arrays["duplicate_negative"])),
        "gt_channel_source": str(arrays["gt_channel_source"][0]) if "gt_channel_source" in arrays and arrays["gt_channel_source"].size else "",
        "drift_gt_channel_enabled": bool(arrays["drift_gt_channel_enabled"][0]) if "drift_gt_channel_enabled" in arrays and arrays["drift_gt_channel_enabled"].size else False,
        "drift_gt_channel_valid": bool(arrays["drift_gt_channel_valid"][0]) if "drift_gt_channel_valid" in arrays and arrays["drift_gt_channel_valid"].size else False,
        "drift_gt_channel_changed_count": int(arrays["drift_gt_channel_changed_count"][0]) if "drift_gt_channel_changed_count" in arrays and arrays["drift_gt_channel_changed_count"].size else 0,
        "gt_align_enabled": bool(arrays["gt_align_enabled"][0]) if "gt_align_enabled" in arrays and arrays["gt_align_enabled"].size else False,
        "gt_align_radius_ms": float(arrays["gt_align_radius_ms"][0]) if "gt_align_radius_ms" in arrays and arrays["gt_align_radius_ms"].size else 0.0,
        "gt_align_median_shift_samples": float(arrays["gt_align_median_shift_samples"][0]) if "gt_align_median_shift_samples" in arrays and arrays["gt_align_median_shift_samples"].size else 0.0,
        "gt_align_p95_abs_shift_samples": float(arrays["gt_align_p95_abs_shift_samples"][0]) if "gt_align_p95_abs_shift_samples" in arrays and arrays["gt_align_p95_abs_shift_samples"].size else 0.0,
        "gt_align_max_abs_shift_samples": int(arrays["gt_align_max_abs_shift_samples"][0]) if "gt_align_max_abs_shift_samples" in arrays and arrays["gt_align_max_abs_shift_samples"].size else 0,
        "gt_align_channel_changed": int(arrays["gt_align_channel_changed"][0]) if "gt_align_channel_changed" in arrays and arrays["gt_align_channel_changed"].size else 0,
        "unit_snr_min_positive": float(arrays["unit_snr_min_positive"][0]) if "unit_snr_min_positive" in arrays and arrays["unit_snr_min_positive"].size else float(labeling_cfg.get("unit_snr_min_positive", 0.0) or 0.0),
        "unit_snr_filter_enabled": bool(arrays["unit_snr_filter_enabled"][0]) if "unit_snr_filter_enabled" in arrays and arrays["unit_snr_filter_enabled"].size else False,
        "high_snr_ignore_near_gt_ms": float(arrays["high_snr_ignore_near_gt_ms"][0]) if "high_snr_ignore_near_gt_ms" in arrays and arrays["high_snr_ignore_near_gt_ms"].size else float(labeling_cfg.get("high_snr_ignore_near_gt_ms", 0.0) or 0.0),
        "high_snr_ignore_neighbor_hops": int(arrays["high_snr_ignore_neighbor_hops"][0]) if "high_snr_ignore_neighbor_hops" in arrays and arrays["high_snr_ignore_neighbor_hops"].size else int(labeling_cfg.get("high_snr_ignore_neighbor_hops", labeling_cfg.get("neighbor_hops", 1))),
        "missed_gt_ignore_enabled": bool(arrays["missed_gt_ignore_enabled"][0]) if "missed_gt_ignore_enabled" in arrays and arrays["missed_gt_ignore_enabled"].size else _cfg_bool(labeling_cfg.get("missed_gt_ignore_enabled", True), default=True),
        "missed_gt_ignore_near_gt_ms": float(arrays["missed_gt_ignore_near_gt_ms"][0]) if "missed_gt_ignore_near_gt_ms" in arrays and arrays["missed_gt_ignore_near_gt_ms"].size else float(labeling_cfg.get("missed_gt_ignore_near_gt_ms", 0.5) or 0.0),
        "missed_gt_ignore_neighbor_hops": int(arrays["missed_gt_ignore_neighbor_hops"][0]) if "missed_gt_ignore_neighbor_hops" in arrays and arrays["missed_gt_ignore_neighbor_hops"].size else int(labeling_cfg.get("missed_gt_ignore_neighbor_hops", labeling_cfg.get("neighbor_hops", 1))),
        "require_drift_gt_channel_for_missed_ignore": bool(arrays["require_drift_gt_channel_for_missed_ignore"][0]) if "require_drift_gt_channel_for_missed_ignore" in arrays and arrays["require_drift_gt_channel_for_missed_ignore"].size else _cfg_bool(labeling_cfg.get("require_drift_gt_channel_for_missed_ignore", True), default=True),
        "missed_gt_drift_channel_valid": bool(arrays["missed_gt_drift_channel_valid"][0]) if "missed_gt_drift_channel_valid" in arrays and arrays["missed_gt_drift_channel_valid"].size else False,
        "gt_snr_high_unit_count": int(arrays["gt_snr_high_unit_count"][0]) if "gt_snr_high_unit_count" in arrays and arrays["gt_snr_high_unit_count"].size else 0,
        "gt_snr_low_unit_count": int(arrays["gt_snr_low_unit_count"][0]) if "gt_snr_low_unit_count" in arrays and arrays["gt_snr_low_unit_count"].size else 0,
        "gt_snr_missing_unit_count": int(arrays["gt_snr_missing_unit_count"][0]) if "gt_snr_missing_unit_count" in arrays and arrays["gt_snr_missing_unit_count"].size else 0,
        "gt_snr_high_event_count": int(arrays["gt_snr_high_event_count"][0]) if "gt_snr_high_event_count" in arrays and arrays["gt_snr_high_event_count"].size else 0,
        "gt_snr_low_event_count": int(arrays["gt_snr_low_event_count"][0]) if "gt_snr_low_event_count" in arrays and arrays["gt_snr_low_event_count"].size else 0,
        "gt_snr_missing_event_count": int(arrays["gt_snr_missing_event_count"][0]) if "gt_snr_missing_event_count" in arrays and arrays["gt_snr_missing_event_count"].size else 0,
        "label_neighbor_k": int(label_neighbor_k),
        "label_neighbor_hops": int(labeling_cfg.get("neighbor_hops", 1)),
        "detector_thr_max_amp_norm": float(det_cfg.thr_max_amp_norm),
        "detector_max_simul_firing": int(det_cfg.max_simul_firing),
        "detector_global_simul_threshold": int(detector.global_simul_threshold),
        "detector_local_simul_threshold": int(detector.local_simul_threshold),
        "detector_saturation_filter_mode": str(getattr(det_cfg, "saturation_filter_mode", "drop")),
        "raw_candidates": int(raw_candidates),
        "merged_events": int(merged_events),
        "hard_drops": int(hard_drops),
        "duplicate_drops": int(duplicate_drops),
        "global_simul_drops": int(global_simul_drops),
        "local_simul_drops": int(local_simul_drops),
        "patch_shape": [int(v) for v in arrays["patches"].shape],
    }
    logger.info(
        "candidate_job name=%s split=%s events=%d spikes=%d noise=%d high_snr_ignored=%d low_snr_ignored=%d missed_gt_ignored=%d overlap_pos=%d patch_shape=%s",
        job.name,
        job.split,
        n,
        summary["spikes"],
        summary["noise"],
        summary["high_snr_ignored_candidates"],
        summary["low_snr_ignored_candidates"],
        summary["missed_gt_ignored_candidates"],
        summary["overlap_positive"],
        tuple(summary["patch_shape"]),
    )
    return summary


def candidate_path(root: Path, job: GateJob) -> Path:
    return root / "candidates" / f"{job.split}_{job.name}.npz"


def ensure_candidates(
    jobs: Sequence[GateJob],
    *,
    cfg: Dict[str, Any],
    output_root: Path,
    rebuild: bool,
    max_seconds: float,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for job in jobs:
        out = candidate_path(output_root, job)
        if out.exists() and not rebuild:
            regenerate_cache = False
            missing_fields: List[str] = []
            stale_version = False
            stale_snr_config = False
            stale_high_snr_ignore_config = False
            stale_missed_gt_ignore_config = False
            with np.load(out, allow_pickle=False) as arr:
                required_fields = {
                    "neighbor_chs",
                    "geo",
                    "rel_pos",
                    "channel_mask",
                    "spatial_stats",
                    "training_mask",
                    "eval_mask",
                    "ignored_neighbor_event",
                    "matched_h5_unit_id",
                    "h5_unit_id",
                    "local_unit_id",
                    "matched_gt_snr",
                    "nearest_gt_snr",
                    "high_snr_near_gt",
                    "low_snr_near_gt",
                    "missed_gt_near_gt",
                    "unit_snr_min_positive",
                    "high_snr_ignore_near_gt_ms",
                    "high_snr_ignore_neighbor_hops",
                    "missed_gt_ignore_enabled",
                    "missed_gt_ignore_near_gt_ms",
                    "missed_gt_ignore_neighbor_hops",
                    "require_drift_gt_channel_for_missed_ignore",
                    "missed_gt_count",
                    "missed_gt_ignored_candidates",
                    "missed_gt_drift_channel_valid",
                    "gt_template_channel",
                    "gt_drift_channel",
                    "gt_anchor_channel",
                    "gt_peak_aligned_channel",
                    "gt_anchor_peak_aligned_channel_changed",
                    "drift_gt_channel_enabled",
                    "drift_gt_channel_valid",
                    "drift_gt_channel_changed_count",
                    "labeling_version",
                }
                missing_fields = sorted(required_fields - set(arr.files))
                stale_version = "labeling_version" in arr.files and arr["labeling_version"].size > 0 and str(arr["labeling_version"][0]) != LABELING_VERSION
                requested_unit_snr_min = float(dict(cfg.get("labeling", {}) or {}).get("unit_snr_min_positive", 0.0) or 0.0)
                stale_snr_config = (
                    "unit_snr_min_positive" in arr.files
                    and arr["unit_snr_min_positive"].size > 0
                    and abs(float(arr["unit_snr_min_positive"][0]) - requested_unit_snr_min) > 1e-6
                )
                labeling_cfg = dict(cfg.get("labeling", {}) or {})
                requested_high_snr_ignore_ms = float(labeling_cfg.get("high_snr_ignore_near_gt_ms", 0.0) or 0.0)
                requested_high_snr_ignore_hops = int(labeling_cfg.get("high_snr_ignore_neighbor_hops", labeling_cfg.get("neighbor_hops", 1)))
                requested_missed_gt_ignore_enabled = _cfg_bool(labeling_cfg.get("missed_gt_ignore_enabled", True), default=True)
                requested_missed_gt_ignore_ms = float(labeling_cfg.get("missed_gt_ignore_near_gt_ms", 0.5) or 0.0)
                requested_missed_gt_ignore_hops = int(labeling_cfg.get("missed_gt_ignore_neighbor_hops", labeling_cfg.get("neighbor_hops", 1)))
                requested_require_drift_missed = _cfg_bool(labeling_cfg.get("require_drift_gt_channel_for_missed_ignore", True), default=True)
                stale_high_snr_ignore_config = (
                    "high_snr_ignore_near_gt_ms" in arr.files
                    and arr["high_snr_ignore_near_gt_ms"].size > 0
                    and abs(float(arr["high_snr_ignore_near_gt_ms"][0]) - requested_high_snr_ignore_ms) > 1e-6
                ) or (
                    "high_snr_ignore_neighbor_hops" in arr.files
                    and arr["high_snr_ignore_neighbor_hops"].size > 0
                    and int(arr["high_snr_ignore_neighbor_hops"][0]) != int(requested_high_snr_ignore_hops)
                )
                stale_missed_gt_ignore_config = (
                    "missed_gt_ignore_enabled" in arr.files
                    and arr["missed_gt_ignore_enabled"].size > 0
                    and bool(arr["missed_gt_ignore_enabled"][0]) != bool(requested_missed_gt_ignore_enabled)
                ) or (
                    "missed_gt_ignore_near_gt_ms" in arr.files
                    and arr["missed_gt_ignore_near_gt_ms"].size > 0
                    and abs(float(arr["missed_gt_ignore_near_gt_ms"][0]) - requested_missed_gt_ignore_ms) > 1e-6
                ) or (
                    "missed_gt_ignore_neighbor_hops" in arr.files
                    and arr["missed_gt_ignore_neighbor_hops"].size > 0
                    and int(arr["missed_gt_ignore_neighbor_hops"][0]) != int(requested_missed_gt_ignore_hops)
                ) or (
                    "require_drift_gt_channel_for_missed_ignore" in arr.files
                    and arr["require_drift_gt_channel_for_missed_ignore"].size > 0
                    and bool(arr["require_drift_gt_channel_for_missed_ignore"][0]) != bool(requested_require_drift_missed)
                )
                if missing_fields or stale_version or stale_snr_config or stale_high_snr_ignore_config or stale_missed_gt_ignore_config:
                    regenerate_cache = True
                else:
                    train_mask = np.asarray(arr["training_mask"], dtype=np.bool_) if "training_mask" in arr else np.ones((arr["spike_label"].shape[0],), dtype=np.bool_)
                    ignored_neighbor_events = np.asarray(arr["ignored_neighbor_event"], dtype=np.bool_) if "ignored_neighbor_event" in arr else np.zeros((arr["spike_label"].shape[0],), dtype=np.bool_)
                    high_snr_near_gt = np.asarray(arr["high_snr_near_gt"], dtype=np.bool_) if "high_snr_near_gt" in arr else np.zeros((arr["spike_label"].shape[0],), dtype=np.bool_)
                    low_snr_near_gt = np.asarray(arr["low_snr_near_gt"], dtype=np.bool_) if "low_snr_near_gt" in arr else np.zeros((arr["spike_label"].shape[0],), dtype=np.bool_)
                    missed_gt_near_gt = np.asarray(arr["missed_gt_near_gt"], dtype=np.bool_) if "missed_gt_near_gt" in arr else np.zeros((arr["spike_label"].shape[0],), dtype=np.bool_)
                    trainable_spikes = (np.asarray(arr["spike_label"], dtype=np.float32) > 0.5) & train_mask
                    trainable_noise = (np.asarray(arr["spike_label"], dtype=np.float32) <= 0.5) & train_mask
                    ignored_total_events = (np.asarray(arr["spike_label"], dtype=np.float32) <= 0.5) & (~train_mask)
                    summaries.append(
                        {
                            "job": job.name,
                            "split": job.split,
                            "input_h5": str(job.input_h5),
                            "output_npz": str(out),
                            "events": int(arr["patches"].shape[0]),
                            "spikes": int(np.sum(arr["spike_label"] > 0.5)),
                            "noise": int(np.sum(trainable_noise)),
                            "trainable_spikes": int(np.sum(trainable_spikes)),
                            "trainable_noise": int(np.sum(trainable_noise)),
                            "ignored_neighbor_events": int(np.sum(ignored_neighbor_events)),
                            "ignored_total_events": int(np.sum(ignored_total_events)),
                            "high_snr_ignored_candidates": int(np.sum(high_snr_near_gt)),
                            "low_snr_ignored_candidates": int(np.sum(low_snr_near_gt)),
                            "missed_gt_ignored_candidates": int(np.sum(missed_gt_near_gt)),
                            "missed_gt_count": int(arr["missed_gt_count"][0]) if "missed_gt_count" in arr and arr["missed_gt_count"].size else 0,
                            "neighbor_primary": int(np.sum(arr["neighbor_primary"])) if "neighbor_primary" in arr else 0,
                            "overlap_positive": int(np.sum(arr["overlap_label"] > 0.5)),
                            "near_gt_negative": int(np.sum(arr["near_gt_negative"])) if "near_gt_negative" in arr else 0,
                            "duplicate_negative": int(np.sum(arr["duplicate_negative"])) if "duplicate_negative" in arr else 0,
                            "gt_channel_source": str(arr["gt_channel_source"][0]) if "gt_channel_source" in arr and arr["gt_channel_source"].size else "",
                            "drift_gt_channel_enabled": bool(arr["drift_gt_channel_enabled"][0]) if "drift_gt_channel_enabled" in arr and arr["drift_gt_channel_enabled"].size else False,
                            "drift_gt_channel_valid": bool(arr["drift_gt_channel_valid"][0]) if "drift_gt_channel_valid" in arr and arr["drift_gt_channel_valid"].size else False,
                            "drift_gt_channel_changed_count": int(arr["drift_gt_channel_changed_count"][0]) if "drift_gt_channel_changed_count" in arr and arr["drift_gt_channel_changed_count"].size else 0,
                            "gt_align_enabled": bool(arr["gt_align_enabled"][0]) if "gt_align_enabled" in arr and arr["gt_align_enabled"].size else False,
                            "gt_align_radius_ms": float(arr["gt_align_radius_ms"][0]) if "gt_align_radius_ms" in arr and arr["gt_align_radius_ms"].size else 0.0,
                            "gt_align_median_shift_samples": float(arr["gt_align_median_shift_samples"][0]) if "gt_align_median_shift_samples" in arr and arr["gt_align_median_shift_samples"].size else 0.0,
                            "gt_align_p95_abs_shift_samples": float(arr["gt_align_p95_abs_shift_samples"][0]) if "gt_align_p95_abs_shift_samples" in arr and arr["gt_align_p95_abs_shift_samples"].size else 0.0,
                            "gt_align_max_abs_shift_samples": int(arr["gt_align_max_abs_shift_samples"][0]) if "gt_align_max_abs_shift_samples" in arr and arr["gt_align_max_abs_shift_samples"].size else 0,
                            "gt_align_channel_changed": int(arr["gt_align_channel_changed"][0]) if "gt_align_channel_changed" in arr and arr["gt_align_channel_changed"].size else 0,
                            "unit_snr_min_positive": float(arr["unit_snr_min_positive"][0]) if "unit_snr_min_positive" in arr and arr["unit_snr_min_positive"].size else 0.0,
                            "unit_snr_filter_enabled": bool(arr["unit_snr_filter_enabled"][0]) if "unit_snr_filter_enabled" in arr and arr["unit_snr_filter_enabled"].size else False,
                            "high_snr_ignore_near_gt_ms": float(arr["high_snr_ignore_near_gt_ms"][0]) if "high_snr_ignore_near_gt_ms" in arr and arr["high_snr_ignore_near_gt_ms"].size else 0.0,
                            "high_snr_ignore_neighbor_hops": int(arr["high_snr_ignore_neighbor_hops"][0]) if "high_snr_ignore_neighbor_hops" in arr and arr["high_snr_ignore_neighbor_hops"].size else 0,
                            "missed_gt_ignore_enabled": bool(arr["missed_gt_ignore_enabled"][0]) if "missed_gt_ignore_enabled" in arr and arr["missed_gt_ignore_enabled"].size else False,
                            "missed_gt_ignore_near_gt_ms": float(arr["missed_gt_ignore_near_gt_ms"][0]) if "missed_gt_ignore_near_gt_ms" in arr and arr["missed_gt_ignore_near_gt_ms"].size else 0.0,
                            "missed_gt_ignore_neighbor_hops": int(arr["missed_gt_ignore_neighbor_hops"][0]) if "missed_gt_ignore_neighbor_hops" in arr and arr["missed_gt_ignore_neighbor_hops"].size else 0,
                            "require_drift_gt_channel_for_missed_ignore": bool(arr["require_drift_gt_channel_for_missed_ignore"][0]) if "require_drift_gt_channel_for_missed_ignore" in arr and arr["require_drift_gt_channel_for_missed_ignore"].size else False,
                            "missed_gt_drift_channel_valid": bool(arr["missed_gt_drift_channel_valid"][0]) if "missed_gt_drift_channel_valid" in arr and arr["missed_gt_drift_channel_valid"].size else False,
                            "gt_snr_high_unit_count": int(arr["gt_snr_high_unit_count"][0]) if "gt_snr_high_unit_count" in arr and arr["gt_snr_high_unit_count"].size else 0,
                            "gt_snr_low_unit_count": int(arr["gt_snr_low_unit_count"][0]) if "gt_snr_low_unit_count" in arr and arr["gt_snr_low_unit_count"].size else 0,
                            "gt_snr_missing_unit_count": int(arr["gt_snr_missing_unit_count"][0]) if "gt_snr_missing_unit_count" in arr and arr["gt_snr_missing_unit_count"].size else 0,
                            "gt_snr_high_event_count": int(arr["gt_snr_high_event_count"][0]) if "gt_snr_high_event_count" in arr and arr["gt_snr_high_event_count"].size else 0,
                            "gt_snr_low_event_count": int(arr["gt_snr_low_event_count"][0]) if "gt_snr_low_event_count" in arr and arr["gt_snr_low_event_count"].size else 0,
                            "gt_snr_missing_event_count": int(arr["gt_snr_missing_event_count"][0]) if "gt_snr_missing_event_count" in arr and arr["gt_snr_missing_event_count"].size else 0,
                            "n_channels": int(arr["n_channels"][0]) if "n_channels" in arr and arr["n_channels"].size else 0,
                            "car_mode": str(arr["car_mode"][0]) if "car_mode" in arr and arr["car_mode"].size else "",
                            "patch_shape": [int(v) for v in arr["patches"].shape],
                            "cached": True,
                        }
                    )
            if regenerate_cache:
                logger.info(
                    "candidate_cache_stale job=%s path=%s missing=%s stale_version=%s stale_snr_config=%s stale_high_snr_ignore_config=%s stale_missed_gt_ignore_config=%s",
                    job.name,
                    out,
                    ",".join(missing_fields),
                    bool(stale_version),
                    bool(stale_snr_config),
                    bool(stale_high_snr_ignore_config),
                    bool(stale_missed_gt_ignore_config),
                )
                summaries.append(generate_candidates_for_job(job, cfg=cfg, output_npz=out, max_seconds=max_seconds, logger=logger))
                continue
            logger.info("candidate_cache_hit job=%s path=%s", job.name, out)
            continue
        summaries.append(generate_candidates_for_job(job, cfg=cfg, output_npz=out, max_seconds=max_seconds, logger=logger))
    return summaries


def load_split_arrays(paths: Sequence[Path]) -> Dict[str, np.ndarray]:
    parts: Dict[str, List[np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as arr:
            for key in arr.files:
                if key not in parts:
                    parts[key] = []
                parts[key].append(np.asarray(arr[key]))
    out: Dict[str, np.ndarray] = {}
    for key, values in parts.items():
        if not values:
            continue
        out[key] = np.concatenate(values, axis=0)
    return out


def freeze_encoder(model: SpikeEmbeddingModel) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("gate_net.")


def compute_natural_sampler_probs(arr: Dict[str, np.ndarray]) -> np.ndarray:
    train_mask = _default_training_mask(arr, "training_mask")
    weights = train_mask.astype(np.float64, copy=False)
    if not np.any(weights > 0):
        raise RuntimeError("no trainable gate candidates after applying training_mask")
    weights /= float(np.sum(weights))
    return weights.astype(np.float64, copy=False)


def predict_gate(
    model: SpikeEmbeddingModel,
    arr: Dict[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    spike_parts: List[np.ndarray] = []
    patches = np.asarray(arr["patches"], dtype=np.float32)
    with torch.no_grad():
        for st in range(0, int(patches.shape[0]), max(1, int(batch_size))):
            ed = min(st + max(1, int(batch_size)), int(patches.shape[0]))
            xb = torch.from_numpy(patches[st:ed].astype(np.float32, copy=False)).to(device)
            meta = build_gate_meta(arr, slice(st, ed), device=device)
            _, spike_prob = model.forward_gate_heads(xb, meta)
            spike_parts.append(spike_prob.detach().cpu().numpy().astype(np.float32, copy=False))
    if not spike_parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(spike_parts, axis=0)


def binary_metrics(prob: np.ndarray, label: np.ndarray, *, threshold: float) -> Dict[str, float]:
    prob = np.asarray(prob, dtype=np.float32)
    label_bool = np.asarray(label, dtype=np.float32) > 0.5
    pred = prob >= float(threshold)
    tp = int(np.sum(pred & label_bool))
    fp = int(np.sum(pred & (~label_bool)))
    fn = int(np.sum((~pred) & label_bool))
    tn = int(np.sum((~pred) & (~label_bool)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2.0 * precision * recall) / max(1e-8, precision + recall))
    beta2 = 0.5 * 0.5
    f05 = float((1.0 + beta2) * precision * recall / max(1e-8, beta2 * precision + recall))
    noise_reject = float(tn / max(1, tn + fp))
    accuracy = float((tp + tn) / max(1, int(prob.size)))
    return {
        "threshold": float(threshold),
        "n": int(prob.size),
        "positive": int(np.sum(label_bool)),
        "negative": int(np.sum(~label_bool)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0_5": f05,
        "accuracy": accuracy,
        "balanced_accuracy": float(0.5 * (recall + noise_reject)),
        "noise_reject_rate": noise_reject,
        "noise_accuracy": noise_reject,
        "noise_pass_count": fp,
        "spike_block_count": fn,
        "pass_count": int(np.sum(pred)),
        "pass_ratio": float(np.mean(pred.astype(np.float32))) if pred.size > 0 else float("nan"),
    }


def average_precision(prob: np.ndarray, label: np.ndarray) -> float:
    prob = np.asarray(prob, dtype=np.float32)
    label_bool = np.asarray(label, dtype=np.float32) > 0.5
    if prob.size == 0 or not np.any(label_bool):
        return float("nan")
    order = np.argsort(-prob, kind="mergesort")
    y = label_bool[order].astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / np.maximum(1.0, np.arange(1, y.size + 1, dtype=np.float64))
    return float(np.sum(precision * y) / max(1.0, float(np.sum(y))))


def threshold_curve_rows(
    *,
    split: str,
    recording: str,
    domain_type: str,
    head: str,
    prob: np.ndarray,
    label: np.ndarray,
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    aucpr = average_precision(prob, label)
    for th in thresholds:
        row = binary_metrics(prob, label, threshold=float(th))
        row.update({"split": split, "recording": recording, "domain_type": domain_type, "head": head, "aucpr": aucpr})
        rows.append(row)
    return rows


def summarize_groups(prob: np.ndarray, label: np.ndarray, arr: Dict[str, np.ndarray], *, threshold: float) -> Dict[str, Any]:
    groups: Dict[str, Any] = {}
    if "n_channels" in arr:
        entries = []
        n_channels_values = np.asarray(arr["n_channels"], dtype=np.int32)
        for n_ch in np.unique(n_channels_values).tolist():
            mask = n_channels_values == int(n_ch)
            if np.any(mask):
                metric = binary_metrics(prob[mask], label[mask], threshold=threshold)
                metric["n_channels"] = int(n_ch)
                entries.append(metric)
        groups["n_channels"] = entries
    bins = {
        "amp_bin": (np.asarray(arr["amp_norm"], dtype=np.float32), [0, 3, 4, 5, 6, 8, 10, 15, float("inf")]),
        "q_bin": (np.asarray(arr["q"], dtype=np.float32), [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.01]),
    }
    for name, (values, edges) in bins.items():
        entries = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (values >= lo) & (values < hi)
            if not np.any(mask):
                continue
            metric = binary_metrics(prob[mask], label[mask], threshold=threshold)
            metric["bin"] = f"[{lo},{hi})" if math.isfinite(hi) else f">={lo}"
            entries.append(metric)
        groups[name] = entries
    for flag_name in ("near_gt_negative", "duplicate_negative"):
        if flag_name in arr:
            mask = np.asarray(arr[flag_name], dtype=np.bool_)
            groups[flag_name] = binary_metrics(prob[mask], label[mask], threshold=threshold) if np.any(mask) else {"n": 0}
    return groups


def evaluate_model(
    model: SpikeEmbeddingModel,
    jobs: Sequence[GateJob],
    *,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    thresholds: Sequence[float],
    calibrated_thresholds: Dict[str, float] | None = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metrics: Dict[str, Any] = {"recordings": {}, "domain": {}}
    curve_rows: List[Dict[str, Any]] = []
    pooled_by_domain: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for job in jobs:
        path = candidate_path(output_root, job)
        with np.load(path, allow_pickle=False) as arr_npz:
            arr = {key: np.asarray(arr_npz[key]) for key in arr_npz.files}
        spike_prob = predict_gate(model, arr, device=device, batch_size=batch_size)
        raw_label = np.asarray(arr["spike_label"], dtype=np.float32)
        eval_mask = _default_training_mask(arr, "eval_mask")
        arr_eval = _mask_event_arrays(arr, eval_mask)
        spike_prob_eval = spike_prob[eval_mask]
        label = raw_label[eval_mask]
        domain_type = str(arr["domain_type"][0]) if raw_label.size else "unknown"
        default_th = float((calibrated_thresholds or {}).get(domain_type, 0.5))
        rec_metrics = {
            "split": job.split,
            "domain_type": domain_type,
            "n_channels": int(arr["n_channels"][0]) if label.size else 0,
            "n": int(label.size),
            "total_candidates": int(raw_label.size),
            "ignored_neighbor_events": int(np.sum(np.asarray(arr.get("ignored_neighbor_event", np.zeros(raw_label.shape, dtype=np.bool_)), dtype=np.bool_))),
            "fixed_0_5": binary_metrics(spike_prob_eval, label, threshold=0.5),
            "calibrated": binary_metrics(spike_prob_eval, label, threshold=default_th),
            "spike_aucpr": average_precision(spike_prob_eval, label),
            "groups": summarize_groups(spike_prob_eval, label, arr_eval, threshold=default_th),
        }
        metrics["recordings"][job.name] = rec_metrics
        curve_rows.extend(
            threshold_curve_rows(
                split=job.split,
                recording=job.name,
                domain_type=domain_type,
                head="spike_noise",
                prob=spike_prob_eval,
                label=label,
                thresholds=thresholds,
            )
        )
        bucket = pooled_by_domain.setdefault(domain_type, {"prob": [], "label": []})
        bucket["prob"].append(spike_prob_eval)
        bucket["label"].append(label)

    for domain_type, bucket in pooled_by_domain.items():
        prob = np.concatenate(bucket["prob"], axis=0) if bucket["prob"] else np.zeros((0,), dtype=np.float32)
        label = np.concatenate(bucket["label"], axis=0) if bucket["label"] else np.zeros((0,), dtype=np.float32)
        metrics["domain"][domain_type] = {
            "fixed_0_5": binary_metrics(prob, label, threshold=0.5),
            "aucpr": average_precision(prob, label),
        }
        for th in thresholds:
            metrics["domain"][domain_type][f"threshold_{th:g}"] = binary_metrics(prob, label, threshold=float(th))
    return metrics, curve_rows


def _summary_float(mapping: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _summary_int(mapping: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(mapping.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _split_evaluation_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    domain = dict(metrics.get("domain", {}) or {})
    out: Dict[str, Any] = {
        "domain": domain,
        "multi_channel": dict(domain.get("multi_channel", {}) or {}),
        "recordings": {},
    }
    for recording, rec_raw in dict(metrics.get("recordings", {}) or {}).items():
        rec = dict(rec_raw or {})
        fixed = dict(rec.get("fixed_0_5", {}) or {})
        out["recordings"][str(recording)] = {
            "split": str(rec.get("split", "")),
            "domain_type": str(rec.get("domain_type", "")),
            "n_channels": _summary_int(rec, "n_channels", 0),
            "fixed_0_5": fixed,
            "spike_aucpr": _summary_float(rec, "spike_aucpr"),
            "groups": rec.get("groups", {}),
        }
    return out


def build_evaluation_summary(
    *,
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    eval_all_metrics: Dict[str, Any],
    train_summary: Dict[str, Any],
    calibration: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    eval_all = _split_evaluation_summary(eval_all_metrics)
    acceptance = fixed_0_5_acceptance(eval_all_metrics, cfg)
    train_best = dict(train_summary.get("best_acceptance", {}) or {})
    return {
        "fixed_threshold": 0.5,
        "acceptance": acceptance,
        "best_epoch": int(train_summary.get("best_epoch", 0)),
        "best_validation_acceptance": train_best,
        "sampler_mode": str(train_summary.get("sampler_mode", "")),
        "loss_weight_mode": str(train_summary.get("loss_weight_mode", "")),
        "calibration": calibration,
        "train": train_summary,
        "val": _split_evaluation_summary(val_metrics),
        "test": _split_evaluation_summary(test_metrics),
        "eval_all": eval_all,
    }


def log_evaluation_summary(logger: logging.Logger, summary: Dict[str, Any]) -> None:
    for split_name in ("val", "test", "eval_all"):
        split = dict(summary.get(split_name, {}) or {})
        domain = dict(split.get("multi_channel", {}) or {})
        fixed = dict(domain.get("fixed_0_5", {}) or {})
        logger.info(
            "eval_summary split=%s threshold=0.5000 n=%d positive=%d negative=%d precision=%.4f recall=%.4f f0_5=%.4f noise_accuracy=%.4f pass_ratio=%.4f",
            split_name,
            _summary_int(fixed, "n"),
            _summary_int(fixed, "positive"),
            _summary_int(fixed, "negative"),
            _summary_float(fixed, "precision"),
            _summary_float(fixed, "recall"),
            _summary_float(fixed, "f0_5"),
            _summary_float(fixed, "noise_accuracy", _summary_float(fixed, "noise_reject_rate")),
            _summary_float(fixed, "pass_ratio"),
        )
        if split_name == "eval_all":
            continue
        for recording, rec_raw in dict(split.get("recordings", {}) or {}).items():
            rec = dict(rec_raw or {})
            fixed_rec = dict(rec.get("fixed_0_5", {}) or {})
            logger.info(
                "%s_eval_recording recording=%s threshold=0.5000 n=%d positive=%d negative=%d "
                "spike_precision=%.4f spike_recall=%.4f spike_f1=%.4f noise_accuracy=%.4f",
                split_name,
                recording,
                _summary_int(fixed_rec, "n"),
                _summary_int(fixed_rec, "positive"),
                _summary_int(fixed_rec, "negative"),
                _summary_float(fixed_rec, "precision"),
                _summary_float(fixed_rec, "recall"),
                _summary_float(fixed_rec, "f1"),
                _summary_float(fixed_rec, "noise_accuracy", _summary_float(fixed_rec, "noise_reject_rate")),
            )


def calibrate_thresholds(
    model: SpikeEmbeddingModel,
    val_jobs: Sequence[GateJob],
    *,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    thresholds: Sequence[float],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    pooled: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for job in val_jobs:
        with np.load(candidate_path(output_root, job), allow_pickle=False) as arr_npz:
            arr = {key: np.asarray(arr_npz[key]) for key in arr_npz.files}
        prob = predict_gate(model, arr, device=device, batch_size=batch_size)
        eval_mask = _default_training_mask(arr, "eval_mask")
        prob = prob[eval_mask]
        label = np.asarray(arr["spike_label"], dtype=np.float32)[eval_mask]
        domain = str(arr["domain_type"][0]) if arr["spike_label"].size else "unknown"
        bucket = pooled.setdefault(domain, {"prob": [], "label": []})
        bucket["prob"].append(prob)
        bucket["label"].append(label)
    cal_cfg = dict(cfg.get("calibration", {}) or {})
    out: Dict[str, Any] = {}
    for domain, bucket in pooled.items():
        prob = np.concatenate(bucket["prob"], axis=0)
        label = np.concatenate(bucket["label"], axis=0)
        rows = [binary_metrics(prob, label, threshold=float(th)) for th in thresholds]
        threshold_mode = str(cal_cfg.get("mode", "adaptive")).strip().lower()
        if threshold_mode in ("fixed", "fixed_0_5", "fixed_threshold"):
            fixed_threshold = float(cal_cfg.get("fixed_threshold", 0.5))
            choice = binary_metrics(prob, label, threshold=fixed_threshold)
            mode = "fixed"
        elif domain == "single_channel":
            min_precision = float(cal_cfg.get("single_channel_min_precision", 0.98))
            eligible = [r for r in rows if float(r["precision"]) >= min_precision]
            if eligible:
                choice = max(eligible, key=lambda r: (float(r["recall"]), float(r["f0_5"]), -float(r["threshold"])))
                mode = "max_recall_at_precision"
            else:
                choice = max(rows, key=lambda r: (float(r["f0_5"]), float(r["precision"])))
                mode = "fallback_max_f0_5"
        else:
            min_recall = float(cal_cfg.get("multi_channel_min_recall", 0.80))
            eligible = [r for r in rows if float(r["recall"]) >= min_recall]
            if eligible:
                choice = max(eligible, key=lambda r: (float(r["f0_5"]), float(r["precision"]), -float(r["threshold"])))
                mode = "max_f0_5_at_recall"
            else:
                choice = max(rows, key=lambda r: (float(r["f0_5"]), float(r["recall"])))
                mode = "fallback_max_f0_5"
        out[domain] = {"threshold": float(choice["threshold"]), "selection_mode": mode, "metrics": choice}
    return out


def clone_model_state_dict(model: SpikeEmbeddingModel) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fixed_0_5_acceptance(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    acc_cfg = dict(cfg.get("acceptance", {}) or {})
    min_precision = float(acc_cfg.get("min_precision", 0.99))
    min_recall = float(acc_cfg.get("min_recall", 0.90))
    domain_metrics = dict(metrics.get("domain", {}) or {})
    if "multi_channel" in domain_metrics:
        domain = "multi_channel"
    elif domain_metrics:
        domain = sorted(domain_metrics.keys())[0]
    else:
        domain = "none"
    fixed = dict(domain_metrics.get(domain, {}).get("fixed_0_5", {}) or {})
    precision = float(fixed.get("precision", 0.0))
    recall = float(fixed.get("recall", 0.0))
    return {
        "domain": domain,
        "threshold": 0.5,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "precision": precision,
        "recall": recall,
        "f0_5": float(fixed.get("f0_5", 0.0)),
        "noise_accuracy": float(fixed.get("noise_accuracy", fixed.get("noise_reject_rate", 0.0))),
        "passes": bool(precision >= min_precision and recall >= min_recall),
        "metrics": fixed,
    }


def acceptance_rank(acceptance: Dict[str, Any]) -> tuple[int, float, float, float]:
    return (
        1 if bool(acceptance.get("passes", False)) else 0,
        float(acceptance.get("f0_5", 0.0)),
        float(acceptance.get("precision", 0.0)),
        float(acceptance.get("recall", 0.0)),
    )


def train_gate(
    model: SpikeEmbeddingModel,
    train_arr: Dict[str, np.ndarray],
    val_jobs: Sequence[GateJob],
    *,
    cfg: Dict[str, Any],
    output_root: Path,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Any]:
    train_cfg = dict(cfg.get("train", {}) or {})
    model.to(device)
    freeze_encoder(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 7e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    sampler_mode = str(train_cfg.get("sampler_mode", "natural")).strip().lower()
    if sampler_mode in ("candidate", "candidate_prior", "deployment"):
        sampler_mode = "natural"
    if sampler_mode != "natural":
        raise ValueError(f"unsupported train.sampler_mode={sampler_mode!r}; expected natural")
    loss_weight_mode = str(train_cfg.get("loss_weight_mode", "none")).strip().lower()
    if loss_weight_mode != "none":
        raise ValueError(f"unsupported train.loss_weight_mode={loss_weight_mode!r}; expected none")
    gate_loss_type = str(train_cfg.get("gate_loss_type", train_cfg.get("loss_type", "bce"))).strip().lower()
    if gate_loss_type in ("binary_cross_entropy", "binary_cross_entropy_with_logits"):
        gate_loss_type = "bce"
    if gate_loss_type != "bce":
        raise ValueError(f"unsupported train.gate_loss_type={gate_loss_type!r}; expected bce")
    if bool(train_cfg.get("use_global_pos_weight", False)):
        raise ValueError("train.use_global_pos_weight=true is unsupported for natural BCE gate training")
    legacy_keys = ("focal_gamma",)
    legacy_present = [key for key in legacy_keys if key in train_cfg]
    if legacy_present:
        raise ValueError(
            "unsupported legacy train keys for natural BCE gate training: "
            + ", ".join(sorted(legacy_present))
        )
    probs = compute_natural_sampler_probs(train_arr)
    n = int(train_arr["patches"].shape[0])
    batch_size = int(train_cfg.get("batch_size", 512))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", max(1, math.ceil(n / max(1, batch_size)))))
    epochs = int(train_cfg.get("epochs", 20))
    validation_every = max(1, int(train_cfg.get("validation_every_epochs", 1)))
    thresholds = [float(x) for x in cfg.get("calibration", {}).get("thresholds", [0.5])]
    history: List[Dict[str, Any]] = []
    best_rank: tuple[int, float, float, float] | None = None
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_acceptance: Dict[str, Any] = {}
    logger.info(
        "train_gate sampler_mode=%s loss_weight_mode=%s fixed_threshold=0.5000 min_precision=%.4f min_recall=%.4f",
        sampler_mode,
        loss_weight_mode,
        float(dict(cfg.get("acceptance", {}) or {}).get("min_precision", 0.99)),
        float(dict(cfg.get("acceptance", {}) or {}).get("min_recall", 0.90)),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        t0 = time.perf_counter()
        for _ in range(steps_per_epoch):
            idx = rng.choice(n, size=batch_size, replace=True, p=probs)
            x = torch.from_numpy(train_arr["patches"][idx].astype(np.float32, copy=False)).to(device)
            spike = torch.from_numpy(train_arr["spike_label"][idx].astype(np.float32, copy=False)).to(device)
            meta = build_gate_meta(train_arr, idx, device=device)
            optimizer.zero_grad(set_to_none=True)
            spike_logit, _ = model.forward_gate_heads(x, meta)
            loss, _terms = gate_detection_loss(
                spike_logit,
                spike,
                gate_sample_weights=None,
                gate_loss_type=gate_loss_type,
                lambda_gate=float(train_cfg.get("lambda_gate", 1.0)),
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item())
        epoch_record: Dict[str, Any] = {
            "epoch": epoch,
            "loss": float(loss_sum / max(1, steps_per_epoch)),
            "seconds": float(time.perf_counter() - t0),
        }
        if epoch % validation_every == 0 or epoch == epochs:
            val_metrics, _ = evaluate_model(
                model,
                val_jobs,
                output_root=output_root,
                device=device,
                batch_size=batch_size,
                thresholds=thresholds,
            )
            epoch_record["val_fixed_0_5"] = {
                domain: metrics.get("fixed_0_5", {})
                for domain, metrics in val_metrics.get("domain", {}).items()
            }
            acceptance = fixed_0_5_acceptance(val_metrics, cfg)
            epoch_record["acceptance"] = dict(acceptance)
            rank = acceptance_rank(acceptance)
            is_best = best_rank is None or rank > best_rank
            epoch_record["is_best"] = bool(is_best)
            if is_best:
                best_rank = rank
                best_state = clone_model_state_dict(model)
                best_epoch = int(epoch)
                best_acceptance = dict(acceptance)
            logger.info(
                "val_fixed_0_5 epoch=%d domain=%s precision=%.4f recall=%.4f f0_5=%.4f noise_accuracy=%.4f passes=%s best_epoch=%d",
                epoch,
                str(acceptance.get("domain", "")),
                float(acceptance.get("precision", float("nan"))),
                float(acceptance.get("recall", float("nan"))),
                float(acceptance.get("f0_5", float("nan"))),
                float(acceptance.get("noise_accuracy", float("nan"))),
                str(bool(acceptance.get("passes", False))),
                int(best_epoch),
            )
        history.append(epoch_record)
        logger.info(
            "epoch=%d/%d loss=%.6f seconds=%.2f",
            epoch,
            epochs,
            epoch_record["loss"],
            epoch_record["seconds"],
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_rank": list(best_rank) if best_rank is not None else [],
        "best_acceptance": dict(best_acceptance),
        "sampler_mode": sampler_mode,
        "loss_weight_mode": loss_weight_mode,
    }


def write_threshold_curves(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "split",
        "recording",
        "domain_type",
        "head",
        "threshold",
        "n",
        "positive",
        "negative",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "f0_5",
        "accuracy",
        "balanced_accuracy",
        "noise_reject_rate",
        "noise_accuracy",
        "noise_pass_count",
        "spike_block_count",
        "pass_count",
        "pass_ratio",
        "aucpr",
    ]
    keys = preferred + sorted({k for row in rows for k in row.keys()} - set(preferred))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def run_sanity_checks(summaries: Sequence[Dict[str, Any]], *, dry_run: bool, cfg: Dict[str, Any]) -> Dict[str, Any]:
    patch_cfg = dict(cfg.get("patch", {}) or {})
    model_cfg = dict(cfg.get("model", {}) or {})
    expected_k = 1 if _cfg_bool(patch_cfg.get("peak_only", False), default=False) else int(patch_cfg.get("k", 1))
    out = {
        "ok": True,
        "candidate_generation_path": "detector_then_neighbor_aware_gt_labeling",
        "direct_gt_patch_extraction": False,
        "gate_spatial_dim": int(model_cfg.get("gate_spatial_dim", 0)),
        "expected_patch_k": int(expected_k),
        "checks": [],
    }
    for summary in summaries:
        shape = summary.get("patch_shape", [])
        ok_shape = len(shape) == 3 and int(shape[1]) == 64 and int(shape[2]) == int(expected_k)
        ignored_total = int(summary.get("ignored_total_events", summary.get("ignored_neighbor_events", 0)))
        labels_accounted = (
            int(summary.get("spikes", 0))
            + int(summary.get("noise", 0))
            + ignored_total
            == int(summary.get("events", 0))
        )
        check = {
            "job": summary.get("job"),
            "patch_shape_ok": bool(ok_shape),
            "labels_accounted": bool(labels_accounted),
            "car_mode": summary.get("car_mode"),
            "events": int(summary.get("events", 0)),
            "spikes": int(summary.get("spikes", 0)),
            "noise": int(summary.get("noise", 0)),
            "trainable_spikes": int(summary.get("trainable_spikes", summary.get("spikes", 0))),
            "trainable_noise": int(summary.get("trainable_noise", summary.get("noise", 0))),
            "ignored_neighbor_events": int(summary.get("ignored_neighbor_events", 0)),
            "ignored_total_events": ignored_total,
            "high_snr_ignored_candidates": int(summary.get("high_snr_ignored_candidates", 0)),
            "low_snr_ignored_candidates": int(summary.get("low_snr_ignored_candidates", 0)),
            "missed_gt_ignored_candidates": int(summary.get("missed_gt_ignored_candidates", 0)),
            "missed_gt_count": int(summary.get("missed_gt_count", 0)),
            "neighbor_primary": int(summary.get("neighbor_primary", 0)),
            "near_gt_negative": int(summary.get("near_gt_negative", 0)),
            "duplicate_negative": int(summary.get("duplicate_negative", 0)),
            "gt_channel_source": summary.get("gt_channel_source", ""),
            "drift_gt_channel_enabled": bool(summary.get("drift_gt_channel_enabled", False)),
            "drift_gt_channel_valid": bool(summary.get("drift_gt_channel_valid", False)),
            "drift_gt_channel_changed_count": int(summary.get("drift_gt_channel_changed_count", 0)),
            "gt_align_enabled": bool(summary.get("gt_align_enabled", False)),
            "gt_align_radius_ms": float(summary.get("gt_align_radius_ms", 0.0)),
            "gt_align_median_shift_samples": float(summary.get("gt_align_median_shift_samples", 0.0)),
            "gt_align_p95_abs_shift_samples": float(summary.get("gt_align_p95_abs_shift_samples", 0.0)),
            "gt_align_channel_changed": int(summary.get("gt_align_channel_changed", 0)),
            "unit_snr_min_positive": float(summary.get("unit_snr_min_positive", 0.0)),
            "unit_snr_filter_enabled": bool(summary.get("unit_snr_filter_enabled", False)),
            "high_snr_ignore_near_gt_ms": float(summary.get("high_snr_ignore_near_gt_ms", 0.0)),
            "high_snr_ignore_neighbor_hops": int(summary.get("high_snr_ignore_neighbor_hops", 0)),
            "missed_gt_ignore_enabled": bool(summary.get("missed_gt_ignore_enabled", False)),
            "missed_gt_ignore_near_gt_ms": float(summary.get("missed_gt_ignore_near_gt_ms", 0.0)),
            "missed_gt_ignore_neighbor_hops": int(summary.get("missed_gt_ignore_neighbor_hops", 0)),
            "require_drift_gt_channel_for_missed_ignore": bool(summary.get("require_drift_gt_channel_for_missed_ignore", False)),
            "missed_gt_drift_channel_valid": bool(summary.get("missed_gt_drift_channel_valid", False)),
            "gt_snr_high_unit_count": int(summary.get("gt_snr_high_unit_count", 0)),
            "gt_snr_low_unit_count": int(summary.get("gt_snr_low_unit_count", 0)),
            "gt_snr_missing_unit_count": int(summary.get("gt_snr_missing_unit_count", 0)),
            "gt_snr_high_event_count": int(summary.get("gt_snr_high_event_count", 0)),
            "gt_snr_low_event_count": int(summary.get("gt_snr_low_event_count", 0)),
            "gt_snr_missing_event_count": int(summary.get("gt_snr_missing_event_count", 0)),
        }
        out["checks"].append(check)
        out["ok"] = bool(out["ok"] and ok_shape and labels_accounted)
    if dry_run:
        has_single = any("single" in str(s.get("job", "")).lower() or int(s.get("n_channels", 0)) == 1 for s in summaries)
        has_multi = any(int(s.get("n_channels", 0)) > 1 for s in summaries)
        out["dry_run_single_present"] = bool(has_single)
        out["dry_run_multi_present"] = bool(has_multi)
    return out


def validate_candidate_files(jobs: Sequence[GateJob], *, output_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "checks": []}
    for job in jobs:
        path = candidate_path(output_root, job)
        with np.load(path, allow_pickle=False) as arr:
            spike = np.asarray(arr["spike_label"], dtype=np.float32) > 0.5
            matched = np.asarray(arr["matched_gt_index"], dtype=np.int64)
            positive_gt = matched[spike & (matched >= 0)]
            unique, counts = np.unique(positive_gt, return_counts=True) if positive_gt.size else (np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64))
            max_positive_per_gt = int(np.max(counts)) if counts.size else 0
            duplicate_negative = int(np.sum(np.asarray(arr["duplicate_negative"], dtype=np.bool_))) if "duplicate_negative" in arr else 0
            near_gt_negative = int(np.sum(np.asarray(arr["near_gt_negative"], dtype=np.bool_))) if "near_gt_negative" in arr else 0
            ignored_neighbor_events = int(np.sum(np.asarray(arr["ignored_neighbor_event"], dtype=np.bool_))) if "ignored_neighbor_event" in arr else 0
            high_snr_ignored_candidates = int(np.sum(np.asarray(arr["high_snr_near_gt"], dtype=np.bool_))) if "high_snr_near_gt" in arr else 0
            low_snr_ignored_candidates = int(np.sum(np.asarray(arr["low_snr_near_gt"], dtype=np.bool_))) if "low_snr_near_gt" in arr else 0
            missed_gt_ignored_candidates = int(np.sum(np.asarray(arr["missed_gt_near_gt"], dtype=np.bool_))) if "missed_gt_near_gt" in arr else 0
            missed_gt_count = int(arr["missed_gt_count"][0]) if "missed_gt_count" in arr and arr["missed_gt_count"].size else 0
            neighbor_primary = int(np.sum(np.asarray(arr["neighbor_primary"], dtype=np.bool_))) if "neighbor_primary" in arr else 0
            train_mask = np.asarray(arr["training_mask"], dtype=np.bool_) if "training_mask" in arr else np.ones(spike.shape, dtype=np.bool_)
            gt_align_enabled = bool(arr["gt_align_enabled"][0]) if "gt_align_enabled" in arr and arr["gt_align_enabled"].size else False
            gt_align_median_shift = float(arr["gt_align_median_shift_samples"][0]) if "gt_align_median_shift_samples" in arr and arr["gt_align_median_shift_samples"].size else 0.0
            unit_snr_min_positive = float(arr["unit_snr_min_positive"][0]) if "unit_snr_min_positive" in arr and arr["unit_snr_min_positive"].size else 0.0
            high_snr_ignore_near_gt_ms = float(arr["high_snr_ignore_near_gt_ms"][0]) if "high_snr_ignore_near_gt_ms" in arr and arr["high_snr_ignore_near_gt_ms"].size else 0.0
            high_snr_ignore_neighbor_hops = int(arr["high_snr_ignore_neighbor_hops"][0]) if "high_snr_ignore_neighbor_hops" in arr and arr["high_snr_ignore_neighbor_hops"].size else 0
            missed_gt_ignore_enabled = bool(arr["missed_gt_ignore_enabled"][0]) if "missed_gt_ignore_enabled" in arr and arr["missed_gt_ignore_enabled"].size else False
            missed_gt_ignore_near_gt_ms = float(arr["missed_gt_ignore_near_gt_ms"][0]) if "missed_gt_ignore_near_gt_ms" in arr and arr["missed_gt_ignore_near_gt_ms"].size else 0.0
            missed_gt_ignore_neighbor_hops = int(arr["missed_gt_ignore_neighbor_hops"][0]) if "missed_gt_ignore_neighbor_hops" in arr and arr["missed_gt_ignore_neighbor_hops"].size else 0
            require_drift_gt_channel_for_missed_ignore = bool(arr["require_drift_gt_channel_for_missed_ignore"][0]) if "require_drift_gt_channel_for_missed_ignore" in arr and arr["require_drift_gt_channel_for_missed_ignore"].size else False
            missed_gt_drift_channel_valid = bool(arr["missed_gt_drift_channel_valid"][0]) if "missed_gt_drift_channel_valid" in arr and arr["missed_gt_drift_channel_valid"].size else False
            drift_gt_channel_enabled_arr = bool(arr["drift_gt_channel_enabled"][0]) if "drift_gt_channel_enabled" in arr and arr["drift_gt_channel_enabled"].size else False
            drift_gt_channel_valid_arr = bool(arr["drift_gt_channel_valid"][0]) if "drift_gt_channel_valid" in arr and arr["drift_gt_channel_valid"].size else False
            drift_gt_channel_changed_count_arr = int(arr["drift_gt_channel_changed_count"][0]) if "drift_gt_channel_changed_count" in arr and arr["drift_gt_channel_changed_count"].size else 0
        check = {
            "job": job.name,
            "max_two_positive_per_gt_ok": bool(max_positive_per_gt <= 2),
            "max_positive_per_gt": max_positive_per_gt,
            "near_gt_negative": near_gt_negative,
            "duplicate_negative": duplicate_negative,
            "ignored_neighbor_events": ignored_neighbor_events,
            "high_snr_ignored_candidates": high_snr_ignored_candidates,
            "low_snr_ignored_candidates": low_snr_ignored_candidates,
            "missed_gt_ignored_candidates": missed_gt_ignored_candidates,
            "missed_gt_count": missed_gt_count,
            "neighbor_primary": neighbor_primary,
            "trainable_spikes": int(np.sum(spike & train_mask)),
            "trainable_noise": int(np.sum((~spike) & train_mask)),
            "gt_align_enabled": gt_align_enabled,
            "gt_align_median_shift_samples": gt_align_median_shift,
            "unit_snr_min_positive": unit_snr_min_positive,
            "high_snr_ignore_near_gt_ms": high_snr_ignore_near_gt_ms,
            "high_snr_ignore_neighbor_hops": high_snr_ignore_neighbor_hops,
            "missed_gt_ignore_enabled": missed_gt_ignore_enabled,
            "missed_gt_ignore_near_gt_ms": missed_gt_ignore_near_gt_ms,
            "missed_gt_ignore_neighbor_hops": missed_gt_ignore_neighbor_hops,
            "require_drift_gt_channel_for_missed_ignore": require_drift_gt_channel_for_missed_ignore,
            "missed_gt_drift_channel_valid": missed_gt_drift_channel_valid,
            "drift_gt_channel_enabled": drift_gt_channel_enabled_arr,
            "drift_gt_channel_valid": drift_gt_channel_valid_arr,
            "drift_gt_channel_changed_count": drift_gt_channel_changed_count_arr,
        }
        out["checks"].append(check)
        out["ok"] = bool(out["ok"] and check["max_two_positive_per_gt_ok"])
    return out


def build_acceptance_report(metrics: Dict[str, Any], calibration: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    acc_cfg = dict(cfg.get("acceptance", {}) or {})
    recordings = dict(metrics.get("recordings", {}) or {})
    report: Dict[str, Any] = {"old_reference": {"s20n10_precision": float(acc_cfg.get("s20n10_old_precision", 0.396))}}
    qdiff = recordings.get("qdiff")
    if qdiff:
        cal = dict(qdiff.get("calibrated", {}) or {})
        report["qdiff"] = {
            "threshold": float(cal.get("threshold", float("nan"))),
            "precision": float(cal.get("precision", float("nan"))),
            "recall": float(cal.get("recall", float("nan"))),
            "passes": bool(
                float(cal.get("precision", 0.0)) >= float(acc_cfg.get("qdiff_min_precision", 0.98))
                and float(cal.get("recall", 0.0)) >= float(acc_cfg.get("qdiff_min_recall", 0.85))
            ),
        }
    s20n10 = recordings.get("s20n10")
    if s20n10:
        cal = dict(s20n10.get("calibrated", {}) or {})
        report["s20n10"] = {
            "threshold": float(cal.get("threshold", float("nan"))),
            "precision": float(cal.get("precision", float("nan"))),
            "recall": float(cal.get("recall", float("nan"))),
            "precision_delta_vs_old": float(cal.get("precision", float("nan"))) - float(acc_cfg.get("s20n10_old_precision", 0.396)),
            "passes_recall_floor": bool(float(cal.get("recall", 0.0)) >= float(acc_cfg.get("s20n10_min_recall", 0.80))),
        }
    report["calibrated_thresholds"] = {
        domain: float(entry.get("threshold", float("nan")))
        for domain, entry in calibration.items()
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an independent operational Msort gate from detector candidates.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--rebuild-candidates", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-seconds-per-recording", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="Override train.epochs for smoke tests.")
    ap.add_argument("--steps-per-epoch", type=int, default=None, help="Override train.steps_per_epoch for smoke tests.")
    ap.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size for smoke tests.")
    ap.add_argument("--output-root", default=None, help="Override output.root and derived files for isolated smoke tests.")
    args = ap.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    if args.epochs is not None or args.steps_per_epoch is not None or args.batch_size is not None:
        cfg["train"] = dict(cfg.get("train", {}) or {})
        if args.epochs is not None:
            cfg["train"]["epochs"] = int(args.epochs)
        if args.steps_per_epoch is not None:
            cfg["train"]["steps_per_epoch"] = int(args.steps_per_epoch)
        if args.batch_size is not None:
            cfg["train"]["batch_size"] = int(args.batch_size)
    if args.output_root:
        root_override = str(resolve_path(args.output_root))
        cfg["output"] = dict(cfg.get("output", {}) or {})
        cfg["output"]["root"] = root_override
        cfg["output"]["checkpoint"] = str(Path(root_override) / "gate_alpha4_peakonly.pt")
        cfg["output"]["metrics_json"] = str(Path(root_override) / "metrics.json")
        cfg["output"]["threshold_curves_csv"] = str(Path(root_override) / "threshold_curves.csv")
        cfg["output"]["log_file"] = str(Path(root_override) / "pretrain_gate.log")
    output_cfg = dict(cfg.get("output", {}) or {})
    output_root = resolve_path(output_cfg.get("root", "data/cache/gate_operational_alpha4_peakonly"))
    logger = build_logger(resolve_path(output_cfg.get("log_file", output_root / "pretrain_gate.log")))
    logger.info("argv=%s", " ".join(sys.argv))
    logger.info("cwd=%s", os.getcwd())
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(args.device or cfg.get("device", "cuda"))
    logger.info("device=%s", device)

    jobs = build_jobs(cfg)
    validate_eval_jobs(jobs)
    if args.dry_run:
        jobs = [job for job in jobs if job.split in ("val", "test")]
        selected: List[GateJob] = []
        seen_domain = set()
        for job in jobs:
            reader = MEArecH5Reader(job.input_h5)
            _, n_channels = reader.get_recordings_shape()
            domain = "single" if int(n_channels) == 1 else "multi"
            if domain not in seen_domain:
                selected.append(job)
                seen_domain.add(domain)
            if len(seen_domain) == 2:
                break
        jobs = selected
        logger.info("dry_run jobs=%s", ",".join(job.name for job in jobs))
    data_cfg = dict(cfg.get("data", {}) or {})
    max_seconds = float(
        args.max_seconds_per_recording
        if args.max_seconds_per_recording is not None
        else (5.0 if args.dry_run else float(data_cfg.get("max_seconds_per_recording", 0.0)))
    )
    rebuild = bool(args.rebuild_candidates or args.dry_run or data_cfg.get("rebuild_candidates", False))

    candidate_summaries: List[Dict[str, Any]] = []
    if not args.train_only:
        candidate_summaries = ensure_candidates(
            jobs,
            cfg=cfg,
            output_root=output_root,
            rebuild=rebuild,
            max_seconds=max_seconds,
            logger=logger,
        )
        sanity = run_sanity_checks(candidate_summaries, dry_run=bool(args.dry_run), cfg=cfg)
        file_sanity = validate_candidate_files(jobs, output_root=output_root)
        sanity["candidate_file_checks"] = file_sanity
        sanity["ok"] = bool(sanity["ok"] and file_sanity["ok"])
        logger.info("sanity=%s", json.dumps(jsonable(sanity), ensure_ascii=True))
        if args.prepare_only or args.dry_run:
            metrics_path = resolve_path(output_cfg.get("metrics_json", output_root / "metrics.json"))
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(jsonable({"candidate_summaries": candidate_summaries, "sanity": sanity}), f, indent=2)
            return

    all_jobs = build_jobs(cfg)
    validate_eval_jobs(all_jobs)
    train_jobs = [job for job in all_jobs if job.split == "train"]
    val_jobs = [job for job in all_jobs if job.split == "val"]
    test_jobs = [job for job in all_jobs if job.split == "test"]
    train_paths = [candidate_path(output_root, job) for job in train_jobs]
    missing = [str(path) for path in train_paths + [candidate_path(output_root, job) for job in val_jobs + test_jobs] if not path.exists()]
    if missing:
        raise FileNotFoundError("candidate caches missing; run without --train-only first: " + ", ".join(missing[:5]))
    train_arr = load_split_arrays(train_paths)
    if train_arr["patches"].shape[0] == 0:
        raise RuntimeError("no train candidates generated")

    model_cfg = build_model_config(dict(cfg.get("model", {}) or {}), embed_dim=int(cfg.get("model", {}).get("embed_dim", 128)))
    model = SpikeEmbeddingModel(model_cfg)
    train_summary = train_gate(model, train_arr, val_jobs, cfg=cfg, output_root=output_root, device=device, logger=logger)

    thresholds = [float(x) for x in cfg.get("calibration", {}).get("thresholds", [0.5])]
    calibration = calibrate_thresholds(
        model,
        val_jobs,
        output_root=output_root,
        device=device,
        batch_size=int(cfg.get("train", {}).get("batch_size", 512)),
        thresholds=thresholds,
        cfg=cfg,
    )
    calibrated_thresholds = {
        domain: float(entry["threshold"])
        for domain, entry in calibration.items()
    }
    val_metrics, val_curve_rows = evaluate_model(
        model,
        val_jobs,
        output_root=output_root,
        device=device,
        batch_size=int(cfg.get("train", {}).get("batch_size", 512)),
        thresholds=thresholds,
        calibrated_thresholds=calibrated_thresholds,
    )
    test_metrics, test_curve_rows = evaluate_model(
        model,
        test_jobs,
        output_root=output_root,
        device=device,
        batch_size=int(cfg.get("train", {}).get("batch_size", 512)),
        thresholds=thresholds,
        calibrated_thresholds=calibrated_thresholds,
    )
    eval_all_metrics, _ = evaluate_model(
        model,
        val_jobs + test_jobs,
        output_root=output_root,
        device=device,
        batch_size=int(cfg.get("train", {}).get("batch_size", 512)),
        thresholds=thresholds,
        calibrated_thresholds=calibrated_thresholds,
    )
    curve_rows = val_curve_rows + test_curve_rows
    evaluation_summary = build_evaluation_summary(
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        eval_all_metrics=eval_all_metrics,
        train_summary=train_summary,
        calibration=calibration,
        cfg=cfg,
    )
    curves_path = resolve_path(output_cfg.get("threshold_curves_csv", output_root / "threshold_curves.csv"))
    evaluation_summary["threshold_curves"] = {"csv": str(curves_path), "rows": curve_rows}
    log_evaluation_summary(logger, evaluation_summary)
    checkpoint_path = resolve_path(output_cfg.get("checkpoint", output_root / "gate_alpha4_peakonly.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": dict(cfg.get("model", {}) or {}),
            "calibration": calibration,
            "train_summary": train_summary,
            "evaluation_summary": evaluation_summary,
            "experiment": str(cfg.get("experiment", "gate_operational_alpha4")),
        },
        checkpoint_path,
    )
    write_threshold_curves(curves_path, curve_rows)
    metrics_path = resolve_path(output_cfg.get("metrics_json", output_root / "metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(evaluation_summary), f, indent=2)
    logger.info("saved checkpoint=%s metrics=%s curves=%s", checkpoint_path, metrics_path, curves_path)


if __name__ == "__main__":
    main()
