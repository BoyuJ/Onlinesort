#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_offline as ro
from src.sort.types import InitializedUnit
from src.sort.online_assign import (
    PrototypeAssignConfig,
    UnitPrototypeSet,
    assign_to_unit_prototypes,
    build_unit_prototype_set,
    update_nearest_unit_prototype,
)
from src.reproducibility import load_recording_manifest, resolve_data_records, sha256, environment, write_json
from src.utils import Event


METRIC_KEYS: tuple[str, ...] = (
    "acc_1to1_global",
    "acc_1to1_tp",
    "precision_1to1",
    "recall_1to1",
    "f1_1to1",
    "sort_f1_micro_mapped",
    "sort_f1_macro_mapped",
    "detect_precision",
    "detect_recall",
    "detect_f1",
)

DEFAULT_REUSE_COMPARISON_NAME = "infer_proto_chain_eval5_online20_embedding_compare"
DEFAULT_REUSE_SOURCE_DIRS: tuple[str, ...] = (
    "infer_proto_chain_eval5_boundary_chain",
    "infer_proto_chain_eval5_boundary",
    "infer_proto_chain_eval5",
)

HDBSCAN_NOGT_SELECTION_RETENTION = "no_gt_retention"
HDBSCAN_NOGT_SELECTION_HIGH_RECALL_K = "no_gt_high_recall_k"
HDBSCAN_ONLINE_PROFILE_EXP090_THRESHOLD_ONLY = "exp090_threshold_only"
HDBSCAN_ONLINE_PROFILE_EXP090_NOGT_K_STOP = "exp090_nogt_k_stop"
HDBSCAN_NOGT_MERGE_PROFILES: Dict[str, Dict[str, Any]] = {
    "strict": {
        "hdbscan_merge_cosine_min": 0.985,
        "hdbscan_merge_channel_gap_max": 1.0,
        "hdbscan_merge_cross_conflict_max": 0.005,
        "hdbscan_merge_refractory_ms": 0.30,
        "hdbscan_merge_max_rounds": 8,
    },
    "balanced": {
        "hdbscan_merge_cosine_min": 0.970,
        "hdbscan_merge_channel_gap_max": 2.0,
        "hdbscan_merge_cross_conflict_max": 0.010,
        "hdbscan_merge_refractory_ms": 0.30,
        "hdbscan_merge_max_rounds": 8,
    },
    "loose": {
        "hdbscan_merge_cosine_min": 0.950,
        "hdbscan_merge_channel_gap_max": 4.0,
        "hdbscan_merge_cross_conflict_max": 0.020,
        "hdbscan_merge_refractory_ms": 0.30,
        "hdbscan_merge_max_rounds": 8,
    },
}
HDBSCAN_ONLINE_PROFILES: Dict[str, Dict[str, Any]] = {
    HDBSCAN_ONLINE_PROFILE_EXP090_THRESHOLD_ONLY: {
        "hdbscan_selection_metric": HDBSCAN_NOGT_SELECTION_HIGH_RECALL_K,
        "hdbscan_high_recall_k_delta": 0.30,
        "hdbscan_high_recall_k_min_clustered_fraction": 0.25,
        "hdbscan_high_recall_k_max_small_unit_pressure": 1.0,
        "hdbscan_high_recall_k_max_noise_ratio": 0.80,
        "hdbscan_merge_enabled": True,
        "hdbscan_merge_profile": "loose",
        "hdbscan_merge_cosine_min": 0.90,
        "hdbscan_merge_channel_gap_max": 2.0,
        "hdbscan_merge_cross_conflict_max": 0.02,
        "nogt_k_stop_enabled": False,
    },
    HDBSCAN_ONLINE_PROFILE_EXP090_NOGT_K_STOP: {
        "hdbscan_selection_metric": HDBSCAN_NOGT_SELECTION_HIGH_RECALL_K,
        "hdbscan_high_recall_k_delta": 0.30,
        "hdbscan_high_recall_k_min_clustered_fraction": 0.25,
        "hdbscan_high_recall_k_max_small_unit_pressure": 1.0,
        "hdbscan_high_recall_k_max_noise_ratio": 0.80,
        "hdbscan_merge_enabled": True,
        "hdbscan_merge_profile": "loose",
        "hdbscan_merge_cosine_min": 0.90,
        "hdbscan_merge_channel_gap_max": 2.0,
        "hdbscan_merge_cross_conflict_max": 0.02,
        "nogt_k_stop_enabled": True,
        "nogt_k_stop_delta": 0.30,
        "nogt_k_stop_quantile": 0.90,
        "nogt_k_stop_min_premerge_fraction": 0.40,
    },
}


@dataclass(frozen=True)
class Case:
    name: str
    input_h5: Path | None
    input_bin: Path | None
    n_channels: int | None
    out_npz: Path
    log_file: Path


@dataclass(frozen=True)
class OnlineConfig:
    init_window_sec: tuple[float, ...]
    assign_mode: str
    max_prototypes_per_unit: int
    min_events_per_prototype: int
    update_enabled: bool
    update_conf_min: float
    update_alpha_mu: float
    update_alpha_stats: float
    assign_cfg: PrototypeAssignConfig


@dataclass
class InferenceContext:
    cfg: Dict[str, Any]
    offline_cfg: Dict[str, Any]
    offline_cluster_cfg: Dict[str, Any]
    online_cfg: OnlineConfig
    input_path: Path
    input_h5: bool
    reader: Any | None
    fs: float
    n_samples: int
    n_channels: int
    channel_positions: np.ndarray | None
    weights_path: Path | None
    weight_payload: Any
    pipeline: Any
    model_cfg: Any
    assignment_gate_thresh: float
    encode_gate_disabled: bool
    candidate_filtering_disabled: bool
    requested_k: int
    requested_k_source: str
    save_embeddings: bool


@dataclass
class BenchmarkContext:
    enabled: bool
    backend: str
    bench_cfg: Any
    bench_cfg_raw: Dict[str, Any]
    gt_events: Any | None
    gt_alignment_stats: Dict[str, Any]
    bench_fs: float
    tol_samples: int
    no_gt_enabled: bool
    no_gt_quality_kwargs: Dict[str, Any]


def build_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mysort.online")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def resolve_path(path_value: str | Path, root: Path = ROOT) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return root / p


def parse_float_list(value: Any, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        raw_values = list(default)
    elif isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        raw_values = list(value)
    else:
        raw_values = [value]
    out: list[float] = []
    for raw in raw_values:
        val = float(raw)
        if val <= 0.0:
            raise ValueError(f"init window must be > 0 seconds, got {raw!r}")
        out.append(val)
    if not out:
        raise ValueError("at least one init window is required")
    return tuple(out)


def init_window_suffix(sec: float) -> str:
    if abs(float(sec) - round(float(sec))) < 1e-6:
        return f"{int(round(float(sec)))}s"
    return f"{float(sec):.3f}".rstrip("0").rstrip(".").replace(".", "p") + "s"


def online_output_path(base_out: Path, init_window_sec: float) -> Path:
    suffix = init_window_suffix(init_window_sec)
    return base_out.with_name(f"{base_out.stem}_online_init{suffix}{base_out.suffix}")


def online_log_path(base_log: Path) -> Path:
    return base_log.with_name(f"{base_log.stem}_online{base_log.suffix}")


def report_path(out_npz: Path) -> Path:
    return ro.benchmark_report_path_for_output(out_npz)


def comparison_subdir(name: str) -> Path:
    raw = str(name).strip()
    if not raw:
        raise ValueError("comparison name must be non-empty")
    p = Path(raw)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise ValueError(f"comparison name must be a relative subdirectory, got {name!r}")
    return p


def resolve_source_npz_dir(path_value: str | Path) -> Path:
    raw = Path(path_value).expanduser()
    candidates = [raw] if raw.is_absolute() else [ROOT / raw, ROOT / "data" / "cache" / raw]
    for candidate in candidates:
        if candidate.exists():
            if not candidate.is_dir():
                raise NotADirectoryError(f"source npz path is not a directory: {candidate}")
            return candidate.resolve()
    raise FileNotFoundError(
        "source npz directory not found; tried: " + ", ".join(str(candidate) for candidate in candidates)
    )


def parse_source_npz_dirs(raw_values: Sequence[str] | None) -> list[Path]:
    values: list[str] = []
    source_values = list(raw_values or DEFAULT_REUSE_SOURCE_DIRS)
    for raw in source_values:
        values.extend(item.strip() for item in str(raw).split(",") if item.strip())
    if not values:
        raise ValueError("at least one source npz directory is required")
    out: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = resolve_source_npz_dir(value)
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arr:
        return {key: np.asarray(arr[key]) for key in arr.files}


def validate_reuse_embedding_arrays(arr: Dict[str, np.ndarray], *, path: Path) -> None:
    required = (
        "embedding",
        "event_time_sample",
        "center_channel",
        "gate_prob",
        "encoded_mask",
        "confidence",
    )
    missing = [key for key in required if key not in arr]
    if missing:
        raise KeyError(f"{path} missing required arrays for reuse mode: {', '.join(missing)}")
    n = int(np.asarray(arr["event_time_sample"]).shape[0])
    for key in required:
        value = np.asarray(arr[key])
        if key == "embedding":
            if value.ndim != 2:
                raise ValueError(f"{path} embedding must be 2D, got {value.shape}")
            if int(value.shape[0]) != n:
                raise ValueError(f"{path} embedding/event length mismatch: {value.shape[0]} != {n}")
        elif value.ndim == 0 or int(value.shape[0]) != n:
            raise ValueError(f"{path} {key} length mismatch: expected {n}, got {value.shape}")
    if n == 0:
        raise ValueError(f"{path} contains no events")


def force_reuse_online_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    online_raw = dict(out.get("online", {}) or {})
    online_raw["assign_mode"] = "multi_prototype"
    out["online"] = online_raw
    return out


def source_benchmark_report_path(source_npz: Path) -> Path:
    return source_npz.with_name(f"{source_npz.stem}_benchmark_report.json")


def read_source_gate_threshold(source_npz: Path) -> float | None:
    report = source_benchmark_report_path(source_npz)
    if not report.exists():
        return None
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return None
    gate_summary = payload.get("gate_summary", {}) if isinstance(payload, dict) else {}
    return finite_or_none(gate_summary.get("gate_threshold")) if isinstance(gate_summary, dict) else None


def resolve_reuse_gate_thresholds(
    cfg: Dict[str, Any],
    *,
    source_npz: Path,
    logger: logging.Logger | None = None,
) -> tuple[float, float]:
    model = dict(cfg.get("model", {}) or {})
    source_assignment = read_source_gate_threshold(source_npz)
    default_assignment = float(model.get("assignment_gate_thresh", model.get("fixed_gate_threshold", 0.5)))
    assignment_gate_thresh = default_assignment if source_assignment is None else float(source_assignment)
    encode_gate_thresh = float(model.get("encode_gate_thresh", min(default_assignment, assignment_gate_thresh)))
    threshold_values, threshold_notes = ro.normalize_runtime_gate_thresholds(
        encode_gate_thresh=float(encode_gate_thresh),
        assignment_gate_thresh=float(assignment_gate_thresh),
        novelty_birth_gate_min=float(dict(cfg.get("clustering", {}) or {}).get("novelty_birth_gate_min", 0.10)),
        keep_gate_min=float(dict(cfg.get("clustering", {}) or {}).get("keep_gate_min", 0.10)),
        init_seed_gate=float(dict(cfg.get("clustering", {}) or {}).get("init_seed_gate", 0.10)),
    )
    if logger is not None:
        if source_assignment is not None:
            logger.info(
                "reuse_gate_threshold source_report=%s assignment_gate_thresh=%.4f",
                source_benchmark_report_path(source_npz),
                float(source_assignment),
            )
        for note in threshold_notes:
            logger.info("%s", note)
    return float(threshold_values["encode_gate_thresh"]), float(threshold_values["assignment_gate_thresh"])


def resolve_online_config(cfg: Dict[str, Any], init_override: str | None) -> OnlineConfig:
    raw = dict(cfg.get("online", {}) or {})
    cluster_raw = dict(cfg.get("clustering", {}) or {})
    default_assign = PrototypeAssignConfig()
    assign_mode = str(raw.get("assign_mode", "multi_prototype")).strip().lower()
    if assign_mode in ("multi", "multi_prototype", "prototype", "prototypes"):
        assign_mode = "multi_prototype"
    else:
        raise ValueError(f"unsupported online.assign_mode={assign_mode!r}; use multi_prototype")
    windows = parse_float_list(
        init_override if init_override is not None else raw.get("init_window_sec"),
        default=(20.0, 30.0),
    )
    default_update_conf = 0.50
    default_update_alpha_mu = 0.05
    default_max_distance = 0.45
    default_margin_min = 0.01
    return OnlineConfig(
        init_window_sec=windows,
        assign_mode=assign_mode,
        max_prototypes_per_unit=int(raw.get("max_prototypes_per_unit", 5)),
        min_events_per_prototype=int(raw.get("min_events_per_prototype", 8)),
        update_enabled=bool(raw.get("update_enabled", True)),
        update_conf_min=float(raw.get("update_conf_min", default_update_conf)),
        update_alpha_mu=float(raw.get("update_alpha_mu", default_update_alpha_mu)),
        update_alpha_stats=float(raw.get("update_alpha_stats", cluster_raw.get("update_alpha_stats", 0.05))),
        assign_cfg=PrototypeAssignConfig(
            max_center_distance=float(raw.get("max_center_distance", default_max_distance)),
            margin_min=float(raw.get("margin_min", default_margin_min)),
            candidate_hop=int(raw.get("candidate_hop", default_assign.candidate_hop)),
            confidence_temperature=float(
                raw.get("confidence_temperature", default_assign.confidence_temperature)
            ),
        ),
    )


def apply_hdbscan_online_profile(raw: Dict[str, Any], logger: logging.Logger | None = None) -> Dict[str, Any]:
    cfg = dict(raw)
    raw_profile = str(
        cfg.get("hdbscan_online_profile", cfg.get("hdbscan_profile", cfg.get("online_profile", "")))
    ).strip().lower()
    if raw_profile in ("", "none", "default", "config"):
        return cfg
    aliases = {
        "threshold_only": HDBSCAN_ONLINE_PROFILE_EXP090_THRESHOLD_ONLY,
        "nogt_k_stop": HDBSCAN_ONLINE_PROFILE_EXP090_NOGT_K_STOP,
    }
    profile = aliases.get(raw_profile, raw_profile)
    if profile not in HDBSCAN_ONLINE_PROFILES:
        raise ValueError(
            "unsupported hdbscan_online_profile="
            f"{raw_profile!r}; expected one of {', '.join(sorted(HDBSCAN_ONLINE_PROFILES))}"
        )
    for key, value in HDBSCAN_ONLINE_PROFILES[profile].items():
        if cfg.get(key) is None:
            cfg[key] = value
        elif key not in cfg:
            cfg[key] = value
    cfg["hdbscan_online_profile"] = profile
    if logger is not None:
        logger.info("hdbscan_online_profile=%s", profile)
    return cfg


def restrict_to_hdbscan(raw: Dict[str, Any], logger: logging.Logger | None = None) -> Dict[str, Any]:
    cfg = dict(raw)
    raw_methods = list(cfg.get("methods", [cfg.get("method", "hdbscan")]))
    ignored = [str(value) for value in raw_methods if str(value).strip().lower() != "hdbscan"]
    raw_selection_method = str(cfg.get("hdbscan_cluster_selection_method", "leaf")).strip().lower() or "leaf"
    cfg["method"] = "hdbscan"
    cfg["methods"] = ["hdbscan"]
    cfg["hdbscan_cluster_selection_method"] = "leaf"
    if "required_methods" in cfg:
        cfg["required_methods"] = [
            value for value in cfg.get("required_methods", []) if str(value).strip().lower() == "hdbscan"
        ]
    if ignored and logger is not None:
        logger.info("online_clustering_methods restricted_to=hdbscan ignored_methods=%s", ",".join(ignored))
    if raw_selection_method != "leaf" and logger is not None:
        logger.info("online_hdbscan_selection_method restricted_to=leaf ignored=%s", raw_selection_method)
    cfg = apply_hdbscan_online_profile(cfg, logger)
    return cfg


def resolve_online_offline_cluster_config(offline_cfg: Dict[str, Any], logger: logging.Logger | None = None) -> Dict[str, Any]:
    resolved = dict(ro.resolve_offline_clustering_config(offline_cfg))
    raw_cluster = dict(offline_cfg.get("clustering", {}) or {})
    preserve_keys = {
        "hdbscan_online_profile",
        "hdbscan_profile",
        "online_profile",
        "nogt_k_stop_enabled",
        "nogt_k_stop_delta",
        "nogt_k_stop_quantile",
        "nogt_k_stop_min_premerge_fraction",
        "nogt_k_stop_min_clustered_fraction",
        "nogt_k_stop_max_small_unit_pressure",
        "nogt_k_stop_max_noise_ratio",
    }
    for key, value in raw_cluster.items():
        if key.startswith("hdbscan_") or key in preserve_keys:
            resolved[key] = value
    return restrict_to_hdbscan(resolved, logger)


def resolve_requested_k_no_gt(cfg: Dict[str, Any], offline_cluster_cfg: Dict[str, Any]) -> tuple[int, str]:
    online_raw = dict(cfg.get("online", {}) or {})
    for raw in (online_raw.get("n_units"), offline_cluster_cfg.get("n_units")):
        if raw is None:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError(f"configured n_units must be positive, got {raw!r}")
        return value, "config"
    return 0, "no_gt"


def select_job_names(raw: str, available: Sequence[str]) -> list[str]:
    if raw.strip().lower() in ("", "all"):
        return list(available)
    wanted = [item.strip() for item in raw.split(",") if item.strip()]
    available_set = set(available)
    missing = sorted(set(wanted) - available_set)
    if missing:
        raise ValueError(f"unknown jobs: {', '.join(missing)}")
    return wanted


def load_batch_cases(
    cfg: Dict[str, Any],
    jobs_arg: str,
    *,
    output_dir_override: str | None = None,
    log_dir_override: str | None = None,
) -> list[Case]:
    batch_cfg = dict(cfg.get("batch", {}) or {})
    jobs = batch_cfg.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("config has no batch.jobs")
    output_dir = output_dir_override if output_dir_override is not None else batch_cfg.get("output_dir")
    log_dir = log_dir_override if log_dir_override is not None else batch_cfg.get("log_dir")
    output_dir_path = resolve_path(output_dir) if output_dir else None
    log_dir_path = resolve_path(log_dir) if log_dir else None

    by_name: dict[str, dict[str, Any]] = {}
    for idx, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"batch.jobs[{idx}] must be a mapping")
        name = str(job.get("name", "")).strip()
        if not name:
            raise ValueError(f"batch.jobs[{idx}] missing name")
        if name in by_name:
            raise ValueError(f"duplicate batch job name: {name}")
        by_name[name] = job

    selected = select_job_names(jobs_arg, list(by_name.keys()))
    job_name_regex = str(batch_cfg.get("job_name_regex", "") or "").strip()
    if job_name_regex:
        pattern = re.compile(job_name_regex)
        selected = [name for name in selected if pattern.search(name)]
        if not selected:
            raise ValueError(f"batch.job_name_regex={job_name_regex!r} matched no selected jobs")
    cases: list[Case] = []
    for name in selected:
        job = by_name[name]
        input_h5 = resolve_path(job["input_h5"]) if job.get("input_h5") else None
        input_bin = resolve_path(job["input_bin"]) if job.get("input_bin") else None
        if input_h5 is None and input_bin is None:
            raise ValueError(f"batch job {name!r} must set input_h5 or input_bin")
        out_base = (
            output_dir_path / f"{name}.npz"
            if output_dir_path is not None
            else resolve_path(job.get("out_npz", f"data/cache/{name}.npz"))
        )
        log_base = (
            log_dir_path / f"{name}.log"
            if log_dir_path is not None
            else resolve_path(job.get("log_file", f"data/logs/{name}.log"))
        )
        cases.append(
            Case(
                name=name,
                input_h5=input_h5,
                input_bin=input_bin,
                n_channels=int(job["n_channels"]) if job.get("n_channels") is not None else None,
                out_npz=out_base,
                log_file=online_log_path(log_base),
            )
        )
    return cases


def single_case_from_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> Case:
    offline_cfg = dict(cfg.get("offline", {}) or {})
    input_h5 = args.input_h5 or offline_cfg.get("input_h5")
    input_bin = args.input_bin or offline_cfg.get("input_bin")
    default_out = ROOT / "data" / "cache" / "online_outputs.npz"
    default_log = ROOT / "data" / "logs" / "online_infer.log"
    out_npz = args.out_npz or offline_cfg.get("out_npz", str(default_out))
    log_file = args.log_file or offline_cfg.get("log_file", str(default_log))
    n_channels = args.n_channels if args.n_channels is not None else offline_cfg.get("n_channels")
    return Case(
        name="single",
        input_h5=resolve_path(input_h5) if input_h5 else None,
        input_bin=resolve_path(input_bin) if input_bin else None,
        n_channels=int(n_channels) if n_channels is not None else None,
        out_npz=resolve_path(out_npz),
        log_file=resolve_path(log_file),
    )


def finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32, copy=False)


def stable_center(z: np.ndarray, trim_quantile: float = 0.90) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        width = int(arr.shape[1]) if arr.ndim == 2 else 0
        return np.zeros((width,), dtype=np.float32)
    center = normalize_vector(np.mean(arr, axis=0))
    if arr.shape[0] <= 2:
        return center
    dist = 1.0 - np.clip(arr @ center, -1.0, 1.0)
    cutoff = float(np.quantile(dist, float(np.clip(trim_quantile, 0.0, 1.0))))
    keep = dist <= cutoff
    if int(np.sum(keep)) < max(2, int(math.ceil(0.5 * arr.shape[0]))):
        keep = np.ones((arr.shape[0],), dtype=np.bool_)
    return normalize_vector(np.mean(arr[keep], axis=0))


def dominant_channel(channels: np.ndarray) -> int:
    arr = np.asarray(channels, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        return 0
    values, counts = np.unique(arr, return_counts=True)
    return int(values[int(np.argmax(counts))])


def subset_arr_by_mask(arr: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    n = int(np.asarray(arr["event_time_sample"]).shape[0])
    keep = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if keep.size != n:
        raise ValueError(f"mask length mismatch: expected {n}, got {keep.size}")
    out: Dict[str, np.ndarray] = {}
    for key, value in arr.items():
        val = np.asarray(value)
        out[key] = val[keep].copy() if val.ndim > 0 and val.shape[0] == n else val.copy()
    return out


def subset_arr_by_indices(arr: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    n = int(np.asarray(arr["event_time_sample"]).shape[0])
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    out: Dict[str, np.ndarray] = {}
    for key, value in arr.items():
        val = np.asarray(value)
        out[key] = val[idx].copy() if val.ndim > 0 and val.shape[0] == n else val.copy()
    return out


def event_for_index(arr: Dict[str, np.ndarray], idx: int) -> Event:
    c0 = int(np.asarray(arr["center_channel"])[int(idx)])
    if "neighbor_chs" in arr:
        neighbors = np.asarray(arr["neighbor_chs"][int(idx)], dtype=np.int32).reshape(-1)
    else:
        neighbors = np.asarray([c0], dtype=np.int32)
    confidence = float(np.asarray(arr.get("confidence", np.zeros((len(arr["event_time_sample"]),))))[int(idx)])
    gate_prob = float(np.asarray(arr.get("gate_prob", np.ones((len(arr["event_time_sample"]),))))[int(idx)])
    overlap_prob = float(
        np.asarray(
            arr.get(
                "overlap_prob",
                arr.get("overlap_prob_unavailable", np.full((len(arr["event_time_sample"]),), np.nan)),
            )
        )[int(idx)]
    )
    encoded = bool(np.asarray(arr.get("encoded_mask", np.ones((len(arr["event_time_sample"]),))))[int(idx)] > 0)
    return Event(
        t0=int(np.asarray(arr["event_time_sample"])[int(idx)]),
        c0=c0,
        amp_norm=confidence,
        spatial_score=confidence,
        neighbors=neighbors,
        q=confidence,
        gate_prob=gate_prob,
        overlap_prob=overlap_prob,
        encoded_mask=encoded,
        qc_flags=int(np.asarray(arr.get("qc_flags", np.zeros((len(arr["event_time_sample"]),))))[int(idx)]),
    )


def events_for_indices(arr: Dict[str, np.ndarray], indices: np.ndarray) -> list[Event]:
    return [event_for_index(arr, int(idx)) for idx in np.asarray(indices, dtype=np.int64).tolist()]


def build_units_from_labels(
    arr: Dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    global_indices: np.ndarray | None = None,
    center_trim_quantile: float = 0.90,
) -> Dict[int, InitializedUnit]:
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    z = np.asarray(arr["embedding"], dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != labels_arr.size:
        raise ValueError(f"embedding/label mismatch: embedding={z.shape} labels={labels_arr.shape}")
    global_idx = (
        np.arange(labels_arr.size, dtype=np.int64)
        if global_indices is None
        else np.asarray(global_indices, dtype=np.int64).reshape(-1)
    )
    if global_idx.size != labels_arr.size:
        raise ValueError("global_indices length must match labels")

    times = np.asarray(arr["event_time_sample"], dtype=np.int64)
    channels = np.asarray(arr["center_channel"], dtype=np.int32)
    confidence = np.asarray(arr.get("confidence", np.zeros((labels_arr.size,), dtype=np.float32)), dtype=np.float32)
    units: Dict[int, InitializedUnit] = {}
    for unit_id in sorted(int(v) for v in np.unique(labels_arr).tolist() if int(v) > 0):
        local = np.flatnonzero(labels_arr == int(unit_id)).astype(np.int64, copy=False)
        if local.size == 0:
            continue
        local_channels = channels[local]
        local_amp = confidence[local]
        local_times = times[local]
        units[int(unit_id)] = InitializedUnit(
            unit_id=int(unit_id),
            center=stable_center(z[local], trim_quantile=float(center_trim_quantile)),
            dominant_channel=dominant_channel(local_channels),
            channel_median=float(np.median(local_channels)) if local_channels.size else 0.0,
            amp_mean=float(np.mean(local_amp)) if local_amp.size else 0.0,
            amp_var=float(np.var(local_amp)) if local_amp.size else 0.0,
            first_t0=int(np.min(local_times)) if local_times.size else 0,
            last_t0=int(np.max(local_times)) if local_times.size else 0,
            n_events=int(local.size),
            member_indices=global_idx[local].astype(np.int64, copy=False),
            source_communities=[int(unit_id)],
        )
    return units


def build_prototypes_from_units(
    arr: Dict[str, np.ndarray],
    units: Dict[int, InitializedUnit],
    *,
    max_prototypes_per_unit: int,
    min_events_per_prototype: int,
    center_trim_quantile: float = 0.90,
) -> Dict[int, UnitPrototypeSet]:
    z = np.asarray(arr["embedding"], dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"embedding must be 2D, got {z.shape}")
    prototypes: Dict[int, UnitPrototypeSet] = {}
    for unit_id, unit in sorted(units.items()):
        member_idx = np.asarray(unit.member_indices, dtype=np.int64).reshape(-1)
        member_idx = member_idx[(member_idx >= 0) & (member_idx < z.shape[0])]
        prototypes[int(unit_id)] = build_unit_prototype_set(
            unit,
            z[member_idx],
            max_prototypes=max(1, int(max_prototypes_per_unit)),
            min_events_per_prototype=max(1, int(min_events_per_prototype)),
            trim_quantile=float(center_trim_quantile),
        )
    return prototypes


def update_unit_center(unit: InitializedUnit, z: np.ndarray, event: Event, *, alpha_mu: float, alpha_stats: float) -> None:
    a_mu = float(np.clip(alpha_mu, 0.0, 1.0))
    a_stats = float(np.clip(alpha_stats, 0.0, 1.0))
    z_norm = normalize_vector(z)
    unit.center = normalize_vector((1.0 - a_mu) * unit.center + a_mu * z_norm)
    old_mean = float(unit.amp_mean)
    amp = float(event.amp_norm)
    new_mean = (1.0 - a_stats) * old_mean + a_stats * amp
    unit.amp_mean = float(new_mean)
    unit.amp_var = float((1.0 - a_stats) * float(unit.amp_var) + a_stats * float((amp - new_mean) ** 2))
    unit.last_t0 = max(int(unit.last_t0), int(event.t0))
    unit.n_events = int(unit.n_events) + 1


def clamp01(value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return float(np.clip(val, 0.0, 1.0))


def _metric_sample(z: np.ndarray, labels: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if z.shape[0] <= int(max_samples):
        return z, labels
    idx = np.linspace(0, z.shape[0] - 1, int(max_samples)).astype(np.int64)
    return z[idx], labels[idx]


def _hdbscan_nogt_metrics(
    z: np.ndarray,
    raw_labels: np.ndarray,
    clusterer: Any,
    *,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    labels = ro._relabel_positive(np.asarray(raw_labels, dtype=np.int32))
    pos = labels > 0
    n = int(labels.shape[0])
    clustered = int(np.sum(pos))
    actual = int(ro._positive_cluster_count(labels))
    noise = int(n - clustered)
    clustered_fraction = float(clustered / max(1, n))
    noise_ratio = float(noise / max(1, n))

    counts = (
        np.asarray([np.sum(labels == int(unit)) for unit in np.unique(labels[pos]).tolist()], dtype=np.float64)
        if actual > 0
        else np.zeros((0,), dtype=np.float64)
    )
    largest_unit_fraction = float(np.max(counts) / max(1, clustered)) if counts.size else 0.0
    small_min = int(cfg.get("hdbscan_nogt_small_unit_min_spikes", 40))
    small_counts = counts[counts < float(small_min)] if counts.size else np.zeros((0,), dtype=np.float64)
    small_unit_fraction = float(small_counts.size / max(1, counts.size)) if counts.size else 0.0
    small_unit_spike_fraction = float(np.sum(small_counts) / max(1, clustered)) if counts.size else 0.0
    small_unit_pressure = float(0.5 * small_unit_fraction + 0.5 * small_unit_spike_fraction)
    size_cv = float(np.std(counts) / max(float(np.mean(counts)), 1.0e-8)) if counts.size else 99.0

    silhouette = -1.0
    dbi = 999.0
    if actual >= 2 and clustered >= 3:
        try:
            from sklearn.metrics import davies_bouldin_score, silhouette_score

            z_sample, y_sample = _metric_sample(
                z[pos],
                labels[pos],
                int(cfg.get("hdbscan_nogt_metric_sample_size", 3000)),
            )
            if np.unique(y_sample).size >= 2 and y_sample.size > np.unique(y_sample).size:
                silhouette = float(silhouette_score(z_sample, y_sample, metric="euclidean"))
                dbi = float(davies_bouldin_score(z_sample, y_sample))
        except Exception:
            silhouette = -1.0
            dbi = 999.0

    persistence_raw = getattr(clusterer, "cluster_persistence_", np.zeros((0,), dtype=np.float32))
    persistence_arr = np.asarray(persistence_raw, dtype=np.float32).reshape(-1)
    persistence_mean = float(np.mean(persistence_arr)) if persistence_arr.size else 0.0

    q_silhouette = clamp01((float(silhouette) + 0.1) / 0.7)
    q_dbi = float(1.0 / (1.0 + max(float(dbi), 0.0) / 2.0)) if math.isfinite(float(dbi)) else 0.0
    q_clustered = clamp01(clustered_fraction)
    q_persistence = clamp01(persistence_mean)
    q_balance = clamp01(1.0 - max(0.0, largest_unit_fraction - 0.65) / 0.35)
    q_small = clamp01(1.0 - small_unit_pressure)

    return {
        "actual_clusters": int(actual),
        "clustered_events": int(clustered),
        "noise_events": int(noise),
        "clustered_fraction": float(clustered_fraction),
        "noise_ratio": float(noise_ratio),
        "silhouette": float(silhouette),
        "DBI": float(dbi),
        "cluster_persistence_mean": float(persistence_mean),
        "largest_unit_fraction": float(largest_unit_fraction),
        "small_unit_fraction": float(small_unit_fraction),
        "small_unit_spike_fraction": float(small_unit_spike_fraction),
        "small_unit_pressure": float(small_unit_pressure),
        "size_cv": float(size_cv),
        "q_silhouette": float(q_silhouette),
        "q_dbi": float(q_dbi),
        "q_clustered": float(q_clustered),
        "q_persistence": float(q_persistence),
        "q_balance": float(q_balance),
        "q_small": float(q_small),
    }


def _annotate_hdbscan_nogt_scores(candidates: List[Dict[str, Any]]) -> None:
    if not candidates:
        return
    counts: Dict[int, int] = {}
    for candidate in candidates:
        k = int(candidate.get("actual_clusters", 0))
        counts[k] = counts.get(k, 0) + 1
    max_freq = max(counts.values()) if counts else 1
    for candidate in candidates:
        k = int(candidate.get("actual_clusters", 0))
        stability = float(counts.get(k, 0) / max(1, max_freq))
        candidate["cluster_count_stability"] = float(stability)
        candidate["hdbscan_nogt_score"] = float(
            0.28 * float(candidate.get("q_silhouette", 0.0))
            + 0.20 * float(candidate.get("q_dbi", 0.0))
            + 0.18 * float(candidate.get("q_clustered", 0.0))
            + 0.12 * float(candidate.get("q_persistence", 0.0))
            + 0.10 * stability
            + 0.07 * float(candidate.get("q_balance", 0.0))
            + 0.05 * float(candidate.get("q_small", 0.0))
        )


def _resolve_hdbscan_nogt_selection_metric(cfg: Dict[str, Any]) -> str:
    raw = cfg.get("hdbscan_selection_metric", cfg.get("hdbscan_nogt_selection_metric"))
    if raw is None:
        generic = str(cfg.get("selection_metric", "")).strip().lower()
        raw = generic if generic in {HDBSCAN_NOGT_SELECTION_RETENTION, HDBSCAN_NOGT_SELECTION_HIGH_RECALL_K} else None
    metric = str(raw or HDBSCAN_NOGT_SELECTION_RETENTION).strip().lower()
    aliases = {
        "retention": HDBSCAN_NOGT_SELECTION_RETENTION,
        "default": HDBSCAN_NOGT_SELECTION_RETENTION,
        HDBSCAN_NOGT_SELECTION_RETENTION: HDBSCAN_NOGT_SELECTION_RETENTION,
        "high_k": HDBSCAN_NOGT_SELECTION_HIGH_RECALL_K,
        "high_recall_k": HDBSCAN_NOGT_SELECTION_HIGH_RECALL_K,
        HDBSCAN_NOGT_SELECTION_HIGH_RECALL_K: HDBSCAN_NOGT_SELECTION_HIGH_RECALL_K,
    }
    if metric not in aliases:
        raise ValueError(
            "offline.clustering.hdbscan_selection_metric must be "
            f"{HDBSCAN_NOGT_SELECTION_RETENTION!r} or {HDBSCAN_NOGT_SELECTION_HIGH_RECALL_K!r}"
        )
    return aliases[metric]


def _candidate_score(candidate: Dict[str, Any]) -> float:
    value = float(candidate.get("hdbscan_nogt_score", float("-inf")))
    return value if math.isfinite(value) else float("-inf")


def _candidate_k(candidate: Dict[str, Any]) -> int:
    return int(candidate.get("actual_clusters", 0))


def _candidate_k_summary(candidates: Sequence[Dict[str, Any]], *, prefix: str) -> Dict[str, Any]:
    count_key = "candidate_count" if prefix == "candidate" else f"{prefix}_candidate_count"
    values = sorted(_candidate_k(candidate) for candidate in candidates)
    if not values:
        return {
            count_key: 0,
            f"{prefix}_k_min": None,
            f"{prefix}_k_max": None,
            f"{prefix}_k_values": [],
            f"{prefix}_k_dominant": None,
            f"{prefix}_k_dominant_fraction": 0.0,
        }
    counts: Dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    dominant_k, dominant_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    return {
        count_key: int(len(values)),
        f"{prefix}_k_min": int(values[0]),
        f"{prefix}_k_max": int(values[-1]),
        f"{prefix}_k_values": sorted(int(value) for value in counts),
        f"{prefix}_k_dominant": int(dominant_k),
        f"{prefix}_k_dominant_fraction": float(dominant_count / max(1, len(values))),
    }


def _hdbscan_k_confidence(near: Sequence[Dict[str, Any]]) -> str:
    summary = _candidate_k_summary(near, prefix="near_best")
    values = list(summary["near_best_k_values"])
    if not values:
        return "unknown"
    if len(values) == 1:
        return "high"
    span = int(summary["near_best_k_max"]) - int(summary["near_best_k_min"])
    dominant_fraction = float(summary["near_best_k_dominant_fraction"])
    if span <= 1 and dominant_fraction >= 0.55:
        return "high"
    if span <= 3 or dominant_fraction >= 0.45:
        return "medium"
    return "low"


def _hdbscan_selection_diag(
    *,
    candidates: Sequence[Dict[str, Any]],
    near: Sequence[Dict[str, Any]],
    pool: Sequence[Dict[str, Any]],
    selected: Dict[str, Any],
    best_score: float,
    delta: float,
    metric: str,
    reason: str,
    constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    diag = {
        "hdbscan_selection_metric": str(metric),
        "selection_reason": str(reason),
        "near_best_score": float(best_score),
        "near_best_score_delta": float(delta),
        "selected_cluster_count": int(_candidate_k(selected)),
        "selected_candidate_index": int(selected.get("candidate_index", -1)),
        "selected_score": float(_candidate_score(selected)),
        "selected_clustered_fraction": float(selected.get("clustered_fraction", 0.0)),
        "selected_small_unit_pressure": float(selected.get("small_unit_pressure", 1.0)),
        "selected_noise_ratio": float(selected.get("noise_ratio", 1.0)),
        "candidate_count": int(len(candidates)),
        "selection_pool_candidate_count": int(len(pool)),
    }
    diag.update(_candidate_k_summary(candidates, prefix="candidate"))
    diag.update(_candidate_k_summary(near, prefix="near_best"))
    diag["k_confidence"] = _hdbscan_k_confidence(near)
    if constraints:
        diag["constraints"] = dict(constraints)
    return diag


def _choose_hdbscan_nogt_candidate(
    candidates: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    requested_k: int = 0,
    requested_k_source: str = "no_gt",
) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("no HDBSCAN no-GT candidates")
    metric = _resolve_hdbscan_nogt_selection_metric(cfg)
    best_score = max(_candidate_score(item) for item in candidates)
    use_config_k = str(requested_k_source).strip().lower() == "config" and int(requested_k) > 0

    if metric == HDBSCAN_NOGT_SELECTION_RETENTION:
        tie_delta = float(cfg.get("hdbscan_nogt_tie_delta", 0.03))
        near = [item for item in candidates if _candidate_score(item) >= best_score - tie_delta]
        selected = max(
            near,
            key=lambda item: (
                float(item.get("cluster_count_stability", 0.0)),
                -float(item.get("small_unit_pressure", 1.0)),
                float(item.get("clustered_fraction", 0.0)),
                _candidate_score(item),
                float(item.get("q_silhouette", 0.0)),
                float(item.get("q_dbi", 0.0)),
                -abs(_candidate_k(item) - int(requested_k)) if use_config_k else 0,
                _candidate_k(item),
            ),
        )
        selected = dict(selected)
        selected["hdbscan_nogt_selection"] = _hdbscan_selection_diag(
            candidates=candidates,
            near=near,
            pool=near,
            selected=selected,
            best_score=best_score,
            delta=tie_delta,
            metric=metric,
            reason="retention_tie_break",
        )
        return selected

    delta = max(0.0, float(cfg.get("hdbscan_high_recall_k_delta", cfg.get("hdbscan_nogt_high_recall_k_delta", 0.05))))
    min_clustered_fraction = clamp01(
        cfg.get("hdbscan_high_recall_k_min_clustered_fraction", cfg.get("hdbscan_nogt_min_clustered_fraction", 0.55))
    )
    max_small_unit_pressure = clamp01(
        cfg.get("hdbscan_high_recall_k_max_small_unit_pressure", cfg.get("hdbscan_nogt_max_small_unit_pressure", 0.35))
    )
    max_noise_ratio = clamp01(
        cfg.get("hdbscan_high_recall_k_max_noise_ratio", cfg.get("hdbscan_nogt_max_noise_ratio", 0.55))
    )
    near = [item for item in candidates if _candidate_score(item) >= best_score - delta]
    constrained = [
        item
        for item in near
        if float(item.get("clustered_fraction", 0.0)) >= min_clustered_fraction
        and float(item.get("small_unit_pressure", 1.0)) <= max_small_unit_pressure
        and float(item.get("noise_ratio", 1.0)) <= max_noise_ratio
    ]
    pool = constrained if constrained else near
    reason = "high_recall_k_constrained" if constrained else "high_recall_k_near_best_fallback"
    if use_config_k:
        selected = max(
            pool,
            key=lambda item: (
                -abs(_candidate_k(item) - int(requested_k)),
                int(_candidate_k(item) >= int(requested_k)),
                _candidate_score(item),
                float(item.get("clustered_fraction", 0.0)),
                -float(item.get("small_unit_pressure", 1.0)),
                -float(item.get("noise_ratio", 1.0)),
            ),
        )
    else:
        selected = max(
            pool,
            key=lambda item: (
                _candidate_k(item),
                float(item.get("clustered_fraction", 0.0)),
                _candidate_score(item),
                float(item.get("q_silhouette", 0.0)),
                float(item.get("q_dbi", 0.0)),
                -float(item.get("small_unit_pressure", 1.0)),
                -float(item.get("noise_ratio", 1.0)),
            ),
        )
    selected = dict(selected)
    selected["hdbscan_nogt_selection"] = _hdbscan_selection_diag(
        candidates=candidates,
        near=near,
        pool=pool,
        selected=selected,
        best_score=best_score,
        delta=delta,
        metric=metric,
        reason=reason,
        constraints={
            "min_clustered_fraction": float(min_clustered_fraction),
            "max_small_unit_pressure": float(max_small_unit_pressure),
            "max_noise_ratio": float(max_noise_ratio),
        },
    )
    return selected


def estimate_hdbscan_nogt_merge_stop_k(
    hdbscan_diag: Dict[str, Any],
    *,
    premerge_k: int,
    cfg: Dict[str, Any],
) -> tuple[int, Dict[str, Any]]:
    if not bool(cfg.get("nogt_k_stop_enabled", False)):
        return 0, {"enabled": False, "target_k": 0}
    candidates = [dict(item) for item in hdbscan_diag.get("hdbscan_nogt_candidates", []) if isinstance(item, dict)]
    if not candidates:
        return 0, {"enabled": True, "target_k": 0, "reason": "missing_candidates"}
    best_score = max(_candidate_score(item) for item in candidates)
    delta = max(0.0, float(cfg.get("nogt_k_stop_delta", 0.30)))
    near = [item for item in candidates if _candidate_score(item) >= best_score - delta]
    if not near:
        near = list(candidates)
    min_clustered = clamp01(cfg.get("nogt_k_stop_min_clustered_fraction", 0.25))
    max_small = clamp01(cfg.get("nogt_k_stop_max_small_unit_pressure", 1.0))
    max_noise = clamp01(cfg.get("nogt_k_stop_max_noise_ratio", 0.80))
    constrained = [
        item
        for item in near
        if float(item.get("clustered_fraction", 0.0)) >= min_clustered
        and float(item.get("small_unit_pressure", 1.0)) <= max_small
        and float(item.get("noise_ratio", 1.0)) <= max_noise
    ]
    pool = constrained if constrained else near
    values = sorted(int(item.get("actual_clusters", 0)) for item in pool if int(item.get("actual_clusters", 0)) > 0)
    if not values:
        return 0, {"enabled": True, "target_k": 0, "reason": "empty_pool"}
    quantile = float(np.clip(float(cfg.get("nogt_k_stop_quantile", 0.90)), 0.0, 1.0))
    target = int(math.ceil(float(np.quantile(np.asarray(values, dtype=np.float64), quantile))))
    min_fraction = float(np.clip(float(cfg.get("nogt_k_stop_min_premerge_fraction", 0.40)), 0.0, 1.0))
    target = max(target, int(math.ceil(float(premerge_k) * min_fraction)))
    target = max(1, min(int(premerge_k), int(target)))
    return target, {
        "enabled": True,
        "target_k": int(target),
        "best_score": float(best_score),
        "delta": float(delta),
        "quantile": float(quantile),
        "min_premerge_fraction": float(min_fraction),
        "min_clustered_fraction": float(min_clustered),
        "max_small_unit_pressure": float(max_small),
        "max_noise_ratio": float(max_noise),
        "candidate_count": int(len(candidates)),
        "near_count": int(len(near)),
        "pool_count": int(len(pool)),
        "pool_k_min": int(values[0]),
        "pool_k_max": int(values[-1]),
        "pool_k_values": sorted(set(int(v) for v in values)),
        "reason": "near_best_high_quantile",
    }


def fit_hdbscan_labels_nogt(
    z: np.ndarray,
    cfg: Dict[str, Any],
    *,
    requested_k: int = 0,
    requested_k_source: str = "no_gt",
) -> tuple[np.ndarray, Dict[str, Any]]:
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:
        raise RuntimeError("run_online requires the hdbscan package") from exc

    min_cluster_sizes = ro._as_search_values(cfg["hdbscan_min_cluster_size"], cast=int, min_value=2)
    min_samples_values = ro._as_search_values(cfg["hdbscan_min_samples"], cast=int, min_value=1)
    epsilon_values = ro._as_search_values(cfg["hdbscan_cluster_selection_epsilon"], cast=float, min_value=0.0)
    selection_method = str(cfg.get("hdbscan_cluster_selection_method", "leaf")).strip().lower() or "leaf"
    if selection_method not in ("leaf", "eom"):
        raise ValueError("offline.clustering.hdbscan_cluster_selection_method must be leaf or eom")
    if not min_cluster_sizes:
        min_cluster_sizes = [max(2, min(10, int(z.shape[0])))]
    if not min_samples_values:
        min_samples_values = [1]
    if not epsilon_values:
        epsilon_values = [0.0]
    selection_metric = _resolve_hdbscan_nogt_selection_metric(cfg)

    labels_by_index: Dict[int, np.ndarray] = {}
    candidates: List[Dict[str, Any]] = []
    for min_cluster_size in min_cluster_sizes:
        if int(min_cluster_size) > int(z.shape[0]):
            continue
        for min_samples in min_samples_values:
            for epsilon in epsilon_values:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples),
                    cluster_selection_epsilon=float(epsilon),
                    metric="euclidean",
                    cluster_selection_method=selection_method,
                )
                raw_labels = clusterer.fit_predict(z).astype(np.int32, copy=False)
                candidate = _hdbscan_nogt_metrics(z, raw_labels, clusterer, cfg=cfg)
                candidate.update(
                    {
                        "candidate_index": int(len(candidates)),
                        "configured_k_delta": (
                            abs(int(candidate["actual_clusters"]) - int(requested_k))
                            if str(requested_k_source).strip().lower() == "config" and int(requested_k) > 0
                            else None
                        ),
                        "min_cluster_size": int(min_cluster_size),
                        "min_samples": int(min_samples),
                        "cluster_selection_epsilon": float(epsilon),
                        "cluster_selection_method": str(selection_method),
                        "metric": "euclidean",
                    }
                )
                labels_by_index[int(candidate["candidate_index"])] = raw_labels.astype(np.int32, copy=True)
                candidates.append(candidate)

    if not candidates:
        empty = {
            "hdbscan_selection_metric": selection_metric,
            "hdbscan_nogt_score": float("-inf"),
            "hdbscan_nogt_selected_params": None,
            "hdbscan_nogt_candidates": [],
            "hdbscan_nogt_selection": {
                "hdbscan_selection_metric": selection_metric,
                "selection_reason": "no_candidates",
                "selected_cluster_count": 0,
                "candidate_count": 0,
                "candidate_k_min": None,
                "candidate_k_max": None,
                "near_best_candidate_count": 0,
                "near_best_k_min": None,
                "near_best_k_max": None,
                "k_confidence": "unknown",
            },
            "selected_cluster_count": 0,
            "candidate_count": 0,
            "candidate_k_min": None,
            "candidate_k_max": None,
            "near_best_candidate_count": 0,
            "near_best_k_min": None,
            "near_best_k_max": None,
            "k_confidence": "unknown",
            "selection_reason": "no_candidates",
            "actual_clusters": 0,
            "noise_events": int(z.shape[0]),
        }
        return np.full((z.shape[0],), -1, dtype=np.int32), empty

    _annotate_hdbscan_nogt_scores(candidates)
    selected = _choose_hdbscan_nogt_candidate(
        candidates,
        cfg,
        requested_k=int(requested_k),
        requested_k_source=str(requested_k_source),
    )
    selected_index = int(selected["candidate_index"])
    selected_labels = labels_by_index[selected_index]
    params = {
        "candidate_index": int(selected_index),
        "min_cluster_size": int(selected["min_cluster_size"]),
        "min_samples": int(selected["min_samples"]),
        "cluster_selection_epsilon": float(selected["cluster_selection_epsilon"]),
        "cluster_selection_method": str(selected["cluster_selection_method"]),
        "metric": "euclidean",
        "actual_clusters": int(selected["actual_clusters"]),
        "noise_events": int(selected["noise_events"]),
        "hdbscan_nogt_score": float(selected["hdbscan_nogt_score"]),
        "configured_k_delta": selected.get("configured_k_delta"),
        "clustered_fraction": float(selected.get("clustered_fraction", 0.0)),
        "small_unit_pressure": float(selected.get("small_unit_pressure", 1.0)),
        "noise_ratio": float(selected.get("noise_ratio", 1.0)),
    }
    selection_diag = dict(selected.get("hdbscan_nogt_selection", {}))
    diag = dict(params)
    diag.update(
        {
            "hdbscan_selection_metric": selection_metric,
            "hdbscan_nogt_score": float(selected["hdbscan_nogt_score"]),
            "hdbscan_nogt_selected_params": params,
            "hdbscan_nogt_candidates": [ro.to_jsonable(item) for item in candidates],
            "hdbscan_nogt_selection": ro.to_jsonable(selection_diag),
            "selected_cluster_count": int(selection_diag.get("selected_cluster_count", selected["actual_clusters"])),
            "candidate_count": int(selection_diag.get("candidate_count", len(candidates))),
            "candidate_k_min": selection_diag.get("candidate_k_min"),
            "candidate_k_max": selection_diag.get("candidate_k_max"),
            "near_best_candidate_count": int(selection_diag.get("near_best_candidate_count", 0)),
            "near_best_k_min": selection_diag.get("near_best_k_min"),
            "near_best_k_max": selection_diag.get("near_best_k_max"),
            "k_confidence": str(selection_diag.get("k_confidence", "unknown")),
            "selection_reason": str(selection_diag.get("selection_reason", "")),
            "score": [
                float(selected["hdbscan_nogt_score"]),
                float(selected.get("cluster_count_stability", 0.0)),
                float(selected.get("clustered_fraction", 0.0)),
                -float(selected.get("small_unit_pressure", 1.0)),
                int(selected.get("actual_clusters", 0)),
            ],
        }
    )
    return selected_labels.astype(np.int32, copy=False), diag


def _resolve_hdbscan_nogt_merge_cfg(cfg: Dict[str, Any]) -> tuple[bool, str, Dict[str, Any]]:
    raw_profile = str(cfg.get("hdbscan_merge_profile", "balanced")).strip().lower() or "balanced"
    if raw_profile in ("none", "off", "false", "disabled"):
        return False, "none", dict(cfg)
    if raw_profile not in HDBSCAN_NOGT_MERGE_PROFILES:
        raise ValueError(
            "offline.clustering.hdbscan_merge_profile must be one of "
            f"{', '.join(sorted(HDBSCAN_NOGT_MERGE_PROFILES))}, none"
        )
    enabled = bool(cfg.get("hdbscan_merge_enabled", False))
    out = dict(cfg)
    defaults = HDBSCAN_NOGT_MERGE_PROFILES[raw_profile]
    for key, value in defaults.items():
        out[key] = cfg.get(key, value)
    out["hdbscan_merge_enabled"] = bool(enabled)
    out["hdbscan_merge_profile"] = raw_profile
    return bool(enabled), raw_profile, out


def _hdbscan_merge_confidence(merge_diag: Dict[str, Any]) -> str:
    if not bool(merge_diag.get("enabled", False)):
        return "disabled"
    pairs = merge_diag.get("pairs", [])
    if not pairs:
        return "none"
    cosines = [float(pair.get("cosine", 0.0)) for pair in pairs if math.isfinite(float(pair.get("cosine", 0.0)))]
    conflicts = [
        float(pair.get("cross_conflict", 1.0))
        for pair in pairs
        if math.isfinite(float(pair.get("cross_conflict", 1.0)))
    ]
    min_cos = min(cosines) if cosines else 0.0
    max_conflict = max(conflicts) if conflicts else 1.0
    if min_cos >= 0.985 and max_conflict <= 0.005:
        return "high"
    if min_cos >= 0.970 and max_conflict <= 0.010:
        return "medium"
    return "low"


def apply_hdbscan_nogt_merge(
    valid_z: np.ndarray,
    valid_labels: np.ndarray,
    cfg: Dict[str, Any],
    *,
    center_channels: np.ndarray | None,
    times_samples: np.ndarray | None,
    requested_k: int = 0,
) -> tuple[np.ndarray, Dict[str, Any]]:
    enabled, profile, merge_cfg = _resolve_hdbscan_nogt_merge_cfg(cfg)
    labels = np.asarray(valid_labels, dtype=np.int32).reshape(-1)
    pre_clusters = int(ro._positive_cluster_count(labels))
    if not enabled:
        return labels.copy(), {
            "enabled": False,
            "profile": profile,
            "premerge_clusters": int(pre_clusters),
            "postmerge_clusters": int(pre_clusters),
            "merges": 0,
            "pairs": [],
            "confidence": "disabled",
            "target_clusters": int(requested_k),
        }
    if labels.size == 0 or pre_clusters <= 1:
        return labels.copy(), {
            "enabled": True,
            "profile": profile,
            "premerge_clusters": int(pre_clusters),
            "postmerge_clusters": int(pre_clusters),
            "merges": 0,
            "pairs": [],
            "confidence": "none",
            "target_clusters": int(requested_k),
        }

    merged_zero_based, merge_diag = ro._merge_seed_labels(
        ro.normalize_embedding_matrix(valid_z),
        labels,
        merge_cfg,
        center_channels=center_channels,
        times_samples=times_samples,
        requested_k=int(requested_k),
        prefix="hdbscan",
    )
    merged = ro._relabel_positive(merged_zero_based)
    post_clusters = int(ro._positive_cluster_count(merged))
    out_diag = dict(merge_diag)
    out_diag.update(
        {
            "enabled": True,
            "profile": profile,
            "premerge_clusters": int(pre_clusters),
            "postmerge_clusters": int(post_clusters),
            "merge_count": int(merge_diag.get("merges", 0)),
            "confidence": _hdbscan_merge_confidence(merge_diag),
        }
    )
    return merged.astype(np.int32, copy=False), out_diag


def offline_cluster_outputs_nogt(
    arr: Dict[str, np.ndarray],
    *,
    cfg: Dict[str, Any],
    requested_k: int,
    requested_k_source: str,
    assignment_gate_thresh: float,
    disable_candidate_filtering: bool = False,
    disable_encode_gate: bool = False,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if "embedding" not in arr:
        raise ValueError("offline clustering requires collected embedding arrays")
    method = str(cfg.get("method", "hdbscan")).strip().lower()
    if method != "hdbscan":
        raise ValueError(f"run_online only supports hdbscan, got method={method!r}")

    out = {key: np.asarray(value).copy() for key, value in arr.items()}
    n_events = int(out["event_time_sample"].shape[0])
    labels = np.full((n_events,), -1, dtype=np.int32)
    gate_prob = np.asarray(out.get("gate_prob", np.zeros((n_events,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    encoded = np.asarray(out.get("encoded_mask", np.zeros((n_events,), dtype=np.int8))).reshape(-1) > 0
    if gate_prob.size != n_events or encoded.size != n_events:
        raise ValueError("gate/encoded arrays must match event count for offline clustering")
    gate_pass = gate_prob >= float(assignment_gate_thresh)
    assignment_pass = encoded if bool(disable_candidate_filtering) else encoded & gate_pass
    valid_idx = np.flatnonzero(assignment_pass)
    z_all = np.asarray(out["embedding"], dtype=np.float32)
    if z_all.ndim != 2 or z_all.shape[0] != n_events:
        raise ValueError(f"embedding shape mismatch: expected ({n_events}, D), got {z_all.shape}")

    fit_start = time.perf_counter()
    selection_metric = _resolve_hdbscan_nogt_selection_metric(cfg)
    if valid_idx.size == 0:
        fit_labels = np.zeros((0,), dtype=np.int32)
        hdbscan_best = {
            "hdbscan_selection_metric": selection_metric,
            "hdbscan_nogt_score": float("-inf"),
            "hdbscan_nogt_selected_params": None,
            "hdbscan_nogt_candidates": [],
            "hdbscan_nogt_selection": {
                "hdbscan_selection_metric": selection_metric,
                "selection_reason": "no_assignment_candidates",
                "selected_cluster_count": 0,
                "candidate_count": 0,
                "candidate_k_min": None,
                "candidate_k_max": None,
                "near_best_candidate_count": 0,
                "near_best_k_min": None,
                "near_best_k_max": None,
                "k_confidence": "unknown",
            },
            "actual_clusters": 0,
            "noise_events": 0,
            "selected_cluster_count": 0,
            "candidate_count": 0,
            "candidate_k_min": None,
            "candidate_k_max": None,
            "near_best_candidate_count": 0,
            "near_best_k_min": None,
            "near_best_k_max": None,
            "k_confidence": "unknown",
            "selection_reason": "no_assignment_candidates",
        }
    else:
        fit_labels, hdbscan_best = fit_hdbscan_labels_nogt(
            ro.normalize_embedding_matrix(z_all[valid_idx]),
            cfg,
            requested_k=int(requested_k),
            requested_k_source=str(requested_k_source),
        )
    fit_sec = time.perf_counter() - fit_start

    labels[valid_idx] = ro._relabel_positive(fit_labels)
    leaf_labels = labels.astype(np.int32, copy=False)
    post_labels = leaf_labels.copy()
    post_label_source = "leaf"
    merge_center_channels = np.asarray(out["center_channel"])[valid_idx] if "center_channel" in out else None
    merge_times_samples = np.asarray(out["event_time_sample"])[valid_idx] if "event_time_sample" in out else None
    merge_valid_z = z_all[valid_idx] if valid_idx.size else np.zeros((0, int(z_all.shape[1])), dtype=np.float32)
    merge_pre_k = int(ro._positive_cluster_count(leaf_labels[valid_idx]))
    nogt_k_stop_target, nogt_k_stop_diag = estimate_hdbscan_nogt_merge_stop_k(
        hdbscan_best,
        premerge_k=merge_pre_k,
        cfg=cfg,
    )
    merged_valid_labels, hdbscan_merge_diag = apply_hdbscan_nogt_merge(
        merge_valid_z,
        leaf_labels[valid_idx],
        cfg,
        center_channels=merge_center_channels,
        times_samples=merge_times_samples,
        requested_k=int(nogt_k_stop_target),
    )
    hdbscan_merge_diag["nogt_k_stop"] = dict(nogt_k_stop_diag)
    if bool(hdbscan_merge_diag.get("enabled", False)):
        post_labels = leaf_labels.copy()
        post_labels[valid_idx] = np.asarray(merged_valid_labels, dtype=np.int32)
        post_label_source = f"hdbscan_merge_{hdbscan_merge_diag.get('profile', 'custom')}"
    out["unit_id_leaf"] = leaf_labels.copy()
    out["unit_id_raw"] = leaf_labels.copy()
    out["unit_id_premerge"] = leaf_labels.copy()

    out["unit_id_post"] = post_labels.astype(np.int32, copy=False)
    out["unit_id"] = out["unit_id_post"].copy()
    diag = {
        "method": method,
        "requested_k": int(requested_k),
        "requested_k_source": str(requested_k_source),
        "actual_clusters": ro._positive_cluster_count(post_labels),
        "clustered_events": int(np.sum(post_labels > 0)),
        "noise_events": int(np.sum(post_labels < 0)),
        "leaf_actual_clusters": ro._positive_cluster_count(leaf_labels),
        "leaf_clustered_events": int(np.sum(leaf_labels > 0)),
        "leaf_noise_events": int(np.sum(leaf_labels < 0)),
        "premerge_clusters": int(hdbscan_merge_diag.get("premerge_clusters", ro._positive_cluster_count(leaf_labels))),
        "postmerge_clusters": int(hdbscan_merge_diag.get("postmerge_clusters", ro._positive_cluster_count(post_labels))),
        "hdbscan_merge_enabled": bool(hdbscan_merge_diag.get("enabled", False)),
        "hdbscan_merge_profile": hdbscan_merge_diag.get("profile", "none"),
        "hdbscan_merge_count": int(hdbscan_merge_diag.get("merge_count", hdbscan_merge_diag.get("merges", 0))),
        "hdbscan_merge_pairs": hdbscan_merge_diag.get("pairs", []),
        "hdbscan_merge_confidence": hdbscan_merge_diag.get("confidence", "disabled"),
        "hdbscan_merge": hdbscan_merge_diag,
        "hdbscan_online_profile": str(cfg.get("hdbscan_online_profile", "config")),
        "nogt_k_stop_enabled": bool(nogt_k_stop_diag.get("enabled", False)),
        "nogt_k_stop_target": int(nogt_k_stop_diag.get("target_k", 0) or 0),
        "nogt_k_stop_reason": str(nogt_k_stop_diag.get("reason", "")),
        "nogt_k_stop_pool_k_min": nogt_k_stop_diag.get("pool_k_min"),
        "nogt_k_stop_pool_k_max": nogt_k_stop_diag.get("pool_k_max"),
        "nogt_k_stop": nogt_k_stop_diag,
        "encode_gate_disabled": bool(disable_encode_gate),
        "encoded_events": int(np.sum(encoded)),
        "gate_pass_events": int(np.sum(gate_pass)),
        "encoded_gate_pass_events": int(np.sum(encoded & gate_pass)),
        "assignment_gate_thresh": float(assignment_gate_thresh),
        "candidate_filtering_disabled": bool(disable_candidate_filtering),
        "candidate_events": int(valid_idx.size),
        "post_label_source": post_label_source,
        "fit_seconds": float(fit_sec),
        "hdbscan_best_params": hdbscan_best,
        "hdbscan_selection_metric": hdbscan_best.get("hdbscan_selection_metric", selection_metric),
        "hdbscan_nogt_selected_params": hdbscan_best.get("hdbscan_nogt_selected_params"),
        "hdbscan_nogt_score": hdbscan_best.get("hdbscan_nogt_score"),
        "hdbscan_nogt_candidates": hdbscan_best.get("hdbscan_nogt_candidates", []),
        "hdbscan_nogt_selection": hdbscan_best.get("hdbscan_nogt_selection", {}),
        "selected_cluster_count": hdbscan_best.get("selected_cluster_count"),
        "candidate_count": hdbscan_best.get("candidate_count"),
        "candidate_k_min": hdbscan_best.get("candidate_k_min"),
        "candidate_k_max": hdbscan_best.get("candidate_k_max"),
        "near_best_candidate_count": hdbscan_best.get("near_best_candidate_count"),
        "near_best_k_min": hdbscan_best.get("near_best_k_min"),
        "near_best_k_max": hdbscan_best.get("near_best_k_max"),
        "k_confidence": hdbscan_best.get("k_confidence"),
        "selection_reason": hdbscan_best.get("selection_reason"),
    }
    return out, diag


def build_assignment_candidate_mask(arr: Dict[str, np.ndarray], ctx: InferenceContext) -> np.ndarray:
    n = int(np.asarray(arr["event_time_sample"]).shape[0])
    encoded = np.asarray(arr.get("encoded_mask", np.zeros((n,), dtype=np.int8))).reshape(-1) > 0
    gate_prob = np.asarray(arr.get("gate_prob", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if encoded.size != n or gate_prob.size != n:
        raise ValueError("gate/encoded arrays must match event count")
    if bool(getattr(ctx, "candidate_filtering_disabled", False)):
        return encoded
    return encoded & (gate_prob >= float(ctx.assignment_gate_thresh))


def apply_online_assignment(
    arr: Dict[str, np.ndarray],
    *,
    init_idx: np.ndarray,
    init_labels: np.ndarray,
    units: Dict[int, InitializedUnit],
    ctx: InferenceContext,
    init_window_sec: float,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    out = {key: np.asarray(value).copy() for key, value in arr.items()}
    n = int(np.asarray(out["event_time_sample"]).shape[0])
    init_idx = np.asarray(init_idx, dtype=np.int64).reshape(-1)
    init_labels = np.asarray(init_labels, dtype=np.int32).reshape(-1)
    if init_idx.size != init_labels.size:
        raise ValueError("init_idx/init_labels length mismatch")

    times = np.asarray(out["event_time_sample"], dtype=np.int64)
    cutoff = int(round(float(init_window_sec) * float(ctx.fs)))
    init_mask = times < cutoff
    online_mask = ~init_mask
    candidate_mask = build_assignment_candidate_mask(out, ctx) & online_mask
    z = np.asarray(out["embedding"], dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != n:
        raise ValueError(f"embedding shape mismatch: expected ({n}, D), got {z.shape}")

    unit_id_init = np.full((n,), -1, dtype=np.int32)
    unit_id_init[init_idx] = init_labels.astype(np.int32, copy=False)
    unit_id_online = unit_id_init.copy()
    accepted = np.zeros((n,), dtype=np.bool_)
    assign_conf = np.zeros((n,), dtype=np.float32)
    assign_dist = np.full((n,), np.nan, dtype=np.float32)
    assign_margin = np.full((n,), np.nan, dtype=np.float32)
    reason = np.full((n,), -1, dtype=np.int16)

    attempted = np.flatnonzero(candidate_mask).astype(np.int64, copy=False)
    assign_mode = str(ctx.online_cfg.assign_mode)
    prototypes: Dict[int, UnitPrototypeSet] = {}
    if assign_mode == "multi_prototype":
        prototypes = build_prototypes_from_units(
            out,
            units,
            max_prototypes_per_unit=int(ctx.online_cfg.max_prototypes_per_unit),
            min_events_per_prototype=int(ctx.online_cfg.min_events_per_prototype),
            center_trim_quantile=float(ctx.offline_cluster_cfg.get("graph_center_trim_quantile", 0.90)),
        )
    updates = 0
    accepted_count = 0
    rejected_count = 0
    for idx in attempted.tolist():
        event = event_for_index(out, int(idx))
        result = assign_to_unit_prototypes(
            [event],
            z[int(idx) : int(idx) + 1],
            prototypes,
            cfg=ctx.online_cfg.assign_cfg,
            neighbors_lookup=ctx.pipeline.neighbors,
        )
        unit_id_online[int(idx)] = int(result.unit_id[0])
        accepted[int(idx)] = bool(result.accepted[0])
        assign_conf[int(idx)] = np.float32(result.confidence[0])
        assign_dist[int(idx)] = np.float32(result.distance[0])
        assign_margin[int(idx)] = np.float32(result.margin[0])
        reason[int(idx)] = np.int16(result.reason_code[0])
        if bool(result.accepted[0]):
            accepted_count += 1
            unit_id = int(result.unit_id[0])
            if (
                ctx.online_cfg.update_enabled
                and float(result.confidence[0]) >= float(ctx.online_cfg.update_conf_min)
                and unit_id in units
            ):
                update_unit_center(
                    units[unit_id],
                    z[int(idx)],
                    event,
                    alpha_mu=float(ctx.online_cfg.update_alpha_mu),
                    alpha_stats=float(ctx.online_cfg.update_alpha_stats),
                )
                if assign_mode == "multi_prototype":
                    update_nearest_unit_prototype(
                        prototypes,
                        unit_id,
                        z[int(idx)],
                        alpha_mu=float(ctx.online_cfg.update_alpha_mu),
                    )
                updates += 1
        else:
            rejected_count += 1

    out["unit_id_init"] = unit_id_init
    out["unit_id_online"] = unit_id_online.astype(np.int32, copy=False)
    out["unit_id_raw"] = unit_id_init.copy()
    out["unit_id_post"] = unit_id_online.astype(np.int32, copy=False)
    out["unit_id"] = unit_id_online.astype(np.int32, copy=False)
    out["online_init_mask"] = init_mask.astype(np.bool_, copy=False)
    out["online_eval_mask"] = online_mask.astype(np.bool_, copy=False)
    out["online_assign_accepted"] = accepted
    out["online_assign_confidence"] = assign_conf
    out["online_assign_distance"] = assign_dist
    out["online_assign_margin"] = assign_margin
    out["online_assign_reason_code"] = reason

    diag = {
        "init_window_sec": float(init_window_sec),
        "init_cutoff_sample": int(cutoff),
        "init_events": int(np.sum(init_mask)),
        "online_events": int(np.sum(online_mask)),
        "units_initialized": int(len(units)),
        "assign_mode": assign_mode,
        "max_prototypes_per_unit": int(ctx.online_cfg.max_prototypes_per_unit),
        "min_events_per_prototype": int(ctx.online_cfg.min_events_per_prototype),
        "prototype_counts_by_unit": {
            str(int(unit_id)): int(np.asarray(proto.centers).shape[0])
            for unit_id, proto in sorted(prototypes.items())
        },
        "prototype_count": int(sum(int(np.asarray(proto.centers).shape[0]) for proto in prototypes.values())),
        "encode_gate_disabled": bool(getattr(ctx, "encode_gate_disabled", False)),
        "candidate_filtering_disabled": bool(getattr(ctx, "candidate_filtering_disabled", False)),
        "encoded_events": int(np.sum(np.asarray(arr.get("encoded_mask", np.zeros((n,), dtype=np.int8))).reshape(-1) > 0)),
        "gate_pass_events": int(
            np.sum(
                np.asarray(arr.get("gate_prob", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
                >= float(ctx.assignment_gate_thresh)
            )
        ),
        "candidate_events": int(attempted.size),
        "assignment_candidates": int(attempted.size),
        "assignment_accepted": int(accepted_count),
        "assignment_rejected": int(rejected_count),
        "assignment_accept_rate": float(accepted_count / max(1, int(attempted.size))),
        "updates": int(updates),
        "update_enabled": bool(ctx.online_cfg.update_enabled),
        "update_conf_min": float(ctx.online_cfg.update_conf_min),
        "update_alpha_mu": float(ctx.online_cfg.update_alpha_mu),
        "update_alpha_stats": float(ctx.online_cfg.update_alpha_stats),
        "assign_cfg": ro.to_jsonable(ctx.online_cfg.assign_cfg),
        "positive_init_labels": int(np.sum(unit_id_init > 0)),
        "positive_online_labels": int(np.sum(unit_id_online > 0)),
    }
    return out, diag


def load_input_metadata(case: Case, cfg: Dict[str, Any], args: argparse.Namespace) -> tuple[Any | None, Path, bool, float, int, int, np.ndarray | None]:
    offline_cfg = dict(cfg.get("offline", {}) or {})
    input_h5 = case.input_h5
    input_bin = case.input_bin
    if bool(input_h5) == bool(input_bin):
        raise ValueError("set exactly one input source: input_h5 or input_bin")

    fs = float(args.fs if args.fs is not None else offline_cfg.get("fs", cfg.get("fs", 30000.0)))
    if input_h5 is not None:
        input_path = input_h5
        if not input_path.exists():
            raise FileNotFoundError(f"input_h5 not found: {input_path}")
        reader = ro.MEArecH5Reader(input_path)
        fs = float(reader.get_fs())
        n_samples, n_channels = reader.get_recordings_shape()
        channel_positions = reader.read_channel_positions()
        return reader, input_path, True, fs, int(n_samples), int(n_channels), channel_positions

    assert input_bin is not None
    input_path = input_bin
    if not input_path.exists():
        raise FileNotFoundError(f"input_bin not found: {input_path}")
    n_channels_arg = args.n_channels if args.n_channels is not None else case.n_channels
    if n_channels_arg is None or int(n_channels_arg) <= 0:
        raise ValueError("--n-channels is required for --input-bin")
    n_samples, n_channels = ro.get_bin_shape(input_path, int(n_channels_arg))
    return None, input_path, False, fs, int(n_samples), int(n_channels), None


def prepare_context(case: Case, cfg: Dict[str, Any], args: argparse.Namespace, logger: logging.Logger) -> InferenceContext:
    offline_cfg = dict(cfg.get("offline", {}) or {})
    offline_cluster_cfg = resolve_online_offline_cluster_config(offline_cfg, logger)
    online_cfg = resolve_online_config(cfg, args.init_window_sec)
    if "hdbscan" in {str(value).strip().lower() for value in offline_cluster_cfg.get("methods", [])}:
        logger.info("optional_dependency hdbscan_available=%s", str(importlib.util.find_spec("hdbscan") is not None))

    reader, input_path, is_h5, fs, n_samples, n_channels, channel_positions_raw = load_input_metadata(case, cfg, args)
    requested_device = args.device or str(offline_cfg.get("device", cfg.get("device", "cuda")))
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = ro.resolve_cuda_device(requested_device)
    logger.info("device=%s", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    chunk_sec = float(args.chunk_sec if args.chunk_sec is not None else offline_cfg.get("chunk_sec", cfg.get("chunk_sec", 0.1)))
    window_sec = float(cfg.get("window_sec", 1.5))
    stride_sec = float(cfg.get("stride_sec", 0.5))
    margin_sec = float(cfg.get("margin_sec", 0.05))
    if bool(offline_cfg.get("fast_offline", True)) and chunk_sec < stride_sec:
        logger.info("online_fast_mode bump chunk_sec from %.3f to stride_sec %.3f", chunk_sec, stride_sec)
        chunk_sec = stride_sec
    chunk = max(1, int(round(float(chunk_sec) * float(fs))))
    chunk_sec = float(chunk / float(fs))

    requested_k, requested_k_source = resolve_requested_k_no_gt(cfg, offline_cluster_cfg)
    patch = dict(cfg.get("patch", {}) or {})
    det = dict(cfg.get("detection", {}) or {})
    model = dict(cfg.get("model", {}) or {})
    fallback_k = int(max(1, patch.get("k", 16)))
    electrode_config = args.electrode_config if args.electrode_config is not None else offline_cfg.get("electrode_config")
    channel_positions, neighbors_lookup, neighbor_graph_source = ro.resolve_neighbor_graph(
        channel_positions=channel_positions_raw,
        n_channels=int(n_channels),
        electrode_config=electrode_config,
        neighbor_source=str(det.get("neighbor_source", "auto")),
        fallback_k=fallback_k,
        root=ROOT,
        logger=logger,
    )
    embed_dim = int(model.get("embed_dim", 64))
    patch_peak_only_cfg = bool(patch.get("peak_only", False))
    resolved_patch_k_for_checkpoint = 1 if patch_peak_only_cfg else fallback_k
    if not patch_peak_only_cfg and neighbors_lookup is not None:
        resolved_patch_k_for_checkpoint = int(neighbors_lookup.shape[1])
    electrode_config_path = resolve_path(electrode_config) if electrode_config else None
    effective_electrode_config_path = (
        electrode_config_path
        if neighbor_graph_source == "config" and electrode_config_path is not None and electrode_config_path.exists()
        else None
    )
    weights_raw = args.weights or offline_cfg.get("weights")
    weights_path = resolve_path(weights_raw) if weights_raw else None
    if weights_path is None:
        raise ValueError("Trained weights are required; pass --weights or import the paper checkpoint")
    weight_payload = None
    if weights_path is not None:
        if not weights_path.exists():
            raise FileNotFoundError(f"weights not found: {weights_path}")
        weight_payload = ro.load_torch_file(weights_path)
    fallback_peak_only_raw = offline_cfg.get("fallback_peak_only_weights")
    fallback_peak_only_path = resolve_path(fallback_peak_only_raw) if fallback_peak_only_raw else None
    weights_path, weight_payload, encoder_checkpoint_info = ro.validate_or_fallback_encoder_checkpoint(
        requested_weights_path=weights_path,
        requested_payload=weight_payload,
        model_cfg=model,
        resolved_patch_k=resolved_patch_k_for_checkpoint,
        electrode_config_path=effective_electrode_config_path,
        fallback_peak_only_weights=fallback_peak_only_path,
    )
    effective_encoder_route = str(
        encoder_checkpoint_info.get("checkpoint_encoder_route", model.get("encoder_route", "peak_only"))
    )
    model_runtime = dict(model)
    model_runtime["encoder_route"] = effective_encoder_route
    patch_peak_only = effective_encoder_route == "peak_only"
    patch_k = 1 if patch_peak_only else int(resolved_patch_k_for_checkpoint)
    encode_gate_thresh, assignment_gate_thresh, gate_calibration = ro.resolve_runtime_gate_thresholds(
        model_runtime,
        weights_path=weights_path,
        payload=weight_payload,
    )
    threshold_values, threshold_notes = ro.normalize_runtime_gate_thresholds(
        encode_gate_thresh=float(encode_gate_thresh),
        assignment_gate_thresh=float(assignment_gate_thresh),
        novelty_birth_gate_min=float(dict(cfg.get("clustering", {}) or {}).get("novelty_birth_gate_min", 0.10)),
        keep_gate_min=float(dict(cfg.get("clustering", {}) or {}).get("keep_gate_min", 0.10)),
        init_seed_gate=float(dict(cfg.get("clustering", {}) or {}).get("init_seed_gate", 0.10)),
    )
    encode_gate_thresh = float(threshold_values["encode_gate_thresh"])
    assignment_gate_thresh = float(threshold_values["assignment_gate_thresh"])
    for note in threshold_notes:
        logger.info("%s", note)
    model_cfg = ro.build_model_config(
        model_runtime,
        embed_dim=embed_dim,
        encode_gate_thresh=encode_gate_thresh,
        assignment_gate_thresh=assignment_gate_thresh,
    )
    min_ring_capacity_sec = window_sec + 2.0 * margin_sec + max(stride_sec, chunk_sec) + 0.1
    ring_capacity_sec = float(offline_cfg.get("ring_capacity_sec", cfg.get("ring_capacity_sec", max(2.0, min_ring_capacity_sec))))
    ring_capacity_sec = max(ring_capacity_sec, min_ring_capacity_sec)
    pipe_cfg = ro.PipelineConfig(
        fs=float(fs),
        n_channels=int(n_channels),
        chunk_sec=float(chunk_sec),
        window_sec=float(window_sec),
        stride_sec=float(stride_sec),
        margin_sec=float(margin_sec),
        ring_capacity_sec=float(ring_capacity_sec),
        device=str(device),
    )
    det_cfg = ro.build_detector_config(det, fs=float(fs), n_channels=int(n_channels))
    pipeline = ro.OnlineSpikePipeline(
        pipe_cfg=pipe_cfg,
        det_cfg=det_cfg,
        patch_cfg=ro.PatchConfig(
            left=int(patch.get("left", 20)),
            right=int(patch.get("right", 44)),
            k=int(patch_k),
            peak_only=bool(patch_peak_only),
        ),
        model_cfg=model_cfg,
        channel_positions=channel_positions,
        neighbors_lookup=neighbors_lookup,
        car_mode=dict(cfg.get("preprocess", {}) or {}).get("car_mode", "global"),
        noise_beta=float(dict(cfg.get("preprocess", {}) or {}).get("noise_beta", 0.05)),
        patch_norm_mode=str(dict(cfg.get("preprocess", {}) or {}).get("patch_norm_mode", "peak_channel_sigma")),
        cluster_enabled=False,
        apply_encode_gate=not bool(dict(cfg.get("runtime", {}) or {}).get("disable_encode_gate", False) or dict(cfg.get("runtime", {}) or {}).get("disable_candidate_filtering", False)),
    )
    if weights_path is not None and weight_payload is not None:
        state = weight_payload
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = pipeline.model.load_state_dict(state, strict=True)
        pipeline.model.eval()
        logger.info(
            "loaded weights: %s (missing_keys=%d unexpected_keys=%d)",
            weights_path,
            len(missing),
            len(unexpected),
        )
        if gate_calibration is not None:
            logger.info(
                "loaded_saved_gate_calibration encode_gate_thresh=%.4f assignment_gate_thresh=%.4f selection_mode=%s",
                float(model_cfg.encode_gate_thresh),
                float(model_cfg.assignment_gate_thresh),
                str(gate_calibration.get("selection_mode", "saved")),
            )

    logger.info(
        (
            "runtime_config case=%s input=%s samples=%d channels=%d fs=%.2f chunk_sec=%.3f "
            "init_windows=%s requested_k=%d requested_k_source=%s out_base=%s"
        ),
        case.name,
        input_path,
        int(n_samples),
        int(n_channels),
        float(fs),
        float(chunk_sec),
        ",".join(init_window_suffix(v) for v in online_cfg.init_window_sec),
        int(requested_k),
        str(requested_k_source),
        str(case.out_npz),
    )
    runtime_cfg = dict(cfg.get("runtime", {}) or {})
    candidate_filtering_disabled = bool(runtime_cfg.get("disable_candidate_filtering", False))
    encode_gate_disabled = bool(runtime_cfg.get("disable_encode_gate", False) or candidate_filtering_disabled)
    logger.info("offline_clustering_config %s", json.dumps(ro.to_jsonable(offline_cluster_cfg), sort_keys=True))
    logger.info("online_config %s", json.dumps(ro.to_jsonable(online_cfg), sort_keys=True))
    logger.info("encode_gate_disabled=%s", str(encode_gate_disabled))
    logger.info("candidate_filtering_disabled=%s", str(candidate_filtering_disabled))
    return InferenceContext(
        cfg=cfg,
        offline_cfg=offline_cfg,
        offline_cluster_cfg=offline_cluster_cfg,
        online_cfg=online_cfg,
        input_path=input_path,
        input_h5=bool(is_h5),
        reader=reader,
        fs=float(fs),
        n_samples=int(n_samples),
        n_channels=int(n_channels),
        channel_positions=channel_positions,
        weights_path=weights_path,
        weight_payload=weight_payload,
        pipeline=pipeline,
        model_cfg=model_cfg,
        assignment_gate_thresh=float(assignment_gate_thresh),
        encode_gate_disabled=encode_gate_disabled,
        candidate_filtering_disabled=candidate_filtering_disabled,
        requested_k=int(requested_k),
        requested_k_source=str(requested_k_source),
        save_embeddings=bool(offline_cfg.get("save_embeddings", True)),
    )


def prepare_reuse_context(
    case: Case,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    logger: logging.Logger,
    *,
    source_npz: Path,
    arr: Dict[str, np.ndarray],
) -> InferenceContext:
    cfg = force_reuse_online_cfg(cfg)
    offline_cfg = dict(cfg.get("offline", {}) or {})
    offline_cluster_cfg = resolve_online_offline_cluster_config(offline_cfg, logger)
    online_cfg = resolve_online_config(cfg, args.init_window_sec or "20")
    if str(online_cfg.assign_mode) != "multi_prototype":
        raise RuntimeError("reuse mode must use online.assign_mode=multi_prototype")
    if "hdbscan" in {str(value).strip().lower() for value in offline_cluster_cfg.get("methods", [])}:
        logger.info("optional_dependency hdbscan_available=%s", str(importlib.util.find_spec("hdbscan") is not None))

    reader, input_path, is_h5, fs, n_samples, n_channels, channel_positions_raw = load_input_metadata(case, cfg, args)
    requested_k, requested_k_source = resolve_requested_k_no_gt(cfg, offline_cluster_cfg)
    patch = dict(cfg.get("patch", {}) or {})
    det = dict(cfg.get("detection", {}) or {})
    model = dict(cfg.get("model", {}) or {})
    fallback_k = int(max(1, patch.get("k", 16)))
    electrode_config = args.electrode_config if args.electrode_config is not None else offline_cfg.get("electrode_config")
    channel_positions, neighbors_lookup, neighbor_graph_source = ro.resolve_neighbor_graph(
        channel_positions=channel_positions_raw,
        n_channels=int(n_channels),
        electrode_config=electrode_config,
        neighbor_source=str(det.get("neighbor_source", "auto")),
        fallback_k=fallback_k,
        root=ROOT,
        logger=logger,
    )

    z = np.asarray(arr["embedding"])
    embed_dim = int(z.shape[1]) if z.ndim == 2 else int(model.get("embed_dim", 64))
    encode_gate_thresh, assignment_gate_thresh = resolve_reuse_gate_thresholds(
        cfg,
        source_npz=source_npz,
        logger=logger,
    )
    model_cfg = ro.build_model_config(
        model,
        embed_dim=int(embed_dim),
        encode_gate_thresh=float(encode_gate_thresh),
        assignment_gate_thresh=float(assignment_gate_thresh),
    )
    pipeline = SimpleNamespace(neighbors=neighbors_lookup, channel_positions=channel_positions)

    logger.info(
        (
            "reuse_runtime_config case=%s source_npz=%s input=%s samples=%d channels=%d fs=%.2f "
            "embedding_events=%d embedding_dim=%d init_windows=%s assign_mode=%s requested_k=%d "
            "requested_k_source=%s out_base=%s neighbor_graph_source=%s"
        ),
        case.name,
        source_npz,
        input_path,
        int(n_samples),
        int(n_channels),
        float(fs),
        int(np.asarray(arr["event_time_sample"]).shape[0]),
        int(embed_dim),
        ",".join(init_window_suffix(v) for v in online_cfg.init_window_sec),
        str(online_cfg.assign_mode),
        int(requested_k),
        str(requested_k_source),
        str(case.out_npz),
        str(neighbor_graph_source),
    )
    runtime_cfg = dict(cfg.get("runtime", {}) or {})
    candidate_filtering_disabled = bool(runtime_cfg.get("disable_candidate_filtering", False))
    encode_gate_disabled = bool(runtime_cfg.get("disable_encode_gate", False) or candidate_filtering_disabled)
    logger.info("offline_clustering_config %s", json.dumps(ro.to_jsonable(offline_cluster_cfg), sort_keys=True))
    logger.info("online_config %s", json.dumps(ro.to_jsonable(online_cfg), sort_keys=True))
    logger.info("encode_gate_disabled=%s", str(encode_gate_disabled))
    logger.info("candidate_filtering_disabled=%s", str(candidate_filtering_disabled))
    return InferenceContext(
        cfg=cfg,
        offline_cfg=offline_cfg,
        offline_cluster_cfg=offline_cluster_cfg,
        online_cfg=online_cfg,
        input_path=input_path,
        input_h5=bool(is_h5),
        reader=reader,
        fs=float(fs),
        n_samples=int(n_samples),
        n_channels=int(n_channels),
        channel_positions=channel_positions,
        weights_path=None,
        weight_payload=None,
        pipeline=pipeline,
        model_cfg=model_cfg,
        assignment_gate_thresh=float(assignment_gate_thresh),
        encode_gate_disabled=encode_gate_disabled,
        candidate_filtering_disabled=candidate_filtering_disabled,
        requested_k=int(requested_k),
        requested_k_source=str(requested_k_source),
        save_embeddings=bool(offline_cfg.get("save_embeddings", True)),
    )


def collect_embedding_outputs(ctx: InferenceContext, logger: logging.Logger) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    embed_dim = int(dict(ctx.cfg.get("model", {}) or {}).get("embed_dim", 64))
    output_parts = ro.init_output_parts(include_embedding=True, embed_dim=embed_dim)
    chunk_samples = max(1, int(round(float(ctx.pipeline.cfg.chunk_sec) * float(ctx.fs))))
    n_chunks = int(math.ceil(float(ctx.n_samples) / float(chunk_samples)))
    log_every = int(ctx.offline_cfg.get("log_every_chunks", 20))
    total_events = 0
    infer_start = time.perf_counter()
    per_chunk_ms: list[float] = []
    pre_ms: list[float] = []
    det_ms: list[float] = []
    patch_ms: list[float] = []
    model_ms: list[float] = []
    cluster_ms: list[float] = []

    if ctx.reader is not None:
        chunk_iter = ctx.reader.iter_recordings(chunk_samples=chunk_samples, start=0, stop=ctx.n_samples)
    else:
        chunk_iter = ro.iter_bin_chunks(ctx.input_path, n_channels=ctx.n_channels, chunk_samples=chunk_samples)

    for chunk_idx, x in enumerate(chunk_iter, start=1):
        out, _ = ctx.pipeline.process_chunk(x)
        new_events = 0
        if out:
            out_arr = ro.outputs_to_numpy(out, include_embedding=True, embed_dim=embed_dim)
            new_events = int(out_arr["event_time_sample"].shape[0])
            for key, value in out_arr.items():
                output_parts[key].append(value)
            total_events += new_events
        per_chunk_ms.append(float(ctx.pipeline.stats.last_latency_ms))
        pre_ms.append(float(ctx.pipeline.stats.last_preprocess_ms))
        det_ms.append(float(ctx.pipeline.stats.last_detect_ms))
        patch_ms.append(float(ctx.pipeline.stats.last_patch_ms))
        model_ms.append(float(ctx.pipeline.stats.last_model_ms))
        cluster_ms.append(float(ctx.pipeline.stats.last_cluster_ms))
        if log_every > 0 and (chunk_idx % log_every == 0 or chunk_idx == n_chunks):
            logger.info(
                (
                    "progress chunk=%d/%d chunk_samples=%d new_events=%d total_events=%d latency_ms=%.2f "
                    "stage_ms(pre=%.2f detect=%.2f patch=%.2f model=%.2f cluster=%.2f)"
                ),
                int(chunk_idx),
                int(n_chunks),
                int(x.shape[0]),
                int(new_events),
                int(total_events),
                float(ctx.pipeline.stats.last_latency_ms),
                float(ctx.pipeline.stats.last_preprocess_ms),
                float(ctx.pipeline.stats.last_detect_ms),
                float(ctx.pipeline.stats.last_patch_ms),
                float(ctx.pipeline.stats.last_model_ms),
                float(ctx.pipeline.stats.last_cluster_ms),
            )

    infer_sec = time.perf_counter() - infer_start
    arr = ro.finalize_output_parts(output_parts, include_embedding=True, embed_dim=embed_dim)
    before = int(arr["event_time_sample"].shape[0])
    arr = ro.dedup_output_events(arr)
    after = int(arr["event_time_sample"].shape[0])
    if after < before:
        logger.info("dedup outputs by (time,channel): %d -> %d removed=%d", before, after, before - after)
    runtime = {
        "infer_sec": float(infer_sec),
        "chunks": int(n_chunks),
        "events_before_dedup": int(before),
        "events_after_dedup": int(after),
        "throughput_samples_per_sec": float(ctx.n_samples / max(infer_sec, 1e-9)),
        "mean_chunk_ms": float(np.mean(per_chunk_ms)) if per_chunk_ms else 0.0,
        "p95_chunk_ms": float(np.percentile(per_chunk_ms, 95)) if per_chunk_ms else 0.0,
        "mean_preprocess_ms": float(np.mean(pre_ms)) if pre_ms else 0.0,
        "mean_detect_ms": float(np.mean(det_ms)) if det_ms else 0.0,
        "mean_patch_ms": float(np.mean(patch_ms)) if patch_ms else 0.0,
        "mean_model_ms": float(np.mean(model_ms)) if model_ms else 0.0,
        "mean_cluster_ms": float(np.mean(cluster_ms)) if cluster_ms else 0.0,
    }
    logger.info(
        (
            "embedding_collection samples=%d events=%d infer_sec=%.2f throughput_samples_per_sec=%.2f "
            "latency_mean_ms=%.2f latency_p95_ms=%.2f"
        ),
        int(ctx.n_samples),
        int(after),
        float(runtime["infer_sec"]),
        float(runtime["throughput_samples_per_sec"]),
        float(runtime["mean_chunk_ms"]),
        float(runtime["p95_chunk_ms"]),
    )
    return arr, runtime


def build_benchmark_context(ctx: InferenceContext, logger: logging.Logger) -> BenchmarkContext:
    label_cfg = dict(ctx.cfg.get("labeling", {}) or {})
    bench_cfg_raw = dict(ctx.cfg.get("benchmark", {}) or {})
    det = dict(ctx.cfg.get("detection", {}) or {})
    backend = ro.resolve_benchmark_backend(bench_cfg_raw)
    bench_cfg = ro.build_benchmark_config(
        backend,
        bench_cfg_raw=bench_cfg_raw,
        label_cfg=label_cfg,
        det_cfg=det,
    )
    ro.log_benchmark_config(logger, backend, bench_cfg)
    no_gt_cfg = bench_cfg_raw.get("no_gt", label_cfg.get("no_gt", {})) or {}
    no_gt_quality_kwargs = {
        "channel_positions": ctx.pipeline.channel_positions,
        "refractory_ms": float(no_gt_cfg.get("refractory_ms", 1.0)),
        "detect_threshold_std": float(no_gt_cfg.get("detect_threshold_std", det.get("alpha", 5.0))),
        "half_window": int(no_gt_cfg.get("half_window", 20)),
        "max_spikes_per_unit": int(no_gt_cfg.get("max_spikes_per_unit", 400)),
        "min_spikes_per_unit": int(no_gt_cfg.get("min_spikes_per_unit", 40)),
        "max_units": int(no_gt_cfg.get("max_units", 256)),
        "rng_seed": int(no_gt_cfg.get("rng_seed", 123)),
        "noise_std_max_points": int(no_gt_cfg.get("noise_std_max_points", 80000)),
    }
    gt_events = None
    gt_alignment_stats: Dict[str, Any] = {
        "gt_time_mode": ro.resolve_benchmark_gt_time_mode(bench_cfg_raw),
        "enabled": False,
        "total": 0,
    }
    if ctx.input_h5:
        try:
            gt_events = ro.load_gt_events(ctx.input_path, format="mearec")
        except Exception as exc:
            logger.warning("benchmark_with_gt skipped: failed to load GT events from %s: %s", ctx.input_path, exc)
        if gt_events is not None:
            gt_events, gt_alignment_stats = ro.align_gt_events_for_benchmark(
                gt_events,
                input_path=ctx.input_path,
                fs=float(ctx.fs),
                n_channels=int(ctx.n_channels),
                cfg=ctx.cfg,
                bench_cfg_raw=bench_cfg_raw,
                neighbors_lookup=ctx.pipeline.neighbors,
                logger=logger,
            )
    tol_samples = ro.resolve_benchmark_tol_samples(backend, bench_cfg, fs=float(ctx.fs))
    return BenchmarkContext(
        enabled=gt_events is not None,
        backend=backend,
        bench_cfg=bench_cfg,
        bench_cfg_raw=bench_cfg_raw,
        gt_events=gt_events,
        gt_alignment_stats=gt_alignment_stats,
        bench_fs=float(ctx.fs),
        tol_samples=int(tol_samples),
        no_gt_enabled=bool(no_gt_cfg.get("enabled", True)),
        no_gt_quality_kwargs=no_gt_quality_kwargs,
    )


def evaluate_arr(
    arr: Dict[str, np.ndarray],
    *,
    label_key: str,
    bench_ctx: BenchmarkContext,
    ctx: InferenceContext,
    out_path: Path,
    source_name: str,
    start_sample: int | None = None,
    include_no_gt: bool = False,
) -> Dict[str, Any]:
    eval_arr = arr
    gt_events = bench_ctx.gt_events
    if start_sample is not None:
        pred_mask = np.asarray(arr["event_time_sample"], dtype=np.int64) >= int(start_sample)
        eval_arr = subset_arr_by_mask(arr, pred_mask)
        if gt_events is not None:
            gt_mask = gt_events.times_samples >= int(start_sample)
            gt_events = gt_events.subset(gt_mask)

    with_gt = None
    no_gt = None
    if gt_events is not None:
        pred = ro.build_pred_event_set(
            eval_arr,
            label_key=label_key,
            fs=float(ctx.fs),
            out_path=out_path,
            source_name=source_name,
        )
        with_gt = ro.eval_mod.benchmark_with_gt(
            pred,
            gt_events,
            fs=float(bench_ctx.bench_fs),
            cfg=bench_ctx.bench_cfg,
            h5_path=ctx.input_path if include_no_gt and bench_ctx.no_gt_enabled and start_sample is None else None,
            quality_kwargs=(
                bench_ctx.no_gt_quality_kwargs
                if include_no_gt and bench_ctx.no_gt_enabled and start_sample is None
                else None
            ),
        )
        no_gt = with_gt.get("no_gt") if isinstance(with_gt, dict) else None
    gate_summary = ro.build_gate_summary(
        eval_arr,
        assignment_gate_thresh=float(ctx.assignment_gate_thresh),
        gt_events=gt_events,
        tol_samples=int(bench_ctx.tol_samples) if gt_events is not None else None,
    )
    summary = ro.build_eval_summary(with_gt, no_gt, gate_summary)
    summary = enrich_eval_summary(summary, with_gt=with_gt)
    return {
        "with_gt": with_gt,
        "no_gt": no_gt,
        "gate_summary": gate_summary,
        "summary": summary,
        "quick_summary": ro.build_quick_eval_summary(summary),
        "start_sample": None if start_sample is None else int(start_sample),
        "n_pred_events": int(np.asarray(eval_arr["event_time_sample"]).shape[0]),
        "n_gt_events": int(gt_events.n_events) if gt_events is not None else 0,
    }


def compact_summary(summary: Dict[str, Any] | None) -> Dict[str, Any]:
    src = summary if isinstance(summary, dict) else {}
    return {key: finite_or_none(src.get(key)) for key in METRIC_KEYS}


def metric_delta(lhs: Dict[str, Any] | None, rhs: Dict[str, Any] | None) -> Dict[str, Any]:
    left = lhs if isinstance(lhs, dict) else {}
    right = rhs if isinstance(rhs, dict) else {}
    out: Dict[str, Any] = {}
    for key in METRIC_KEYS:
        a = finite_or_none(left.get(key))
        b = finite_or_none(right.get(key))
        out[key] = None if a is None or b is None else float(a - b)
    return out


def enrich_eval_summary(summary: Dict[str, Any], *, with_gt: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out = dict(summary)
    source_summary = dict(with_gt.get("summary", {})) if isinstance(with_gt, dict) else {}
    for key in ("sort_f1_micro_mapped", "sort_f1_macro_mapped"):
        value = finite_or_none(source_summary.get(key))
        if value is not None:
            out[key] = float(value)
    if finite_or_none(out.get("f1_1to1")) is None:
        precision = finite_or_none(out.get("precision_1to1"))
        recall = finite_or_none(out.get("recall_1to1"))
        if precision is not None and recall is not None and precision + recall > 0.0:
            out["f1_1to1"] = float(2.0 * precision * recall / (precision + recall))
    return out


def online_primary_f1(summary: Dict[str, Any] | None) -> tuple[float | None, str]:
    src = summary if isinstance(summary, dict) else {}
    for key in ("sort_f1_micro_mapped", "f1_1to1"):
        value = finite_or_none(src.get(key))
        if value is not None:
            return float(value), key
    precision = finite_or_none(src.get("precision_1to1"))
    recall = finite_or_none(src.get("recall_1to1"))
    if precision is not None and recall is not None and precision + recall > 0.0:
        return float(2.0 * precision * recall / (precision + recall)), "computed_pr_f1"
    return None, "missing"


def nan_if_none(value: Any) -> float:
    return float("nan") if value is None else float(value)


def run_full_offline_baseline(
    arr: Dict[str, np.ndarray],
    *,
    ctx: InferenceContext,
    bench_ctx: BenchmarkContext,
    logger: logging.Logger,
) -> Dict[str, Any]:
    methods = list(ctx.offline_cluster_cfg.get("methods", [ctx.offline_cluster_cfg.get("method", "hdbscan")]))
    fail_on_method_error = bool(ctx.offline_cluster_cfg.get("fail_on_method_error", False))
    required = {
        str(value).strip().lower()
        for value in ctx.offline_cluster_cfg.get("required_methods", [])
        if str(value).strip()
    }
    results: list[Dict[str, Any]] = []
    for method in methods:
        method_cfg = dict(ctx.offline_cluster_cfg)
        method_cfg["method"] = str(method)
        method_cfg["fs"] = float(ctx.fs)
        logger.info("offline_baseline_candidate_start method=%s", method)
        try:
            clustered, diag = offline_cluster_outputs_nogt(
                arr,
                cfg=method_cfg,
                requested_k=int(ctx.requested_k),
                requested_k_source=str(ctx.requested_k_source),
                assignment_gate_thresh=float(ctx.assignment_gate_thresh),
                disable_candidate_filtering=bool(getattr(ctx, "candidate_filtering_disabled", False)),
                disable_encode_gate=bool(getattr(ctx, "encode_gate_disabled", False)),
            )
            score = (float(diag.get("hdbscan_nogt_score", float("-inf"))), 0.0, 0.0)
        except Exception as exc:
            logger.warning("offline_baseline_candidate_failed method=%s error=%s", method, exc)
            if fail_on_method_error or str(method).strip().lower() in required:
                raise
            continue
        logger.info(
            "offline_baseline_candidate_result method=%s score=%.6f actual_clusters=%d clustered_events=%d",
            str(method),
            float(score[0]),
            int(diag.get("actual_clusters", 0)),
            int(diag.get("clustered_events", 0)),
        )
        results.append({"method": str(method), "arr": clustered, "diag": diag, "score": score})
    if not results:
        raise RuntimeError("all full-offline baseline clustering methods failed")
    best = max(results, key=lambda item: item["score"])
    best["eval_full"] = evaluate_arr(
        best["arr"],
        label_key="unit_id_post" if "unit_id_post" in best["arr"] else "unit_id",
        bench_ctx=bench_ctx,
        ctx=ctx,
        out_path=ctx.input_path.with_suffix(".offline_baseline.npz"),
        source_name="mysort_run_online_offline_baseline",
        include_no_gt=True,
    )
    logger.info(
        "offline_baseline_best method=%s score=%.6f",
        str(best["method"]),
        float(best["score"][0]),
    )
    return best


def run_online_candidate(
    arr: Dict[str, np.ndarray],
    *,
    ctx: InferenceContext,
    bench_ctx: BenchmarkContext,
    init_window_sec: float,
    method: str,
    out_path: Path,
    baseline: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    times = np.asarray(arr["event_time_sample"], dtype=np.int64)
    cutoff = int(round(float(init_window_sec) * float(ctx.fs)))
    init_idx = np.flatnonzero(times < cutoff).astype(np.int64, copy=False)
    if init_idx.size == 0:
        raise RuntimeError(f"init_window_sec={init_window_sec} has no detected events")
    init_arr = subset_arr_by_indices(arr, init_idx)
    method_cfg = dict(ctx.offline_cluster_cfg)
    method_cfg["method"] = str(method)
    method_cfg["fs"] = float(ctx.fs)
    clustered_init, init_diag = offline_cluster_outputs_nogt(
        init_arr,
        cfg=method_cfg,
        requested_k=int(ctx.requested_k),
        requested_k_source=str(ctx.requested_k_source),
        assignment_gate_thresh=float(ctx.assignment_gate_thresh),
        disable_candidate_filtering=bool(getattr(ctx, "candidate_filtering_disabled", False)),
        disable_encode_gate=bool(getattr(ctx, "encode_gate_disabled", False)),
    )
    init_labels = np.asarray(clustered_init.get("unit_id_post", clustered_init["unit_id"]), dtype=np.int32)
    units = build_units_from_labels(
        clustered_init,
        init_labels,
        global_indices=init_idx,
        center_trim_quantile=float(ctx.offline_cluster_cfg.get("graph_center_trim_quantile", 0.90)),
    )
    online_arr, online_diag = apply_online_assignment(
        arr,
        init_idx=init_idx,
        init_labels=init_labels,
        units=units,
        ctx=ctx,
        init_window_sec=float(init_window_sec),
    )
    full_eval = evaluate_arr(
        online_arr,
        label_key="unit_id",
        bench_ctx=bench_ctx,
        ctx=ctx,
        out_path=out_path,
        source_name=f"mysort_run_online_init{init_window_suffix(init_window_sec)}_{method}_full",
        include_no_gt=False,
    )
    online_eval = evaluate_arr(
        online_arr,
        label_key="unit_id",
        bench_ctx=bench_ctx,
        ctx=ctx,
        out_path=out_path,
        source_name=f"mysort_run_online_init{init_window_suffix(init_window_sec)}_{method}_online_only",
        start_sample=cutoff,
        include_no_gt=False,
    )
    baseline_online_eval = evaluate_arr(
        baseline["arr"],
        label_key="unit_id_post" if "unit_id_post" in baseline["arr"] else "unit_id",
        bench_ctx=bench_ctx,
        ctx=ctx,
        out_path=out_path.with_name(f"{out_path.stem}_offline_baseline{out_path.suffix}"),
        source_name=f"mysort_run_online_baseline_online_only_init{init_window_suffix(init_window_sec)}",
        start_sample=cutoff,
        include_no_gt=False,
    )
    diag = dict(online_diag)
    diag.update(
        {
            "init_clustering": init_diag,
            "init_method": str(method),
            "baseline_method": str(baseline.get("method")),
            "requested_k": int(ctx.requested_k),
            "requested_k_source": str(ctx.requested_k_source),
        }
    )
    return {
        "method": str(method),
        "arr": online_arr,
        "init_diag": init_diag,
        "online_diag": diag,
        "full_eval": full_eval,
        "online_eval": online_eval,
        "baseline_online_eval": baseline_online_eval,
        "score": (float(init_diag.get("hdbscan_nogt_score", float("-inf"))), 0.0, 0.0),
    }


def save_online_result(
    *,
    result: Dict[str, Any],
    baseline: Dict[str, Any],
    ctx: InferenceContext,
    bench_ctx: BenchmarkContext,
    out_path: Path,
    runtime: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    arr = result["arr"]
    save_arr = {key: np.asarray(value) for key, value in arr.items()}
    if not bool(ctx.save_embeddings):
        save_arr = dict(save_arr)
        save_arr.pop("embedding", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save_arr)

    baseline_full_eval = baseline["eval_full"]
    baseline_online_eval = result["baseline_online_eval"]
    full_summary = result["full_eval"].get("summary", {})
    online_summary = result["online_eval"].get("summary", {})
    baseline_full_summary = baseline_full_eval.get("summary", {})
    baseline_online_summary = baseline_online_eval.get("summary", {})
    payload = {
        "quick_summary": ro.build_quick_eval_summary(full_summary),
        "backend": bench_ctx.backend,
        "benchmark_config": bench_ctx.bench_cfg,
        "benchmark_config_raw": bench_ctx.bench_cfg_raw,
        "input_h5": ctx.input_path if ctx.input_h5 else None,
        "input_bin": None if ctx.input_h5 else ctx.input_path,
        "prediction_npz": out_path,
        "label_key": "unit_id",
        "init_window_sec": float(result["online_diag"]["init_window_sec"]),
        "requested_k_source": str(ctx.requested_k_source),
        "hdbscan_selection_metric": result["init_diag"].get("hdbscan_selection_metric", HDBSCAN_NOGT_SELECTION_RETENTION),
        "hdbscan_nogt_selected_params": result["init_diag"].get("hdbscan_nogt_selected_params"),
        "hdbscan_nogt_score": result["init_diag"].get("hdbscan_nogt_score"),
        "hdbscan_nogt_candidates": result["init_diag"].get("hdbscan_nogt_candidates", []),
        "hdbscan_nogt_selection": result["init_diag"].get("hdbscan_nogt_selection", {}),
        "selected_cluster_count": result["init_diag"].get("selected_cluster_count"),
        "candidate_count": result["init_diag"].get("candidate_count"),
        "candidate_k_min": result["init_diag"].get("candidate_k_min"),
        "candidate_k_max": result["init_diag"].get("candidate_k_max"),
        "near_best_candidate_count": result["init_diag"].get("near_best_candidate_count"),
        "near_best_k_min": result["init_diag"].get("near_best_k_min"),
        "near_best_k_max": result["init_diag"].get("near_best_k_max"),
        "k_confidence": result["init_diag"].get("k_confidence"),
        "selection_reason": result["init_diag"].get("selection_reason"),
        "premerge_clusters": result["init_diag"].get("premerge_clusters"),
        "postmerge_clusters": result["init_diag"].get("postmerge_clusters"),
        "hdbscan_online_profile": result["init_diag"].get("hdbscan_online_profile"),
        "hdbscan_merge_enabled": result["init_diag"].get("hdbscan_merge_enabled"),
        "hdbscan_merge_profile": result["init_diag"].get("hdbscan_merge_profile"),
        "hdbscan_merge_count": result["init_diag"].get("hdbscan_merge_count"),
        "hdbscan_merge_pairs": result["init_diag"].get("hdbscan_merge_pairs", []),
        "hdbscan_merge_confidence": result["init_diag"].get("hdbscan_merge_confidence"),
        "hdbscan_merge_nogt_k_stop": result["init_diag"].get("nogt_k_stop", {}),
        "encode_gate_disabled": result["init_diag"].get("encode_gate_disabled", False),
        "encoded_events": result["init_diag"].get("encoded_events"),
        "gate_pass_events": result["init_diag"].get("gate_pass_events"),
        "encoded_gate_pass_events": result["init_diag"].get("encoded_gate_pass_events"),
        "candidate_filtering_disabled": result["init_diag"].get("candidate_filtering_disabled", False),
        "candidate_events": result["init_diag"].get("candidate_events"),
        "nogt_k_stop_enabled": result["init_diag"].get("nogt_k_stop_enabled"),
        "nogt_k_stop_target": result["init_diag"].get("nogt_k_stop_target"),
        "nogt_k_stop_reason": result["init_diag"].get("nogt_k_stop_reason"),
        "nogt_k_stop_pool_k_min": result["init_diag"].get("nogt_k_stop_pool_k_min"),
        "nogt_k_stop_pool_k_max": result["init_diag"].get("nogt_k_stop_pool_k_max"),
        "full_record": result["full_eval"],
        "online_only": result["online_eval"],
        "offline_baseline": {
            "method": baseline.get("method"),
            "diag": baseline.get("diag"),
            "summary": baseline_full_summary,
            "online_only_summary": baseline_online_summary,
            "full_record": baseline_full_eval,
            "online_only": baseline_online_eval,
        },
        "delta_vs_offline": {
            "full_record_minus_offline_full": metric_delta(full_summary, baseline_full_summary),
            "online_only_minus_offline_online_only": metric_delta(online_summary, baseline_online_summary),
            "online_only_minus_offline_full": metric_delta(online_summary, baseline_full_summary),
        },
        "online_diagnostics": result["online_diag"],
        "runtime": runtime,
        "gt_alignment": bench_ctx.gt_alignment_stats,
    }
    report = report_path(out_path)
    ro.save_benchmark_report(report, payload)
    logger.info("saved online outputs: %s (%d events)", out_path, int(np.asarray(arr["event_time_sample"]).shape[0]))
    logger.info("online_report=%s", report)
    delta_online = metric_delta(online_summary, baseline_online_summary)
    logger.info(
        (
            "online_summary init=%.1fs method=%s full_acc=%.4f full_precision=%.4f full_f1=%.4f "
            "online_acc=%.4f online_precision=%.4f online_f1=%.4f "
            "delta_online_acc=%.4f delta_online_precision=%.4f delta_online_f1=%.4f"
        ),
        float(result["online_diag"]["init_window_sec"]),
        str(result["method"]),
        float(full_summary.get("acc_1to1_global", float("nan"))),
        float(full_summary.get("precision_1to1", float("nan"))),
        float(full_summary.get("f1_1to1", float("nan"))),
        float(online_summary.get("acc_1to1_global", float("nan"))),
        float(online_summary.get("precision_1to1", float("nan"))),
        float(online_summary.get("f1_1to1", float("nan"))),
        nan_if_none(delta_online.get("acc_1to1_global")),
        nan_if_none(delta_online.get("precision_1to1")),
        nan_if_none(delta_online.get("f1_1to1")),
    )


def run_online_window(
    arr: Dict[str, np.ndarray],
    *,
    ctx: InferenceContext,
    bench_ctx: BenchmarkContext,
    baseline: Dict[str, Any],
    init_window_sec: float,
    out_path: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    methods = list(ctx.offline_cluster_cfg.get("methods", [ctx.offline_cluster_cfg.get("method", "hdbscan")]))
    fail_on_method_error = bool(ctx.offline_cluster_cfg.get("fail_on_method_error", False))
    required = {
        str(value).strip().lower()
        for value in ctx.offline_cluster_cfg.get("required_methods", [])
        if str(value).strip()
    }
    candidates: list[Dict[str, Any]] = []
    for method in methods:
        logger.info("online_candidate_start init=%.3fs method=%s", float(init_window_sec), str(method))
        try:
            candidate = run_online_candidate(
                arr,
                ctx=ctx,
                bench_ctx=bench_ctx,
                init_window_sec=float(init_window_sec),
                method=str(method),
                out_path=out_path,
                baseline=baseline,
                logger=logger,
            )
        except Exception as exc:
            logger.warning("online_candidate_failed init=%.3fs method=%s error=%s", float(init_window_sec), str(method), exc)
            if fail_on_method_error or str(method).strip().lower() in required:
                raise
            continue
        candidates.append(candidate)
        selection_metric = candidate["init_diag"].get("hdbscan_selection_metric", HDBSCAN_NOGT_SELECTION_RETENTION)
        logger.info(
            (
                "online_candidate_result init=%.3fs method=%s selection_metric=%s score=%.6f "
                "units=%d accepted=%d updates=%d"
            ),
            float(init_window_sec),
            str(method),
            selection_metric,
            float(candidate["score"][0]),
            int(candidate["online_diag"].get("units_initialized", 0)),
            int(candidate["online_diag"].get("assignment_accepted", 0)),
            int(candidate["online_diag"].get("updates", 0)),
        )
    if not candidates:
        raise RuntimeError(f"all online init clustering methods failed for init_window_sec={init_window_sec}")
    best = max(candidates, key=lambda item: item["score"])
    logger.info(
        "online_best init=%.3fs method=%s score=%.6f",
        float(init_window_sec),
        str(best["method"]),
        float(best["score"][0]),
    )
    return best


def infer_identity_arrays(arr, ctx, init_window_sec):
    """Initialize from the unlabeled prefix, then assign in timestamp order; no GT reads."""
    cutoff = int(round(init_window_sec * ctx.fs))
    times = np.asarray(arr["event_time_sample"], dtype=np.int64)
    init_idx = np.flatnonzero((times >= 0) & (times < cutoff))
    init_arr = subset_arr_by_indices(arr, init_idx)
    init_cfg = dict(ctx.offline_cluster_cfg, method="hdbscan", fs=float(ctx.fs))
    clustered, init_diag = offline_cluster_outputs_nogt(
        init_arr, cfg=init_cfg, requested_k=0, requested_k_source="no_gt",
        assignment_gate_thresh=ctx.assignment_gate_thresh)
    labels = np.asarray(clustered.get("unit_id_post", clustered["unit_id"]), dtype=np.int32)
    units = build_units_from_labels(clustered, labels, global_indices=init_idx,
        center_trim_quantile=float(ctx.offline_cluster_cfg.get("graph_center_trim_quantile", 0.90)))
    result, assignment_diag = apply_online_assignment(arr, init_idx=init_idx,
        init_labels=labels, units=units, ctx=ctx, init_window_sec=init_window_sec)
    return result, {"initialization": init_diag, "assignment": assignment_diag}


def run_inference_only_case(case, cfg, args):
    windows = resolve_online_config(cfg, args.init_window_sec).init_window_sec
    if not args.force:
        for seconds in windows:
            output = online_output_path(case.out_npz, seconds)
            if output.exists() or output.with_suffix(".run.json").exists():
                raise FileExistsError(f"Output already exists: {output}; choose a new output or use --force")
    logger = build_logger(case.log_file)
    ctx = prepare_context(case, cfg, args, logger)
    arr, runtime = collect_embedding_outputs(ctx, logger)
    for seconds in ctx.online_cfg.init_window_sec:
        output = online_output_path(case.out_npz, seconds)
        metadata_path = output.with_suffix(".run.json")
        if (output.exists() or metadata_path.exists()) and not args.force:
            raise FileExistsError(f"Output already exists: {output}; choose a new output or use --force")
        result, diagnostics = infer_identity_arrays(arr, ctx, seconds)
        if not ctx.save_embeddings:
            result.pop("embedding", None)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **result)
        write_json(metadata_path, ro.to_jsonable({
            "schema_version": 1, "recording": case.name, "fs": ctx.fs,
            "arguments": vars(args),
            "initialization_s": seconds, "config": cfg,
            "resolved_online_config": vars(ctx.online_cfg),
            "weights_sha256": sha256(ctx.weights_path),
            "predictions_sha256": sha256(output),
            "environment": environment(), "runtime": runtime,
            "diagnostics": diagnostics, "gt_used_for_inference": False}))
        logger.info("predictions_saved=%s; score using scripts/evaluate.py", output)
    return 0


def run_case(case: Case, cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    if cfg.get("online", {}).get("inference_only", False):
        return run_inference_only_case(case, cfg, args)
    total_start = time.perf_counter()
    logger = build_logger(case.log_file)
    logger.info("launch online inference")
    logger.info("argv=%s", " ".join(sys.argv))
    logger.info("python_executable=%s", sys.executable)
    logger.info("cwd=%s", os.getcwd())
    logger.info("case=%s config=%s", case.name, args.config)
    ctx = prepare_context(case, cfg, args, logger)
    arr, runtime = collect_embedding_outputs(ctx, logger)
    bench_ctx = build_benchmark_context(ctx, logger)
    baseline = run_full_offline_baseline(arr, ctx=ctx, bench_ctx=bench_ctx, logger=logger)

    for init_sec in ctx.online_cfg.init_window_sec:
        out_path = online_output_path(case.out_npz, float(init_sec))
        result = run_online_window(
            arr,
            ctx=ctx,
            bench_ctx=bench_ctx,
            baseline=baseline,
            init_window_sec=float(init_sec),
            out_path=out_path,
            logger=logger,
        )
        save_online_result(
            result=result,
            baseline=baseline,
            ctx=ctx,
            bench_ctx=bench_ctx,
            out_path=out_path,
            runtime=runtime,
            logger=logger,
        )

    total_sec = time.perf_counter() - total_start
    logger.info(
        "summary case=%s samples=%d events=%d total_sec=%.2f log_file=%s",
        case.name,
        int(ctx.n_samples),
        int(np.asarray(arr["event_time_sample"]).shape[0]),
        float(total_sec),
        case.log_file,
    )
    return 0


def reuse_output_case(case: Case, *, comparison_name: str, embedding_set: str) -> Case:
    subdir = comparison_subdir(comparison_name)
    out_base = ROOT / "data" / "cache" / subdir / embedding_set / f"{case.name}.npz"
    log_file = ROOT / "data" / "logs" / subdir / embedding_set / f"{case.name}_online.log"
    return Case(
        name=case.name,
        input_h5=case.input_h5,
        input_bin=case.input_bin,
        n_channels=case.n_channels,
        out_npz=out_base,
        log_file=log_file,
    )


def reports_exist_for_reuse_case(case: Case, cfg: Dict[str, Any], init_override: str | None) -> bool:
    reuse_cfg = force_reuse_online_cfg(cfg)
    online_cfg = resolve_online_config(reuse_cfg, init_override or "20")
    return all(report_path(online_output_path(case.out_npz, sec)).exists() for sec in online_cfg.init_window_sec)


def reuse_report_entries_for_case(
    case: Case,
    cfg: Dict[str, Any],
    init_override: str | None,
    *,
    embedding_set: str,
    source_npz: Path,
) -> list[Dict[str, Any]]:
    reuse_cfg = force_reuse_online_cfg(cfg)
    online_cfg = resolve_online_config(reuse_cfg, init_override or "20")
    entries: list[Dict[str, Any]] = []
    for init_sec in online_cfg.init_window_sec:
        out_path = online_output_path(case.out_npz, float(init_sec))
        entries.append(
            {
                "embedding_set": str(embedding_set),
                "job": str(case.name),
                "init_window_sec": float(init_sec),
                "source_npz": source_npz,
                "out_npz": out_path,
                "report_json": report_path(out_path),
            }
        )
    return entries


def run_reuse_case(
    case: Case,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    *,
    source_npz: Path,
    embedding_set: str,
) -> list[Dict[str, Any]]:
    total_start = time.perf_counter()
    logger = build_logger(case.log_file)
    logger.info("launch online reuse evaluation")
    logger.info("argv=%s", " ".join(sys.argv))
    logger.info("python_executable=%s", sys.executable)
    logger.info("cwd=%s", os.getcwd())
    logger.info("case=%s config=%s source_npz=%s", case.name, args.config, source_npz)
    load_start = time.perf_counter()
    arr = load_npz_dict(source_npz)
    validate_reuse_embedding_arrays(arr, path=source_npz)
    load_sec = time.perf_counter() - load_start
    ctx = prepare_reuse_context(case, cfg, args, logger, source_npz=source_npz, arr=arr)
    bench_ctx = build_benchmark_context(ctx, logger)
    baseline = run_full_offline_baseline(arr, ctx=ctx, bench_ctx=bench_ctx, logger=logger)
    runtime = {
        "reused_embeddings": True,
        "source_npz": str(source_npz),
        "embedding_load_sec": float(load_sec),
        "events_before_dedup": int(np.asarray(arr["event_time_sample"]).shape[0]),
        "events_after_dedup": int(np.asarray(arr["event_time_sample"]).shape[0]),
        "embedding_dim": int(np.asarray(arr["embedding"]).shape[1]),
        "infer_sec": 0.0,
    }

    for init_sec in ctx.online_cfg.init_window_sec:
        out_path = online_output_path(case.out_npz, float(init_sec))
        result = run_online_window(
            arr,
            ctx=ctx,
            bench_ctx=bench_ctx,
            baseline=baseline,
            init_window_sec=float(init_sec),
            out_path=out_path,
            logger=logger,
        )
        save_online_result(
            result=result,
            baseline=baseline,
            ctx=ctx,
            bench_ctx=bench_ctx,
            out_path=out_path,
            runtime=runtime,
            logger=logger,
        )

    total_sec = time.perf_counter() - total_start
    logger.info(
        "summary case=%s source_npz=%s samples=%d events=%d total_sec=%.2f log_file=%s",
        case.name,
        source_npz,
        int(ctx.n_samples),
        int(np.asarray(arr["event_time_sample"]).shape[0]),
        float(total_sec),
        case.log_file,
    )
    return reuse_report_entries_for_case(
        case,
        cfg,
        args.init_window_sec,
        embedding_set=embedding_set,
        source_npz=source_npz,
    )


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return None if not finite else float(np.mean(np.asarray(finite, dtype=np.float64)))


def _median(values: Sequence[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return None if not finite else float(np.median(np.asarray(finite, dtype=np.float64)))


def _fmt_metric(value: Any, digits: int = 4) -> str:
    numeric = finite_or_none(value)
    return "" if numeric is None else f"{numeric:.{int(digits)}f}"


def _summary_metric(summary: Dict[str, Any], key: str) -> float | None:
    return finite_or_none(summary.get(key))


def comparison_row_from_report(entry: Dict[str, Any]) -> Dict[str, Any]:
    report_json = Path(entry["report_json"])
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    online_summary = dict(payload.get("online_only", {}).get("summary", {}) or {})
    full_summary = dict(payload.get("full_record", {}).get("summary", {}) or {})
    baseline_online_summary = dict(
        payload.get("offline_baseline", {}).get("online_only_summary", {}) or {}
    )
    delta_online = dict(payload.get("delta_vs_offline", {}).get("online_only_minus_offline_online_only", {}) or {})
    diag = dict(payload.get("online_diagnostics", {}) or {})
    online_f1, online_f1_source = online_primary_f1(online_summary)
    baseline_f1, baseline_f1_source = online_primary_f1(baseline_online_summary)
    delta_f1 = None if online_f1 is None or baseline_f1 is None else float(online_f1 - baseline_f1)
    return {
        "row_type": "job",
        "embedding_set": str(entry["embedding_set"]),
        "job": str(entry["job"]),
        "init_window_sec": float(entry["init_window_sec"]),
        "online_f1": online_f1,
        "online_f1_source": online_f1_source,
        "online_acc_1to1_global": _summary_metric(online_summary, "acc_1to1_global"),
        "online_precision_1to1": _summary_metric(online_summary, "precision_1to1"),
        "online_recall_1to1": _summary_metric(online_summary, "recall_1to1"),
        "online_sorted_score_1to1": _summary_metric(online_summary, "sorted_score_1to1"),
        "full_f1": online_primary_f1(full_summary)[0],
        "full_acc_1to1_global": _summary_metric(full_summary, "acc_1to1_global"),
        "baseline_online_f1": baseline_f1,
        "baseline_online_f1_source": baseline_f1_source,
        "baseline_online_acc_1to1_global": _summary_metric(baseline_online_summary, "acc_1to1_global"),
        "delta_online_f1": delta_f1,
        "delta_online_acc_1to1_global": _summary_metric(delta_online, "acc_1to1_global"),
        "delta_online_precision_1to1": _summary_metric(delta_online, "precision_1to1"),
        "delta_online_recall_1to1": _summary_metric(delta_online, "recall_1to1"),
        "assign_mode": str(diag.get("assign_mode", "")),
        "units_initialized": finite_or_none(diag.get("units_initialized")),
        "prototype_count": finite_or_none(diag.get("prototype_count")),
        "assignment_candidates": finite_or_none(diag.get("assignment_candidates")),
        "assignment_accepted": finite_or_none(diag.get("assignment_accepted")),
        "assignment_accept_rate": finite_or_none(diag.get("assignment_accept_rate")),
        "updates": finite_or_none(diag.get("updates")),
        "winner_for_job": "",
        "job_rank": "",
        "source_npz": str(entry["source_npz"]),
        "out_npz": str(entry["out_npz"]),
        "report_json": str(entry["report_json"]),
    }


def aggregate_comparison_rows(job_rows: list[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], str]:
    by_job: Dict[str, list[Dict[str, Any]]] = {}
    for row in job_rows:
        by_job.setdefault(str(row["job"]), []).append(row)
    wins: Dict[str, int] = {}
    for job, rows in by_job.items():
        ranked = sorted(
            rows,
            key=lambda item: (
                float(item["online_f1"]) if item["online_f1"] is not None else float("-inf"),
                float(item["online_acc_1to1_global"])
                if item["online_acc_1to1_global"] is not None
                else float("-inf"),
            ),
            reverse=True,
        )
        winner = str(ranked[0]["embedding_set"]) if ranked else ""
        if winner:
            wins[winner] = wins.get(winner, 0) + 1
        for rank, row in enumerate(ranked, start=1):
            row["winner_for_job"] = winner
            row["job_rank"] = int(rank)

    by_set: Dict[str, list[Dict[str, Any]]] = {}
    for row in job_rows:
        by_set.setdefault(str(row["embedding_set"]), []).append(row)
    aggregate_rows: list[Dict[str, Any]] = []
    for embedding_set, rows in sorted(by_set.items()):
        f1_values = [float(row["online_f1"]) for row in rows if row["online_f1"] is not None]
        acc_values = [
            float(row["online_acc_1to1_global"])
            for row in rows
            if row["online_acc_1to1_global"] is not None
        ]
        precision_values = [
            float(row["online_precision_1to1"])
            for row in rows
            if row["online_precision_1to1"] is not None
        ]
        recall_values = [
            float(row["online_recall_1to1"])
            for row in rows
            if row["online_recall_1to1"] is not None
        ]
        delta_f1_values = [float(row["delta_online_f1"]) for row in rows if row["delta_online_f1"] is not None]
        aggregate_rows.append(
            {
                "row_type": "aggregate",
                "embedding_set": embedding_set,
                "job": "ALL",
                "init_window_sec": rows[0].get("init_window_sec", "") if rows else "",
                "online_f1": _mean(f1_values),
                "online_f1_median": _median(f1_values),
                "online_acc_1to1_global": _mean(acc_values),
                "online_precision_1to1": _mean(precision_values),
                "online_recall_1to1": _mean(recall_values),
                "delta_online_f1": _mean(delta_f1_values),
                "wins": int(wins.get(embedding_set, 0)),
                "n_jobs": int(len(rows)),
                "winner_for_job": "",
                "job_rank": "",
                "source_npz": "",
                "out_npz": "",
                "report_json": "",
            }
        )
    ranked_aggregate = sorted(
        aggregate_rows,
        key=lambda item: (
            float(item["online_f1"]) if item["online_f1"] is not None else float("-inf"),
            int(item.get("wins", 0)),
        ),
        reverse=True,
    )
    overall_winner = str(ranked_aggregate[0]["embedding_set"]) if ranked_aggregate else ""
    return aggregate_rows, overall_winner


def write_comparison_summary(
    entries: Sequence[Dict[str, Any]],
    *,
    comparison_name: str,
    logger: logging.Logger | None = None,
) -> tuple[Path, Path, str]:
    job_rows: list[Dict[str, Any]] = []
    for entry in entries:
        report_json = Path(entry["report_json"])
        if report_json.exists():
            job_rows.append(comparison_row_from_report(entry))
        elif logger is not None:
            logger.warning("comparison_report_missing report=%s", report_json)
    aggregate_rows, overall_winner = aggregate_comparison_rows(job_rows)

    subdir = comparison_subdir(comparison_name)
    csv_path = ROOT / "data" / "reports" / f"{subdir.name}_summary.csv"
    md_path = ROOT / "data" / "reports" / f"{subdir.name}_summary.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "row_type",
        "embedding_set",
        "job",
        "init_window_sec",
        "online_f1",
        "online_f1_median",
        "online_f1_source",
        "online_acc_1to1_global",
        "online_precision_1to1",
        "online_recall_1to1",
        "online_sorted_score_1to1",
        "full_f1",
        "full_acc_1to1_global",
        "baseline_online_f1",
        "baseline_online_f1_source",
        "baseline_online_acc_1to1_global",
        "delta_online_f1",
        "delta_online_acc_1to1_global",
        "delta_online_precision_1to1",
        "delta_online_recall_1to1",
        "assign_mode",
        "units_initialized",
        "prototype_count",
        "assignment_candidates",
        "assignment_accepted",
        "assignment_accept_rate",
        "updates",
        "wins",
        "n_jobs",
        "winner_for_job",
        "job_rank",
        "source_npz",
        "out_npz",
        "report_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(aggregate_rows, key=lambda item: str(item["embedding_set"])):
            writer.writerow({field: row.get(field, "") for field in fields})
        for row in sorted(job_rows, key=lambda item: (str(item["job"]), int(item.get("job_rank") or 999))):
            writer.writerow({field: row.get(field, "") for field in fields})

    md_lines = [
        f"# {subdir.name}",
        "",
        f"- Overall winner by mean online F1: `{overall_winner}`" if overall_winner else "- Overall winner: unavailable",
        "- Ranking metric: online-only F1 after 20s warmup (`sort_f1_micro_mapped`, fallback `f1_1to1`, fallback computed precision/recall F1).",
        "",
        "## Aggregate",
        "",
        "| Embedding set | Jobs | Wins | Mean online F1 | Median online F1 | Mean acc | Mean precision | Mean recall | Mean delta F1 vs offline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        aggregate_rows,
        key=lambda item: float(item["online_f1"]) if item["online_f1"] is not None else float("-inf"),
        reverse=True,
    ):
        md_lines.append(
            "| {embedding_set} | {n_jobs} | {wins} | {mean_f1} | {median_f1} | {acc} | {precision} | {recall} | {delta_f1} |".format(
                embedding_set=row.get("embedding_set", ""),
                n_jobs=row.get("n_jobs", ""),
                wins=row.get("wins", ""),
                mean_f1=_fmt_metric(row.get("online_f1")),
                median_f1=_fmt_metric(row.get("online_f1_median")),
                acc=_fmt_metric(row.get("online_acc_1to1_global")),
                precision=_fmt_metric(row.get("online_precision_1to1")),
                recall=_fmt_metric(row.get("online_recall_1to1")),
                delta_f1=_fmt_metric(row.get("delta_online_f1")),
            )
        )
    md_lines.extend(
        [
            "",
            "## Per Recording",
            "",
            "| Job | Winner | Embedding set | Rank | Online F1 | Online acc | Precision | Recall | Delta F1 vs offline | Accept rate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(job_rows, key=lambda item: (str(item["job"]), int(item.get("job_rank") or 999))):
        md_lines.append(
            "| {job} | {winner} | {embedding_set} | {rank} | {f1} | {acc} | {precision} | {recall} | {delta_f1} | {accept_rate} |".format(
                job=row.get("job", ""),
                winner=row.get("winner_for_job", ""),
                embedding_set=row.get("embedding_set", ""),
                rank=row.get("job_rank", ""),
                f1=_fmt_metric(row.get("online_f1")),
                acc=_fmt_metric(row.get("online_acc_1to1_global")),
                precision=_fmt_metric(row.get("online_precision_1to1")),
                recall=_fmt_metric(row.get("online_recall_1to1")),
                delta_f1=_fmt_metric(row.get("delta_online_f1")),
                accept_rate=_fmt_metric(row.get("assignment_accept_rate")),
            )
        )
    md_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- CSV: `{csv_path}`",
            f"- Reports/NPZ root: `{ROOT / 'data' / 'cache' / subdir}`",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if logger is not None:
        logger.info("comparison_summary_csv=%s", csv_path)
        logger.info("comparison_summary_md=%s", md_path)
        logger.info("comparison_overall_winner=%s", overall_winner)
    return csv_path, md_path, overall_winner


def run_reuse_embeddings(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    if args.jobs is not None or cfg.get("batch"):
        base_cases = load_batch_cases(cfg, str(args.jobs or "all"))
    else:
        base_cases = [single_case_from_args(args, cfg)]
    source_dirs = parse_source_npz_dirs(args.source_npz_dir)
    comparison_name = str(args.comparison_name or DEFAULT_REUSE_COMPARISON_NAME)
    summary_logger = build_logger(
        ROOT / "data" / "logs" / comparison_subdir(comparison_name) / "reuse_embedding_compare.log"
    )
    summary_logger.info(
        "reuse_compare_start comparison=%s source_dirs=%s jobs=%s",
        comparison_name,
        ",".join(str(path) for path in source_dirs),
        ",".join(case.name for case in base_cases),
    )

    all_entries: list[Dict[str, Any]] = []
    total = len(source_dirs) * len(base_cases)
    run_idx = 0
    for source_dir in source_dirs:
        embedding_set = source_dir.name
        for base_case in base_cases:
            run_idx += 1
            source_npz = source_dir / f"{base_case.name}.npz"
            if not source_npz.exists():
                raise FileNotFoundError(f"missing source npz for job={base_case.name}: {source_npz}")
            case = reuse_output_case(
                base_case,
                comparison_name=comparison_name,
                embedding_set=embedding_set,
            )
            expected_entries = reuse_report_entries_for_case(
                case,
                cfg,
                args.init_window_sec,
                embedding_set=embedding_set,
                source_npz=source_npz,
            )
            all_entries.extend(expected_entries)
            if not bool(args.force) and reports_exist_for_reuse_case(case, cfg, args.init_window_sec):
                print(f"[skip reuse {run_idx}/{total}] {embedding_set}/{base_case.name}: reports already exist")
                continue
            print(f"[reuse {run_idx}/{total}] {embedding_set}/{base_case.name}")
            run_reuse_case(
                case,
                cfg,
                args,
                source_npz=source_npz,
                embedding_set=embedding_set,
            )

    csv_path, md_path, winner = write_comparison_summary(
        all_entries,
        comparison_name=comparison_name,
        logger=summary_logger,
    )
    print(f"[summary] winner={winner} csv={csv_path} md={md_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online closed-set assignment evaluation from offline warmup clusters.")
    parser.add_argument("--manifest", help="recording manifest for batch inference")
    parser.add_argument("--data-root", help="root containing manifest recording paths")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and model without reading signal samples")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "infer.yaml"))
    parser.add_argument("--input-bin", type=str, default=None)
    parser.add_argument("--input-h5", type=str, default=None)
    parser.add_argument("--n-channels", type=int, default=None)
    parser.add_argument("--fs", type=float, default=None)
    parser.add_argument("--chunk-sec", type=float, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--electrode-config", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cuda", "auto", "cpu"], default=None)
    parser.add_argument("--out-npz", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--jobs", type=str, default=None, help="'all' or comma-separated batch job names")
    parser.add_argument("--output-dir", type=str, default=None, help="Override batch output_dir from config")
    parser.add_argument("--log-dir", type=str, default=None, help="Override batch log_dir from config")
    parser.add_argument("--init-window-sec", type=str, default=None, help="comma-separated init windows, e.g. 20,30")
    parser.add_argument("--force", action="store_true", help="rerun batch jobs even if all online reports already exist")
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="skip inference and run online assignment from existing per-job embedding NPZ files",
    )
    parser.add_argument(
        "--source-npz-dir",
        action="append",
        default=None,
        help=(
            "directory containing existing <job>.npz embeddings; may be repeated or comma-separated. "
            "Relative values are resolved under the repo root and data/cache."
        ),
    )
    parser.add_argument(
        "--comparison-name",
        type=str,
        default=DEFAULT_REUSE_COMPARISON_NAME,
        help="relative output subdirectory/name for reuse comparison outputs",
    )
    return parser


def reports_exist_for_case(case: Case, cfg: Dict[str, Any], init_override: str | None) -> bool:
    online_cfg = resolve_online_config(cfg, init_override)
    return all(report_path(online_output_path(case.out_npz, sec)).exists() for sec in online_cfg.init_window_sec)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.config = str(resolve_path(args.config, ROOT))
    cfg = ro.load_yaml(args.config)
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))
    manifest_spec = args.manifest or cfg.get("batch", {}).get("manifest")
    if manifest_spec and args.jobs is not None:
        manifest = load_recording_manifest(resolve_path(manifest_spec))
        if args.jobs != "all":
            requested = {name.strip() for name in args.jobs.split(",")}
            known = {record["name"] for record in manifest["records"]}
            if requested - known:
                raise ValueError(f"Unknown recording(s): {sorted(requested - known)}")
            manifest["records"] = [r for r in manifest["records"] if r["name"] in requested]
        paths = resolve_data_records(manifest, args.data_root or cfg["batch"].get("data_root", "data/recordings"), ROOT)
        cfg["batch"]["jobs"] = [dict(name=r["name"], input_h5=str(path)) for r, path in zip(manifest["records"], paths)]
    if args.reuse_embeddings:
        return run_reuse_embeddings(cfg, args)
    if args.jobs is not None:
        cases = load_batch_cases(
            cfg,
            str(args.jobs),
            output_dir_override=args.output_dir,
            log_dir_override=args.log_dir,
        )
    else:
        cases = [single_case_from_args(args, cfg)]

    if args.dry_run:
        for case in cases:
            ctx = prepare_context(case, cfg, args, build_logger(case.log_file))
            print(f"[preflight passed] {case.name}: {ctx.fs:g} Hz, {ctx.n_channels} channels, weights loaded")
        return 0

    for idx, case in enumerate(cases, start=1):
        if args.jobs is not None and not args.force and not cfg.get("online", {}).get("inference_only", False) and reports_exist_for_case(case, cfg, args.init_window_sec):
            print(f"[skip] {case.name}: online reports already exist")
            continue
        print(f"[case {idx}/{len(cases)}] {case.name}")
        code = run_case(case, cfg, args)
        if code != 0:
            return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
