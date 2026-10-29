from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reproducibility import load_recording_manifest, resolve_data_records
from scripts.check_uid_consistency import compare_files, load_file_info
from src.dataset import (
    IGNORED_SAMPLE_SOURCE,
    MEArecH5Reader,
    OPERATIONAL_SOURCE_NAME_TO_CODE,
    OPERATIONAL_SOURCE_SCHEMA_VERSION,
    build_operational_pretrain_config,
    build_unit_recording_support,
    ensure_global_unit_id_labels,
    load_mearec_pretrain_data,
    load_npz,
    save_npz,
    save_optional_patches,
)
from src.model_io import build_model_config, electrode_config_hash, write_gate_metadata_sidecar
from src.nn.losses import gate_detection_loss, stage2_total_loss
from src.nn.model import SpikeEmbeddingModel, _normalize_encoder_route
from src.operational_gate import (
    LABELING_VERSION as OPERATIONAL_LABELING_VERSION,
    binary_metrics as operational_binary_metrics,
    compute_natural_sampler_probs as operational_compute_natural_sampler_probs,
)
from src.utils import load_yaml, set_seed

GATE_SOURCE_ORDER: Tuple[str, ...] = (
    "primary_matched_positive",
    "secondary_matched_positive",
    "detector_negative",
)
GATE_SOURCE_NAME_TO_CODE: Dict[str, int] = dict(OPERATIONAL_SOURCE_NAME_TO_CODE)
GATE_SOURCE_CODE_TO_NAME: Dict[int, str] = {
    code: name for name, code in GATE_SOURCE_NAME_TO_CODE.items()
}

PRETRAIN_CACHE_VERSION = 6
PRETRAIN_REQUIRED_LABEL_FIELDS: Tuple[str, ...] = (
    "unit_id",
    "local_unit_id",
    "global_unit_id",
    "geo",
    "rel_pos",
    "channel_mask",
    "neighbor_chs",
    "spatial_stats",
    "spike_label",
    "sample_source",
    "center_channel",
    "recording_id",
    "event_time_sample",
    "anchor_time_sample",
    "recording_fs_hz",
    "quality_flag",
    "align_shift_samples",
    "peak_channel",
    "strict_overlap_label",
    "overlap_label",
    "qc_allowed",
    "strict_matching_ambiguous",
    "relaxed_matching_ambiguous",
    "snr_peak",
    "event_amp_norm",
    "event_spatial_score",
    "event_simul_local",
    "event_simul_global",
    "training_mask",
    "eval_mask",
    "domain_type",
    "n_channels",
    "label_role",
    "matched_gt_label",
    "matched_gt_index",
    "matched_gt_channel",
    "neighbor_primary",
    "gt_anchor_channel",
    "gt_peak_aligned_channel",
    "gt_anchor_peak_aligned_channel_changed",
    "missed_gt_near_gt",
    "missed_gt_ignore_enabled",
    "missed_gt_ignore_near_gt_ms",
    "missed_gt_ignore_neighbor_hops",
    "require_drift_gt_channel_for_missed_ignore",
    "missed_gt_count",
    "missed_gt_ignored_candidates",
    "missed_gt_drift_channel_valid",
    "labeling_version",
    "source_schema_version",
)
PRETRAIN_OPERATIONAL_LABELING_CONTRACT: Dict[str, Any] = {
    "gt_align_to_peak": True,
    "gt_align_radius_ms": 0.5,
    "gt_time_reference": "near_peak",
    "single_channel_gt_align_radius_ms": 2.0,
    "single_channel_gt_time_reference": "onset",
    "gt_align_polarity": "negative",
    "gt_align_channel_hops": 1,
    "primary_channel_mode": "exact_then_neighbor",
    "neighbor_hops": 1,
    "neighbor_k": 9,
    "max_positive_per_gt": 2,
    "unit_snr_min_positive": 4.0,
    "high_snr_ignore_near_gt_ms": 0.5,
    "high_snr_ignore_neighbor_hops": 1,
    "missed_gt_ignore_enabled": True,
    "missed_gt_ignore_near_gt_ms": 0.5,
    "missed_gt_ignore_neighbor_hops": 1,
    "require_drift_gt_channel_for_missed_ignore": True,
    "drift_gt_channel_detection": "h5_metadata",
    "gt_neighbor_exclusion": "anchor_peak_aligned_union",
}
OPERATIONAL_FIXED_GATE_THRESHOLD = 0.5
GATE_SAMPLER_MODE_ALIASES: Dict[str, str] = {
    "natural": "natural",
    "candidate": "natural",
    "candidate_prior": "natural",
    "deployment": "natural",
}
GATE_LOSS_WEIGHT_MODE_ALIASES: Dict[str, str] = {
    "none": "none",
}
OPERATIONAL_IGNORED_DATA_DEFAULTS: Dict[str, Any] = {
    "normalize_templates": True,
    "max_per_unit": 0,
    "gt_max_per_unit": None,
    "merge_dt_ms": 0.5,
    "dedup_spikes": True,
    "dedup_mode": "neighbors",
    "align_radius_ms": 0.2,
    "align_polarity": "negative",
    "gt_time_reference": "near_peak",
    "robust_norm_samples": 60000,
    "min_snr": 4.0,
    "gt_min_snr": None,
    "detector_min_snr": None,
    "min_unit_snr": 0.0,
    "max_snr": 0.0,
    "drop_border_spikes": True,
    "sample_strategy": "uniform_time",
    "include_detector_candidates": True,
    "secondary_matched_positive_ratio": 1.0,
    "detector_negative_ratio": 2.0,
    "detector_negative_exclusion_ms": 0.6,
    "detector_negative_ambiguous_tol_ms": 0.6,
    "secondary_matched_positive_tol_samples": 3,
    "detector_disable_negative_for_clean": False,
    "detector_tol_ms": 0.3,
    "detector_alpha_overrides": {},
}


def resolve_path(path_value: str | Path, root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return root / p


def pretrain_cache_manifest_path(labels_path: Path) -> Path:
    return labels_path.with_name(labels_path.name + ".meta.yaml")


def _array_summary(arr: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(arr)
    return {
        "shape": [int(v) for v in x.shape],
        "dtype": str(x.dtype),
        "nbytes": int(x.nbytes),
    }


def _label_summaries(labels: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in sorted(labels.items()):
        if isinstance(value, np.ndarray):
            out[str(key)] = _array_summary(value)
    return out


def _pretrain_cache_source_signatures(h5_paths: List[str | Path], *, root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw_path in h5_paths:
        path = resolve_path(raw_path, root)
        entry: Dict[str, Any] = {"path": str(path), "exists": bool(path.exists())}
        if path.exists():
            stat = path.stat()
            entry["size"] = int(stat.st_size)
            entry["mtime_ns"] = int(stat.st_mtime_ns)
        out.append(entry)
    return out


def build_pretrain_cache_manifest(
    *,
    mearec_h5_paths: List[str | Path],
    loader_kwargs: Dict[str, Any],
    patches_path: Path,
    labels_path: Path,
    patches: np.ndarray,
    labels: Dict[str, np.ndarray],
    root: Path,
) -> Dict[str, Any]:
    patches_stat = patches_path.stat()
    labels_stat = labels_path.stat()
    return {
        "version": PRETRAIN_CACHE_VERSION,
        "mearec_h5_paths": [str(resolve_path(p, root)) for p in mearec_h5_paths],
        "sources": _pretrain_cache_source_signatures(mearec_h5_paths, root=root),
        "loader_kwargs": dict(loader_kwargs),
        "operational_labeling_version": OPERATIONAL_LABELING_VERSION,
        "operational_labeling_contract": dict(PRETRAIN_OPERATIONAL_LABELING_CONTRACT),
        "cache": {
            "patches": {
                "path": str(patches_path),
                "file_size": int(patches_stat.st_size),
                "summary": _array_summary(patches),
            },
            "labels": {
                "path": str(labels_path),
                "file_size": int(labels_stat.st_size),
                "summary": _label_summaries(labels),
            },
        },
    }


def write_pretrain_cache_manifest(manifest_path: Path, payload: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)


def check_pretrain_cache(
    *,
    mearec_h5_paths: List[str | Path],
    loader_kwargs: Dict[str, Any],
    patches_path: Path,
    labels_path: Path,
    root: Path,
) -> tuple[str, str, Path]:
    manifest_path = pretrain_cache_manifest_path(labels_path)
    if not patches_path.exists() or not labels_path.exists():
        missing = []
        if not patches_path.exists():
            missing.append("patches")
        if not labels_path.exists():
            missing.append("labels")
        return "miss", f"missing_{'_'.join(missing)}", manifest_path
    if not manifest_path.exists():
        return "miss", "missing_manifest", manifest_path
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return "stale", "manifest_read_error", manifest_path
    if not isinstance(payload, dict):
        return "stale", "invalid_manifest", manifest_path
    if int(payload.get("version", -1)) != PRETRAIN_CACHE_VERSION:
        return "stale", "version", manifest_path
    try:
        with np.load(labels_path, allow_pickle=False) as label_npz:
            missing_fields = [field for field in PRETRAIN_REQUIRED_LABEL_FIELDS if field not in label_npz.files]
            schema_values = (
                np.asarray(label_npz["source_schema_version"]).reshape(-1)
                if "source_schema_version" in label_npz.files
                else np.asarray([], dtype=str)
            )
            labeling_values = (
                np.asarray(label_npz["labeling_version"]).reshape(-1)
                if "labeling_version" in label_npz.files
                else np.asarray([], dtype=str)
            )
    except Exception:
        return "stale", "label_read_error", manifest_path
    if missing_fields:
        return "stale", "missing_label_fields_" + "_".join(missing_fields[:4]), manifest_path
    if schema_values.size == 0 or not np.all(schema_values.astype(str) == OPERATIONAL_SOURCE_SCHEMA_VERSION):
        return "stale", "source_schema_version", manifest_path
    if labeling_values.size == 0 or not np.all(labeling_values.astype(str) == OPERATIONAL_LABELING_VERSION):
        return "stale", "operational_labeling_version", manifest_path
    expected_paths = [str(resolve_path(p, root)) for p in mearec_h5_paths]
    if payload.get("mearec_h5_paths") != expected_paths:
        return "stale", "mearec_h5_paths", manifest_path
    if payload.get("sources") != _pretrain_cache_source_signatures(mearec_h5_paths, root=root):
        return "stale", "source_signature", manifest_path
    if payload.get("loader_kwargs") != dict(loader_kwargs):
        return "stale", "loader_kwargs", manifest_path
    if str(payload.get("operational_labeling_version", "")) != OPERATIONAL_LABELING_VERSION:
        return "stale", "operational_labeling_version", manifest_path
    if payload.get("operational_labeling_contract") != dict(PRETRAIN_OPERATIONAL_LABELING_CONTRACT):
        return "stale", "operational_labeling_contract", manifest_path
    return "hit", "manifest_match", manifest_path


def _normalize_pretrain_encoder_route(route: Any) -> str:
    return _normalize_encoder_route(str(route or "peak_only"))


def resolve_pretrain_encoder_route(model_cfg: Dict[str, Any], data_peak_only: bool) -> str:
    if "encoder_route" not in model_cfg:
        return "peak_only"
    return _normalize_pretrain_encoder_route(model_cfg.get("encoder_route"))


def validate_pretrain_config_compatibility(
    *,
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    loader_kwargs: Dict[str, Any],
) -> str:
    data_peak_only = bool(data_cfg.get("peak_only", False))
    route = resolve_pretrain_encoder_route(model_cfg, data_peak_only)
    configured_top_k = loader_kwargs.get("top_k_channels", data_cfg.get("top_k_channels", 1))
    configured_top_k = int(configured_top_k)
    if data_peak_only and route != "peak_only":
        raise ValueError(f"data.peak_only=true requires model.encoder_route=peak_only, got {route}")
    if route == "peak_only" and configured_top_k != 1:
        raise ValueError("model.encoder_route=peak_only requires data.top_k_channels=1")
    if route != "peak_only" and configured_top_k <= 1:
        raise ValueError(f"model.encoder_route={route} requires data.top_k_channels>=2")
    norm_mode = str(loader_kwargs.get("normalization_mode", data_cfg.get("normalization_mode", "peak_channel_sigma"))).lower()
    if norm_mode in ("per_spike", "patch_zscore", "zscore"):
        raise ValueError("per-spike waveform normalization is disabled; use peak_channel_sigma")
    return route


def validate_pretrain_array_compatibility(
    *,
    patches: np.ndarray,
    labels: Dict[str, np.ndarray],
    data_peak_only: bool,
    encoder_route: str,
    context: str,
) -> None:
    if patches.ndim != 3:
        raise ValueError(f"{context} patches must have shape [N,T,K], got {tuple(patches.shape)}")
    n_samples = int(patches.shape[0])
    patch_k = int(patches.shape[2])
    if bool(data_peak_only) and patch_k != 1:
        raise ValueError(
            f"{context} data.peak_only=true requires single-channel patches, got patches.shape[2]={patch_k}"
        )
    if _normalize_pretrain_encoder_route(encoder_route) == "peak_only" and patch_k != 1:
        raise ValueError(
            f"{context} model.encoder_route=peak_only requires single-channel patches, got patches.shape[2]={patch_k}"
        )
    if _normalize_pretrain_encoder_route(encoder_route) != "peak_only" and patch_k <= 1:
        raise ValueError(
            f"{context} model.encoder_route={encoder_route} requires multi-channel patches, got patches.shape[2]={patch_k}"
        )

    expected_shapes = {
        "geo": (n_samples, patch_k, 4),
        "rel_pos": (n_samples, patch_k, 2),
        "channel_mask": (n_samples, patch_k),
        "neighbor_chs": (n_samples, patch_k),
    }
    for key, expected in expected_shapes.items():
        if key not in labels:
            raise ValueError(f"{context} labels missing required field {key!r}")
        actual = tuple(int(v) for v in np.asarray(labels[key]).shape)
        if actual != expected:
            raise ValueError(f"{context} labels[{key!r}] shape mismatch: expected {expected}, got {actual}")


def _config_values_equivalent(actual: Any, default: Any) -> bool:
    if actual is None or default is None:
        return actual is None and default is None
    if isinstance(default, bool):
        if isinstance(actual, bool):
            return actual == default
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            return bool(actual) == default if actual in (0, 1) else False
        if isinstance(actual, str):
            normalized = actual.strip().lower()
            if normalized in {"true", "false"}:
                return (normalized == "true") == default
        return False
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        try:
            return math.isclose(float(actual), float(default), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(default, dict):
        return dict(actual or {}) == default if isinstance(actual, dict) or actual is None else False
    return str(actual) == str(default)


def validate_pretrain_config_structure(cfg: Dict[str, Any]) -> None:
    if "stage1_gate" in cfg:
        raise ValueError("top-level stage1_gate is ignored; move stage settings under train.stage1_gate")


def audit_operational_pretrain_data_config(
    data_cfg: Dict[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> None:
    ignored_default_keys: List[str] = []
    ignored_nondefault: List[str] = []
    for key, default_value in OPERATIONAL_IGNORED_DATA_DEFAULTS.items():
        if key not in data_cfg:
            continue
        actual = data_cfg[key]
        if _config_values_equivalent(actual, default_value):
            ignored_default_keys.append(key)
        else:
            ignored_nondefault.append(f"data.{key}={actual!r} (default {default_value!r})")
    if ignored_nondefault:
        raise ValueError(
            "operational pretraining ignores these data.* keys with non-default values: "
            + ", ".join(ignored_nondefault)
            + "; move the setting into data.detector/stage2 if supported or remove it"
        )
    if ignored_default_keys and logger is not None:
        logger.warning(
            "operational_pretrain_ignored_default_data_keys keys=%s",
            ",".join(sorted(ignored_default_keys)),
        )


def audit_operational_pretrain_cli_args(args: argparse.Namespace) -> None:
    ignored_args = []
    if args.max_per_unit is not None:
        ignored_args.append("--max-per-unit")
    if args.merge_dt_ms is not None:
        ignored_args.append("--merge-dt-ms")
    if args.dedup_mode is not None:
        ignored_args.append("--dedup-mode")
    if bool(args.no_dedup_spikes):
        ignored_args.append("--no-dedup-spikes")
    if bool(args.no_normalize_templates):
        ignored_args.append("--no-normalize-templates")
    if ignored_args:
        raise ValueError(
            "operational pretraining ignores these CLI options: "
            + ", ".join(ignored_args)
            + "; remove them or add explicit support before running"
        )


def build_effective_operational_pretrain_config(loader_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return build_operational_pretrain_config(
        left=int(loader_kwargs.get("left", 20)),
        right=int(loader_kwargs.get("right", 44)),
        patch_k=int(loader_kwargs.get("top_k_channels", 1)),
        peak_only=bool(loader_kwargs.get("peak_only", False)),
        car_mode=str(loader_kwargs.get("car_mode", "auto")),
        normalization_mode=str(loader_kwargs.get("normalization_mode", "peak_channel_sigma")),
        detector_chunk_sec=float(loader_kwargs.get("detector_chunk_sec", 0.5)),
        detector_config=dict(loader_kwargs.get("detector_config", {}) or {}),
        electrode_config=loader_kwargs.get("electrode_config"),
        strict_overlap_ms=float(loader_kwargs.get("strict_overlap_ms", 0.5)),
        strict_overlap_um=float(loader_kwargs.get("strict_overlap_um", 60.0)),
    )


def _compact_yaml(value: Any) -> str:
    return yaml.safe_dump(value, sort_keys=True, default_flow_style=True).strip()


def log_effective_operational_pretrain_config(
    logger: logging.Logger,
    effective_cfg: Dict[str, Any],
) -> None:
    for section in ("patch", "preprocess", "stream", "detection", "labeling", "offline"):
        logger.info(
            "operational_pretrain_effective_config section=%s value=%s",
            section,
            _compact_yaml(effective_cfg.get(section, {})),
        )


def validate_mearec_h5_paths(h5_paths: List[str | Path]) -> None:
    missing = [str(p) for p in h5_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("configured MEArec H5 files are missing: " + ", ".join(missing))


def recording_id_path_map(h5_paths: List[str | Path]) -> Dict[int, str]:
    return {int(i): str(Path(p)) for i, p in enumerate(h5_paths)}


def format_recording_id_path_map(mapping: Dict[int, str], recordings: List[int] | np.ndarray | None = None) -> str:
    if recordings is None:
        keys = sorted(mapping.keys())
    else:
        keys = sorted(int(v) for v in np.asarray(recordings, dtype=np.int64).reshape(-1).tolist())
    return " ".join(f"{int(k)}={mapping.get(int(k), '<unknown>')}" for k in keys)


def validate_recording_holdout_validation_recordings(
    val_recordings: List[int] | np.ndarray,
    mapping: Dict[int, str],
) -> None:
    blocked: List[str] = []
    for rec_value in np.asarray(val_recordings, dtype=np.int64).reshape(-1).tolist():
        path_raw = mapping.get(int(rec_value))
        if not path_raw:
            continue
        path = Path(path_raw)
        if "noise20" not in path.name.lower():
            continue
        try:
            _, n_channels = MEArecH5Reader(path).get_recordings_shape()
        except Exception as exc:
            raise RuntimeError(f"could not inspect validation recording {rec_value} at {path}: {exc}") from exc
        if int(n_channels) == 32:
            blocked.append(f"recording_id={int(rec_value)} path={path}")
    if blocked:
        raise ValueError(
            "recording_holdout validation must not use 32-channel noise20 recordings: "
            + "; ".join(blocked)
        )


def build_file_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mysort.pretrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _parse_epoch_from_name(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return -1


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    files = list(checkpoint_dir.glob("pretrain_epoch_*.ckpt"))
    if not files:
        return None
    return max(files, key=_parse_epoch_from_name)


def save_training_checkpoint(
    checkpoint_path: Path,
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if metadata is not None:
        state["metadata"] = metadata
    torch.save(state, checkpoint_path)


def load_training_checkpoint_payload(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> Any:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)



def restore_training_checkpoint_payload(
    ckpt: Any,
    *,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    restore_optimizer: bool = True,
    restore_scheduler: bool = True,
) -> tuple[int, int, bool]:
    _ = device

    def _normalize_rng_state(state: Any) -> torch.Tensor:
        if isinstance(state, torch.Tensor):
            return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()

    if "model_state_dict" in ckpt:
        model.load_state_dict({k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("gate_net.overlap_head.")})
    else:
        model.load_state_dict(ckpt)
    if restore_optimizer and isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if isinstance(ckpt, dict) and "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if isinstance(ckpt, dict) and "torch_rng_state" in ckpt:
        torch.set_rng_state(_normalize_rng_state(ckpt["torch_rng_state"]))
    if isinstance(ckpt, dict) and "torch_cuda_rng_state_all" in ckpt and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(_normalize_rng_state(state) for state in ckpt["torch_cuda_rng_state_all"])
    scheduler_loaded = False
    if restore_scheduler and scheduler is not None and isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scheduler_loaded = True
    epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    global_step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0
    return epoch, global_step, scheduler_loaded


def load_model_state_from_checkpoint_payload(
    ckpt: Any,
    *,
    model: torch.nn.Module,
) -> None:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict({k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("gate_net.overlap_head.")})
    else:
        model.load_state_dict(ckpt)


def build_optimizer_for_parameters(
    parameters: List[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    if not parameters:
        raise ValueError("optimizer requires at least one parameter")
    return optim.AdamW(
        parameters,
        lr=float(lr),
        weight_decay=float(weight_decay),
    )


def load_training_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, int, bool]:
    ckpt = load_training_checkpoint_payload(checkpoint_path, device=device)
    return restore_training_checkpoint_payload(
        ckpt,
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
    )


def resolve_device(requested: str, *, allow_cpu: bool = False) -> torch.device:
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("device=auto resolved to CPU, but this project requires GPU for training.")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available. Check CUDA driver and PyTorch CUDA build.")
        return torch.device("cuda")
    if req == "cpu":
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("CPU training is disabled. Please run with --device cuda on a GPU machine.")
    raise ValueError(f"unsupported device: {requested}")


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> optim.lr_scheduler.LambdaLR:
    min_lr = float(max(0.0, min_lr))
    base_lr = float(max(1e-12, base_lr))
    total_steps = int(max(1, total_steps))
    warmup_steps = int(max(0, min(warmup_steps, total_steps - 1)))
    min_factor = float(np.clip(min_lr / base_lr, 0.0, 1.0))

    def _lr_lambda(step: int) -> float:
        s = int(np.clip(step, 0, total_steps))
        if warmup_steps > 0 and s < warmup_steps:
            return max(1e-8, float(s + 1) / float(warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        prog = float(s - warmup_steps) / float(max(1, total_steps - warmup_steps))
        prog = float(np.clip(prog, 0.0, 1.0))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return float(min_factor + (1.0 - min_factor) * cos)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def build_scheduler_from_config(
    optimizer: optim.Optimizer,
    *,
    scheduler_cfg: Dict[str, Any],
    base_lr: float,
    total_steps: int,
    steps_per_epoch: int,
    warmup_epochs: int | None = None,
) -> tuple[optim.lr_scheduler.LRScheduler | None, Dict[str, Any]]:
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    info: Dict[str, Any] = {"type": scheduler_type}
    if scheduler_type in ("warmup_cosine", "cosine_warmup", "warmup-cosine"):
        resolved_warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0)) if warmup_epochs is None else int(warmup_epochs)
        resolved_warmup_epochs = int(max(0, resolved_warmup_epochs))
        warmup_steps = int(max(0, resolved_warmup_epochs * max(1, steps_per_epoch)))
        min_lr = float(scheduler_cfg.get("min_lr", 1e-5))
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            base_lr=float(base_lr),
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            total_steps=int(max(1, total_steps)),
        )
        info.update(
            {
                "warmup_epochs": int(resolved_warmup_epochs),
                "warmup_steps": int(warmup_steps),
                "total_steps": int(max(1, total_steps)),
                "min_lr": float(min_lr),
            }
        )
        return scheduler, info
    return None, info


def stratified_train_val_split(
    labels: np.ndarray,
    *,
    val_ratio: float,
    seed: int,
    min_val_per_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    tr_parts: List[np.ndarray] = []
    va_parts: List[np.ndarray] = []
    for c in classes.tolist():
        idx = np.where(y == int(c))[0]
        if idx.size <= 1 or val_ratio <= 0:
            tr_parts.append(idx.astype(np.int64, copy=False))
            continue
        idx = idx.copy()
        rng.shuffle(idx)
        n_val = int(round(idx.size * float(val_ratio)))
        n_val = max(int(min_val_per_class), n_val)
        n_val = min(n_val, int(idx.size - 1))
        if n_val <= 0:
            tr_parts.append(idx.astype(np.int64, copy=False))
            continue
        va_parts.append(idx[:n_val].astype(np.int64, copy=False))
        tr_parts.append(idx[n_val:].astype(np.int64, copy=False))
    tr_idx = np.concatenate(tr_parts, axis=0) if tr_parts else np.zeros((0,), dtype=np.int64)
    va_idx = np.concatenate(va_parts, axis=0) if va_parts else np.zeros((0,), dtype=np.int64)
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    return tr_idx, va_idx


def build_recording_holdout_candidate_mask(
    unit_id: np.ndarray,
    spike_label: np.ndarray | None,
    sample_source: np.ndarray | None,
    *,
    strict_overlap_label: np.ndarray | None = None,
    overlap_label: np.ndarray | None = None,
    qc_allowed: np.ndarray | None = None,
    strict_matching_ambiguous: np.ndarray | None = None,
    relaxed_matching_ambiguous: np.ndarray | None = None,
) -> np.ndarray:
    unit_arr = np.asarray(unit_id, dtype=np.int64)
    keep = unit_arr >= 0
    if spike_label is not None:
        keep = keep & (np.asarray(spike_label, dtype=np.float32) > 0.5)
    if sample_source is not None:
        keep = keep & (np.asarray(sample_source, dtype=np.int8) == int(GATE_SOURCE_NAME_TO_CODE["primary_matched_positive"]))
    if strict_overlap_label is not None:
        keep = keep & (~np.asarray(strict_overlap_label, dtype=np.bool_))
    if overlap_label is not None:
        keep = keep & (np.asarray(overlap_label, dtype=np.float32) <= 0.5)
    if qc_allowed is not None:
        keep = keep & np.asarray(qc_allowed, dtype=np.bool_)
    if strict_matching_ambiguous is not None:
        keep = keep & (~np.asarray(strict_matching_ambiguous, dtype=np.bool_))
    if relaxed_matching_ambiguous is not None:
        keep = keep & (~np.asarray(relaxed_matching_ambiguous, dtype=np.bool_))
    return keep.astype(np.bool_, copy=False)


def _valid_units_for_recording_holdout(
    unit_id: np.ndarray,
    recording_id: np.ndarray,
    candidate_mask: np.ndarray,
    recordings: np.ndarray,
    *,
    min_val_samples_per_unit: int,
) -> Dict[int, np.ndarray]:
    unit_arr = np.asarray(unit_id, dtype=np.int64)
    rec_arr = np.asarray(recording_id, dtype=np.int64)
    candidate = np.asarray(candidate_mask, dtype=np.bool_)
    rec_values = np.asarray(recordings, dtype=np.int64)
    out: Dict[int, np.ndarray] = {}
    for rec_value in rec_values.tolist():
        val_mask = candidate & (rec_arr == int(rec_value))
        val_units, val_counts = np.unique(unit_arr[val_mask], return_counts=True)
        valid_units = [
            int(u)
            for u, c in zip(val_units.tolist(), val_counts.tolist())
            if int(c) >= int(min_val_samples_per_unit)
        ]
        out[int(rec_value)] = np.asarray(sorted(valid_units), dtype=np.int64)
    return out


def _recording_holdout_combo_valid(
    unit_id: np.ndarray,
    recording_id: np.ndarray,
    candidate_mask: np.ndarray,
    selected_recordings: np.ndarray,
    *,
    min_train_recordings: int,
    min_holdout_units: int,
    min_val_samples_per_unit: int,
) -> tuple[bool, Dict[int, np.ndarray]]:
    rec_arr = np.asarray(recording_id, dtype=np.int64)
    selected = np.asarray(selected_recordings, dtype=np.int64)
    if selected.size == 0:
        return False, {}
    train_recording_count = int(np.unique(rec_arr[~np.isin(rec_arr, selected)]).size)
    if train_recording_count < int(min_train_recordings):
        return False, {}
    valid_units = _valid_units_for_recording_holdout(
        unit_id,
        recording_id,
        candidate_mask,
        selected,
        min_val_samples_per_unit=int(min_val_samples_per_unit),
    )
    valid = all(int(units.size) >= int(min_holdout_units) for units in valid_units.values())
    return bool(valid), valid_units


def recording_holdout_train_val_split(
    labels: np.ndarray,
    recording_id: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    val_recording_ratio: float,
    seed: int,
    min_val_recordings: int = 1,
    min_train_recordings: int = 1,
    min_holdout_units: int = 2,
    min_train_samples_per_unit: int = 1,
    min_val_samples_per_unit: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    unit_arr = np.asarray(labels, dtype=np.int64)
    rec_arr = np.asarray(recording_id, dtype=np.int64)
    candidate = np.asarray(candidate_mask, dtype=np.bool_) & (unit_arr >= 0)
    if unit_arr.shape[0] != rec_arr.shape[0] or unit_arr.shape[0] != candidate.shape[0]:
        raise ValueError("labels, recording_id, and candidate_mask must have matching shape")
    rec_values = np.unique(rec_arr).astype(np.int64, copy=False)
    min_val_recordings = int(max(1, min_val_recordings))
    min_train_recordings = int(max(1, min_train_recordings))
    min_holdout_units = int(max(1, min_holdout_units))
    min_val_samples_per_unit = int(max(1, min_val_samples_per_unit))
    if rec_values.size < min_val_recordings + min_train_recordings:
        raise RuntimeError(
            "recording_holdout split requires at least "
            f"{min_val_recordings + min_train_recordings} recordings; got {int(rec_values.size)}"
        )
    desired_val = int(round(float(rec_values.size) * float(val_recording_ratio)))
    desired_val = max(min_val_recordings, desired_val)
    desired_val = min(desired_val, int(rec_values.size) - min_train_recordings)
    rng = np.random.default_rng(seed)
    shuffled_recs = rec_values.copy()
    rng.shuffle(shuffled_recs)
    eligible_recs: List[int] = []
    skipped: Dict[int, str] = {}
    for rec_value in shuffled_recs.tolist():
        ok, valid_units = _recording_holdout_combo_valid(
            unit_arr,
            rec_arr,
            candidate,
            np.asarray([int(rec_value)], dtype=np.int64),
            min_train_recordings=min_train_recordings,
            min_holdout_units=min_holdout_units,
            min_val_samples_per_unit=min_val_samples_per_unit,
        )
        if ok:
            eligible_recs.append(int(rec_value))
        else:
            valid_count = int(valid_units.get(int(rec_value), np.zeros((0,), dtype=np.int64)).size) if valid_units else 0
            skipped[int(rec_value)] = f"valid_units={valid_count}"
    if len(eligible_recs) < min_val_recordings:
        raise RuntimeError(
            "recording_holdout split found no eligible validation recording with enough clean valid units; "
            "lower train.validation.min_holdout_units/min_val_samples_per_unit or add more recordings"
        )
    desired_val = min(desired_val, len(eligible_recs))
    selected: np.ndarray | None = None
    selected_valid_units: Dict[int, np.ndarray] = {}
    for k in range(desired_val, min_val_recordings - 1, -1):
        if len(eligible_recs) <= 12 or math.comb(len(eligible_recs), k) <= 4096:
            combo_iter = combinations(eligible_recs, k)
        else:
            seen: set[tuple[int, ...]] = set()
            random_combos: List[tuple[int, ...]] = []
            for _ in range(4096):
                combo = tuple(sorted(rng.choice(np.asarray(eligible_recs, dtype=np.int64), size=k, replace=False).tolist()))
                if combo not in seen:
                    seen.add(combo)
                    random_combos.append(combo)
            combo_iter = iter(random_combos)
        for combo in combo_iter:
            combo_arr = np.asarray(combo, dtype=np.int64)
            ok, valid_units = _recording_holdout_combo_valid(
                unit_arr,
                rec_arr,
                candidate,
                combo_arr,
                min_train_recordings=min_train_recordings,
                min_holdout_units=min_holdout_units,
                min_val_samples_per_unit=min_val_samples_per_unit,
            )
            if ok:
                selected = combo_arr
                selected_valid_units = valid_units
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError(
            "recording_holdout split could not choose a compatible validation recording set; "
            "lower val_recording_ratio/min_holdout_units or add more eligible recordings"
        )
    val_mask = np.isin(rec_arr, selected)
    train_idx = np.flatnonzero(~val_mask).astype(np.int64, copy=False)
    val_idx = np.flatnonzero(val_mask).astype(np.int64, copy=False)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    summary = {
        "mode": "recording_holdout",
        "recordings_total": int(rec_values.size),
        "train_recordings": sorted(int(v) for v in np.unique(rec_arr[train_idx]).tolist()),
        "val_recordings": sorted(int(v) for v in selected.tolist()),
        "eligible_val_recordings": sorted(int(v) for v in eligible_recs),
        "skipped_val_recordings": dict(skipped),
        "valid_units_by_val_recording": {
            int(k): [int(v) for v in vals.tolist()] for k, vals in selected_valid_units.items()
        },
        "min_holdout_units": int(min_holdout_units),
        "min_val_samples_per_unit": int(min_val_samples_per_unit),
    }
    return train_idx, val_idx, summary


def embed_numpy(
    model: SpikeEmbeddingModel,
    x_np: np.ndarray,
    geo_np: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    apply_encode_gate: bool = False,
) -> np.ndarray:
    if x_np.shape[0] == 0:
        return np.zeros((0, model.cfg.embed_dim), dtype=np.float32)
    was_training = model.training
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for st in range(0, x_np.shape[0], max(1, int(batch_size))):
            ed = min(st + max(1, int(batch_size)), x_np.shape[0])
            xb = torch.from_numpy(x_np[st:ed]).to(device)
            gb = torch.from_numpy(geo_np[st:ed]).to(device)
            if apply_encode_gate:
                out = model(xb, {"geo": gb}, apply_encode_gate=True)
            else:
                out = model.forward_encoder_only(xb, {"geo": gb})
            outs.append(out.embedding.detach().cpu().numpy().astype(np.float32, copy=False))
    if was_training:
        model.train()
    if not outs:
        return np.zeros((0, model.cfg.embed_dim), dtype=np.float32)
    return np.concatenate(outs, axis=0)


def augment_batch_views(
    x: torch.Tensor,
    *,
    noise_std: float,
    gain_jitter: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def _apply(inp: torch.Tensor) -> torch.Tensor:
        out = inp
        if gain_jitter > 0.0:
            gain = 1.0 + (2.0 * torch.rand((inp.shape[0], 1, 1), device=inp.device, dtype=inp.dtype) - 1.0) * gain_jitter
            out = out * gain
        if noise_std > 0.0:
            out = out + torch.randn_like(out) * noise_std
        return out

    return _apply(x), _apply(x)


def sample_pair_cosine_stats(
    z: np.ndarray,
    y: np.ndarray,
    *,
    n_pairs: int,
    seed: int,
) -> Dict[str, float]:
    out = {
        "pos_cos_mean": float("nan"),
        "neg_cos_mean": float("nan"),
        "pos_neg_gap": float("nan"),
    }
    n = int(z.shape[0])
    if n < 2:
        return out
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    if classes.size == 0:
        return out
    pair_target = max(1000, int(n_pairs))
    per_unit_target = max(1, int(math.ceil(pair_target / max(1, classes.size))))
    pos_unit_vals: List[float] = []
    neg_unit_vals: List[float] = []
    for unit_value in classes.tolist():
        idx = np.flatnonzero(y == int(unit_value)).astype(np.int64, copy=False)
        other = np.flatnonzero(y != int(unit_value)).astype(np.int64, copy=False)
        pos_vals: List[float] = []
        neg_vals: List[float] = []
        if idx.size >= 2:
            for _ in range(per_unit_target):
                pair = rng.choice(idx, size=2, replace=False)
                pos_vals.append(float(np.dot(z[int(pair[0])], z[int(pair[1])])))
        if idx.size > 0 and other.size > 0:
            for _ in range(per_unit_target):
                i = int(rng.choice(idx))
                j = int(rng.choice(other))
                neg_vals.append(float(np.dot(z[i], z[j])))
        if pos_vals:
            pos_unit_vals.append(float(np.mean(np.asarray(pos_vals, dtype=np.float32))))
        if neg_vals:
            neg_unit_vals.append(float(np.mean(np.asarray(neg_vals, dtype=np.float32))))
    if pos_unit_vals:
        out["pos_cos_mean"] = float(np.mean(np.asarray(pos_unit_vals, dtype=np.float32)))
    if neg_unit_vals:
        out["neg_cos_mean"] = float(np.mean(np.asarray(neg_unit_vals, dtype=np.float32)))
    if pos_unit_vals and neg_unit_vals:
        out["pos_neg_gap"] = float(out["pos_cos_mean"] - out["neg_cos_mean"])
    return out


def evaluate_encoder_separation(
    model: SpikeEmbeddingModel,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_geo: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_geo: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    support_per_class: int = 32,
    max_eval_samples: int = 20000,
    eval_batch_size: int = 1024,
    pair_samples: int = 10000,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "val_proto_acc": float("nan"),
        "val_pos_cos": float("nan"),
        "val_neg_cos": float("nan"),
        "val_pos_neg_gap": float("nan"),
        "val_samples": 0.0,
        "val_classes": 0.0,
    }
    if val_x.shape[0] == 0 or train_x.shape[0] == 0:
        return metrics

    rng = np.random.default_rng(seed)
    y_train = np.asarray(train_y, dtype=np.int64)
    y_val = np.asarray(val_y, dtype=np.int64)
    train_keep = y_train >= 0
    val_keep = y_val >= 0
    train_x = train_x[train_keep]
    train_geo = train_geo[train_keep]
    y_train = y_train[train_keep]
    val_x = val_x[val_keep]
    val_geo = val_geo[val_keep]
    y_val = y_val[val_keep]
    if val_x.shape[0] == 0 or train_x.shape[0] == 0:
        return metrics

    # Build class prototypes from train split using a bounded support set per class.
    proto_labels: List[int] = []
    proto_parts: List[np.ndarray] = []
    for c in np.unique(y_train).tolist():
        idx = np.where(y_train == int(c))[0]
        if idx.size == 0:
            continue
        if idx.size > int(support_per_class):
            idx = rng.choice(idx, size=int(support_per_class), replace=False)
        zc = embed_numpy(
            model,
            train_x[idx],
            train_geo[idx],
            device=device,
            batch_size=eval_batch_size,
        )
        if zc.shape[0] == 0:
            continue
        pc = np.mean(zc, axis=0).astype(np.float32, copy=False)
        nrm = float(np.linalg.norm(pc))
        if nrm > 1e-8:
            pc = pc / nrm
        proto_labels.append(int(c))
        proto_parts.append(pc)

    if not proto_parts:
        return metrics

    p_labels = np.asarray(proto_labels, dtype=np.int64)
    p_mat = np.stack(proto_parts, axis=0).astype(np.float32, copy=False)

    # Optional cap to keep validation fast on very large datasets.
    val_idx = np.arange(val_x.shape[0], dtype=np.int64)
    if val_idx.size > int(max_eval_samples) > 0:
        val_idx = rng.choice(val_idx, size=int(max_eval_samples), replace=False)
    val_idx = np.sort(val_idx)

    z_val = embed_numpy(
        model,
        val_x[val_idx],
        val_geo[val_idx],
        device=device,
        batch_size=eval_batch_size,
    )
    y_val_sub = y_val[val_idx]

    logits = z_val @ p_mat.T
    pred = p_labels[np.argmax(logits, axis=1)]
    metrics["val_proto_acc"] = float(np.mean((pred == y_val_sub).astype(np.float32))) if pred.size > 0 else float("nan")

    sep = sample_pair_cosine_stats(
        z_val,
        y_val_sub,
        n_pairs=int(pair_samples),
        seed=seed + 17,
    )
    metrics["val_pos_cos"] = float(sep["pos_cos_mean"])
    metrics["val_neg_cos"] = float(sep["neg_cos_mean"])
    metrics["val_pos_neg_gap"] = float(sep["pos_neg_gap"])
    metrics["val_samples"] = float(y_val_sub.size)
    metrics["val_classes"] = float(np.unique(y_val_sub).size)
    return metrics


def collect_gate_probabilities(
    model: SpikeEmbeddingModel,
    x_np: np.ndarray,
    geo_np: np.ndarray,
    spatial_stats_np: np.ndarray | None = None,
    channel_mask_np: np.ndarray | None = None,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if x_np.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    was_training = model.training
    model.eval()
    spike_probs: List[np.ndarray] = []
    with torch.no_grad():
        for st in range(0, x_np.shape[0], max(1, int(batch_size))):
            ed = min(st + max(1, int(batch_size)), x_np.shape[0])
            xb = torch.from_numpy(x_np[st:ed]).to(device)
            gb = torch.from_numpy(geo_np[st:ed]).to(device)
            meta = {"geo": gb}
            if spatial_stats_np is not None:
                meta["spatial_stats"] = torch.from_numpy(np.asarray(spatial_stats_np[st:ed], dtype=np.float32)).to(device)
            if channel_mask_np is not None:
                meta["channel_mask"] = torch.from_numpy(np.asarray(channel_mask_np[st:ed], dtype=np.bool_)).to(device)
            _, spike_prob = model.forward_gate_only(xb, meta)
            spike_probs.append(spike_prob.detach().cpu().numpy().astype(np.float32, copy=False))
    if was_training:
        model.train()
    if not spike_probs:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(spike_probs, axis=0).astype(np.float32, copy=False)


def gate_metrics_from_probs(
    spike_prob: np.ndarray,
    spike_label: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    metrics = {
        "gate_recall": float("nan"),
        "gate_precision": float("nan"),
        "gate_f1": float("nan"),
        "gate_f0_5": float("nan"),
        "noise_accuracy": float("nan"),
        "noise_specificity": float("nan"),
        "noise_recall": float("nan"),
        "balanced_accuracy": float("nan"),
        "encode_pass_ratio": float("nan"),
        "positive_rate": float("nan"),
        "threshold": float(threshold),
    }
    if spike_prob.size == 0:
        return metrics
    prob = np.asarray(spike_prob, dtype=np.float32)
    target = (np.asarray(spike_label, dtype=np.float32) > 0.5).astype(np.int8, copy=False)
    pred = (prob >= float(threshold)).astype(np.int8, copy=False)
    tp = int(np.sum((pred == 1) & (target == 1)))
    fp = int(np.sum((pred == 1) & (target == 0)))
    fn = int(np.sum((pred == 0) & (target == 1)))
    tn = int(np.sum((pred == 0) & (target == 0)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2.0 * precision * recall) / max(1e-8, precision + recall))
    beta2 = 0.5 * 0.5
    f05 = float((1.0 + beta2) * precision * recall / max(1e-8, beta2 * precision + recall))
    noise_accuracy = float(tn / max(1, tn + fp))
    noise_specificity = float(noise_accuracy)
    noise_recall_deprecated = float(tn / max(1, tn + fn))
    metrics["gate_recall"] = recall
    metrics["gate_precision"] = precision
    metrics["gate_f1"] = f1
    metrics["gate_f0_5"] = f05
    metrics["noise_accuracy"] = noise_accuracy
    metrics["noise_specificity"] = noise_specificity
    metrics["noise_recall"] = noise_recall_deprecated
    metrics["balanced_accuracy"] = float(0.5 * (recall + noise_accuracy))
    metrics["encode_pass_ratio"] = float(np.mean(pred.astype(np.float32)))
    metrics["positive_rate"] = float(np.mean(target.astype(np.float32)))
    return metrics

def validate_fixed_gate_thresholds(
    *,
    encode_gate_thresh: float,
    assignment_gate_thresh: float,
    fixed_gate_threshold: float | None,
    context: str,
) -> float | None:
    if fixed_gate_threshold is None:
        return None
    fixed = float(fixed_gate_threshold)
    if abs(fixed - 0.5) > 1e-8:
        raise ValueError(f"{context} fixed_gate_threshold currently only supports 0.5, got {fixed:.4f}")
    if abs(float(encode_gate_thresh) - fixed) > 1e-8:
        raise ValueError(
            f"{context} requires encode_gate_thresh={fixed:.4f}, got {float(encode_gate_thresh):.4f}"
        )
    if abs(float(assignment_gate_thresh) - fixed) > 1e-8:
        raise ValueError(
            f"{context} requires assignment_gate_thresh={fixed:.4f}, got {float(assignment_gate_thresh):.4f}"
        )
    return fixed


def build_fixed_gate_summary(
    spike_prob: np.ndarray,
    spike_label: np.ndarray,
    *,
    threshold: float,
    gate_goal_spike_recall: float,
    gate_goal_noise_accuracy: float,
    domain_id: np.ndarray | None = None,
    sample_source: np.ndarray | None = None,
) -> Dict[str, Any]:
    domain_metrics = gate_metrics_by_domain(
        spike_prob,
        spike_label,
        threshold=float(threshold),
        domain_id=domain_id,
    )
    overall = dict(domain_metrics["overall"])
    source_metrics = gate_metrics_by_source(
        spike_prob,
        spike_label,
        threshold=float(threshold),
        sample_source=sample_source,
    )
    detector_summary = detector_domain_gate_summary(source_metrics)
    secondary_matched_positive_recall = float(detector_summary["secondary_matched_positive_recall"])
    detector_negative_noise_accuracy = float(detector_summary["detector_negative_noise_accuracy"])
    detector_domain_balanced_accuracy = float(detector_summary["detector_domain_balanced_accuracy"])
    detector_negative_pass_ratio = float(detector_summary["detector_negative_pass_ratio"])
    detector_negative_goal_reached = bool(
        not np.isfinite(detector_negative_noise_accuracy)
        or detector_negative_noise_accuracy + 1e-8 >= float(gate_goal_noise_accuracy)
    )
    goal_reached = bool(
        float(overall.get("gate_recall", float("nan"))) + 1e-8 >= float(gate_goal_spike_recall)
        and float(overall.get("noise_accuracy", float("nan"))) + 1e-8 >= float(gate_goal_noise_accuracy)
        and detector_negative_goal_reached
    )
    return {
        "fixed_gate_threshold": float(threshold),
        "encode_gate_thresh": float(threshold),
        "assignment_gate_thresh": float(threshold),
        "thresholds": {
            "encode_gate_thresh": float(threshold),
            "assignment_gate_thresh": float(threshold),
        },
        "selection_mode": "fixed_threshold",
        "selection_modes": {"encode": "fixed_threshold", "assignment": "fixed_threshold"},
        "encode_metrics": dict(overall),
        "assignment_metrics": dict(overall),
        "domain_metrics": dict(domain_metrics["by_domain"]),
        "source_metrics": dict(source_metrics),
        "gate_precision": float(overall.get("gate_precision", float("nan"))),
        "gate_recall": float(overall.get("gate_recall", float("nan"))),
        "gate_f1": float(overall.get("gate_f1", float("nan"))),
        "gate_f0_5": float(overall.get("gate_f0_5", float("nan"))),
        "noise_accuracy": float(overall.get("noise_accuracy", float("nan"))),
        "noise_specificity": float(overall.get("noise_specificity", overall.get("noise_accuracy", float("nan")))),
        "noise_recall": float(overall.get("noise_recall", float("nan"))),
        "balanced_accuracy": float(overall.get("balanced_accuracy", float("nan"))),
        "encode_pass_ratio": float(overall.get("encode_pass_ratio", float("nan"))),
        "assignment_pass_ratio": float(overall.get("encode_pass_ratio", float("nan"))),
        "positive_rate": float(overall.get("positive_rate", float("nan"))),
        "goal_spike_recall": float(gate_goal_spike_recall),
        "goal_noise_accuracy": float(gate_goal_noise_accuracy),
        "goal_reached": goal_reached,
        "fixed_gate_goal_reached": goal_reached,
        "detector_negative_noise_accuracy": detector_negative_noise_accuracy,
        "secondary_matched_positive_recall": secondary_matched_positive_recall,
        "detector_domain_balanced_accuracy": detector_domain_balanced_accuracy,
        "secondary_matched_positive_pass_ratio": float(detector_summary["secondary_matched_positive_pass_ratio"]),
        "detector_negative_pass_ratio": detector_negative_pass_ratio,
        "detector_negative_goal_reached": detector_negative_goal_reached,
    }


def _current_gate_metric_names(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metrics)
    out["gate_precision"] = float(out.get("gate_precision", out.get("precision", float("nan"))))
    out["gate_recall"] = float(out.get("gate_recall", out.get("recall", float("nan"))))
    out["gate_f1"] = float(out.get("gate_f1", out.get("f1", float("nan"))))
    out["gate_f0_5"] = float(out.get("gate_f0_5", out.get("f0_5", float("nan"))))
    out["noise_accuracy"] = float(out.get("noise_accuracy", out.get("noise_reject_rate", float("nan"))))
    out["noise_specificity"] = float(out.get("noise_specificity", out.get("noise_accuracy", float("nan"))))
    out["balanced_accuracy"] = float(out.get("balanced_accuracy", float("nan")))
    out["encode_pass_ratio"] = float(out.get("encode_pass_ratio", out.get("pass_ratio", float("nan"))))
    out["positive_rate"] = float(
        out.get(
            "positive_rate",
            float(out.get("positive", 0.0)) / max(1.0, float(out.get("n", out.get("n_samples", 0.0)))),
        )
    )
    out["n_samples"] = int(out.get("n_samples", out.get("n", 0)))
    out["n_positive"] = int(out.get("n_positive", out.get("positive", 0)))
    return out


def _gate_metrics_from_predictions(
    pred: np.ndarray,
    label: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, Any]:
    target = np.asarray(label, dtype=np.float32) > 0.5
    pred_bool = np.asarray(pred, dtype=np.bool_)
    if pred_bool.shape[0] != target.shape[0]:
        raise ValueError("pred and label must have matching length")
    tp = int(np.sum(pred_bool & target))
    fp = int(np.sum(pred_bool & (~target)))
    fn = int(np.sum((~pred_bool) & target))
    tn = int(np.sum((~pred_bool) & (~target)))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2.0 * precision * recall) / max(1e-8, precision + recall))
    noise_accuracy = float(tn / max(1, tn + fp))
    out = {
        "threshold": float(threshold),
        "n": int(target.shape[0]),
        "positive": int(np.sum(target)),
        "negative": int(np.sum(~target)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float((tp + tn) / max(1, target.shape[0])),
        "balanced_accuracy": float(0.5 * (recall + noise_accuracy)),
        "noise_reject_rate": noise_accuracy,
        "noise_accuracy": noise_accuracy,
        "pass_count": int(np.sum(pred_bool)),
        "pass_ratio": float(np.mean(pred_bool.astype(np.float32))) if pred_bool.size > 0 else float("nan"),
    }
    return _current_gate_metric_names(out)


def build_operational_gate_summary(
    spike_prob: np.ndarray,
    spike_label: np.ndarray,
    *,
    fixed_gate_threshold: float | None,
    gate_goal_spike_recall: float,
    gate_goal_noise_accuracy: float,
    domain_type: np.ndarray | None,
    sample_source: np.ndarray | None,
) -> Dict[str, Any]:
    prob = np.asarray(spike_prob, dtype=np.float32).reshape(-1)
    label = np.asarray(spike_label, dtype=np.float32).reshape(-1)
    if prob.shape[0] != label.shape[0]:
        raise ValueError("spike_prob and spike_label must have matching length")
    threshold = OPERATIONAL_FIXED_GATE_THRESHOLD if fixed_gate_threshold is None else float(fixed_gate_threshold)
    if not math.isclose(float(threshold), OPERATIONAL_FIXED_GATE_THRESHOLD, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("operational gate validation uses fixed threshold 0.5")
    domain_arr = _normalize_domain_type_array(domain_type, int(prob.shape[0]))
    domain_metrics: Dict[str, Dict[str, Any]] = {}
    domain_thresholds: Dict[str, float] = {}
    domain_selection_modes: Dict[str, str] = {}
    overall = _current_gate_metric_names(operational_binary_metrics(prob, label, threshold=threshold))
    for domain_value in np.unique(domain_arr).tolist():
        mask = domain_arr == str(domain_value)
        if not np.any(mask):
            continue
        metric = _current_gate_metric_names(operational_binary_metrics(prob[mask], label[mask], threshold=threshold))
        domain_metrics[str(domain_value)] = metric
        domain_thresholds[str(domain_value)] = threshold
        domain_selection_modes[str(domain_value)] = "fixed"
    selection_mode = "fixed_threshold"
    assignment_threshold = threshold
    source_metrics = gate_metrics_by_source(
        prob,
        label,
        threshold=float(assignment_threshold),
        sample_source=sample_source,
    )
    detector_summary = detector_domain_gate_summary(source_metrics)
    detector_negative_noise_accuracy = float(detector_summary["detector_negative_noise_accuracy"])
    detector_negative_goal_reached = bool(
        not np.isfinite(detector_negative_noise_accuracy)
        or detector_negative_noise_accuracy + 1e-8 >= float(gate_goal_noise_accuracy)
    )
    goal_reached = bool(
        float(overall.get("gate_recall", float("nan"))) + 1e-8 >= float(gate_goal_spike_recall)
        and float(overall.get("noise_accuracy", float("nan"))) + 1e-8 >= float(gate_goal_noise_accuracy)
        and detector_negative_goal_reached
    )
    return {
        "metric_style": "operational",
        "fixed_gate_threshold": float(threshold),
        "encode_gate_thresh": float(assignment_threshold),
        "assignment_gate_thresh": float(assignment_threshold),
        "thresholds": {
            "encode_gate_thresh": float(assignment_threshold),
            "assignment_gate_thresh": float(assignment_threshold),
        },
        "domain_type_thresholds": dict(domain_thresholds),
        "selection_mode": selection_mode,
        "selection_modes": {"encode": selection_mode, "assignment": selection_mode, "domain": dict(domain_selection_modes)},
        "encode_metrics": dict(overall),
        "assignment_metrics": dict(overall),
        "domain_metrics": dict(domain_metrics),
        "source_metrics": dict(source_metrics),
        "gate_precision": float(overall.get("gate_precision", float("nan"))),
        "gate_recall": float(overall.get("gate_recall", float("nan"))),
        "gate_f1": float(overall.get("gate_f1", float("nan"))),
        "gate_f0_5": float(overall.get("gate_f0_5", float("nan"))),
        "noise_accuracy": float(overall.get("noise_accuracy", float("nan"))),
        "noise_specificity": float(overall.get("noise_specificity", overall.get("noise_accuracy", float("nan")))),
        "balanced_accuracy": float(overall.get("balanced_accuracy", float("nan"))),
        "encode_pass_ratio": float(overall.get("encode_pass_ratio", float("nan"))),
        "assignment_pass_ratio": float(overall.get("encode_pass_ratio", float("nan"))),
        "positive_rate": float(overall.get("positive_rate", float("nan"))),
        "goal_spike_recall": float(gate_goal_spike_recall),
        "goal_noise_accuracy": float(gate_goal_noise_accuracy),
        "goal_reached": goal_reached,
        "fixed_gate_goal_reached": goal_reached,
        "detector_negative_noise_accuracy": detector_negative_noise_accuracy,
        "secondary_matched_positive_recall": float(detector_summary["secondary_matched_positive_recall"]),
        "detector_domain_balanced_accuracy": float(detector_summary["detector_domain_balanced_accuracy"]),
        "secondary_matched_positive_pass_ratio": float(detector_summary["secondary_matched_positive_pass_ratio"]),
        "detector_negative_pass_ratio": float(detector_summary["detector_negative_pass_ratio"]),
        "detector_negative_goal_reached": detector_negative_goal_reached,
    }


def _resolve_domain_ids(domain_id: np.ndarray | None, n_samples: int) -> np.ndarray:
    if domain_id is None:
        return np.zeros((int(n_samples),), dtype=np.int32)
    arr = np.asarray(domain_id)
    if arr.shape[0] != int(n_samples):
        raise ValueError("domain_id shape must match spike_prob/spike_label")
    return arr.astype(np.int32, copy=False)


def gate_metrics_by_domain(
    spike_prob: np.ndarray,
    spike_label: np.ndarray,
    *,
    threshold: float,
    domain_id: np.ndarray | None = None,
) -> Dict[str, Any]:
    prob = np.asarray(spike_prob, dtype=np.float32)
    label = np.asarray(spike_label, dtype=np.float32)
    domain_arr = _resolve_domain_ids(domain_id, prob.shape[0])
    overall = gate_metrics_from_probs(prob, label, threshold=float(threshold))
    by_domain: Dict[str, Dict[str, float]] = {}
    min_domain_recall = float("nan")
    recalls: List[float] = []
    for dom in np.unique(domain_arr).tolist():
        mask = domain_arr == int(dom)
        metrics = gate_metrics_from_probs(prob[mask], label[mask], threshold=float(threshold))
        n_samples = int(np.sum(mask))
        n_positive = int(np.sum(label[mask] > 0.5))
        metrics["n_samples"] = n_samples
        metrics["n_positive"] = n_positive
        by_domain[str(int(dom))] = metrics
        if n_positive > 0 and np.isfinite(metrics["gate_recall"]):
            recalls.append(float(metrics["gate_recall"]))
    if recalls:
        min_domain_recall = float(min(recalls))
    overall["min_domain_recall"] = float(min_domain_recall)
    overall["n_domains"] = int(len(by_domain))
    overall["n_domains_with_positive"] = int(sum(1 for m in by_domain.values() if int(m["n_positive"]) > 0))
    return {
        "overall": overall,
        "by_domain": by_domain,
    }


def _select_encode_threshold(
    candidates: List[Dict[str, Any]],
    *,
    target_recall: float,
) -> tuple[Dict[str, Any], str]:
    eligible = [c for c in candidates if float(c["overall"]["min_domain_recall"]) + 1e-8 >= float(target_recall)]
    if eligible:
        chosen = max(eligible, key=lambda c: float(c["overall"]["threshold"]))
        return chosen, "target_recall"
    chosen = max(
        candidates,
        key=lambda c: (
            float(c["overall"]["min_domain_recall"]) if np.isfinite(c["overall"]["min_domain_recall"]) else -1.0,
            float(c["overall"]["gate_recall"]),
            float(c["overall"]["threshold"]),
        ),
    )
    return chosen, "max_min_domain_recall"


def _select_assignment_threshold(
    candidates: List[Dict[str, Any]],
    *,
    target_recall_min: float,
) -> tuple[Dict[str, Any], str]:
    eligible = [c for c in candidates if float(c["overall"]["min_domain_recall"]) + 1e-8 >= float(target_recall_min)]
    if eligible:
        chosen = max(
            eligible,
            key=lambda c: (float(c["overall"]["gate_f1"]), float(c["overall"]["threshold"])),
        )
        return chosen, "target_recall_max_f1"
    chosen = max(
        candidates,
        key=lambda c: (float(c["overall"]["gate_f1"]), float(c["overall"]["threshold"])),
    )
    return chosen, "max_f1"


def calibrate_gate_thresholds_dual(
    spike_prob: np.ndarray,
    spike_label: np.ndarray,
    *,
    target_recall_encode: float,
    target_recall_assignment_min: float,
    domain_id: np.ndarray | None = None,
) -> Dict[str, Any]:
    if spike_prob.size == 0:
        return {
            "encode_gate_thresh": float("nan"),
            "assignment_gate_thresh": float("nan"),
            "gate_precision": float("nan"),
            "gate_recall": float("nan"),
            "gate_f1": float("nan"),
            "encode_pass_ratio": float("nan"),
            "assignment_pass_ratio": float("nan"),
            "target_recall": float(target_recall_encode),
            "target_recall_encode": float(target_recall_encode),
            "target_recall_assignment_min": float(target_recall_assignment_min),
            "selection_mode": "empty",
            "selection_modes": {"encode": "empty", "assignment": "empty"},
            "all_positive_precision": float("nan"),
            "all_positive_f1": float("nan"),
            "positive_rate": float("nan"),
            "thresholds": {
                "encode_gate_thresh": float("nan"),
                "assignment_gate_thresh": float("nan"),
            },
            "encode_metrics": {},
            "assignment_metrics": {},
            "domain_metrics": {},
        }
    prob = np.asarray(spike_prob, dtype=np.float32)
    label = np.asarray(spike_label, dtype=np.float32)
    domain_arr = _resolve_domain_ids(domain_id, prob.shape[0])
    thresholds = np.linspace(0.0, 1.0, num=100, dtype=np.float32)
    candidates: List[Dict[str, Any]] = []
    for thr in thresholds.tolist():
        candidates.append(
            gate_metrics_by_domain(
                prob,
                label,
                threshold=float(thr),
                domain_id=domain_arr,
            )
        )

    encode_choice, encode_mode = _select_encode_threshold(candidates, target_recall=float(target_recall_encode))
    assignment_choice, assignment_mode = _select_assignment_threshold(
        candidates,
        target_recall_min=float(target_recall_assignment_min),
    )
    baseline = gate_metrics_by_domain(
        prob,
        label,
        threshold=0.0,
        domain_id=domain_arr,
    )["overall"]
    domain_metrics: Dict[str, Dict[str, Any]] = {}
    for dom_key in sorted(encode_choice["by_domain"].keys(), key=lambda x: int(x)):
        encode_metrics = dict(encode_choice["by_domain"][dom_key])
        assignment_metrics = dict(assignment_choice["by_domain"].get(dom_key, {}))
        domain_metrics[dom_key] = {
            "n_samples": int(encode_metrics.get("n_samples", assignment_metrics.get("n_samples", 0))),
            "n_positive": int(encode_metrics.get("n_positive", assignment_metrics.get("n_positive", 0))),
            "encode": encode_metrics,
            "assignment": assignment_metrics,
        }
    out = {
        "encode_gate_thresh": float(encode_choice["overall"]["threshold"]),
        "assignment_gate_thresh": float(assignment_choice["overall"]["threshold"]),
        "thresholds": {
            "encode_gate_thresh": float(encode_choice["overall"]["threshold"]),
            "assignment_gate_thresh": float(assignment_choice["overall"]["threshold"]),
        },
        "encode_metrics": dict(encode_choice["overall"]),
        "assignment_metrics": dict(assignment_choice["overall"]),
        "domain_metrics": domain_metrics,
        "gate_precision": float(assignment_choice["overall"]["gate_precision"]),
        "gate_recall": float(assignment_choice["overall"]["gate_recall"]),
        "gate_f1": float(assignment_choice["overall"]["gate_f1"]),
        "encode_pass_ratio": float(encode_choice["overall"]["encode_pass_ratio"]),
        "assignment_pass_ratio": float(assignment_choice["overall"]["encode_pass_ratio"]),
        "positive_rate": float(assignment_choice["overall"]["positive_rate"]),
        "target_recall": float(target_recall_encode),
        "target_recall_encode": float(target_recall_encode),
        "target_recall_assignment_min": float(target_recall_assignment_min),
        "selection_mode": "dual_threshold",
        "selection_modes": {"encode": encode_mode, "assignment": assignment_mode},
        "all_positive_precision": float(baseline["gate_precision"]),
        "all_positive_f1": float(baseline["gate_f1"]),
    }
    return out


def calibrate_gate_threshold(
    spike_prob: np.ndarray,
    spike_label: np.ndarray,
    *,
    target_recall: float,
) -> Dict[str, Any]:
    return calibrate_gate_thresholds_dual(
        spike_prob,
        spike_label,
        target_recall_encode=float(target_recall),
        target_recall_assignment_min=float(target_recall),
        domain_id=None,
    )


def evaluate_gate_metrics(
    model: SpikeEmbeddingModel,
    x_np: np.ndarray,
    geo_np: np.ndarray,
    spike_label: np.ndarray,
    spatial_stats_np: np.ndarray | None = None,
    channel_mask_np: np.ndarray | None = None,
    *,
    device: torch.device,
    batch_size: int,
    threshold: float | None = None,
) -> Dict[str, float]:
    prob = collect_gate_probabilities(
        model,
        x_np,
        geo_np,
        spatial_stats_np=spatial_stats_np,
        channel_mask_np=channel_mask_np,
        device=device,
        batch_size=batch_size,
    )
    default_threshold = model.cfg.encode_gate_thresh
    if model.cfg.fixed_gate_threshold is not None:
        default_threshold = float(model.cfg.fixed_gate_threshold)
    use_threshold = float(default_threshold if threshold is None else threshold)
    return gate_metrics_from_probs(prob, spike_label, threshold=use_threshold)


def format_domain_gate_metrics(domain_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    def _domain_sort_key(value: str) -> tuple[int, Any]:
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))

    for dom_key in sorted(domain_metrics.keys(), key=_domain_sort_key):
        dom = domain_metrics[dom_key]
        if "encode" not in dom and "assignment" not in dom:
            lines.append(
                (
                    "domain=%s n=%d pos=%d spike_recall=%.4f spike_f1=%.4f "
                    "noise_acc=%.4f balanced_acc=%.4f pass=%.4f"
                )
                % (
                    dom_key,
                    int(dom.get("n_samples", 0)),
                    int(dom.get("n_positive", 0)),
                    float(dom.get("gate_recall", float("nan"))),
                    float(dom.get("gate_f1", float("nan"))),
                    float(dom.get("noise_accuracy", float("nan"))),
                    float(dom.get("balanced_accuracy", float("nan"))),
                    float(dom.get("encode_pass_ratio", float("nan"))),
                )
            )
            continue
        encode = dict(dom.get("encode", {}))
        assignment = dict(dom.get("assignment", {}))
        lines.append(
            (
                "domain=%s n=%d pos=%d encode_recall=%.4f encode_pass=%.4f "
                "assignment_recall=%.4f assignment_f1=%.4f assignment_pass=%.4f"
            )
            % (
                dom_key,
                int(dom.get("n_samples", 0)),
                int(dom.get("n_positive", 0)),
                float(encode.get("gate_recall", float("nan"))),
                float(encode.get("encode_pass_ratio", float("nan"))),
                float(assignment.get("gate_recall", float("nan"))),
                float(assignment.get("gate_f1", float("nan"))),
                float(assignment.get("encode_pass_ratio", float("nan"))),
            )
        )
    return lines


def gate_metrics_by_source(
    spike_prob: np.ndarray,
    spike_label: np.ndarray,
    *,
    threshold: float,
    sample_source: np.ndarray | None,
) -> Dict[str, Dict[str, float]]:
    if sample_source is None:
        return {}
    prob = np.asarray(spike_prob, dtype=np.float32)
    label = np.asarray(spike_label, dtype=np.float32)
    source_arr = np.asarray(sample_source, dtype=np.int8)
    if source_arr.shape[0] != prob.shape[0]:
        raise ValueError("sample_source shape must match spike_prob/spike_label")
    out: Dict[str, Dict[str, float]] = {}
    for source_name in GATE_SOURCE_ORDER:
        source_code = int(GATE_SOURCE_NAME_TO_CODE[source_name])
        mask = source_arr == source_code
        if not np.any(mask):
            continue
        metrics = gate_metrics_from_probs(prob[mask], label[mask], threshold=float(threshold))
        metrics["n_samples"] = int(np.sum(mask))
        metrics["n_positive"] = int(np.sum(label[mask] > 0.5))
        out[source_name] = metrics
    return out


def format_source_gate_metrics(source_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for source_name in GATE_SOURCE_ORDER:
        metrics = source_metrics.get(source_name)
        if not metrics:
            continue
        lines.append(
            (
                "source=%s n=%d pos=%d precision=%.4f recall=%.4f f1=%.4f noise_acc=%.4f pass=%.4f"
            )
            % (
                source_name,
                int(metrics.get("n_samples", 0)),
                int(metrics.get("n_positive", 0)),
                float(metrics.get("gate_precision", float("nan"))),
                float(metrics.get("gate_recall", float("nan"))),
                float(metrics.get("gate_f1", float("nan"))),
                float(metrics.get("noise_accuracy", float("nan"))),
                float(metrics.get("encode_pass_ratio", float("nan"))),
            )
        )
    return lines


def detector_domain_gate_summary(source_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    det_pos = dict(source_metrics.get("secondary_matched_positive", {}))
    det_neg = dict(source_metrics.get("detector_negative", {}))
    det_pos_recall = float(det_pos.get("gate_recall", float("nan")))
    det_neg_noise_accuracy = float(det_neg.get("noise_accuracy", float("nan")))
    parts = [v for v in (det_pos_recall, det_neg_noise_accuracy) if np.isfinite(v)]
    return {
        "secondary_matched_positive_recall": det_pos_recall,
        "detector_negative_noise_accuracy": det_neg_noise_accuracy,
        "detector_domain_balanced_accuracy": float(np.mean(np.asarray(parts, dtype=np.float32))) if parts else float("nan"),
        "secondary_matched_positive_pass_ratio": float(det_pos.get("encode_pass_ratio", float("nan"))),
        "detector_negative_pass_ratio": float(det_neg.get("encode_pass_ratio", float("nan"))),
    }


def _binary_label_counts(labels: np.ndarray) -> Tuple[int, int]:
    target = (np.asarray(labels, dtype=np.float32) > 0.5).astype(np.int8, copy=False)
    pos = int(np.sum(target == 1))
    neg = int(np.sum(target == 0))
    return pos, neg


def _normalize_domain_type_array(domain_type: np.ndarray | None, n_samples: int) -> np.ndarray:
    if domain_type is None:
        return np.asarray(["unknown"] * int(n_samples), dtype=str)
    arr = np.asarray(domain_type).astype(str).reshape(-1)
    if arr.shape[0] != int(n_samples):
        raise ValueError("domain_type must have the same length as spike_label")
    return arr


def canonical_gate_sampler_mode(raw_mode: Any, *, logger: logging.Logger | None = None) -> str:
    raw = str(raw_mode or "natural").strip().lower()
    canonical = GATE_SAMPLER_MODE_ALIASES.get(raw, raw)
    if canonical != raw and logger is not None:
        logger.warning(
            "deprecated train.gate.sampler_mode=%s; use %s",
            raw,
            canonical,
        )
    return canonical


def canonical_gate_loss_weight_mode(raw_mode: Any, *, logger: logging.Logger | None = None) -> str:
    raw = str(raw_mode or "none").strip().lower()
    canonical = GATE_LOSS_WEIGHT_MODE_ALIASES.get(raw, raw)
    if canonical != raw and logger is not None:
        logger.warning(
            "deprecated train.gate.loss_weight_mode=%s; use %s",
            raw,
            canonical,
        )
    return canonical


def compute_natural_gate_sampler_probs(
    spike_label: np.ndarray,
    *,
    training_mask: np.ndarray,
) -> np.ndarray:
    label = np.asarray(spike_label, dtype=np.float32).reshape(-1)
    train_mask = np.asarray(training_mask, dtype=np.bool_).reshape(-1)
    if train_mask.shape[0] != label.shape[0]:
        raise ValueError("spike_label and training_mask must have matching length")
    return operational_compute_natural_sampler_probs({"spike_label": label, "training_mask": train_mask})


def build_gate_batch_sampler_natural(
    spike_label: np.ndarray,
    *,
    training_mask: np.ndarray,
    batch_size: int,
    seed: int,
) -> Callable[[int | None], np.ndarray]:
    probs = compute_natural_gate_sampler_probs(spike_label, training_mask=training_mask)
    indices = np.arange(probs.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    batch_size = max(1, int(batch_size))

    def sample_gate_batch(epoch: int | None = None) -> np.ndarray:
        _ = epoch
        return rng.choice(indices, size=batch_size, replace=True, p=probs).astype(np.int64, copy=False)

    return sample_gate_batch


def resolve_embedding_class_indices(
    unit_id: np.ndarray,
    sample_source: np.ndarray | None,
    *,
    epoch: int | None,
    embedding_gt_warmup_epochs: int,
) -> Dict[int, np.ndarray]:
    unit_arr = np.asarray(unit_id, dtype=np.int64)
    source_arr = np.asarray(sample_source, dtype=np.int8) if sample_source is not None else None
    positive_idx = np.flatnonzero(unit_arr >= 0).astype(np.int64, copy=False)
    classes = np.unique(unit_arr[positive_idx]).astype(np.int64, copy=False) if positive_idx.size > 0 else np.zeros((0,), dtype=np.int64)
    class_indices_all: Dict[int, np.ndarray] = {}
    for c in classes.tolist():
        class_indices_all[int(c)] = np.flatnonzero(unit_arr == int(c)).astype(np.int64, copy=False)
    use_primary_only = bool(
        source_arr is not None
        and embedding_gt_warmup_epochs > 0
        and epoch is not None
        and int(epoch) < int(embedding_gt_warmup_epochs)
    )
    if not use_primary_only:
        return {int(c): idxs.copy() for c, idxs in class_indices_all.items()}
    out: Dict[int, np.ndarray] = {}
    gt_code = int(GATE_SOURCE_NAME_TO_CODE["primary_matched_positive"])
    for c, idxs in class_indices_all.items():
        gt_idx = idxs[source_arr[idxs] == gt_code]
        out[int(c)] = gt_idx.astype(np.int64, copy=False) if gt_idx.size > 0 else idxs.copy()
    return out


def resolve_embedding_positive_indices(
    unit_id: np.ndarray,
    sample_source: np.ndarray | None,
    *,
    epoch: int | None,
    embedding_gt_warmup_epochs: int,
) -> np.ndarray:
    class_indices = resolve_embedding_class_indices(
        unit_id,
        sample_source,
        epoch=epoch,
        embedding_gt_warmup_epochs=embedding_gt_warmup_epochs,
    )
    if not class_indices:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(list(class_indices.values()), axis=0).astype(np.int64, copy=False)


def compute_gate_consistency_loss(
    teacher_embedding: torch.Tensor,
    student_embedding: torch.Tensor,
    labels: torch.Tensor,
    encoded_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    keep_mask = (labels >= 0) & encoded_mask.to(dtype=torch.bool)
    n_keep = int(torch.sum(keep_mask).detach().cpu().item())
    if n_keep <= 0:
        return teacher_embedding.new_zeros(()), 0
    teacher = F.normalize(teacher_embedding[keep_mask].detach(), p=2, dim=1)
    student = F.normalize(student_embedding[keep_mask], p=2, dim=1)
    cosine = torch.sum(student * teacher, dim=1)
    return torch.mean(1.0 - cosine), n_keep


def filter_embedding_loss_batch(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    *,
    dch: torch.Tensor | None = None,
    encoded_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    keep_mask = labels >= 0
    if encoded_mask is not None:
        keep_mask = keep_mask & encoded_mask.to(dtype=torch.bool)
    n_keep = int(torch.sum(keep_mask).detach().cpu().item())
    if n_keep <= 0:
        empty_z = embedding.new_zeros((0, embedding.shape[1]))
        empty_y = labels.new_zeros((0,), dtype=labels.dtype)
        empty_dch = dch.new_zeros((0, 0)) if dch is not None else None
        return empty_z, empty_y, empty_dch, 0
    z_keep = embedding[keep_mask]
    y_keep = labels[keep_mask]
    dch_keep = None
    if dch is not None:
        if dch.ndim != 2 or dch.shape[0] != embedding.shape[0] or dch.shape[1] != embedding.shape[0]:
            raise ValueError(
                "dch must be a square matrix whose size matches the embedding batch before gate filtering"
            )
        dch_keep = dch[keep_mask][:, keep_mask]
    return z_keep, y_keep, dch_keep, n_keep


def build_mearec_loader_kwargs(args: argparse.Namespace, data_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "top_k_channels": int(args.top_k_channels if args.top_k_channels is not None else data_cfg.get("top_k_channels", 16)),
        "normalize": bool(data_cfg.get("normalize_templates", True)),
        "max_per_unit": int(args.max_per_unit if args.max_per_unit is not None else data_cfg.get("max_per_unit", 0)),
        "gt_max_per_unit": None if data_cfg.get("gt_max_per_unit") is None else int(data_cfg.get("gt_max_per_unit")),
        "seed": int(seed),
        "merge_dt_ms": float(args.merge_dt_ms if args.merge_dt_ms is not None else data_cfg.get("merge_dt_ms", 0.5)),
        "dedup_enabled": bool(data_cfg.get("dedup_spikes", True)),
        "dedup_mode": str(args.dedup_mode if args.dedup_mode is not None else data_cfg.get("dedup_mode", "neighbors")),
        "electrode_config": args.electrode_config or data_cfg.get("electrode_config"),
        "peak_only": bool(data_cfg.get("peak_only", False)),
        "align_radius_ms": float(data_cfg.get("align_radius_ms", 0.2)),
        "align_polarity": str(data_cfg.get("align_polarity", "negative")),
        "gt_time_reference": str(data_cfg.get("gt_time_reference", "near_peak")),
        "car_mode": str(data_cfg.get("car_mode", "auto")),
        "normalization_mode": str(data_cfg.get("normalization_mode", "peak_channel_sigma")),
        "robust_norm_samples": int(data_cfg.get("robust_norm_samples", 60000)),
        "min_snr": float(data_cfg.get("min_snr", 4.0)),
        "gt_min_snr": None if data_cfg.get("gt_min_snr") is None else float(data_cfg.get("gt_min_snr")),
        "detector_min_snr": None if data_cfg.get("detector_min_snr") is None else float(data_cfg.get("detector_min_snr")),
        "min_unit_snr": float(data_cfg.get("min_unit_snr", 0.0)),
        "max_snr": float(data_cfg.get("max_snr", 0.0)),
        "drop_border_spikes": bool(data_cfg.get("drop_border_spikes", True)),
        "sample_strategy": str(data_cfg.get("sample_strategy", "uniform_time")),
        "include_detector_candidates": bool(data_cfg.get("include_detector_candidates", True)),
        "secondary_matched_positive_ratio": float(data_cfg.get("secondary_matched_positive_ratio", 1.0)),
        "detector_negative_ratio": float(data_cfg.get("detector_negative_ratio", 2.0)),
        "detector_negative_exclusion_ms": float(data_cfg.get("detector_negative_exclusion_ms", 0.6)),
        "secondary_matched_positive_min_spatial_score": float(data_cfg.get("secondary_matched_positive_min_spatial_score", 0.05)),
        "detector_negative_ambiguous_tol_ms": float(data_cfg.get("detector_negative_ambiguous_tol_ms", 0.6)),
        "secondary_matched_positive_tol_samples": (
            None
            if data_cfg.get("secondary_matched_positive_tol_samples", 3) is None
            else int(data_cfg.get("secondary_matched_positive_tol_samples", 3))
        ),
        "detector_disable_negative_for_clean": bool(data_cfg.get("detector_disable_negative_for_clean", False)),
        "detector_chunk_sec": float(data_cfg.get("detector_chunk_sec", 0.5)),
        "detector_tol_ms": float(data_cfg.get("detector_tol_ms", 0.3)),
        "strict_overlap_ms": float(data_cfg.get("strict_overlap_ms", data_cfg.get("strict_overlap", {}).get("ms", 0.5) if isinstance(data_cfg.get("strict_overlap"), dict) else 0.5)),
        "strict_overlap_um": float(data_cfg.get("strict_overlap_um", data_cfg.get("strict_overlap", {}).get("um", 60.0) if isinstance(data_cfg.get("strict_overlap"), dict) else 60.0)),
        "detector_config": dict(data_cfg.get("detector", {}) or {}),
        "detector_alpha_overrides": dict(data_cfg.get("detector_alpha_overrides", {}) or {}),
    }
    if args.no_dedup_spikes:
        kwargs["dedup_enabled"] = False
        kwargs["dedup_mode"] = "none"
    if args.no_normalize_templates:
        kwargs["normalize"] = False
    return kwargs


def format_snr_summary(label: str, snr_peak: np.ndarray | None) -> str:
    if snr_peak is None:
        return f"{label}=missing"
    arr = np.asarray(snr_peak, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"{label}=empty"
    return (
        f"{label}_n={int(arr.size)} {label}_p10={float(np.percentile(arr, 10)):.4f} "
        f"{label}_p50={float(np.percentile(arr, 50)):.4f} {label}_p90={float(np.percentile(arr, 90)):.4f}"
    )


def derive_spatial_stats_from_patches(patches: np.ndarray) -> np.ndarray:
    x = np.asarray(patches, dtype=np.float32)
    if x.ndim != 3:
        return np.zeros((int(x.shape[0]) if x.ndim > 0 else 0, 2), dtype=np.float32)
    peak_per_ch = np.max(np.abs(x), axis=1)
    amp_proxy = np.max(peak_per_ch, axis=1, keepdims=True) / 30.0
    mean_amp = np.mean(peak_per_ch, axis=1, keepdims=True)
    max_amp = np.max(peak_per_ch, axis=1, keepdims=True)
    spatial_proxy = mean_amp / np.maximum(max_amp, 1e-6)
    return np.concatenate([amp_proxy, spatial_proxy], axis=1).astype(np.float32, copy=False)


def resolve_label_spatial_stats(
    labels: Dict[str, np.ndarray],
    patches: np.ndarray,
    *,
    logger: logging.Logger | None,
    context: str,
) -> np.ndarray:
    n_samples = int(patches.shape[0])
    spatial_stats = labels.get("spatial_stats")
    if spatial_stats is not None:
        arr = np.asarray(spatial_stats, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == n_samples:
            return arr.astype(np.float32, copy=False)
        if logger is not None:
            logger.warning(
                "%s labels spatial_stats invalid shape=%s expected_first_dim=%d; deriving gate spatial stats from patches",
                context,
                str(arr.shape),
                n_samples,
            )
    elif logger is not None:
        logger.warning("%s labels missing spatial_stats; deriving gate spatial stats from patches", context)
    return derive_spatial_stats_from_patches(patches)


def summarize_sample_sources(sample_source: np.ndarray | None) -> Dict[str, int]:
    out = {
        "primary_matched_positive": 0,
        "secondary_matched_positive": 0,
        "detector_negative": 0,
        "unknown": 0,
    }
    if sample_source is None:
        out["unknown"] = -1
        return out
    arr = np.asarray(sample_source, dtype=np.int8)
    out["primary_matched_positive"] = int(np.sum(arr == 0))
    out["secondary_matched_positive"] = int(np.sum(arr == 1))
    out["detector_negative"] = int(np.sum(arr == 2))
    out["unknown"] = int(np.sum(arr < 0))
    return out


def accumulate_gate_source_counts(counter: Dict[str, int], sample_source: np.ndarray | None) -> None:
    counts = summarize_sample_sources(sample_source)
    for source_name in GATE_SOURCE_ORDER:
        counter[source_name] = int(counter.get(source_name, 0)) + int(counts.get(source_name, 0))


def format_gate_source_mix(counter: Dict[str, int]) -> str:
    total = int(sum(int(counter.get(source_name, 0)) for source_name in GATE_SOURCE_ORDER))
    if total <= 0:
        return "none"
    parts = []
    for source_name in GATE_SOURCE_ORDER:
        count = int(counter.get(source_name, 0))
        parts.append(f"{source_name}={count}({count / total:.3f})")
    return " ".join(parts)


def accumulate_recording_counts(counter: Dict[int, int], recording_id: np.ndarray) -> None:
    values, counts = np.unique(np.asarray(recording_id, dtype=np.int64), return_counts=True)
    for rec_value, rec_count in zip(values.tolist(), counts.tolist()):
        rec_key = int(rec_value)
        counter[rec_key] = int(counter.get(rec_key, 0)) + int(rec_count)


def format_recording_mix(counter: Dict[int, int]) -> str:
    total = int(sum(int(v) for v in counter.values()))
    if total <= 0:
        return "none"
    parts = []
    for rec_value in sorted(counter.keys()):
        count = int(counter[rec_value])
        parts.append(f"recording_{int(rec_value)}={count}({count / total:.3f})")
    return " ".join(parts)


def run_uid_consistency_preflight(
    h5_paths: List[str | Path],
    *,
    logger: logging.Logger,
    seed: int,
    enabled: bool,
    left: int = 20,
    right: int = 44,
    max_spikes_per_uid: int = 200,
    time_atol_sec: float = 1e-7,
    wave_corr_thresh: float = 0.8,
) -> List[Dict[str, Any]]:
    resolved = [Path(p) for p in h5_paths]
    if not enabled or len(resolved) <= 1:
        return []
    logger.info(
        "uid_preflight start files=%d left=%d right=%d max_spikes_per_uid=%d time_atol_sec=%.2e wave_corr_thresh=%.4f",
        len(resolved),
        int(left),
        int(right),
        int(max_spikes_per_uid),
        float(time_atol_sec),
        float(wave_corr_thresh),
    )
    infos = [
        load_file_info(
            path,
            left=int(left),
            right=int(right),
            max_spikes_per_uid=int(max_spikes_per_uid),
            seed=int(seed + idx),
        )
        for idx, path in enumerate(resolved)
    ]
    reports: List[Dict[str, Any]] = []
    all_ok = True
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            report = compare_files(
                infos[i],
                infos[j],
                atol_sec=float(time_atol_sec),
                wave_corr_thresh=float(wave_corr_thresh),
            )
            reports.append(report)
            wave_ratio = report.get("wave_corr_ge_thresh_ratio")
            logger.info(
                "uid_preflight_pair base=%s other=%s shared=%d close_time=%.4f overlap=%.4f same_count=%.4f wave_corr_ge_thresh=%s likely_same=%s",
                Path(str(report["base_file"])) .name,
                Path(str(report["other_file"])) .name,
                int(report.get("shared_uids", 0)),
                float(report.get("close_time_match_ratio", 0.0)),
                float(report.get("mean_sample_overlap_ratio", 0.0)),
                float(report.get("same_spike_count_ratio", 0.0)),
                "nan" if wave_ratio is None else f"{float(wave_ratio):.4f}",
                str(bool(report.get("likely_same_uid_semantics", False))),
            )
            mismatch_examples = list(report.get("mismatched_uid_examples", []))
            if mismatch_examples:
                first = dict(mismatch_examples[0])
                logger.warning(
                    "uid_preflight_mismatch base=%s other=%s uid=%d count_base=%d count_other=%d overlap=%.4f wave_corr=%s",
                    Path(str(report["base_file"])) .name,
                    Path(str(report["other_file"])) .name,
                    int(first.get("uid", -1)),
                    int(first.get("count_base", 0)),
                    int(first.get("count_other", 0)),
                    float(first.get("sample_overlap_ratio", 0.0)),
                    str(first.get("wave_corr")),
                )
            all_ok = all_ok and bool(report.get("likely_same_uid_semantics", False))
    logger.info("uid_preflight done all_ok=%s", str(bool(all_ok)))
    if not all_ok:
        raise RuntimeError(
            "UID consistency preflight failed: same uid across MEArec files does not look like the same neuron. "
            "Please fix the data or disable the preflight explicitly."
        )
    return reports


def summarize_unit_domain_support(
    support_map: Dict[int, np.ndarray],
    *,
    min_domains_required: int,
) -> Dict[str, float]:
    counts = np.asarray([np.asarray(v).size for v in support_map.values()], dtype=np.int64)
    if counts.size == 0:
        return {
            "n_units": 0.0,
            "eligible_units": 0.0,
            "mean_domains": 0.0,
            "max_domains": 0.0,
        }
    return {
        "n_units": float(counts.size),
        "eligible_units": float(np.sum(counts >= int(min_domains_required))),
        "mean_domains": float(np.mean(counts.astype(np.float32))),
        "max_domains": float(np.max(counts)),
    }


def build_gt_spike_mask(
    unit_id: np.ndarray,
    spike_label: np.ndarray | None,
    sample_source: np.ndarray | None,
) -> np.ndarray:
    unit_arr = np.asarray(unit_id, dtype=np.int64)
    keep_mask = unit_arr >= 0
    if spike_label is not None:
        spike_arr = np.asarray(spike_label, dtype=np.float32)
        if spike_arr.shape[0] != unit_arr.shape[0]:
            raise ValueError("spike_label must match unit_id shape")
        keep_mask = keep_mask & (spike_arr > 0.5)
    if sample_source is not None:
        source_arr = np.asarray(sample_source, dtype=np.int8)
        if source_arr.shape[0] != unit_arr.shape[0]:
            raise ValueError("sample_source must match unit_id shape")
        keep_mask = keep_mask & (source_arr == int(GATE_SOURCE_NAME_TO_CODE["primary_matched_positive"]))
    return keep_mask



def build_stage2_positive_mask(
    unit_id: np.ndarray,
    spike_label: np.ndarray | None,
    sample_source: np.ndarray | None,
    *,
    source_mode: str = "primary_only",
) -> np.ndarray:
    unit_arr = np.asarray(unit_id, dtype=np.int64)
    keep_mask = unit_arr >= 0
    if spike_label is not None:
        spike_arr = np.asarray(spike_label, dtype=np.float32)
        if spike_arr.shape[0] != unit_arr.shape[0]:
            raise ValueError("spike_label must match unit_id shape")
        keep_mask = keep_mask & (spike_arr > 0.5)
    normalized_mode = str(source_mode).strip().lower()
    if normalized_mode not in {"primary_only", "matched_positives"}:
        raise ValueError(f"unsupported stage2 source_mode: {source_mode}")
    if sample_source is not None:
        source_arr = np.asarray(sample_source, dtype=np.int8)
        if source_arr.shape[0] != unit_arr.shape[0]:
            raise ValueError("sample_source must match unit_id shape")
        if normalized_mode == "primary_only":
            allowed_codes = np.asarray([int(GATE_SOURCE_NAME_TO_CODE["primary_matched_positive"])], dtype=np.int8)
        elif normalized_mode == "matched_positives":
            allowed_codes = np.asarray(
                [
                    int(GATE_SOURCE_NAME_TO_CODE["primary_matched_positive"]),
                    int(GATE_SOURCE_NAME_TO_CODE["secondary_matched_positive"]),
                ],
                dtype=np.int8,
            )
        keep_mask = keep_mask & np.isin(source_arr, allowed_codes)
    return keep_mask



def summarize_gate_pass_ratios_by_domain(
    recording_id: np.ndarray,
    gt_mask: np.ndarray,
    gate_pass_mask: np.ndarray,
) -> Dict[int, float]:
    recording_arr = np.asarray(recording_id, dtype=np.int16)
    gt_arr = np.asarray(gt_mask, dtype=np.bool_)
    pass_arr = np.asarray(gate_pass_mask, dtype=np.bool_)
    ratios: Dict[int, float] = {}
    if recording_arr.shape[0] != gt_arr.shape[0] or gt_arr.shape[0] != pass_arr.shape[0]:
        raise ValueError("recording_id, gt_mask, and gate_pass_mask must have matching shape")
    if not np.any(gt_arr):
        return ratios
    for rec_value in np.unique(recording_arr[gt_arr]).tolist():
        rec_mask = gt_arr & (recording_arr == int(rec_value))
        denom = int(np.sum(rec_mask))
        if denom <= 0:
            continue
        ratios[int(rec_value)] = float(np.sum(pass_arr & rec_mask) / denom)
    return ratios



def format_gate_pass_ratio_summary(ratios: Dict[int, float]) -> str:
    if not ratios:
        return "none"
    return " ".join(f"{int(dom)}={float(val):.4f}" for dom, val in sorted(ratios.items()))



def resolve_stage2_gt_class_indices(
    unit_id: np.ndarray,
    recording_id: np.ndarray,
    spike_label: np.ndarray | None,
    sample_source: np.ndarray | None,
    *,
    min_domains_per_unit: int,
    source_mode: str = "primary_only",
    extra_mask: np.ndarray | None = None,
    min_samples_per_unit: int = 1,
) -> tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray]]:
    unit_arr = np.asarray(unit_id, dtype=np.int64)
    recording_arr = np.asarray(recording_id, dtype=np.int16)
    keep_mask = build_stage2_positive_mask(unit_arr, spike_label, sample_source, source_mode=source_mode)
    if extra_mask is not None:
        extra_arr = np.asarray(extra_mask, dtype=np.bool_)
        if extra_arr.shape[0] != unit_arr.shape[0]:
            raise ValueError("extra_mask must match unit_id shape")
        keep_mask = keep_mask & extra_arr
    if not np.any(keep_mask):
        return {}, np.zeros((0,), dtype=np.int64), {}
    filtered_units = unit_arr[keep_mask]
    filtered_recordings = recording_arr[keep_mask]
    support_map = build_unit_recording_support(filtered_units, filtered_recordings, primary_only=False)
    min_domains = int(max(1, min_domains_per_unit))
    min_samples = int(max(1, min_samples_per_unit))
    eligible_units = sorted(
        int(unit_value)
        for unit_value, recs in support_map.items()
        if np.asarray(recs).size >= min_domains and int(np.sum(filtered_units == int(unit_value))) >= min_samples
    )
    if not eligible_units:
        return {}, np.zeros((0,), dtype=np.int64), support_map
    allowed_mask = np.isin(unit_arr, np.asarray(eligible_units, dtype=np.int64))
    final_mask = keep_mask & allowed_mask
    class_indices: Dict[int, np.ndarray] = {}
    for unit_value in eligible_units:
        idx = np.flatnonzero(final_mask & (unit_arr == int(unit_value))).astype(np.int64, copy=False)
        if idx.size > 0:
            class_indices[int(unit_value)] = idx
    if not class_indices:
        return {}, np.zeros((0,), dtype=np.int64), support_map
    positive_indices = np.concatenate(list(class_indices.values()), axis=0).astype(np.int64, copy=False)
    return class_indices, positive_indices, support_map


def build_stage2_quality_mask_np(
    *,
    unit_id: np.ndarray,
    spike_label: np.ndarray | None,
    sample_source: np.ndarray | None,
    gate_prob: np.ndarray | None,
    gate_threshold: float,
    allowed_sources: List[str] | Tuple[str, ...] | None,
    strict_overlap_label: np.ndarray | None = None,
    overlap_label: np.ndarray | None = None,
    qc_allowed: np.ndarray | None = None,
    strict_matching_ambiguous: np.ndarray | None = None,
    relaxed_matching_ambiguous: np.ndarray | None = None,
    snr_peak: np.ndarray | None = None,
    event_amp_norm: np.ndarray | None = None,
    event_spatial_score: np.ndarray | None = None,
    event_simul_local: np.ndarray | None = None,
    event_simul_global: np.ndarray | None = None,
    qc_cfg: Dict[str, Any] | None = None,
    apply_overlap_filter: bool = True,
    apply_qc_filter: bool = True,
) -> tuple[np.ndarray, Dict[str, int]]:
    unit_arr = np.asarray(unit_id, dtype=np.int64)
    valid = unit_arr >= 0
    if spike_label is not None:
        valid = valid & (np.asarray(spike_label, dtype=np.float32) > 0.5)
    if sample_source is not None:
        source_arr = np.asarray(sample_source, dtype=np.int8)
        source_map = {name: code for name, code in GATE_SOURCE_NAME_TO_CODE.items()}
        if allowed_sources is None:
            allowed_sources = ("primary_matched_positive",)
        allowed_codes = np.asarray([source_map.get(str(name), -999) for name in allowed_sources], dtype=np.int8)
        source_allowed = np.isin(source_arr, allowed_codes)
    else:
        source_allowed = np.ones(unit_arr.shape, dtype=np.bool_)
    valid = valid & source_allowed

    if gate_prob is not None:
        gate_pass = np.asarray(gate_prob, dtype=np.float32) >= float(gate_threshold)
    else:
        gate_pass = np.ones(unit_arr.shape, dtype=np.bool_)
    valid = valid & gate_pass

    strict_overlap = (
        np.asarray(strict_overlap_label, dtype=np.bool_)
        if strict_overlap_label is not None
        else np.zeros(unit_arr.shape, dtype=np.bool_)
    )
    overlap = (
        np.asarray(overlap_label, dtype=np.float32) > 0.5
        if overlap_label is not None
        else np.zeros(unit_arr.shape, dtype=np.bool_)
    )
    if bool(apply_overlap_filter):
        valid = valid & (~strict_overlap)
        valid = valid & (~overlap)

    qc = np.ones(unit_arr.shape, dtype=np.bool_)
    if qc_allowed is not None:
        qc = qc & np.asarray(qc_allowed, dtype=np.bool_)
    if strict_matching_ambiguous is not None:
        qc = qc & (~np.asarray(strict_matching_ambiguous, dtype=np.bool_))
    if relaxed_matching_ambiguous is not None:
        qc = qc & (~np.asarray(relaxed_matching_ambiguous, dtype=np.bool_))
    qc_cfg = dict(qc_cfg or {})
    if snr_peak is not None:
        qc = qc & (np.asarray(snr_peak, dtype=np.float32) >= float(qc_cfg.get("min_snr_peak", 0.0)))
    if event_amp_norm is not None:
        qc = qc & (np.asarray(event_amp_norm, dtype=np.float32) >= float(qc_cfg.get("min_amp_norm", 0.0)))
    if event_spatial_score is not None:
        qc = qc & (np.asarray(event_spatial_score, dtype=np.float32) >= float(qc_cfg.get("min_spatial_score", 0.0)))
    if event_simul_local is not None:
        qc = qc & (np.asarray(event_simul_local, dtype=np.int32) <= int(qc_cfg.get("max_simul_local", 1_000_000)))
    if event_simul_global is not None:
        qc = qc & (np.asarray(event_simul_global, dtype=np.int32) <= int(qc_cfg.get("max_simul_global", 1_000_000)))
    if bool(apply_qc_filter):
        valid = valid & qc
    stats = {
        "valid": int(np.sum(valid)),
        "source_allowed": int(np.sum(source_allowed)),
        "gate_pass": int(np.sum(gate_pass)),
        "strict_overlap": int(np.sum(strict_overlap)),
        "overlap": int(np.sum(overlap)),
        "qc_pass": int(np.sum(qc)),
        "overlap_filter_enabled": int(bool(apply_overlap_filter)),
        "qc_filter_enabled": int(bool(apply_qc_filter)),
    }
    return valid.astype(np.bool_, copy=False), stats


def build_boundary_candidate_mask_np(
    *,
    valid_mask: np.ndarray,
    overlap_label: np.ndarray | None,
    strict_overlap_label: np.ndarray | None,
    gate_prob: np.ndarray | None,
    gate_threshold: float,
    snr_peak: np.ndarray | None,
    event_amp_norm: np.ndarray | None,
    event_spatial_score: np.ndarray | None,
    event_simul_local: np.ndarray | None,
    event_simul_global: np.ndarray | None,
    strict_matching_ambiguous: np.ndarray | None,
    relaxed_matching_ambiguous: np.ndarray | None,
    qc_cfg: Dict[str, Any],
    qc_allowed: np.ndarray | None = None,
    apply_overlap_filter: bool = True,
    apply_qc_filter: bool = True,
) -> np.ndarray:
    cand = np.asarray(valid_mask, dtype=np.bool_).copy()
    if bool(apply_overlap_filter):
        if strict_overlap_label is not None:
            cand = cand & (~np.asarray(strict_overlap_label, dtype=np.bool_))
        if overlap_label is not None:
            cand = cand & (np.asarray(overlap_label, dtype=np.float32) <= 0.5)
    if gate_prob is not None:
        cand = cand & (np.asarray(gate_prob, dtype=np.float32) >= float(gate_threshold))
    if bool(apply_qc_filter):
        if qc_allowed is not None:
            cand = cand & np.asarray(qc_allowed, dtype=np.bool_)
        if snr_peak is not None:
            cand = cand & (np.asarray(snr_peak, dtype=np.float32) >= float(qc_cfg.get("min_snr_peak", 0.0)))
        if event_amp_norm is not None:
            cand = cand & (np.asarray(event_amp_norm, dtype=np.float32) >= float(qc_cfg.get("min_amp_norm", 0.0)))
        if event_spatial_score is not None:
            cand = cand & (np.asarray(event_spatial_score, dtype=np.float32) >= float(qc_cfg.get("min_spatial_score", 0.0)))
        if event_simul_local is not None:
            cand = cand & (np.asarray(event_simul_local, dtype=np.int32) <= int(qc_cfg.get("max_simul_local", 1_000_000)))
        if event_simul_global is not None:
            cand = cand & (np.asarray(event_simul_global, dtype=np.int32) <= int(qc_cfg.get("max_simul_global", 1_000_000)))
        if strict_matching_ambiguous is not None:
            cand = cand & (~np.asarray(strict_matching_ambiguous, dtype=np.bool_))
        if relaxed_matching_ambiguous is not None:
            cand = cand & (~np.asarray(relaxed_matching_ambiguous, dtype=np.bool_))
    return cand.astype(np.bool_, copy=False)

def _time_block_ids_for_indices(idx: np.ndarray, event_time: np.ndarray, n_blocks: int) -> np.ndarray:
    idx_arr = np.asarray(idx, dtype=np.int64)
    if idx_arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    t = np.asarray(event_time, dtype=np.int64)[idx_arr].astype(np.float64, copy=False)
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    if t_max <= t_min:
        return np.zeros((idx_arr.size,), dtype=np.int64)
    block = np.floor((t - t_min) / max(1.0, (t_max - t_min + 1.0) / float(max(2, int(n_blocks))))).astype(np.int64)
    return np.clip(block, 0, int(max(2, n_blocks)) - 1)


def build_temporal_episode(
    batch: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor] | None:
    global_unit_id = batch["unit_id"].detach().cpu().numpy().astype(np.int64, copy=False)
    recording_id = batch["recording_id"].detach().cpu().numpy().astype(np.int64, copy=False)
    event_time = batch["event_time_sample"].detach().cpu().numpy().astype(np.int64, copy=False)
    valid = valid_mask.detach().cpu().numpy().astype(np.bool_, copy=False)
    ep_cfg = dict(cfg.get("stage2", {}).get("episode", {}) or {})
    sampling_mode = str(ep_cfg.get("sampling_mode", "")).strip().lower()
    fixed_support_once = bool(ep_cfg.get("fixed_support_once", False)) or sampling_mode in {
        "fixed_support_once",
        "single_support_once",
    }
    max_spikes_per_task = int(ep_cfg.get("max_spikes_per_task", ep_cfg.get("max_samples_per_task", 256)))
    n_blocks = int(max(2, ep_cfg.get("n_time_blocks", 6)))
    anchor_blocks_min = int(max(1, ep_cfg.get("anchor_blocks_min", 1)))
    anchor_blocks_max = int(max(anchor_blocks_min, ep_cfg.get("anchor_blocks_max", 2)))
    causal_fraction = float(ep_cfg.get("causal_stress_fraction", 0.25))
    min_way = int(ep_cfg.get("min_way", 2))
    if not np.any(valid):
        return None
    anchor_parts: List[np.ndarray] = []
    loss_parts: List[np.ndarray] = []
    all_parts: List[np.ndarray] = []
    rng = np.random.default_rng(int(torch.sum(batch["event_time_sample"]).detach().cpu().item()) % (2**32 - 1))
    for rec in np.unique(recording_id[valid]).tolist():
        rec_idx = np.flatnonzero(valid & (recording_id == int(rec)) & (global_unit_id >= 0)).astype(np.int64, copy=False)
        if rec_idx.size < 4 or np.unique(global_unit_id[rec_idx]).size < min_way:
            continue
        rec_idx = rec_idx[np.argsort(event_time[rec_idx], kind="mergesort")]
        if fixed_support_once:
            fixed_mask_t = batch.get("fixed_support_mask")
            if fixed_mask_t is None:
                raise ValueError("stage2.episode.fixed_support_once requires batch['fixed_support_mask']")
            fixed_support_mask = fixed_mask_t.detach().cpu().numpy().astype(np.bool_, copy=False)
            anchor = rec_idx[fixed_support_mask[rec_idx]]
            loss_pool = rec_idx[~fixed_support_mask[rec_idx]]
            if anchor.size == 0 or loss_pool.size == 0:
                continue
            support_units = np.unique(global_unit_id[anchor])
            if support_units.size < min_way:
                continue
            keep_query = np.isin(global_unit_id[loss_pool], support_units)
            loss_pool = loss_pool[keep_query]
            if loss_pool.size == 0:
                continue
            anchor_parts.append(anchor.astype(np.int64, copy=False))
            loss_parts.append(loss_pool.astype(np.int64, copy=False))
            all_parts.append(np.concatenate([anchor, loss_pool], axis=0).astype(np.int64, copy=False))
            continue
        if int(max_spikes_per_task) > 0 and rec_idx.size > int(max_spikes_per_task):
            per_unit_parts: List[np.ndarray] = []
            units = np.unique(global_unit_id[rec_idx])
            per_unit_cap = max(1, int(max_spikes_per_task) // max(1, int(units.size)))
            for unit_value in units.tolist():
                unit_idx = rec_idx[global_unit_id[rec_idx] == int(unit_value)]
                take = min(int(unit_idx.size), int(per_unit_cap))
                if take > 0:
                    per_unit_parts.append(rng.choice(unit_idx, size=take, replace=False).astype(np.int64, copy=False))
            rec_idx = np.sort(np.concatenate(per_unit_parts, axis=0).astype(np.int64, copy=False)) if per_unit_parts else rec_idx[:0]
            if rec_idx.size < 4 or np.unique(global_unit_id[rec_idx]).size < min_way:
                continue
        t_rec = event_time[rec_idx].astype(np.float64, copy=False)
        t_min = float(np.min(t_rec))
        t_max = float(np.max(t_rec))
        if t_max <= t_min:
            split = max(1, rec_idx.size // 2)
            anchor = rec_idx[:split]
            loss_pool = rec_idx[split:]
        else:
            block_id = np.floor((t_rec - t_min) / max(1.0, (t_max - t_min + 1.0) / float(n_blocks))).astype(np.int64)
            block_id = np.clip(block_id, 0, n_blocks - 1)
            n_anchor_low = int(min(max(1, anchor_blocks_min), n_blocks - 1))
            n_anchor_high = int(min(max(n_anchor_low, anchor_blocks_max), n_blocks - 1))
            if rng.random() < causal_fraction:
                n_anchor = int(rng.integers(n_anchor_low, n_anchor_high + 1))
                anchor_blocks = set(range(min(n_anchor, n_blocks - 1)))
            else:
                n_anchor = int(rng.integers(n_anchor_low, n_anchor_high + 1))
                anchor_blocks = set(rng.choice(np.arange(n_blocks), size=min(n_anchor, n_blocks - 1), replace=False).tolist())
            anchor_mask = np.asarray([int(b) in anchor_blocks for b in block_id.tolist()], dtype=np.bool_)
            loss_mask = ~anchor_mask
            anchor = rec_idx[anchor_mask]
            loss_pool = rec_idx[loss_mask]
            if anchor.size == 0 or loss_pool.size == 0:
                split = max(1, rec_idx.size // 2)
                anchor = rec_idx[:split]
                loss_pool = rec_idx[split:]
        if anchor.size == 0 or loss_pool.size == 0:
            continue
        if np.unique(global_unit_id[anchor]).size < 1 or np.unique(global_unit_id[loss_pool]).size < 1:
            continue
        anchor_parts.append(anchor.astype(np.int64, copy=False))
        loss_parts.append(loss_pool.astype(np.int64, copy=False))
        all_parts.append(rec_idx.astype(np.int64, copy=False))
    if not all_parts:
        return None
    anchor_idx = np.concatenate(anchor_parts, axis=0).astype(np.int64, copy=False)
    loss_idx = np.concatenate(loss_parts, axis=0).astype(np.int64, copy=False)
    all_idx = np.concatenate(all_parts, axis=0).astype(np.int64, copy=False)
    device = batch["unit_id"].device
    episode_way = int(np.unique(global_unit_id[all_idx]).size)
    return {
        "anchor_idx": torch.from_numpy(anchor_idx).to(device=device),
        "loss_idx": torch.from_numpy(loss_idx).to(device=device),
        "all_idx": torch.from_numpy(all_idx).to(device=device),
        "support_idx": torch.from_numpy(anchor_idx).to(device=device),
        "query_idx": torch.from_numpy(loss_idx).to(device=device),
        "recording_id": torch.from_numpy(np.unique(recording_id[all_idx]).astype(np.int64)).to(device=device),
        "episode_way": torch.tensor(episode_way, device=device),
        "episode_effective_n_way": torch.tensor(episode_way, device=device),
        "episode_support_count": torch.tensor(int(anchor_idx.size), device=device),
        "episode_query_count": torch.tensor(int(loss_idx.size), device=device),
    }


def build_stage2_temporal_episode_sampler(
    unit_id: np.ndarray,
    recording_id: np.ndarray,
    event_time_sample: np.ndarray,
    unit_channel_source: np.ndarray | None,
    valid_indices: np.ndarray,
    *,
    seed: int,
    min_way: int,
    batch_size_tasks: int = 1,
    max_spikes_per_task: int = 256,
    fixed_support_once: bool = False,
    fixed_support_per_unit: int = 1,
) -> Callable[[int | None], np.ndarray]:
    unit_arr = np.asarray(unit_id, dtype=np.int64)
    rec_arr = np.asarray(recording_id, dtype=np.int64)
    time_arr = np.asarray(event_time_sample, dtype=np.int64)
    _ = unit_channel_source
    valid_idx = np.asarray(valid_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rec_candidates: Dict[int, np.ndarray] = {}
    rec_fixed_support: Dict[int, np.ndarray] = {}
    fixed_support_parts: List[np.ndarray] = []
    support_rng = np.random.default_rng(int(seed) + 1009)
    for rec_value in np.unique(rec_arr[valid_idx]).tolist():
        idx = valid_idx[(rec_arr[valid_idx] == int(rec_value)) & (unit_arr[valid_idx] >= 0)]
        if idx.size < 4:
            continue
        units = np.unique(unit_arr[idx])
        if units.size < int(max(2, min_way)):
            continue
        idx = idx[np.argsort(time_arr[idx], kind="mergesort")].astype(np.int64, copy=False)
        if fixed_support_once:
            support_parts: List[np.ndarray] = []
            for unit_value in units.tolist():
                unit_idx = idx[unit_arr[idx] == int(unit_value)]
                if unit_idx.size < 2:
                    continue
                take = min(int(unit_idx.size) - 1, max(1, int(fixed_support_per_unit)))
                support_parts.append(
                    np.sort(support_rng.choice(unit_idx, size=take, replace=False).astype(np.int64, copy=False))
                )
            support = (
                np.sort(np.concatenate(support_parts, axis=0).astype(np.int64, copy=False))
                if support_parts
                else np.zeros((0,), dtype=np.int64)
            )
            support_units = np.unique(unit_arr[support]) if support.size else np.zeros((0,), dtype=np.int64)
            query = np.setdiff1d(idx, support, assume_unique=False).astype(np.int64, copy=False)
            query = query[np.isin(unit_arr[query], support_units)]
            if support.size == 0 or query.size == 0 or support_units.size < int(max(2, min_way)):
                continue
            rec_fixed_support[int(rec_value)] = support
            fixed_support_parts.append(support)
        rec_candidates[int(rec_value)] = idx
    if not rec_candidates:
        raise RuntimeError("recording task sampler found no recording with enough valid units")
    fixed_support_indices = (
        np.sort(np.concatenate(fixed_support_parts, axis=0).astype(np.int64, copy=False))
        if fixed_support_parts
        else np.zeros((0,), dtype=np.int64)
    )
    rec_cycle = np.asarray(sorted(rec_candidates.keys()), dtype=np.int64)
    rng.shuffle(rec_cycle)
    cursor = 0

    def sample_recording_task_batch(epoch: int | None = None) -> np.ndarray:
        nonlocal cursor, rec_cycle
        _ = epoch
        task_parts: List[np.ndarray] = []
        for _task in range(max(1, int(batch_size_tasks))):
            if cursor >= rec_cycle.size:
                rng.shuffle(rec_cycle)
                cursor = 0
            rec_value = int(rec_cycle[cursor])
            cursor += 1
            idx = rec_candidates[rec_value]
            cap = int(max(1, max_spikes_per_task))
            if fixed_support_once:
                support = rec_fixed_support[int(rec_value)]
                query_pool = np.setdiff1d(idx, support, assume_unique=False).astype(np.int64, copy=False)
                query_budget = max(0, cap - int(support.size))
                if query_budget <= 0 or query_pool.size == 0:
                    task_parts.append(support.copy())
                    continue
                if query_pool.size > query_budget:
                    units = np.unique(unit_arr[query_pool])
                    per_unit = max(1, query_budget // max(1, int(units.size)))
                    query_parts: List[np.ndarray] = []
                    for unit_value in units.tolist():
                        unit_idx = query_pool[unit_arr[query_pool] == int(unit_value)]
                        take = min(int(unit_idx.size), int(per_unit))
                        if take > 0:
                            query_parts.append(rng.choice(unit_idx, size=take, replace=False).astype(np.int64, copy=False))
                    query = (
                        np.concatenate(query_parts, axis=0).astype(np.int64, copy=False)
                        if query_parts
                        else query_pool[:0]
                    )
                    if query.size < query_budget:
                        missing = np.setdiff1d(query_pool, query, assume_unique=False)
                        if missing.size > 0:
                            fill = rng.choice(
                                missing,
                                size=min(query_budget - query.size, missing.size),
                                replace=False,
                            ).astype(np.int64, copy=False)
                            query = np.concatenate([query, fill], axis=0).astype(np.int64, copy=False)
                else:
                    query = query_pool
                chosen = np.concatenate([support, query[:query_budget]], axis=0).astype(np.int64, copy=False)
                task_parts.append(np.sort(chosen))
                continue
            if idx.size <= cap:
                task_parts.append(idx.copy())
                continue
            units = np.unique(unit_arr[idx])
            per_unit = max(1, cap // max(1, int(units.size)))
            parts: List[np.ndarray] = []
            for unit_value in units.tolist():
                unit_idx = idx[unit_arr[idx] == int(unit_value)]
                take = min(int(unit_idx.size), int(per_unit))
                if take > 0:
                    parts.append(rng.choice(unit_idx, size=take, replace=False).astype(np.int64, copy=False))
            chosen = np.concatenate(parts, axis=0).astype(np.int64, copy=False) if parts else idx[:cap]
            if chosen.size < cap:
                missing = np.setdiff1d(idx, chosen, assume_unique=False)
                if missing.size > 0:
                    fill = rng.choice(missing, size=min(cap - chosen.size, missing.size), replace=False).astype(np.int64, copy=False)
                    chosen = np.concatenate([chosen, fill], axis=0).astype(np.int64, copy=False)
            task_parts.append(np.sort(chosen[:cap].astype(np.int64, copy=False)))
        return np.concatenate(task_parts, axis=0).astype(np.int64, copy=False)

    setattr(sample_recording_task_batch, "fixed_support_indices", fixed_support_indices)
    return sample_recording_task_batch


def temporal_connection_loss(
    z: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    episode: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses_cfg = dict(cfg.get("stage2", {}).get("losses", {}) or {})
    conn_cfg = dict(losses_cfg.get("temporal_connection", {}) or {})
    mode = str(conn_cfg.get("mode", "adjacent_hinge"))
    if mode == "global_compact_margin":
        from src.nn.global_identity_loss import global_identity_connection_loss
        return global_identity_connection_loss(z, batch, episode, cfg)
    if mode != "adjacent_hinge":
        raise ValueError(f"Unknown auxiliary connection mode: {mode}")
    enabled = bool(conn_cfg.get("enabled", False))
    weight = float(conn_cfg.get("weight", conn_cfg.get("lambda", 0.0)))
    stats = {
        "temporal_conn_loss": 0.0,
        "lambda_temporal_conn_eff": weight if enabled else 0.0,
        "temporal_conn_pair_count": 0.0,
        "temporal_conn_active_ratio": 0.0,
    }
    if not enabled or weight <= 0.0:
        return z.new_zeros(()), stats
    all_idx = episode.get("all_idx")
    if all_idx is None or int(all_idx.numel()) == 0:
        return z.new_zeros(()), stats
    unit_id = batch["unit_id"].to(device=z.device, dtype=torch.long)
    recording_id = batch["recording_id"].to(device=z.device, dtype=torch.long)
    event_time = batch["event_time_sample"].to(device=z.device, dtype=torch.long)
    n_blocks = int(max(2, cfg.get("stage2", {}).get("episode", {}).get("n_time_blocks", 8)))
    tau_conn = float(conn_cfg.get("tau_conn", conn_cfg.get("margin", 0.20)))
    max_pairs_per_recording = int(conn_cfg.get("max_pairs_per_recording", 128))
    bsz = int(unit_id.shape[0])
    two_view = bool(z.shape[0] == 2 * bsz)
    views = [z[:bsz], z[bsz:]] if two_view else [z]

    pair_i: List[int] = []
    pair_j: List[int] = []
    all_idx_cpu = all_idx.detach().cpu().numpy().astype(np.int64, copy=False)
    unit_cpu = unit_id.detach().cpu().numpy().astype(np.int64, copy=False)
    rec_cpu = recording_id.detach().cpu().numpy().astype(np.int64, copy=False)
    time_cpu = event_time.detach().cpu().numpy().astype(np.int64, copy=False)
    for rec_value in np.unique(rec_cpu[all_idx_cpu]).tolist():
        rec_idx = all_idx_cpu[rec_cpu[all_idx_cpu] == int(rec_value)]
        if rec_idx.size < 2:
            continue
        rec_blocks = _time_block_ids_for_indices(rec_idx, time_cpu, n_blocks)
        block_by_idx = {int(idx_value): int(block_value) for idx_value, block_value in zip(rec_idx.tolist(), rec_blocks.tolist())}
        rec_pairs: List[tuple[int, int]] = []
        for unit_value in np.unique(unit_cpu[rec_idx]).tolist():
            unit_idx = rec_idx[unit_cpu[rec_idx] == int(unit_value)]
            if unit_idx.size < 2:
                continue
            for block_value in range(n_blocks - 1):
                left = np.asarray([int(v) for v in unit_idx.tolist() if block_by_idx.get(int(v)) == block_value], dtype=np.int64)
                right = np.asarray([int(v) for v in unit_idx.tolist() if block_by_idx.get(int(v)) == block_value + 1], dtype=np.int64)
                if left.size == 0 or right.size == 0:
                    continue
                left = left[np.argsort(time_cpu[left], kind="mergesort")]
                right = right[np.argsort(time_cpu[right], kind="mergesort")]
                rec_pairs.append((int(left[left.size // 2]), int(right[right.size // 2])))
        if max_pairs_per_recording > 0 and len(rec_pairs) > max_pairs_per_recording:
            step = float(len(rec_pairs)) / float(max_pairs_per_recording)
            take = sorted({int(min(len(rec_pairs) - 1, round(i * step))) for i in range(max_pairs_per_recording)})
            rec_pairs = [rec_pairs[i] for i in take[:max_pairs_per_recording]]
        for i_value, j_value in rec_pairs:
            pair_i.append(i_value)
            pair_j.append(j_value)

    if not pair_i:
        return z.new_zeros(()), stats
    i_t = torch.tensor(pair_i, device=z.device, dtype=torch.long)
    j_t = torch.tensor(pair_j, device=z.device, dtype=torch.long)
    view_losses = []
    active_vals = []
    for z_view in views:
        z_norm = F.normalize(z_view, p=2, dim=1)
        d = 1.0 - torch.sum(z_norm[i_t] * z_norm[j_t], dim=1)
        violation = F.relu(d - float(tau_conn)).pow(2)
        view_losses.append(torch.mean(violation))
        active_vals.append(torch.mean((violation > 0.0).to(dtype=torch.float32)).detach())
    loss = torch.mean(torch.stack(view_losses)) if view_losses else z.new_zeros(())
    stats["temporal_conn_loss"] = float(loss.detach().cpu().item())
    stats["temporal_conn_pair_count"] = float(len(pair_i))
    stats["temporal_conn_active_ratio"] = float(torch.mean(torch.stack(active_vals)).cpu().item()) if active_vals else 0.0
    return loss, stats


def sample_cross_domain_pair_cosine_stats(
    z: np.ndarray,
    y: np.ndarray,
    recording_id: np.ndarray,
    *,
    n_pairs: int,
    seed: int,
) -> Dict[str, float]:
    out = {
        "cross_pos_cos_mean": float("nan"),
        "cross_neg_cos_mean": float("nan"),
        "cross_domain_gap": float("nan"),
    }
    n = int(z.shape[0])
    if n < 2:
        return out
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y, dtype=np.int64)
    rec_arr = np.asarray(recording_id, dtype=np.int16)
    pair_target = max(1000, int(n_pairs))
    classes = np.unique(y_arr)
    if classes.size == 0:
        return out
    per_unit_target = max(1, int(math.ceil(pair_target / max(1, classes.size))))
    pos_unit_vals: List[float] = []
    neg_unit_vals: List[float] = []
    for unit_value in classes.tolist():
        idx = np.flatnonzero(y_arr == int(unit_value)).astype(np.int64, copy=False)
        other = np.flatnonzero(y_arr != int(unit_value)).astype(np.int64, copy=False)
        pos_vals: List[float] = []
        neg_vals: List[float] = []
        if idx.size >= 2 and np.unique(rec_arr[idx]).size >= 2:
            tries = 0
            while len(pos_vals) < per_unit_target and tries < per_unit_target * 20:
                pair = rng.choice(idx, size=2, replace=False)
                if int(rec_arr[int(pair[0])]) != int(rec_arr[int(pair[1])]):
                    pos_vals.append(float(np.dot(z[int(pair[0])], z[int(pair[1])])))
                tries += 1
        if idx.size > 0 and other.size > 0:
            for _ in range(per_unit_target):
                i = int(rng.choice(idx))
                j = int(rng.choice(other))
                neg_vals.append(float(np.dot(z[i], z[j])))
        if pos_vals:
            pos_unit_vals.append(float(np.mean(np.asarray(pos_vals, dtype=np.float32))))
        if neg_vals:
            neg_unit_vals.append(float(np.mean(np.asarray(neg_vals, dtype=np.float32))))
    if pos_unit_vals:
        out["cross_pos_cos_mean"] = float(np.mean(np.asarray(pos_unit_vals, dtype=np.float32)))
    if neg_unit_vals:
        out["cross_neg_cos_mean"] = float(np.mean(np.asarray(neg_unit_vals, dtype=np.float32)))
    if pos_unit_vals and neg_unit_vals:
        out["cross_domain_gap"] = float(out["cross_pos_cos_mean"] - out["cross_neg_cos_mean"])
    return out


def _balanced_holdout_eval_indices(
    y: np.ndarray,
    recording_id: np.ndarray,
    *,
    max_eval_samples: int,
    seed: int,
) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.int64)
    rec_arr = np.asarray(recording_id, dtype=np.int64)
    all_idx = np.arange(y_arr.shape[0], dtype=np.int64)
    if int(max_eval_samples) <= 0 or all_idx.size <= int(max_eval_samples):
        return all_idx
    groups: List[np.ndarray] = []
    for rec_value in np.unique(rec_arr).tolist():
        rec_mask = rec_arr == int(rec_value)
        for unit_value in np.unique(y_arr[rec_mask]).tolist():
            idx = np.flatnonzero(rec_mask & (y_arr == int(unit_value))).astype(np.int64, copy=False)
            if idx.size > 0:
                groups.append(idx)
    if not groups:
        return all_idx[:0]
    rng = np.random.default_rng(seed)
    per_group = max(2, int(math.floor(int(max_eval_samples) / max(1, len(groups)))))
    selected_parts: List[np.ndarray] = []
    selected_set: set[int] = set()
    for idx in groups:
        take = min(int(idx.size), per_group)
        chosen = idx if idx.size <= take else rng.choice(idx, size=take, replace=False).astype(np.int64, copy=False)
        selected_parts.append(chosen)
        selected_set.update(int(v) for v in chosen.tolist())
    selected = np.concatenate(selected_parts, axis=0) if selected_parts else np.zeros((0,), dtype=np.int64)
    remaining_budget = int(max_eval_samples) - int(selected.size)
    if remaining_budget > 0:
        remaining = np.asarray([int(v) for v in all_idx.tolist() if int(v) not in selected_set], dtype=np.int64)
        if remaining.size > 0:
            fill = remaining if remaining.size <= remaining_budget else rng.choice(
                remaining,
                size=remaining_budget,
                replace=False,
            ).astype(np.int64, copy=False)
            selected = np.concatenate([selected, fill], axis=0)
    if selected.size > int(max_eval_samples):
        selected = rng.choice(selected, size=int(max_eval_samples), replace=False).astype(np.int64, copy=False)
    return np.sort(selected.astype(np.int64, copy=False))


def sample_holdout_recording_embedding_stats(
    z_val: np.ndarray,
    y_val: np.ndarray,
    recording_id: np.ndarray,
    *,
    support_per_unit: int,
    seed: int,
) -> Dict[str, float]:
    out = {
        "holdout_proto_acc": float("nan"),
        "holdout_pos_cos": float("nan"),
        "holdout_neg_cos": float("nan"),
        "holdout_gap": float("nan"),
        "holdout_recordings": 0.0,
        "holdout_units": 0.0,
        "holdout_samples": 0.0,
    }
    if z_val.shape[0] == 0:
        return out
    z_arr = np.asarray(z_val, dtype=np.float32)
    z_norm = np.linalg.norm(z_arr, axis=1, keepdims=True)
    z_arr = z_arr / np.maximum(z_norm, 1.0e-8)
    y_arr = np.asarray(y_val, dtype=np.int64)
    rec_arr = np.asarray(recording_id, dtype=np.int16)
    support_target = max(1, int(support_per_unit))
    rng = np.random.default_rng(seed)
    rec_acc_vals: List[float] = []
    rec_pos_vals: List[float] = []
    rec_neg_vals: List[float] = []
    rec_unit_counts: List[int] = []
    rec_sample_counts: List[int] = []
    for rec_value in np.unique(rec_arr).tolist():
        rec_mask = rec_arr == int(rec_value)
        if not np.any(rec_mask):
            continue
        y_rec = y_arr[rec_mask]
        z_rec = z_arr[rec_mask]
        unit_splits: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
        proto_labels: List[int] = []
        proto_parts: List[np.ndarray] = []
        rec_sample_total = 0
        for unit_value in np.unique(y_rec).tolist():
            local_idx = np.flatnonzero(y_rec == int(unit_value)).astype(np.int64, copy=False)
            if local_idx.size < 2:
                continue
            shuffled = local_idx.copy()
            rng.shuffle(shuffled)
            n_support = min(support_target, int(shuffled.size - 1))
            n_support = max(1, int(n_support))
            support_local = np.sort(shuffled[:n_support].astype(np.int64, copy=False))
            query_local = np.sort(shuffled[n_support:].astype(np.int64, copy=False))
            if query_local.size == 0:
                continue
            proto = np.mean(z_rec[support_local], axis=0).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(proto))
            if norm > 1.0e-8:
                proto = proto / norm
            unit_splits[int(unit_value)] = (support_local, query_local)
            proto_labels.append(int(unit_value))
            proto_parts.append(proto)
            rec_sample_total += int(support_local.size + query_local.size)
        if len(proto_parts) < 2:
            continue
        proto_label_arr = np.asarray(proto_labels, dtype=np.int64)
        proto_mat = np.stack(proto_parts, axis=0).astype(np.float32, copy=False)
        proto_index = {int(label): idx for idx, label in enumerate(proto_label_arr.tolist())}
        acc_unit_vals: List[float] = []
        pos_unit_vals: List[float] = []
        neg_unit_vals: List[float] = []
        for unit_value, (support_local, query_local) in unit_splits.items():
            proto_idx = proto_index.get(int(unit_value))
            if proto_idx is None:
                continue
            z_query = z_rec[query_local]
            logits = z_query @ proto_mat.T
            pred = proto_label_arr[np.argmax(logits, axis=1)]
            acc_unit_vals.append(float(np.mean((pred == int(unit_value)).astype(np.float32))))
            z_support = z_rec[support_local]
            pos_unit_vals.append(float(np.mean(z_query @ z_support.T)))
            neg_proto = proto_mat[proto_label_arr != int(unit_value)]
            if neg_proto.shape[0] > 0:
                neg_unit_vals.append(float(np.mean(z_query @ neg_proto.T)))
        if acc_unit_vals:
            rec_acc_vals.append(float(np.mean(np.asarray(acc_unit_vals, dtype=np.float32))))
        if pos_unit_vals:
            rec_pos_vals.append(float(np.mean(np.asarray(pos_unit_vals, dtype=np.float32))))
        if neg_unit_vals:
            rec_neg_vals.append(float(np.mean(np.asarray(neg_unit_vals, dtype=np.float32))))
        rec_unit_counts.append(int(len(unit_splits)))
        rec_sample_counts.append(int(rec_sample_total))
    if rec_acc_vals:
        out["holdout_proto_acc"] = float(np.mean(np.asarray(rec_acc_vals, dtype=np.float32)))
        out["holdout_recordings"] = float(len(rec_acc_vals))
    if rec_pos_vals:
        out["holdout_pos_cos"] = float(np.mean(np.asarray(rec_pos_vals, dtype=np.float32)))
    if rec_neg_vals:
        out["holdout_neg_cos"] = float(np.mean(np.asarray(rec_neg_vals, dtype=np.float32)))
    if rec_pos_vals and rec_neg_vals:
        out["holdout_gap"] = float(out["holdout_pos_cos"] - out["holdout_neg_cos"])
    if rec_unit_counts:
        out["holdout_units"] = float(np.sum(np.asarray(rec_unit_counts, dtype=np.int64)))
    if rec_sample_counts:
        out["holdout_samples"] = float(np.sum(np.asarray(rec_sample_counts, dtype=np.int64)))
    return out


def _select_time_spaced_indices(idx: np.ndarray, times: np.ndarray, *, max_count: int) -> np.ndarray:
    idx_arr = np.asarray(idx, dtype=np.int64)
    if idx_arr.size == 0:
        return idx_arr
    order = np.argsort(np.asarray(times, dtype=np.int64)[idx_arr], kind="mergesort")
    ordered = idx_arr[order]
    limit = int(max_count)
    if limit <= 0 or ordered.size <= limit:
        return ordered.astype(np.int64, copy=False)
    positions = np.linspace(0, ordered.size - 1, num=limit)
    take = np.unique(np.round(positions).astype(np.int64))
    if take.size < limit:
        missing = np.asarray([i for i in range(ordered.size) if i not in set(take.tolist())], dtype=np.int64)
        if missing.size > 0:
            take = np.concatenate([take, missing[: limit - take.size]], axis=0)
    take = np.sort(take[:limit].astype(np.int64, copy=False))
    return ordered[take].astype(np.int64, copy=False)


def _select_recording_eval_unit_candidates(
    unit_candidates: List[tuple[int, np.ndarray]],
    time_arr: np.ndarray,
    *,
    min_unit_samples: int,
    max_unit_samples: int,
    max_recording_samples: int,
) -> List[tuple[int, np.ndarray]]:
    selected: List[tuple[int, np.ndarray]] = []
    for unit_value, unit_idx in unit_candidates:
        unit_arr = np.asarray(unit_idx, dtype=np.int64)
        if int(max_unit_samples) > 0:
            unit_arr = _select_time_spaced_indices(unit_arr, time_arr, max_count=int(max_unit_samples))
        else:
            unit_arr = unit_arr[np.argsort(np.asarray(time_arr, dtype=np.int64)[unit_arr], kind="mergesort")]
        if unit_arr.size >= int(min_unit_samples):
            selected.append((int(unit_value), unit_arr.astype(np.int64, copy=False)))
    if len(selected) < 2:
        return []
    max_rec = int(max_recording_samples)
    total = int(sum(int(idx.size) for _, idx in selected))
    if max_rec <= 0 or total <= max_rec:
        return selected
    if max_rec < int(min_unit_samples) * 2:
        return []
    if max_rec < int(min_unit_samples) * len(selected):
        selected = sorted(selected, key=lambda item: (-int(item[1].size), int(item[0])))[: max(2, max_rec // int(min_unit_samples))]
        total = int(sum(int(idx.size) for _, idx in selected))
    quotas: Dict[int, int] = {}
    for unit_value, idx in selected:
        raw = int(math.floor(float(max_rec) * float(idx.size) / float(max(1, total))))
        quotas[int(unit_value)] = min(int(idx.size), max(int(min_unit_samples), raw))
    while sum(quotas.values()) > max_rec:
        reducible = [
            (unit_value, quotas[unit_value] - int(min_unit_samples))
            for unit_value in quotas
            if quotas[unit_value] > int(min_unit_samples)
        ]
        if not reducible:
            break
        unit_value = max(reducible, key=lambda item: (item[1], -item[0]))[0]
        quotas[unit_value] -= 1
    while sum(quotas.values()) < max_rec:
        expandable = [
            (unit_value, int(idx.size) - quotas[int(unit_value)])
            for unit_value, idx in selected
            if quotas[int(unit_value)] < int(idx.size)
        ]
        if not expandable:
            break
        unit_value = max(expandable, key=lambda item: (item[1], -item[0]))[0]
        quotas[unit_value] += 1
    capped: List[tuple[int, np.ndarray]] = []
    for unit_value, idx in selected:
        take = int(quotas.get(int(unit_value), 0))
        if take >= int(min_unit_samples):
            capped_idx = _select_time_spaced_indices(idx, time_arr, max_count=take)
            if capped_idx.size >= int(min_unit_samples):
                capped.append((int(unit_value), capped_idx.astype(np.int64, copy=False)))
    return capped if len(capped) >= 2 else []


def _stable_recording_cluster_eval_indices(
    y_val: np.ndarray,
    recording_id: np.ndarray,
    event_time_sample: np.ndarray | None = None,
    *,
    min_samples_per_unit: int = 8,
    max_samples_per_unit: int = 128,
    max_samples_per_recording: int = 4096,
) -> np.ndarray:
    y_arr = np.asarray(y_val, dtype=np.int64)
    rec_arr = np.asarray(recording_id, dtype=np.int16)
    if y_arr.shape[0] != rec_arr.shape[0]:
        raise ValueError("y_val and recording_id must have matching first dimension")
    if y_arr.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if event_time_sample is None:
        time_arr = np.arange(y_arr.shape[0], dtype=np.int64)
    else:
        time_arr = np.asarray(event_time_sample, dtype=np.int64)
        if time_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("event_time_sample must match y_val first dimension")

    min_unit_samples = max(1, int(min_samples_per_unit))
    max_unit_samples = int(max_samples_per_unit)
    max_recording_samples = int(max_samples_per_recording)
    selected_parts: List[np.ndarray] = []
    for rec_value in np.unique(rec_arr).tolist():
        rec_idx_all = np.flatnonzero(rec_arr == int(rec_value)).astype(np.int64, copy=False)
        unit_candidates: List[tuple[int, np.ndarray]] = []
        for unit_value in np.unique(y_arr[rec_idx_all]).tolist():
            unit_idx = rec_idx_all[y_arr[rec_idx_all] == int(unit_value)]
            if unit_idx.size >= min_unit_samples:
                unit_candidates.append((int(unit_value), unit_idx.astype(np.int64, copy=False)))
        if len(unit_candidates) < 2:
            continue
        selected_units = _select_recording_eval_unit_candidates(
            unit_candidates,
            time_arr,
            min_unit_samples=min_unit_samples,
            max_unit_samples=max_unit_samples,
            max_recording_samples=max_recording_samples,
        )
        for _, selected in selected_units:
            selected_parts.append(selected)
    if not selected_parts:
        return np.zeros((0,), dtype=np.int64)
    return np.sort(np.concatenate(selected_parts, axis=0).astype(np.int64, copy=False))


def compute_knn_purity(z: np.ndarray, labels: np.ndarray, k_list: List[int] | tuple[int, ...] = (5, 10, 20), macro_by_label: bool = True) -> Dict[str, float]:
    z_arr = _normalized_embedding(z)
    y = np.asarray(labels, dtype=np.int64)
    out: Dict[str, float] = {}
    if z_arr.shape[0] <= 1:
        return {f"knn_purity{k}": float("nan") for k in k_list}
    sim = z_arr @ z_arr.T
    np.fill_diagonal(sim, -np.inf)
    for k in k_list:
        k_eff = min(int(k), int(z_arr.shape[0] - 1))
        if k_eff <= 0:
            out[f"knn_purity{k}"] = float("nan")
            continue
        nbr = np.argsort(-sim, axis=1, kind="mergesort")[:, :k_eff]
        purity = np.mean((y[nbr] == y[:, None]).astype(np.float32), axis=1)
        if macro_by_label:
            vals = [float(np.mean(purity[y == int(c)])) for c in np.unique(y).tolist() if np.any(y == int(c))]
            out[f"knn_purity{k}"] = float(np.mean(np.asarray(vals, dtype=np.float32))) if vals else float("nan")
        else:
            out[f"knn_purity{k}"] = float(np.mean(purity.astype(np.float32)))
    return out


def compute_label_conductance(z: np.ndarray, labels: np.ndarray, k: int = 15, mutual: bool = True, tau: float = 0.1) -> Dict[str, float]:
    z_arr = _normalized_embedding(z)
    y = np.asarray(labels, dtype=np.int64)
    n = int(z_arr.shape[0])
    if n <= 1 or np.unique(y).size < 2:
        return {f"label_conductance{k}": float("nan"), f"graph_separation{k}": float("nan")}
    k_eff = min(int(k), n - 1)
    sim = z_arr @ z_arr.T
    np.fill_diagonal(sim, -np.inf)
    nbr = np.argsort(-sim, axis=1, kind="mergesort")[:, :k_eff]
    nbr_sets = [set(int(v) for v in row.tolist()) for row in nbr]
    weights: Dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in nbr[i].tolist():
            jj = int(j)
            if bool(mutual) and i not in nbr_sets[jj]:
                continue
            a, b = (i, jj) if i < jj else (jj, i)
            if a == b:
                continue
            weights[(a, b)] = float(np.exp(float(np.clip(sim[a, b], -1.0, 1.0)) / max(float(tau), 1.0e-8)))
    if not weights:
        return {f"label_conductance{k}": float("nan"), f"graph_separation{k}": float("nan")}
    vals: List[float] = []
    for c in np.unique(y).tolist():
        inside = y == int(c)
        if not np.any(inside) or np.all(inside):
            continue
        cut = 0.0
        vol_s = 0.0
        vol_t = 0.0
        for (i, j), w in weights.items():
            if inside[i] or inside[j]:
                vol_s += w if inside[i] else 0.0
                vol_s += w if inside[j] else 0.0
            if (not inside[i]) or (not inside[j]):
                vol_t += w if not inside[i] else 0.0
                vol_t += w if not inside[j] else 0.0
            if bool(inside[i]) != bool(inside[j]):
                cut += w
        denom = max(min(vol_s, vol_t), 1.0e-8)
        vals.append(float(cut / denom))
    conductance = float(np.mean(np.asarray(vals, dtype=np.float32))) if vals else float("nan")
    return {f"label_conductance{k}": conductance, f"graph_separation{k}": float(1.0 - conductance) if np.isfinite(conductance) else float("nan")}


def sample_stable_recording_cluster_stats(
    z_val: np.ndarray,
    y_val: np.ndarray,
    recording_id: np.ndarray,
    event_time_sample: np.ndarray | None = None,
    *,
    knn_k_target: int = 15,
    min_knn_k: int = 3,
    min_samples_per_unit: int = 8,
    max_samples_per_unit: int = 128,
    max_samples_per_recording: int = 4096,
) -> Dict[str, float]:
    out = {
        "cluster_knn_purity": float("nan"),
        "cluster_knn_acc": float("nan"),
        "cluster_centroid_acc": float("nan"),
        "cluster_centroid_margin": float("nan"),
        "cluster_recordings": 0.0,
        "cluster_units": 0.0,
        "cluster_samples": 0.0,
        "cluster_mean_k": float("nan"),
    }
    if z_val.shape[0] == 0:
        return out
    z_arr = np.asarray(z_val, dtype=np.float32)
    z_norm = np.linalg.norm(z_arr, axis=1, keepdims=True)
    z_arr = z_arr / np.maximum(z_norm, 1.0e-8)
    y_arr = np.asarray(y_val, dtype=np.int64)
    rec_arr = np.asarray(recording_id, dtype=np.int16)
    if y_arr.shape[0] != z_arr.shape[0] or rec_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("z_val, y_val, and recording_id must have matching first dimension")
    if event_time_sample is None:
        time_arr = np.arange(z_arr.shape[0], dtype=np.int64)
    else:
        time_arr = np.asarray(event_time_sample, dtype=np.int64)
        if time_arr.shape[0] != z_arr.shape[0]:
            raise ValueError("event_time_sample must match z_val first dimension")

    rec_purity_vals: List[float] = []
    rec_knn_acc_vals: List[float] = []
    rec_centroid_acc_vals: List[float] = []
    rec_margin_vals: List[float] = []
    rec_unit_counts: List[int] = []
    rec_sample_counts: List[int] = []
    rec_k_vals: List[int] = []
    min_unit_samples = max(1, int(min_samples_per_unit))
    max_unit_samples = int(max_samples_per_unit)
    max_recording_samples = int(max_samples_per_recording)
    k_target = max(1, int(knn_k_target))
    k_min = max(1, int(min_knn_k))

    for rec_value in np.unique(rec_arr).tolist():
        rec_idx_all = np.flatnonzero(rec_arr == int(rec_value)).astype(np.int64, copy=False)
        if rec_idx_all.size == 0:
            continue
        unit_candidates: List[tuple[int, np.ndarray]] = []
        for unit_value in np.unique(y_arr[rec_idx_all]).tolist():
            unit_idx = rec_idx_all[y_arr[rec_idx_all] == int(unit_value)]
            if unit_idx.size >= min_unit_samples:
                unit_candidates.append((int(unit_value), unit_idx.astype(np.int64, copy=False)))
        if len(unit_candidates) < 2:
            continue
        selected_units_with_idx = _select_recording_eval_unit_candidates(
            unit_candidates,
            time_arr,
            min_unit_samples=min_unit_samples,
            max_unit_samples=max_unit_samples,
            max_recording_samples=max_recording_samples,
        )
        if len(selected_units_with_idx) < 2:
            continue
        selected_units = [int(unit_value) for unit_value, _ in selected_units_with_idx]
        selected_parts = [idx for _, idx in selected_units_with_idx]
        rec_selected = np.concatenate(selected_parts, axis=0).astype(np.int64, copy=False)
        y_rec = y_arr[rec_selected]
        z_rec = z_arr[rec_selected]
        unit_counts = [int(np.sum(y_rec == int(unit_value))) for unit_value in selected_units]
        if not unit_counts:
            continue
        k_eff = min(k_target, min(unit_counts) - 1, int(y_rec.shape[0] - 1))
        if k_eff < k_min:
            continue

        sim = z_rec @ z_rec.T
        np.fill_diagonal(sim, -np.inf)
        neighbor_idx = np.argsort(-sim, axis=1, kind="mergesort")[:, :k_eff]
        neighbor_labels = y_rec[neighbor_idx]
        same_neighbor = neighbor_labels == y_rec[:, None]
        sample_purity = np.mean(same_neighbor.astype(np.float32), axis=1)
        pred_labels = np.zeros_like(y_rec)
        for i in range(y_rec.shape[0]):
            labels_i, counts_i = np.unique(neighbor_labels[i], return_counts=True)
            pred_labels[i] = int(labels_i[int(np.argmax(counts_i))])

        unit_purity_vals: List[float] = []
        unit_knn_acc_vals: List[float] = []
        for unit_value in selected_units:
            mask = y_rec == int(unit_value)
            if np.any(mask):
                unit_purity_vals.append(float(np.mean(sample_purity[mask].astype(np.float32))))
                unit_knn_acc_vals.append(float(np.mean((pred_labels[mask] == int(unit_value)).astype(np.float32))))

        proto_labels = np.asarray(selected_units, dtype=np.int64)
        proto_parts = []
        for unit_value in selected_units:
            proto = np.mean(z_rec[y_rec == int(unit_value)], axis=0).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(proto))
            if norm > 1.0e-8:
                proto = proto / norm
            proto_parts.append(proto)
        proto_mat = np.stack(proto_parts, axis=0).astype(np.float32, copy=False)
        centroid_acc_unit_vals: List[float] = []
        margin_unit_vals: List[float] = []
        for unit_value in selected_units:
            mask = y_rec == int(unit_value)
            z_unit = z_rec[mask]
            if z_unit.shape[0] < 2:
                continue
            other_proto = proto_mat[proto_labels != int(unit_value)]
            if other_proto.shape[0] == 0:
                continue
            own_sum = np.sum(z_unit, axis=0)
            own_proto = (own_sum[None, :] - z_unit) / float(max(1, z_unit.shape[0] - 1))
            own_proto = own_proto / np.maximum(np.linalg.norm(own_proto, axis=1, keepdims=True), 1.0e-8)
            own_sim = np.sum(z_unit * own_proto, axis=1)
            other_sim = z_unit @ other_proto.T
            other_best = np.max(other_sim, axis=1)
            own_global_idx = int(np.flatnonzero(proto_labels == int(unit_value))[0])
            all_sim = z_unit @ proto_mat.T
            all_sim[:, own_global_idx] = own_sim
            pred = proto_labels[np.argmax(all_sim, axis=1)]
            centroid_acc_unit_vals.append(float(np.mean((pred == int(unit_value)).astype(np.float32))))
            margin_unit_vals.append(float(np.mean((own_sim - other_best).astype(np.float32))))

        if unit_purity_vals:
            rec_purity_vals.append(float(np.mean(np.asarray(unit_purity_vals, dtype=np.float32))))
        if unit_knn_acc_vals:
            rec_knn_acc_vals.append(float(np.mean(np.asarray(unit_knn_acc_vals, dtype=np.float32))))
        if centroid_acc_unit_vals:
            rec_centroid_acc_vals.append(float(np.mean(np.asarray(centroid_acc_unit_vals, dtype=np.float32))))
        if margin_unit_vals:
            rec_margin_vals.append(float(np.mean(np.asarray(margin_unit_vals, dtype=np.float32))))
        rec_unit_counts.append(int(len(selected_units)))
        rec_sample_counts.append(int(y_rec.shape[0]))
        rec_k_vals.append(int(k_eff))

    if rec_purity_vals:
        out["cluster_knn_purity"] = float(np.mean(np.asarray(rec_purity_vals, dtype=np.float32)))
        out["cluster_recordings"] = float(len(rec_purity_vals))
    if rec_knn_acc_vals:
        out["cluster_knn_acc"] = float(np.mean(np.asarray(rec_knn_acc_vals, dtype=np.float32)))
    if rec_centroid_acc_vals:
        out["cluster_centroid_acc"] = float(np.mean(np.asarray(rec_centroid_acc_vals, dtype=np.float32)))
    if rec_margin_vals:
        out["cluster_centroid_margin"] = float(np.mean(np.asarray(rec_margin_vals, dtype=np.float32)))
    if rec_unit_counts:
        out["cluster_units"] = float(np.sum(np.asarray(rec_unit_counts, dtype=np.int64)))
    if rec_sample_counts:
        out["cluster_samples"] = float(np.sum(np.asarray(rec_sample_counts, dtype=np.int64)))
    if rec_k_vals:
        out["cluster_mean_k"] = float(np.mean(np.asarray(rec_k_vals, dtype=np.float32)))
    return out


def _normalized_embedding(z: np.ndarray) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1.0e-8)


def _local_distance_arrays(z: np.ndarray, labels: np.ndarray, *, k_pos: int = 10, k_neg: int = 10) -> tuple[np.ndarray, np.ndarray]:
    z_arr = _normalized_embedding(z)
    y = np.asarray(labels, dtype=np.int64)
    n = int(y.shape[0])
    d_pos = np.full((n,), np.nan, dtype=np.float32)
    d_neg = np.full((n,), np.nan, dtype=np.float32)
    if n <= 1 or np.unique(y).size < 2:
        return d_pos, d_neg
    dist = 1.0 - np.clip(z_arr @ z_arr.T, -1.0, 1.0)
    np.fill_diagonal(dist, np.inf)
    for i in range(n):
        same = np.flatnonzero(y == int(y[i])).astype(np.int64, copy=False)
        same = same[same != i]
        if same.size > 0:
            kk = min(int(max(1, k_pos)), int(same.size))
            d_pos[i] = float(np.mean(np.partition(dist[i, same], kk - 1)[:kk]))
        neg_means: List[float] = []
        for c in np.unique(y).tolist():
            if int(c) == int(y[i]):
                continue
            idx = np.flatnonzero(y == int(c)).astype(np.int64, copy=False)
            if idx.size == 0:
                continue
            kk = min(int(max(1, k_neg)), int(idx.size))
            neg_means.append(float(np.mean(np.partition(dist[i, idx], kk - 1)[:kk])))
        if neg_means:
            d_neg[i] = float(np.min(np.asarray(neg_means, dtype=np.float32)))
    return d_pos, d_neg


def compute_local_separation_margin(z: np.ndarray, labels: np.ndarray, k_pos: int = 10, k_neg: int = 10) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    d_pos, d_neg = _local_distance_arrays(z, y, k_pos=k_pos, k_neg=k_neg)
    nmargin = (d_neg - d_pos) / (d_neg + d_pos + 1.0e-8)
    vals = []
    for c in np.unique(y).tolist():
        m = (y == int(c)) & np.isfinite(nmargin)
        if np.any(m):
            vals.append(float(np.mean(nmargin[m].astype(np.float32))))
    return {f"lsm{k_pos}": float(np.mean(np.asarray(vals, dtype=np.float32))) if vals else float("nan")}


def compute_boundary_error_rate(z: np.ndarray, labels: np.ndarray, k: int = 10, margins: List[float] | tuple[float, ...] = (0.0, 0.02, 0.05)) -> Dict[str, float]:
    d_pos, d_neg = _local_distance_arrays(z, labels, k_pos=k, k_neg=k)
    valid = np.isfinite(d_pos) & np.isfinite(d_neg)
    out: Dict[str, float] = {}
    for margin in margins:
        key = f"ber{int(round(float(margin) * 100)):03d}"
        out[key] = float(np.mean((d_neg[valid] <= d_pos[valid] + float(margin)).astype(np.float32))) if np.any(valid) else float("nan")
    return out


def compute_local_silhouette(z: np.ndarray, labels: np.ndarray, k: int = 10) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    d_pos, d_neg = _local_distance_arrays(z, y, k_pos=k, k_neg=k)
    sil = (d_neg - d_pos) / np.maximum(np.maximum(d_pos, d_neg), 1.0e-8)
    vals = []
    for c in np.unique(y).tolist():
        m = (y == int(c)) & np.isfinite(sil)
        if np.any(m):
            vals.append(float(np.mean(sil[m].astype(np.float32))))
    return {f"local_silhouette{k}": float(np.mean(np.asarray(vals, dtype=np.float32))) if vals else float("nan")}


def sample_recording_task_embedding_stats(
    z_val: np.ndarray,
    y_val: np.ndarray,
    recording_id: np.ndarray,
    event_time_sample: np.ndarray | None = None,
    *,
    min_samples_per_unit: int = 8,
    max_samples_per_unit: int = 128,
    max_samples_per_recording: int = 4096,
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "knn_purity5": float("nan"),
        "knn_purity10": float("nan"),
        "knn_purity20": float("nan"),
        "lsm10": float("nan"),
        "ber000": float("nan"),
        "ber002": float("nan"),
        "ber005": float("nan"),
        "local_silhouette10": float("nan"),
        "label_conductance15": float("nan"),
        "graph_separation15": float("nan"),
        "recording_task_recordings": 0.0,
        "recording_task_units": 0.0,
        "recording_task_samples": 0.0,
    }
    if z_val.shape[0] == 0:
        return out
    y_arr = np.asarray(y_val, dtype=np.int64)
    rec_arr = np.asarray(recording_id, dtype=np.int64)
    idx = _stable_recording_cluster_eval_indices(
        y_arr,
        rec_arr,
        event_time_sample,
        min_samples_per_unit=min_samples_per_unit,
        max_samples_per_unit=max_samples_per_unit,
        max_samples_per_recording=max_samples_per_recording,
    )
    if idx.size == 0:
        return out
    metric_parts: Dict[str, List[float]] = {key: [] for key in out if key not in ("recording_task_recordings", "recording_task_units", "recording_task_samples")}
    unit_total = 0
    sample_total = 0
    rec_count = 0
    for rec_value in np.unique(rec_arr[idx]).tolist():
        rec_idx = idx[rec_arr[idx] == int(rec_value)]
        if rec_idx.size == 0 or np.unique(y_arr[rec_idx]).size < 2:
            continue
        z_rec = z_val[rec_idx]
        y_rec = y_arr[rec_idx]
        metrics_rec: Dict[str, float] = {}
        metrics_rec.update(compute_knn_purity(z_rec, y_rec, k_list=(5, 10, 20), macro_by_label=True))
        metrics_rec.update(compute_local_separation_margin(z_rec, y_rec, k_pos=10, k_neg=10))
        metrics_rec.update(compute_boundary_error_rate(z_rec, y_rec, k=10, margins=(0.0, 0.02, 0.05)))
        metrics_rec.update(compute_local_silhouette(z_rec, y_rec, k=10))
        metrics_rec.update(compute_label_conductance(z_rec, y_rec, k=15, mutual=True, tau=0.1))
        for key, value in metrics_rec.items():
            if key in metric_parts and np.isfinite(value):
                metric_parts[key].append(float(value))
        rec_count += 1
        unit_total += int(np.unique(y_rec).size)
        sample_total += int(y_rec.size)
    for key, values in metric_parts.items():
        if values:
            out[key] = float(np.mean(np.asarray(values, dtype=np.float32)))
    out["recording_task_recordings"] = float(rec_count)
    out["recording_task_units"] = float(unit_total)
    out["recording_task_samples"] = float(sample_total)
    return out


def evaluate_stage2_encoder_metrics(
    model: SpikeEmbeddingModel,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_geo: np.ndarray,
    train_recording_id: np.ndarray,
    train_spike_label: np.ndarray | None,
    train_sample_source: np.ndarray | None,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_geo: np.ndarray,
    val_recording_id: np.ndarray,
    val_spike_label: np.ndarray | None,
    val_sample_source: np.ndarray | None,
    *,
    allowed_units: np.ndarray | None,
    device: torch.device,
    seed: int,
    support_per_class: int = 32,
    max_eval_samples: int = 20000,
    eval_batch_size: int = 1024,
    pair_samples: int = 10000,
    val_event_time_sample: np.ndarray | None = None,
    apply_encode_gate: bool = False,
    cluster_enabled: bool = True,
    cluster_knn_k_target: int = 15,
    cluster_min_knn_k: int = 3,
    cluster_min_samples_per_unit: int = 8,
    cluster_max_samples_per_unit: int = 128,
    cluster_max_samples_per_recording: int = 4096,
    metric_prefix: str = "val",
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        f"{metric_prefix}_proto_acc_gt": float("nan"),
        f"{metric_prefix}_pos_cos_gt": float("nan"),
        f"{metric_prefix}_neg_cos_gt": float("nan"),
        f"{metric_prefix}_gap_gt": float("nan"),
        f"{metric_prefix}_cross_domain_pos_cos": float("nan"),
        f"{metric_prefix}_cross_domain_neg_cos": float("nan"),
        f"{metric_prefix}_cross_domain_gap": float("nan"),
        f"{metric_prefix}_recording_proto_acc_gt": float("nan"),
        f"{metric_prefix}_recording_pos_cos_gt": float("nan"),
        f"{metric_prefix}_recording_neg_cos_gt": float("nan"),
        f"{metric_prefix}_recording_gap_gt": float("nan"),
        f"{metric_prefix}_recordings_gt": 0.0,
        f"{metric_prefix}_holdout_proto_acc_gt": float("nan"),
        f"{metric_prefix}_holdout_pos_cos_gt": float("nan"),
        f"{metric_prefix}_holdout_neg_cos_gt": float("nan"),
        f"{metric_prefix}_holdout_gap_gt": float("nan"),
        f"{metric_prefix}_holdout_recordings_gt": 0.0,
        f"{metric_prefix}_holdout_units_gt": 0.0,
        f"{metric_prefix}_holdout_samples_gt": 0.0,
        f"{metric_prefix}_cluster_knn_purity_gt": float("nan"),
        f"{metric_prefix}_cluster_knn_acc_gt": float("nan"),
        f"{metric_prefix}_cluster_centroid_acc_gt": float("nan"),
        f"{metric_prefix}_cluster_centroid_margin_gt": float("nan"),
        f"{metric_prefix}_cluster_recordings_gt": 0.0,
        f"{metric_prefix}_cluster_units_gt": 0.0,
        f"{metric_prefix}_cluster_samples_gt": 0.0,
        f"{metric_prefix}_cluster_mean_k_gt": float("nan"),
        f"{metric_prefix}_knn_purity5_gt": float("nan"),
        f"{metric_prefix}_knn_purity10_gt": float("nan"),
        f"{metric_prefix}_knn_purity20_gt": float("nan"),
        f"{metric_prefix}_lsm10_gt": float("nan"),
        f"{metric_prefix}_ber000_gt": float("nan"),
        f"{metric_prefix}_ber002_gt": float("nan"),
        f"{metric_prefix}_ber005_gt": float("nan"),
        f"{metric_prefix}_local_silhouette10_gt": float("nan"),
        f"{metric_prefix}_label_conductance15_gt": float("nan"),
        f"{metric_prefix}_graph_separation15_gt": float("nan"),
        f"{metric_prefix}_recording_task_recordings_gt": 0.0,
        f"{metric_prefix}_recording_task_units_gt": 0.0,
        f"{metric_prefix}_recording_task_samples_gt": 0.0,
        f"{metric_prefix}_samples_gt": 0.0,
        f"{metric_prefix}_classes_gt": 0.0,
    }
    val_mask = build_gt_spike_mask(val_y, val_spike_label, val_sample_source)
    # Stage2 recording-holdout validation is intentionally internal to the held-out
    # recording.  The legacy train/allowed-unit arguments are kept for call-site
    # compatibility, but train prototypes are no longer used here.
    val_x = val_x[val_mask]
    val_geo = val_geo[val_mask]
    val_recording_id = np.asarray(val_recording_id, dtype=np.int16)[val_mask]
    y_val = np.asarray(val_y, dtype=np.int64)[val_mask]
    if val_event_time_sample is not None:
        val_event_time = np.asarray(val_event_time_sample, dtype=np.int64)[val_mask]
    else:
        val_event_time = None
    if val_x.shape[0] == 0:
        return metrics

    if cluster_enabled:
        cluster_idx = _stable_recording_cluster_eval_indices(
            y_val,
            val_recording_id,
            val_event_time,
            min_samples_per_unit=int(cluster_min_samples_per_unit),
            max_samples_per_unit=int(cluster_max_samples_per_unit),
            max_samples_per_recording=int(cluster_max_samples_per_recording),
        )
        if cluster_idx.size > 0:
            z_cluster = embed_numpy(
                model,
                val_x[cluster_idx],
                val_geo[cluster_idx],
                device=device,
                batch_size=eval_batch_size,
                apply_encode_gate=apply_encode_gate,
            )
            cluster_stats = sample_stable_recording_cluster_stats(
                z_cluster,
                y_val[cluster_idx],
                val_recording_id[cluster_idx],
                val_event_time[cluster_idx] if val_event_time is not None else None,
                knn_k_target=int(cluster_knn_k_target),
                min_knn_k=int(cluster_min_knn_k),
                min_samples_per_unit=int(cluster_min_samples_per_unit),
                max_samples_per_unit=int(cluster_max_samples_per_unit),
                max_samples_per_recording=int(cluster_max_samples_per_recording),
            )
            metrics[f"{metric_prefix}_cluster_knn_purity_gt"] = float(cluster_stats["cluster_knn_purity"])
            metrics[f"{metric_prefix}_cluster_knn_acc_gt"] = float(cluster_stats["cluster_knn_acc"])
            metrics[f"{metric_prefix}_cluster_centroid_acc_gt"] = float(cluster_stats["cluster_centroid_acc"])
            metrics[f"{metric_prefix}_cluster_centroid_margin_gt"] = float(cluster_stats["cluster_centroid_margin"])
            metrics[f"{metric_prefix}_cluster_recordings_gt"] = float(cluster_stats["cluster_recordings"])
            metrics[f"{metric_prefix}_cluster_units_gt"] = float(cluster_stats["cluster_units"])
            metrics[f"{metric_prefix}_cluster_samples_gt"] = float(cluster_stats["cluster_samples"])
            metrics[f"{metric_prefix}_cluster_mean_k_gt"] = float(cluster_stats["cluster_mean_k"])
            recording_task_stats = sample_recording_task_embedding_stats(
                z_cluster,
                y_val[cluster_idx],
                val_recording_id[cluster_idx],
                val_event_time[cluster_idx] if val_event_time is not None else None,
                min_samples_per_unit=int(cluster_min_samples_per_unit),
                max_samples_per_unit=int(cluster_max_samples_per_unit),
                max_samples_per_recording=int(cluster_max_samples_per_recording),
            )
            for key, value in recording_task_stats.items():
                metrics[f"{metric_prefix}_{key}_gt"] = float(value)

    val_idx = _balanced_holdout_eval_indices(
        y_val,
        val_recording_id,
        max_eval_samples=int(max_eval_samples),
        seed=seed + 5,
    )
    if val_idx.size == 0:
        return metrics
    z_val = embed_numpy(
        model,
        val_x[val_idx],
        val_geo[val_idx],
        device=device,
        batch_size=eval_batch_size,
        apply_encode_gate=apply_encode_gate,
    )
    y_val_sub = y_val[val_idx]
    val_recording_sub = val_recording_id[val_idx]
    if not np.isfinite(float(metrics.get(f"{metric_prefix}_lsm10_gt", float("nan")))):
        val_event_time_sub = val_event_time[val_idx] if val_event_time is not None else None
        recording_task_stats = sample_recording_task_embedding_stats(
            z_val,
            y_val_sub,
            val_recording_sub,
            val_event_time_sub,
            min_samples_per_unit=int(cluster_min_samples_per_unit),
            max_samples_per_unit=int(cluster_max_samples_per_unit),
            max_samples_per_recording=int(cluster_max_samples_per_recording),
        )
        for key, value in recording_task_stats.items():
            metrics[f"{metric_prefix}_{key}_gt"] = float(value)
    holdout_sep = sample_holdout_recording_embedding_stats(
        z_val,
        y_val_sub,
        val_recording_sub,
        support_per_unit=int(support_per_class),
        seed=seed + 17,
    )
    metrics[f"{metric_prefix}_holdout_proto_acc_gt"] = float(holdout_sep["holdout_proto_acc"])
    metrics[f"{metric_prefix}_holdout_pos_cos_gt"] = float(holdout_sep["holdout_pos_cos"])
    metrics[f"{metric_prefix}_holdout_neg_cos_gt"] = float(holdout_sep["holdout_neg_cos"])
    metrics[f"{metric_prefix}_holdout_gap_gt"] = float(holdout_sep["holdout_gap"])
    metrics[f"{metric_prefix}_holdout_recordings_gt"] = float(holdout_sep["holdout_recordings"])
    metrics[f"{metric_prefix}_holdout_units_gt"] = float(holdout_sep["holdout_units"])
    metrics[f"{metric_prefix}_holdout_samples_gt"] = float(holdout_sep["holdout_samples"])

    metrics[f"{metric_prefix}_proto_acc_gt"] = float(holdout_sep["holdout_proto_acc"])
    metrics[f"{metric_prefix}_pos_cos_gt"] = float(holdout_sep["holdout_pos_cos"])
    metrics[f"{metric_prefix}_neg_cos_gt"] = float(holdout_sep["holdout_neg_cos"])
    metrics[f"{metric_prefix}_gap_gt"] = float(holdout_sep["holdout_gap"])
    metrics[f"{metric_prefix}_recording_proto_acc_gt"] = float(holdout_sep["holdout_proto_acc"])
    metrics[f"{metric_prefix}_recording_pos_cos_gt"] = float(holdout_sep["holdout_pos_cos"])
    metrics[f"{metric_prefix}_recording_neg_cos_gt"] = float(holdout_sep["holdout_neg_cos"])
    metrics[f"{metric_prefix}_recording_gap_gt"] = float(holdout_sep["holdout_gap"])
    metrics[f"{metric_prefix}_recordings_gt"] = float(holdout_sep["holdout_recordings"])
    metrics[f"{metric_prefix}_samples_gt"] = float(holdout_sep["holdout_samples"])
    metrics[f"{metric_prefix}_classes_gt"] = float(np.unique(y_val_sub).size)
    return metrics


def stage2_encoder_model_rank(
    metrics: Dict[str, float],
    *,
    primary_key: str | None = None,
    secondary_key: str | None = None,
    tertiary_key: str | None = None,
) -> tuple[float, ...]:
    if primary_key is None:
        for candidate in (
            "val_lsm10_operational_gt",
            "val_lsm10_gt",
            "val_knn_purity10_operational_gt",
            "val_knn_purity10_gt",
        ):
            if candidate in metrics:
                primary_key = candidate
                break
        else:
            primary_key = "val_lsm10_operational_gt"
    if secondary_key is None:
        for candidate in (
            "val_knn_purity10_operational_gt",
            "val_knn_purity10_gt",
            "val_lsm10_operational_gt",
            "val_lsm10_gt",
        ):
            if candidate in metrics:
                secondary_key = candidate
                break
        else:
            secondary_key = "val_knn_purity10_operational_gt"
    if tertiary_key is None:
        tertiary_key = "val_ber002_operational_gt"

    def _rank_value(key: str, *, lower_is_better: bool = False) -> float:
        value = float(metrics.get(key, float("nan")))
        if not np.isfinite(value):
            return -1.0
        return -value if lower_is_better else value

    primary = float(metrics.get(primary_key, float("nan")))
    secondary = float(metrics.get(secondary_key, float("nan")))
    return (
        primary if np.isfinite(primary) else -1.0,
        secondary if np.isfinite(secondary) else -1.0,
        _rank_value(tertiary_key, lower_is_better=("ber" in tertiary_key.lower())),
    )


def stage1_gate_model_rank(
    gate_summary: Dict[str, Any],
    encoder_metrics: Dict[str, float],
    *,
    goal_spike_recall: float,
    goal_noise_accuracy: float,
    selection_mode: str = "fixed_0_5_acceptance",
    acceptance_min_precision: float = 0.99,
    acceptance_min_recall: float = 0.90,
) -> tuple[float, ...]:
    _ = (encoder_metrics, goal_spike_recall, goal_noise_accuracy)
    mode = str(selection_mode).strip().lower()
    if mode not in ("fixed_0_5_acceptance", "fixed_0.5_acceptance", "acceptance"):
        raise ValueError(
            "unsupported train.stage1_gate.selection_mode: "
            f"{selection_mode!r}; expected fixed_0_5_acceptance"
        )
    precision = float(gate_summary.get("gate_precision", float("nan")))
    recall = float(gate_summary.get("gate_recall", float("nan")))
    f05 = float(gate_summary.get("gate_f0_5", float("nan")))
    return (
        1.0 if precision + 1e-8 >= float(acceptance_min_precision) and recall + 1e-8 >= float(acceptance_min_recall) else 0.0,
        f05 if np.isfinite(f05) else -1.0,
        precision if np.isfinite(precision) else -1.0,
        recall if np.isfinite(recall) else -1.0,
    )



def build_stage2_metric_view(
    raw_metrics: Dict[str, float],
    *,
    scope: str,
    train_samples: float | None = None,
    train_classes: float | None = None,
) -> Dict[str, float]:
    suffix = str(scope)
    out = {
        f"val_proto_acc_{suffix}_gt": float(raw_metrics.get("val_proto_acc_gt", float("nan"))),
        f"val_pos_cos_{suffix}_gt": float(raw_metrics.get("val_pos_cos_gt", float("nan"))),
        f"val_neg_cos_{suffix}_gt": float(raw_metrics.get("val_neg_cos_gt", float("nan"))),
        f"val_gap_{suffix}_gt": float(raw_metrics.get("val_gap_gt", float("nan"))),
        f"val_holdout_proto_acc_{suffix}_gt": float(raw_metrics.get("val_holdout_proto_acc_gt", float("nan"))),
        f"val_holdout_pos_cos_{suffix}_gt": float(raw_metrics.get("val_holdout_pos_cos_gt", float("nan"))),
        f"val_holdout_neg_cos_{suffix}_gt": float(raw_metrics.get("val_holdout_neg_cos_gt", float("nan"))),
        f"val_holdout_gap_{suffix}_gt": float(raw_metrics.get("val_holdout_gap_gt", float("nan"))),
        f"val_holdout_recordings_{suffix}_gt": float(raw_metrics.get("val_holdout_recordings_gt", 0.0)),
        f"val_holdout_units_{suffix}_gt": float(raw_metrics.get("val_holdout_units_gt", 0.0)),
        f"val_holdout_samples_{suffix}_gt": float(raw_metrics.get("val_holdout_samples_gt", 0.0)),
        f"val_cluster_knn_purity_{suffix}_gt": float(raw_metrics.get("val_cluster_knn_purity_gt", float("nan"))),
        f"val_cluster_knn_acc_{suffix}_gt": float(raw_metrics.get("val_cluster_knn_acc_gt", float("nan"))),
        f"val_cluster_centroid_acc_{suffix}_gt": float(raw_metrics.get("val_cluster_centroid_acc_gt", float("nan"))),
        f"val_cluster_centroid_margin_{suffix}_gt": float(
            raw_metrics.get("val_cluster_centroid_margin_gt", float("nan"))
        ),
        f"val_cluster_recordings_{suffix}_gt": float(raw_metrics.get("val_cluster_recordings_gt", 0.0)),
        f"val_cluster_units_{suffix}_gt": float(raw_metrics.get("val_cluster_units_gt", 0.0)),
        f"val_cluster_samples_{suffix}_gt": float(raw_metrics.get("val_cluster_samples_gt", 0.0)),
        f"val_cluster_mean_k_{suffix}_gt": float(raw_metrics.get("val_cluster_mean_k_gt", float("nan"))),
        f"val_knn_purity5_{suffix}_gt": float(raw_metrics.get("val_knn_purity5_gt", float("nan"))),
        f"val_knn_purity10_{suffix}_gt": float(raw_metrics.get("val_knn_purity10_gt", float("nan"))),
        f"val_knn_purity20_{suffix}_gt": float(raw_metrics.get("val_knn_purity20_gt", float("nan"))),
        f"val_lsm10_{suffix}_gt": float(raw_metrics.get("val_lsm10_gt", float("nan"))),
        f"val_ber000_{suffix}_gt": float(raw_metrics.get("val_ber000_gt", float("nan"))),
        f"val_ber002_{suffix}_gt": float(raw_metrics.get("val_ber002_gt", float("nan"))),
        f"val_ber005_{suffix}_gt": float(raw_metrics.get("val_ber005_gt", float("nan"))),
        f"val_local_silhouette10_{suffix}_gt": float(raw_metrics.get("val_local_silhouette10_gt", float("nan"))),
        f"val_label_conductance15_{suffix}_gt": float(raw_metrics.get("val_label_conductance15_gt", float("nan"))),
        f"val_graph_separation15_{suffix}_gt": float(raw_metrics.get("val_graph_separation15_gt", float("nan"))),
        f"val_recording_task_recordings_{suffix}_gt": float(
            raw_metrics.get("val_recording_task_recordings_gt", 0.0)
        ),
        f"val_recording_task_units_{suffix}_gt": float(raw_metrics.get("val_recording_task_units_gt", 0.0)),
        f"val_recording_task_samples_{suffix}_gt": float(raw_metrics.get("val_recording_task_samples_gt", 0.0)),
        f"val_samples_{suffix}_gt": float(raw_metrics.get("val_samples_gt", 0.0)),
        f"val_classes_{suffix}_gt": float(raw_metrics.get("val_classes_gt", 0.0)),
    }
    if train_samples is not None:
        out[f"train_samples_{suffix}_gt"] = float(train_samples)
    if train_classes is not None:
        out[f"train_classes_{suffix}_gt"] = float(train_classes)
    return out



def resolve_stage_training_config(
    train_cfg: Dict[str, Any],
    val_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    has_explicit_stage1 = "stage1_gate" in train_cfg
    has_explicit_stage2 = "stage2_encoder" in train_cfg
    stage1_cfg = dict(train_cfg.get("stage1_gate", {}) or {})
    stage2_cfg = dict(train_cfg.get("stage2_encoder", {}) or {})
    legacy_total_epochs = int(train_cfg.get("epochs", 0))
    if not has_explicit_stage1 and not has_explicit_stage2:
        stage1_epochs = int(max(0, legacy_total_epochs))
        stage2_epochs = 0
    else:
        default_stage1_epochs = int(stage1_cfg.get("epochs", val_cfg.get("gate_warmup_epochs", 0)))
        if legacy_total_epochs > 0:
            default_stage1_epochs = int(min(default_stage1_epochs, legacy_total_epochs))
        stage1_epochs = int(stage1_cfg.get("epochs", default_stage1_epochs))
        if "epochs" in stage2_cfg:
            stage2_epochs = int(stage2_cfg.get("epochs", 0))
        elif legacy_total_epochs > 0:
            stage2_epochs = int(max(legacy_total_epochs - stage1_epochs, 0))
        else:
            stage2_epochs = 0
    stage1_cfg["epochs"] = int(max(0, stage1_epochs))
    stage2_cfg.setdefault("epochs", int(max(0, stage2_epochs)))
    stage2_cfg.setdefault("source_mode", "matched_positives")
    stage2_cfg.setdefault("freeze_scope", "gate_net_only")
    stage2_cfg.setdefault("data_filter", "frozen_encode_gate")
    stage2_cfg.setdefault("min_domains_per_unit", 1)
    stage2_cfg.setdefault("min_samples_per_unit_after_gate", None)
    stage2_cfg.setdefault("domain_sampling", "round_robin")
    stage2_cfg.setdefault("use_encode_gate", False)
    stage2_cfg.setdefault("select_metric_primary", "val_lsm10_operational_gt")
    stage2_cfg.setdefault("select_metric_secondary", "val_knn_purity10_operational_gt")
    stage2_cfg.setdefault("select_metric_tertiary", "val_ber002_operational_gt")
    stage2_cfg.setdefault("reset_optimizer_on_entry", True)
    stage2_cfg.setdefault("reset_scheduler_warmup_epochs", 2)
    stage1_encoder_aux_cfg = dict(stage1_cfg.get("encoder_aux", {}) or {})
    stage1_encoder_aux_cfg.setdefault("enabled", False)
    stage1_encoder_aux_cfg.setdefault("batch_size", None)
    stage1_encoder_aux_cfg.setdefault("loss_scale", 0.5)
    stage1_cfg["encoder_aux"] = stage1_encoder_aux_cfg
    return {
        "stage1_gate": stage1_cfg,
        "stage2_encoder": stage2_cfg,
        "total_epochs": int(stage1_cfg["epochs"] + int(stage2_cfg["epochs"])),
    }


def apply_freeze_scope(model: SpikeEmbeddingModel, *, freeze_scope: str) -> Dict[str, int]:
    for param in model.parameters():
        param.requires_grad = False
    frozen_params = 0
    trainable_params = 0
    scope = str(freeze_scope).strip().lower()
    if scope not in ("gate_net_only", "gate_path_frozen", "spike_head_only", "none", ""):
        raise ValueError(f"unsupported stage2 freeze_scope: {freeze_scope}")
    for param in model.encoder_net.parameters():
        param.requires_grad = True
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += int(param.numel())
        else:
            frozen_params += int(param.numel())
    return {
        "frozen_params": int(frozen_params),
        "trainable_params": int(trainable_params),
    }



def enforce_freeze_scope_modules(model: SpikeEmbeddingModel, *, freeze_scope: str) -> None:
    scope = str(freeze_scope).strip().lower()
    if scope not in ("gate_net_only", "gate_path_frozen", "spike_head_only", "none", ""):
        raise ValueError(f"unsupported stage2 freeze_scope: {freeze_scope}")
    model.gate_net.eval()


def apply_stage_trainability(model: SpikeEmbeddingModel, *, stage_name: str) -> Dict[str, int]:
    for param in model.parameters():
        param.requires_grad = False
    if stage_name == "stage1_gate":
        modules = (model.gate_net,)
    elif stage_name == "stage2_encoder":
        modules = (model.encoder_net,)
    else:
        raise ValueError(f"unsupported train stage: {stage_name}")
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True

    frozen_params = 0
    trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += int(param.numel())
        else:
            frozen_params += int(param.numel())
    return {
        "frozen_params": int(frozen_params),
        "trainable_params": int(trainable_params),
    }


def enforce_stage_modules(model: SpikeEmbeddingModel, *, stage_name: str) -> None:
    if stage_name == "stage1_gate":
        model.encoder_net.eval()
    elif stage_name == "stage2_encoder":
        model.gate_net.eval()
    else:
        raise ValueError(f"unsupported train stage: {stage_name}")



def clone_model_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}



def build_stage2_frozen_gate_subset(
    model: SpikeEmbeddingModel,
    x_np: np.ndarray,
    unit_id: np.ndarray,
    geo_np: np.ndarray,
    recording_id: np.ndarray,
    spike_label: np.ndarray | None,
    sample_source: np.ndarray | None,
    spatial_stats_np: np.ndarray | None = None,
    channel_mask_np: np.ndarray | None = None,
    strict_overlap_label: np.ndarray | None = None,
    overlap_label: np.ndarray | None = None,
    qc_allowed: np.ndarray | None = None,
    strict_matching_ambiguous: np.ndarray | None = None,
    relaxed_matching_ambiguous: np.ndarray | None = None,
    snr_peak: np.ndarray | None = None,
    event_amp_norm: np.ndarray | None = None,
    event_spatial_score: np.ndarray | None = None,
    event_simul_local: np.ndarray | None = None,
    event_simul_global: np.ndarray | None = None,
    stage2_qc_cfg: Dict[str, Any] | None = None,
    *,
    device: torch.device,
    batch_size: int,
    encode_gate_thresh: float,
    min_domains_per_unit: int,
    min_samples_per_unit: int,
    source_mode: str = "primary_only",
    apply_overlap_filter: bool = True,
    apply_qc_filter: bool = True,
) -> Dict[str, Any]:
    gate_prob = collect_gate_probabilities(
        model,
        x_np,
        geo_np,
        spatial_stats_np=spatial_stats_np,
        channel_mask_np=channel_mask_np,
        device=device,
        batch_size=batch_size,
    )
    gt_mask = build_gt_spike_mask(unit_id, spike_label, sample_source)
    positive_mask = build_stage2_positive_mask(
        unit_id,
        spike_label,
        sample_source,
        source_mode=source_mode,
    )
    gt_gate_pass_mask = gt_mask & (gate_prob >= float(encode_gate_thresh))
    gate_pass_mask = positive_mask & (gate_prob >= float(encode_gate_thresh))
    allowed_source_names = (
        ["primary_matched_positive"]
        if str(source_mode).strip().lower() == "primary_only"
        else ["primary_matched_positive", "secondary_matched_positive"]
    )
    valid_mask, valid_stats = build_stage2_quality_mask_np(
        unit_id=unit_id,
        spike_label=spike_label,
        sample_source=sample_source,
        gate_prob=gate_prob,
        gate_threshold=float(encode_gate_thresh),
        allowed_sources=allowed_source_names,
        strict_overlap_label=strict_overlap_label,
        overlap_label=overlap_label,
        qc_allowed=qc_allowed,
        strict_matching_ambiguous=strict_matching_ambiguous,
        relaxed_matching_ambiguous=relaxed_matching_ambiguous,
        snr_peak=snr_peak,
        event_amp_norm=event_amp_norm,
        event_spatial_score=event_spatial_score,
        event_simul_local=event_simul_local,
        event_simul_global=event_simul_global,
        qc_cfg=stage2_qc_cfg,
        apply_overlap_filter=apply_overlap_filter,
        apply_qc_filter=apply_qc_filter,
    )
    boundary_candidate_mask = build_boundary_candidate_mask_np(
        valid_mask=valid_mask,
        overlap_label=overlap_label,
        strict_overlap_label=strict_overlap_label,
        qc_allowed=qc_allowed,
        gate_prob=gate_prob,
        gate_threshold=float(encode_gate_thresh),
        snr_peak=snr_peak,
        event_amp_norm=event_amp_norm,
        event_spatial_score=event_spatial_score,
        event_simul_local=event_simul_local,
        event_simul_global=event_simul_global,
        strict_matching_ambiguous=strict_matching_ambiguous,
        relaxed_matching_ambiguous=relaxed_matching_ambiguous,
        qc_cfg=dict(stage2_qc_cfg or {}),
        apply_overlap_filter=apply_overlap_filter,
        apply_qc_filter=apply_qc_filter,
    )
    class_indices, positive_idx, support_map = resolve_stage2_gt_class_indices(
        unit_id,
        recording_id,
        spike_label,
        sample_source,
        min_domains_per_unit=min_domains_per_unit,
        extra_mask=valid_mask,
        min_samples_per_unit=min_samples_per_unit,
        source_mode=source_mode,
    )
    allowed_units = np.asarray(sorted(class_indices.keys()), dtype=np.int64)
    support_summary = summarize_unit_domain_support(support_map, min_domains_required=min_domains_per_unit)
    secondary_matched_positive_total = 0
    secondary_matched_positive_included = 0
    if sample_source is not None:
        sample_source_arr = np.asarray(sample_source, dtype=np.int8)
        secondary_matched_positive_mask = sample_source_arr == int(GATE_SOURCE_NAME_TO_CODE["secondary_matched_positive"])
        secondary_matched_positive_total = int(np.sum(secondary_matched_positive_mask))
        secondary_matched_positive_included = int(np.sum(secondary_matched_positive_mask & gate_pass_mask))
    return {
        "source_mode": str(source_mode).strip().lower(),
        "gate_prob": gate_prob.astype(np.float32, copy=False),
        "gt_mask": gt_mask.astype(np.bool_, copy=False),
        "positive_mask": positive_mask.astype(np.bool_, copy=False),
        "gate_pass_mask": gate_pass_mask.astype(np.bool_, copy=False),
        "gt_gate_pass_mask": gt_gate_pass_mask.astype(np.bool_, copy=False),
        "valid_mask": valid_mask.astype(np.bool_, copy=False),
        "boundary_candidate_mask": boundary_candidate_mask.astype(np.bool_, copy=False),
        "valid_stats": dict(valid_stats),
        "class_indices": class_indices,
        "positive_idx": positive_idx.astype(np.int64, copy=False),
        "support_map": support_map,
        "allowed_units": allowed_units,
        "support_summary": support_summary,
        "gate_pass_ratio_by_domain": summarize_gate_pass_ratios_by_domain(recording_id, gt_mask, gt_gate_pass_mask),
        "secondary_matched_positive_total": secondary_matched_positive_total,
        "secondary_matched_positive_included": secondary_matched_positive_included,
        "secondary_matched_positive_excluded": int(max(0, secondary_matched_positive_total - secondary_matched_positive_included)),
    }



def capture_gate_snapshot(
    model: SpikeEmbeddingModel,
    x_np: np.ndarray,
    geo_np: np.ndarray,
    spike_label: np.ndarray | None,
    spatial_stats_np: np.ndarray | None = None,
    channel_mask_np: np.ndarray | None = None,
    *,
    device: torch.device,
    batch_size: int,
    threshold: float,
) -> Dict[str, Any]:
    spike_prob = collect_gate_probabilities(
        model,
        x_np,
        geo_np,
        spatial_stats_np=spatial_stats_np,
        channel_mask_np=channel_mask_np,
        device=device,
        batch_size=batch_size,
    )
    metrics: Dict[str, float] = {}
    if spike_label is not None and np.asarray(spike_label).shape[0] == spike_prob.shape[0]:
        metrics = gate_metrics_from_probs(spike_prob, spike_label, threshold=float(threshold))
    return {
        "spike_prob": spike_prob.astype(np.float32, copy=False),
        "metrics": metrics,
        "n_samples": int(spike_prob.shape[0]),
    }



def compare_gate_snapshots(
    reference: Dict[str, Any] | None,
    current: Dict[str, Any] | None,
    *,
    atol: float,
) -> Dict[str, float]:
    if reference is None or current is None:
        return {"ok": 1.0, "max_abs_diff": 0.0}
    ref_prob = np.asarray(reference.get("spike_prob", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    cur_prob = np.asarray(current.get("spike_prob", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    if ref_prob.shape != cur_prob.shape:
        return {"ok": 0.0, "max_abs_diff": float("inf")}
    max_abs_diff = float(np.max(np.abs(ref_prob - cur_prob))) if ref_prob.size > 0 else 0.0
    return {"ok": 1.0 if max_abs_diff <= float(atol) else 0.0, "max_abs_diff": max_abs_diff}



def run_gate_validation(
    model: SpikeEmbeddingModel,
    x_np: np.ndarray,
    geo_np: np.ndarray,
    spike_label: np.ndarray,
    spatial_stats_np: np.ndarray | None = None,
    channel_mask_np: np.ndarray | None = None,
    *,
    device: torch.device,
    batch_size: int,
    fixed_gate_threshold: float | None,
    gate_goal_spike_recall: float,
    gate_goal_noise_accuracy: float,
    gate_target_recall_encode: float,
    gate_target_recall_assignment_min: float,
    domain_id: np.ndarray | None = None,
    domain_type: np.ndarray | None = None,
    eval_mask: np.ndarray | None = None,
    sample_source: np.ndarray | None = None,
    gate_log_source_metrics: bool = False,
) -> Dict[str, Any]:
    gate_prob = collect_gate_probabilities(
        model,
        x_np,
        geo_np,
        spatial_stats_np=spatial_stats_np,
        channel_mask_np=channel_mask_np,
        device=device,
        batch_size=batch_size,
    )
    metric_mask = np.ones((gate_prob.shape[0],), dtype=np.bool_)
    if eval_mask is not None:
        metric_mask = np.asarray(eval_mask, dtype=np.bool_).reshape(-1)
        if metric_mask.shape[0] != gate_prob.shape[0]:
            raise ValueError("eval_mask must match validation sample count")
    metric_gate_prob = gate_prob[metric_mask]
    metric_spike_label = np.asarray(spike_label, dtype=np.float32)[metric_mask]
    metric_sample_source = np.asarray(sample_source)[metric_mask] if sample_source is not None else None
    metric_domain_type = _normalize_domain_type_array(domain_type, int(gate_prob.shape[0]))[metric_mask] if domain_type is not None else None
    _ = (domain_id, gate_log_source_metrics, gate_target_recall_encode, gate_target_recall_assignment_min)
    calibration = build_operational_gate_summary(
        metric_gate_prob,
        metric_spike_label,
        fixed_gate_threshold=fixed_gate_threshold,
        gate_goal_spike_recall=gate_goal_spike_recall,
        gate_goal_noise_accuracy=gate_goal_noise_accuracy,
        domain_type=metric_domain_type,
        sample_source=metric_sample_source,
    )
    encode_metrics = dict(calibration.get("encode_metrics", {}))
    assignment_metrics = dict(calibration.get("assignment_metrics", {}))
    source_metrics = dict(calibration.get("source_metrics", {}))
    detector_summary = detector_domain_gate_summary(source_metrics)
    calibration.update(detector_summary)
    return {
        "calibration": calibration,
        "encode_metrics": encode_metrics,
        "assignment_metrics": assignment_metrics,
        "source_metrics": source_metrics,
        "spike_prob": gate_prob.astype(np.float32, copy=False),
    }


def gate_selection_summary(calibration: Dict[str, Any]) -> Dict[str, float]:
    assignment_metrics = dict(calibration.get("assignment_metrics", {}))
    return {
        "gate_recall": float(calibration.get("gate_recall", assignment_metrics.get("gate_recall", float("nan")))),
        "gate_precision": float(calibration.get("gate_precision", assignment_metrics.get("gate_precision", float("nan")))),
        "noise_accuracy": float(calibration.get("noise_accuracy", assignment_metrics.get("noise_accuracy", float("nan")))),
        "balanced_accuracy": float(calibration.get("balanced_accuracy", assignment_metrics.get("balanced_accuracy", float("nan")))),
        "gate_f1": float(calibration.get("gate_f1", assignment_metrics.get("gate_f1", float("nan")))),
        "gate_f0_5": float(calibration.get("gate_f0_5", assignment_metrics.get("gate_f0_5", float("nan")))),
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "train.yaml"))
    src_group = ap.add_mutually_exclusive_group(required=False)
    src_group.add_argument("--patches", type=str, default=None, help="path to patches.npy")
    src_group.add_argument(
        "--mearec-h5",
        type=str,
        nargs="+",
        default=None,
        help="one or more MEArec h5 files for pretraining",
    )
    ap.add_argument("--labels", type=str, default=None, help="path to events_labels.npz (required with --patches)")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--device", type=str, choices=["auto", "cuda"], default=None)
    ap.add_argument("--log-file", type=str, default=None, help="training log file path")
    ap.add_argument("--resume-from", type=str, default=None, help="resume checkpoint path")
    ap.add_argument("--auto-resume", action="store_true", help="resume from latest checkpoint in checkpoint_dir")
    ap.add_argument("--top-k-channels", type=int, default=None, help="keep top-K channels by energy for each template")
    ap.add_argument("--max-per-unit", type=int, default=None, help="max samples per neuron id (0 = keep all)")
    ap.add_argument("--merge-dt-ms", type=float, default=None, help="dedup time window in ms for multi-channel spikes")
    ap.add_argument("--no-dedup-spikes", action="store_true", help="disable multi-channel spike deduplication")
    ap.add_argument(
        "--dedup-mode",
        type=str,
        default=None,
        help="dedup mode: neighbors | same_channel | time_only | none",
    )
    ap.add_argument("--electrode-config", type=str, default=None, help="electrode config for building neighbor graph")
    ap.add_argument(
        "--no-normalize-templates",
        action="store_true",
        help="disable normalization when using --mearec-h5",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config, inputs, cache state, device, and model construction without loading data or training",
    )
    ap.add_argument("--manifest", help="ordered training recording manifest")
    ap.add_argument("--data-root", help="root containing manifest recording paths")
    ap.add_argument("--stage2-init-from", help="start encoder training from a stage-1 checkpoint")
    args = ap.parse_args()
    args.config = str(resolve_path(args.config, ROOT))

    cfg_preview = load_yaml(args.config)
    manifest_spec = args.manifest or cfg_preview.get("data", {}).get("manifest")
    training_manifest = None
    if manifest_spec and not args.mearec_h5 and not args.patches:
        training_manifest = load_recording_manifest(resolve_path(manifest_spec, ROOT))
        cfg_preview["data"]["mearec_h5_paths"] = [str(p) for p in resolve_data_records(
            training_manifest, args.data_root or cfg_preview["data"].get("data_root", "data/recordings"), ROOT)]
    train_preview = cfg_preview.get("train", {})
    if args.log_file:
        log_path = resolve_path(args.log_file, ROOT)
    elif train_preview.get("log_file"):
        log_path = resolve_path(str(train_preview.get("log_file")), ROOT)
    else:
        log_path = ROOT / "data" / "logs" / "pretrain.log"

    logger = build_file_logger(log_path)
    logger.info("launch pretrain")
    logger.info("argv=%s", " ".join(sys.argv))
    logger.info("config_path=%s", args.config)
    logger.info("cwd=%s", os.getcwd())

    logger.info("stage=load_config start")
    cfg = cfg_preview
    loss_ablation_cfg = cfg.get("loss_ablation")
    if isinstance(loss_ablation_cfg, dict) and loss_ablation_cfg.get("mode") is not None:
        logger.warning(
            "loss_ablation.mode is deprecated and does not control stage2 loss; "
            "use stage2.losses.*.enabled/weight instead."
        )
    validate_pretrain_config_structure(cfg)
    logger.info("stage=load_config done")
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    val_cfg = train_cfg.get("validation", {})
    stage_cfg = resolve_stage_training_config(train_cfg, val_cfg)
    stage1_cfg = dict(stage_cfg["stage1_gate"])
    stage2_cfg = dict(stage_cfg["stage2_encoder"])
    stage1_epochs = int(stage1_cfg.get("epochs", 0))
    stage2_epochs = int(stage2_cfg.get("epochs", 0))
    total_epochs = int(stage_cfg.get("total_epochs", stage1_epochs + stage2_epochs))
    stage2_schedule_epochs = int(stage2_cfg.get("schedule_epochs", stage2_epochs))
    if stage2_schedule_epochs < stage2_epochs:
        raise ValueError("stage2 schedule_epochs must cover all execution epochs")

    seed = int(cfg.get("seed", 42))
    logger.info("stage=set_seed start seed=%d", seed)
    set_seed(seed)
    logger.info("stage=set_seed done")

    cfg_mearec = data_cfg.get("mearec_h5_paths", [])
    if isinstance(cfg_mearec, (list, tuple)):
        mearec_h5 = [str(resolve_path(p, ROOT)) for p in cfg_mearec]
    elif cfg_mearec:
        mearec_h5 = [str(resolve_path(cfg_mearec, ROOT))]
    else:
        mearec_h5 = []
    if args.mearec_h5:
        mearec_h5 = [str(resolve_path(p, ROOT)) for p in args.mearec_h5]

    patches_raw = args.patches or data_cfg.get("patches")
    labels_raw = args.labels or data_cfg.get("labels")
    patches_path = resolve_path(patches_raw, ROOT) if patches_raw else None
    labels_path = resolve_path(labels_raw, ROOT) if labels_raw else None

    if not mearec_h5 and patches_path is None:
        raise ValueError("no training data source configured; set data.mearec_h5_paths or pass --mearec-h5 / --patches")
    if patches_path is not None and labels_path is None:
        raise ValueError("labels path is required when using cached patches")

    default_save = str(ROOT / "data" / "cache" / "pretrained.pt")
    save_out = str(resolve_path(args.save or train_cfg.get("save", default_save), ROOT))
    checkpoint_dir = resolve_path(train_cfg.get("checkpoint_dir", "data/checkpoints/pretrain"), ROOT)
    checkpoint_every = int(train_cfg.get("checkpoint_every_epochs", 20))
    auto_resume = bool(train_cfg.get("auto_resume", False)) or args.auto_resume
    resume_from_cfg = train_cfg.get("resume_from", "")
    resume_from = args.resume_from or (str(resume_from_cfg) if resume_from_cfg else None)
    stage2_init_raw = args.stage2_init_from or train_cfg.get("stage2_init_from")
    stage2_init_path = resolve_path(stage2_init_raw, ROOT) if stage2_init_raw else None
    if stage2_init_path is not None:
        if resume_from or auto_resume:
            raise ValueError("stage2-init-from cannot be combined with resume")
        if stage1_epochs <= 0 or stage2_epochs <= 0:
            raise ValueError("stage2-init-from requires configured stage-1 and stage-2 epochs")
        if not stage2_init_path.is_file():
            raise FileNotFoundError(f"stage-1 checkpoint not found: {stage2_init_path}")
    log_every_steps = int(train_cfg.get("log_every_steps", 0))
    logger.info("data_source mearec_files=%d cached_patches=%s", len(mearec_h5), str(patches_path))
    logger.info("log_every_steps=%d", log_every_steps)
    logger.info("checkpoint_dir=%s checkpoint_every_epochs=%d", checkpoint_dir, checkpoint_every)
    logger.info("resume_from=%s auto_resume=%s", str(resume_from), str(auto_resume))
    logger.info(
        "train_stage_setup total_epochs=%d stage1_gate_epochs=%d stage2_encoder_epochs=%d",
        total_epochs,
        stage1_epochs,
        stage2_epochs,
    )
    loader_kwargs = build_mearec_loader_kwargs(args, data_cfg, seed) if mearec_h5 else {}
    model_cfg_raw = dict(cfg.get("model", {}) or {})
    resolved_encoder_route = validate_pretrain_config_compatibility(
        data_cfg=data_cfg,
        model_cfg=model_cfg_raw,
        loader_kwargs=loader_kwargs,
    )
    if "encoder_route" not in model_cfg_raw:
        model_cfg_raw["encoder_route"] = resolved_encoder_route
    logger.info(
        "pretrain_mode data_peak_only=%s top_k_channels=%s encoder_route=%s",
        str(bool(data_cfg.get("peak_only", False))),
        str(loader_kwargs.get("top_k_channels", data_cfg.get("top_k_channels", ""))),
        resolved_encoder_route,
    )
    if mearec_h5:
        validate_mearec_h5_paths(mearec_h5)
        audit_operational_pretrain_cli_args(args)
        audit_operational_pretrain_data_config(data_cfg, logger=logger)
        effective_operational_cfg = build_effective_operational_pretrain_config(loader_kwargs)
        log_effective_operational_pretrain_config(logger, effective_operational_cfg)
        logger.info("recording_id_map %s", format_recording_id_path_map(recording_id_path_map(mearec_h5)))

    auto_cache_enabled = bool(mearec_h5) and patches_path is not None and labels_path is not None
    use_cached_data = bool(patches_path is not None and not mearec_h5)
    cache_state: str | None = None
    cache_reason: str | None = None
    manifest_path: Path | None = None
    if auto_cache_enabled:
        assert patches_path is not None
        assert labels_path is not None
        cache_state, cache_reason, manifest_path = check_pretrain_cache(
            mearec_h5_paths=mearec_h5,
            loader_kwargs=loader_kwargs,
            patches_path=patches_path,
            labels_path=labels_path,
            root=ROOT,
        )
        logger.info("cache_check patches=%s labels=%s manifest=%s", patches_path, labels_path, manifest_path)
        if cache_state == "hit":
            use_cached_data = True
            logger.info("cache_hit reason=%s manifest=%s", cache_reason, manifest_path)
        elif cache_state == "miss":
            logger.info("cache_miss reason=%s manifest=%s", cache_reason, manifest_path)
        else:
            logger.info("cache_stale reason=%s manifest=%s", cache_reason, manifest_path)

    cfg_device = str(train_cfg.get("device", "auto"))
    requested_device = args.device or cfg_device
    logger.info("stage=resolve_device requested=%s", requested_device)
    device = resolve_device(requested_device, allow_cpu=bool(args.dry_run))
    if device.type == "cuda":
        logger.info("device=cuda name=%s", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        logger.info("device=cpu")
    logger.info("stage=resolve_device done")

    if args.dry_run:
        logger.info("stage=dry_run_preflight start")
        _ = SpikeEmbeddingModel(build_model_config(model_cfg_raw)).to(device)
        logger.info(
            "dry_run_preflight_ok mearec_files=%d cache_state=%s cache_reason=%s total_epochs=%d stage1_gate_epochs=%d stage2_encoder_epochs=%d",
            len(mearec_h5),
            str(cache_state),
            str(cache_reason),
            total_epochs,
            stage1_epochs,
            stage2_epochs,
        )
        logger.info("stage=dry_run_preflight done")
        return

    uid_preflight_enabled = bool(data_cfg.get("uid_consistency_preflight_enabled", True))
    if mearec_h5 and uid_preflight_enabled:
        run_uid_consistency_preflight(
            mearec_h5,
            logger=logger,
            seed=seed + 1000,
            enabled=True,
            left=int(data_cfg.get("uid_consistency_left", 20)),
            right=int(data_cfg.get("uid_consistency_right", 44)),
            max_spikes_per_uid=int(data_cfg.get("uid_consistency_max_spikes_per_uid", 200)),
            time_atol_sec=float(data_cfg.get("uid_consistency_time_atol_sec", 1e-7)),
            wave_corr_thresh=float(data_cfg.get("uid_consistency_wave_corr_thresh", 0.8)),
        )
    elif mearec_h5:
        logger.info("uid_preflight skipped enabled=%s", str(uid_preflight_enabled))

    if use_cached_data:
        assert patches_path is not None
        assert labels_path is not None
        logger.info("stage=load_cached_data start patches=%s labels=%s", patches_path, labels_path)
        patches = np.load(patches_path).astype(np.float32)
        labels = load_npz(labels_path)
        logger.info("stage=load_cached_data done")
    else:
        logger.info("stage=load_mearec_data start")
        logger.info("stage=load_mearec_data using_effective_operational_config=true")
        for i, p in enumerate(mearec_h5):
            logger.info("mearec_file[%d]=%s exists=%s", i, p, str(Path(p).exists()))
        patches, labels = load_mearec_pretrain_data(
            h5_paths=mearec_h5,
            progress_cb=logger.info,
            **loader_kwargs,
        )
        labels = ensure_global_unit_id_labels(labels)
        validate_pretrain_array_compatibility(
            patches=patches,
            labels=labels,
            data_peak_only=bool(data_cfg.get("peak_only", False)),
            encoder_route=resolved_encoder_route,
            context="generated",
        )
        if auto_cache_enabled:
            assert patches_path is not None
            assert labels_path is not None
            save_optional_patches(patches_path, patches)
            save_npz(labels_path, labels)
            manifest_path = pretrain_cache_manifest_path(labels_path)
            write_pretrain_cache_manifest(
                manifest_path,
                build_pretrain_cache_manifest(
                    mearec_h5_paths=mearec_h5,
                    loader_kwargs=loader_kwargs,
                    patches_path=patches_path,
                    labels_path=labels_path,
                    patches=patches,
                    labels=labels,
                    root=ROOT,
                ),
            )
            logger.info(
                "cache_write_done patches=%s labels=%s manifest=%s samples=%d",
                patches_path,
                labels_path,
                manifest_path,
                int(patches.shape[0]),
            )
        logger.info("stage=load_mearec_data done")

    labels = ensure_global_unit_id_labels(labels)
    validate_pretrain_array_compatibility(
        patches=patches,
        labels=labels,
        data_peak_only=bool(data_cfg.get("peak_only", False)),
        encoder_route=resolved_encoder_route,
        context="cached" if use_cached_data else "generated",
    )
    unit_id = labels["unit_id"].astype(np.int64)
    local_unit_id = labels["local_unit_id"].astype(np.int64)
    global_unit_id = labels["global_unit_id"].astype(np.int64)
    geo_all = labels.get("geo")
    rel_pos_all = labels.get("rel_pos")
    channel_mask_all = labels.get("channel_mask")
    spike_label = labels.get("spike_label")
    spatial_stats_all = resolve_label_spatial_stats(
        labels,
        patches,
        logger=logger,
        context="cached" if use_cached_data else "mearec",
    )
    sample_source = labels.get("sample_source")
    center_channel = labels.get("center_channel")
    event_time_sample = labels.get("event_time_sample")
    peak_channel = labels.get("peak_channel")
    recording_id = labels.get("recording_id")
    training_mask = labels.get("training_mask")
    eval_mask = labels.get("eval_mask")
    domain_type = labels.get("domain_type")
    n_channels_label = labels.get("n_channels")
    strict_overlap_label = labels.get("strict_overlap_label")
    overlap_label = labels.get("overlap_label")
    qc_allowed = labels.get("qc_allowed")
    strict_matching_ambiguous = labels.get("strict_matching_ambiguous")
    relaxed_matching_ambiguous = labels.get("relaxed_matching_ambiguous")
    snr_peak = labels.get("snr_peak")
    event_amp_norm = labels.get("event_amp_norm")
    event_spatial_score = labels.get("event_spatial_score")
    event_simul_local = labels.get("event_simul_local")
    event_simul_global = labels.get("event_simul_global")
    valid = unit_id >= 0
    if spike_label is None:
        spike_label = valid.astype(np.float32)
    else:
        spike_label = np.asarray(spike_label, dtype=np.float32)
    if geo_all is None:
        geo_all = np.zeros((patches.shape[0], patches.shape[2], 4), dtype=np.float32)
    else:
        geo_all = np.asarray(geo_all, dtype=np.float32)
    if rel_pos_all is not None:
        rel_pos_arr = np.asarray(rel_pos_all, dtype=np.float32)
        if rel_pos_arr.shape[:2] == geo_all.shape[:2] and rel_pos_arr.shape[2] >= 2:
            geo_all = geo_all.copy()
            geo_all[:, :, :2] = rel_pos_arr[:, :, :2]
    if channel_mask_all is None:
        channel_mask_all = np.ones((patches.shape[0], patches.shape[2]), dtype=np.bool_)
    else:
        channel_mask_all = np.asarray(channel_mask_all, dtype=np.bool_)
    if sample_source is not None:
        sample_source = np.asarray(sample_source, dtype=np.int8)
    if center_channel is None:
        center_channel = np.zeros((patches.shape[0],), dtype=np.int32)
        if use_cached_data:
            logger.warning("cached labels missing center_channel; channel regularization will use zeros")
    else:
        center_channel = np.asarray(center_channel, dtype=np.int32)
    if event_time_sample is None:
        event_time_sample = np.arange(patches.shape[0], dtype=np.int64)
        logger.warning("labels missing event_time_sample; temporal Stage2 will use sample order fallback")
    else:
        event_time_sample = np.asarray(event_time_sample, dtype=np.int64)
    if peak_channel is None:
        peak_channel = center_channel.astype(np.int32, copy=False)
        logger.warning("labels missing peak_channel; falling back to center_channel")
    else:
        peak_channel = np.asarray(peak_channel, dtype=np.int32)
    if recording_id is None:
        recording_id = np.zeros((patches.shape[0],), dtype=np.int16)
    else:
        recording_id = np.asarray(recording_id, dtype=np.int16)
    if training_mask is None:
        training_mask = np.ones((patches.shape[0],), dtype=np.bool_)
        logger.warning("labels missing training_mask; defaulting to all true")
    else:
        training_mask = np.asarray(training_mask, dtype=np.bool_)
    if eval_mask is None:
        eval_mask = training_mask.astype(np.bool_, copy=True)
        logger.warning("labels missing eval_mask; defaulting to training_mask")
    else:
        eval_mask = np.asarray(eval_mask, dtype=np.bool_)
    if n_channels_label is None:
        fallback_n_channels = 1 if bool(data_cfg.get("peak_only", False)) else int(patches.shape[2])
        n_channels_label = np.full((patches.shape[0],), fallback_n_channels, dtype=np.int32)
        logger.warning("labels missing n_channels; deriving domain_type from fallback channel count")
    else:
        n_channels_label = np.asarray(n_channels_label, dtype=np.int32)
    if domain_type is None:
        domain_type = np.where(np.asarray(n_channels_label, dtype=np.int32) <= 1, "single_channel", "multi_channel")
        logger.warning("labels missing domain_type; deriving from n_channels")
    else:
        domain_type = np.asarray(domain_type).astype(str)
    if not np.array_equal(unit_id, global_unit_id):
        raise RuntimeError("pretrain labels must use global_unit_id as unit_id")
    multi_pair_global_units = 0
    positive_label = unit_id >= 0
    if np.any(positive_label):
        pairs = np.stack([recording_id[positive_label].astype(np.int64), local_unit_id[positive_label]], axis=1)
        global_values = global_unit_id[positive_label]
        for gid in np.unique(global_values).tolist():
            pair_for_gid = np.unique(pairs[global_values == int(gid)], axis=0)
            if pair_for_gid.shape[0] != 1:
                multi_pair_global_units += 1
    logger.info(
        "unit_label_scope global_units=%d local_unit_values=%d recordings=%d multi_pair_global_units=%d",
        int(np.unique(global_unit_id[global_unit_id >= 0]).size),
        int(np.unique(local_unit_id[local_unit_id >= 0]).size),
        int(np.unique(recording_id).size),
        int(multi_pair_global_units),
    )
    if strict_overlap_label is None:
        strict_overlap_label = np.zeros((patches.shape[0],), dtype=np.bool_)
        logger.warning("labels missing strict_overlap_label; defaulting to all false")
    else:
        strict_overlap_label = np.asarray(strict_overlap_label, dtype=np.bool_)
    if overlap_label is None:
        overlap_label = strict_overlap_label.astype(np.float32, copy=False)
    else:
        overlap_label = np.maximum(np.asarray(overlap_label, dtype=np.float32), strict_overlap_label.astype(np.float32, copy=False))
    if qc_allowed is None:
        qc_allowed = np.ones((patches.shape[0],), dtype=np.bool_)
        logger.warning("labels missing qc_allowed; defaulting to all true")
    else:
        qc_allowed = np.asarray(qc_allowed, dtype=np.bool_)
    if strict_matching_ambiguous is None:
        strict_matching_ambiguous = np.zeros((patches.shape[0],), dtype=np.bool_)
    else:
        strict_matching_ambiguous = np.asarray(strict_matching_ambiguous, dtype=np.bool_)
    if relaxed_matching_ambiguous is None:
        relaxed_matching_ambiguous = np.zeros((patches.shape[0],), dtype=np.bool_)
    else:
        relaxed_matching_ambiguous = np.asarray(relaxed_matching_ambiguous, dtype=np.bool_)
    if snr_peak is not None:
        snr_peak = np.asarray(snr_peak, dtype=np.float32)
    event_amp_norm = (
        np.asarray(event_amp_norm, dtype=np.float32)
        if event_amp_norm is not None
        else np.zeros((patches.shape[0],), dtype=np.float32)
    )
    event_spatial_score = (
        np.asarray(event_spatial_score, dtype=np.float32)
        if event_spatial_score is not None
        else np.ones((patches.shape[0],), dtype=np.float32)
    )
    event_simul_local = (
        np.asarray(event_simul_local, dtype=np.int32)
        if event_simul_local is not None
        else np.zeros((patches.shape[0],), dtype=np.int32)
    )
    event_simul_global = (
        np.asarray(event_simul_global, dtype=np.int32)
        if event_simul_global is not None
        else np.zeros((patches.shape[0],), dtype=np.int32)
    )
    if use_cached_data:
        logger.info(
            "loaded_cached patches=%d time=%d channels=%d classes=%d",
            int(patches.shape[0]),
            int(patches.shape[1]),
            int(patches.shape[2]),
            int(np.unique(unit_id).size),
        )
    else:
        logger.info(
            "loaded_mearec files=%d samples=%d time=%d channels=%d classes=%d",
            len(mearec_h5),
            int(patches.shape[0]),
            int(patches.shape[1]),
            int(patches.shape[2]),
            int(np.unique(unit_id).size),
        )
    source_counts = summarize_sample_sources(sample_source)
    logger.info(
        "sample_source_counts total_primary_matched_positive=%d total_secondary_matched_positive=%d total_detector_negative=%d total_unknown=%d",
        int(source_counts["primary_matched_positive"]),
        int(source_counts["secondary_matched_positive"]),
        int(source_counts["detector_negative"]),
        int(source_counts["unknown"]),
    )
    logger.info(
        "gate_spatial_stats shape=%s source=%s",
        str(tuple(int(v) for v in spatial_stats_all.shape)),
        "labels" if "spatial_stats" in labels else "derived_from_patches",
    )

    val_enabled = bool(val_cfg.get("enabled", True))
    val_ratio = float(val_cfg.get("split_ratio", 0.1))
    val_min_per_class = int(val_cfg.get("min_per_class", 1))
    val_seed = int(val_cfg.get("seed", seed + 1))
    val_split_mode = str(
        val_cfg.get("split_mode", "recording_holdout" if stage2_epochs > 0 else "unit_stratified")
    ).lower()
    val_recording_ratio = float(val_cfg.get("val_recording_ratio", 0.2))
    val_min_val_recordings = int(val_cfg.get("min_val_recordings", 1))
    val_min_train_recordings = int(val_cfg.get("min_train_recordings", 1))
    val_min_holdout_units = int(val_cfg.get("min_holdout_units", 2))
    val_min_train_samples_per_unit = int(val_cfg.get("min_train_samples_per_unit", max(1, val_min_per_class)))
    val_min_val_samples_per_unit = int(val_cfg.get("min_val_samples_per_unit", max(1, val_min_per_class)))
    val_every_epochs = max(1, int(val_cfg.get("every_epochs", 1)))
    val_support_per_class = int(val_cfg.get("support_per_class", 32))
    val_max_eval_samples = int(val_cfg.get("max_eval_samples", 20000))
    val_pair_samples = int(val_cfg.get("pair_samples", 10000))
    val_batch_size = int(val_cfg.get("batch_size", 1024))
    gate_fixed_threshold = float(val_cfg.get("gate_fixed_threshold", OPERATIONAL_FIXED_GATE_THRESHOLD))
    if not math.isclose(float(gate_fixed_threshold), OPERATIONAL_FIXED_GATE_THRESHOLD, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("train.validation.gate_fixed_threshold must remain 0.5")
    gate_target_recall_encode = float(val_cfg.get("gate_target_recall_encode", val_cfg.get("gate_target_recall", 0.995)))
    gate_target_recall_assignment_min = float(
        val_cfg.get("gate_target_recall_assignment_min", val_cfg.get("gate_target_recall_assignment", 0.90))
    )
    gate_warmup_epochs = int(val_cfg.get("gate_warmup_epochs", 3))
    gate_patience = int(val_cfg.get("gate_patience", 3))
    gate_enable_early_stop = bool(val_cfg.get("gate_enable_early_stop", False))
    gate_early_stop_enabled = bool(gate_enable_early_stop and gate_patience > 0)
    gate_goal_spike_recall = float(val_cfg.get("gate_goal_spike_recall", 0.95))
    gate_goal_noise_accuracy = float(val_cfg.get("gate_goal_noise_accuracy", 0.95))
    for deprecated_key in (
        "gate_metric_style",
        "gate_validation_style",
        "gate_alpha4_calibration",
        "gate_alpha4_thresholds",
        "calibration",
        "thresholds",
    ):
        if deprecated_key in val_cfg:
            logger.warning(
                "deprecated train.validation.%s ignored; operational gate validation uses fixed_threshold=0.5",
                deprecated_key,
    )
    val_cluster_cfg = dict(val_cfg.get("cluster", {}) or {})
    val_cluster_enabled = bool(val_cluster_cfg.get("enabled", val_cfg.get("cluster_enabled", True)))
    val_cluster_min_samples_per_unit = int(
        val_cluster_cfg.get("min_samples_per_unit", val_cfg.get("cluster_min_samples_per_unit", 8))
    )
    val_cluster_max_samples_per_unit = int(
        val_cluster_cfg.get("max_samples_per_unit", val_cfg.get("cluster_max_samples_per_unit", 128))
    )
    val_cluster_max_samples_per_recording = int(
        val_cluster_cfg.get("max_samples_per_recording", val_cfg.get("cluster_max_samples_per_recording", 4096))
    )
    logger.info(
        "validation_embedding enabled=%s min_samples_per_unit=%d max_samples_per_unit=%d max_samples_per_recording=%d",
        str(val_cluster_enabled),
        int(val_cluster_min_samples_per_unit),
        int(val_cluster_max_samples_per_unit),
        int(val_cluster_max_samples_per_recording),
    )
    logger.info(
        "gate_validation metric_style=operational fixed_threshold=%.4f goal_spike_recall=%.4f goal_noise_accuracy=%.4f",
        float(gate_fixed_threshold),
        float(gate_goal_spike_recall),
        float(gate_goal_noise_accuracy),
    )

    train_idx = np.arange(unit_id.shape[0], dtype=np.int64)
    val_idx = np.zeros((0,), dtype=np.int64)
    val_recording_split = val_split_mode in ("recording_holdout", "recording", "holdout_recording")
    unit_stratified_split = val_split_mode in ("unit_stratified", "stratified", "unit")
    if stage2_epochs > 0 and unit_stratified_split:
        raise ValueError(
            "Stage2 validation requires train.validation.split_mode=recording_holdout; "
            "unit_stratified does not validate unseen recordings"
        )
    if not val_recording_split and not unit_stratified_split:
        raise ValueError("train.validation.split_mode must be one of {recording_holdout, unit_stratified}")
    split_requested = (val_recording_split and val_recording_ratio > 0.0) or (unit_stratified_split and val_ratio > 0.0)
    if val_enabled and split_requested and unit_id.shape[0] > 1:
        if val_recording_split:
            split_candidate_mask = build_recording_holdout_candidate_mask(
                unit_id,
                spike_label,
                sample_source,
                strict_overlap_label=strict_overlap_label,
                overlap_label=overlap_label,
                qc_allowed=qc_allowed,
                strict_matching_ambiguous=strict_matching_ambiguous,
                relaxed_matching_ambiguous=relaxed_matching_ambiguous,
            )
            train_idx, val_idx, split_summary = recording_holdout_train_val_split(
                unit_id,
                recording_id,
                split_candidate_mask,
                val_recording_ratio=val_recording_ratio,
                seed=val_seed,
                min_val_recordings=val_min_val_recordings,
                min_train_recordings=val_min_train_recordings,
                min_holdout_units=val_min_holdout_units,
                min_train_samples_per_unit=val_min_train_samples_per_unit,
                min_val_samples_per_unit=val_min_val_samples_per_unit,
            )
            if training_manifest and training_manifest.get("validation_indices") is not None:
                expected = list(training_manifest["validation_indices"])
                if list(split_summary["val_recordings"]) != expected:
                    raise ValueError(f"training split differs from paper manifest: expected {expected}, got {split_summary['val_recordings']}")
            logger.info(
                "validation_split mode=recording_holdout val_recording_ratio=%.4f train_recordings=%s val_recordings=%s eligible_val_recordings=%s candidate_spikes=%d valid_units=%s",
                val_recording_ratio,
                ",".join(str(v) for v in split_summary["train_recordings"]),
                ",".join(str(v) for v in split_summary["val_recordings"]),
                ",".join(str(v) for v in split_summary["eligible_val_recordings"]),
                int(np.sum(split_candidate_mask)),
                str(split_summary["valid_units_by_val_recording"]),
            )
            if mearec_h5:
                rec_path_map = recording_id_path_map(mearec_h5)
                logger.info(
                    "validation_split_paths train=%s val=%s",
                    format_recording_id_path_map(rec_path_map, split_summary["train_recordings"]),
                    format_recording_id_path_map(rec_path_map, split_summary["val_recordings"]),
                )
                validate_recording_holdout_validation_recordings(split_summary["val_recordings"], rec_path_map)
        elif unit_stratified_split:
            train_idx, val_idx = stratified_train_val_split(
                unit_id,
                val_ratio=val_ratio,
                seed=val_seed,
                min_val_per_class=val_min_per_class,
            )
            logger.info(
                "validation_split mode=unit_stratified split_ratio=%.4f min_val_per_class=%d",
                val_ratio,
                val_min_per_class,
            )
        else:
            raise ValueError("train.validation.split_mode must be one of {recording_holdout, unit_stratified}")
    train_patches = patches[train_idx]
    train_geo_all = geo_all[train_idx]
    train_channel_mask_all = channel_mask_all[train_idx]
    train_spatial_stats = spatial_stats_all[train_idx]
    train_unit_id = unit_id[train_idx]
    train_spike_label = spike_label[train_idx]
    train_sample_source = sample_source[train_idx] if sample_source is not None else None
    train_center_channel = center_channel[train_idx]
    train_event_time_sample = event_time_sample[train_idx]
    train_peak_channel = peak_channel[train_idx]
    train_recording_id = recording_id[train_idx]
    train_training_mask = training_mask[train_idx]
    train_eval_mask = eval_mask[train_idx]
    train_domain_type = domain_type[train_idx]
    train_strict_overlap_label = strict_overlap_label[train_idx]
    train_overlap_label = overlap_label[train_idx]
    train_qc_allowed = qc_allowed[train_idx]
    train_strict_matching_ambiguous = strict_matching_ambiguous[train_idx]
    train_relaxed_matching_ambiguous = relaxed_matching_ambiguous[train_idx]
    train_snr_peak = snr_peak[train_idx] if snr_peak is not None else None
    train_event_amp_norm = event_amp_norm[train_idx]
    train_event_spatial_score = event_spatial_score[train_idx]
    train_event_simul_local = event_simul_local[train_idx]
    train_event_simul_global = event_simul_global[train_idx]
    if val_idx.size > 0:
        val_patches = patches[val_idx]
        val_geo_all = geo_all[val_idx]
        val_channel_mask_all = channel_mask_all[val_idx]
        val_spatial_stats = spatial_stats_all[val_idx]
        val_unit_id = unit_id[val_idx]
        val_spike_label = spike_label[val_idx]
        val_sample_source = sample_source[val_idx] if sample_source is not None else None
        val_center_channel = center_channel[val_idx]
        val_event_time_sample = event_time_sample[val_idx]
        val_peak_channel = peak_channel[val_idx]
        val_recording_id = recording_id[val_idx]
        val_training_mask = training_mask[val_idx]
        val_eval_mask = eval_mask[val_idx]
        val_domain_type = domain_type[val_idx]
        val_strict_overlap_label = strict_overlap_label[val_idx]
        val_overlap_label = overlap_label[val_idx]
        val_qc_allowed = qc_allowed[val_idx]
        val_strict_matching_ambiguous = strict_matching_ambiguous[val_idx]
        val_relaxed_matching_ambiguous = relaxed_matching_ambiguous[val_idx]
        val_snr_peak = snr_peak[val_idx] if snr_peak is not None else None
        val_event_amp_norm = event_amp_norm[val_idx]
        val_event_spatial_score = event_spatial_score[val_idx]
        val_event_simul_local = event_simul_local[val_idx]
        val_event_simul_global = event_simul_global[val_idx]
    else:
        val_patches = np.zeros((0,) + patches.shape[1:], dtype=np.float32)
        val_geo_all = np.zeros((0,) + geo_all.shape[1:], dtype=np.float32)
        val_channel_mask_all = np.zeros((0, channel_mask_all.shape[1]), dtype=np.bool_)
        val_spatial_stats = np.zeros((0, spatial_stats_all.shape[1]), dtype=np.float32)
        val_unit_id = np.zeros((0,), dtype=np.int64)
        val_spike_label = np.zeros((0,), dtype=np.float32)
        val_sample_source = None
        val_center_channel = np.zeros((0,), dtype=np.int32)
        val_event_time_sample = np.zeros((0,), dtype=np.int64)
        val_peak_channel = np.zeros((0,), dtype=np.int32)
        val_recording_id = np.zeros((0,), dtype=np.int16)
        val_training_mask = np.zeros((0,), dtype=np.bool_)
        val_eval_mask = np.zeros((0,), dtype=np.bool_)
        val_domain_type = np.zeros((0,), dtype=str)
        val_strict_overlap_label = np.zeros((0,), dtype=np.bool_)
        val_overlap_label = np.zeros((0,), dtype=np.float32)
        val_qc_allowed = np.zeros((0,), dtype=np.bool_)
        val_strict_matching_ambiguous = np.zeros((0,), dtype=np.bool_)
        val_relaxed_matching_ambiguous = np.zeros((0,), dtype=np.bool_)
        val_snr_peak = None
        val_event_amp_norm = np.zeros((0,), dtype=np.float32)
        val_event_spatial_score = np.zeros((0,), dtype=np.float32)
        val_event_simul_local = np.zeros((0,), dtype=np.int32)
        val_event_simul_global = np.zeros((0,), dtype=np.int32)

    logger.info(
        "data_split train_samples=%d train_classes=%d val_samples=%d val_classes=%d val_enabled=%s val_ratio=%.4f",
        int(train_patches.shape[0]),
        int(np.unique(train_unit_id).size) if train_unit_id.size > 0 else 0,
        int(val_patches.shape[0]),
        int(np.unique(val_unit_id).size) if val_unit_id.size > 0 else 0,
        str(bool(val_enabled and val_idx.size > 0)),
        float(val_ratio),
    )
    train_pos_labels, train_neg_labels = _binary_label_counts(train_spike_label)
    val_pos_labels, val_neg_labels = _binary_label_counts(val_spike_label)
    logger.info(
        "spike_label_split train_pos=%d train_neg=%d val_pos=%d val_neg=%d",
        int(train_pos_labels),
        int(train_neg_labels),
        int(val_pos_labels),
        int(val_neg_labels),
    )
    train_source_counts = summarize_sample_sources(train_sample_source)
    val_source_counts = summarize_sample_sources(val_sample_source)
    train_gt_mask = (train_spike_label > 0.5) & (
        (train_sample_source == 0) if train_sample_source is not None else np.ones(train_spike_label.shape, dtype=bool)
    )
    logger.info(
        "sample_source_split train_primary_matched_positive=%d train_secondary_matched_positive=%d train_detector_negative=%d "
        "val_primary_matched_positive=%d val_secondary_matched_positive=%d val_detector_negative=%d",
        int(train_source_counts["primary_matched_positive"]),
        int(train_source_counts["secondary_matched_positive"]),
        int(train_source_counts["detector_negative"]),
        int(val_source_counts["primary_matched_positive"]),
        int(val_source_counts["secondary_matched_positive"]),
        int(val_source_counts["detector_negative"]),
    )
    logger.info(
        "snr_split train %s | val %s",
        format_snr_summary("train_snr", train_snr_peak),
        format_snr_summary("val_snr", val_snr_peak),
    )
    if stage1_epochs > 0 and train_neg_labels <= 0:
        raise RuntimeError("stage-1 gate training requires detector negative samples, but training negatives=0")
    if val_enabled and val_patches.shape[0] > 0:
        val_binary = np.unique((val_spike_label > 0.5).astype(np.int8, copy=False))
        if val_binary.size < 2:
            raise RuntimeError("validation spike_label must contain both positive and negative samples")

    sampler_cfg = train_cfg.get("sampler", {})
    sampler_mode = str(sampler_cfg.get("mode", "random")).lower()
    sampler_max_n_way = int(sampler_cfg.get("max_n_way", 0))
    n_per_class = int(sampler_cfg.get("n_per_class", 0))
    k_shot = int(sampler_cfg.get("k_shot", 0))
    episodic_proto_ce = sampler_mode in ("balanced", "class_balanced")
    stage2_min_domains = int(stage2_cfg.get("min_domains_per_unit", 1))
    stage2_data_filter = str(stage2_cfg.get("data_filter", "frozen_encode_gate")).lower()
    stage2_reset_optimizer_on_entry = bool(stage2_cfg.get("reset_optimizer_on_entry", True))
    stage2_reset_scheduler_warmup_epochs = int(max(0, stage2_cfg.get("reset_scheduler_warmup_epochs", 2)))
    stage1_encoder_aux_cfg = dict(stage1_cfg.get("encoder_aux", {}) or {})
    stage1_encoder_aux_requested = bool(stage1_encoder_aux_cfg.get("enabled", False))
    stage1_encoder_aux_enabled = False
    raw_stage2_min_samples = stage2_cfg.get("min_samples_per_unit_after_gate")
    if raw_stage2_min_samples is None:
        stage2_min_samples_after_gate = int(max(1, n_per_class)) if episodic_proto_ce and n_per_class > 0 else 2
    else:
        stage2_min_samples_after_gate = int(max(1, int(raw_stage2_min_samples)))
    full_support = build_unit_recording_support(
        unit_id,
        recording_id,
        sample_source=sample_source,
        spike_label=spike_label,
        primary_only=True,
    )
    full_support_summary = summarize_unit_domain_support(full_support, min_domains_required=stage2_min_domains)
    logger.info(
        "stage2_domain_support split=all gt_units=%d eligible_units=%d mean_domains=%.4f max_domains=%d",
        int(full_support_summary["n_units"]),
        int(full_support_summary["eligible_units"]),
        float(full_support_summary["mean_domains"]),
        int(full_support_summary["max_domains"]),
    )
    train_stage2_reference_class_indices, train_stage2_reference_positive_idx, train_stage2_reference_support = resolve_stage2_gt_class_indices(
        train_unit_id,
        train_recording_id,
        train_spike_label,
        train_sample_source,
        min_domains_per_unit=stage2_min_domains,
    )
    train_stage2_reference_allowed_units = np.asarray(sorted(train_stage2_reference_class_indices.keys()), dtype=np.int64)
    train_stage2_reference_support_summary = summarize_unit_domain_support(
        train_stage2_reference_support,
        min_domains_required=stage2_min_domains,
    )
    logger.info(
        "stage2_domain_support split=train_reference gt_units=%d eligible_units=%d mean_domains=%.4f max_domains=%d eligible_samples=%d",
        int(train_stage2_reference_support_summary["n_units"]),
        int(train_stage2_reference_support_summary["eligible_units"]),
        float(train_stage2_reference_support_summary["mean_domains"]),
        int(train_stage2_reference_support_summary["max_domains"]),
        int(train_stage2_reference_positive_idx.shape[0]),
    )
    logger.info(
        "stage2_reference_gt_counts train_gt=%d reference_gt=%d filtered_gt=%d",
        int(np.sum(train_gt_mask)),
        int(train_stage2_reference_positive_idx.shape[0]),
        int(np.sum(train_gt_mask) - train_stage2_reference_positive_idx.shape[0]),
    )
    if stage2_epochs > 0:
        stage2_source_mode = str(stage2_cfg.get("source_mode", "matched_positives")).lower()
        if stage2_source_mode not in {"primary_only", "matched_positives"}:
            raise ValueError(
                "stage2_encoder.source_mode must be one of {primary_only, matched_positives}"
            )
        if str(stage2_cfg.get("domain_sampling", "round_robin")).lower() != "round_robin":
            raise ValueError("stage2_encoder.domain_sampling currently only supports round_robin")
        if bool(stage2_cfg.get("use_encode_gate", False)):
            raise ValueError("stage2_encoder.use_encode_gate must remain false in the two-stage setup")
        if stage2_data_filter != "frozen_encode_gate":
            raise ValueError("stage2_encoder.data_filter currently only supports frozen_encode_gate")
        if train_stage2_reference_allowed_units.size == 0 or train_stage2_reference_positive_idx.size == 0:
            raise RuntimeError(
                "stage-2 encoder training requires GT-only units that appear in at least "
                f"{stage2_min_domains} recordings, but none were found"
            )

    loss_cfg = cfg.get("loss", {})
    stage2_losses_preview = dict(cfg.get("stage2", {}).get("losses", {}) or {})
    graph_cfg: Dict[str, Any] = {}
    graph_enabled = False

    logger.info("stage=build_model start")
    model = SpikeEmbeddingModel(build_model_config(model_cfg_raw)).to(device)
    model.train()
    logger.info(
        "model_encoder route=%s embed_dim=%d hidden_dim=%d n_blocks=%d top_k_channels=%d peak_only=%s",
        str(model.encoder_net.encoder_route),
        int(model.cfg.embed_dim),
        int(model.cfg.hidden_dim),
        int(model.cfg.n_blocks),
        int(patches.shape[2]),
        str(bool(data_cfg.get("peak_only", False))),
    )
    logger.info("stage2_loss_config %s", _compact_yaml(stage2_losses_preview))
    logger.info("validation_cluster_config %s", _compact_yaml(val_cluster_cfg))
    electrode_cfg_raw = loader_kwargs.get("electrode_config") if loader_kwargs else data_cfg.get("electrode_config")
    electrode_cfg_path = resolve_path(electrode_cfg_raw, ROOT) if electrode_cfg_raw else None
    electrode_cfg_name = None
    if electrode_cfg_path is not None and electrode_cfg_path.exists():
        try:
            electrode_cfg_name = str(load_yaml(electrode_cfg_path).get("name", electrode_cfg_path.stem))
        except Exception:
            electrode_cfg_name = electrode_cfg_path.stem
    encoder_checkpoint_metadata: Dict[str, Any] = {
        "recording_as_task": True,
        "encoder_route": str(model.encoder_net.encoder_route),
        "embed_dim": int(model.cfg.embed_dim),
        "train_patch_k": int(patches.shape[2]),
        "normalization_mode": str(loader_kwargs.get("normalization_mode", data_cfg.get("normalization_mode", "peak_channel_sigma"))),
        "no_support_inference": True,
        "requires_dense_electrode": bool(model.encoder_net.encoder_route != "peak_only" or int(patches.shape[2]) > 1),
        "electrode_config_name": electrode_cfg_name,
        "electrode_config_hash": electrode_config_hash(electrode_cfg_path),
        "model_config": {
            "encoder_route": str(model.encoder_net.encoder_route),
            "embed_dim": int(model.cfg.embed_dim),
            "hidden_dim": int(model.cfg.hidden_dim),
            "n_blocks": int(model.cfg.n_blocks),
            "shared_stem_layers": int(model.cfg.shared_stem_layers),
            "footprint_stats_dim": int(model.cfg.footprint_stats_dim),
        },
        "stage2_loss_config": stage2_losses_preview,
        "validation_cluster_config": dict(val_cluster_cfg),
    }
    logger.info(
        "encoder_checkpoint_metadata route=%s train_patch_k=%d requires_dense=%s electrode_config=%s",
        encoder_checkpoint_metadata["encoder_route"],
        int(encoder_checkpoint_metadata["train_patch_k"]),
        str(encoder_checkpoint_metadata["requires_dense_electrode"]),
        str(encoder_checkpoint_metadata["electrode_config_name"]),
    )
    logger.info("stage=build_model done")
    fixed_gate_threshold = validate_fixed_gate_thresholds(
        encode_gate_thresh=float(model.cfg.encode_gate_thresh),
        assignment_gate_thresh=float(model.cfg.assignment_gate_thresh),
        fixed_gate_threshold=model.cfg.fixed_gate_threshold if gate_fixed_threshold is None else gate_fixed_threshold,
        context="pretrain",
    )
    teacher_model: SpikeEmbeddingModel | None = None

    bs = int(train_cfg.get("batch_size", 128))
    gate_bs_cfg = int(train_cfg.get("gate_batch_size", min(512, bs)))
    gate_bs = max(2, gate_bs_cfg)
    if gate_bs % 2 == 1:
        gate_bs -= 1
    stage1_steps_cfg = stage1_cfg.get("steps_per_epoch")
    if stage1_epochs > 0 and stage1_steps_cfg is not None:
        stage1_steps_per_epoch = int(max(1, int(stage1_steps_cfg)))
    else:
        stage1_steps_per_epoch = int(max(1, math.ceil(max(1, train_patches.shape[0]) / max(1, gate_bs)))) if stage1_epochs > 0 else 0
    stage2_steps_per_epoch = (
        int(max(1, math.ceil(max(1, train_stage2_reference_positive_idx.shape[0]) / max(1, bs)))) if stage2_epochs > 0 else 0
    )
    stage1_total_train_steps = int(max(1, stage1_epochs * max(1, stage1_steps_per_epoch))) if stage1_epochs > 0 else 0
    stage2_reference_total_train_steps = int(max(1, stage2_schedule_epochs * max(1, stage2_steps_per_epoch))) if stage2_epochs > 0 else 0
    total_train_steps = int(max(1, (stage1_epochs * stage1_steps_per_epoch) + (stage2_epochs * stage2_steps_per_epoch)))

    base_lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    scheduler_cfg = train_cfg.get("scheduler", {})
    initial_scheduler_total_steps = total_train_steps
    initial_scheduler_steps_per_epoch = max(stage1_steps_per_epoch, stage2_steps_per_epoch, 1)
    if stage2_epochs > 0 and stage2_reset_optimizer_on_entry and stage1_epochs > 0:
        initial_scheduler_total_steps = int(max(1, stage1_total_train_steps))
        initial_scheduler_steps_per_epoch = max(1, stage1_steps_per_epoch)

    logger.info("stage=build_optimizer start")
    opt = build_optimizer_for_parameters(
        list(model.parameters()),
        lr=base_lr,
        weight_decay=weight_decay,
    )
    logger.info("stage=build_optimizer done")

    scheduler, scheduler_info = build_scheduler_from_config(
        opt,
        scheduler_cfg=scheduler_cfg,
        base_lr=base_lr,
        total_steps=initial_scheduler_total_steps,
        steps_per_epoch=initial_scheduler_steps_per_epoch,
    )
    scheduler_type = str(scheduler_info.get("type", "none"))
    if scheduler is not None:
        logger.info(
            "scheduler type=%s warmup_epochs=%d warmup_steps=%d total_steps=%d min_lr=%.6g scope=%s",
            scheduler_type,
            int(scheduler_info.get("warmup_epochs", 0)),
            int(scheduler_info.get("warmup_steps", 0)),
            int(scheduler_info.get("total_steps", 0)),
            float(scheduler_info.get("min_lr", 0.0)),
            "stage1_only" if (stage2_epochs > 0 and stage2_reset_optimizer_on_entry and stage1_epochs > 0) else "full_train",
        )
    else:
        logger.info("scheduler type=none")

    aug_cfg = train_cfg.get("contrastive_views", {})
    contrastive_views_enabled = bool(aug_cfg.get("enabled", False))
    contrastive_noise_std = float(aug_cfg.get("noise_std", 0.01))
    contrastive_gain_jitter = float(aug_cfg.get("gain_jitter", 0.05))
    logger.info(
        "contrastive_views enabled=%s noise_std=%.6g gain_jitter=%.6g",
        str(contrastive_views_enabled),
        contrastive_noise_std,
        contrastive_gain_jitter,
    )

    loss_cfg = cfg.get("loss", {})
    supcon_tau = float(loss_cfg.get("supcon_tau", 0.1))
    proto_gamma = float(loss_cfg.get("proto_gamma", 20.0))
    lambda_supcon = float(loss_cfg.get("lambda_supcon", 1.0))
    lambda_proto = float(loss_cfg.get("lambda_proto", 1.0))
    lambda_gate = float(loss_cfg.get("lambda_gate", 1.0))
    lambda_channel = float(loss_cfg.get("lambda_channel", 0.1))
    if "graph" in loss_cfg and "lambda_channel" not in loss_cfg:
        lambda_channel = 0.0
    channel_bias_beta = float(loss_cfg.get("channel_bias_beta", 0.2))
    proto_tau = float(loss_cfg.get("proto_tau", train_cfg.get("proto_tau", 0.1)))
    if proto_tau <= 0.0:
        raise ValueError(f"proto_tau must be > 0, got {proto_tau}")
    lambda_proto_ce = float(loss_cfg.get("lambda_proto_ce", train_cfg.get("lambda_proto_ce", 1.0)))
    stage2_losses_cfg = dict(cfg.get("stage2", {}).get("losses", {}) or {})
    _ = stage2_losses_cfg
    graph_cfg = {}
    graph_enabled = False
    graph_lambda_umap_cfg = 0.0
    stage2_global_cfg = dict(cfg.get("stage2", {}) or {})
    stage2_qc_cfg = dict(stage2_global_cfg.get("qc", {}) or {})
    stage2_episode_cfg = dict(stage2_global_cfg.get("episode", {}) or {})
    stage2_temporal_episode_enabled = bool(stage2_episode_cfg.get("enabled", False))
    if stage2_epochs > 0 and not stage2_temporal_episode_enabled:
        raise ValueError("stage-2 main path requires stage2.episode.enabled=true")

    gate_train_cfg = dict(train_cfg.get("gate", {}) or {})
    stage1_gate_selection_mode = str(stage1_cfg.get("selection_mode", "fixed_0_5_acceptance")).strip().lower()
    if stage1_gate_selection_mode in ("fixed_0.5_acceptance", "acceptance"):
        stage1_gate_selection_mode = "fixed_0_5_acceptance"
    stage1_acceptance_min_precision = float(stage1_cfg.get("acceptance_min_precision", 0.99))
    stage1_acceptance_min_recall = float(stage1_cfg.get("acceptance_min_recall", 0.90))
    gate_sampler_mode = canonical_gate_sampler_mode(gate_train_cfg.get("sampler_mode", "natural"), logger=logger)
    gate_loss_type = str(gate_train_cfg.get("loss_type", "bce")).lower()
    if gate_loss_type in ("binary_cross_entropy", "binary_cross_entropy_with_logits"):
        gate_loss_type = "bce"
    gate_use_global_pos_weight = bool(gate_train_cfg.get("use_global_pos_weight", False))
    gate_loss_weight_mode = canonical_gate_loss_weight_mode(
        gate_train_cfg.get("loss_weight_mode", "none"),
        logger=logger,
    )
    gate_log_source_metrics = bool(gate_train_cfg.get("log_source_metrics", False))
    legacy_gate_keys = (
        "balance_recordings",
        "curriculum_warmup_epochs",
        "batch_mix_warmup",
        "batch_mix_main",
        "source_weights",
        "focal_gamma",
        "overlap_positive_boost",
    )
    if stage1_epochs > 0:
        if stage1_gate_selection_mode != "fixed_0_5_acceptance":
            raise ValueError(
                "unsupported train.stage1_gate.selection_mode: "
                f"{stage1_gate_selection_mode!r}; expected fixed_0_5_acceptance"
            )
        legacy_present = [key for key in legacy_gate_keys if key in gate_train_cfg]
        if legacy_present:
            raise ValueError(
                "unsupported legacy train.gate keys for natural BCE gate training: "
                + ", ".join(sorted(legacy_present))
            )
        if gate_sampler_mode != "natural":
            raise ValueError(f"unsupported train.gate.sampler_mode: {gate_sampler_mode}; expected natural")
        if gate_loss_type != "bce":
            raise ValueError(f"unsupported train.gate.loss_type: {gate_loss_type}; expected bce")
        if gate_loss_weight_mode != "none":
            raise ValueError(f"unsupported train.gate.loss_weight_mode: {gate_loss_weight_mode}; expected none")
        if gate_use_global_pos_weight:
            raise ValueError("train.gate.use_global_pos_weight=true is unsupported for natural BCE gate training")

    n_pos_spike = int(np.sum(train_spike_label > 0.5))
    n_neg_spike = int(train_spike_label.shape[0] - n_pos_spike)
    gate_pos_weight_value = float(n_neg_spike) / max(1.0, float(n_pos_spike))
    gate_pos_weight_value *= float(model.cfg.gate_pos_weight_scale)
    gate_pos_weight = None
    gate_sampler = None
    train_stage1_reference_class_indices = dict(train_stage2_reference_class_indices)
    train_stage1_reference_positive_idx = np.asarray(train_stage2_reference_positive_idx, dtype=np.int64)
    train_stage1_reference_allowed_units = np.asarray(train_stage2_reference_allowed_units, dtype=np.int64)
    if stage1_epochs > 0:
        if n_neg_spike <= 0:
            raise RuntimeError("gate training requires negative spike samples, but n_neg_spike == 0")
        gate_sampler = build_gate_batch_sampler_natural(
            train_spike_label,
            training_mask=train_training_mask,
            batch_size=gate_bs,
            seed=seed + 97,
        )
    logger.info(
        "stage1_gate_setup epochs=%d positives=%d negatives=%d pos_weight=%.4f pos_weight_enabled=%s gate_batch_size=%d "
        "encode_gate_thresh=%.4f assignment_gate_thresh=%.4f lambda_gate=%.4f sampler_mode=%s loss_type=%s",
        stage1_epochs,
        n_pos_spike,
        n_neg_spike,
        gate_pos_weight_value,
        str(gate_use_global_pos_weight),
        gate_bs,
        float(model.cfg.encode_gate_thresh),
        float(model.cfg.assignment_gate_thresh),
        lambda_gate,
        gate_sampler_mode,
        gate_loss_type,
    )
    logger.info(
        "stage1_gate_train loss_weight_mode=%s selection_mode=%s acceptance_min_precision=%.4f acceptance_min_recall=%.4f "
        "log_source_metrics=%s early_stop_enabled=%s steps_per_epoch=%d",
        gate_loss_weight_mode,
        stage1_gate_selection_mode,
        stage1_acceptance_min_precision,
        stage1_acceptance_min_recall,
        str(gate_log_source_metrics),
        str(gate_early_stop_enabled),
        int(stage1_steps_per_epoch),
    )
    if stage1_encoder_aux_requested:
        logger.warning("stage1 encoder_aux is deprecated in the decoupled gate/encoder setup and will be ignored")
    logger.info(
        "stage2_encoder_setup epochs=%d source_mode=%s freeze_scope=%s data_filter=%s min_domains_per_unit=%d min_samples_per_unit_after_gate=%d domain_sampling=%s use_encode_gate=%s select_primary=%s select_secondary=%s reset_optimizer_on_entry=%s reset_scheduler_warmup_epochs=%d reference_units=%d reference_samples=%d",
        stage2_epochs,
        str(stage2_cfg.get("source_mode", "matched_positives")),
        str(stage2_cfg.get("freeze_scope", "gate_net_only")),
        stage2_data_filter,
        stage2_min_domains,
        stage2_min_samples_after_gate,
        str(stage2_cfg.get("domain_sampling", "round_robin")),
        str(stage2_cfg.get("use_encode_gate", False)),
        str(stage2_cfg.get("select_metric_primary", "val_lsm10_operational_gt")),
        str(stage2_cfg.get("select_metric_secondary", "val_knn_purity10_operational_gt")),
        str(stage2_reset_optimizer_on_entry),
        int(stage2_reset_scheduler_warmup_epochs),
        int(train_stage2_reference_allowed_units.size),
        int(train_stage2_reference_positive_idx.shape[0]),
    )
    if stage2_epochs > 0:
        logger.info(
            "stage2 gate early-stop bypassed because gate_net is frozen enabled=%s patience=%d warmup_epochs=%d",
            str(gate_early_stop_enabled),
            gate_patience,
            gate_warmup_epochs,
        )

    stage2_batch_sampler: Callable[[int | None], np.ndarray] | None = None
    stage2_fixed_support_index_set: set[int] | None = None
    if stage2_epochs > 0 and episodic_proto_ce and not stage2_temporal_episode_enabled:
        if sampler_max_n_way <= 0 or n_per_class <= 0:
            raise ValueError("sampler.mode=balanced requires sampler.max_n_way and sampler.n_per_class > 0")
        if k_shot <= 0:
            raise ValueError("sampler.mode=balanced with episodic proto_ce requires sampler.k_shot > 0")
        if k_shot >= n_per_class:
            logger.warning(
                "proto_episode_ce may become zero in stage-2: k_shot(%d) >= n_per_class(%d), query set can be empty",
                k_shot,
                n_per_class,
            )
        logger.info(
            "stage2_proto_ce enabled max_n_way=%d n_per_class=%d k_shot=%d proto_tau=%.6g lambda_proto_ce=%.6g",
            sampler_max_n_way,
            n_per_class,
            k_shot,
            proto_tau,
            lambda_proto_ce,
        )
    elif stage2_epochs > 0 and sampler_mode not in ("random", "balanced", "class_balanced"):
        raise ValueError(f"unsupported train.sampler.mode: {sampler_mode}")

    start_epoch = 0
    global_step = 0
    resume_path: Path | None = None
    resume_stage2_checkpoint_payload: Any | None = None
    resume_gate_calibration: Dict[str, Any] | None = None
    if resume_from:
        resume_path = resolve_path(resume_from, ROOT)
    elif auto_resume:
        resume_path = find_latest_checkpoint(checkpoint_dir)

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        logger.info("stage=resume start path=%s", resume_path)
        resume_ckpt_payload = load_training_checkpoint_payload(resume_path, device=device)
        resume_metadata = dict(resume_ckpt_payload.get("metadata", {}) or {}) if isinstance(resume_ckpt_payload, dict) else {}
        resume_stage_name = str(resume_metadata.get("stage", "")).lower()
        if "gate_calibration" in resume_metadata:
            resume_gate_calibration = dict(resume_metadata["gate_calibration"])
        if resume_stage_name == "stage2_encoder" and stage2_reset_optimizer_on_entry:
            load_model_state_from_checkpoint_payload(resume_ckpt_payload, model=model)
            start_epoch = int(resume_ckpt_payload.get("epoch", 0)) if isinstance(resume_ckpt_payload, dict) else 0
            global_step = int(resume_ckpt_payload.get("global_step", 0)) if isinstance(resume_ckpt_payload, dict) else 0
            resume_stage2_checkpoint_payload = resume_ckpt_payload
            logger.info(
                "stage=resume deferred_stage2_optimizer_restore start_epoch=%d global_step=%d",
                start_epoch,
                global_step,
            )
        else:
            start_epoch, global_step, scheduler_loaded = restore_training_checkpoint_payload(
                resume_ckpt_payload,
                model=model,
                optimizer=opt,
                device=device,
                scheduler=scheduler,
            )
            if scheduler is not None and not scheduler_loaded and global_step > 0:
                fast_steps = int(min(global_step, initial_scheduler_total_steps))
                scheduler.step(fast_steps)
                logger.info("scheduler_fast_forwarded steps=%d", fast_steps)
        logger.info("stage=resume done start_epoch=%d global_step=%d", start_epoch, global_step)

    if stage2_init_path is not None:
        stage1_payload = load_training_checkpoint_payload(stage2_init_path, device=device)
        stage1_metadata = dict(stage1_payload.get("metadata", {}) or {})
        if stage1_metadata.get("stage") != "stage1_gate":
            raise ValueError("stage2-init-from requires a stage-1 checkpoint")
        resume_gate_calibration = dict(stage1_metadata.get("gate_calibration", {}) or {})
        if not resume_gate_calibration.get("fixed_gate_goal_reached", False):
            raise ValueError("stage-1 checkpoint did not reach the fixed gate goal")
        start_epoch = stage1_epochs
        del stage1_payload

    latest_gate_calibration: Dict[str, Any] = {
        "encode_gate_thresh": float(model.cfg.encode_gate_thresh),
        "assignment_gate_thresh": float(model.cfg.assignment_gate_thresh),
        "thresholds": {
            "encode_gate_thresh": float(model.cfg.encode_gate_thresh),
            "assignment_gate_thresh": float(model.cfg.assignment_gate_thresh),
        },
        "encode_metrics": {"threshold": float(model.cfg.encode_gate_thresh)},
        "assignment_metrics": {"threshold": float(model.cfg.assignment_gate_thresh)},
        "domain_metrics": {},
        "source_metrics": {},
        "gate_precision": float("nan"),
        "gate_recall": float("nan"),
        "gate_f1": float("nan"),
        "noise_accuracy": float("nan"),
        "noise_specificity": float("nan"),
        "balanced_accuracy": float("nan"),
        "encode_pass_ratio": float("nan"),
        "assignment_pass_ratio": float("nan"),
        "positive_rate": float("nan"),
        "target_recall": float(gate_target_recall_encode),
        "target_recall_encode": float(gate_target_recall_encode),
        "target_recall_assignment_min": float(gate_target_recall_assignment_min),
        "selection_mode": "config",
        "selection_modes": {"encode": "config", "assignment": "config"},
        "all_positive_precision": float("nan"),
        "all_positive_f1": float("nan"),
    }
    if resume_gate_calibration is not None:
        latest_gate_calibration = dict(resume_gate_calibration)

    best_stage1_rank: tuple[float, ...] | None = None
    best_stage1_epoch = 0
    best_stage1_summary: Dict[str, Any] | None = None
    best_stage1_reference_metrics: Dict[str, float] | None = None
    best_stage1_model_state: Dict[str, torch.Tensor] | None = None
    best_stage2_rank: tuple[float, ...] | None = None
    best_stage2_epoch = 0
    best_stage2_metrics: Dict[str, float] | None = None
    best_stage2_reference_metrics: Dict[str, float] | None = None
    best_stage2_mixed_metrics: Dict[str, float] | None = None
    best_stage2_model_state: Dict[str, torch.Tensor] | None = None
    best_stage1_checkpoint_path = stage2_init_path or checkpoint_dir / "stage1_gate_best.ckpt"
    best_stage2_checkpoint_path = checkpoint_dir / "pretrain_best.ckpt"
    stage2_freeze_scope = str(stage2_cfg.get("freeze_scope", "gate_net_only"))
    stage2_source_mode = str(stage2_cfg.get("source_mode", "matched_positives")).lower()
    stage2_select_primary = str(stage2_cfg.get("select_metric_primary", "val_lsm10_operational_gt"))
    stage2_select_secondary = str(stage2_cfg.get("select_metric_secondary", "val_knn_purity10_operational_gt"))
    stage2_select_tertiary = str(stage2_cfg.get("select_metric_tertiary", "val_ber002_operational_gt"))
    stage2_prepared = False
    stage2_fallback_reason: str | None = None
    stage2_gate_filter_summary: Dict[str, Any] = {}
    train_stage2_operational_class_indices: Dict[int, np.ndarray] = {}
    train_stage2_operational_positive_idx = np.zeros((0,), dtype=np.int64)
    train_stage2_operational_allowed_units = np.zeros((0,), dtype=np.int64)
    train_stage2_operational_support: Dict[int, np.ndarray] = {}
    train_stage2_operational_support_summary = summarize_unit_domain_support({}, min_domains_required=stage2_min_domains)
    val_stage2_operational_positive_idx = np.zeros((0,), dtype=np.int64)
    train_stage2_training_class_indices: Dict[int, np.ndarray] = {}
    train_stage2_training_positive_idx = np.zeros((0,), dtype=np.int64)
    train_stage2_training_allowed_units = np.zeros((0,), dtype=np.int64)
    train_stage2_training_support: Dict[int, np.ndarray] = {}
    train_stage2_training_support_summary = summarize_unit_domain_support({}, min_domains_required=stage2_min_domains)
    val_stage2_training_positive_idx = np.zeros((0,), dtype=np.int64)
    train_stage2_training_valid_mask = np.zeros((train_patches.shape[0],), dtype=np.bool_)
    train_stage2_boundary_candidate_mask = np.zeros((train_patches.shape[0],), dtype=np.bool_)
    train_stage2_gate_prob = np.zeros((train_patches.shape[0],), dtype=np.float32)

    def prepare_stage2_state(*, restore_stage1_best: bool, init_reason: str) -> bool:
        nonlocal opt
        nonlocal scheduler
        nonlocal global_step
        nonlocal latest_gate_calibration
        nonlocal stage2_batch_sampler
        nonlocal stage2_fixed_support_index_set
        nonlocal stage2_steps_per_epoch
        nonlocal stage2_prepared
        nonlocal stage2_fallback_reason
        nonlocal stage2_gate_filter_summary
        nonlocal train_stage2_operational_class_indices
        nonlocal train_stage2_operational_positive_idx
        nonlocal train_stage2_operational_allowed_units
        nonlocal train_stage2_operational_support
        nonlocal train_stage2_operational_support_summary
        nonlocal val_stage2_operational_positive_idx
        nonlocal train_stage2_training_class_indices
        nonlocal train_stage2_training_positive_idx
        nonlocal train_stage2_training_allowed_units
        nonlocal train_stage2_training_support
        nonlocal train_stage2_training_support_summary
        nonlocal val_stage2_training_positive_idx
        nonlocal train_stage2_training_valid_mask
        nonlocal train_stage2_boundary_candidate_mask
        nonlocal train_stage2_gate_prob
        nonlocal resume_stage2_checkpoint_payload
        ckpt_metadata: Dict[str, Any] = {}
        deferred_stage2_resume = resume_stage2_checkpoint_payload if not restore_stage1_best else None
        if restore_stage1_best:
            if not best_stage1_checkpoint_path.exists():
                raise RuntimeError(f"stage-1 best checkpoint missing: {best_stage1_checkpoint_path}")
            ckpt_payload = load_training_checkpoint_payload(best_stage1_checkpoint_path, device=device)
            _, restored_global_step, scheduler_loaded = restore_training_checkpoint_payload(
                ckpt_payload,
                model=model,
                optimizer=opt,
                device=device,
                scheduler=scheduler,
                restore_optimizer=not stage2_reset_optimizer_on_entry,
                restore_scheduler=not stage2_reset_optimizer_on_entry,
            )
            global_step = int(restored_global_step)
            if (
                not stage2_reset_optimizer_on_entry
                and scheduler is not None
                and not scheduler_loaded
                and global_step > 0
            ):
                fast_steps = int(min(global_step, total_train_steps))
                scheduler.step(fast_steps)
                logger.info("stage2_scheduler_fast_forwarded steps=%d", fast_steps)
            ckpt_metadata = dict(ckpt_payload.get("metadata", {}) or {}) if isinstance(ckpt_payload, dict) else {}
            if "gate_calibration" in ckpt_metadata:
                latest_gate_calibration = dict(ckpt_metadata["gate_calibration"])
            elif best_stage1_summary is not None:
                latest_gate_calibration = dict(best_stage1_summary)
            logger.info(
                "stage_transition to=stage2_encoder init=stage1_gate_best best_epoch=%d restored_global_step=%d",
                best_stage1_epoch,
                global_step,
            )
        else:
            logger.info(
                "stage_transition to=stage2_encoder init=%s current_global_step=%d",
                init_reason,
                global_step,
            )
        freeze_stats = apply_freeze_scope(model, freeze_scope=stage2_freeze_scope)
        model.train()
        enforce_freeze_scope_modules(model, freeze_scope=stage2_freeze_scope)
        logger.info(
            "stage2_freeze scope=%s frozen_params=%d trainable_params=%d",
            stage2_freeze_scope,
            int(freeze_stats["frozen_params"]),
            int(freeze_stats["trainable_params"]),
        )
        encode_gate_thresh = float(latest_gate_calibration.get("encode_gate_thresh", model.cfg.encode_gate_thresh))
        assignment_gate_thresh = float(latest_gate_calibration.get("assignment_gate_thresh", model.cfg.assignment_gate_thresh))
        logger.info(
            "stage2_gate_thresholds encode_thresh=%.4f assignment_thresh=%.4f",
            encode_gate_thresh,
            assignment_gate_thresh,
        )
        logger.info(
            "stage2_train_filter mode=high_recall_gate_source_only overlap_filter=false qc_filter=false"
        )
        train_gate_subset_gt = build_stage2_frozen_gate_subset(
            model,
            train_patches,
            train_unit_id,
            train_geo_all,
            train_recording_id,
            train_spike_label,
            train_sample_source,
            spatial_stats_np=train_spatial_stats,
            channel_mask_np=train_channel_mask_all,
            strict_overlap_label=train_strict_overlap_label,
            overlap_label=train_overlap_label,
            qc_allowed=train_qc_allowed,
            strict_matching_ambiguous=train_strict_matching_ambiguous,
            relaxed_matching_ambiguous=train_relaxed_matching_ambiguous,
            snr_peak=train_snr_peak,
            event_amp_norm=train_event_amp_norm,
            event_spatial_score=train_event_spatial_score,
            event_simul_local=train_event_simul_local,
            event_simul_global=train_event_simul_global,
            stage2_qc_cfg=stage2_qc_cfg,
            device=device,
            batch_size=val_batch_size,
            encode_gate_thresh=encode_gate_thresh,
            min_domains_per_unit=stage2_min_domains,
            min_samples_per_unit=stage2_min_samples_after_gate,
            source_mode="primary_only",
            apply_overlap_filter=False,
            apply_qc_filter=False,
        )
        train_gate_subset = train_gate_subset_gt
        if stage2_source_mode != "primary_only":
            train_gate_subset = build_stage2_frozen_gate_subset(
                model,
                train_patches,
                train_unit_id,
                train_geo_all,
                train_recording_id,
                train_spike_label,
                train_sample_source,
                spatial_stats_np=train_spatial_stats,
                channel_mask_np=train_channel_mask_all,
                strict_overlap_label=train_strict_overlap_label,
                overlap_label=train_overlap_label,
                qc_allowed=train_qc_allowed,
                strict_matching_ambiguous=train_strict_matching_ambiguous,
                relaxed_matching_ambiguous=train_relaxed_matching_ambiguous,
                snr_peak=train_snr_peak,
                event_amp_norm=train_event_amp_norm,
                event_spatial_score=train_event_spatial_score,
                event_simul_local=train_event_simul_local,
                event_simul_global=train_event_simul_global,
                stage2_qc_cfg=stage2_qc_cfg,
                device=device,
                batch_size=val_batch_size,
                encode_gate_thresh=encode_gate_thresh,
                min_domains_per_unit=stage2_min_domains,
                min_samples_per_unit=stage2_min_samples_after_gate,
                source_mode=stage2_source_mode,
                apply_overlap_filter=False,
                apply_qc_filter=False,
            )
        val_gate_prob = collect_gate_probabilities(
            model,
            val_patches,
            val_geo_all,
            spatial_stats_np=val_spatial_stats,
            channel_mask_np=val_channel_mask_all,
            device=device,
            batch_size=val_batch_size,
        ) if val_patches.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
        val_gt_mask = build_gt_spike_mask(val_unit_id, val_spike_label, val_sample_source)
        val_training_positive_mask = build_stage2_positive_mask(
            val_unit_id,
            val_spike_label,
            val_sample_source,
            source_mode=stage2_source_mode,
        )
        val_gate_pass_mask = val_gt_mask & (val_gate_prob >= encode_gate_thresh)
        val_gate_pass_training_mask = val_training_positive_mask & (val_gate_prob >= encode_gate_thresh)
        val_allowed_source_names = (
            ["primary_matched_positive"]
            if stage2_source_mode == "primary_only"
            else ["primary_matched_positive", "secondary_matched_positive"]
        )
        val_valid_mask, _val_valid_stats = build_stage2_quality_mask_np(
            unit_id=val_unit_id,
            spike_label=val_spike_label,
            sample_source=val_sample_source,
            gate_prob=val_gate_prob,
            gate_threshold=encode_gate_thresh,
            allowed_sources=val_allowed_source_names,
            strict_overlap_label=val_strict_overlap_label,
            qc_allowed=val_qc_allowed,
            strict_matching_ambiguous=val_strict_matching_ambiguous,
            relaxed_matching_ambiguous=val_relaxed_matching_ambiguous,
            snr_peak=val_snr_peak,
            event_amp_norm=val_event_amp_norm,
            event_spatial_score=val_event_spatial_score,
            event_simul_local=val_event_simul_local,
            event_simul_global=val_event_simul_global,
            qc_cfg=stage2_qc_cfg,
            apply_overlap_filter=False,
            apply_qc_filter=False,
        )
        val_operational_valid_mask, _ = build_stage2_quality_mask_np(
            unit_id=val_unit_id,
            spike_label=val_spike_label,
            sample_source=val_sample_source,
            gate_prob=val_gate_prob,
            gate_threshold=encode_gate_thresh,
            allowed_sources=["primary_matched_positive"],
            strict_overlap_label=val_strict_overlap_label,
            qc_allowed=val_qc_allowed,
            strict_matching_ambiguous=val_strict_matching_ambiguous,
            relaxed_matching_ambiguous=val_relaxed_matching_ambiguous,
            snr_peak=val_snr_peak,
            event_amp_norm=val_event_amp_norm,
            event_spatial_score=val_event_spatial_score,
            event_simul_local=val_event_simul_local,
            event_simul_global=val_event_simul_global,
            qc_cfg=stage2_qc_cfg,
            apply_overlap_filter=False,
            apply_qc_filter=False,
        )
        train_stage2_operational_class_indices = dict(train_gate_subset_gt["class_indices"])
        train_stage2_operational_positive_idx = np.asarray(train_gate_subset_gt["positive_idx"], dtype=np.int64)
        train_stage2_operational_allowed_units = np.asarray(train_gate_subset_gt["allowed_units"], dtype=np.int64)
        train_stage2_operational_support = dict(train_gate_subset_gt["support_map"])
        train_stage2_operational_support_summary = dict(train_gate_subset_gt["support_summary"])
        val_stage2_operational_positive_idx = np.flatnonzero(val_operational_valid_mask).astype(np.int64, copy=False)
        train_stage2_training_class_indices = dict(train_gate_subset["class_indices"])
        train_stage2_training_positive_idx = np.asarray(train_gate_subset["positive_idx"], dtype=np.int64)
        train_stage2_training_allowed_units = np.asarray(train_gate_subset["allowed_units"], dtype=np.int64)
        train_stage2_training_support = dict(train_gate_subset["support_map"])
        train_stage2_training_support_summary = dict(train_gate_subset["support_summary"])
        train_stage2_training_valid_mask = np.asarray(train_gate_subset["valid_mask"], dtype=np.bool_)
        train_stage2_boundary_candidate_mask = np.asarray(train_gate_subset["boundary_candidate_mask"], dtype=np.bool_)
        train_stage2_gate_prob = np.asarray(train_gate_subset["gate_prob"], dtype=np.float32)
        val_stage2_training_positive_idx = np.flatnonzero(val_valid_mask).astype(np.int64, copy=False)
        logger.info(
            "stage2_validation_operational_filter mode=high_recall_gate_source_only overlap_filter=false qc_filter=false"
        )
        dropped_units = int(max(0, train_stage2_reference_allowed_units.size - train_stage2_operational_allowed_units.size))
        val_secondary_matched_positive_mask = (
            np.asarray(val_sample_source, dtype=np.int8) == int(GATE_SOURCE_NAME_TO_CODE["secondary_matched_positive"])
        ) if val_sample_source is not None else np.zeros((val_unit_id.shape[0],), dtype=np.bool_)
        stage2_gate_filter_summary = {
            "encode_gate_thresh": encode_gate_thresh,
            "assignment_gate_thresh": assignment_gate_thresh,
            "source_mode": stage2_source_mode,
            "train_gt": int(np.sum(train_gate_subset_gt["gt_mask"])),
            "train_gate_pass_gt": int(np.sum(train_gate_subset_gt["gt_gate_pass_mask"])),
            "train_valid_gt": int(np.sum(train_gate_subset_gt["valid_mask"])),
            "val_gt": int(np.sum(val_gt_mask)),
            "val_gate_pass_gt": int(np.sum(val_gate_pass_mask)),
            "val_valid_gt": int(np.sum(val_operational_valid_mask)),
            "reference_units": int(train_stage2_reference_allowed_units.size),
            "operational_units": int(train_stage2_operational_allowed_units.size),
            "dropped_units": dropped_units,
            "training_units": int(train_stage2_training_allowed_units.size),
            "train_samples_operational_gt": int(train_stage2_operational_positive_idx.size),
            "val_samples_operational_gt": int(val_stage2_operational_positive_idx.size),
            "train_samples_training_source": int(train_stage2_training_positive_idx.size),
            "val_samples_training_source": int(val_stage2_training_positive_idx.size),
            "train_gate_pass_ratio_by_domain": dict(train_gate_subset_gt["gate_pass_ratio_by_domain"]),
            "val_gate_pass_ratio_by_domain": dict(summarize_gate_pass_ratios_by_domain(val_recording_id, val_gt_mask, val_gate_pass_mask)),
            "secondary_matched_positive_excluded_train": int(train_gate_subset_gt["secondary_matched_positive_excluded"]),
            "secondary_matched_positive_excluded_val": int(
                np.sum(np.asarray(val_sample_source, dtype=np.int8) == int(GATE_SOURCE_NAME_TO_CODE["secondary_matched_positive"]))
            ) if val_sample_source is not None else 0,
            "secondary_matched_positive_included_train": int(train_gate_subset.get("secondary_matched_positive_included", 0)),
            "secondary_matched_positive_included_val": int(np.sum(val_secondary_matched_positive_mask & val_gate_pass_training_mask)),
            "train_valid_stats": dict(train_gate_subset.get("valid_stats", {})),
            "val_valid_stats": dict(_val_valid_stats),
        }
        logger.info(
            "stage2_gate_filter_split train_gt=%d train_gate_pass_gt=%d train_valid_gt=%d val_gt=%d val_gate_pass_gt=%d val_valid_gt=%d",
            int(stage2_gate_filter_summary["train_gt"]),
            int(stage2_gate_filter_summary["train_gate_pass_gt"]),
            int(stage2_gate_filter_summary["train_valid_gt"]),
            int(stage2_gate_filter_summary["val_gt"]),
            int(stage2_gate_filter_summary["val_gate_pass_gt"]),
            int(stage2_gate_filter_summary["val_valid_gt"]),
        )
        logger.info(
            "stage2_gate_filter_units pre_units=%d post_units=%d dropped_units=%d",
            int(stage2_gate_filter_summary["reference_units"]),
            int(stage2_gate_filter_summary["operational_units"]),
            int(stage2_gate_filter_summary["dropped_units"]),
        )
        logger.info(
            "stage2_gate_filter_support min_domains=%d min_samples=%d eligible_units=%d",
            stage2_min_domains,
            stage2_min_samples_after_gate,
            int(train_stage2_operational_allowed_units.size),
        )
        logger.info(
            "stage2_training_subset source_mode=%s train_units=%d train_samples=%d val_samples=%d secondary_matched_positive_included_train=%d secondary_matched_positive_included_val=%d",
            str(stage2_gate_filter_summary["source_mode"]),
            int(stage2_gate_filter_summary["training_units"]),
            int(stage2_gate_filter_summary["train_samples_training_source"]),
            int(stage2_gate_filter_summary["val_samples_training_source"]),
            int(stage2_gate_filter_summary["secondary_matched_positive_included_train"]),
            int(stage2_gate_filter_summary["secondary_matched_positive_included_val"]),
        )
        logger.info(
            "stage2_gate_filter_domain_pass split=train %s",
            format_gate_pass_ratio_summary(dict(stage2_gate_filter_summary["train_gate_pass_ratio_by_domain"])),
        )
        logger.info(
            "stage2_gate_filter_domain_pass split=val %s",
            format_gate_pass_ratio_summary(dict(stage2_gate_filter_summary["val_gate_pass_ratio_by_domain"])),
        )
        logger.info(
            "stage2_gate_filter_audit secondary_matched_positive_excluded_train=%d secondary_matched_positive_excluded_val=%d",
            int(stage2_gate_filter_summary["secondary_matched_positive_excluded_train"]),
            int(stage2_gate_filter_summary["secondary_matched_positive_excluded_val"]),
        )
        if train_stage2_operational_positive_idx.size == 0 or train_stage2_operational_allowed_units.size == 0 or val_stage2_operational_positive_idx.size == 0:
            stage2_steps_per_epoch = 0
            stage2_batch_sampler = None
            stage2_fixed_support_index_set = None
            stage2_prepared = True
            stage2_fallback_reason = "insufficient_gate_pass_support"
            logger.warning(
                "stage2_skipped reason=%s train_operational_samples=%d val_operational_samples=%d operational_units=%d",
                stage2_fallback_reason,
                int(train_stage2_operational_positive_idx.size),
                int(val_stage2_operational_positive_idx.size),
                int(train_stage2_operational_allowed_units.size),
            )
            return False
        if train_stage2_training_positive_idx.size == 0 or train_stage2_training_allowed_units.size == 0:
            stage2_steps_per_epoch = 0
            stage2_batch_sampler = None
            stage2_fixed_support_index_set = None
            stage2_prepared = True
            stage2_fallback_reason = "insufficient_stage2_training_support"
            logger.warning(
                "stage2_skipped reason=%s train_training_samples=%d training_units=%d",
                stage2_fallback_reason,
                int(train_stage2_training_positive_idx.size),
                int(train_stage2_training_allowed_units.size),
            )
            return False
        if stage2_temporal_episode_enabled:
            temporal_batch_size_tasks = int(train_cfg.get("batch_size_tasks", stage2_episode_cfg.get("batch_size_tasks", 2)))
            temporal_max_spikes = int(stage2_episode_cfg.get("max_spikes_per_task", stage2_episode_cfg.get("max_samples_per_task", 256)))
            temporal_episode_size = max(1, temporal_batch_size_tasks * temporal_max_spikes)
            stage2_steps_per_epoch = int(max(1, math.ceil(max(1, train_stage2_training_positive_idx.shape[0]) / temporal_episode_size)))
        else:
            stage2_steps_per_epoch = int(max(1, math.ceil(max(1, train_stage2_training_positive_idx.shape[0]) / max(1, bs))))
        if stage2_reset_optimizer_on_entry and (restore_stage1_best or deferred_stage2_resume is not None):
            trainable_parameters = [param for param in model.parameters() if param.requires_grad]
            opt = build_optimizer_for_parameters(
                trainable_parameters,
                lr=base_lr,
                weight_decay=weight_decay,
            )
            stage2_total_train_steps = int(max(1, stage2_schedule_epochs * max(1, stage2_steps_per_epoch)))
            scheduler, stage2_scheduler_info = build_scheduler_from_config(
                opt,
                scheduler_cfg=scheduler_cfg,
                base_lr=base_lr,
                total_steps=stage2_total_train_steps,
                steps_per_epoch=stage2_steps_per_epoch,
                warmup_epochs=stage2_reset_scheduler_warmup_epochs,
            )
            logger.info(
                "stage2_optimizer_reset mode=%s trainable_params=%d lr=%.6g scheduler_type=%s warmup_epochs=%d total_steps=%d",
                "resume_current_stage2" if deferred_stage2_resume is not None else init_reason,
                int(sum(int(param.numel()) for param in trainable_parameters)),
                base_lr,
                str(stage2_scheduler_info.get("type", "none")),
                int(stage2_scheduler_info.get("warmup_epochs", 0)),
                int(stage2_scheduler_info.get("total_steps", 0)),
            )
            if deferred_stage2_resume is not None:
                _, restored_global_step, scheduler_loaded = restore_training_checkpoint_payload(
                    deferred_stage2_resume, model=model, optimizer=opt, device=device, scheduler=scheduler)
                global_step = int(restored_global_step)
                if scheduler is not None and not scheduler_loaded:
                    stage2_completed_epochs = int(max(0, start_epoch - stage1_epochs))
                    fast_steps = int(min(stage2_completed_epochs * stage2_steps_per_epoch, stage2_total_train_steps))
                    scheduler.step(fast_steps)
                    logger.info(
                        "stage2_resume scheduler_fast_forwarded steps=%d restored_global_step=%d",
                        fast_steps,
                        global_step,
                    )
                else:
                    logger.info("stage2_resume restored_global_step=%d", global_step)
                deferred_metadata = (
                    dict(deferred_stage2_resume.get("metadata", {}) or {}) if isinstance(deferred_stage2_resume, dict) else {}
                )
                if "gate_calibration" in deferred_metadata:
                    latest_gate_calibration = dict(deferred_metadata["gate_calibration"])
                resume_stage2_checkpoint_payload = None
        if stage2_temporal_episode_enabled:
            temporal_min_way = int(stage2_episode_cfg.get("min_way", 2))
            temporal_batch_size_tasks = int(train_cfg.get("batch_size_tasks", stage2_episode_cfg.get("batch_size_tasks", 2)))
            temporal_max_spikes = int(stage2_episode_cfg.get("max_spikes_per_task", stage2_episode_cfg.get("max_samples_per_task", 256)))
            stage2_batch_sampler = build_stage2_temporal_episode_sampler(
                train_unit_id,
                train_recording_id,
                train_event_time_sample,
                train_peak_channel,
                train_stage2_training_positive_idx,
                seed=seed + 311,
                min_way=temporal_min_way,
                batch_size_tasks=temporal_batch_size_tasks,
                max_spikes_per_task=temporal_max_spikes,
                fixed_support_once=bool(stage2_episode_cfg.get("fixed_support_once", False))
                or str(stage2_episode_cfg.get("sampling_mode", "")).strip().lower()
                in {"fixed_support_once", "single_support_once"},
                fixed_support_per_unit=int(stage2_episode_cfg.get("fixed_support_per_unit", 1)),
            )
            fixed_support_indices = getattr(stage2_batch_sampler, "fixed_support_indices", np.zeros((0,), dtype=np.int64))
            fixed_support_enabled = bool(stage2_episode_cfg.get("fixed_support_once", False)) or str(
                stage2_episode_cfg.get("sampling_mode", "")
            ).strip().lower() in {"fixed_support_once", "single_support_once"}
            stage2_fixed_support_index_set = (
                set(int(v) for v in np.asarray(fixed_support_indices, dtype=np.int64).reshape(-1).tolist())
                if fixed_support_enabled
                else None
            )
            logger.info(
                "stage2_recording_task_sampler enabled batch_size_tasks=%d max_spikes_per_task=%d n_time_blocks=%d anchor_blocks_min=%d anchor_blocks_max=%d causal_stress_fraction=%.3f min_way=%d steps_per_epoch=%d fixed_support_once=%s fixed_support_count=%d",
                temporal_batch_size_tasks,
                temporal_max_spikes,
                int(stage2_episode_cfg.get("n_time_blocks", 6)),
                int(stage2_episode_cfg.get("anchor_blocks_min", 1)),
                int(stage2_episode_cfg.get("anchor_blocks_max", 2)),
                float(stage2_episode_cfg.get("causal_stress_fraction", 0.25)),
                temporal_min_way,
                stage2_steps_per_epoch,
                str(fixed_support_enabled),
                int(len(stage2_fixed_support_index_set or ())),
            )
        elif episodic_proto_ce:
            raise RuntimeError("deprecated episodic_proto_ce sampler path was removed; enable stage2.episode")
        else:
            stage2_batch_sampler = None
            stage2_fixed_support_index_set = None
        stage2_prepared = True
        return True

    if start_epoch >= stage1_epochs and stage2_epochs > 0:
        prepare_stage2_state(restore_stage1_best=False, init_reason="resume_current_stage2")

    proto_metric_keys = (
        "local_proto_count",
        "local_proto_query_count",
        "local_proto_anchor_unit_count",
        "local_proto_query_unit_count",
        "local_proto_pos_terms",
        "local_proto_neg_terms",
        "local_proto_split_attempts",
        "local_proto_split_accepts",
        "local_proto_split_reject_small_child",
        "local_proto_split_reject_low_gain",
        "local_proto_split_reject_low_residual",
        "local_proto_split_reject_child_close",
        "local_proto_split_reject_child_far",
        "local_proto_residual_p90_before",
        "local_proto_residual_p90_after",
        "local_proto_split_gain_mean",
        "local_proto_units_p1",
        "local_proto_units_p2",
        "local_proto_units_p3",
        "local_proto_units_p4",
        "mean_proto_per_anchor_unit",
        "local_proto_nce_loss",
        "local_proto_same_loss",
        "local_proto_chain_loss",
        "same_proto_cos_mean",
        "same_proto_cos_min",
        "same_proto_pair_count",
        "temporal_chain_cos_mean",
        "temporal_chain_cos_min",
        "temporal_chain_pair_count",
        "local_proto_warmup_active",
        "local_proto_temporal_fallback",
    )

    def train_step(
        idx: np.ndarray,
        gate_idx: np.ndarray,
        *,
        use_proto_ce: bool,
        stage_name: str,
    ) -> Dict[str, float]:
        main_loss = torch.zeros((), device=device)
        main_terms = {
            "supcon": torch.zeros((), device=device),
            "proto": torch.zeros((), device=device),
            "proto_ce": torch.zeros((), device=device),
            "channel": torch.zeros((), device=device),
            "umap_teacher_graph": torch.zeros((), device=device),
            "supervised_graph": torch.zeros((), device=device),
            "boundary": torch.zeros((), device=device),
            "lambda_umap_eff": torch.zeros((), device=device),
            "lambda_supervised_graph_eff": torch.zeros((), device=device),
            "lambda_boundary_eff": torch.zeros((), device=device),
            "stage2_valid_count": torch.zeros((), device=device),
            "episode_way": torch.zeros((), device=device),
            "episode_support_count": torch.zeros((), device=device),
            "episode_query_count": torch.zeros((), device=device),
            "teacher_graph_pos_edges": torch.zeros((), device=device),
            "teacher_graph_neg_edges": torch.zeros((), device=device),
            "teacher_graph_random_neg_edges": torch.zeros((), device=device),
            "teacher_graph_hard_neg_edges": torch.zeros((), device=device),
            "teacher_graph_hard_neg_fraction": torch.zeros((), device=device),
            "supervised_graph_pos_edges": torch.zeros((), device=device),
            "supervised_graph_neg_edges": torch.zeros((), device=device),
            "supervised_graph_hard_neg_edges": torch.zeros((), device=device),
            "supervised_graph_mean_q_pos": torch.zeros((), device=device),
            "supervised_graph_mean_q_neg": torch.zeros((), device=device),
            "boundary_anchor_count": torch.zeros((), device=device),
            "temporal_conn_loss": torch.zeros((), device=device),
            "lambda_temporal_conn_eff": torch.zeros((), device=device),
            "temporal_conn_pair_count": torch.zeros((), device=device),
            "temporal_conn_active_ratio": torch.zeros((), device=device),
        }
        for key in proto_metric_keys:
            main_terms[key] = torch.zeros((), device=device)
        gate_terms = {"gate": torch.zeros((), device=device)}
        encode_pass = float("nan")

        def _compute_embedding_batch_loss(batch_idx: np.ndarray, *, use_proto_ce_local: bool) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            x = torch.from_numpy(train_patches[batch_idx]).to(device)
            geo = torch.from_numpy(train_geo_all[batch_idx]).to(device)
            channel_mask = torch.from_numpy(train_channel_mask_all[batch_idx].astype(np.bool_, copy=False)).to(device)
            y = torch.from_numpy(train_unit_id[batch_idx]).to(device)
            gate_threshold_step = float(stage2_gate_filter_summary.get("encode_gate_thresh", model.cfg.encode_gate_thresh))
            batch_base: Dict[str, torch.Tensor] = {
                "unit_id": y,
                "spike_label": torch.from_numpy(train_spike_label[batch_idx]).to(device),
                "sample_source": torch.from_numpy(train_sample_source[batch_idx] if train_sample_source is not None else np.zeros((batch_idx.shape[0],), dtype=np.int8)).to(device),
                "recording_id": torch.from_numpy(train_recording_id[batch_idx]).to(device),
                "event_time_sample": torch.from_numpy(train_event_time_sample[batch_idx]).to(device),
                "center_channel": torch.from_numpy(train_center_channel[batch_idx]).to(device),
                "peak_channel": torch.from_numpy(train_peak_channel[batch_idx]).to(device),
                "strict_overlap_label": torch.from_numpy(train_strict_overlap_label[batch_idx]).to(device),
                "overlap_label": torch.from_numpy(train_overlap_label[batch_idx]).to(device),
                "qc_allowed": torch.from_numpy(train_qc_allowed[batch_idx]).to(device),
                "strict_matching_ambiguous": torch.from_numpy(train_strict_matching_ambiguous[batch_idx]).to(device),
                "relaxed_matching_ambiguous": torch.from_numpy(train_relaxed_matching_ambiguous[batch_idx]).to(device),
                "event_amp_norm": torch.from_numpy(train_event_amp_norm[batch_idx]).to(device),
                "event_spatial_score": torch.from_numpy(train_event_spatial_score[batch_idx]).to(device),
                "event_simul_local": torch.from_numpy(train_event_simul_local[batch_idx]).to(device),
                "event_simul_global": torch.from_numpy(train_event_simul_global[batch_idx]).to(device),
                "valid_mask": torch.from_numpy(train_stage2_training_valid_mask[batch_idx]).to(device),
                "boundary_candidate_mask": torch.from_numpy(train_stage2_boundary_candidate_mask[batch_idx]).to(device),
                "gate_confidence": torch.from_numpy(train_stage2_gate_prob[batch_idx]).to(device),
                "gate_pass": torch.from_numpy((train_stage2_gate_prob[batch_idx] >= gate_threshold_step).astype(np.bool_)).to(device),
                "stage2_epoch": torch.tensor(max(0, int(stage_epoch) - 1), device=device),
            }
            if stage2_fixed_support_index_set is not None:
                fixed_support_mask = np.fromiter(
                    (int(sample_idx) in stage2_fixed_support_index_set for sample_idx in batch_idx.tolist()),
                    dtype=np.bool_,
                    count=int(batch_idx.shape[0]),
                )
                batch_base["fixed_support_mask"] = torch.from_numpy(fixed_support_mask).to(device)
            if contrastive_views_enabled:
                x1, x2 = augment_batch_views(
                    x,
                    noise_std=contrastive_noise_std,
                    gain_jitter=contrastive_gain_jitter,
                )
                x_in = torch.cat([x1, x2], dim=0)
                geo_in = torch.cat([geo, geo], dim=0)
                channel_mask_in = torch.cat([channel_mask, channel_mask], dim=0)
                y_in = torch.cat([y, y], dim=0)
            else:
                x_in = x
                geo_in = geo
                channel_mask_in = channel_mask
                y_in = y
            center_batch = torch.from_numpy(train_center_channel[batch_idx].astype(np.float32, copy=False)).to(device)
            center_in = torch.cat([center_batch, center_batch], dim=0) if contrastive_views_enabled else center_batch
            dch = torch.abs(center_in[:, None] - center_in[None, :])
            dch_scale = torch.amax(dch)
            if float(dch_scale.detach().cpu().item()) > 0.0:
                dch = dch / dch_scale
            out_main = model.forward_encoder_only(x_in, {"geo": geo_in, "channel_mask": channel_mask_in})
            if stage2_temporal_episode_enabled:
                valid_mask_t = batch_base["valid_mask"].to(device=device, dtype=torch.bool)
                valid_stats = {
                    "stage2_valid_count": int(torch.sum(valid_mask_t).detach().cpu().item()),
                    "stage2_valid_ratio": (
                        float(torch.mean(valid_mask_t.to(dtype=torch.float32)).detach().cpu().item())
                        if int(valid_mask_t.numel()) > 0
                        else 0.0
                    ),
                }
                episode = build_temporal_episode(batch_base, valid_mask_t, cfg)
                if episode is None:
                    zero_loss = out_main.embedding.sum() * 0.0
                    terms = {
                        "supcon": zero_loss.detach(),
                        "proto": zero_loss.detach(),
                        "proto_ce": zero_loss.detach(),
                        "channel": zero_loss.detach(),
                        "umap_teacher_graph": zero_loss.detach(),
                        "supervised_graph": zero_loss.detach(),
                        "boundary": zero_loss.detach(),
                        "lambda_umap_eff": torch.tensor(0.0, device=device),
                        "lambda_supervised_graph_eff": torch.tensor(0.0, device=device),
                        "lambda_boundary_eff": torch.tensor(0.0, device=device),
                        "stage2_valid_count": torch.tensor(float(valid_stats.get("stage2_valid_count", 0)), device=device),
                        "episode_way": torch.tensor(0.0, device=device),
                        "episode_support_count": torch.tensor(0.0, device=device),
                        "episode_query_count": torch.tensor(0.0, device=device),
                        "teacher_graph_pos_edges": torch.tensor(0.0, device=device),
                        "teacher_graph_neg_edges": torch.tensor(0.0, device=device),
                        "teacher_graph_random_neg_edges": torch.tensor(0.0, device=device),
                        "teacher_graph_hard_neg_edges": torch.tensor(0.0, device=device),
                        "teacher_graph_hard_neg_fraction": torch.tensor(0.0, device=device),
                        "supervised_graph_pos_edges": torch.tensor(0.0, device=device),
                        "supervised_graph_neg_edges": torch.tensor(0.0, device=device),
                        "supervised_graph_hard_neg_edges": torch.tensor(0.0, device=device),
                        "supervised_graph_mean_q_pos": torch.tensor(0.0, device=device),
                        "supervised_graph_mean_q_neg": torch.tensor(0.0, device=device),
                        "boundary_anchor_count": torch.tensor(0.0, device=device),
                        "temporal_conn_loss": torch.tensor(0.0, device=device),
                        "lambda_temporal_conn_eff": torch.tensor(0.0, device=device),
                        "temporal_conn_pair_count": torch.tensor(0.0, device=device),
                        "temporal_conn_active_ratio": torch.tensor(0.0, device=device),
                    }
                    for key in proto_metric_keys:
                        terms[key] = torch.tensor(0.0, device=device)
                    return zero_loss, terms
                teacher_feature = None
                if teacher_model is not None and graph_lambda_umap_cfg > 0.0:
                    with torch.no_grad():
                        teacher_out = teacher_model.forward_encoder_only(x, {"geo": geo, "channel_mask": channel_mask})
                        target_feature = str(graph_cfg.get("target_feature", "fused_feature")).lower()
                        if target_feature in ("embedding", "teacher_embedding"):
                            teacher_feature = F.normalize(teacher_out.embedding.detach(), p=2, dim=1)
                        else:
                            if teacher_out.fused_feature is None:
                                raise RuntimeError("teacher model did not return fused_feature")
                            teacher_feature = F.normalize(teacher_out.fused_feature.detach(), p=2, dim=1)
                main_loss_new, loss_stats = stage2_total_loss(
                    student_out=out_main,
                    teacher_feature=teacher_feature,
                    batch=batch_base,
                    episode=episode,
                    cfg=cfg,
                )
                conn_loss, conn_stats = temporal_connection_loss(out_main.embedding, batch_base, episode, cfg)
                main_loss_new = main_loss_new + float(conn_stats.get("lambda_temporal_conn_eff", 0.0)) * conn_loss
                loss_stats.update(conn_stats)
                for key in ("episode_way", "episode_support_count", "episode_query_count"):
                    if key in episode:
                        loss_stats[key] = episode[key].detach()
                terms = {
                    "supcon": loss_stats.get("supcon", out_main.embedding.new_zeros(())),
                    "proto": loss_stats.get("proto", out_main.embedding.new_zeros(())),
                    "proto_ce": loss_stats.get("proto_ce", out_main.embedding.new_zeros(())),
                    "channel": loss_stats.get("channel", out_main.embedding.new_zeros(())),
                    "umap_teacher_graph": loss_stats.get("umap_teacher_graph", out_main.embedding.new_zeros(())),
                    "supervised_graph": loss_stats.get("supervised_graph", out_main.embedding.new_zeros(())),
                    "boundary": loss_stats.get("boundary", out_main.embedding.new_zeros(())),
                    "lambda_umap_eff": torch.tensor(float(loss_stats.get("lambda_umap_eff", 0.0)), device=device),
                    "lambda_supervised_graph_eff": torch.tensor(float(loss_stats.get("lambda_supervised_graph_eff", 0.0)), device=device),
                    "lambda_boundary_eff": torch.tensor(float(loss_stats.get("lambda_boundary_eff", 0.0)), device=device),
                    "stage2_valid_count": torch.tensor(float(loss_stats.get("stage2_valid_count", 0)), device=device),
                    "episode_way": torch.as_tensor(loss_stats.get("episode_way", 0.0), device=device, dtype=torch.float32),
                    "episode_support_count": torch.as_tensor(loss_stats.get("episode_support_count", 0.0), device=device, dtype=torch.float32),
                    "episode_query_count": torch.as_tensor(loss_stats.get("episode_query_count", 0.0), device=device, dtype=torch.float32),
                    "teacher_graph_pos_edges": torch.tensor(float(loss_stats.get("teacher_graph_pos_edges", 0)), device=device),
                    "teacher_graph_neg_edges": torch.tensor(float(loss_stats.get("teacher_graph_neg_edges", 0)), device=device),
                    "teacher_graph_random_neg_edges": torch.tensor(float(loss_stats.get("teacher_graph_random_neg_edges", 0)), device=device),
                    "teacher_graph_hard_neg_edges": torch.tensor(float(loss_stats.get("teacher_graph_hard_neg_edges", 0)), device=device),
                    "teacher_graph_hard_neg_fraction": torch.tensor(float(loss_stats.get("teacher_graph_hard_neg_fraction", 0.0)), device=device),
                    "supervised_graph_pos_edges": torch.tensor(float(loss_stats.get("supervised_graph_pos_edges", 0)), device=device),
                    "supervised_graph_neg_edges": torch.tensor(float(loss_stats.get("supervised_graph_neg_edges", 0)), device=device),
                    "supervised_graph_hard_neg_edges": torch.tensor(float(loss_stats.get("supervised_graph_hard_neg_edges", 0)), device=device),
                    "supervised_graph_mean_q_pos": torch.tensor(float(loss_stats.get("supervised_graph_mean_q_pos", 0.0)), device=device),
                    "supervised_graph_mean_q_neg": torch.tensor(float(loss_stats.get("supervised_graph_mean_q_neg", 0.0)), device=device),
                    "boundary_anchor_count": torch.tensor(float(loss_stats.get("boundary_anchor_count", 0)), device=device),
                    "temporal_conn_loss": torch.tensor(float(loss_stats.get("temporal_conn_loss", 0.0)), device=device),
                    "lambda_temporal_conn_eff": torch.tensor(float(loss_stats.get("lambda_temporal_conn_eff", 0.0)), device=device),
                    "temporal_conn_pair_count": torch.tensor(float(loss_stats.get("temporal_conn_pair_count", 0.0)), device=device),
                    "temporal_conn_active_ratio": torch.tensor(float(loss_stats.get("temporal_conn_active_ratio", 0.0)), device=device),
                }
                for key in proto_metric_keys:
                    terms[key] = torch.tensor(float(loss_stats.get(key, 0.0)), device=device)
                return main_loss_new, terms
            raise RuntimeError("stage-2 encoder training requires stage2.episode.enabled=true")

        if stage_name == "stage2_encoder" and idx.size > 0:
            model.train()
            enforce_stage_modules(model, stage_name=stage_name)
            main_loss, computed_terms = _compute_embedding_batch_loss(idx, use_proto_ce_local=use_proto_ce)
            main_terms.update(computed_terms)
        elif stage_name == "stage1_gate" and gate_idx.size > 0:
            model.train()
            enforce_stage_modules(model, stage_name=stage_name)
            x_gate = torch.from_numpy(train_patches[gate_idx]).to(device)
            geo_gate = torch.from_numpy(train_geo_all[gate_idx]).to(device)
            channel_mask_gate = torch.from_numpy(train_channel_mask_all[gate_idx].astype(np.bool_, copy=False)).to(device)
            spatial_gate = torch.from_numpy(train_spatial_stats[gate_idx]).to(device)
            spike_gate = torch.from_numpy(train_spike_label[gate_idx]).to(device)
            gate_logit, gate_prob = model.forward_gate_heads(
                x_gate,
                {"geo": geo_gate, "spatial_stats": spatial_gate, "channel_mask": channel_mask_gate},
            )
            main_loss, gate_terms = gate_detection_loss(
                gate_logit,
                spike_gate,
                gate_pos_weight=gate_pos_weight,
                gate_sample_weights=None,
                lambda_gate=lambda_gate,
                gate_loss_type=gate_loss_type,
            )
            encode_pass = float((gate_prob >= model.cfg.encode_gate_thresh).float().mean().detach().cpu().item())
        else:
            raise RuntimeError(f"invalid training batch for stage={stage_name}: idx={idx.size} gate_idx={gate_idx.size}")

        opt.zero_grad(set_to_none=True)
        if not main_loss.requires_grad:
            trainable_anchor = None
            for param in model.parameters():
                if param.requires_grad:
                    trainable_anchor = param.sum() * 0.0
                    break
            if trainable_anchor is not None:
                main_loss = main_loss + trainable_anchor
        main_loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        step_result = {
            "loss": float(main_loss.detach().cpu().item()),
            "gate": float(gate_terms["gate"].detach().cpu().item()),
            "supcon": float(main_terms["supcon"].detach().cpu().item()),
            "proto": float(main_terms["proto"].detach().cpu().item()),
            "proto_ce": float(main_terms["proto_ce"].detach().cpu().item()),
            "channel": float(main_terms["channel"].detach().cpu().item()),
            "umap_teacher_graph": float(main_terms["umap_teacher_graph"].detach().cpu().item()),
            "supervised_graph": float(main_terms["supervised_graph"].detach().cpu().item()),
            "boundary": float(main_terms["boundary"].detach().cpu().item()),
            "lambda_umap_eff": float(main_terms["lambda_umap_eff"].detach().cpu().item()),
            "lambda_supervised_graph_eff": float(main_terms["lambda_supervised_graph_eff"].detach().cpu().item()),
            "lambda_boundary_eff": float(main_terms["lambda_boundary_eff"].detach().cpu().item()),
            "stage2_valid_count": float(main_terms["stage2_valid_count"].detach().cpu().item()),
            "episode_way": float(main_terms["episode_way"].detach().cpu().item()),
            "episode_support_count": float(main_terms["episode_support_count"].detach().cpu().item()),
            "episode_query_count": float(main_terms["episode_query_count"].detach().cpu().item()),
            "teacher_graph_pos_edges": float(main_terms["teacher_graph_pos_edges"].detach().cpu().item()),
            "teacher_graph_neg_edges": float(main_terms["teacher_graph_neg_edges"].detach().cpu().item()),
            "teacher_graph_random_neg_edges": float(main_terms["teacher_graph_random_neg_edges"].detach().cpu().item()),
            "teacher_graph_hard_neg_edges": float(main_terms["teacher_graph_hard_neg_edges"].detach().cpu().item()),
            "teacher_graph_hard_neg_fraction": float(main_terms["teacher_graph_hard_neg_fraction"].detach().cpu().item()),
            "supervised_graph_pos_edges": float(main_terms["supervised_graph_pos_edges"].detach().cpu().item()),
            "supervised_graph_neg_edges": float(main_terms["supervised_graph_neg_edges"].detach().cpu().item()),
            "supervised_graph_hard_neg_edges": float(main_terms["supervised_graph_hard_neg_edges"].detach().cpu().item()),
            "supervised_graph_mean_q_pos": float(main_terms["supervised_graph_mean_q_pos"].detach().cpu().item()),
            "supervised_graph_mean_q_neg": float(main_terms["supervised_graph_mean_q_neg"].detach().cpu().item()),
            "boundary_anchor_count": float(main_terms["boundary_anchor_count"].detach().cpu().item()),
            "temporal_conn_loss": float(main_terms["temporal_conn_loss"].detach().cpu().item()),
            "lambda_temporal_conn_eff": float(main_terms["lambda_temporal_conn_eff"].detach().cpu().item()),
            "temporal_conn_pair_count": float(main_terms["temporal_conn_pair_count"].detach().cpu().item()),
            "temporal_conn_active_ratio": float(main_terms["temporal_conn_active_ratio"].detach().cpu().item()),
            "encode_pass": encode_pass,
        }
        for key in proto_metric_keys:
            step_result[key] = float(main_terms[key].detach().cpu().item())
        return step_result

    logger.info(
        "train_setup total_epochs=%d batch_size=%d gate_batch_size=%d stage1_steps_per_epoch=%d stage2_steps_per_epoch=%d lr=%.6g weight_decay=%.6g save=%s",
        total_epochs,
        bs,
        gate_bs,
        stage1_steps_per_epoch,
        stage2_steps_per_epoch,
        float(train_cfg.get("lr", 1e-3)),
        float(train_cfg.get("weight_decay", 1e-4)),
        save_out,
    )
    logger.info("stage2 execution_epochs=%d schedule_epochs=%d", stage2_epochs, stage2_schedule_epochs)
    logger.info("stage=train_loop start")
    total_start = time.perf_counter()
    if start_epoch >= total_epochs:
        logger.info("resume epoch (%d) >= total epochs (%d), skipping train loop", start_epoch, total_epochs)

    prev_stage_name: str | None = None
    for ep in range(start_epoch, total_epochs):
        stage_name = "stage1_gate" if ep < stage1_epochs else "stage2_encoder"
        stage_epoch = (ep + 1) if stage_name == "stage1_gate" else (ep - stage1_epochs + 1)
        stage_total = stage1_epochs if stage_name == "stage1_gate" else stage2_epochs
        if stage_name != prev_stage_name and stage_name == "stage1_gate":
            stage1_train_stats = apply_stage_trainability(model, stage_name=stage_name)
            model.train()
            enforce_stage_modules(model, stage_name=stage_name)
            logger.info(
                "stage1_freeze gate_only frozen_params=%d trainable_params=%d",
                int(stage1_train_stats["frozen_params"]),
                int(stage1_train_stats["trainable_params"]),
            )
        elif stage_name != prev_stage_name and stage_name == "stage2_encoder" and not stage2_prepared:
            if fixed_gate_threshold is not None and stage1_epochs > 0:
                fixed_summary = dict(best_stage1_summary or latest_gate_calibration)
                fixed_goal_reached = bool(
                    fixed_summary.get("fixed_gate_goal_reached", fixed_summary.get("goal_reached", False))
                )
                if not fixed_goal_reached:
                    stage2_prepared = True
                    stage2_fallback_reason = "fixed_gate_goal_not_reached"
                    logger.warning(
                        "stage2_skipped reason=%s fixed_gate_goal_reached=false "
                        "assignment_recall=%.4f noise_accuracy=%.4f detector_negative_noise_accuracy=%.4f "
                        "balanced_accuracy=%.4f fixed_gate_threshold=%.4f",
                        stage2_fallback_reason,
                        float(fixed_summary.get("gate_recall", float("nan"))),
                        float(fixed_summary.get("noise_accuracy", float("nan"))),
                        float(fixed_summary.get("detector_negative_noise_accuracy", float("nan"))),
                        float(fixed_summary.get("balanced_accuracy", float("nan"))),
                        float(fixed_gate_threshold),
                    )
                    break
            stage2_ready = prepare_stage2_state(restore_stage1_best=True, init_reason="stage1_gate_best")
            prev_stage_name = stage_name
            if not stage2_ready:
                break
        elif stage_name == "stage2_encoder":
            model.train()
            enforce_stage_modules(model, stage_name=stage_name)
        if stage_name == "stage2_encoder" and stage2_fallback_reason == "insufficient_gate_pass_support":
            logger.info("stage2_training_bypassed reason=%s", stage2_fallback_reason)
            break
        prev_stage_name = stage_name

        ep_start = time.perf_counter()
        losses: List[float] = []
        gate_losses: List[float] = []
        sup_losses: List[float] = []
        pro_losses: List[float] = []
        proto_ce_losses: List[float] = []
        ch_losses: List[float] = []
        umap_losses: List[float] = []
        supervised_graph_losses: List[float] = []
        boundary_losses: List[float] = []
        lambda_umap_values: List[float] = []
        lambda_supervised_graph_values: List[float] = []
        lambda_boundary_values: List[float] = []
        graph_random_neg_values: List[float] = []
        graph_hard_neg_values: List[float] = []
        graph_hard_frac_values: List[float] = []
        supervised_graph_pos_edge_values: List[float] = []
        supervised_graph_neg_edge_values: List[float] = []
        supervised_graph_hard_neg_values: List[float] = []
        supervised_graph_q_pos_values: List[float] = []
        supervised_graph_q_neg_values: List[float] = []
        episode_way_values: List[float] = []
        temporal_conn_loss_values: List[float] = []
        lambda_temporal_conn_values: List[float] = []
        temporal_conn_pair_values: List[float] = []
        temporal_conn_active_values: List[float] = []
        proto_metric_values: Dict[str, List[float]] = {key: [] for key in proto_metric_keys}
        encode_pass_rates: List[float] = []
        gate_source_epoch_counts = {source_name: 0 for source_name in GATE_SOURCE_ORDER}
        gate_recording_epoch_counts: Dict[int, int] = {}
        proto_ce_zero_query_steps = 0
        steps = 0
        epoch_seen = np.zeros((train_patches.shape[0],), dtype=np.bool_) if (stage_name == "stage2_encoder" and stage2_batch_sampler is not None) else None

        if stage_name == "stage1_gate":
            if gate_sampler is None:
                raise RuntimeError("stage-1 gate sampler was not initialized")
            for _ in range(stage1_steps_per_epoch):
                gate_idx = gate_sampler(stage_epoch - 1)
                if train_sample_source is not None and gate_idx.size > 0:
                    accumulate_gate_source_counts(gate_source_epoch_counts, train_sample_source[gate_idx])
                if gate_idx.size > 0:
                    accumulate_recording_counts(gate_recording_epoch_counts, train_recording_id[gate_idx])
                step_metrics = train_step(
                    np.zeros((0,), dtype=np.int64),
                    gate_idx,
                    use_proto_ce=False,
                    stage_name=stage_name,
                )
                losses.append(float(step_metrics["loss"]))
                gate_losses.append(float(step_metrics["gate"]))
                sup_losses.append(float(step_metrics["supcon"]))
                pro_losses.append(float(step_metrics["proto"]))
                proto_ce_losses.append(float(step_metrics["proto_ce"]))
                ch_losses.append(float(step_metrics["channel"]))
                umap_losses.append(float(step_metrics["umap_teacher_graph"]))
                supervised_graph_losses.append(float(step_metrics["supervised_graph"]))
                boundary_losses.append(float(step_metrics["boundary"]))
                lambda_umap_values.append(float(step_metrics["lambda_umap_eff"]))
                lambda_supervised_graph_values.append(float(step_metrics["lambda_supervised_graph_eff"]))
                lambda_boundary_values.append(float(step_metrics["lambda_boundary_eff"]))
                graph_random_neg_values.append(float(step_metrics["teacher_graph_random_neg_edges"]))
                graph_hard_neg_values.append(float(step_metrics["teacher_graph_hard_neg_edges"]))
                graph_hard_frac_values.append(float(step_metrics["teacher_graph_hard_neg_fraction"]))
                supervised_graph_pos_edge_values.append(float(step_metrics["supervised_graph_pos_edges"]))
                supervised_graph_neg_edge_values.append(float(step_metrics["supervised_graph_neg_edges"]))
                supervised_graph_hard_neg_values.append(float(step_metrics["supervised_graph_hard_neg_edges"]))
                supervised_graph_q_pos_values.append(float(step_metrics["supervised_graph_mean_q_pos"]))
                supervised_graph_q_neg_values.append(float(step_metrics["supervised_graph_mean_q_neg"]))
                episode_way_values.append(float(step_metrics["episode_way"]))
                temporal_conn_loss_values.append(float(step_metrics["temporal_conn_loss"]))
                lambda_temporal_conn_values.append(float(step_metrics["lambda_temporal_conn_eff"]))
                temporal_conn_pair_values.append(float(step_metrics["temporal_conn_pair_count"]))
                temporal_conn_active_values.append(float(step_metrics["temporal_conn_active_ratio"]))
                for key in proto_metric_keys:
                    proto_metric_values[key].append(float(step_metrics.get(key, 0.0)))
                encode_pass_rates.append(float(step_metrics["encode_pass"]))
                steps += 1
                global_step += 1
                if log_every_steps > 0 and steps % log_every_steps == 0:
                    logger.info(
                        "epoch=%d step=%d/%d stage=%s batch_loss=%.6f mean_loss=%.6f mean_gate=%.6f mean_supcon=%.6f mean_proto=%.6f mean_proto_ce=%.6f mean_channel=%.6f mean_encode_pass=%.6f",
                        ep + 1,
                        steps,
                        stage1_steps_per_epoch,
                        stage_name,
                        float(step_metrics["loss"]),
                        float(np.mean(losses)),
                        float(np.mean(gate_losses)),
                        float(np.mean(sup_losses)),
                        float(np.mean(pro_losses)),
                        float(np.mean(proto_ce_losses)),
                        float(np.mean(ch_losses)),
                        float(np.mean(encode_pass_rates)),
                    )
        else:
            if stage2_batch_sampler is not None:
                for _ in range(stage2_steps_per_epoch):
                    idx_batch = stage2_batch_sampler(stage_epoch - 1)
                    if epoch_seen is not None and idx_batch.size > 0:
                        epoch_seen[idx_batch] = True
                    step_metrics = train_step(
                        idx_batch,
                        np.zeros((0,), dtype=np.int64),
                        use_proto_ce=episodic_proto_ce,
                        stage_name=stage_name,
                    )
                    losses.append(float(step_metrics["loss"]))
                    gate_losses.append(float(step_metrics["gate"]))
                    sup_losses.append(float(step_metrics["supcon"]))
                    pro_losses.append(float(step_metrics["proto"]))
                    proto_ce_value = float(step_metrics["proto_ce"])
                    proto_ce_losses.append(proto_ce_value)
                    if episodic_proto_ce and proto_ce_value == 0.0:
                        proto_ce_zero_query_steps += 1
                    ch_losses.append(float(step_metrics["channel"]))
                    umap_losses.append(float(step_metrics["umap_teacher_graph"]))
                    supervised_graph_losses.append(float(step_metrics["supervised_graph"]))
                    boundary_losses.append(float(step_metrics["boundary"]))
                    lambda_umap_values.append(float(step_metrics["lambda_umap_eff"]))
                    lambda_supervised_graph_values.append(float(step_metrics["lambda_supervised_graph_eff"]))
                    lambda_boundary_values.append(float(step_metrics["lambda_boundary_eff"]))
                    graph_random_neg_values.append(float(step_metrics["teacher_graph_random_neg_edges"]))
                    graph_hard_neg_values.append(float(step_metrics["teacher_graph_hard_neg_edges"]))
                    graph_hard_frac_values.append(float(step_metrics["teacher_graph_hard_neg_fraction"]))
                    supervised_graph_pos_edge_values.append(float(step_metrics["supervised_graph_pos_edges"]))
                    supervised_graph_neg_edge_values.append(float(step_metrics["supervised_graph_neg_edges"]))
                    supervised_graph_hard_neg_values.append(float(step_metrics["supervised_graph_hard_neg_edges"]))
                    supervised_graph_q_pos_values.append(float(step_metrics["supervised_graph_mean_q_pos"]))
                    supervised_graph_q_neg_values.append(float(step_metrics["supervised_graph_mean_q_neg"]))
                    episode_way_values.append(float(step_metrics["episode_way"]))
                    temporal_conn_loss_values.append(float(step_metrics["temporal_conn_loss"]))
                    lambda_temporal_conn_values.append(float(step_metrics["lambda_temporal_conn_eff"]))
                    temporal_conn_pair_values.append(float(step_metrics["temporal_conn_pair_count"]))
                    temporal_conn_active_values.append(float(step_metrics["temporal_conn_active_ratio"]))
                    for key in proto_metric_keys:
                        proto_metric_values[key].append(float(step_metrics.get(key, 0.0)))
                    steps += 1
                    global_step += 1
                    if log_every_steps > 0 and steps % log_every_steps == 0:
                        logger.info(
                            "epoch=%d step=%d/%d stage=%s batch_loss=%.6f mean_loss=%.6f mean_supcon=%.6f mean_proto=%.6f mean_proto_ce=%.6f mean_supervised_graph=%.6f mean_channel=%.6f mean_graph_random_neg=%.2f mean_graph_hard_neg=%.2f mean_graph_hard_frac=%.4f mean_supervised_graph_pos=%.2f mean_supervised_graph_neg=%.2f",
                            ep + 1,
                            steps,
                            stage2_steps_per_epoch,
                            stage_name,
                            float(step_metrics["loss"]),
                            float(np.mean(losses)),
                            float(np.mean(sup_losses)),
                            float(np.mean(pro_losses)),
                            float(np.mean(proto_ce_losses)),
                            float(np.mean(supervised_graph_losses)),
                            float(np.mean(ch_losses)),
                            float(np.mean(graph_random_neg_values)),
                            float(np.mean(graph_hard_neg_values)),
                            float(np.mean(graph_hard_frac_values)),
                            float(np.mean(supervised_graph_pos_edge_values)),
                            float(np.mean(supervised_graph_neg_edge_values)),
                        )
                if epoch_seen is not None and train_stage2_training_positive_idx.size > 0:
                    coverage = float(np.mean(epoch_seen[train_stage2_training_positive_idx].astype(np.float32)))
                    logger.info("epoch=%d stage=%s balanced_coverage=%.4f", ep + 1, stage_name, coverage)
            else:
                perm = np.random.permutation(train_stage2_training_positive_idx)
                for st in range(0, perm.shape[0], bs):
                    idx_batch = perm[st : st + bs]
                    step_metrics = train_step(
                        idx_batch,
                        np.zeros((0,), dtype=np.int64),
                        use_proto_ce=False,
                        stage_name=stage_name,
                    )
                    losses.append(float(step_metrics["loss"]))
                    gate_losses.append(float(step_metrics["gate"]))
                    sup_losses.append(float(step_metrics["supcon"]))
                    pro_losses.append(float(step_metrics["proto"]))
                    proto_ce_losses.append(float(step_metrics["proto_ce"]))
                    ch_losses.append(float(step_metrics["channel"]))
                    umap_losses.append(float(step_metrics["umap_teacher_graph"]))
                    supervised_graph_losses.append(float(step_metrics["supervised_graph"]))
                    boundary_losses.append(float(step_metrics["boundary"]))
                    lambda_umap_values.append(float(step_metrics["lambda_umap_eff"]))
                    lambda_supervised_graph_values.append(float(step_metrics["lambda_supervised_graph_eff"]))
                    lambda_boundary_values.append(float(step_metrics["lambda_boundary_eff"]))
                    graph_random_neg_values.append(float(step_metrics["teacher_graph_random_neg_edges"]))
                    graph_hard_neg_values.append(float(step_metrics["teacher_graph_hard_neg_edges"]))
                    graph_hard_frac_values.append(float(step_metrics["teacher_graph_hard_neg_fraction"]))
                    supervised_graph_pos_edge_values.append(float(step_metrics["supervised_graph_pos_edges"]))
                    supervised_graph_neg_edge_values.append(float(step_metrics["supervised_graph_neg_edges"]))
                    supervised_graph_hard_neg_values.append(float(step_metrics["supervised_graph_hard_neg_edges"]))
                    supervised_graph_q_pos_values.append(float(step_metrics["supervised_graph_mean_q_pos"]))
                    supervised_graph_q_neg_values.append(float(step_metrics["supervised_graph_mean_q_neg"]))
                    episode_way_values.append(float(step_metrics["episode_way"]))
                    temporal_conn_loss_values.append(float(step_metrics["temporal_conn_loss"]))
                    lambda_temporal_conn_values.append(float(step_metrics["lambda_temporal_conn_eff"]))
                    temporal_conn_pair_values.append(float(step_metrics["temporal_conn_pair_count"]))
                    temporal_conn_active_values.append(float(step_metrics["temporal_conn_active_ratio"]))
                    for key in proto_metric_keys:
                        proto_metric_values[key].append(float(step_metrics.get(key, 0.0)))
                    steps += 1
                    global_step += 1
                    if log_every_steps > 0 and steps % log_every_steps == 0:
                        logger.info(
                            "epoch=%d step=%d/%d stage=%s batch_loss=%.6f mean_loss=%.6f mean_supcon=%.6f mean_proto=%.6f mean_proto_ce=%.6f mean_supervised_graph=%.6f mean_channel=%.6f mean_graph_random_neg=%.2f mean_graph_hard_neg=%.2f mean_graph_hard_frac=%.4f mean_supervised_graph_pos=%.2f mean_supervised_graph_neg=%.2f",
                            ep + 1,
                            steps,
                            max(1, int(math.ceil(max(1, perm.shape[0]) / max(1, bs)))),
                            stage_name,
                            float(step_metrics["loss"]),
                            float(np.mean(losses)),
                            float(np.mean(sup_losses)),
                            float(np.mean(pro_losses)),
                            float(np.mean(proto_ce_losses)),
                            float(np.mean(supervised_graph_losses)),
                            float(np.mean(ch_losses)),
                            float(np.mean(graph_random_neg_values)),
                            float(np.mean(graph_hard_neg_values)),
                            float(np.mean(graph_hard_frac_values)),
                            float(np.mean(supervised_graph_pos_edge_values)),
                            float(np.mean(supervised_graph_neg_edge_values)),
                        )

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_gate = float(np.mean(gate_losses)) if gate_losses else float("nan")
        epoch_sup = float(np.mean(sup_losses)) if sup_losses else float("nan")
        epoch_pro = float(np.mean(pro_losses)) if pro_losses else float("nan")
        epoch_proto_ce = float(np.mean(proto_ce_losses)) if proto_ce_losses else float("nan")
        epoch_ch = float(np.mean(ch_losses)) if ch_losses else float("nan")
        epoch_umap = float(np.mean(umap_losses)) if umap_losses else float("nan")
        epoch_supervised_graph = float(np.mean(supervised_graph_losses)) if supervised_graph_losses else float("nan")
        epoch_boundary = float(np.mean(boundary_losses)) if boundary_losses else float("nan")
        epoch_lambda_umap = float(np.mean(lambda_umap_values)) if lambda_umap_values else float("nan")
        epoch_lambda_supervised_graph = float(np.mean(lambda_supervised_graph_values)) if lambda_supervised_graph_values else float("nan")
        epoch_lambda_boundary = float(np.mean(lambda_boundary_values)) if lambda_boundary_values else float("nan")
        epoch_graph_random_neg = float(np.mean(graph_random_neg_values)) if graph_random_neg_values else float("nan")
        epoch_graph_hard_neg = float(np.mean(graph_hard_neg_values)) if graph_hard_neg_values else float("nan")
        epoch_graph_hard_frac = float(np.mean(graph_hard_frac_values)) if graph_hard_frac_values else float("nan")
        epoch_supervised_graph_pos = float(np.mean(supervised_graph_pos_edge_values)) if supervised_graph_pos_edge_values else float("nan")
        epoch_supervised_graph_neg = float(np.mean(supervised_graph_neg_edge_values)) if supervised_graph_neg_edge_values else float("nan")
        epoch_supervised_graph_hard_neg = float(np.mean(supervised_graph_hard_neg_values)) if supervised_graph_hard_neg_values else float("nan")
        epoch_supervised_graph_q_pos = float(np.mean(supervised_graph_q_pos_values)) if supervised_graph_q_pos_values else float("nan")
        epoch_supervised_graph_q_neg = float(np.mean(supervised_graph_q_neg_values)) if supervised_graph_q_neg_values else float("nan")
        epoch_episode_way = float(np.mean(episode_way_values)) if episode_way_values else float("nan")
        epoch_temporal_conn_loss = float(np.mean(temporal_conn_loss_values)) if temporal_conn_loss_values else float("nan")
        epoch_lambda_temporal_conn = float(np.mean(lambda_temporal_conn_values)) if lambda_temporal_conn_values else float("nan")
        epoch_temporal_conn_pairs = float(np.mean(temporal_conn_pair_values)) if temporal_conn_pair_values else float("nan")
        epoch_temporal_conn_active = float(np.mean(temporal_conn_active_values)) if temporal_conn_active_values else float("nan")
        epoch_encode_pass = float(np.mean(encode_pass_rates)) if encode_pass_rates else float("nan")
        epoch_proto_metrics = {
            key: (float(np.mean(values)) if values else float("nan"))
            for key, values in proto_metric_values.items()
        }
        epoch_time = time.perf_counter() - ep_start
        lr_now = float(opt.param_groups[0]["lr"])
        logger.info(
            "epoch=%d/%d stage=%s stage_epoch=%d/%d loss=%.6f gate=%.6f supcon=%.6f proto=%.6f proto_ce=%.6f umap=%.6f supervised_graph=%.6f boundary=%.6f channel=%.6f lambda_umap_eff=%.6f lambda_supervised_graph_eff=%.6f lambda_boundary_eff=%.6f graph_random_neg=%.2f graph_hard_neg=%.2f graph_hard_frac=%.4f supervised_graph_pos=%.2f supervised_graph_neg=%.2f supervised_graph_hard_neg=%.2f supervised_graph_q_pos=%.4f supervised_graph_q_neg=%.4f episode_way=%.2f encode_pass=%.6f lr=%.6g seconds=%.2f",
            ep + 1,
            total_epochs,
            stage_name,
            stage_epoch,
            stage_total,
            epoch_loss,
            epoch_gate,
            epoch_sup,
            epoch_pro,
            epoch_proto_ce,
            epoch_umap,
            epoch_supervised_graph,
            epoch_boundary,
            epoch_ch,
            epoch_lambda_umap,
            epoch_lambda_supervised_graph,
            epoch_lambda_boundary,
            epoch_graph_random_neg,
            epoch_graph_hard_neg,
            epoch_graph_hard_frac,
            epoch_supervised_graph_pos,
            epoch_supervised_graph_neg,
            epoch_supervised_graph_hard_neg,
            epoch_supervised_graph_q_pos,
            epoch_supervised_graph_q_neg,
            epoch_episode_way,
            epoch_encode_pass,
            lr_now,
            epoch_time,
        )
        if stage_name == "stage2_encoder":
            logger.info(
                "epoch=%d/%d stage=%s proto_diag mean_proto_per_anchor_unit=%.4f proto_count=%.2f anchor_units=%.2f query_units=%.2f split_attempts=%.2f split_accepts=%.2f reject_low_residual=%.2f reject_small_child=%.2f reject_low_gain=%.2f reject_child_close=%.2f reject_child_far=%.2f residual_p90_before=%.4f residual_p90_after=%.4f split_gain=%.4f units_p1=%.2f units_p2=%.2f units_p3=%.2f units_p4=%.2f nce=%.6f same_loss=%.6f chain_loss=%.6f same_cos_mean=%.4f same_cos_min=%.4f chain_cos_mean=%.4f chain_cos_min=%.4f warmup=%.2f temporal_fallback=%.2f",
                ep + 1,
                total_epochs,
                stage_name,
                epoch_proto_metrics["mean_proto_per_anchor_unit"],
                epoch_proto_metrics["local_proto_count"],
                epoch_proto_metrics["local_proto_anchor_unit_count"],
                epoch_proto_metrics["local_proto_query_unit_count"],
                epoch_proto_metrics["local_proto_split_attempts"],
                epoch_proto_metrics["local_proto_split_accepts"],
                epoch_proto_metrics["local_proto_split_reject_low_residual"],
                epoch_proto_metrics["local_proto_split_reject_small_child"],
                epoch_proto_metrics["local_proto_split_reject_low_gain"],
                epoch_proto_metrics["local_proto_split_reject_child_close"],
                epoch_proto_metrics["local_proto_split_reject_child_far"],
                epoch_proto_metrics["local_proto_residual_p90_before"],
                epoch_proto_metrics["local_proto_residual_p90_after"],
                epoch_proto_metrics["local_proto_split_gain_mean"],
                epoch_proto_metrics["local_proto_units_p1"],
                epoch_proto_metrics["local_proto_units_p2"],
                epoch_proto_metrics["local_proto_units_p3"],
                epoch_proto_metrics["local_proto_units_p4"],
                epoch_proto_metrics["local_proto_nce_loss"],
                epoch_proto_metrics["local_proto_same_loss"],
                epoch_proto_metrics["local_proto_chain_loss"],
                epoch_proto_metrics["same_proto_cos_mean"],
                epoch_proto_metrics["same_proto_cos_min"],
                epoch_proto_metrics["temporal_chain_cos_mean"],
                epoch_proto_metrics["temporal_chain_cos_min"],
                epoch_proto_metrics["local_proto_warmup_active"],
                epoch_proto_metrics["local_proto_temporal_fallback"],
            )
            logger.info(
                "epoch=%d/%d stage=%s temporal_conn_diag loss=%.6f lambda_temporal_conn_eff=%.6f pairs=%.2f active=%.4f",
                ep + 1,
                total_epochs,
                stage_name,
                epoch_temporal_conn_loss,
                epoch_lambda_temporal_conn,
                epoch_temporal_conn_pairs,
                epoch_temporal_conn_active,
            )
        if stage_name == "stage1_gate":
            logger.info("epoch=%d stage=%s gate_batch_source_mix %s", ep + 1, stage_name, format_gate_source_mix(gate_source_epoch_counts))
            logger.info("epoch=%d stage=%s gate_batch_recording_mix %s", ep + 1, stage_name, format_recording_mix(gate_recording_epoch_counts))
        if episodic_proto_ce and stage_name == "stage2_encoder" and proto_ce_zero_query_steps > 0:
            logger.warning(
                "epoch=%d stage=%s proto_ce had zero valid queries on %d/%d steps",
                ep + 1,
                stage_name,
                proto_ce_zero_query_steps,
                max(1, steps),
            )
        if not np.isfinite(epoch_loss):
            logger.warning("non-finite loss detected at epoch=%d stage=%s", ep + 1, stage_name)

        should_validate = bool(
            val_enabled
            and val_patches.shape[0] > 0
            and (((ep + 1) % val_every_epochs == 0) or ((ep + 1) == total_epochs))
        )
        if should_validate:
            t_val = time.perf_counter()
            gate_eval = run_gate_validation(
                model,
                val_patches,
                val_geo_all,
                val_spike_label,
                spatial_stats_np=val_spatial_stats,
                channel_mask_np=val_channel_mask_all,
                device=device,
                batch_size=val_batch_size,
                fixed_gate_threshold=fixed_gate_threshold,
                gate_goal_spike_recall=gate_goal_spike_recall,
                gate_goal_noise_accuracy=gate_goal_noise_accuracy,
                gate_target_recall_encode=gate_target_recall_encode,
                gate_target_recall_assignment_min=gate_target_recall_assignment_min,
                domain_id=val_recording_id,
                domain_type=val_domain_type,
                eval_mask=val_eval_mask,
                sample_source=val_sample_source,
                gate_log_source_metrics=gate_log_source_metrics,
            )
            latest_gate_calibration = dict(gate_eval["calibration"])
            encode_metrics = dict(gate_eval["encode_metrics"])
            assignment_metrics = dict(gate_eval["assignment_metrics"])
            source_metrics = dict(gate_eval["source_metrics"])
            gate_recall = float(assignment_metrics.get("gate_recall", float("nan")))
            gate_precision = float(assignment_metrics.get("gate_precision", float("nan")))
            gate_f1 = float(assignment_metrics.get("gate_f1", float("nan")))
            noise_accuracy = float(assignment_metrics.get("noise_accuracy", float("nan")))
            balanced_accuracy = float(assignment_metrics.get("balanced_accuracy", float("nan")))
            secondary_matched_positive_recall = float(latest_gate_calibration.get("secondary_matched_positive_recall", float("nan")))
            detector_negative_noise_accuracy = float(
                latest_gate_calibration.get("detector_negative_noise_accuracy", float("nan"))
            )
            detector_domain_balanced_accuracy = float(
                latest_gate_calibration.get("detector_domain_balanced_accuracy", float("nan"))
            )
            if (
                not np.isfinite(gate_recall)
                or not np.isfinite(gate_precision)
                or not np.isfinite(gate_f1)
                or not np.isfinite(noise_accuracy)
                or not np.isfinite(balanced_accuracy)
            ):
                raise RuntimeError(
                    f"non-finite gate validation metrics at epoch={ep + 1}: "
                    f"gate_recall={gate_recall} gate_precision={gate_precision} gate_f1={gate_f1} "
                    f"noise_accuracy={noise_accuracy} balanced_accuracy={balanced_accuracy}"
                )
            if stage_name == "stage1_gate":
                stage1_reference_metrics = build_stage2_metric_view({}, scope="stage1_reference")
                if (
                    stage1_encoder_aux_enabled
                    and train_stage1_reference_allowed_units.size > 0
                    and train_stage1_reference_positive_idx.size > 0
                ):
                    stage1_reference_raw = evaluate_stage2_encoder_metrics(
                        model,
                        train_patches,
                        train_unit_id,
                        train_geo_all,
                        train_recording_id,
                        train_spike_label,
                        train_sample_source,
                        val_patches,
                        val_unit_id,
                        val_geo_all,
                        val_recording_id,
                        val_spike_label,
                        val_sample_source,
                        allowed_units=train_stage1_reference_allowed_units,
                        device=device,
                        seed=val_seed,
                        support_per_class=val_support_per_class,
                        max_eval_samples=val_max_eval_samples,
                        eval_batch_size=val_batch_size,
                        pair_samples=val_pair_samples,
                        val_event_time_sample=val_event_time_sample,
                        cluster_enabled=val_cluster_enabled,
                        cluster_min_samples_per_unit=val_cluster_min_samples_per_unit,
                        cluster_max_samples_per_unit=val_cluster_max_samples_per_unit,
                        cluster_max_samples_per_recording=val_cluster_max_samples_per_recording,
                    )
                    stage1_reference_metrics = build_stage2_metric_view(
                        stage1_reference_raw,
                        scope="stage1_reference",
                    )
                logger.info(
                    (
                        "val epoch=%d/%d stage=%s assignment_recall=%.4f assignment_precision=%.4f assignment_f1=%.4f "
                        "noise_accuracy=%.4f balanced_accuracy=%.4f encode_recall=%.4f encode_pass=%.4f "
                        "secondary_matched_positive_recall=%.4f detector_negative_noise_accuracy=%.4f "
                        "detector_domain_balanced_accuracy=%.4f encode_thresh=%.4f assignment_thresh=%.4f val_sec=%.2f"
                    ),
                    ep + 1,
                    total_epochs,
                    stage_name,
                    gate_recall,
                    gate_precision,
                    gate_f1,
                    noise_accuracy,
                    balanced_accuracy,
                    float(encode_metrics.get("gate_recall", float("nan"))),
                    float(encode_metrics.get("encode_pass_ratio", float("nan"))),
                    secondary_matched_positive_recall,
                    detector_negative_noise_accuracy,
                    detector_domain_balanced_accuracy,
                    float(latest_gate_calibration.get("encode_gate_thresh", float("nan"))),
                    float(latest_gate_calibration.get("assignment_gate_thresh", float("nan"))),
                    float(time.perf_counter() - t_val),
                )
                for line in format_domain_gate_metrics(dict(latest_gate_calibration.get("domain_metrics", {}))):
                    logger.info("val_gate_domain epoch=%d stage=%s %s", ep + 1, stage_name, line)
                for line in format_source_gate_metrics(source_metrics):
                    logger.info("val_gate_source epoch=%d stage=%s %s", ep + 1, stage_name, line)
                logger.info(
                    (
                        "stage1 reference encoder metrics epoch=%d/%d stage=%s lsm10=%.4f "
                        "ber002=%.4f local_silhouette10=%.4f "
                        "recordings=%d units=%d samples=%d"
                    ),
                    ep + 1,
                    total_epochs,
                    stage_name,
                    float(stage1_reference_metrics["val_lsm10_stage1_reference_gt"]),
                    float(stage1_reference_metrics["val_ber002_stage1_reference_gt"]),
                    float(stage1_reference_metrics["val_local_silhouette10_stage1_reference_gt"]),
                    int(stage1_reference_metrics["val_recording_task_recordings_stage1_reference_gt"]),
                    int(stage1_reference_metrics["val_recording_task_units_stage1_reference_gt"]),
                    int(stage1_reference_metrics["val_recording_task_samples_stage1_reference_gt"]),
                )
                current_rank = stage1_gate_model_rank(
                    gate_selection_summary(latest_gate_calibration),
                    stage1_reference_metrics,
                    goal_spike_recall=gate_goal_spike_recall,
                    goal_noise_accuracy=gate_goal_noise_accuracy,
                    selection_mode=stage1_gate_selection_mode,
                    acceptance_min_precision=stage1_acceptance_min_precision,
                    acceptance_min_recall=stage1_acceptance_min_recall,
                )
                if best_stage1_rank is None or current_rank > best_stage1_rank:
                    best_stage1_rank = current_rank
                    best_stage1_epoch = ep + 1
                    best_stage1_summary = dict(latest_gate_calibration)
                    best_stage1_reference_metrics = dict(stage1_reference_metrics)
                    best_stage1_model_state = clone_model_state_dict(model)
                    best_ckpt_path = best_stage1_checkpoint_path
                    save_training_checkpoint(
                        best_ckpt_path,
                        epoch=ep + 1,
                        global_step=global_step,
                        model=model,
                        optimizer=opt,
                        scheduler=scheduler,
                        metadata={
                            "encoder_checkpoint": dict(encoder_checkpoint_metadata),
                            "gate_calibration": dict(latest_gate_calibration),
                            "stage": stage_name,
                            "stage1_reference_metrics": dict(stage1_reference_metrics),
                            "stage1_selection_metrics": {
                                "primary": stage1_gate_selection_mode,
                                "acceptance_min_precision": float(stage1_acceptance_min_precision),
                                "acceptance_min_recall": float(stage1_acceptance_min_recall),
                            },
                        },
                    )
                    logger.info("stage1_best_checkpoint_saved path=%s", best_ckpt_path)
            else:
                stage2_reference_raw = evaluate_stage2_encoder_metrics(
                    model,
                    train_patches,
                    train_unit_id,
                    train_geo_all,
                    train_recording_id,
                    train_spike_label,
                    train_sample_source,
                    val_patches,
                    val_unit_id,
                    val_geo_all,
                    val_recording_id,
                    val_spike_label,
                    val_sample_source,
                    allowed_units=None,
                    device=device,
                    seed=val_seed,
                    support_per_class=val_support_per_class,
                    max_eval_samples=val_max_eval_samples,
                    eval_batch_size=val_batch_size,
                    pair_samples=val_pair_samples,
                    val_event_time_sample=val_event_time_sample,
                    cluster_enabled=val_cluster_enabled,
                    cluster_min_samples_per_unit=val_cluster_min_samples_per_unit,
                    cluster_max_samples_per_unit=val_cluster_max_samples_per_unit,
                    cluster_max_samples_per_recording=val_cluster_max_samples_per_recording,
                )
                stage2_reference_metrics = build_stage2_metric_view(stage2_reference_raw, scope="reference")
                stage2_operational_raw = evaluate_stage2_encoder_metrics(
                    model,
                    train_patches[train_stage2_operational_positive_idx],
                    train_unit_id[train_stage2_operational_positive_idx],
                    train_geo_all[train_stage2_operational_positive_idx],
                    train_recording_id[train_stage2_operational_positive_idx],
                    None,
                    None,
                    val_patches[val_stage2_operational_positive_idx],
                    val_unit_id[val_stage2_operational_positive_idx],
                    val_geo_all[val_stage2_operational_positive_idx],
                    val_recording_id[val_stage2_operational_positive_idx],
                    None,
                    None,
                    allowed_units=None,
                    device=device,
                    seed=val_seed,
                    support_per_class=val_support_per_class,
                    max_eval_samples=val_max_eval_samples,
                    eval_batch_size=val_batch_size,
                    pair_samples=val_pair_samples,
                    val_event_time_sample=val_event_time_sample[val_stage2_operational_positive_idx],
                    cluster_enabled=val_cluster_enabled,
                    cluster_min_samples_per_unit=val_cluster_min_samples_per_unit,
                    cluster_max_samples_per_unit=val_cluster_max_samples_per_unit,
                    cluster_max_samples_per_recording=val_cluster_max_samples_per_recording,
                )
                stage2_operational_metrics = build_stage2_metric_view(
                    stage2_operational_raw,
                    scope="operational",
                    train_samples=float(train_stage2_operational_positive_idx.size),
                    train_classes=float(train_stage2_operational_allowed_units.size),
                )
                stage2_mixed_metrics: Dict[str, float] = {}
                if (
                    stage2_source_mode != "primary_only"
                    and train_stage2_training_allowed_units.size > 0
                    and train_stage2_training_positive_idx.size > 0
                    and val_stage2_training_positive_idx.size > 0
                ):
                    stage2_mixed_raw = evaluate_stage2_encoder_metrics(
                        model,
                        train_patches[train_stage2_training_positive_idx],
                        train_unit_id[train_stage2_training_positive_idx],
                        train_geo_all[train_stage2_training_positive_idx],
                        train_recording_id[train_stage2_training_positive_idx],
                        None,
                        None,
                        val_patches[val_stage2_training_positive_idx],
                        val_unit_id[val_stage2_training_positive_idx],
                        val_geo_all[val_stage2_training_positive_idx],
                        val_recording_id[val_stage2_training_positive_idx],
                        None,
                        None,
                        allowed_units=None,
                        device=device,
                        seed=val_seed,
                        support_per_class=val_support_per_class,
                        max_eval_samples=val_max_eval_samples,
                        eval_batch_size=val_batch_size,
                        pair_samples=val_pair_samples,
                        val_event_time_sample=val_event_time_sample[val_stage2_training_positive_idx],
                        cluster_enabled=val_cluster_enabled,
                        cluster_min_samples_per_unit=val_cluster_min_samples_per_unit,
                        cluster_max_samples_per_unit=val_cluster_max_samples_per_unit,
                        cluster_max_samples_per_recording=val_cluster_max_samples_per_recording,
                    )
                    stage2_mixed_metrics = build_stage2_metric_view(
                        stage2_mixed_raw,
                        scope="operational_mixed",
                        train_samples=float(train_stage2_training_positive_idx.size),
                        train_classes=float(train_stage2_training_allowed_units.size),
                    )
                logger.info(
                    (
                        "val epoch=%d/%d stage=%s reference_lsm10=%.4f "
                        "reference_ber002=%.4f reference_local_silhouette10=%.4f "
                        "reference_recordings=%d reference_units=%d reference_samples=%d"
                    ),
                    ep + 1,
                    total_epochs,
                    stage_name,
                    float(stage2_reference_metrics["val_lsm10_reference_gt"]),
                    float(stage2_reference_metrics["val_ber002_reference_gt"]),
                    float(stage2_reference_metrics["val_local_silhouette10_reference_gt"]),
                    int(stage2_reference_metrics["val_recording_task_recordings_reference_gt"]),
                    int(stage2_reference_metrics["val_recording_task_units_reference_gt"]),
                    int(stage2_reference_metrics["val_recording_task_samples_reference_gt"]),
                )
                logger.info(
                    (
                        "val epoch=%d/%d stage=%s operational_lsm10=%.4f "
                        "operational_ber002=%.4f operational_local_silhouette10=%.4f "
                        "train_samples_operational=%d train_classes_operational=%d "
                        "val_recordings_operational=%d val_units_operational=%d val_samples_operational=%d "
                        "assignment_recall=%.4f assignment_f1=%.4f "
                        "noise_accuracy=%.4f secondary_matched_positive_recall=%.4f detector_negative_noise_accuracy=%.4f "
                        "detector_domain_balanced_accuracy=%.4f encode_thresh=%.4f assignment_thresh=%.4f val_sec=%.2f"
                    ),
                    ep + 1,
                    total_epochs,
                    stage_name,
                    float(stage2_operational_metrics["val_lsm10_operational_gt"]),
                    float(stage2_operational_metrics["val_ber002_operational_gt"]),
                    float(stage2_operational_metrics["val_local_silhouette10_operational_gt"]),
                    int(stage2_operational_metrics["train_samples_operational_gt"]),
                    int(stage2_operational_metrics["train_classes_operational_gt"]),
                    int(stage2_operational_metrics["val_recording_task_recordings_operational_gt"]),
                    int(stage2_operational_metrics["val_recording_task_units_operational_gt"]),
                    int(stage2_operational_metrics["val_recording_task_samples_operational_gt"]),
                    gate_recall,
                    gate_f1,
                    noise_accuracy,
                    secondary_matched_positive_recall,
                    detector_negative_noise_accuracy,
                    detector_domain_balanced_accuracy,
                    float(latest_gate_calibration.get("encode_gate_thresh", float("nan"))),
                    float(latest_gate_calibration.get("assignment_gate_thresh", float("nan"))),
                    float(time.perf_counter() - t_val),
                )
                if stage2_mixed_metrics:
                    logger.info(
                        (
                            "val epoch=%d/%d stage=%s operational_mixed_lsm10=%.4f "
                            "operational_mixed_ber002=%.4f operational_mixed_local_silhouette10=%.4f "
                            "train_samples_operational_mixed=%d train_classes_operational_mixed=%d "
                            "val_recordings_operational_mixed=%d val_units_operational_mixed=%d val_samples_operational_mixed=%d"
                        ),
                        ep + 1,
                        total_epochs,
                        stage_name,
                        float(stage2_mixed_metrics["val_lsm10_operational_mixed_gt"]),
                        float(stage2_mixed_metrics["val_ber002_operational_mixed_gt"]),
                        float(stage2_mixed_metrics["val_local_silhouette10_operational_mixed_gt"]),
                        int(stage2_mixed_metrics["train_samples_operational_mixed_gt"]),
                        int(stage2_mixed_metrics["train_classes_operational_mixed_gt"]),
                        int(stage2_mixed_metrics["val_recording_task_recordings_operational_mixed_gt"]),
                        int(stage2_mixed_metrics["val_recording_task_units_operational_mixed_gt"]),
                        int(stage2_mixed_metrics["val_recording_task_samples_operational_mixed_gt"]),
                    )
                for line in format_domain_gate_metrics(dict(latest_gate_calibration.get("domain_metrics", {}))):
                    logger.info("val_gate_domain epoch=%d stage=%s %s", ep + 1, stage_name, line)
                for line in format_source_gate_metrics(source_metrics):
                    logger.info("val_gate_source epoch=%d stage=%s %s", ep + 1, stage_name, line)
                operational_primary = float(stage2_operational_metrics.get(stage2_select_primary, float("nan")))
                recording_metric_tokens = (
                    "lsm",
                    "ber",
                    "silhouette",
                    "conductance",
                )
                operational_eval_samples_key = (
                    "val_cluster_samples_operational_gt"
                    if "_cluster_" in stage2_select_primary
                    else "val_recording_task_samples_operational_gt"
                    if any(token in stage2_select_primary for token in recording_metric_tokens)
                    else "val_holdout_samples_operational_gt"
                )
                operational_eval_samples = float(stage2_operational_metrics.get(operational_eval_samples_key, 0.0))
                operational_valid = bool(
                    np.isfinite(operational_primary)
                    and operational_eval_samples > 0.0
                    and float(stage2_operational_metrics.get("train_samples_operational_gt", 0.0)) > 0.0
                )
                if not operational_valid:
                    logger.warning(
                        "stage2_operational_eval_invalid epoch=%d primary=%s value=%.4f train_samples=%d eval_samples_key=%s eval_samples=%d",
                        ep + 1,
                        stage2_select_primary,
                        operational_primary,
                        int(stage2_operational_metrics.get("train_samples_operational_gt", 0.0)),
                        operational_eval_samples_key,
                        int(operational_eval_samples),
                    )
                else:
                    current_rank = stage2_encoder_model_rank(
                        stage2_operational_metrics,
                        primary_key=stage2_select_primary,
                        secondary_key=stage2_select_secondary,
                        tertiary_key=stage2_select_tertiary,
                    )
                    if best_stage2_rank is None or current_rank > best_stage2_rank:
                        best_stage2_rank = current_rank
                        best_stage2_epoch = ep + 1
                        best_stage2_metrics = dict(stage2_operational_metrics)
                        best_stage2_reference_metrics = dict(stage2_reference_metrics)
                        best_stage2_mixed_metrics = dict(stage2_mixed_metrics)
                        best_stage2_model_state = clone_model_state_dict(model)
                        save_training_checkpoint(
                            best_stage2_checkpoint_path,
                            epoch=ep + 1,
                            global_step=global_step,
                            model=model,
                            optimizer=opt,
                            scheduler=scheduler,
                            metadata={
                                "encoder_checkpoint": dict(encoder_checkpoint_metadata),
                                "gate_calibration": dict(latest_gate_calibration),
                                "stage": stage_name,
                                "stage2_operational_metrics": dict(stage2_operational_metrics),
                                "stage2_operational_mixed_metrics": dict(stage2_mixed_metrics),
                                "stage2_reference_metrics": dict(stage2_reference_metrics),
                                "stage2_gate_filter": dict(stage2_gate_filter_summary),
                                "stage2_selection_metrics": {
                                    "primary": stage2_select_primary,
                                    "secondary": stage2_select_secondary,
                                    "tertiary": stage2_select_tertiary,
                                },
                            },
                        )
                        logger.info(
                            "stage2_best_checkpoint_saved path=%s primary=%s secondary=%s tertiary=%s",
                            best_stage2_checkpoint_path,
                            stage2_select_primary,
                            stage2_select_secondary,
                            stage2_select_tertiary,
                        )

        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"pretrain_epoch_{ep + 1:04d}.ckpt"
            save_training_checkpoint(
                ckpt_path,
                epoch=ep + 1,
                global_step=global_step,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                metadata={
                    "encoder_checkpoint": dict(encoder_checkpoint_metadata),
                    "gate_calibration": dict(latest_gate_calibration),
                    "stage": stage_name,
                },
            )
            logger.info("checkpoint_saved path=%s", ckpt_path)

    if stage2_epochs > 0 and best_stage2_model_state is None:
        raise RuntimeError(f"No valid encoder checkpoint: {stage2_fallback_reason}")
    logger.info("stage=train_loop done")
    final_model_stage = "current"
    final_stage2_reference_metrics: Dict[str, float] = {}
    final_stage2_operational_metrics: Dict[str, float] = {}
    final_stage2_operational_mixed_metrics: Dict[str, float] = {}
    if best_stage2_model_state is not None:
        model.load_state_dict(best_stage2_model_state)
        final_model_stage = "stage2_best"
        logger.info(
            "restored_best_model stage=stage2_encoder epoch=%d lsm10_operational=%.4f ber002_operational=%.4f local_silhouette10_operational=%.4f",
            best_stage2_epoch,
            float((best_stage2_metrics or {}).get("val_lsm10_operational_gt", float("nan"))),
            float((best_stage2_metrics or {}).get("val_ber002_operational_gt", float("nan"))),
            float((best_stage2_metrics or {}).get("val_local_silhouette10_operational_gt", float("nan"))),
        )
    elif best_stage1_model_state is not None:
        model.load_state_dict(best_stage1_model_state)
        if best_stage1_summary is not None:
            latest_gate_calibration = dict(best_stage1_summary)
        final_model_stage = "stage1_best_fallback" if stage2_epochs > 0 else "stage1_best"
        if stage2_epochs > 0 and stage2_fallback_reason is None:
            stage2_fallback_reason = "no_valid_stage2_operational_checkpoint"
        logger.info(
            "restored_best_model stage=stage1_gate epoch=%d balanced_accuracy=%.4f spike_recall=%.4f noise_accuracy=%.4f",
            best_stage1_epoch,
            float(latest_gate_calibration.get("balanced_accuracy", float("nan"))),
            float(latest_gate_calibration.get("gate_recall", float("nan"))),
            float(latest_gate_calibration.get("noise_accuracy", float("nan"))),
        )

    if val_enabled and val_patches.shape[0] > 0:
        final_gate_eval = run_gate_validation(
            model,
            val_patches,
            val_geo_all,
            val_spike_label,
            spatial_stats_np=val_spatial_stats,
            channel_mask_np=val_channel_mask_all,
            device=device,
            batch_size=val_batch_size,
            fixed_gate_threshold=fixed_gate_threshold,
            gate_goal_spike_recall=gate_goal_spike_recall,
            gate_goal_noise_accuracy=gate_goal_noise_accuracy,
            gate_target_recall_encode=gate_target_recall_encode,
            gate_target_recall_assignment_min=gate_target_recall_assignment_min,
            domain_id=val_recording_id,
            domain_type=val_domain_type,
            eval_mask=val_eval_mask,
            sample_source=val_sample_source,
            gate_log_source_metrics=gate_log_source_metrics,
        )
        latest_gate_calibration = dict(final_gate_eval["calibration"])
        final_assignment_metrics = dict(final_gate_eval["assignment_metrics"])
        logger.info(
            "final_gate_calibration stage=%s assignment_recall=%.4f assignment_f1=%.4f noise_accuracy=%.4f "
            "secondary_matched_positive_recall=%.4f detector_negative_noise_accuracy=%.4f detector_domain_balanced_accuracy=%.4f "
            "encode_thresh=%.4f assignment_thresh=%.4f",
            final_model_stage,
            float(final_assignment_metrics.get("gate_recall", float("nan"))),
            float(final_assignment_metrics.get("gate_f1", float("nan"))),
            float(final_assignment_metrics.get("noise_accuracy", float("nan"))),
            float(latest_gate_calibration.get("secondary_matched_positive_recall", float("nan"))),
            float(latest_gate_calibration.get("detector_negative_noise_accuracy", float("nan"))),
            float(latest_gate_calibration.get("detector_domain_balanced_accuracy", float("nan"))),
            float(latest_gate_calibration.get("encode_gate_thresh", float("nan"))),
            float(latest_gate_calibration.get("assignment_gate_thresh", float("nan"))),
        )
        if stage2_epochs > 0 and train_stage2_reference_allowed_units.size > 0:
            final_reference_raw = evaluate_stage2_encoder_metrics(
                model,
                train_patches,
                train_unit_id,
                train_geo_all,
                train_recording_id,
                train_spike_label,
                train_sample_source,
                val_patches,
                val_unit_id,
                val_geo_all,
                val_recording_id,
                val_spike_label,
                val_sample_source,
                allowed_units=None,
                device=device,
                seed=val_seed,
                support_per_class=val_support_per_class,
                max_eval_samples=val_max_eval_samples,
                eval_batch_size=val_batch_size,
                pair_samples=val_pair_samples,
                val_event_time_sample=val_event_time_sample,
                cluster_enabled=val_cluster_enabled,
                cluster_min_samples_per_unit=val_cluster_min_samples_per_unit,
                cluster_max_samples_per_unit=val_cluster_max_samples_per_unit,
                cluster_max_samples_per_recording=val_cluster_max_samples_per_recording,
            )
            final_stage2_reference_metrics = build_stage2_metric_view(final_reference_raw, scope="reference")
            logger.info(
                "final_encoder_reference_metrics stage=%s lsm10=%.4f ber002=%.4f local_silhouette10=%.4f recordings=%d units=%d samples=%d",
                final_model_stage,
                float(final_stage2_reference_metrics.get("val_lsm10_reference_gt", float("nan"))),
                float(final_stage2_reference_metrics.get("val_ber002_reference_gt", float("nan"))),
                float(final_stage2_reference_metrics.get("val_local_silhouette10_reference_gt", float("nan"))),
                int(final_stage2_reference_metrics.get("val_recording_task_recordings_reference_gt", 0.0)),
                int(final_stage2_reference_metrics.get("val_recording_task_units_reference_gt", 0.0)),
                int(final_stage2_reference_metrics.get("val_recording_task_samples_reference_gt", 0.0)),
            )
        if stage2_epochs > 0 and train_stage2_operational_allowed_units.size > 0 and train_stage2_operational_positive_idx.size > 0 and val_stage2_operational_positive_idx.size > 0:
            final_operational_raw = evaluate_stage2_encoder_metrics(
                model,
                train_patches[train_stage2_operational_positive_idx],
                train_unit_id[train_stage2_operational_positive_idx],
                train_geo_all[train_stage2_operational_positive_idx],
                train_recording_id[train_stage2_operational_positive_idx],
                None,
                None,
                val_patches[val_stage2_operational_positive_idx],
                val_unit_id[val_stage2_operational_positive_idx],
                val_geo_all[val_stage2_operational_positive_idx],
                val_recording_id[val_stage2_operational_positive_idx],
                None,
                None,
                allowed_units=None,
                device=device,
                seed=val_seed,
                support_per_class=val_support_per_class,
                max_eval_samples=val_max_eval_samples,
                eval_batch_size=val_batch_size,
                pair_samples=val_pair_samples,
                val_event_time_sample=val_event_time_sample[val_stage2_operational_positive_idx],
                cluster_enabled=val_cluster_enabled,
                cluster_min_samples_per_unit=val_cluster_min_samples_per_unit,
                cluster_max_samples_per_unit=val_cluster_max_samples_per_unit,
                cluster_max_samples_per_recording=val_cluster_max_samples_per_recording,
            )
            final_stage2_operational_metrics = build_stage2_metric_view(
                final_operational_raw,
                scope="operational",
                train_samples=float(train_stage2_operational_positive_idx.size),
                train_classes=float(train_stage2_operational_allowed_units.size),
            )
            logger.info(
                "final_encoder_operational_metrics stage=%s lsm10=%.4f ber002=%.4f local_silhouette10=%.4f train_samples=%d val_recordings=%d val_units=%d val_samples=%d",
                final_model_stage,
                float(final_stage2_operational_metrics.get("val_lsm10_operational_gt", float("nan"))),
                float(final_stage2_operational_metrics.get("val_ber002_operational_gt", float("nan"))),
                float(final_stage2_operational_metrics.get("val_local_silhouette10_operational_gt", float("nan"))),
                int(final_stage2_operational_metrics.get("train_samples_operational_gt", 0.0)),
                int(final_stage2_operational_metrics.get("val_recording_task_recordings_operational_gt", 0.0)),
                int(final_stage2_operational_metrics.get("val_recording_task_units_operational_gt", 0.0)),
                int(final_stage2_operational_metrics.get("val_recording_task_samples_operational_gt", 0.0)),
            )
        elif stage2_epochs > 0:
            logger.info(
                "final_encoder_operational_metrics stage=%s unavailable reason=%s",
                final_model_stage,
                stage2_fallback_reason or "no_operational_subset",
            )
        if (
            stage2_epochs > 0
            and stage2_source_mode != "primary_only"
            and train_stage2_training_allowed_units.size > 0
            and train_stage2_training_positive_idx.size > 0
            and val_stage2_training_positive_idx.size > 0
        ):
            final_operational_mixed_raw = evaluate_stage2_encoder_metrics(
                model,
                train_patches[train_stage2_training_positive_idx],
                train_unit_id[train_stage2_training_positive_idx],
                train_geo_all[train_stage2_training_positive_idx],
                train_recording_id[train_stage2_training_positive_idx],
                None,
                None,
                val_patches[val_stage2_training_positive_idx],
                val_unit_id[val_stage2_training_positive_idx],
                val_geo_all[val_stage2_training_positive_idx],
                val_recording_id[val_stage2_training_positive_idx],
                None,
                None,
                allowed_units=None,
                device=device,
                seed=val_seed,
                support_per_class=val_support_per_class,
                max_eval_samples=val_max_eval_samples,
                eval_batch_size=val_batch_size,
                pair_samples=val_pair_samples,
                val_event_time_sample=val_event_time_sample[val_stage2_training_positive_idx],
                cluster_enabled=val_cluster_enabled,
                cluster_min_samples_per_unit=val_cluster_min_samples_per_unit,
                cluster_max_samples_per_unit=val_cluster_max_samples_per_unit,
                cluster_max_samples_per_recording=val_cluster_max_samples_per_recording,
            )
            final_stage2_operational_mixed_metrics = build_stage2_metric_view(
                final_operational_mixed_raw,
                scope="operational_mixed",
                train_samples=float(train_stage2_training_positive_idx.size),
                train_classes=float(train_stage2_training_allowed_units.size),
            )
            logger.info(
                "final_encoder_operational_mixed_metrics stage=%s lsm10=%.4f ber002=%.4f local_silhouette10=%.4f train_samples=%d val_recordings=%d val_units=%d val_samples=%d",
                final_model_stage,
                float(final_stage2_operational_mixed_metrics.get("val_lsm10_operational_mixed_gt", float("nan"))),
                float(final_stage2_operational_mixed_metrics.get("val_ber002_operational_mixed_gt", float("nan"))),
                float(final_stage2_operational_mixed_metrics.get("val_local_silhouette10_operational_mixed_gt", float("nan"))),
                int(final_stage2_operational_mixed_metrics.get("train_samples_operational_mixed_gt", 0.0)),
                int(final_stage2_operational_mixed_metrics.get("val_recording_task_recordings_operational_mixed_gt", 0.0)),
                int(final_stage2_operational_mixed_metrics.get("val_recording_task_units_operational_mixed_gt", 0.0)),
                int(final_stage2_operational_mixed_metrics.get("val_recording_task_samples_operational_mixed_gt", 0.0)),
            )

    save_path = Path(save_out)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("stage=save_model start")
    torch.save(model.state_dict(), save_path)
    sidecar_path = write_gate_metadata_sidecar(
        save_path,
        gate_calibration=dict(latest_gate_calibration),
        metadata={
            "encoder_checkpoint": dict(encoder_checkpoint_metadata),
            "weights_path": str(save_path),
            "generated_by": "scripts/pretrain.py",
            "final_model_stage": final_model_stage,
            "stage2_gate_filter": dict(stage2_gate_filter_summary),
            "stage2_selection_metrics": {
                "primary": stage2_select_primary,
                "secondary": stage2_select_secondary,
                "tertiary": stage2_select_tertiary,
            },
            "stage2_reference_metrics": dict(final_stage2_reference_metrics or (best_stage2_reference_metrics or {})),
            "stage2_operational_metrics": dict(final_stage2_operational_metrics or (best_stage2_metrics or {})),
            "stage2_operational_mixed_metrics": dict(
                final_stage2_operational_mixed_metrics or (best_stage2_mixed_metrics or {})
            ),
            "stage2_fallback_reason": stage2_fallback_reason,
        },
    )
    logger.info(
        "gate_calibration_saved path=%s encode_gate_thresh=%.4f assignment_gate_thresh=%.4f gate_f1=%.4f",
        sidecar_path,
        float(latest_gate_calibration["encode_gate_thresh"]),
        float(latest_gate_calibration["assignment_gate_thresh"]),
        float(latest_gate_calibration.get("gate_f1", float("nan"))),
    )
    logger.info("stage=save_model done")
    total_sec = time.perf_counter() - total_start
    logger.info("saved=%s", save_path)
    logger.info("done total_seconds=%.2f", total_sec)
    logger.info("log_file=%s", log_path)


if __name__ == "__main__":
    main()
