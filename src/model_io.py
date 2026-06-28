from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib

import torch
import yaml

from src.nn.model import ModelConfig, _normalize_encoder_route


def load_torch_file(path: str | Path) -> Any:
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def gate_metadata_sidecar_path(weights_path: str | Path) -> Path:
    weights_path = Path(weights_path)
    return weights_path.with_name(weights_path.name + ".meta.yaml")


def extract_gate_calibration(payload: Any) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    direct = payload.get("gate_calibration")
    if isinstance(direct, dict):
        return dict(direct)
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        gate = meta.get("gate_calibration")
        if isinstance(gate, dict):
            return dict(gate)
    return None


def load_saved_gate_calibration(weights_path: str | Path, payload: Any | None = None) -> Dict[str, Any] | None:
    weights_path = Path(weights_path)
    gate = extract_gate_calibration(payload)
    if gate is not None:
        return gate

    sidecar_path = gate_metadata_sidecar_path(weights_path)
    if not sidecar_path.exists():
        return None
    with open(sidecar_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    if isinstance(meta.get("gate_calibration"), dict):
        return dict(meta["gate_calibration"])
    if "encode_gate_thresh" in meta and "assignment_gate_thresh" in meta:
        return dict(meta)
    return None


def write_gate_metadata_sidecar(
    weights_path: str | Path,
    *,
    gate_calibration: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> Path:
    sidecar_path = gate_metadata_sidecar_path(weights_path)
    payload: Dict[str, Any] = {}
    if metadata:
        payload.update(metadata)
    payload["gate_calibration"] = dict(gate_calibration)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
    return sidecar_path


def build_model_config(
    model_cfg: Dict[str, Any],
    *,
    embed_dim: int | None = None,
    hidden_dim: int | None = None,
    encode_gate_thresh: float | None = None,
    assignment_gate_thresh: float | None = None,
) -> ModelConfig:
    return ModelConfig(
        embed_dim=int(model_cfg.get("embed_dim", 64) if embed_dim is None else embed_dim),
        hidden_dim=int(model_cfg.get("hidden_dim", 64) if hidden_dim is None else hidden_dim),
        n_blocks=int(model_cfg.get("n_blocks", 4)),
        shared_stem_layers=int(model_cfg.get("shared_stem_layers", 2)),
        gate_stem_layers=int(model_cfg.get("gate_stem_layers", 2)),
        gate_hidden_dim=int(model_cfg.get("gate_hidden_dim", 64)),
        gate_n_blocks=int(model_cfg.get("gate_n_blocks", 1)),
        gate_init_bias=float(model_cfg.get("gate_init_bias", 0.0)),
        encode_gate_thresh=float(model_cfg.get("encode_gate_thresh", 0.2) if encode_gate_thresh is None else encode_gate_thresh),
        assignment_gate_thresh=float(
            model_cfg.get("assignment_gate_thresh", 0.1)
            if assignment_gate_thresh is None
            else assignment_gate_thresh
        ),
        fixed_gate_threshold=(
            None if model_cfg.get("fixed_gate_threshold") is None else float(model_cfg.get("fixed_gate_threshold"))
        ),
        gate_pos_weight_scale=float(model_cfg.get("gate_pos_weight_scale", 1.0)),
        gate_use_deep=bool(model_cfg.get("gate_use_deep", False)),
        gate_spatial_dim=int(model_cfg.get("gate_spatial_dim", 5)),
        gate_shared_backbone=bool(model_cfg.get("gate_shared_backbone", False)),
        encoder_route=_normalize_encoder_route(str(model_cfg.get("encoder_route", "peak_only"))),
        spatial_pool_hidden_dim=int(model_cfg.get("spatial_pool_hidden_dim", 64)),
        spatial_gamma_init=float(model_cfg.get("spatial_gamma_init", 0.05)),
        footprint_stats_dim=int(model_cfg.get("footprint_stats_dim", 4)),
    )


def electrode_config_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract_checkpoint_metadata(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        return dict(meta)
    return {}


def extract_encoder_checkpoint_metadata(payload: Any) -> Dict[str, Any]:
    meta = extract_checkpoint_metadata(payload)
    out = dict(meta.get("encoder_checkpoint", {}) or {}) if isinstance(meta.get("encoder_checkpoint"), dict) else {}
    for key in (
        "encoder_route",
        "train_patch_k",
        "requires_dense_electrode",
        "electrode_config_name",
        "electrode_config_hash",
    ):
        if key in meta and key not in out:
            out[key] = meta[key]
    return out


def load_encoder_checkpoint_metadata(weights_path: str | Path | None, payload: Any | None = None) -> Dict[str, Any]:
    out = extract_encoder_checkpoint_metadata(payload)
    if out:
        return out
    if weights_path is None:
        return {}
    sidecar_path = gate_metadata_sidecar_path(weights_path)
    if not sidecar_path.exists():
        return {}
    with open(sidecar_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    if not isinstance(meta, dict):
        return {}
    enc = meta.get("encoder_checkpoint")
    if isinstance(enc, dict):
        return dict(enc)
    return {
        key: meta[key]
        for key in (
            "encoder_route",
            "train_patch_k",
            "requires_dense_electrode",
            "electrode_config_name",
            "electrode_config_hash",
        )
        if key in meta
    }


def infer_encoder_route_from_payload(payload: Any, model_cfg: Dict[str, Any]) -> str:
    ckpt_meta = extract_encoder_checkpoint_metadata(payload)
    route = ckpt_meta.get("encoder_route", model_cfg.get("encoder_route", "peak_only"))
    return _normalize_encoder_route(str(route or "peak_only"))


def validate_or_fallback_encoder_checkpoint(
    *,
    requested_weights_path: Path | None,
    requested_payload: Any | None,
    model_cfg: Dict[str, Any],
    resolved_patch_k: int,
    electrode_config_path: Path | None,
    fallback_peak_only_weights: Path | None,
) -> Tuple[Path | None, Any | None, Dict[str, Any]]:
    ckpt_meta = load_encoder_checkpoint_metadata(requested_weights_path, requested_payload)
    route = _normalize_encoder_route(
        str(ckpt_meta.get("encoder_route") or infer_encoder_route_from_payload(requested_payload, model_cfg))
    )
    info: Dict[str, Any] = {
        "requested_weights_path": None if requested_weights_path is None else str(requested_weights_path),
        "resolved_weights_path": None if requested_weights_path is None else str(requested_weights_path),
        "checkpoint_encoder_route": route,
        "resolved_patch_k": int(resolved_patch_k),
        "fallback_used": False,
        "fallback_reason": "",
        "checkpoint_metadata": ckpt_meta,
    }
    _ = fallback_peak_only_weights
    resolved_k = int(resolved_patch_k)
    train_patch_k_raw = ckpt_meta.get("train_patch_k")
    train_patch_k = None if train_patch_k_raw is None else int(train_patch_k_raw)
    if requested_weights_path is not None and route == "peak_only" and resolved_k != 1:
        raise RuntimeError(f"recording-as-task peak-only checkpoints require resolved K=1, got K={int(resolved_patch_k)}")
    if requested_weights_path is not None and route != "peak_only":
        if train_patch_k is not None and resolved_k != int(train_patch_k):
            raise RuntimeError(
                f"{route} checkpoint was trained with K={int(train_patch_k)}, got resolved K={resolved_k}"
            )
        if resolved_k <= 1:
            raise RuntimeError(f"{route} checkpoints require multi-channel patches, got resolved K={resolved_k}")
        if bool(ckpt_meta.get("requires_dense_electrode", False)) and electrode_config_path is None:
            raise RuntimeError(f"{route} checkpoint requires a resolved dense electrode_config")
    return requested_weights_path, requested_payload, info


def normalize_runtime_gate_thresholds(
    *,
    encode_gate_thresh: float,
    assignment_gate_thresh: float,
    novelty_birth_gate_min: float,
    keep_gate_min: float,
    init_seed_gate: float,
) -> Tuple[Dict[str, float], List[str]]:
    values = {
        "encode_gate_thresh": float(encode_gate_thresh),
        "assignment_gate_thresh": float(assignment_gate_thresh),
        "novelty_birth_gate_min": float(novelty_birth_gate_min),
        "keep_gate_min": float(keep_gate_min),
        "init_seed_gate": float(init_seed_gate),
    }
    notes: List[str] = []
    if values["encode_gate_thresh"] <= 0.0 or values["assignment_gate_thresh"] <= 0.0:
        raise ValueError(
            "main online/offline path requires positive gate thresholds; "
            f"encode_gate_thresh={values['encode_gate_thresh']:.4f} "
            f"assignment_gate_thresh={values['assignment_gate_thresh']:.4f}"
        )
    if values["encode_gate_thresh"] > values["assignment_gate_thresh"]:
        old = float(values["encode_gate_thresh"])
        values["encode_gate_thresh"] = float(values["assignment_gate_thresh"])
        notes.append(
            f"clamped encode_gate_thresh from {old:.4f} to {values['encode_gate_thresh']:.4f} to satisfy encode<=assignment"
        )
    for key in ("novelty_birth_gate_min", "keep_gate_min", "init_seed_gate"):
        if values[key] > values["assignment_gate_thresh"]:
            old = float(values[key])
            values[key] = float(values["assignment_gate_thresh"])
            notes.append(
                f"clamped {key} from {old:.4f} to {values[key]:.4f} to satisfy {key}<=assignment"
            )
    return values, notes


def resolve_runtime_gate_thresholds(
    model_cfg: Dict[str, Any],
    *,
    weights_path: str | Path | None,
    payload: Any | None = None,
) -> Tuple[float, float, Dict[str, Any] | None]:
    fixed_gate_threshold = model_cfg.get("fixed_gate_threshold")
    if fixed_gate_threshold is not None:
        fixed_gate_threshold = float(fixed_gate_threshold)
        encode_gate_thresh = float(model_cfg.get("encode_gate_thresh", fixed_gate_threshold))
        assignment_gate_thresh = float(model_cfg.get("assignment_gate_thresh", fixed_gate_threshold))
        if abs(encode_gate_thresh - fixed_gate_threshold) > 1e-8:
            raise ValueError(
                "fixed_gate_threshold mode requires "
                f"model.encode_gate_thresh={fixed_gate_threshold:.4f}, got {encode_gate_thresh:.4f}"
            )
        if abs(assignment_gate_thresh - fixed_gate_threshold) > 1e-8:
            raise ValueError(
                "fixed_gate_threshold mode requires "
                f"model.assignment_gate_thresh={fixed_gate_threshold:.4f}, got {assignment_gate_thresh:.4f}"
            )
        if abs(fixed_gate_threshold - 0.5) > 1e-8:
            raise ValueError(
                "fixed_gate_threshold mode currently only supports 0.5, "
                f"got {fixed_gate_threshold:.4f}"
            )
        return fixed_gate_threshold, fixed_gate_threshold, None

    encode_gate_thresh = float(model_cfg.get("encode_gate_thresh", 0.2))
    assignment_gate_thresh = float(model_cfg.get("assignment_gate_thresh", 0.1))
    gate_threshold_source = str(model_cfg.get("gate_threshold_source", "config")).strip().lower()
    use_saved_gate_calibration = bool(model_cfg.get("use_saved_gate_calibration", False))
    if gate_threshold_source not in ("config", "saved", "hybrid", "auto", ""):
        gate_threshold_source = "config"
    prefer_saved = bool(use_saved_gate_calibration or gate_threshold_source in ("saved", "hybrid", "auto"))
    if prefer_saved and weights_path is not None:
        gate_calibration = load_saved_gate_calibration(weights_path, payload=payload)
        if isinstance(gate_calibration, dict):
            saved_encode = gate_calibration.get("encode_gate_thresh")
            saved_assignment = gate_calibration.get("assignment_gate_thresh")
            if saved_encode is not None and saved_assignment is not None:
                return float(saved_encode), float(saved_assignment), dict(gate_calibration)
    return encode_gate_thresh, assignment_gate_thresh, None
