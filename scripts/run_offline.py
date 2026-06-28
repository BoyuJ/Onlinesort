from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.eval as eval_mod
from src.benchmark_io import load_event_arrays, load_gt_events, resolve_metadata
from src.dataset import MEArecH5Reader
from src.model_io import (
    build_model_config,
    load_torch_file,
    normalize_runtime_gate_thresholds,
    resolve_runtime_gate_thresholds,
    validate_or_fallback_encoder_checkpoint,
)
from src.nn.model import ModelConfig
from src.operational_gate import (
    align_gt_to_filtered_peaks,
    drift_gt_channel_enabled,
    infer_drifted_gt_peak_channels,
    infer_static_gt_peak_channels,
)
from src.sort.types import normalize_embedding_matrix
from src.stream import OnlineSpikePipeline, PatchConfig, PipelineConfig, build_detector_config
from src.utils import load_yaml, outputs_to_numpy, resolve_channel_geometry


def read_bin(path: Path, n_channels: int, dtype=np.int16) -> np.ndarray:
    raw = np.fromfile(path, dtype=dtype)
    n = raw.size // int(n_channels)
    return raw[: n * int(n_channels)].reshape(n, int(n_channels)).astype(np.float32)


def get_bin_shape(path: Path, n_channels: int, dtype=np.int16) -> tuple[int, int]:
    bytes_per_sample = np.dtype(dtype).itemsize * int(n_channels)
    if bytes_per_sample <= 0:
        raise ValueError(f"invalid n_channels={n_channels}")
    return int(path.stat().st_size // bytes_per_sample), int(n_channels)


def iter_bin_chunks(path: Path, n_channels: int, chunk_samples: int, dtype=np.int16) -> Iterator[np.ndarray]:
    n_channels = int(n_channels)
    chunk_samples = int(chunk_samples)
    if chunk_samples <= 0:
        raise ValueError(f"chunk_samples must be > 0, got {chunk_samples}")
    with open(path, "rb") as f:
        while True:
            raw = np.fromfile(f, dtype=dtype, count=chunk_samples * n_channels)
            if raw.size == 0:
                break
            n = raw.size // n_channels
            yield raw[: n * n_channels].reshape(n, n_channels).astype(np.float32, copy=False)


def init_output_parts(*, include_embedding: bool = False, embed_dim: int | None = None) -> Dict[str, List[np.ndarray]]:
    empty = outputs_to_numpy([], include_embedding=include_embedding, embed_dim=embed_dim)
    return {key: [] for key in empty}


def finalize_output_parts(
    parts: Dict[str, List[np.ndarray]],
    *,
    include_embedding: bool = False,
    embed_dim: int | None = None,
) -> Dict[str, np.ndarray]:
    empty = outputs_to_numpy([], include_embedding=include_embedding, embed_dim=embed_dim)
    out: Dict[str, np.ndarray] = {}
    for key, default in empty.items():
        chunks = parts.get(key, [])
        out[key] = np.concatenate(chunks, axis=0) if chunks else default
    return out


def dedup_output_events(arr: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    t = arr["event_time_sample"]
    c = arr["center_channel"]
    if t.size == 0:
        return arr
    keys = np.empty((t.size,), dtype=[("t", np.int64), ("c", np.int32)])
    keys["t"] = t.astype(np.int64, copy=False)
    keys["c"] = c.astype(np.int32, copy=False)
    _, inv = np.unique(keys, return_inverse=True)
    n_groups = int(inv.max()) + 1
    best_idx = np.full((n_groups,), -1, dtype=np.int64)
    best_conf = np.full((n_groups,), -np.inf, dtype=np.float32)
    best_qc = np.full((n_groups,), np.iinfo(np.int32).max, dtype=np.int32)
    conf = arr["confidence"].astype(np.float32, copy=False)
    qc = arr["qc_flags"].astype(np.int32, copy=False)
    for i in range(inv.size):
        group = int(inv[i])
        ci = float(conf[i])
        qi = int(qc[i])
        if ci > float(best_conf[group]) or (ci == float(best_conf[group]) and qi < int(best_qc[group])):
            best_conf[group] = ci
            best_qc[group] = qi
            best_idx[group] = i
    keep = best_idx[best_idx >= 0]
    if keep.size == 0:
        width = int(arr.get("embedding", np.zeros((0, 0))).shape[-1])
        return outputs_to_numpy([], include_embedding="embedding" in arr, embed_dim=width)
    order = np.lexsort((arr["center_channel"][keep], arr["event_time_sample"][keep]))
    keep = keep[order]
    return {key: value[keep] for key, value in arr.items()}


def resolve_path(path_value: str | Path, root: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else root / p


def resolve_neighbor_graph(
    *,
    channel_positions: np.ndarray | None,
    n_channels: int,
    electrode_config: str | Path | None,
    neighbor_source: str,
    fallback_k: int,
    root: Path,
    logger: logging.Logger,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    n_ch = int(n_channels)
    source_req = str(neighbor_source).lower()
    if source_req not in ("auto", "h5", "config"):
        logger.warning("unsupported neighbor_source=%s; fallback to auto", neighbor_source)
        source_req = "auto"

    h5_positions = None
    if channel_positions is not None:
        h5_try = np.asarray(channel_positions, dtype=np.float32)
        if h5_try.ndim == 2 and h5_try.shape[0] == n_ch:
            h5_positions = h5_try
        else:
            logger.warning("ignore H5 channel_positions: shape=%s channels=%d", tuple(h5_try.shape), n_ch)

    electrode_cfg_path = resolve_path(electrode_config, root) if electrode_config else None
    electrode_cfg = None
    if n_ch > 1 and electrode_cfg_path is None:
        raise ValueError("multi-channel inference requires an electrode_config")
    if electrode_cfg_path is not None and not electrode_cfg_path.exists() and n_ch > 1:
        raise FileNotFoundError(f"electrode_config not found: {electrode_cfg_path}")
    if electrode_cfg_path is not None and electrode_cfg_path.exists():
        electrode_cfg = load_yaml(electrode_cfg_path)

    positions, neighbors, geometry_source, metrics = resolve_channel_geometry(
        n_channels=n_ch,
        fallback_k=fallback_k,
        electrode_cfg=electrode_cfg,
        h5_positions=h5_positions,
    )
    if n_ch > 1 and geometry_source != "config":
        raise ValueError(
            f"multi-channel inference requires electrode_config matching n_channels={n_ch}; "
            f"resolved neighbor source={geometry_source}"
        )
    if metrics is not None:
        logger.info(
            "neighbor_compare nn1_match_ratio=%.4f nnk_match_ratio=%.4f source_pref=%s",
            float(metrics.get("nn1_match_ratio", float("nan"))),
            float(metrics.get("nnk_match_ratio", float("nan"))),
            source_req,
        )
    if positions is not None and neighbors is not None:
        logger.info("neighbor source=%s k_neighbors=%d", geometry_source, int(neighbors.shape[1]))
        return positions, neighbors, str(geometry_source)
    return channel_positions, None, "none"


def resolve_cuda_device(requested: str) -> torch.device:
    req = str(requested).lower()
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("device=auto resolved to CPU, but inference requires GPU.")
    if req == "cpu":
        raise RuntimeError("CPU inference is disabled. Use --device cuda.")
    raise ValueError(f"unsupported device: {requested}")


def build_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mysort.helper")
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


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if is_dataclass(obj):
        return {field.name: to_jsonable(getattr(obj, field.name)) for field in fields(obj)}
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    return obj


def benchmark_report_path_for_output(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_benchmark_report.json")


def save_benchmark_report(report_path: Path, payload: Dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_search_values(value: Any, *, cast: Any, min_value: Any | None = None) -> List[Any]:
    values = [cast(v) for v in _as_list(value)]
    if min_value is not None:
        values = [v for v in values if v >= min_value]
    seen: List[Any] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def resolve_offline_clustering_config(offline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(offline_cfg.get("clustering", {}) or {})
    methods = [str(value).strip().lower() for value in _as_list(raw.get("methods", raw.get("method", "hdbscan")))]
    methods = [value for value in methods if value]
    if not methods:
        methods = ["hdbscan"]
    if any(value != "hdbscan" for value in methods):
        raise ValueError("only HDBSCAN warmup clustering is supported")
    n_units_raw = raw.get("n_units", None)
    n_units = None if n_units_raw is None else int(n_units_raw)
    if n_units is not None and n_units <= 0:
        raise ValueError(f"offline.clustering.n_units must be positive or null, got {n_units}")
    return {
        "method": "hdbscan",
        "methods": ["hdbscan"],
        "selection_metric": str(raw.get("selection_metric", "no_gt_high_recall_k")),
        "fail_on_method_error": bool(raw.get("fail_on_method_error", False)),
        "required_methods": [
            str(value).strip().lower()
            for value in _as_list(raw.get("required_methods", []))
            if str(value).strip().lower() == "hdbscan"
        ],
        "n_units": n_units,
        "random_state": int(raw.get("random_state", 0)),
        "hdbscan_min_cluster_size": _as_list(raw.get("hdbscan_min_cluster_size", [8, 16, 32, 64])),
        "hdbscan_min_samples": _as_list(raw.get("hdbscan_min_samples", [1, 5, 10])),
        "hdbscan_cluster_selection_epsilon": _as_list(raw.get("hdbscan_cluster_selection_epsilon", [0.0])),
        "hdbscan_cluster_selection_method": str(raw.get("hdbscan_cluster_selection_method", "leaf")),
        "hdbscan_selection_metric": str(raw.get("hdbscan_selection_metric", raw.get("selection_metric", "no_gt_high_recall_k"))),
        "hdbscan_high_recall_k_delta": float(raw.get("hdbscan_high_recall_k_delta", 0.30)),
        "hdbscan_high_recall_k_min_clustered_fraction": float(raw.get("hdbscan_high_recall_k_min_clustered_fraction", 0.25)),
        "hdbscan_high_recall_k_max_small_unit_pressure": float(raw.get("hdbscan_high_recall_k_max_small_unit_pressure", 1.0)),
        "hdbscan_high_recall_k_max_noise_ratio": float(raw.get("hdbscan_high_recall_k_max_noise_ratio", 0.80)),
        "hdbscan_online_profile": str(raw.get("hdbscan_online_profile", "exp090_threshold_only")),
        "hdbscan_merge_enabled": bool(raw.get("hdbscan_merge_enabled", True)),
        "hdbscan_merge_profile": str(raw.get("hdbscan_merge_profile", "loose")),
        "hdbscan_merge_cosine_min": float(raw.get("hdbscan_merge_cosine_min", 0.90)),
        "hdbscan_merge_channel_gap_max": float(raw.get("hdbscan_merge_channel_gap_max", 2.0)),
        "hdbscan_merge_refractory_ms": float(raw.get("hdbscan_merge_refractory_ms", 0.30)),
        "hdbscan_merge_cross_conflict_max": float(raw.get("hdbscan_merge_cross_conflict_max", 0.02)),
        "hdbscan_merge_max_rounds": int(raw.get("hdbscan_merge_max_rounds", 8)),
        "nogt_k_stop_enabled": bool(raw.get("nogt_k_stop_enabled", False)),
        "nogt_k_stop_delta": float(raw.get("nogt_k_stop_delta", 0.30)),
        "nogt_k_stop_quantile": float(raw.get("nogt_k_stop_quantile", 0.90)),
        "nogt_k_stop_min_premerge_fraction": float(raw.get("nogt_k_stop_min_premerge_fraction", 0.40)),
    }


def _positive_cluster_count(labels: np.ndarray) -> int:
    arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    return int(np.unique(arr[arr > 0]).size)


def _relabel_positive(labels_zero_based: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels_zero_based, dtype=np.int32).reshape(-1)
    out = np.full(labels.shape, -1, dtype=np.int32)
    non_noise = labels >= 0
    if not np.any(non_noise):
        return out
    unique = np.unique(labels[non_noise])
    mapping = {int(src): int(i + 1) for i, src in enumerate(unique.tolist())}
    for src, dst in mapping.items():
        out[labels == int(src)] = int(dst)
    return out


def _compress_positive_labels(labels: np.ndarray) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    out = np.full(labels_arr.shape, -1, dtype=np.int32)
    unique = [int(value) for value in np.unique(labels_arr).tolist() if int(value) > 0]
    for dst, src in enumerate(unique, start=1):
        out[labels_arr == int(src)] = int(dst)
    return out


def _normalized_centroids(z: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    z_norm = normalize_embedding_matrix(np.asarray(z, dtype=np.float32))
    label_values = np.asarray([v for v in np.unique(labels_arr).tolist() if int(v) > 0], dtype=np.int32)
    centers = []
    for label in label_values.tolist():
        members = z_norm[labels_arr == int(label)]
        center = np.mean(members, axis=0)
        center = center / max(float(np.linalg.norm(center)), 1e-8)
        centers.append(center.astype(np.float32, copy=False))
    if not centers:
        return label_values, np.zeros((0, z_norm.shape[1]), dtype=np.float32)
    return label_values, normalize_embedding_matrix(np.stack(centers, axis=0))


def _cross_refractory_conflict_rate(times_a: np.ndarray, times_b: np.ndarray, refractory_samples: int) -> float:
    a = np.sort(np.asarray(times_a, dtype=np.int64).reshape(-1))
    b = np.sort(np.asarray(times_b, dtype=np.int64).reshape(-1))
    if a.size == 0 or b.size == 0 or int(refractory_samples) <= 0:
        return 0.0
    if a.size > b.size:
        a, b = b, a
    ref = int(refractory_samples)
    j0 = 0
    conflicts = 0
    for t_raw in a.tolist():
        t = int(t_raw)
        while j0 < b.size and int(b[j0]) < t - ref:
            j0 += 1
        j = j0
        while j < b.size and int(b[j]) <= t + ref:
            conflicts += 1
            break
    return float(conflicts / max(1, int(a.size)))


def _merge_seed_labels(
    z: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    *,
    center_channels: np.ndarray | None,
    times_samples: np.ndarray | None,
    requested_k: int,
    prefix: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if not bool(cfg.get(f"{prefix}_merge_enabled", True)):
        return _compress_positive_labels(labels), {"enabled": False, "merges": 0}

    out = _compress_positive_labels(labels)
    cosine_min = float(cfg.get(f"{prefix}_merge_cosine_min", 0.97))
    channel_gap_max = float(cfg.get(f"{prefix}_merge_channel_gap_max", -1.0))
    conflict_max = float(cfg.get(f"{prefix}_merge_cross_conflict_max", 0.01))
    refractory_ms = float(cfg.get(f"{prefix}_merge_refractory_ms", 0.30))
    fs = float(cfg.get("fs", 32000.0))
    refractory_samples = max(1, int(round(refractory_ms * fs / 1000.0)))
    max_rounds = max(1, int(cfg.get(f"{prefix}_merge_max_rounds", 8)))
    requested = int(requested_k)
    total_merges = 0
    total_comparisons = 0
    accepted_pairs: List[Dict[str, Any]] = []
    rounds = 0

    for _ in range(max_rounds):
        label_values, centroids = _normalized_centroids(z, out)
        n_clusters = int(len(label_values))
        if n_clusters <= 1 or (requested > 0 and n_clusters <= requested):
            break
        parent = {int(label): int(label) for label in label_values}

        def find(x: int) -> int:
            while parent[int(x)] != int(x):
                parent[int(x)] = parent[parent[int(x)]]
                x = parent[int(x)]
            return int(x)

        def union(a: int, b: int) -> bool:
            ra = find(int(a))
            rb = find(int(b))
            if ra == rb:
                return False
            parent[max(ra, rb)] = min(ra, rb)
            return True

        channel_medians: Dict[int, float] = {}
        if center_channels is not None and channel_gap_max >= 0.0:
            ch = np.asarray(center_channels, dtype=np.float32).reshape(-1)
            if ch.shape[0] == out.shape[0]:
                for label in label_values:
                    channel_medians[int(label)] = float(np.median(ch[out == int(label)]))

        label_sizes = {int(label): int(np.sum(out == int(label))) for label in label_values}
        candidates: List[tuple[float, float, float, int, int, int, float, float]] = []
        for i, label_i in enumerate(label_values):
            for j in range(i + 1, len(label_values)):
                label_j = int(label_values[j])
                total_comparisons += 1
                cosine = float(np.clip(np.dot(centroids[i], centroids[j]), -1.0, 1.0))
                if cosine < cosine_min:
                    continue
                channel_gap = 0.0
                if channel_gap_max >= 0.0 and channel_medians:
                    channel_gap = abs(channel_medians.get(int(label_i), 0.0) - channel_medians.get(label_j, 0.0))
                    if channel_gap > channel_gap_max:
                        continue
                conflict = 0.0
                if times_samples is not None:
                    times_arr = np.asarray(times_samples, dtype=np.int64).reshape(-1)
                    if times_arr.shape[0] == out.shape[0]:
                        conflict = _cross_refractory_conflict_rate(
                            times_arr[out == int(label_i)],
                            times_arr[out == label_j],
                            refractory_samples,
                        )
                        if conflict > conflict_max:
                            continue
                min_size = min(label_sizes.get(int(label_i), 0), label_sizes.get(label_j, 0))
                candidates.append((cosine, -conflict, -channel_gap, -min_size, int(label_i), label_j, channel_gap, conflict))

        candidates.sort(reverse=True, key=lambda item: item[:4])
        merge_budget = int(1e9) if requested <= 0 else max(0, n_clusters - requested)
        merges_this_round = 0
        for score, _neg_conflict, _neg_channel_gap, neg_min_size, label_i, label_j, channel_gap, conflict in candidates:
            if merges_this_round >= merge_budget:
                break
            if union(label_i, label_j):
                merges_this_round += 1
                total_merges += 1
                accepted_pairs.append(
                    {
                        "round": int(rounds),
                        "label_i": int(label_i),
                        "label_j": int(label_j),
                        "cosine": float(score),
                        "channel_gap": float(channel_gap),
                        "cross_conflict": float(conflict),
                        "size_i": int(label_sizes.get(int(label_i), 0)),
                        "size_j": int(label_sizes.get(int(label_j), 0)),
                        "min_size": int(-neg_min_size),
                    }
                )
        if merges_this_round <= 0:
            break
        merged = np.full(out.shape, -1, dtype=np.int32)
        for label in label_values:
            merged[out == int(label)] = find(int(label))
        out = _compress_positive_labels(merged)
        rounds += 1

    return out, {
        "enabled": True,
        "comparisons": int(total_comparisons),
        "merges": int(total_merges),
        "pairs": accepted_pairs,
        "rounds": int(rounds),
        "target_clusters": int(requested),
        "cosine_min": float(cosine_min),
        "channel_gap_max": float(channel_gap_max),
        "cross_conflict_max": float(conflict_max),
        "refractory_ms": float(refractory_ms),
        "refractory_samples": int(refractory_samples),
    }


def resolve_benchmark_backend(bench_cfg_raw: Dict[str, Any]) -> str:
    backend = str(bench_cfg_raw.get("backend", "kilo")).strip().lower()
    return backend if backend in ("kilo", "legacy") else "kilo"


def build_benchmark_config(
    backend: str,
    *,
    bench_cfg_raw: Dict[str, Any],
    label_cfg: Dict[str, Any],
    det_cfg: Dict[str, Any],
) -> Any:
    if backend == "kilo":
        tol_samples_raw = bench_cfg_raw.get("tol_samples", 6)
        tol_ms_raw = bench_cfg_raw.get("tol_ms", None)
        return eval_mod.BenchmarkConfig(
            tol_samples=None if tol_samples_raw is None else int(tol_samples_raw),
            tol_ms=None if tol_ms_raw is None else float(tol_ms_raw),
            min_sort_label=int(bench_cfg_raw.get("min_sort_label", label_cfg.get("min_pred_label", 0))),
            n_check=int(bench_cfg_raw.get("n_check", 0)),
            candidate_match_mode=str(bench_cfg_raw.get("candidate_match_mode", "all")),
            channel_radius_ch=int(
                bench_cfg_raw.get("channel_radius_ch", label_cfg.get("channel_radius_ch", det_cfg.get("merge_radius_ch", 1)))
            ),
            score_threshold=float(bench_cfg_raw.get("score_threshold", 0.8)),
            window_sec=float(bench_cfg_raw.get("window_sec", 60.0)),
            diagnostics=bool(bench_cfg_raw.get("diagnostics", True)),
        )
    return eval_mod.BenchmarkConfig(
        tol_ms=float(bench_cfg_raw.get("tol_ms", label_cfg.get("tol_ms", 0.3))),
        min_sort_label=int(bench_cfg_raw.get("min_sort_label", label_cfg.get("min_pred_label", 0))),
        event_match_mode=str(bench_cfg_raw.get("event_match_mode", "auto")),
        channel_radius_ch=int(
            bench_cfg_raw.get("channel_radius_ch", label_cfg.get("channel_radius_ch", det_cfg.get("merge_radius_ch", 1)))
        ),
        window_sec=float(bench_cfg_raw.get("window_sec", 60.0)),
        recovery_f1_threshold=float(bench_cfg_raw.get("recovery_f1_threshold", 0.8)),
        diagnostics=bool(bench_cfg_raw.get("diagnostics", True)),
    )


def log_benchmark_config(logger: logging.Logger, backend: str, bench_cfg: Any) -> None:
    if backend == "kilo":
        logger.info(
            "benchmark_config backend=%s tol_samples=%s tol_ms=%s min_sort_label=%d",
            backend,
            "None" if getattr(bench_cfg, "tol_samples", None) is None else str(int(bench_cfg.tol_samples)),
            "None" if getattr(bench_cfg, "tol_ms", None) is None else f"{float(bench_cfg.tol_ms):.3f}",
            int(bench_cfg.min_sort_label),
        )
    else:
        logger.info(
            "benchmark_config backend=%s tol_ms=%.3f min_sort_label=%d",
            backend,
            float(bench_cfg.tol_ms),
            int(bench_cfg.min_sort_label),
        )


def resolve_benchmark_tol_samples(backend: str, bench_cfg: Any, *, fs: float) -> int:
    if backend == "kilo":
        tol_samples = getattr(bench_cfg, "tol_samples", None)
        if tol_samples is not None:
            return max(0, int(tol_samples))
        tol_ms = getattr(bench_cfg, "tol_ms", None)
        return 6 if tol_ms is None else max(0, int(round(float(tol_ms) * float(fs) / 1000.0)))
    return max(0, int(round(float(bench_cfg.tol_ms) * float(fs) / 1000.0)))


def resolve_benchmark_gt_time_mode(bench_cfg_raw: Dict[str, Any]) -> str:
    mode = str(bench_cfg_raw.get("gt_time_mode", "raw") or "raw").strip().lower()
    if mode in ("raw", "original", "none", "off"):
        return "raw"
    if mode in ("peak", "peak_aligned", "aligned"):
        return "peak_aligned"
    raise ValueError(f"unsupported benchmark.gt_time_mode={mode!r}; use raw | peak_aligned")


def _resolve_eval_car_mode(n_channels: int, preprocess_cfg: Dict[str, Any]) -> str:
    mode = str(preprocess_cfg.get("car_mode", "auto") or "auto").strip().lower()
    if mode == "auto":
        return "none" if int(n_channels) == 1 else "global"
    return mode


def build_eval_gt_alignment_config(
    *,
    bench_cfg_raw: Dict[str, Any],
    label_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    align_cfg = dict(label_cfg)
    align_cfg.update(
        {
            "gt_align_to_peak": True,
            "gt_align_radius_ms": float(bench_cfg_raw.get("gt_align_radius_ms", label_cfg.get("gt_align_radius_ms", 0.5))),
            "gt_time_reference": str(bench_cfg_raw.get("gt_time_reference", label_cfg.get("gt_time_reference", "near_peak"))),
            "single_channel_gt_align_radius_ms": float(
                bench_cfg_raw.get("single_channel_gt_align_radius_ms", label_cfg.get("single_channel_gt_align_radius_ms", 2.0))
            ),
            "single_channel_gt_time_reference": str(
                bench_cfg_raw.get("single_channel_gt_time_reference", label_cfg.get("single_channel_gt_time_reference", "onset"))
            ),
            "gt_align_polarity": str(bench_cfg_raw.get("gt_align_polarity", label_cfg.get("gt_align_polarity", "negative"))),
            "gt_align_channel_hops": int(bench_cfg_raw.get("gt_align_channel_hops", label_cfg.get("gt_align_channel_hops", 1))),
        }
    )
    return align_cfg


def align_gt_events_for_benchmark(
    gt_events: eval_mod.EventSet,
    *,
    input_path: Path,
    fs: float,
    n_channels: int,
    cfg: Dict[str, Any],
    bench_cfg_raw: Dict[str, Any],
    neighbors_lookup: np.ndarray | None,
    logger: logging.Logger | None = None,
) -> tuple[eval_mod.EventSet, Dict[str, Any]]:
    gt_time_mode = resolve_benchmark_gt_time_mode(bench_cfg_raw)
    stats: Dict[str, Any] = {"gt_time_mode": gt_time_mode, "enabled": False, "total": int(gt_events.times_samples.size)}
    if gt_time_mode == "raw":
        if logger is not None:
            logger.info("gt_eval_time_reference mode=raw total=%d", int(gt_events.times_samples.size))
        return gt_events, stats

    n_ch = int(n_channels)
    neigh = np.asarray(neighbors_lookup, dtype=np.int32) if neighbors_lookup is not None else np.arange(n_ch, dtype=np.int32)[:, None]
    label_cfg = cfg.get("labeling", {})
    preprocess_cfg = dict(cfg.get("preprocess", {}))
    align_cfg = build_eval_gt_alignment_config(bench_cfg_raw=bench_cfg_raw, label_cfg=label_cfg)

    gt_template_channels, gt_channel_source = infer_static_gt_peak_channels(input_path, gt_events, fs=float(fs), n_channels=n_ch)
    gt_channels = gt_template_channels.astype(np.int32, copy=True)
    if drift_gt_channel_enabled(input_path, n_channels=n_ch, neighbors_lookup=neigh):
        gt_drift_channels, drift_source = infer_drifted_gt_peak_channels(input_path, gt_events, fs=float(fs), n_channels=n_ch)
        if drift_source == "drift_templates":
            gt_channels = gt_drift_channels.astype(np.int32, copy=True)
            gt_channel_source = drift_source

    aligned_gt, align_stats = align_gt_to_filtered_peaks(
        gt_events,
        input_h5=input_path,
        fs=float(fs),
        n_channels=n_ch,
        gt_channels=gt_channels,
        labeling_cfg=align_cfg,
        preprocess_cfg=preprocess_cfg,
        car_mode=_resolve_eval_car_mode(n_ch, preprocess_cfg),
        neighbors_lookup=neigh,
    )
    stats.update(dict(align_stats))
    stats["gt_time_mode"] = gt_time_mode
    stats["gt_channel_source"] = str(gt_channel_source)
    stats["n_channels"] = n_ch
    if logger is not None:
        logger.info(
            "gt_peak_align_eval mode=%s total=%d aligned=%d channel_source=%s",
            str(gt_time_mode),
            int(stats.get("total", 0)),
            int(stats.get("aligned", 0)),
            str(gt_channel_source),
        )
    return aligned_gt, stats


def build_pred_event_set(
    arr: Dict[str, np.ndarray],
    *,
    label_key: str,
    fs: float,
    out_path: Path,
    source_name: str,
) -> eval_mod.EventSet:
    return load_event_arrays(
        arr["event_time_sample"],
        arr[label_key],
        channels=arr["center_channel"],
        confidences=arr["confidence"],
        fs=fs,
        times_unit="samples",
        metadata={"source": source_name, "path": str(out_path), "fs": float(fs), "label_key": str(label_key)},
    )


def summarize_detection_against_gt(
    pred_events: eval_mod.EventSet,
    gt_events: eval_mod.EventSet,
    *,
    tol_samples: int,
    min_sort_label: int,
) -> Dict[str, Any]:
    all_metrics = eval_mod.detection_metrics(pred_events.times_samples, gt_events.times_samples, tol_samples)
    keep_mask = pred_events.labels >= int(min_sort_label)
    kept_metrics = eval_mod.detection_metrics(pred_events.times_samples[keep_mask], gt_events.times_samples, tol_samples)
    return {"tol_samples": int(tol_samples), "all": dict(all_metrics), "kept": dict(kept_metrics)}


def _gate_binary_metrics(*, spike_pass: int, spike_block: int, noise_pass: int, noise_block: int) -> Dict[str, float]:
    spike_total = int(spike_pass) + int(spike_block)
    noise_total = int(noise_pass) + int(noise_block)
    total = spike_total + noise_total
    precision_den = int(spike_pass) + int(noise_pass)
    spike_recall = float(spike_pass / spike_total) if spike_total > 0 else float("nan")
    noise_reject = float(noise_block / noise_total) if noise_total > 0 else float("nan")
    precision = float(spike_pass / precision_den) if precision_den > 0 else float("nan")
    f1_den = precision + spike_recall
    gate_f1 = float(2.0 * precision * spike_recall / f1_den) if np.isfinite(f1_den) and f1_den > 0 else float("nan")
    balanced = float(0.5 * (spike_recall + noise_reject)) if np.isfinite(spike_recall) and np.isfinite(noise_reject) else float("nan")
    overall = float((int(spike_pass) + int(noise_block)) / total) if total > 0 else float("nan")
    return {
        "gate_spike_recall": spike_recall,
        "gate_noise_reject_rate": noise_reject,
        "gate_f1": gate_f1,
        "gate_balanced_accuracy": balanced,
        "gate_overall_accuracy": overall,
    }


def build_gate_summary(
    arr: Dict[str, np.ndarray],
    *,
    assignment_gate_thresh: float,
    gt_events: eval_mod.EventSet | None = None,
    tol_samples: int | None = None,
) -> Dict[str, Any]:
    gate_prob = np.asarray(arr.get("gate_prob", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    encoded_mask = np.asarray(arr.get("encoded_mask", np.zeros((gate_prob.size,), dtype=np.int8))).reshape(-1) > 0
    if encoded_mask.shape != gate_prob.shape:
        n = min(int(encoded_mask.size), int(gate_prob.size))
        encoded_mask = encoded_mask[:n]
        gate_prob = gate_prob[:n]
    assignment_pass = encoded_mask & (gate_prob >= float(assignment_gate_thresh))
    out: Dict[str, Any] = {"gate_threshold": float(assignment_gate_thresh)}
    if gt_events is None or tol_samples is None or "event_time_sample" not in arr:
        return out

    labels = np.asarray(arr.get("unit_id", np.full((gate_prob.size,), -1, dtype=np.int32)), dtype=np.int64)
    pred_events = eval_mod.EventSet(
        times_samples=np.asarray(arr["event_time_sample"], dtype=np.int64)[: assignment_pass.size],
        labels=labels[: assignment_pass.size],
        channels=np.asarray(arr["center_channel"], dtype=np.int32)[: assignment_pass.size] if "center_channel" in arr else None,
    )
    matches = eval_mod.match_events(
        pred_events,
        gt_events,
        tol_samples=int(tol_samples),
        event_match_mode="time",
        channel_radius_ch=0,
    )
    spike_idx = {int(match.pred_index) for match in matches}
    is_spike = np.asarray([idx in spike_idx for idx in range(assignment_pass.size)], dtype=np.bool_)
    is_noise = ~is_spike
    out.update(
        _gate_binary_metrics(
            spike_pass=int(np.sum(is_spike & assignment_pass)),
            spike_block=int(np.sum(is_spike & (~assignment_pass))),
            noise_pass=int(np.sum(is_noise & assignment_pass)),
            noise_block=int(np.sum(is_noise & (~assignment_pass))),
        )
    )
    return out


def build_eval_summary(with_gt: Dict[str, Any] | None, no_gt: Dict[str, Any] | None, gate_summary: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(with_gt, dict):
        summary = dict(with_gt.get("summary", {}))
        out: Dict[str, Any] = {
            "n_gt_units": int(summary.get("n_gt_units", 0)),
            "n_gt_events": int(summary.get("n_gt_events", 0)),
            "acc_1to1_global": float(summary.get("acc_1to1_global", float("nan"))),
            "acc_1to1_tp": float(summary.get("acc_1to1_tp", summary.get("classification_acc_true_positive", float("nan")))),
            "precision_1to1": float(summary.get("precision_1to1", float("nan"))),
            "recall_1to1": float(summary.get("recall_1to1", float("nan"))),
            "sorted_score_1to1": float(summary.get("sorted_score_1to1", summary.get("sorted_score_mean_1to1_all_gt", float("nan")))),
            "snr_gt8_n_gt_units": int(summary.get("snr_gt8_n_gt_units", 0)),
            "snr_gt8_n_gt_events": int(summary.get("snr_gt8_n_gt_events", 0)),
            "snr_gt8_acc_1to1_global": float(summary.get("snr_gt8_acc_1to1_global", float("nan"))),
            "snr_gt8_acc_1to1_tp": float(summary.get("snr_gt8_acc_1to1_tp", float("nan"))),
            "snr_gt8_precision_1to1": float(summary.get("snr_gt8_precision_1to1", float("nan"))),
            "snr_gt8_recall_1to1": float(summary.get("snr_gt8_recall_1to1", float("nan"))),
            "snr_gt8_sorted_score_1to1": float(summary.get("snr_gt8_sorted_score_1to1", float("nan"))),
        }
    else:
        no_gt_summary = dict(no_gt.get("summary", {})) if isinstance(no_gt, dict) else {}
        out = {
            "no_gt_sorted_score_mean": float(no_gt_summary.get("sorted_score_mean", float("nan"))),
            "no_gt_SNR_mean": float(no_gt_summary.get("SNR_mean", float("nan"))),
            "no_gt_FP_est_mean": float(no_gt_summary.get("FP_est_mean", float("nan"))),
            "no_gt_missed_spikes_FN_mean": float(no_gt_summary.get("missed_spikes_FN_mean", float("nan"))),
        }
    for key in (
        "gate_threshold",
        "gate_spike_recall",
        "gate_noise_reject_rate",
        "gate_f1",
        "gate_balanced_accuracy",
        "gate_overall_accuracy",
    ):
        if key in gate_summary:
            out[key] = gate_summary[key]
    return out


def build_quick_eval_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gate": {
            "threshold": float(summary.get("gate_threshold", float("nan"))),
            "spike_recall": float(summary.get("gate_spike_recall", float("nan"))),
            "noise_reject_rate": float(summary.get("gate_noise_reject_rate", float("nan"))),
            "gate_f1": float(summary.get("gate_f1", float("nan"))),
            "balanced_accuracy": float(summary.get("gate_balanced_accuracy", float("nan"))),
            "overall_accuracy": float(summary.get("gate_overall_accuracy", float("nan"))),
        },
        "classification_snr_gt8": {
            "n_gt_units": int(summary.get("snr_gt8_n_gt_units", 0)),
            "n_gt_events": int(summary.get("snr_gt8_n_gt_events", 0)),
            "acc_1to1_tp": float(summary.get("snr_gt8_acc_1to1_tp", float("nan"))),
            "acc_1to1_global": float(summary.get("snr_gt8_acc_1to1_global", float("nan"))),
            "precision_1to1": float(summary.get("snr_gt8_precision_1to1", float("nan"))),
            "recall_1to1": float(summary.get("snr_gt8_recall_1to1", float("nan"))),
            "sorted_score_1to1": float(summary.get("snr_gt8_sorted_score_1to1", float("nan"))),
        },
        "classification_all_snr": {
            "n_gt_units": int(summary.get("n_gt_units", 0)),
            "n_gt_events": int(summary.get("n_gt_events", 0)),
            "acc_1to1_tp": float(summary.get("acc_1to1_tp", float("nan"))),
            "acc_1to1_global": float(summary.get("acc_1to1_global", float("nan"))),
            "precision_1to1": float(summary.get("precision_1to1", float("nan"))),
            "recall_1to1": float(summary.get("recall_1to1", float("nan"))),
            "sorted_score_1to1": float(summary.get("sorted_score_1to1", float("nan"))),
        },
    }


def method_output_path(out_path: Path, method: str) -> Path:
    return out_path.with_name(f"{out_path.stem}_{method}{out_path.suffix}")


def _finite_metric(value: Any, default: float = float("-inf")) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return score if math.isfinite(score) else default


def score_eval_summary(summary: Dict[str, Any] | None, metric: str) -> tuple[float, float, float]:
    if not isinstance(summary, dict):
        return (float("-inf"), float("-inf"), float("-inf"))
    primary = _finite_metric(summary.get(metric))
    if primary == float("-inf"):
        primary = _finite_metric(summary.get("sorted_score_1to1"))
    if primary == float("-inf"):
        primary = _finite_metric(summary.get("no_gt_sorted_score_mean"))
    return (
        primary,
        _finite_metric(summary.get("acc_1to1_global")),
        _finite_metric(summary.get("acc_1to1_tp")),
    )


if __name__ == "__main__":
    raise SystemExit("This module now provides shared helpers for scripts/run_online.py; use scripts/run_online.py.")
