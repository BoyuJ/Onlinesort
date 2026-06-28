from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import h5py
import numpy as np

from .dataset import MEArecH5Reader
from .eval import EventSet


def _to_1d(arr: np.ndarray | Sequence[Any], *, dtype: Any) -> np.ndarray:
    return np.asarray(arr, dtype=dtype).reshape(-1)


def _times_to_samples(
    times: np.ndarray | Sequence[Any],
    *,
    fs: float | None,
    times_unit: str,
) -> np.ndarray:
    arr = np.asarray(times).reshape(-1)
    unit = str(times_unit).lower()
    if unit in ("sample", "samples"):
        return np.rint(arr.astype(np.float64, copy=False)).astype(np.int64, copy=False)
    if unit in ("second", "seconds", "sec", "s"):
        if fs is None:
            raise ValueError("fs is required when times_unit='seconds'")
        return np.rint(arr.astype(np.float64, copy=False) * float(fs)).astype(np.int64, copy=False)
    if unit != "auto":
        raise ValueError(f"unsupported times_unit={times_unit}")
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    if fs is None:
        raise ValueError("fs is required for floating-point times when times_unit='auto'")
    return np.rint(arr.astype(np.float64, copy=False) * float(fs)).astype(np.int64, copy=False)


def _normalize_unit_positions(unit_positions: Dict[int, np.ndarray] | None) -> Dict[int, np.ndarray] | None:
    if unit_positions is None:
        return None
    return {
        int(unit): np.asarray(pos, dtype=np.float32).reshape(-1)
        for unit, pos in dict(unit_positions).items()
    }


def load_event_arrays(
    times: np.ndarray | Sequence[Any],
    labels: np.ndarray | Sequence[Any],
    *,
    channels: np.ndarray | Sequence[Any] | None = None,
    confidences: np.ndarray | Sequence[Any] | None = None,
    fs: float | None = None,
    times_unit: str = "samples",
    unit_positions: Dict[int, np.ndarray] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> EventSet:
    meta = dict(metadata or {})
    if fs is not None:
        meta.setdefault("fs", float(fs))
    return EventSet(
        times_samples=_times_to_samples(times, fs=fs, times_unit=times_unit),
        labels=_to_1d(labels, dtype=np.int64),
        channels=None if channels is None else _to_1d(channels, dtype=np.int32),
        confidences=None if confidences is None else _to_1d(confidences, dtype=np.float32),
        unit_positions=_normalize_unit_positions(unit_positions),
        metadata=meta,
    )


def load_mysort_npz(npz_path: str | Path, *, fs: float | None = None) -> EventSet:
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as arr:
        if "event_time_sample" in arr:
            times = arr["event_time_sample"]
            times_unit = "samples"
        elif "event_time_sec" in arr:
            times = arr["event_time_sec"]
            times_unit = "seconds"
        else:
            raise KeyError("mysort npz requires event_time_sample or event_time_sec")
        if "unit_id" not in arr:
            raise KeyError("mysort npz requires unit_id")
        channels = arr["center_channel"] if "center_channel" in arr else None
        confidences = arr["confidence"] if "confidence" in arr else None
        meta = {"source": "mysort_npz", "path": str(path)}
        if fs is not None:
            meta["fs"] = float(fs)
        return load_event_arrays(
            times,
            arr["unit_id"],
            channels=channels,
            confidences=confidences,
            fs=fs,
            times_unit=times_unit,
            metadata=meta,
        )


def _best_channel_from_templates(templates: np.ndarray) -> np.ndarray:
    arr = np.asarray(templates)
    if arr.ndim < 3:
        raise ValueError(f"templates.npy must be at least 3D, got shape={arr.shape}")
    rel_axes = list(range(1, arr.ndim))
    ch_axis = max(rel_axes, key=lambda axis: int(arr.shape[axis]))
    arr_move = np.moveaxis(arr, ch_axis, -1)
    flat = np.reshape(np.abs(arr_move), (arr_move.shape[0], -1, arr_move.shape[-1]))
    return np.argmax(np.max(flat, axis=1), axis=1).astype(np.int32, copy=False)


def _load_cluster_filter(results_dir: Path) -> Dict[int, str]:
    for name in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        path = results_dir / name
        if not path.exists():
            continue
        out: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                cluster_raw = row.get("cluster_id") or row.get("cluster") or row.get("id")
                label = row.get("group") or row.get("KSLabel") or row.get("label")
                if cluster_raw is None or label is None:
                    continue
                try:
                    out[int(cluster_raw)] = str(label)
                except ValueError:
                    continue
        if out:
            return out
    return {}


def load_phy_results_dir(
    results_dir: str | Path,
    *,
    fs: float | None = None,
    allowed_cluster_labels: Sequence[str] | None = None,
) -> EventSet:
    path = Path(results_dir)
    spike_times_path = path / "spike_times.npy"
    spike_clusters_path = path / "spike_clusters.npy"
    if not spike_times_path.exists() or not spike_clusters_path.exists():
        raise FileNotFoundError("Phy/Kilosort results require spike_times.npy and spike_clusters.npy")

    spike_times = np.load(spike_times_path, allow_pickle=False).reshape(-1)
    spike_clusters = np.load(spike_clusters_path, allow_pickle=False).reshape(-1)
    if spike_times.shape != spike_clusters.shape:
        raise ValueError("spike_times.npy and spike_clusters.npy must have the same shape")

    keep = np.ones((spike_times.shape[0],), dtype=np.bool_)
    cluster_labels = _load_cluster_filter(path)
    if allowed_cluster_labels is not None and cluster_labels:
        allow = {str(label) for label in allowed_cluster_labels}
        keep = np.array(
            [cluster_labels.get(int(cluster), "") in allow for cluster in spike_clusters.tolist()],
            dtype=np.bool_,
        )

    event_channels = None
    unit_positions = None
    templates_path = path / "templates.npy"
    if templates_path.exists():
        template_best_channels = _best_channel_from_templates(np.load(templates_path, mmap_mode="r"))
        cluster_best: Dict[int, int] = {}
        spike_templates_path = path / "spike_templates.npy"
        if spike_templates_path.exists():
            spike_templates = np.load(spike_templates_path, allow_pickle=False).reshape(-1)
            if spike_templates.shape == spike_clusters.shape:
                for cluster_id in np.unique(spike_clusters).tolist():
                    idx = np.flatnonzero(spike_clusters == int(cluster_id))
                    if idx.size == 0:
                        continue
                    tpl_vals, tpl_counts = np.unique(spike_templates[idx], return_counts=True)
                    best_template = int(tpl_vals[int(np.argmax(tpl_counts))])
                    if 0 <= best_template < template_best_channels.shape[0]:
                        cluster_best[int(cluster_id)] = int(template_best_channels[best_template])
        else:
            for cluster_id in np.unique(spike_clusters).tolist():
                if 0 <= int(cluster_id) < template_best_channels.shape[0]:
                    cluster_best[int(cluster_id)] = int(template_best_channels[int(cluster_id)])
        if cluster_best:
            event_channels = np.array(
                [cluster_best.get(int(cluster), -1) for cluster in spike_clusters.tolist()],
                dtype=np.int32,
            )
            if np.any(event_channels >= 0):
                event_channels = event_channels[keep]
            else:
                event_channels = None

            channel_pos_path = path / "channel_positions.npy"
            if channel_pos_path.exists():
                channel_positions = np.load(channel_pos_path, allow_pickle=False)
                if channel_positions.ndim == 2 and channel_positions.shape[0] > 0:
                    unit_positions = {}
                    for cluster_id, best_ch in cluster_best.items():
                        if 0 <= int(best_ch) < int(channel_positions.shape[0]):
                            unit_positions[int(cluster_id)] = np.asarray(channel_positions[int(best_ch)], dtype=np.float32)

    return EventSet(
        times_samples=_times_to_samples(spike_times[keep], fs=fs, times_unit="auto"),
        labels=_to_1d(spike_clusters[keep], dtype=np.int64),
        channels=event_channels,
        unit_positions=_normalize_unit_positions(unit_positions),
        metadata={"source": "phy_results_dir", "path": str(path), **({"fs": float(fs)} if fs is not None else {})},
    )


def _infer_unit_peak_channels_from_h5(h5: h5py.File) -> Dict[int, int]:
    if "voltage_peaks" not in h5:
        return {}
    peaks = np.asarray(h5["voltage_peaks"][:], dtype=np.float32)
    if peaks.ndim != 2 or peaks.shape[0] == 0 or peaks.shape[1] == 0:
        return {}
    peak_ch = np.argmax(np.abs(peaks), axis=1).astype(np.int32, copy=False)
    if "template_ids" in h5:
        template_ids = np.asarray(h5["template_ids"][:]).reshape(-1)
    else:
        template_ids = np.arange(peaks.shape[0], dtype=np.int64)
    if template_ids.size != peaks.shape[0]:
        template_ids = np.arange(peaks.shape[0], dtype=np.int64)
    out: Dict[int, int] = {}
    for idx, (uid_raw, ch_raw) in enumerate(zip(template_ids.tolist(), peak_ch.tolist())):
        out[int(uid_raw)] = int(ch_raw)
        out.setdefault(int(idx), int(ch_raw))
    if "spiketrains" in h5:
        unit_keys = []
        for key in h5["spiketrains"].keys():
            try:
                unit_keys.append(int(key))
            except ValueError:
                try:
                    unit_keys.append(int(str(key).split("_")[-1]))
                except Exception:
                    continue
        unit_keys = sorted(unit_keys)
        if len(unit_keys) == int(peak_ch.shape[0]):
            for unit_id, ch_raw in zip(unit_keys, peak_ch.tolist()):
                out[int(unit_id)] = int(ch_raw)
    return out


def load_mearec_h5(h5_path: str | Path) -> EventSet:
    path = Path(h5_path)
    reader = MEArecH5Reader(path)
    fs = float(reader.get_fs())
    try:
        channel_positions = np.asarray(reader.read_channel_positions(), dtype=np.float32)
    except Exception:
        channel_positions = None
    try:
        peak_channels = reader.read_unit_peak_channels()
    except Exception:
        peak_channels = {}

    times_parts = []
    label_parts = []
    channel_parts = []
    unit_positions = {}
    for unit_id, times_sec in reader.iter_spiketrains():
        times_samp = _times_to_samples(times_sec, fs=fs, times_unit="seconds")
        if times_samp.size == 0:
            continue
        times_parts.append(times_samp)
        label_parts.append(np.full((times_samp.size,), int(unit_id), dtype=np.int64))
        peak_ch = int(peak_channels.get(int(unit_id), -1))
        if peak_ch >= 0:
            channel_parts.append(np.full((times_samp.size,), peak_ch, dtype=np.int32))
            if channel_positions is not None and 0 <= peak_ch < int(channel_positions.shape[0]):
                unit_positions[int(unit_id)] = np.asarray(channel_positions[peak_ch], dtype=np.float32)

    if times_parts:
        times = np.concatenate(times_parts, axis=0)
        labels = np.concatenate(label_parts, axis=0)
        channels = np.concatenate(channel_parts, axis=0) if channel_parts and len(channel_parts) == len(times_parts) else None
    else:
        times = np.zeros((0,), dtype=np.int64)
        labels = np.zeros((0,), dtype=np.int64)
        channels = None

    return EventSet(
        times_samples=times,
        labels=labels,
        channels=channels,
        unit_positions=_normalize_unit_positions(unit_positions),
        metadata={"source": "mearec_h5", "path": str(path), "fs": fs},
    )


def _read_optional_dataset(h5: h5py.File, candidates: Sequence[str]) -> np.ndarray | None:
    for name in candidates:
        key = str(name).lstrip("/")
        if key in h5:
            obj = h5[key]
            if isinstance(obj, h5py.Dataset):
                return np.asarray(obj[:])
    return None


def _read_fs_from_h5(h5: h5py.File, explicit_fs: float | None = None) -> float | None:
    if explicit_fs is not None:
        return float(explicit_fs)
    for key in (
        "info/recordings/fs",
        "fs",
        "sampling_frequency",
        "sampling_rate",
        "recording/fs",
        "gt/fs",
        "sorting/fs",
    ):
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            raw = h5[key][()]
            return float(np.asarray(raw).reshape(()))
    return None


def _is_spiketrain_group(h5: h5py.File, group_path: str) -> bool:
    key = str(group_path).strip("/")
    if key not in h5 or not isinstance(h5[key], h5py.Group):
        return False
    grp = h5[key]
    for child_name in grp.keys():
        child = grp[child_name]
        if isinstance(child, h5py.Group) and "times" in child:
            return True
    return False


def load_generic_h5_events(
    h5_path: str | Path,
    *,
    schema: str = "auto",
    fs: float | None = None,
) -> EventSet:
    path = Path(h5_path)
    with h5py.File(path, "r") as h5:
        resolved_fs = _read_fs_from_h5(h5, explicit_fs=fs)
        schema_req = str(schema).lower()
        if schema_req == "auto":
            if _is_spiketrain_group(h5, "spiketrains"):
                schema_req = "spiketrains"
            elif _is_spiketrain_group(h5, "gt/spiketrains"):
                schema_req = "gt_spiketrains"
            elif "spike_times" in h5 and "spike_clusters" in h5:
                schema_req = "flat_sorting"
            elif "sorting/spike_times" in h5 and ("sorting/spike_labels" in h5 or "sorting/spike_clusters" in h5):
                schema_req = "sorting_group"
            else:
                raise ValueError(f"unsupported H5 schema in {path}")

        if schema_req in ("spiketrains", "gt_spiketrains"):
            group_key = "spiketrains" if schema_req == "spiketrains" else "gt/spiketrains"
            grp = h5[group_key]
            peak_channels = _infer_unit_peak_channels_from_h5(h5)
            channel_positions = _read_optional_dataset(h5, ["channel_positions"])
            times_parts = []
            label_parts = []
            channel_parts = []
            unit_positions = {}
            if resolved_fs is None:
                raise ValueError(f"fs is required to convert spiketrain times in {path}")
            for key in grp.keys():
                child = grp[key]
                if not isinstance(child, h5py.Group) or "times" not in child:
                    continue
                try:
                    unit_id = int(key)
                except ValueError:
                    unit_id = int(str(key).split("_")[-1])
                times_sec = np.asarray(child["times"][:], dtype=np.float64)
                times_parts.append(_times_to_samples(times_sec, fs=resolved_fs, times_unit="seconds"))
                label_parts.append(np.full((times_sec.shape[0],), int(unit_id), dtype=np.int64))
                peak_ch = int(peak_channels.get(int(unit_id), -1))
                if peak_ch >= 0:
                    channel_parts.append(np.full((times_sec.shape[0],), peak_ch, dtype=np.int32))
                    if channel_positions is not None and 0 <= peak_ch < int(channel_positions.shape[0]):
                        unit_positions[int(unit_id)] = np.asarray(channel_positions[int(peak_ch)], dtype=np.float32)
            times = np.concatenate(times_parts, axis=0) if times_parts else np.zeros((0,), dtype=np.int64)
            labels = np.concatenate(label_parts, axis=0) if label_parts else np.zeros((0,), dtype=np.int64)
            channels = np.concatenate(channel_parts, axis=0) if channel_parts and len(channel_parts) == len(times_parts) else None
            return EventSet(
                times_samples=times,
                labels=labels,
                channels=channels,
                unit_positions=_normalize_unit_positions(unit_positions),
                metadata={"source": f"generic_h5:{schema_req}", "path": str(path), **({"fs": resolved_fs} if resolved_fs is not None else {})},
            )

        if schema_req in ("flat_sorting", "sorting_group"):
            prefix = "" if schema_req == "flat_sorting" else "sorting/"
            times = _read_optional_dataset(h5, [f"{prefix}spike_times"])
            labels = _read_optional_dataset(h5, [f"{prefix}spike_labels", f"{prefix}spike_clusters"])
            if times is None or labels is None:
                raise ValueError(f"missing spike_times/spike_labels in {path}")
            channels = _read_optional_dataset(
                h5,
                [
                    f"{prefix}spike_channels",
                    f"{prefix}center_channel",
                    f"{prefix}channels",
                ],
            )
            event_set = load_event_arrays(
                times,
                labels,
                channels=channels,
                fs=resolved_fs,
                times_unit="auto",
                metadata={"source": f"generic_h5:{schema_req}", "path": str(path), **({"fs": resolved_fs} if resolved_fs is not None else {})},
            )
            if event_set.channels is not None:
                channel_positions = _read_optional_dataset(h5, ["channel_positions", "sorting/channel_positions"])
                if channel_positions is not None and channel_positions.ndim == 2:
                    unit_positions = {}
                    for unit_id in np.unique(event_set.labels).tolist():
                        idx = np.flatnonzero(event_set.labels == int(unit_id))
                        if idx.size == 0:
                            continue
                        ch_vals, ch_counts = np.unique(event_set.channels[idx], return_counts=True)
                        best_ch = int(ch_vals[int(np.argmax(ch_counts))])
                        if 0 <= best_ch < int(channel_positions.shape[0]):
                            unit_positions[int(unit_id)] = np.asarray(channel_positions[best_ch], dtype=np.float32)
                    event_set.unit_positions = _normalize_unit_positions(unit_positions)
            return event_set

        raise ValueError(f"unsupported H5 schema={schema}")


def load_prediction_events(
    source: str | Path | EventSet,
    *,
    format: str = "auto",
    fs: float | None = None,
    allowed_cluster_labels: Sequence[str] | None = None,
) -> EventSet:
    if isinstance(source, EventSet):
        return source
    path = Path(source)
    fmt = str(format).lower()
    if fmt == "auto":
        if path.is_dir():
            fmt = "phy"
        elif path.suffix.lower() == ".npz":
            fmt = "mysort_npz"
        elif path.suffix.lower() in (".h5", ".hdf5"):
            fmt = "generic_h5"
        else:
            raise ValueError(f"cannot infer prediction format from {path}")
    if fmt in ("mysort", "mysort_npz", "npz"):
        return load_mysort_npz(path, fs=fs)
    if fmt in ("phy", "phy_results_dir", "kilosort"):
        return load_phy_results_dir(path, fs=fs, allowed_cluster_labels=allowed_cluster_labels)
    if fmt in ("generic_h5", "h5"):
        return load_generic_h5_events(path, schema="auto", fs=fs)
    raise ValueError(f"unsupported prediction format={format}")


def load_gt_events(
    source: str | Path | EventSet,
    *,
    format: str = "auto",
    fs: float | None = None,
) -> EventSet:
    if isinstance(source, EventSet):
        return source
    path = Path(source)
    fmt = str(format).lower()
    if fmt == "auto":
        if path.is_dir():
            fmt = "phy"
        elif path.suffix.lower() in (".h5", ".hdf5"):
            try:
                return load_mearec_h5(path)
            except Exception:
                return load_generic_h5_events(path, schema="auto", fs=fs)
        elif path.suffix.lower() == ".npz":
            fmt = "mysort_npz"
        else:
            raise ValueError(f"cannot infer GT format from {path}")
    if fmt in ("mearec", "mearec_h5"):
        return load_mearec_h5(path)
    if fmt in ("generic_h5", "h5"):
        return load_generic_h5_events(path, schema="auto", fs=fs)
    if fmt in ("phy", "phy_results_dir", "kilosort"):
        return load_phy_results_dir(path, fs=fs)
    if fmt in ("mysort", "mysort_npz", "npz"):
        return load_mysort_npz(path, fs=fs)
    raise ValueError(f"unsupported GT format={format}")


def resolve_metadata(*event_sets: EventSet, fs: float | None = None) -> Dict[str, Any]:
    fs_values: List[float] = []
    if fs is not None:
        fs_values.append(float(fs))
    sources: List[str] = []
    paths: List[str] = []
    for event_set in event_sets:
        meta = dict(event_set.metadata or {})
        if "fs" in meta:
            fs_values.append(float(meta["fs"]))
        if "source" in meta:
            sources.append(str(meta["source"]))
        if "path" in meta:
            paths.append(str(meta["path"]))
    resolved_fs = None
    for value in fs_values:
        if resolved_fs is None:
            resolved_fs = float(value)
            continue
        if not np.isclose(float(value), float(resolved_fs), rtol=0.0, atol=1e-6):
            raise ValueError(f"inconsistent fs values: {resolved_fs} vs {value}")
    return {
        "fs": resolved_fs,
        "sources": sources,
        "paths": paths,
    }


__all__ = [
    "load_event_arrays",
    "load_generic_h5_events",
    "load_gt_events",
    "load_mearec_h5",
    "load_mysort_npz",
    "load_phy_results_dir",
    "load_prediction_events",
    "resolve_metadata",
]
