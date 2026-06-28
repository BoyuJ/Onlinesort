from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import h5py
import numpy as np


OPERATIONAL_SOURCE_NAME_TO_CODE: Dict[str, int] = {
    "primary_matched_positive": 0,
    "secondary_matched_positive": 1,
    "detector_negative": 2,
}
OPERATIONAL_SOURCE_CODE_TO_NAME: Dict[int, str] = {
    code: name for name, code in OPERATIONAL_SOURCE_NAME_TO_CODE.items()
}
OPERATIONAL_SOURCE_SCHEMA_VERSION = "operational_detector_matched_v1"
IGNORED_SAMPLE_SOURCE = -1


@dataclass(frozen=True)
class Recording:
    waveforms: np.ndarray
    labels: np.ndarray | None
    times: np.ndarray
    peak_channel_id: np.ndarray
    peak_amplitude: np.ndarray
    quality_flag: np.ndarray
    recording_id: int
    fs_hz: float


def build_global_unit_ids(local_unit_id: np.ndarray, recording_id: np.ndarray) -> np.ndarray:
    """Map recording-local unit ids to a stable contiguous global id space."""
    local = np.asarray(local_unit_id, dtype=np.int64).reshape(-1)
    rec = np.asarray(recording_id, dtype=np.int64).reshape(-1)
    if local.shape[0] != rec.shape[0]:
        raise ValueError("local_unit_id and recording_id must have matching shape")
    out = np.full(local.shape, -1, dtype=np.int64)
    positive = local >= 0
    if not np.any(positive):
        return out
    pairs = sorted({(int(r), int(u)) for r, u in zip(rec[positive].tolist(), local[positive].tolist())})
    mapping = {pair: idx for idx, pair in enumerate(pairs)}
    for idx in np.flatnonzero(positive).tolist():
        out[int(idx)] = int(mapping[(int(rec[int(idx)]), int(local[int(idx)]))])
    return out


def ensure_global_unit_id_labels(labels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return labels with explicit local/global unit ids and unit_id set to global."""
    out = dict(labels)
    if "unit_id" not in out:
        raise KeyError("labels must contain unit_id")
    unit = np.asarray(out["unit_id"], dtype=np.int64)
    rec = np.asarray(out.get("recording_id", np.zeros(unit.shape, dtype=np.int16)), dtype=np.int64)
    if rec.shape[0] != unit.shape[0]:
        raise ValueError("recording_id must match unit_id shape")
    local = np.asarray(out.get("local_unit_id", unit), dtype=np.int64)
    if local.shape[0] != unit.shape[0]:
        raise ValueError("local_unit_id must match unit_id shape")
    if "global_unit_id" in out:
        global_unit = np.asarray(out["global_unit_id"], dtype=np.int64)
        if global_unit.shape[0] != unit.shape[0]:
            raise ValueError("global_unit_id must match unit_id shape")
    else:
        global_unit = build_global_unit_ids(local, rec)
    out["local_unit_id"] = local.astype(np.int64, copy=False)
    out["global_unit_id"] = global_unit.astype(np.int64, copy=False)
    out["unit_id"] = global_unit.astype(np.int64, copy=False)
    return out


def _list_h5_datasets(h5: h5py.File) -> List[str]:
    paths: List[str] = []

    def _walk(obj: h5py.Group, prefix: str) -> None:
        for k, v in obj.items():
            cur = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Dataset):
                paths.append(cur)
            elif isinstance(v, h5py.Group):
                _walk(v, cur)

    _walk(h5, "")
    return paths


def _find_h5_dataset(h5: h5py.File, candidates: List[str], *, ndim: int | None = None) -> np.ndarray | None:
    # Try exact path first.
    for c in candidates:
        key = c.lstrip("/")
        if key in h5 and isinstance(h5[key], h5py.Dataset):
            arr = h5[key][:]
            if ndim is None or arr.ndim == ndim:
                return arr
    # Fallback: suffix/path-name match.
    all_paths = _list_h5_datasets(h5)
    cand_norm = [c.lstrip("/").lower() for c in candidates]
    for p in all_paths:
        pl = p.lower()
        if any(pl == c or pl.endswith("/" + c) for c in cand_norm):
            arr = h5[p][:]
            if ndim is None or arr.ndim == ndim:
                return arr
    return None


def _infer_template_axis(shape: Tuple[int, ...], n_ids: int) -> int:
    axes = [i for i, s in enumerate(shape) if s == n_ids]
    if len(axes) == 1:
        return axes[0]
    if len(axes) > 1:
        return axes[0]
    return 0


def _to_tkc_and_ids(templates: np.ndarray, template_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(templates, dtype=np.float32)
    ids = np.asarray(template_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError("template_ids is empty")

    if arr.ndim < 3:
        raise ValueError(f"unsupported templates ndim={arr.ndim}, expected >=3")

    # Move template axis to front.
    id_axis = _infer_template_axis(tuple(arr.shape), ids.size)
    arr = np.moveaxis(arr, id_axis, 0)  # [N, ...]
    if arr.shape[0] != ids.size:
        raise ValueError(
            f"template_ids size mismatch: ids={ids.size}, inferred_template_axis={id_axis}, templates_shape={arr.shape}"
        )

    tail = list(arr.shape[1:])
    rel_axes = list(range(1, arr.ndim))

    # Choose channel axis first: prefer exact 384 for NP384, else largest dimension.
    ch_candidates = [ax for ax in rel_axes if arr.shape[ax] == 384]
    if ch_candidates:
        ch_axis = ch_candidates[0]
    else:
        ch_axis = max(rel_axes, key=lambda ax: arr.shape[ax])

    # Choose time axis from remaining: prefer exact 224, else largest remaining dimension.
    rem_after_ch = [ax for ax in rel_axes if ax != ch_axis]
    t_candidates = [ax for ax in rem_after_ch if arr.shape[ax] == 224]
    if t_candidates:
        t_axis = t_candidates[0]
    else:
        t_axis = max(rem_after_ch, key=lambda ax: arr.shape[ax])

    variant_axes = [ax for ax in rel_axes if ax not in (t_axis, ch_axis)]
    perm = [0] + variant_axes + [t_axis, ch_axis]
    arr = np.transpose(arr, perm)

    n = arr.shape[0]
    t = arr.shape[-2]
    c = arr.shape[-1]
    variant_size = int(np.prod(arr.shape[1:-2])) if arr.ndim > 3 else 1
    out = arr.reshape(n * variant_size, t, c)
    out_ids = np.repeat(ids, variant_size)
    return out, out_ids


def _top_k_channels(patches_tkc: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or patches_tkc.shape[2] <= k:
        return patches_tkc
    energy = np.max(np.abs(patches_tkc), axis=1)  # [N, C]
    idx = np.argpartition(-energy, k - 1, axis=1)[:, :k]
    ord_local = np.argsort(-np.take_along_axis(energy, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, ord_local, axis=1)
    return np.take_along_axis(patches_tkc, idx[:, None, :], axis=2)


def _normalize_patch(patches_tkc: np.ndarray) -> np.ndarray:
    x = patches_tkc.astype(np.float32, copy=False)
    mean = np.mean(x, axis=(1, 2), keepdims=True)
    std = np.std(x, axis=(1, 2), keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def _estimate_channel_robust_stats(
    recordings: h5py.Dataset,
    *,
    sample_count: int = 60000,
) -> Tuple[np.ndarray, np.ndarray]:
    n_samples = int(recordings.shape[0])
    n_channels = int(recordings.shape[1])
    if n_samples <= 0 or n_channels <= 0:
        return np.zeros((n_channels,), dtype=np.float32), np.ones((n_channels,), dtype=np.float32)

    target = max(1, int(sample_count))
    step = max(1, n_samples // target)
    sampled = recordings[::step, :].astype(np.float32, copy=False)
    if sampled.shape[0] > target:
        sampled = sampled[:target]
    if sampled.shape[0] == 0:
        sampled = recordings[:1, :].astype(np.float32, copy=False)

    med = np.median(sampled, axis=0).astype(np.float32, copy=False)
    mad = np.median(np.abs(sampled - med[None, :]), axis=0).astype(np.float32, copy=False)
    sigma = np.maximum(1.4826 * mad, 1e-6).astype(np.float32, copy=False)
    return med, sigma


def _subsample_spike_samples(
    samples: np.ndarray,
    *,
    max_per_unit: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(samples, dtype=np.int64)
    if out.size == 0:
        return out
    out = np.sort(out)
    cap = int(max_per_unit)
    if cap <= 0 or out.size <= cap:
        return out
    mode = str(strategy).lower()
    if mode in ("uniform_time", "uniform", "coverage", "time_uniform"):
        step = float(out.size) / float(cap)
        idx = np.floor((np.arange(cap, dtype=np.float64) + 0.5) * step).astype(np.int64)
        idx = np.clip(idx, 0, out.size - 1)
        return out[idx]
    sel = rng.choice(out, size=cap, replace=False)
    return np.sort(sel.astype(np.int64, copy=False))


def compute_strict_overlap_label(
    event_time_sample: np.ndarray,
    peak_channel: np.ndarray,
    unit_id: np.ndarray,
    spike_label: np.ndarray,
    recording_id: np.ndarray,
    channel_positions: Any,
    fs: float,
    strict_overlap_ms: float = 0.5,
    strict_overlap_um: float = 60.0,
) -> np.ndarray:
    """Mark positive events that are very close in time and space within a recording."""
    t = np.asarray(event_time_sample, dtype=np.int64).reshape(-1)
    ch = np.asarray(peak_channel, dtype=np.int64).reshape(-1)
    uid = np.asarray(unit_id, dtype=np.int64).reshape(-1)
    spike = np.asarray(spike_label, dtype=np.float32).reshape(-1)
    rec = np.asarray(recording_id, dtype=np.int64).reshape(-1)
    if not (t.shape == ch.shape == uid.shape == spike.shape == rec.shape):
        raise ValueError("strict overlap inputs must have matching 1D shapes")

    pos: np.ndarray | None = None
    pos_by_rec: np.ndarray | None = None
    pos_list: List[np.ndarray] | None = None
    if isinstance(channel_positions, (list, tuple)):
        pos_list = []
        for rec_pos in channel_positions:
            arr = np.asarray(rec_pos, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError("channel_positions list entries must have shape [C, D]")
            pos_list.append(arr)
    else:
        pos = np.asarray(channel_positions, dtype=np.float32)
        if pos.ndim == 2:
            pos_by_rec = None
        elif pos.ndim == 3:
            pos_by_rec = pos
        else:
            raise ValueError("channel_positions must have shape [C, D], [R, C, D], or a list of [C, D] arrays")

    out = np.zeros(t.shape, dtype=np.bool_)
    dt_max = max(0, int(round(float(strict_overlap_ms) * float(fs) / 1000.0)))
    dist_max = float(strict_overlap_um)
    if t.size == 0 or dt_max < 0 or dist_max < 0.0:
        return out

    positive = (uid >= 0) & (spike > 0.5) & (ch >= 0) & (t >= 0)
    if not np.any(positive):
        return out

    for rec_value in np.unique(rec[positive]).tolist():
        idx = np.flatnonzero(positive & (rec == int(rec_value))).astype(np.int64, copy=False)
        if pos_list is not None:
            if int(rec_value) < 0 or int(rec_value) >= len(pos_list):
                continue
            rec_pos = pos_list[int(rec_value)]
        else:
            assert pos is not None
            if pos_by_rec is None:
                rec_pos = pos
            elif int(rec_value) < 0 or int(rec_value) >= pos_by_rec.shape[0]:
                continue
            else:
                rec_pos = pos_by_rec[int(rec_value)]
        n_channels = int(rec_pos.shape[0])
        idx = idx[ch[idx] < n_channels]
        if idx.size < 2:
            continue
        order = np.argsort(t[idx], kind="mergesort")
        idx_sorted = idx[order]
        t_sorted = t[idx_sorted]
        ch_sorted = ch[idx_sorted]
        left = 0
        for right in range(idx_sorted.size):
            while left < right and int(t_sorted[right] - t_sorted[left]) > dt_max:
                left += 1
            if right <= left:
                continue
            cand = np.arange(left, right, dtype=np.int64)
            c0 = int(ch_sorted[right])
            cands_ch = ch_sorted[cand].astype(np.int64, copy=False)
            delta = rec_pos[cands_ch] - rec_pos[c0]
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            near = cand[dist <= dist_max]
            if near.size > 0:
                out[idx_sorted[right]] = True
                out[idx_sorted[near]] = True
    return out


def _pick_peak_channel_and_time(
    seg_align: np.ndarray,
    *,
    ch_med: np.ndarray,
    polarity: str,
) -> Tuple[int, int, float, float]:
    """
    Pick a peak channel first, then the peak time on that channel.
    This avoids selecting a single global extremum across all channels/times.
    """
    seg = np.asarray(seg_align, dtype=np.float32)
    if seg.ndim != 2 or seg.shape[0] == 0 or seg.shape[1] == 0:
        raise ValueError("seg_align must be [T, C] with non-zero shape")

    med = np.asarray(ch_med, dtype=np.float32).reshape(1, -1)
    if med.shape[1] != seg.shape[1]:
        raise ValueError(f"ch_med size mismatch: med={med.shape[1]} seg_channels={seg.shape[1]}")

    centered = seg - med
    mode = str(polarity).lower()
    c_idx = np.arange(seg.shape[1], dtype=np.int64)

    if mode in ("negative", "neg", "min"):
        t_idx = np.argmin(centered, axis=0).astype(np.int64, copy=False)
        peak_centered = centered[t_idx, c_idx]
        amp = -peak_centered
    elif mode in ("positive", "pos", "max"):
        t_idx = np.argmax(centered, axis=0).astype(np.int64, copy=False)
        peak_centered = centered[t_idx, c_idx]
        amp = peak_centered
    else:
        abs_centered = np.abs(centered)
        t_idx = np.argmax(abs_centered, axis=0).astype(np.int64, copy=False)
        peak_centered = centered[t_idx, c_idx]
        amp = np.abs(peak_centered)

    amp = np.maximum(amp.astype(np.float32, copy=False), 0.0)
    c_peak = int(np.argmax(amp))
    t_rel = int(t_idx[c_peak])
    peak_val = float(seg[t_rel, c_peak])
    peak_amp = float(amp[c_peak])
    return t_rel, c_peak, peak_val, peak_amp


def _resolve_pretrain_car_mode(n_channels: int, car_mode: str = "auto") -> str:
    mode = str(car_mode or "auto").lower()
    if mode == "auto":
        return "none" if int(n_channels) == 1 else "global"
    if mode not in ("none", "global"):
        raise ValueError(f"unsupported car_mode={car_mode!r}; use auto | none | global")
    if mode == "global" and int(n_channels) == 1:
        raise ValueError("car_mode='global' would zero a single-channel recording; use car_mode='auto' or 'none'")
    return mode


def _normalize_gt_time_reference(gt_time_reference: str) -> str:
    mode = str(gt_time_reference or "near_peak").lower()
    if mode in ("near_peak", "peak", "centered", "symmetric"):
        return "near_peak"
    if mode in ("onset", "start", "forward"):
        return "onset"
    raise ValueError(f"unsupported gt_time_reference={gt_time_reference!r}; use near_peak | onset")


def _alignment_window_bounds(
    *,
    t_anchor: int,
    align_radius_samples: int,
    n_samples_total: int,
    gt_time_reference: str,
) -> Tuple[int, int]:
    radius = max(1, int(align_radius_samples))
    t = int(t_anchor)
    n = int(n_samples_total)
    mode = _normalize_gt_time_reference(gt_time_reference)
    if mode == "onset":
        return max(0, t), min(n, t + radius + 1)
    return max(0, t - radius), min(n, t + radius + 1)


def _read_h5_spiketrain_unit_ids(f: h5py.File) -> List[int]:
    out: List[int] = []
    if "spiketrains" not in f:
        return out
    for key in f["spiketrains"].keys():
        try:
            out.append(int(key))
        except ValueError:
            try:
                out.append(int(str(key).split("_")[-1]))
            except Exception:
                continue
    return sorted(out)


def _read_unit_snr_annotations_from_h5(f: h5py.File) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if "spiketrains" not in f:
        return out
    for key, group in f["spiketrains"].items():
        try:
            uid = int(key)
        except ValueError:
            try:
                uid = int(str(key).split("_")[-1])
            except Exception:
                continue
        try:
            ann = group["annotations"]
            if "snr" not in ann:
                continue
            snr = float(ann["snr"][()])
        except Exception:
            continue
        out[int(uid)] = snr
    return out


def _top_k_channel_indices(scores_tc: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores_tc, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.int32)
    k_eff = max(1, min(int(k), int(scores.shape[1])))
    if k_eff >= int(scores.shape[1]):
        return np.argsort(-scores, axis=1).astype(np.int32, copy=False)
    idx = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
    ord_local = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, ord_local, axis=1).astype(np.int32, copy=False)


def _map_unit_channel_sets(
    unit_channels: np.ndarray,
    *,
    template_ids: np.ndarray,
    spiketrain_unit_ids: List[int],
) -> Dict[int, np.ndarray]:
    arr = np.asarray(unit_channels, dtype=np.int32)
    ids = np.asarray(template_ids, dtype=np.int64).reshape(-1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[0] != ids.size:
        return {}
    out: Dict[int, np.ndarray] = {}
    for idx, (uid_raw, ch_raw) in enumerate(zip(ids.tolist(), arr.tolist())):
        ch_arr = np.asarray(ch_raw, dtype=np.int32)
        out[int(uid_raw)] = ch_arr
        out.setdefault(int(idx), ch_arr)
    if len(spiketrain_unit_ids) == int(arr.shape[0]):
        for uid, ch_raw in zip(spiketrain_unit_ids, arr):
            out[int(uid)] = np.asarray(ch_raw, dtype=np.int32)
    return out


def _read_unit_match_channels_from_h5(f: h5py.File, *, top_k: int = 9) -> Dict[int, np.ndarray]:
    k = max(1, int(top_k))
    spiketrain_unit_ids = _read_h5_spiketrain_unit_ids(f)
    if "template_ids" in f:
        template_ids = np.asarray(f["template_ids"][:]).reshape(-1)
    else:
        template_ids = np.zeros((0,), dtype=np.int64)

    if "voltage_peaks" in f:
        peaks = np.asarray(f["voltage_peaks"][:], dtype=np.float32)
        if peaks.ndim == 2 and peaks.shape[0] > 0 and peaks.shape[1] > 0:
            ids = template_ids
            if ids.size != peaks.shape[0]:
                ids = np.arange(peaks.shape[0], dtype=np.int64)
            top_idx = _top_k_channel_indices(np.abs(peaks), k)
            mapped = _map_unit_channel_sets(top_idx, template_ids=ids, spiketrain_unit_ids=spiketrain_unit_ids)
            if mapped:
                return mapped

    templates = _find_h5_dataset(
        f,
        [
            "templates",
            "recordings/templates",
            "gt/templates",
        ],
    )
    if templates is None:
        return {}
    ids = template_ids
    if ids.size == 0:
        fallback_n = int(len(spiketrain_unit_ids)) if spiketrain_unit_ids else int(np.asarray(templates).shape[0])
        ids = np.arange(fallback_n, dtype=np.int64)
    try:
        patches_tkc, patch_ids = _to_tkc_and_ids(templates, ids)
    except Exception:
        return {}
    if patches_tkc.size == 0 or patch_ids.size == 0:
        return {}
    energy_tc = np.max(np.abs(patches_tkc), axis=1)
    uniq_ids = np.unique(patch_ids)
    scores_by_unit = np.zeros((uniq_ids.size, energy_tc.shape[1]), dtype=np.float32)
    for i, uid in enumerate(uniq_ids.tolist()):
        mask = patch_ids == uid
        scores_by_unit[i] = np.max(energy_tc[mask], axis=0)
    top_idx = _top_k_channel_indices(scores_by_unit, k)
    return _map_unit_channel_sets(top_idx, template_ids=uniq_ids, spiketrain_unit_ids=spiketrain_unit_ids)


def _read_unit_peak_channels_from_h5(f: h5py.File) -> Dict[int, int]:
    unit_match_channels = _read_unit_match_channels_from_h5(f, top_k=1)
    out: Dict[int, int] = {}
    for uid, chs in unit_match_channels.items():
        arr = np.asarray(chs, dtype=np.int32).reshape(-1)
        if arr.size > 0:
            out[int(uid)] = int(arr[0])
    return out


def _normalize_candidate_patch(
    patch: np.ndarray,
    *,
    normalize: bool,
    norm_mode: str,
    neigh: np.ndarray,
    ch_med: np.ndarray,
    ch_sigma: np.ndarray,
    sigma_evt: np.ndarray | None = None,
) -> np.ndarray:
    out = np.asarray(patch, dtype=np.float32, copy=True)
    if not normalize:
        return out
    if norm_mode in ("peak_channel_sigma", "peak_sigma", "baseline_sigma"):
        # Default event center is at sample index 20 for current patches.
        peak_index = min(20, max(0, out.shape[0] - 1))
        offsets = np.arange(out.shape[0], dtype=np.int32) - int(peak_index)
        baseline_mask = (offsets >= -20) & (offsets <= -6)
        if not np.any(baseline_mask):
            baseline_mask[: min(out.shape[0], max(1, peak_index))] = True
        baseline = np.median(out[baseline_mask], axis=0).astype(np.float32, copy=False)
        if sigma_evt is not None and sigma_evt.shape[0] == neigh.shape[0]:
            scale = np.maximum(np.asarray(sigma_evt, dtype=np.float32), 1e-6)
        else:
            scale = np.maximum(ch_sigma[neigh], 1e-6)
        out = (out - baseline[None, :]) / scale[None, :]
    elif norm_mode in ("robust_channel", "channel_robust", "global_channel", "per_spike", "patch_zscore", "zscore"):
        raise ValueError("legacy waveform normalization is disabled; use peak_channel_sigma")
    return out.astype(np.float32, copy=False)


def _realign_detector_event(
    *,
    event_time_sample: int,
    event_center_channel: int,
    seg: np.ndarray,
    seg_t0: int,
    fs: float,
    ch_med: np.ndarray,
) -> Tuple[int, int, int]:
    local_t = int(event_time_sample - seg_t0)
    align_r = max(1, int(round(0.2 * fs / 1000.0)))
    a_lo_local = max(0, local_t - align_r)
    a_hi_local = min(seg.shape[0], local_t + align_r + 1)
    if a_hi_local <= a_lo_local:
        local_t_aligned = local_t
        c_aligned = int(event_center_channel)
    else:
        seg_det = seg[a_lo_local:a_hi_local, :]
        try:
            t_rel_det, c_aligned, _, _ = _pick_peak_channel_and_time(
                seg_det, ch_med=ch_med, polarity="negative",
            )
            local_t_aligned = a_lo_local + t_rel_det
        except (ValueError, IndexError):
            local_t_aligned = local_t
            c_aligned = int(event_center_channel)
    t_aligned = int(seg_t0 + local_t_aligned)
    return local_t_aligned, t_aligned, int(c_aligned)


def _detector_event_channel_set(event: Any, *, c_aligned: int, neighbors_lookup: np.ndarray) -> np.ndarray:
    n_channels = int(neighbors_lookup.shape[0])
    parts: List[np.ndarray] = [
        np.asarray([int(getattr(event, "c0", c_aligned)), int(c_aligned)], dtype=np.int32),
    ]
    support_channels = getattr(event, "support_channels", None)
    if support_channels is not None:
        parts.append(np.asarray(support_channels, dtype=np.int32).reshape(-1))
    event_neighbors = getattr(event, "neighbors", None)
    if event_neighbors is not None:
        parts.append(np.asarray(event_neighbors, dtype=np.int32).reshape(-1))
    for c in (int(getattr(event, "c0", c_aligned)), int(c_aligned)):
        if 0 <= c < n_channels:
            parts.append(np.asarray(neighbors_lookup[c], dtype=np.int32).reshape(-1))
    if not parts:
        return np.asarray([int(c_aligned)], dtype=np.int32)
    merged = np.concatenate(parts, axis=0).astype(np.int32, copy=False)
    merged = merged[(merged >= 0) & (merged < n_channels)]
    if merged.size == 0:
        return np.asarray([int(np.clip(c_aligned, 0, max(0, n_channels - 1)))], dtype=np.int32)
    return np.unique(merged).astype(np.int32, copy=False)


def _detector_event_stats_row(event: Any, *, snr_peak: float) -> np.ndarray:
    return np.asarray(
        [
            float(getattr(event, "amp_norm", snr_peak)),
            float(getattr(event, "spatial_score", 1.0)),
            float(getattr(event, "support_count", 1)),
            float(getattr(event, "simul_global", 0)),
            float(getattr(event, "simul_local", 0)),
        ],
        dtype=np.float32,
    )


def _build_peak_aligned_spiketrains(
    *,
    spiketrains: Iterable[Tuple[int, np.ndarray]],
    filtered_recordings: np.ndarray | None,
    fs: float,
    ch_med: np.ndarray,
    align_radius_ms: float,
    polarity: str,
    gt_time_reference: str = "near_peak",
) -> Tuple[List[Tuple[int, np.ndarray]], np.ndarray, Dict[str, int]]:
    stats = {
        "gt_align_total": 0,
        "gt_align_boundary_hits": 0,
        "gt_align_max_abs_shift": 0,
    }
    spiketrains_list = [
        (int(uid), np.asarray(times_sec, dtype=np.float64).reshape(-1))
        for uid, times_sec in spiketrains
    ]
    if filtered_recordings is None or not spiketrains_list:
        raw_parts: List[np.ndarray] = []
        for _, times_sec in spiketrains_list:
            t_samples = np.round(times_sec * float(fs)).astype(np.int64, copy=False)
            if t_samples.size > 0:
                raw_parts.append(t_samples)
        if raw_parts:
            gt_times_all = np.sort(np.concatenate(raw_parts, axis=0).astype(np.int64, copy=False), kind="mergesort")
        else:
            gt_times_all = np.zeros((0,), dtype=np.int64)
        return spiketrains_list, gt_times_all, stats

    align_r = max(1, int(round(float(align_radius_ms) * float(fs) / 1000.0)))
    time_ref = _normalize_gt_time_reference(gt_time_reference)
    n_samples_total = int(filtered_recordings.shape[0])
    aligned_spiketrains: List[Tuple[int, np.ndarray]] = []
    aligned_parts: List[np.ndarray] = []
    for uid, times_sec in spiketrains_list:
        t_samples = np.round(times_sec * float(fs)).astype(np.int64, copy=False)
        if t_samples.size == 0:
            aligned_spiketrains.append((int(uid), times_sec))
            continue
        aligned_samples = np.empty_like(t_samples)
        for idx, t_anchor in enumerate(t_samples.tolist()):
            stats["gt_align_total"] += 1
            a_lo, a_hi = _alignment_window_bounds(
                t_anchor=int(t_anchor),
                align_radius_samples=align_r,
                n_samples_total=n_samples_total,
                gt_time_reference=time_ref,
            )
            if a_hi <= a_lo:
                aligned_samples[idx] = int(t_anchor)
                continue
            seg_align = filtered_recordings[a_lo:a_hi, :].astype(np.float32, copy=False)
            if seg_align.size == 0:
                aligned_samples[idx] = int(t_anchor)
                continue
            try:
                t_rel, _, _, _ = _pick_peak_channel_and_time(
                    seg_align,
                    ch_med=ch_med,
                    polarity=polarity,
                )
            except (ValueError, IndexError):
                aligned_samples[idx] = int(t_anchor)
                continue
            if t_rel == 0 or t_rel == (seg_align.shape[0] - 1):
                stats["gt_align_boundary_hits"] += 1
            t_peak = int(a_lo + t_rel)
            aligned_samples[idx] = t_peak
            dt = abs(int(t_peak) - int(t_anchor))
            if dt > stats["gt_align_max_abs_shift"]:
                stats["gt_align_max_abs_shift"] = int(dt)
        aligned_parts.append(aligned_samples)
        aligned_spiketrains.append((int(uid), aligned_samples.astype(np.float64) / float(fs)))
    if aligned_parts:
        gt_times_all = np.sort(np.concatenate(aligned_parts, axis=0).astype(np.int64, copy=False), kind="mergesort")
    else:
        gt_times_all = np.zeros((0,), dtype=np.int64)
    return aligned_spiketrains, gt_times_all, stats


def _prefilter_recording(
    recordings,
    *,
    fs: float,
    n_channels: int,
    car_mode: str = "auto",
    chunk_samples: int = 0,
    progress_cb: Callable[[str], None] | None = None,
) -> np.ndarray:
    from .stream import StreamBandpass, StreamCAR

    n_samples = int(recordings.shape[0])
    resolved_car_mode = _resolve_pretrain_car_mode(n_channels, car_mode)
    if chunk_samples <= 0:
        chunk_samples = max(int(fs), 32000)
    bandpass = StreamBandpass(fs=fs, low_hz=300.0, high_hz=6000.0, order=3, n_channels=n_channels)
    car = StreamCAR(mode=resolved_car_mode)
    filtered = np.zeros((n_samples, n_channels), dtype=np.float32)
    for t0 in range(0, n_samples, chunk_samples):
        t1 = min(t0 + chunk_samples, n_samples)
        x = recordings[t0:t1].astype(np.float32, copy=False)
        filtered[t0:t1] = car.process(bandpass.process(x))
    return filtered


def load_mearec_template_patches(
    h5_path: str | Path,
    *,
    top_k_channels: int = 16,
    normalize: bool = True,
    left: int = 20,
    right: int = 44,
    electrode_config: str | Path | None = None,
    peak_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(h5_path)

    reader = MEArecH5Reader(path)
    cfg_path = None
    if electrode_config is None:
        cfg_path = Path(__file__).resolve().parents[1] / "configs" / "electrodes" / "neuropixels_384.yaml"
    else:
        cfg_path = Path(electrode_config)
        if not cfg_path.is_absolute():
            cfg_path = Path(__file__).resolve().parents[1] / cfg_path
    electrode_cfg = None
    if cfg_path is not None and cfg_path.exists():
        from .utils import load_yaml

        electrode_cfg = load_yaml(cfg_path)

    try:
        h5_positions = reader.read_channel_positions()
    except Exception:
        h5_positions = None

    from src.utils import relative_geo, resolve_channel_geometry

    positions, neighbors_lookup, _, _ = resolve_channel_geometry(
        n_channels=reader.get_recordings_shape()[1],
        fallback_k=top_k_channels,
        electrode_cfg=electrode_cfg,
        h5_positions=h5_positions,
    )
    if positions is None or neighbors_lookup is None:
        raise KeyError("channel_positions not found in h5 and electrode_config unavailable")

    with h5py.File(path, "r") as f:
        templates = _find_h5_dataset(
            f,
            candidates=[
                "templates",
                "template_waveforms",
                "recordings/templates",
                "recordings/template_waveforms",
                "gt/templates",
            ],
        )
        template_ids = _find_h5_dataset(
            f,
            candidates=[
                "template_ids",
                "templates/template_ids",
                "info/templates/template_ids",
                "recordings/template_ids",
                "gt/template_ids",
            ],
            ndim=1,
        )
        if templates is None:
            paths = _list_h5_datasets(f)
            raise KeyError(f"cannot find templates dataset in {path}. available datasets: {paths[:20]}")
        if template_ids is None:
            paths = _list_h5_datasets(f)
            raise KeyError(f"cannot find template_ids dataset in {path}. available datasets: {paths[:20]}")

    patches_full, unit_ids = _to_tkc_and_ids(templates, template_ids)
    
    N, T, C = patches_full.shape
    k_out = 1 if peak_only else top_k_channels
    out_patches = np.zeros((N, left + right, k_out), dtype=np.float32)
    out_geo = np.zeros((N, k_out, 4), dtype=np.float32)
    
    for i in range(N):
        t_peak, c_peak = np.unravel_index(np.argmax(np.abs(patches_full[i])), (T, C))
        t_start = t_peak - left
        t_end = t_peak + right
        
        pad_left = max(0, -t_start)
        pad_right = max(0, t_end - T)
        
        valid_start = max(0, t_start)
        valid_end = min(T, t_end)
        
        if peak_only:
            neigh = np.array([c_peak], dtype=np.int32)
            patch_slice = patches_full[i, valid_start:valid_end][:, neigh]
        else:
            neigh = neighbors_lookup[c_peak]
            patch_slice = patches_full[i, valid_start:valid_end][:, neigh]
        
        if pad_left > 0 or pad_right > 0:
            padded = np.zeros((left + right, k_out), dtype=np.float32)
            padded[pad_left:(left+right)-pad_right] = patch_slice
            out_patches[i] = padded
        else:
            out_patches[i] = patch_slice
            
        geo_i = relative_geo(positions, np.array([c_peak]), np.array([neigh]))[0].astype(np.float32, copy=False)
        if geo_i.shape[1] < 4:
            pad = np.zeros((geo_i.shape[0], 4 - int(geo_i.shape[1])), dtype=np.float32)
            geo_i = np.concatenate([geo_i, pad], axis=1)
        out_geo[i] = geo_i[:, :4]

    if normalize:
        out_patches = _normalize_patch(out_patches)
        
    return out_patches.astype(np.float32, copy=False), unit_ids.astype(np.int64, copy=False), out_geo.astype(np.float32)


def build_unit_recording_support(
    unit_id: np.ndarray,
    recording_id: np.ndarray,
    *,
    sample_source: np.ndarray | None = None,
    spike_label: np.ndarray | None = None,
    primary_only: bool = False,
) -> Dict[int, np.ndarray]:
    unit_arr = np.asarray(unit_id, dtype=np.int64)
    recording_arr = np.asarray(recording_id, dtype=np.int16)
    if unit_arr.shape[0] != recording_arr.shape[0]:
        raise ValueError("unit_id and recording_id must have matching shape")
    keep_mask = unit_arr >= 0
    if primary_only:
        if sample_source is not None:
            source_arr = np.asarray(sample_source, dtype=np.int8)
            if source_arr.shape[0] != unit_arr.shape[0]:
                raise ValueError("sample_source must match unit_id shape")
            keep_mask = keep_mask & (
                source_arr == int(OPERATIONAL_SOURCE_NAME_TO_CODE["primary_matched_positive"])
            )
        if spike_label is not None:
            spike_arr = np.asarray(spike_label, dtype=np.float32)
            if spike_arr.shape[0] != unit_arr.shape[0]:
                raise ValueError("spike_label must match unit_id shape")
            keep_mask = keep_mask & (spike_arr > 0.5)
    if not np.any(keep_mask):
        return {}
    filtered_units = unit_arr[keep_mask]
    filtered_recordings = recording_arr[keep_mask]
    out: Dict[int, np.ndarray] = {}
    for unit_value in np.unique(filtered_units).tolist():
        recs = np.unique(filtered_recordings[filtered_units == int(unit_value)]).astype(np.int16, copy=False)
        out[int(unit_value)] = recs
    return out


def save_npz(path: str | Path, arrays: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_optional_patches(path: str | Path, patches: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, patches.astype(np.float32))


class MEArecH5Reader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def get_fs(self) -> float:
        with h5py.File(self.path, "r") as f:
            fs = f["info"]["recordings"]["fs"][()]
        return float(fs)

    def get_recordings_shape(self) -> Tuple[int, int]:
        with h5py.File(self.path, "r") as f:
            shape = f["recordings"].shape
        if len(shape) < 2:
            raise ValueError(f"recordings dataset must be 2D, got shape={shape}")
        return int(shape[0]), int(shape[1])

    def read_recordings(self, start: int, stop: int) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            return f["recordings"][start:stop].astype(np.float32)

    def iter_recordings(self, chunk_samples: int, start: int = 0, stop: Optional[int] = None) -> Iterator[np.ndarray]:
        chunk = int(chunk_samples)
        if chunk <= 0:
            raise ValueError(f"chunk_samples must be > 0, got {chunk_samples}")
        with h5py.File(self.path, "r") as f:
            ds = f["recordings"]
            total = int(ds.shape[0])
            s = max(0, int(start))
            e = total if stop is None else min(total, int(stop))
            for t0 in range(s, e, chunk):
                t1 = min(t0 + chunk, e)
                yield np.asarray(ds[t0:t1], dtype=np.float32)

    def read_channel_positions(self) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            return f["channel_positions"][:].astype(np.float32)

    def read_unit_peak_channels(self) -> Dict[int, int]:
        with h5py.File(self.path, "r") as f:
            return _read_unit_peak_channels_from_h5(f)

    def iter_spiketrains(self) -> Iterator[Tuple[int, np.ndarray]]:
        with h5py.File(self.path, "r") as f:
            grp = f["spiketrains"]
            for key in grp.keys():
                times = grp[key]["times"][:].astype(np.float64)
                try:
                    unit_id = int(key)
                except ValueError:
                    unit_id = int(key.split("_")[-1])
                yield unit_id, times


def _progress_logger(progress_cb: Callable[[str], None] | None):
    class _Logger:
        def info(self, message: str, *args: Any) -> None:
            if progress_cb is None:
                return
            progress_cb(str(message) % args if args else str(message))

        def warning(self, message: str, *args: Any) -> None:
            self.info("warning: " + str(message), *args)

    return _Logger()


def _operational_pretrain_cfg(
    *,
    left: int,
    right: int,
    patch_k: int,
    peak_only: bool,
    car_mode: str,
    normalization_mode: str,
    detector_chunk_sec: float,
    detector_config: Dict[str, Any] | None,
    electrode_config: str | Path | None,
    strict_overlap_ms: float,
    strict_overlap_um: float,
) -> Dict[str, Any]:
    det = {
        "alpha": 5.0,
        "refractory_ms": 0.30,
        "merge_dt_ms": 0.20,
        "merge_enabled": True,
        "merge_mode": "neighbors",
        "merge_radius_ch": 1,
        "method": "fast_local",
        "stats_update_every_windows": 4,
        "stats_subsample_step": 8,
        "active_halo_ms": 0.25,
        "max_simul_firing": 0.25,
        "thr_max_amp_norm": 30.0,
        "low_spatial_score": 0.1,
        "low_amp_norm": 5.0,
        "neighbor_source": "auto",
        "duplicate_dt_ms": 0.45,
        "duplicate_radius_ch": 1,
        "hard_min_spatial_score": 0.0,
        "hard_min_support": 1,
        "hard_filter_mode": "soft",
    }
    if detector_config:
        det.update(dict(detector_config))
    return {
        "stream": {
            "window_sec": 1.5,
            "stride_sec": 0.5,
            "margin_sec": 0.05,
            "chunk_sec": float(detector_chunk_sec),
            "ring_capacity_sec": 4.0,
        },
        "patch": {
            "left": int(left),
            "right": int(right),
            "k": 1 if bool(peak_only) else max(1, int(patch_k)),
            "peak_only": bool(peak_only),
        },
        "preprocess": {
            "bandpass_low": 300,
            "bandpass_high": 6000,
            "car_mode": str(car_mode),
            "noise_beta": 0.05,
            "patch_norm_mode": str(normalization_mode or "peak_channel_sigma"),
        },
        "detection": det,
        "labeling": {
            "match_ms": 0.2,
            "secondary_match_ms": 0.1,
            "gt_align_to_peak": True,
            "gt_align_radius_ms": 0.5,
            "gt_time_reference": "near_peak",
            "single_channel_gt_align_radius_ms": 2.0,
            "single_channel_gt_time_reference": "onset",
            "gt_align_polarity": "negative",
            "gt_align_channel_hops": 1,
            "primary_channel_mode": "exact_then_neighbor",
            "unit_snr_min_positive": 4.0,
            "high_snr_ignore_near_gt_ms": 0.5,
            "high_snr_ignore_neighbor_hops": 1,
            "missed_gt_ignore_enabled": True,
            "missed_gt_ignore_near_gt_ms": 0.5,
            "missed_gt_ignore_neighbor_hops": 1,
            "require_drift_gt_channel_for_missed_ignore": True,
            "neighbor_k": 9,
            "neighbor_hops": 1,
            "max_positive_per_gt": 2,
            "near_higher_peak_ratio": 0.85,
            "strict_overlap_ms": float(strict_overlap_ms),
            "strict_overlap_um": float(strict_overlap_um),
            "near_gt_ms": 0.5,
        },
        "offline": {"electrode_config": None if electrode_config is None else str(electrode_config)},
    }


def build_operational_pretrain_config(
    *,
    left: int,
    right: int,
    patch_k: int,
    peak_only: bool,
    car_mode: str,
    normalization_mode: str,
    detector_chunk_sec: float,
    detector_config: Dict[str, Any] | None,
    electrode_config: str | Path | None,
    strict_overlap_ms: float,
    strict_overlap_um: float,
) -> Dict[str, Any]:
    return _operational_pretrain_cfg(
        left=int(left),
        right=int(right),
        patch_k=int(patch_k),
        peak_only=bool(peak_only),
        car_mode=str(car_mode),
        normalization_mode=str(normalization_mode),
        detector_chunk_sec=float(detector_chunk_sec),
        detector_config=detector_config,
        electrode_config=electrode_config,
        strict_overlap_ms=float(strict_overlap_ms),
        strict_overlap_um=float(strict_overlap_um),
    )


def _labels_from_operational_arrays(
    arrays: Dict[str, np.ndarray],
    *,
    recording_idx: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    patches = np.asarray(arrays["patches"], dtype=np.float32)
    n = int(patches.shape[0])
    label_role = np.asarray(arrays.get("label_role", np.zeros((n,), dtype=np.int8)), dtype=np.int8).reshape(-1)
    spike_label = np.asarray(arrays.get("spike_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    training_mask = np.asarray(arrays.get("training_mask", np.ones((n,), dtype=np.bool_)), dtype=np.bool_).reshape(-1)
    eval_mask = np.asarray(arrays.get("eval_mask", training_mask), dtype=np.bool_).reshape(-1)
    matched_gt_label = np.asarray(arrays.get("matched_gt_label", np.full((n,), -1, dtype=np.int64)), dtype=np.int64).reshape(-1)

    positive = (spike_label > 0.5) & training_mask & (matched_gt_label >= 0)
    local_unit_id = np.where(positive, matched_gt_label, -1).astype(np.int64, copy=False)
    recording_id = np.full((n,), int(recording_idx), dtype=np.int16)
    global_unit_id = build_global_unit_ids(local_unit_id, recording_id)

    sample_source = np.full((n,), IGNORED_SAMPLE_SOURCE, dtype=np.int8)
    sample_source[(label_role == 1) & positive] = int(OPERATIONAL_SOURCE_NAME_TO_CODE["primary_matched_positive"])
    sample_source[(label_role == 2) & positive] = int(OPERATIONAL_SOURCE_NAME_TO_CODE["secondary_matched_positive"])
    sample_source[(spike_label <= 0.5) & training_mask] = int(OPERATIONAL_SOURCE_NAME_TO_CODE["detector_negative"])

    event_time_sample = np.asarray(arrays.get("event_time_sample", np.zeros((n,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    center_channel = np.asarray(arrays.get("center_channel", np.zeros((n,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    event_amp_norm = np.asarray(arrays.get("amp_norm", np.ones((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    event_spatial_score = np.asarray(arrays.get("spatial_score", np.ones((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    event_support_count = np.asarray(arrays.get("support_count", np.ones((n,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    event_simul_global = np.asarray(arrays.get("simul_global", np.zeros((n,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    event_simul_local = np.asarray(arrays.get("simul_local", np.zeros((n,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    fs_arr = np.asarray(arrays.get("fs", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if fs_arr.size != n:
        fs_arr = np.full((n,), float(fs_arr[0]) if fs_arr.size else 0.0, dtype=np.float32)
    n_channels_arr = np.asarray(arrays.get("n_channels", np.ones((n,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    n_channels = int(n_channels_arr[0]) if n_channels_arr.size else 1
    spatial_stats = np.stack(
        [
            event_amp_norm / 30.0,
            event_spatial_score,
            event_support_count.astype(np.float32),
            event_simul_global.astype(np.float32) / float(max(1, n_channels)),
            event_simul_local.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    dt_samples = np.asarray(arrays.get("dt_samples", np.zeros((n,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    dt_samples = np.where(dt_samples == np.iinfo(np.int32).max, 0, dt_samples).astype(np.int32, copy=False)
    overlap_label = np.asarray(arrays.get("overlap_label", np.zeros((n,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    geo = np.asarray(arrays.get("geo", np.zeros((n, int(patches.shape[2]), 4), dtype=np.float32)), dtype=np.float32)
    if geo.shape != (n, int(patches.shape[2]), 4):
        geo = np.zeros((n, int(patches.shape[2]), 4), dtype=np.float32)
    rel_pos = np.asarray(arrays.get("rel_pos", np.zeros((n, int(patches.shape[2]), 2), dtype=np.float32)), dtype=np.float32)
    if rel_pos.shape != (n, int(patches.shape[2]), 2):
        rel_pos = np.zeros((n, int(patches.shape[2]), 2), dtype=np.float32)
    channel_mask = np.asarray(arrays.get("channel_mask", np.ones((n, int(patches.shape[2])), dtype=np.bool_)), dtype=np.bool_)
    if channel_mask.shape != (n, int(patches.shape[2])):
        channel_mask = np.ones((n, int(patches.shape[2])), dtype=np.bool_)
    neighbor_chs = np.asarray(arrays.get("neighbor_chs", np.tile(center_channel[:, None], (1, int(patches.shape[2])))), dtype=np.int32)
    if neighbor_chs.shape != (n, int(patches.shape[2])):
        neighbor_chs = np.tile(center_channel[:, None], (1, int(patches.shape[2]))).astype(np.int32, copy=False)
    quality_flag = (
        (local_unit_id >= 0)
        & training_mask
        & (overlap_label <= 0.5)
        & (spike_label > 0.5)
    ).astype(np.bool_, copy=False)

    labels: Dict[str, np.ndarray] = {
        "unit_id": global_unit_id.astype(np.int64, copy=False),
        "local_unit_id": local_unit_id.astype(np.int64, copy=False),
        "global_unit_id": global_unit_id.astype(np.int64, copy=False),
        "matched_gt_label": matched_gt_label.astype(np.int64, copy=False),
        "matched_h5_unit_id": np.asarray(arrays.get("matched_h5_unit_id", matched_gt_label), dtype=np.int64).reshape(-1),
        "h5_unit_id": np.asarray(arrays.get("h5_unit_id", matched_gt_label), dtype=np.int64).reshape(-1),
        "matched_gt_index": np.asarray(arrays.get("matched_gt_index", np.full((n,), -1, dtype=np.int64)), dtype=np.int64).reshape(-1),
        "matched_gt_channel": np.asarray(arrays.get("matched_gt_channel", np.full((n,), -1, dtype=np.int32)), dtype=np.int32).reshape(-1),
        "matched_gt_snr": np.asarray(arrays.get("matched_gt_snr", np.full((n,), np.nan, dtype=np.float32)), dtype=np.float32).reshape(-1),
        "geo": geo,
        "rel_pos": rel_pos.astype(np.float32, copy=False),
        "channel_mask": channel_mask.astype(np.bool_, copy=False),
        "neighbor_chs": neighbor_chs.astype(np.int32, copy=False),
        "spike_label": spike_label.astype(np.float32, copy=False),
        "sample_source": sample_source.astype(np.int8, copy=False),
        "sample_source_schema": np.asarray([OPERATIONAL_SOURCE_SCHEMA_VERSION] * n),
        "source_schema_version": np.asarray([OPERATIONAL_SOURCE_SCHEMA_VERSION] * n),
        "label_role": label_role.astype(np.int8, copy=False),
        "training_mask": training_mask.astype(np.bool_, copy=False),
        "eval_mask": eval_mask.astype(np.bool_, copy=False),
        "center_channel": center_channel.astype(np.int32, copy=False),
        "event_time_sample": event_time_sample.astype(np.int64, copy=False),
        "anchor_time_sample": event_time_sample.astype(np.int64, copy=False),
        "recording_fs_hz": fs_arr.astype(np.float32, copy=False),
        "align_shift_samples": dt_samples.astype(np.int32, copy=False),
        "peak_channel": center_channel.astype(np.int32, copy=False),
        "recording_id": recording_id.astype(np.int16, copy=False),
        "quality_flag": quality_flag,
        "strict_overlap_label": overlap_label.astype(np.float32, copy=False),
        "overlap_label": overlap_label.astype(np.float32, copy=False),
        "overlap_supervision_mask": np.asarray(arrays.get("overlap_supervision_mask", positive), dtype=np.bool_).reshape(-1),
        "qc_allowed": training_mask.astype(np.bool_, copy=False),
        "strict_matching_ambiguous": ((~training_mask) & (spike_label <= 0.5)).astype(np.bool_, copy=False),
        "relaxed_matching_ambiguous": np.zeros((n,), dtype=np.bool_),
        "snr_peak": event_amp_norm.astype(np.float32, copy=False),
        "unit_snr": np.asarray(arrays.get("matched_gt_snr", np.full((n,), np.nan, dtype=np.float32)), dtype=np.float32).reshape(-1),
        "event_amp_norm": event_amp_norm.astype(np.float32, copy=False),
        "event_spatial_score": event_spatial_score.astype(np.float32, copy=False),
        "event_support_count": event_support_count.astype(np.int32, copy=False),
        "event_simul_global": event_simul_global.astype(np.int32, copy=False),
        "event_simul_local": event_simul_local.astype(np.int32, copy=False),
        "event_stats_source": np.ones((n,), dtype=np.int8),
        "spatial_stats": spatial_stats,
    }
    for key in (
        "ignored_neighbor_event",
        "neighbor_primary",
        "domain_type",
        "n_channels",
        "nearest_gt_dt_samples",
        "nearest_gt_index",
        "nearest_gt_snr",
        "near_gt_negative",
        "high_snr_near_gt",
        "low_snr_near_gt",
        "missed_gt_near_gt",
        "duplicate_negative",
        "gt_channel_source",
        "gt_template_channel",
        "gt_drift_channel",
        "gt_anchor_channel",
        "gt_peak_aligned_channel",
        "gt_anchor_peak_aligned_channel_changed",
        "drift_gt_channel_enabled",
        "drift_gt_channel_valid",
        "drift_gt_channel_changed_count",
        "gt_align_enabled",
        "gt_align_radius_ms",
        "gt_align_median_shift_samples",
        "gt_align_p95_abs_shift_samples",
        "gt_align_max_abs_shift_samples",
        "gt_align_channel_changed",
        "unit_snr_min_positive",
        "unit_snr_filter_enabled",
        "high_snr_ignore_near_gt_ms",
        "high_snr_ignore_neighbor_hops",
        "missed_gt_ignore_enabled",
        "missed_gt_ignore_near_gt_ms",
        "missed_gt_ignore_neighbor_hops",
        "require_drift_gt_channel_for_missed_ignore",
        "missed_gt_count",
        "missed_gt_ignored_candidates",
        "missed_gt_drift_channel_valid",
        "gt_snr_high_unit_count",
        "gt_snr_low_unit_count",
        "gt_snr_missing_unit_count",
        "gt_snr_high_event_count",
        "gt_snr_low_event_count",
        "gt_snr_missing_event_count",
        "labeling_version",
    ):
        if key in arrays:
            labels[key] = np.asarray(arrays[key])
    return patches, labels


def load_mearec_pretrain_data(
    h5_paths: Iterable[str | Path],
    *,
    top_k_channels: int = 16,
    normalize: bool = True,
    max_per_unit: int = 100,
    seed: int = 42,
    progress_cb: Callable[[str], None] | None = None,
    left: int = 20,
    right: int = 44,
    electrode_config: str | Path | None = None,
    merge_dt_ms: float = 0.5,
    dedup_enabled: bool = True,
    dedup_mode: str = "neighbors",
    peak_only: bool = False,
    align_radius_ms: float = 0.2,
    align_polarity: str = "negative",
    gt_time_reference: str = "near_peak",
    car_mode: str = "auto",
    normalization_mode: str = "peak_channel_sigma",
    robust_norm_samples: int = 60000,
    min_snr: float = 4.0,
    max_snr: float = 0.0,
    gt_max_per_unit: int | None = None,
    gt_min_snr: float | None = None,
    detector_min_snr: float | None = None,
    min_unit_snr: float = 0.0,
    drop_border_spikes: bool = True,
    sample_strategy: str = "uniform_time",
    include_detector_candidates: bool = True,
    secondary_matched_positive_ratio: float = 1.0,
    detector_negative_ratio: float = 2.0,
    detector_negative_exclusion_ms: float = 0.6,
    secondary_matched_positive_min_spatial_score: float = 0.05,
    detector_negative_ambiguous_tol_ms: float = 0.6,
    secondary_matched_positive_tol_samples: int | None = 3,
    detector_disable_negative_for_clean: bool = False,
    detector_chunk_sec: float = 0.5,
    detector_tol_ms: float = 0.3,
    strict_overlap_ms: float = 0.5,
    strict_overlap_um: float = 60.0,
    detector_config: Dict[str, Any] | None = None,
    detector_alpha_overrides: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    del (
        normalize,
        max_per_unit,
        seed,
        merge_dt_ms,
        dedup_enabled,
        dedup_mode,
        align_radius_ms,
        align_polarity,
        gt_time_reference,
        robust_norm_samples,
        min_snr,
        max_snr,
        gt_max_per_unit,
        gt_min_snr,
        detector_min_snr,
        min_unit_snr,
        drop_border_spikes,
        sample_strategy,
        include_detector_candidates,
        secondary_matched_positive_ratio,
        detector_negative_ratio,
        detector_negative_exclusion_ms,
        secondary_matched_positive_min_spatial_score,
        detector_negative_ambiguous_tol_ms,
        secondary_matched_positive_tol_samples,
        detector_disable_negative_for_clean,
        detector_tol_ms,
        detector_alpha_overrides,
    )
    import tempfile

    from . import operational_gate

    cfg = build_operational_pretrain_config(
        left=int(left),
        right=int(right),
        patch_k=int(top_k_channels),
        peak_only=bool(peak_only),
        car_mode=str(car_mode),
        normalization_mode=str(normalization_mode or "peak_channel_sigma"),
        detector_chunk_sec=float(detector_chunk_sec),
        detector_config=detector_config,
        electrode_config=electrode_config,
        strict_overlap_ms=float(strict_overlap_ms),
        strict_overlap_um=float(strict_overlap_um),
    )
    logger = _progress_logger(progress_cb)
    all_patches: List[np.ndarray] = []
    label_parts: Dict[str, List[np.ndarray]] = {}
    for recording_idx, raw_path in enumerate(h5_paths):
        path = Path(raw_path)
        if progress_cb is not None:
            progress_cb(f"operational_pretrain_candidates path={path}")
        with tempfile.TemporaryDirectory(prefix="msort_operational_candidates_") as tmpdir:
            out_npz = Path(tmpdir) / f"{recording_idx}_{path.stem}.npz"
            job = operational_gate.GateJob(name=path.stem, input_h5=path, split="train")
            operational_gate.generate_candidates_for_job(
                job,
                cfg=cfg,
                output_npz=out_npz,
                max_seconds=0.0,
                logger=logger,
            )
            with np.load(out_npz, allow_pickle=False) as arr:
                arrays = {key: arr[key] for key in arr.files}
        patches, labels = _labels_from_operational_arrays(arrays, recording_idx=int(recording_idx))
        if patches.shape[0] == 0:
            continue
        all_patches.append(patches.astype(np.float32, copy=False))
        for key, value in labels.items():
            label_parts.setdefault(key, []).append(np.asarray(value))
        if progress_cb is not None:
            source = labels["sample_source"]
            progress_cb(
                "operational_pretrain_loaded "
                f"path={path} samples={int(patches.shape[0])} "
                f"primary={int(np.sum(source == OPERATIONAL_SOURCE_NAME_TO_CODE['primary_matched_positive']))} "
                f"secondary={int(np.sum(source == OPERATIONAL_SOURCE_NAME_TO_CODE['secondary_matched_positive']))} "
                f"negative={int(np.sum(source == OPERATIONAL_SOURCE_NAME_TO_CODE['detector_negative']))} "
                f"ignored={int(np.sum(source < 0))}"
            )

    if not all_patches:
        raise ValueError("no operational detector candidates extracted from h5 files")

    max_k = max(int(p.shape[2]) for p in all_patches)
    if any(int(p.shape[2]) != max_k for p in all_patches):
        padded_patches: List[np.ndarray] = []
        for p in all_patches:
            if int(p.shape[2]) == max_k:
                padded_patches.append(p)
                continue
            pad = np.zeros((int(p.shape[0]), int(p.shape[1]), max_k - int(p.shape[2])), dtype=np.float32)
            padded_patches.append(np.concatenate([p, pad], axis=2).astype(np.float32, copy=False))
        all_patches = padded_patches

        def _pad_label_parts(key: str, width: int, fill_value: float | int | bool) -> None:
            parts = label_parts.get(key)
            if not parts:
                return
            out_parts: List[np.ndarray] = []
            for part in parts:
                arr = np.asarray(part)
                if arr.ndim < 2 or int(arr.shape[1]) == width:
                    out_parts.append(arr)
                    continue
                pad_shape = list(arr.shape)
                pad_shape[1] = width - int(arr.shape[1])
                pad = np.full(pad_shape, fill_value, dtype=arr.dtype)
                out_parts.append(np.concatenate([arr, pad], axis=1))
            label_parts[key] = out_parts

        _pad_label_parts("geo", max_k, 0.0)
        _pad_label_parts("rel_pos", max_k, 0.0)
        _pad_label_parts("channel_mask", max_k, False)
        _pad_label_parts("neighbor_chs", max_k, -1)
    patches_all = np.concatenate(all_patches, axis=0).astype(np.float32, copy=False)
    labels_all = {key: np.concatenate(parts, axis=0) for key, parts in label_parts.items()}
    labels_all = ensure_global_unit_id_labels(labels_all)
    if progress_cb is not None:
        sample_source = labels_all["sample_source"]
        progress_cb(
            "merged_operational_pretrain "
            f"total_samples={int(patches_all.shape[0])} "
            f"classes={int(np.unique(labels_all['unit_id'][labels_all['unit_id'] >= 0]).size)} "
            f"primary={int(np.sum(sample_source == OPERATIONAL_SOURCE_NAME_TO_CODE['primary_matched_positive']))} "
            f"secondary={int(np.sum(sample_source == OPERATIONAL_SOURCE_NAME_TO_CODE['secondary_matched_positive']))} "
            f"negative={int(np.sum(sample_source == OPERATIONAL_SOURCE_NAME_TO_CODE['detector_negative']))} "
            f"ignored={int(np.sum(sample_source < 0))}"
        )
    return patches_all, labels_all


__all__ = [
    "build_global_unit_ids",
    "build_operational_pretrain_config",
    "MEArecH5Reader",
    "Recording",
    "compute_strict_overlap_label",
    "ensure_global_unit_id_labels",
    "IGNORED_SAMPLE_SOURCE",
    "load_mearec_pretrain_data",
    "load_mearec_template_patches",
    "load_npz",
    "OPERATIONAL_SOURCE_CODE_TO_NAME",
    "OPERATIONAL_SOURCE_NAME_TO_CODE",
    "OPERATIONAL_SOURCE_SCHEMA_VERSION",
    "save_npz",
    "save_optional_patches",
]
