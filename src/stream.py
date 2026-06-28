from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import butter, find_peaks, sosfilt, sosfilt_zi

from .nn.model import ModelConfig, SpikeEmbeddingModel
from .online.buffer import RingBuffer, RingView
from .utils import (
    CandidatePeak,
    Event,
    EventOutput,
    Monitor,
    PatchBatch,
    PipelineStats,
    QCFlag,
    build_knn_neighbors,
    ensure_channel_positions,
    get_logger,
    pairwise_distances,
    relative_geo,
    resolve_max_simul_firing,
    update_qc_flag,
)


@dataclass
class FileTail:
    path: Path

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def size_bytes(self) -> int:
        if not self.path.exists():
            return 0
        return self.path.stat().st_size

    def wait_for(self, n_bytes: int, timeout_sec: float = 0.5, poll_interval_sec: float = 0.02) -> bool:
        n_bytes = int(n_bytes)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.size_bytes() >= n_bytes:
                return True
            time.sleep(poll_interval_sec)
        return self.size_bytes() >= n_bytes


class IntanStreamReader:
    def __init__(self, path: str | Path, n_channels: int, scale_uV: Optional[float] = None):
        self.path = Path(path)
        self.n_channels = int(n_channels)
        self.scale_uV = scale_uV
        self.file_offset_samples = 0
        self._dtype = np.int16
        self._bytes_per_sample = np.dtype(self._dtype).itemsize * self.n_channels
        self._tail = FileTail(self.path)

    def available_samples(self) -> int:
        size = self._tail.size_bytes()
        return size // self._bytes_per_sample - self.file_offset_samples

    def read_chunk(self, n_samples: int, timeout_sec: float = 0.0) -> np.ndarray:
        n_samples = int(n_samples)
        if n_samples <= 0:
            return np.zeros((0, self.n_channels), dtype=np.float32)
        target_bytes = (self.file_offset_samples + n_samples) * self._bytes_per_sample
        if timeout_sec > 0:
            _ = self._tail.wait_for(target_bytes, timeout_sec=timeout_sec)
        avail = self.available_samples()
        if avail <= 0:
            return np.zeros((0, self.n_channels), dtype=np.float32)
        n_read = min(n_samples, avail)
        offset_bytes = self.file_offset_samples * self._bytes_per_sample
        with open(self.path, "rb") as f:
            f.seek(offset_bytes, os.SEEK_SET)
            raw = np.fromfile(f, dtype=self._dtype, count=n_read * self.n_channels)
        if raw.size == 0:
            return np.zeros((0, self.n_channels), dtype=np.float32)
        x = raw.reshape(-1, self.n_channels).astype(np.float32)
        if self.scale_uV is not None:
            x *= float(self.scale_uV)
        self.file_offset_samples += x.shape[0]
        return x


class StreamBandpass:
    def __init__(self, fs: float, low_hz: float = 300.0, high_hz: float = 6000.0, order: int = 3, n_channels: int = 1):
        nyq = 0.5 * float(fs)
        low = low_hz / nyq
        high = min(high_hz / nyq, 0.99)
        self.sos = butter(order, [low, high], btype="bandpass", output="sos")
        self.n_channels = int(n_channels)
        base_zi = sosfilt_zi(self.sos).astype(np.float32)
        self.zi = np.repeat(base_zi[:, :, None], self.n_channels, axis=2)

    def process(self, x_new: np.ndarray) -> np.ndarray:
        if x_new.size == 0:
            return x_new.astype(np.float32, copy=False)
        y, self.zi = sosfilt(self.sos, x_new, axis=0, zi=self.zi)
        return y.astype(np.float32, copy=False)


class StreamCAR:
    def __init__(self, mode: str = "global", groups: Optional[Iterable[Iterable[int]]] = None):
        self.mode = mode
        self.groups: List[np.ndarray] = []
        if groups is not None:
            self.groups = [np.array(g, dtype=np.int32) for g in groups]

    def process(self, y_new: np.ndarray) -> np.ndarray:
        if y_new.size == 0:
            return y_new.astype(np.float32, copy=False)
        y = y_new.astype(np.float32, copy=True)
        if self.mode == "none":
            return y
        if self.mode == "global" or not self.groups:
            mean = np.mean(y, axis=1, keepdims=True)
            return y - mean
        for group in self.groups:
            if group.size == 0:
                continue
            mean = np.mean(y[:, group], axis=1, keepdims=True)
            y[:, group] -= mean
        return y


class RobustNoiseEstimator:
    def __init__(
        self,
        n_channels: int,
        beta: float = 0.05,
        warmup_windows: int = 2,
        eps: float = 1e-6,
        update_every_windows: int = 1,
        subsample_step: int = 1,
    ):
        self.n_channels = int(n_channels)
        self.beta = float(beta)
        self.warmup_windows = int(warmup_windows)
        self.eps = float(eps)
        self.update_every_windows = max(1, int(update_every_windows))
        self.subsample_step = max(1, int(subsample_step))
        self.sigma = np.ones((self.n_channels,), dtype=np.float32)
        self.median = np.zeros((self.n_channels,), dtype=np.float32)
        self._seen = 0

    def _mad_sigma_and_median(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.subsample_step > 1 and x.shape[0] > self.subsample_step:
            x = x[:: self.subsample_step]
        med = np.median(x, axis=0).astype(np.float32, copy=False)
        mad = np.median(np.abs(x - med[None, :]), axis=0)
        sigma = 1.4826 * mad
        return np.maximum(sigma.astype(np.float32), self.eps), med

    def _mad_sigma(self, x: np.ndarray) -> np.ndarray:
        sigma, _ = self._mad_sigma_and_median(x)
        return sigma

    def update(self, x_window: np.ndarray) -> np.ndarray:
        if x_window.size == 0:
            return self.sigma
        if self._seen >= self.warmup_windows and (self._seen % self.update_every_windows) != 0:
            self._seen += 1
            return self.sigma
        sigma_w, med_w = self._mad_sigma_and_median(x_window)
        if self._seen < self.warmup_windows:
            self.sigma = sigma_w
            self.median = med_w
        else:
            self.sigma = (1.0 - self.beta) * self.sigma + self.beta * sigma_w
            self.median = (1.0 - self.beta) * self.median + self.beta * med_w
        self._seen += 1
        return self.sigma


def loose_detector_defaults(n_channels: int | None = None) -> Dict[str, Any]:
    _ = n_channels
    return {
        "alpha": 4.0,
        "refractory_ms": 0.30,
        "merge_dt_ms": 0.20,
        "merge_enabled": True,
        "merge_mode": "neighbors",
        "merge_radius_ch": 1,
        "thr_max_amp_norm": 30.0,
        "low_spatial_score": 0.1,
        "low_amp_norm": 5.0,
        "method": "fast_local",
        "stats_update_every_windows": 4,
        "stats_subsample_step": 8,
        "active_halo_ms": 0.25,
        "max_simul_firing": 0,
        "global_simul_threshold": 0.2,
        "local_simul_threshold": 0.8,
        "duplicate_dt_ms": 0.30,
        "duplicate_radius_ch": 1,
        "hard_min_spatial_score": 0.0,
        "hard_min_support": 1,
        "hard_filter_mode": "soft",
        "saturation_filter_mode": "drop",
    }


def _normalize_saturation_filter_mode(value: Any) -> str:
    mode = str(value if value is not None else "drop").strip().lower()
    if mode in ("drop", "hard", "reject", "true", "1"):
        return "drop"
    if mode in ("flag", "soft", "mark"):
        return "flag"
    if mode in ("off", "none", "false", "0"):
        return "off"
    raise ValueError("detection.saturation_filter_mode must be one of drop, flag, off")


def event_spatial_stats(
    events: List[Event],
    *,
    n_channels: int,
    neighbor_count: int,
) -> np.ndarray:
    if not events:
        return np.zeros((0, 5), dtype=np.float32)
    denom_ch = max(1, int(n_channels))
    denom_neigh = max(1, int(neighbor_count))
    rows = [
        [
            float(e.amp_norm) / 30.0,
            float(e.spatial_score),
            float(e.support_count) / float(denom_neigh),
            float(e.simul_global) / float(denom_ch),
            float(e.simul_local) / float(denom_neigh),
        ]
        for e in events
    ]
    return np.asarray(rows, dtype=np.float32)


def _pick_negative_peaks_legacy(
    x_seg: np.ndarray,
    sigma_c: np.ndarray,
    fs: float,
    alpha: float = 5.0,
    refractory_ms: float = 0.8,
    align_radius: int = 3,
) -> List[CandidatePeak]:
    if x_seg.size == 0:
        return []
    t_len, n_channels = x_seg.shape
    refractory = max(1, int(round(refractory_ms * fs / 1000.0)))
    cands: List[CandidatePeak] = []

    for c in range(n_channels):
        sigma = max(float(sigma_c[c]), 1e-6)
        thr = alpha * sigma
        signal = -x_seg[:, c]
        peaks, _ = find_peaks(signal, height=thr, distance=refractory)
        if peaks.size == 0:
            continue
        for p in peaks:
            lo = max(0, p - align_radius)
            hi = min(t_len, p + align_radius + 1)
            local_min = lo + int(np.argmin(x_seg[lo:hi, c]))
            amp = abs(float(x_seg[local_min, c]))
            amp_norm = amp / sigma
            cands.append(
                CandidatePeak(
                    t=int(local_min),
                    c=int(c),
                    amp_norm=float(amp_norm),
                    prominence=float(amp_norm),
                    width=1.0,
                )
            )
    cands.sort(key=lambda p: p.t)
    return cands


def _pick_negative_peaks_fast(
    x_seg: np.ndarray,
    sigma_c: np.ndarray,
    fs: float,
    alpha: float = 5.0,
    refractory_ms: float = 0.8,
    align_radius: int = 3,
    active_start: int | None = None,
    active_stop: int | None = None,
    active_halo: int = 0,
    thr_max_amp_norm: float = 30.0,
    max_simul_firing: int = 0,
    global_simul_threshold: int = 0,
    local_simul_threshold: int = 0,
    saturation_filter_mode: str = "drop",
    neighbors_lookup: np.ndarray | None = None,
    stats_out: Dict[str, int] | None = None,
) -> List[CandidatePeak]:
    if x_seg.size == 0:
        return []
    t_len, n_channels = x_seg.shape
    active_start = 0 if active_start is None else int(np.clip(active_start, 0, t_len))
    active_stop = t_len if active_stop is None else int(np.clip(active_stop, active_start, t_len))
    if active_stop <= active_start:
        return []

    detect_lo = max(0, active_start - int(active_halo))
    detect_hi = min(t_len, active_stop + int(active_halo))
    x_local = x_seg[detect_lo:detect_hi]
    if x_local.size == 0:
        return []

    refractory = max(1, int(round(refractory_ms * fs / 1000.0)))
    thr = alpha * np.maximum(np.asarray(sigma_c, dtype=np.float32), 1e-6)
    mask = x_local < (-thr[None, :])
    simultaneous = np.sum(mask, axis=1)
    global_threshold = int(global_simul_threshold)
    if global_threshold <= 0 and int(max_simul_firing) > 0:
        global_threshold = int(max_simul_firing)
    local_threshold = int(local_simul_threshold)
    thrmax = np.maximum(float(thr_max_amp_norm), 0.0) * np.maximum(np.asarray(sigma_c, dtype=np.float32), 1e-6)
    saturation_mode = _normalize_saturation_filter_mode(saturation_filter_mode)
    active_lo_local = active_start - detect_lo
    active_hi_local = active_stop - detect_lo

    cands: List[CandidatePeak] = []
    for c in range(n_channels):
        m = mask[:, c]
        if not np.any(m):
            continue
        changes = np.diff(np.pad(m.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        kept_t: List[int] = []
        kept_amp: List[float] = []
        kept_simul_global: List[int] = []
        kept_simul_local: List[int] = []
        kept_qc_flags: List[int] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            lo = max(0, int(s) - int(align_radius))
            hi = min(x_local.shape[0], int(e) + int(align_radius))
            if hi <= lo:
                continue
            local_min = lo + int(np.argmin(x_local[lo:hi, c]))
            if local_min < active_lo_local or local_min >= active_hi_local:
                continue
            sim_lo = max(0, local_min - 1)
            sim_hi = min(x_local.shape[0], local_min + 2)
            simul_global = int(np.max(simultaneous[sim_lo:sim_hi])) if sim_hi > sim_lo else 0
            simul_local = 0
            if neighbors_lookup is not None and local_threshold > 0:
                neigh = np.asarray(neighbors_lookup[c], dtype=np.int32).reshape(-1)
                neigh = neigh[(neigh >= 0) & (neigh < n_channels)]
                if neigh.size > 0:
                    local_counts = np.sum(mask[sim_lo:sim_hi][:, neigh], axis=1)
                    simul_local = int(np.max(local_counts)) if local_counts.size > 0 else 0
            if global_threshold > 0 and simul_global > global_threshold:
                if stats_out is not None:
                    stats_out["global_simul_drops"] = int(stats_out.get("global_simul_drops", 0)) + 1
                continue
            if local_threshold > 0 and simul_local > local_threshold:
                if stats_out is not None:
                    stats_out["local_simul_drops"] = int(stats_out.get("local_simul_drops", 0)) + 1
                continue
            qc_flags = int(QCFlag.NONE)
            if thr_max_amp_norm > 0.0 and saturation_mode != "off":
                art_lo = max(0, local_min - 5)
                art_hi = min(x_local.shape[0], local_min + 6)
                saturated = bool(np.any(np.max(np.abs(x_local[art_lo:art_hi, :]), axis=0) >= thrmax))
                if saturated:
                    if saturation_mode == "drop":
                        continue
                    qc_flags = update_qc_flag(qc_flags, QCFlag.SATURATED_ARTIFACT)
            global_t = detect_lo + local_min
            amp = abs(float(x_local[local_min, c]))
            amp_norm = amp / max(float(sigma_c[c]), 1e-6)
            if kept_t and (global_t - kept_t[-1]) < refractory:
                if amp_norm <= kept_amp[-1]:
                    continue
                kept_t.pop()
                kept_amp.pop()
                kept_simul_global.pop()
                kept_simul_local.pop()
                kept_qc_flags.pop()
            kept_t.append(int(global_t))
            kept_amp.append(float(amp_norm))
            kept_simul_global.append(int(simul_global))
            kept_simul_local.append(int(simul_local))
            kept_qc_flags.append(int(qc_flags))
        for t_peak, amp_norm, sim_g, sim_l, qc_flags in zip(
            kept_t,
            kept_amp,
            kept_simul_global,
            kept_simul_local,
            kept_qc_flags,
        ):
            cands.append(
                CandidatePeak(
                    t=int(t_peak),
                    c=int(c),
                    amp_norm=float(amp_norm),
                    prominence=float(amp_norm),
                    width=1.0,
                    simul_global=int(sim_g),
                    simul_local=int(sim_l),
                    qc_flags=int(qc_flags),
                )
            )
    cands.sort(key=lambda p: p.t)
    return cands


def _build_hop_matrix(neighbors: np.ndarray) -> np.ndarray:
    neigh = np.asarray(neighbors, dtype=np.int32)
    if neigh.ndim != 2 or neigh.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.int16)
    n = int(neigh.shape[0])
    out = np.full((n, n), 32767, dtype=np.int16)
    for src in range(n):
        dist = np.full((n,), -1, dtype=np.int16)
        dist[src] = 0
        queue: List[int] = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            d_next = int(dist[u]) + 1
            for v in neigh[u].tolist():
                vv = int(v)
                if vv < 0 or vv >= n:
                    continue
                if dist[vv] >= 0:
                    continue
                dist[vv] = d_next
                queue.append(vv)
        mask = dist < 0
        if np.any(mask):
            idx = np.nonzero(mask)[0]
            dist[idx] = np.abs(idx - src).astype(np.int16)
        out[src] = dist
    return out


def _hop_distance(
    c1: int,
    c2: int,
    *,
    hop_matrix: np.ndarray | None,
    n_channels: int,
) -> int:
    if hop_matrix is None or hop_matrix.size == 0 or n_channels <= 0:
        return abs(int(c1) - int(c2))
    a = int(np.clip(int(c1), 0, n_channels - 1))
    b = int(np.clip(int(c2), 0, n_channels - 1))
    return int(hop_matrix[a, b])


def pick_negative_peaks(
    x_seg: np.ndarray,
    sigma_c: np.ndarray,
    fs: float,
    alpha: float = 5.0,
    refractory_ms: float = 0.8,
    align_radius: int = 3,
    method: str = "legacy",
    active_start: int | None = None,
    active_stop: int | None = None,
    active_halo: int = 0,
    thr_max_amp_norm: float = 30.0,
    max_simul_firing: int = 0,
    global_simul_threshold: int = 0,
    local_simul_threshold: int = 0,
    saturation_filter_mode: str = "drop",
    neighbors_lookup: np.ndarray | None = None,
) -> List[CandidatePeak]:
    if str(method).lower() in ("legacy", "find_peaks"):
        return _pick_negative_peaks_legacy(
            x_seg=x_seg,
            sigma_c=sigma_c,
            fs=fs,
            alpha=alpha,
            refractory_ms=refractory_ms,
            align_radius=align_radius,
        )
    return _pick_negative_peaks_fast(
        x_seg=x_seg,
        sigma_c=sigma_c,
        fs=fs,
        alpha=alpha,
        refractory_ms=refractory_ms,
        align_radius=align_radius,
        active_start=active_start,
        active_stop=active_stop,
        active_halo=active_halo,
        thr_max_amp_norm=thr_max_amp_norm,
        max_simul_firing=max_simul_firing,
        global_simul_threshold=global_simul_threshold,
        local_simul_threshold=local_simul_threshold,
        saturation_filter_mode=saturation_filter_mode,
        neighbors_lookup=neighbors_lookup,
    )


def _spatial_close(
    c1: int,
    c2: int,
    merge_radius_ch: int,
    neighbors_lookup: np.ndarray | None = None,
    channel_dist: np.ndarray | None = None,
    hop_matrix: np.ndarray | None = None,
    merge_mode: str = "neighbors",
) -> bool:
    mode = str(merge_mode).lower()
    if mode in ("none", "off", "false", "0"):
        return False
    if mode in ("same_channel", "channel"):
        return c1 == c2
    if mode in ("index", "idx", "channel_index"):
        return abs(c1 - c2) <= merge_radius_ch
    # Default: graph-hop based if provided, else distance/index fallback.
    if neighbors_lookup is not None:
        if merge_radius_ch <= 0:
            return c1 == c2
        return _hop_distance(
            c1,
            c2,
            hop_matrix=hop_matrix,
            n_channels=int(neighbors_lookup.shape[0]),
        ) <= int(merge_radius_ch)
    if channel_dist is None:
        return abs(c1 - c2) <= merge_radius_ch
    row = channel_dist[c1]
    nn_pitch = float(np.min(row[row > 0])) if np.any(row > 0) else 1.0
    return channel_dist[c1, c2] <= float(merge_radius_ch) * nn_pitch


def _item_time_sample(item: CandidatePeak | Event) -> int:
    if isinstance(item, Event):
        return int(item.t0)
    return int(item.t)


def _cluster_by_time(items: List[CandidatePeak] | List[Event], merge_dt_samples: int) -> List[List[CandidatePeak] | List[Event]]:
    if not items:
        return []
    clusters: List[List[CandidatePeak] | List[Event]] = []
    cur: List[CandidatePeak] | List[Event] = [items[0]]
    for item in items[1:]:
        if _item_time_sample(item) - _item_time_sample(cur[-1]) <= merge_dt_samples:
            cur.append(item)
        else:
            clusters.append(cur)
            cur = [item]
    clusters.append(cur)
    return clusters


def merge_candidates_to_events(
    cands: List[CandidatePeak],
    seg_t0: int,
    merge_dt_samples: int,
    merge_radius_ch: int,
    neighbors_lookup: np.ndarray,
    channel_dist: np.ndarray | None = None,
    hop_matrix: np.ndarray | None = None,
    merge_mode: str = "neighbors",
) -> List[Event]:
    if not cands:
        return []
    events: List[Event] = []
    for cluster_any in _cluster_by_time(cands, merge_dt_samples):
        cluster = cluster_any
        n = len(cluster)
        used = [False] * n
        order = sorted(range(n), key=lambda idx: float(cluster[idx].amp_norm), reverse=True)
        for idx in order:
            if used[idx]:
                continue
            anchor = cluster[idx]
            used[idx] = True
            group = [anchor]
            for j, cand in enumerate(cluster):
                if used[j]:
                    continue
                if abs(int(cand.t) - int(anchor.t)) > int(merge_dt_samples):
                    continue
                if _spatial_close(
                    anchor.c,
                    cand.c,
                    merge_radius_ch,
                    neighbors_lookup=neighbors_lookup,
                    channel_dist=channel_dist,
                    hop_matrix=hop_matrix,
                    merge_mode=merge_mode,
                ):
                    used[j] = True
                    group.append(cand)
            amps = np.array([g.amp_norm for g in group], dtype=np.float32)
            spatial_score = float(np.mean(amps) / (float(np.max(amps)) + 1e-6))
            t_vals = np.array([g.t for g in group], dtype=np.int32)
            support_channels = np.unique(np.array([g.c for g in group], dtype=np.int32)).astype(np.int32, copy=False)
            peak_ratio = 1.0
            if amps.size > 1:
                amps_desc = np.sort(amps)[::-1]
                peak_ratio = float(amps_desc[1] / max(amps_desc[0], 1e-6))
            qc_flags = int(QCFlag.NONE)
            for g in group:
                qc_flags = int(QCFlag(qc_flags) | QCFlag(getattr(g, "qc_flags", 0)))
            events.append(
                Event(
                    t0=int(seg_t0 + anchor.t),
                    c0=int(anchor.c),
                    amp_norm=float(anchor.amp_norm),
                    spatial_score=spatial_score,
                    neighbors=neighbors_lookup[anchor.c].copy(),
                    support_count=int(len(group)),
                    support_channels=support_channels,
                    time_spread_samples=int(np.max(t_vals) - np.min(t_vals)) if t_vals.size > 0 else 0,
                    peak_ratio=float(np.clip(peak_ratio, 0.0, 1.0)),
                    simul_global=int(max(int(g.simul_global) for g in group)),
                    simul_local=int(max(int(g.simul_local) for g in group)),
                    qc_flags=qc_flags,
                )
            )
    events.sort(key=lambda e: e.t0)
    return events


def _event_duplicate_score(e: Event) -> float:
    return float(e.amp_norm) * (0.5 + float(e.spatial_score))


def suppress_duplicate_events(
    events: List[Event],
    *,
    duplicate_dt_samples: int,
    duplicate_radius_ch: int,
    hop_matrix: np.ndarray | None,
    n_channels: int,
) -> tuple[List[Event], int]:
    if not events:
        return [], 0
    if duplicate_dt_samples <= 0:
        return list(events), 0
    out: List[Event] = []
    dropped = 0
    for cluster_any in _cluster_by_time(events, duplicate_dt_samples):
        cluster = cluster_any
        n = len(cluster)
        used = [False] * n
        order = sorted(range(n), key=lambda idx: _event_duplicate_score(cluster[idx]), reverse=True)
        for idx in order:
            if used[idx]:
                continue
            used[idx] = True
            keep = cluster[idx]
            out.append(keep)
            for j, cand in enumerate(cluster):
                if used[j]:
                    continue
                if abs(int(cand.t0) - int(keep.t0)) > int(duplicate_dt_samples):
                    continue
                hop = _hop_distance(
                    keep.c0,
                    cand.c0,
                    hop_matrix=hop_matrix,
                    n_channels=n_channels,
                )
                if hop <= int(duplicate_radius_ch):
                    cand.qc_flags = update_qc_flag(cand.qc_flags, QCFlag.DUPLICATE_NEARBY)
                    used[j] = True
                    dropped += 1
    out.sort(key=lambda e: e.t0)
    return out, int(dropped)


@dataclass
class DetectorConfig:
    fs: float
    alpha: float = 4.0
    refractory_ms: float = 0.30
    merge_dt_ms: float = 0.20
    merge_enabled: bool = True
    merge_mode: str = "neighbors"
    merge_radius_ch: int = 1
    thr_max_amp_norm: float = 30.0
    low_spatial_score: float = 0.2
    low_amp_norm: float = 5.0
    width_low: float = 0.1
    width_high: float = 12.0
    method: str = "fast_local"
    stats_update_every_windows: int = 4
    stats_subsample_step: int = 8
    active_halo_ms: float = 0.25
    max_simul_firing: int = 0
    global_simul_threshold: float = 0.0
    local_simul_threshold: float = 0.0
    neighbor_source: str = "auto"
    duplicate_dt_ms: float = 0.30
    duplicate_radius_ch: int = 1
    hard_min_spatial_score: float = 0.0
    hard_min_support: int = 1
    hard_filter_mode: str = "soft"
    saturation_filter_mode: str = "drop"


def build_detector_config(
    det_cfg_raw: Dict[str, Any] | None,
    *,
    fs: float,
    n_channels: int,
    defaults: Dict[str, Any] | None = None,
) -> DetectorConfig:
    raw: Dict[str, Any] = loose_detector_defaults(n_channels)
    raw.update(defaults or {})
    det_cfg_raw = det_cfg_raw or {}
    if det_cfg_raw:
        raw.update(det_cfg_raw)
    legacy_simul_raw = raw.get("max_simul_firing", 0)
    if "max_simul_firing" in det_cfg_raw and "global_simul_threshold" not in det_cfg_raw:
        global_simul_raw = det_cfg_raw.get("max_simul_firing", 0)
    else:
        global_simul_raw = raw.get("global_simul_threshold", legacy_simul_raw if legacy_simul_raw else 0.2)
    return DetectorConfig(
        fs=float(raw.get("fs", fs)),
        alpha=float(raw.get("alpha", 4.0)),
        refractory_ms=float(raw.get("refractory_ms", 0.30)),
        merge_dt_ms=float(raw.get("merge_dt_ms", 0.20)),
        merge_enabled=bool(raw.get("merge_enabled", True)),
        merge_mode=str(raw.get("merge_mode", "neighbors")),
        merge_radius_ch=int(raw.get("merge_radius_ch", 1)),
        thr_max_amp_norm=float(raw.get("thr_max_amp_norm", 30.0)),
        low_spatial_score=float(raw.get("low_spatial_score", 0.2)),
        low_amp_norm=float(raw.get("low_amp_norm", 5.0)),
        width_low=float(raw.get("width_low", 0.1)),
        width_high=float(raw.get("width_high", 12.0)),
        method=str(raw.get("method", "fast_local")),
        stats_update_every_windows=int(raw.get("stats_update_every_windows", 4)),
        stats_subsample_step=int(raw.get("stats_subsample_step", 8)),
        active_halo_ms=float(raw.get("active_halo_ms", 0.25)),
        max_simul_firing=resolve_max_simul_firing(legacy_simul_raw, int(n_channels)),
        global_simul_threshold=float(global_simul_raw),
        local_simul_threshold=float(raw.get("local_simul_threshold", 0.8)),
        neighbor_source=str(raw.get("neighbor_source", "auto")),
        duplicate_dt_ms=float(raw.get("duplicate_dt_ms", 0.30)),
        duplicate_radius_ch=int(raw.get("duplicate_radius_ch", 1)),
        hard_min_spatial_score=float(raw.get("hard_min_spatial_score", 0.0)),
        hard_min_support=int(raw.get("hard_min_support", 1)),
        hard_filter_mode=str(raw.get("hard_filter_mode", "soft")),
        saturation_filter_mode=_normalize_saturation_filter_mode(raw.get("saturation_filter_mode", "drop")),
    )


class EventDetector:
    def __init__(self, cfg: DetectorConfig, neighbors_lookup: np.ndarray, channel_dist: np.ndarray | None = None):
        mode = str(cfg.merge_mode).lower()
        if cfg.merge_enabled and mode not in ("none", "off", "false", "0", "same_channel", "channel"):
            if cfg.merge_dt_ms >= cfg.refractory_ms:
                raise ValueError("merge_dt_ms must be < refractory_ms")
        self.cfg = cfg
        self.neighbors_lookup = neighbors_lookup
        self.channel_dist = channel_dist
        self.hop_matrix = _build_hop_matrix(neighbors_lookup)
        self.global_simul_threshold = resolve_max_simul_firing(cfg.global_simul_threshold, int(neighbors_lookup.shape[0]))
        if self.global_simul_threshold <= 0 and int(cfg.max_simul_firing) > 0:
            self.global_simul_threshold = int(cfg.max_simul_firing)
        self.local_simul_threshold = resolve_max_simul_firing(
            cfg.local_simul_threshold,
            int(neighbors_lookup.shape[1]) if neighbors_lookup.ndim == 2 and neighbors_lookup.shape[1] > 0 else 1,
        )
        self.last_stats: Dict[str, int] = {
            "raw_candidates": 0,
            "merged_events": 0,
            "hard_drops": 0,
            "duplicate_drops": 0,
            "global_simul_drops": 0,
            "local_simul_drops": 0,
            "qc_low_spatial": 0,
            "qc_low_amp": 0,
            "qc_saturated": 0,
            "qc_refractory": 0,
            "final_events": 0,
        }

    def detect(
        self,
        x_seg: np.ndarray,
        seg_t0: int,
        sigma_c: np.ndarray,
        active_start: int | None = None,
        active_stop: int | None = None,
    ) -> List[Event]:
        self.last_stats = {
            "raw_candidates": 0,
            "merged_events": 0,
            "hard_drops": 0,
            "duplicate_drops": 0,
            "global_simul_drops": 0,
            "local_simul_drops": 0,
            "qc_low_spatial": 0,
            "qc_low_amp": 0,
            "qc_saturated": 0,
            "qc_refractory": 0,
            "final_events": 0,
        }
        pick_stats = {"global_simul_drops": 0, "local_simul_drops": 0}
        active_halo = max(0, int(round(float(self.cfg.active_halo_ms) * self.cfg.fs / 1000.0)))
        method = str(self.cfg.method).lower()
        if method in ("legacy", "find_peaks"):
            cands = _pick_negative_peaks_legacy(
                x_seg=x_seg,
                sigma_c=sigma_c,
                fs=self.cfg.fs,
                alpha=self.cfg.alpha,
                refractory_ms=self.cfg.refractory_ms,
            )
        else:
            cands = _pick_negative_peaks_fast(
                x_seg=x_seg,
                sigma_c=sigma_c,
                fs=self.cfg.fs,
                alpha=self.cfg.alpha,
                refractory_ms=self.cfg.refractory_ms,
                active_start=active_start,
                active_stop=active_stop,
                active_halo=active_halo,
                thr_max_amp_norm=self.cfg.thr_max_amp_norm,
                max_simul_firing=self.cfg.max_simul_firing,
                global_simul_threshold=self.global_simul_threshold,
                local_simul_threshold=self.local_simul_threshold,
                saturation_filter_mode=self.cfg.saturation_filter_mode,
                neighbors_lookup=self.neighbors_lookup,
                stats_out=pick_stats,
            )
        self.last_stats["global_simul_drops"] = int(pick_stats.get("global_simul_drops", 0))
        self.last_stats["local_simul_drops"] = int(pick_stats.get("local_simul_drops", 0))
        self.last_stats["raw_candidates"] = int(len(cands))
        if self.cfg.merge_enabled:
            merge_dt_samples = max(1, int(round(self.cfg.merge_dt_ms * self.cfg.fs / 1000.0)))
            events = merge_candidates_to_events(
                cands=cands,
                seg_t0=seg_t0,
                merge_dt_samples=merge_dt_samples,
                merge_radius_ch=self.cfg.merge_radius_ch,
                neighbors_lookup=self.neighbors_lookup,
                channel_dist=self.channel_dist,
                hop_matrix=self.hop_matrix,
                merge_mode=self.cfg.merge_mode,
            )
        else:
            events = [
                Event(
                    t0=int(seg_t0 + c.t),
                    c0=int(c.c),
                    amp_norm=float(c.amp_norm),
                    spatial_score=1.0,
                    neighbors=self.neighbors_lookup[c.c].copy(),
                    support_channels=np.array([int(c.c)], dtype=np.int32),
                    simul_global=int(c.simul_global),
                    simul_local=int(c.simul_local),
                    qc_flags=int(c.qc_flags),
                )
                for c in cands
            ]
        self.last_stats["merged_events"] = int(len(events))
        hard_kept: List[Event] = []
        hard_drops = 0
        hard_filter_mode = str(self.cfg.hard_filter_mode).strip().lower()
        for e in events:
            reject_isolated = int(e.support_count) < int(self.cfg.hard_min_support) and float(e.amp_norm) < 7.0
            reject_low_spatial = float(e.spatial_score) < float(self.cfg.hard_min_spatial_score) and float(e.amp_norm) < 8.0
            if reject_isolated or reject_low_spatial:
                if reject_low_spatial:
                    e.qc_flags = update_qc_flag(e.qc_flags, QCFlag.LOW_SPATIAL_SCORE)
                if reject_isolated:
                    e.qc_flags = update_qc_flag(e.qc_flags, QCFlag.LOW_AMP)
                if hard_filter_mode == "hard":
                    hard_drops += 1
                    continue
            hard_kept.append(e)
        events = hard_kept
        self.last_stats["hard_drops"] = int(hard_drops)
        duplicate_dt_samples = max(0, int(round(float(self.cfg.duplicate_dt_ms) * self.cfg.fs / 1000.0)))
        events, duplicate_drops = suppress_duplicate_events(
            events,
            duplicate_dt_samples=duplicate_dt_samples,
            duplicate_radius_ch=int(self.cfg.duplicate_radius_ch),
            hop_matrix=self.hop_matrix,
            n_channels=int(self.neighbors_lookup.shape[0]),
        )
        self.last_stats["duplicate_drops"] = int(duplicate_drops)
        refractory_samples = max(1, int(round(self.cfg.refractory_ms * self.cfg.fs / 1000.0)))
        prev_t_by_ch = {}
        saturation_mode = _normalize_saturation_filter_mode(self.cfg.saturation_filter_mode)
        for e in events:
            flags = QCFlag(e.qc_flags)
            if saturation_mode != "off" and self.cfg.thr_max_amp_norm > 0.0 and e.amp_norm > self.cfg.thr_max_amp_norm:
                flags = QCFlag(update_qc_flag(int(flags), QCFlag.SATURATED_ARTIFACT))
            if flags & QCFlag.SATURATED_ARTIFACT:
                self.last_stats["qc_saturated"] += 1
            if e.spatial_score < self.cfg.low_spatial_score:
                flags = QCFlag(update_qc_flag(int(flags), QCFlag.LOW_SPATIAL_SCORE))
                self.last_stats["qc_low_spatial"] += 1
            if e.amp_norm < self.cfg.low_amp_norm:
                flags = QCFlag(update_qc_flag(int(flags), QCFlag.LOW_AMP))
                self.last_stats["qc_low_amp"] += 1
            last_t = prev_t_by_ch.get(e.c0)
            if last_t is not None and e.t0 - last_t < refractory_samples:
                flags = QCFlag(update_qc_flag(int(flags), QCFlag.TOO_CLOSE_REFRACTORY))
                self.last_stats["qc_refractory"] += 1
            prev_t_by_ch[e.c0] = e.t0
            e.qc_flags = int(flags)
            amp_span = max(float(self.cfg.thr_max_amp_norm) - float(self.cfg.low_amp_norm), 1e-6)
            amp_q = float(np.clip((float(e.amp_norm) - float(self.cfg.low_amp_norm)) / amp_span, 0.0, 1.0))
            spatial_q = float(np.clip(float(e.spatial_score), 0.0, 1.0))
            q = 0.6 * amp_q + 0.4 * spatial_q
            if flags & QCFlag.LOW_AMP:
                q *= 0.5
            if flags & QCFlag.LOW_SPATIAL_SCORE:
                q *= 0.7
            if flags & QCFlag.SATURATED_ARTIFACT:
                q *= 0.2
            if flags & QCFlag.TOO_CLOSE_REFRACTORY:
                q *= 0.4
            e.q = float(np.clip(q, 0.0, 1.0))
        self.last_stats["final_events"] = int(len(events))
        return events


@dataclass
class PatchConfig:
    left: int = 20
    right: int = 44
    k: int = 16
    peak_only: bool = False

    @property
    def time_offsets(self) -> np.ndarray:
        return np.arange(-self.left, self.right, dtype=np.int32)


class PatchBuilder:
    def __init__(self, cfg: PatchConfig, neighbors_lookup: np.ndarray, channel_positions: np.ndarray):
        self.cfg = cfg
        self.neighbors_lookup = neighbors_lookup
        self.channel_positions = channel_positions
        self.time_offsets = cfg.time_offsets

    def _rel_pos_2d(self, center_ch: np.ndarray, neighbor_chs: np.ndarray) -> np.ndarray:
        pos = np.asarray(self.channel_positions, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[0] == 0:
            return np.zeros((neighbor_chs.shape[0], neighbor_chs.shape[1], 2), dtype=np.float32)
        coord = pos[:, : min(3, pos.shape[1])]
        if coord.shape[1] < 2:
            coord = np.pad(coord, ((0, 0), (0, 2 - coord.shape[1])), mode="constant")
        var = np.var(coord, axis=0)
        axes = np.argsort(-var, kind="mergesort")[:2]
        if axes.size < 2:
            axes = np.asarray([0, 1], dtype=np.int64)
        center = coord[center_ch][:, None, :][:, :, axes]
        neigh = coord[neighbor_chs][:, :, axes]
        return (neigh - center).astype(np.float32, copy=False)

    def build(self, x_seg: np.ndarray, seg_t0: int, events: List[Event]) -> Tuple[PatchBatch, List[Event]]:
        k_eff = 1 if self.cfg.peak_only else self.cfg.k
        if not events:
            empty = PatchBatch(
                x=np.zeros((0, self.time_offsets.size, k_eff), dtype=np.float32),
                meta={
                    "center_ch": np.zeros((0,), dtype=np.int32),
                    "neighbor_chs": np.zeros((0, k_eff), dtype=np.int32),
                    "geo": np.zeros((0, k_eff, 4), dtype=np.float32),
                    "rel_pos": np.zeros((0, k_eff, 2), dtype=np.float32),
                    "channel_mask": np.zeros((0, k_eff), dtype=np.bool_),
                    "t0": np.zeros((0,), dtype=np.int64),
                },
            )
            return empty, []
        local_t0 = np.array([e.t0 - seg_t0 for e in events], dtype=np.int64)
        center_ch = np.array([e.c0 for e in events], dtype=np.int32)
        if self.cfg.peak_only:
            neighbor_chs = center_ch[:, None]
        else:
            neighbor_chs = self.neighbors_lookup[center_ch]
        t_len = x_seg.shape[0]

        min_t = local_t0 + int(self.time_offsets.min())
        max_t = local_t0 + int(self.time_offsets.max())
        valid = (min_t >= 0) & (max_t < t_len)

        kept_events: List[Event] = []
        keep_idx = np.where(valid)[0]
        drop_idx = np.where(~valid)[0]
        for i in drop_idx:
            events[i].qc_flags = update_qc_flag(events[i].qc_flags, QCFlag.BORDER_CUT)
        for i in keep_idx:
            kept_events.append(events[i])

        if keep_idx.size == 0:
            empty = PatchBatch(
                x=np.zeros((0, self.time_offsets.size, k_eff), dtype=np.float32),
                meta={
                    "center_ch": np.zeros((0,), dtype=np.int32),
                    "neighbor_chs": np.zeros((0, k_eff), dtype=np.int32),
                    "geo": np.zeros((0, k_eff, 4), dtype=np.float32),
                    "rel_pos": np.zeros((0, k_eff, 2), dtype=np.float32),
                    "channel_mask": np.zeros((0, k_eff), dtype=np.bool_),
                    "t0": np.zeros((0,), dtype=np.int64),
                },
            )
            return empty, []

        local_t0 = local_t0[keep_idx].astype(np.int32)
        center_ch = center_ch[keep_idx]
        neighbor_chs = neighbor_chs[keep_idx]
        t0_global = np.array([e.t0 for e in kept_events], dtype=np.int64)

        idx_t = local_t0[:, None, None] + self.time_offsets[None, :, None]
        idx_c = neighbor_chs[:, None, :]
        x = x_seg[idx_t, idx_c].astype(np.float32, copy=False)
        geo = relative_geo(self.channel_positions, center_ch=center_ch, neighbor_chs=neighbor_chs)
        rel_pos = self._rel_pos_2d(center_ch, neighbor_chs)
        channel_mask = np.ones(neighbor_chs.shape, dtype=np.bool_)

        return (
            PatchBatch(
                x=x,
                meta={
                    "center_ch": center_ch,
                    "neighbor_chs": neighbor_chs,
                    "geo": geo,
                    "rel_pos": rel_pos,
                    "channel_mask": channel_mask,
                    "t0": t0_global,
                },
            ),
            kept_events,
        )


@dataclass
class Segment:
    data: RingView
    center_start: int
    center_stop: int


class WindowScheduler:
    def __init__(self, fs: float, window_sec: float, stride_sec: float, margin_sec: float):
        self.fs = float(fs)
        self.window = int(round(window_sec * fs))
        self.stride = int(round(stride_sec * fs))
        self.margin = int(round(margin_sec * fs))
        self.segment_len = self.window + 2 * self.margin
        self.next_now: int | None = None

    def ready(self, ring: RingBuffer) -> bool:
        self._initialize_if_needed(ring)
        if self.next_now is None:
            return False
        return self._is_now_ready(ring, self.next_now)

    def iter_ready_segments(self, ring: RingBuffer) -> Iterator[Segment]:
        self._initialize_if_needed(ring)
        if self.next_now is None:
            return
        while self._is_now_ready(ring, self.next_now):
            now = self.next_now
            seg_start = now - self.window - self.margin
            seg_stop = now + self.margin
            view = ring.get_range(seg_start, seg_stop)
            yield Segment(data=view, center_start=now - self.window, center_stop=now)
            self.next_now += self.stride

    def keep_center_window(self, events: List[Event], center_start: int, center_stop: int) -> List[Event]:
        # Keep only the non-overlapping stride sub-window to avoid emitting
        # the same event repeatedly from overlapped center windows.
        keep_stop = min(center_stop, center_start + self.stride)
        return [e for e in events if center_start <= e.t0 < keep_stop]

    def _initialize_if_needed(self, ring: RingBuffer) -> None:
        if self.next_now is not None:
            return
        if not ring.has(self.segment_len):
            return
        self.next_now = ring.global_t0_of_buffer_start + self.window + self.margin
        while not self._is_now_ready(ring, self.next_now):
            self.next_now += self.stride
            if self.next_now > ring.global_t1_exclusive:
                break

    def _is_now_ready(self, ring: RingBuffer, now: int) -> bool:
        seg_start = now - self.window - self.margin
        seg_stop = now + self.margin
        return seg_start >= ring.global_t0_of_buffer_start and seg_stop <= ring.global_t1_exclusive


@dataclass
class PipelineConfig:
    fs: float
    n_channels: int
    chunk_sec: float = 0.1
    window_sec: float = 1.5
    stride_sec: float = 0.5
    margin_sec: float = 0.05
    ring_capacity_sec: float = 8.0
    device: str = "cpu"


class OnlineSpikePipeline:
    def __init__(
        self,
        pipe_cfg: PipelineConfig,
        det_cfg: DetectorConfig,
        patch_cfg: PatchConfig,
        model_cfg: ModelConfig,
        cluster_cfg: Optional[Any] = None,
        channel_positions: Optional[np.ndarray] = None,
        neighbors_lookup: Optional[np.ndarray] = None,
        car_mode: str = "global",
        car_groups: Optional[List[List[int]]] = None,
        noise_beta: float = 0.05,
        patch_norm_mode: str = "peak_channel_sigma",
        clusterer: Optional[Any] = None,
        cluster_enabled: bool = True,
        apply_encode_gate: bool = True,
    ):
        self.cfg = pipe_cfg
        self.fs = pipe_cfg.fs
        self.n_channels = pipe_cfg.n_channels
        self.device = torch.device(pipe_cfg.device)
        self.patch_norm_mode = str(patch_norm_mode).lower()
        self.apply_encode_gate = bool(apply_encode_gate)
        if self.patch_norm_mode not in ("peak_channel_sigma", "none"):
            raise ValueError(
                f"unsupported patch_norm_mode={patch_norm_mode}. use peak_channel_sigma | none"
            )

        self.channel_positions = ensure_channel_positions(self.n_channels, channel_positions)
        if neighbors_lookup is None:
            self.neighbors = build_knn_neighbors(self.channel_positions, patch_cfg.k)
        else:
            neigh = np.asarray(neighbors_lookup, dtype=np.int32)
            if neigh.ndim != 2 or neigh.shape[0] != self.n_channels:
                raise ValueError(
                    f"neighbors_lookup shape mismatch: expected [{self.n_channels}, K], got {neigh.shape}"
                )
            if not patch_cfg.peak_only and neigh.shape[1] != patch_cfg.k:
                raise ValueError(
                    f"patch_cfg.k ({patch_cfg.k}) must match neighbors_lookup width ({neigh.shape[1]})"
                )
            self.neighbors = neigh
        self.channel_dist = pairwise_distances(self.channel_positions)

        cap_samples = int(round(pipe_cfg.ring_capacity_sec * self.fs))
        self.ring_raw = RingBuffer(cap_samples, self.n_channels)
        self.ring_filt = RingBuffer(cap_samples, self.n_channels)

        self.bandpass = StreamBandpass(
            fs=self.fs,
            low_hz=300.0,
            high_hz=6000.0,
            order=3,
            n_channels=self.n_channels,
        )
        self.car = StreamCAR(mode=car_mode, groups=car_groups)
        self.noise = RobustNoiseEstimator(
            self.n_channels,
            beta=noise_beta,
            update_every_windows=int(det_cfg.stats_update_every_windows),
            subsample_step=int(det_cfg.stats_subsample_step),
        )
        self.scheduler = WindowScheduler(self.fs, pipe_cfg.window_sec, pipe_cfg.stride_sec, pipe_cfg.margin_sec)

        self.detector = EventDetector(det_cfg, neighbors_lookup=self.neighbors, channel_dist=self.channel_dist)
        self.patch_builder = PatchBuilder(patch_cfg, neighbors_lookup=self.neighbors, channel_positions=self.channel_positions)

        self.model = SpikeEmbeddingModel(model_cfg).to(self.device)
        self.model.eval()
        _ = cluster_cfg
        self.cluster_enabled = bool(cluster_enabled and clusterer is not None)
        self.clusterer = clusterer

        self.logger = get_logger("mysort.pipeline")
        self.monitor = Monitor()
        self.stats = PipelineStats()
        self.window_index = 0

    def process_chunk(self, x_new: np.ndarray) -> Tuple[List[EventOutput], Dict[str, float]]:
        t_start = time.perf_counter()
        outputs: List[EventOutput] = []
        preprocess_ms = 0.0
        detect_ms = 0.0
        patch_ms = 0.0
        model_ms = 0.0
        cluster_ms = 0.0
        seg_count = 0
        chunk_events = 0
        chunk_patches = 0
        chunk_encoded = 0
        chunk_cluster_inputs = 0
        chunk_detect_raw_candidates = 0
        chunk_detect_merged_events = 0
        chunk_detect_hard_drops = 0
        chunk_detect_duplicate_drops = 0
        chunk_detect_global_simul_drops = 0
        chunk_detect_local_simul_drops = 0
        if x_new.size == 0:
            self.monitor.inc("stall", 1)
            self.stats.n_stalls += 1
            return outputs, self.monitor.snapshot()

        t_stage = time.perf_counter()
        self.ring_raw.push(x_new)
        self.stats.n_raw_samples += x_new.shape[0]

        y = self.bandpass.process(x_new)
        y = self.car.process(y)
        self.ring_filt.push(y)
        self.stats.n_filt_samples += y.shape[0]
        preprocess_ms += (time.perf_counter() - t_stage) * 1_000.0

        for seg in self.scheduler.iter_ready_segments(self.ring_filt):
            self.window_index += 1
            seg_count += 1

            t_stage = time.perf_counter()
            sigma = self.noise.update(seg.data.x[self.scheduler.margin : self.scheduler.margin + self.scheduler.window])
            active_start = int(self.scheduler.margin)
            active_stop = int(self.scheduler.margin + self.scheduler.stride)
            events = self.detector.detect(
                seg.data.x,
                seg.data.t0,
                sigma,
                active_start=active_start,
                active_stop=active_stop,
            )
            chunk_detect_raw_candidates += int(self.detector.last_stats.get("raw_candidates", 0))
            chunk_detect_merged_events += int(self.detector.last_stats.get("merged_events", 0))
            chunk_detect_hard_drops += int(self.detector.last_stats.get("hard_drops", 0))
            chunk_detect_duplicate_drops += int(self.detector.last_stats.get("duplicate_drops", 0))
            chunk_detect_global_simul_drops += int(self.detector.last_stats.get("global_simul_drops", 0))
            chunk_detect_local_simul_drops += int(self.detector.last_stats.get("local_simul_drops", 0))
            events = self.scheduler.keep_center_window(events, seg.center_start, seg.center_stop)
            detect_ms += (time.perf_counter() - t_stage) * 1_000.0
            self.stats.n_events += len(events)
            chunk_events += len(events)

            t_stage = time.perf_counter()
            patch_batch, kept_events = self.patch_builder.build(seg.data.x, seg.data.t0, events)
            patch_ms += (time.perf_counter() - t_stage) * 1_000.0
            self.stats.n_patches += patch_batch.x.shape[0]
            chunk_patches += int(patch_batch.x.shape[0])
            if patch_batch.x.shape[0] == 0:
                continue

            t_stage = time.perf_counter()
            x_np = patch_batch.x.astype(np.float32, copy=False)
            if self.patch_norm_mode == "peak_channel_sigma":
                neigh = patch_batch.meta["neighbor_chs"].astype(np.int64, copy=False)
                sigma_neigh = sigma[neigh].astype(np.float32, copy=False)
                offsets = self.patch_builder.time_offsets.astype(np.int32, copy=False)
                if offsets.shape[0] != x_np.shape[1]:
                    offsets = np.arange(-int(self.patch_builder.cfg.left), -int(self.patch_builder.cfg.left) + x_np.shape[1], dtype=np.int32)
                baseline_mask = (offsets >= -20) & (offsets <= -6)
                if not np.any(baseline_mask):
                    baseline_mask[: min(x_np.shape[1], max(1, int(self.patch_builder.cfg.left)))] = True
                baseline = np.median(x_np[:, baseline_mask, :], axis=1, keepdims=True)
                x_np = (x_np - baseline) / np.maximum(sigma_neigh[:, None, :], 1e-6)
            x_t = torch.from_numpy(x_np).float().to(self.device)
            meta_t = {"geo": torch.from_numpy(patch_batch.meta["geo"]).float().to(self.device)}
            if "rel_pos" in patch_batch.meta:
                meta_t["rel_pos"] = torch.from_numpy(patch_batch.meta["rel_pos"]).float().to(self.device)
            if "channel_mask" in patch_batch.meta:
                meta_t["channel_mask"] = torch.from_numpy(patch_batch.meta["channel_mask"].astype(np.bool_, copy=False)).to(self.device)
            if kept_events and self.model.cfg.gate_spatial_dim > 0:
                ss = event_spatial_stats(
                    kept_events,
                    n_channels=int(self.n_channels),
                    neighbor_count=int(self.neighbors.shape[1]) if self.neighbors.ndim == 2 else 1,
                )[:, : int(self.model.cfg.gate_spatial_dim)]
                meta_t["spatial_stats"] = torch.from_numpy(ss).float().to(self.device)

            with torch.no_grad():
                if self.apply_encode_gate:
                    model_out = self.model(x_t, meta_t)
                else:
                    model_out = self.model(x_t, meta_t, apply_encode_gate=False)
            z_np = model_out.embedding.detach().cpu().numpy().astype(np.float32, copy=False)
            gate_prob_np = model_out.spike_prob.detach().cpu().numpy().astype(np.float32, copy=False)
            overlap_prob_np = np.full((gate_prob_np.shape[0],), np.nan, dtype=np.float32)
            encoded_mask_np = model_out.encoded_mask.detach().cpu().numpy().astype(np.bool_, copy=False)
            model_ms += (time.perf_counter() - t_stage) * 1_000.0

            cluster_mask_np = (
                encoded_mask_np
                & (gate_prob_np >= float(self.model.cfg.assignment_gate_thresh))
            )
            cluster_idx = np.flatnonzero(cluster_mask_np)
            self.stats.n_encoded += int(np.sum(encoded_mask_np))
            self.stats.n_cluster_inputs += int(cluster_idx.size)
            chunk_encoded += int(np.sum(encoded_mask_np))
            chunk_cluster_inputs += int(cluster_idx.size)
            for ev, gate_prob, overlap_prob, encoded in zip(kept_events, gate_prob_np.tolist(), overlap_prob_np.tolist(), encoded_mask_np.tolist()):
                ev.gate_prob = float(gate_prob)
                ev.overlap_prob = float(overlap_prob)
                ev.encoded_mask = bool(encoded)

            t_stage = time.perf_counter()
            cluster_events = [kept_events[int(i)] for i in cluster_idx.tolist()]
            cluster_z = z_np[cluster_idx] if cluster_idx.size > 0 else np.zeros((0, z_np.shape[1]), dtype=np.float32)
            outs: List[EventOutput] = []
            cl_stats = {"births": 0, "matches": 0, "drops": 0, "init_flushed": 0}
            if not self.cluster_enabled:
                for i, ev in enumerate(kept_events):
                    outs.append(
                        EventOutput(
                            event_time_sample=int(ev.t0),
                            event_time_sec=float(ev.t0 / self.fs),
                            center_channel=int(ev.c0),
                            unit_id=-1,
                            confidence=float(np.clip(float(ev.q) * float(ev.gate_prob), 0.0, 1.0)),
                            overlap_prob_unavailable=float(ev.overlap_prob),
                            gate_prob=float(ev.gate_prob),
                            encoded_mask=int(bool(ev.encoded_mask)),
                            init_buffered=0,
                            qc_flags=int(ev.qc_flags),
                            embedding=z_np[i].copy(),
                            overlap_prob=float(ev.overlap_prob),
                        )
                    )
            elif cluster_events:
                outs, cl_stats = self.clusterer.update(cluster_events, cluster_z)
            if self.cluster_enabled and cluster_idx.size < len(kept_events):
                for i, ev in enumerate(kept_events):
                    if cluster_mask_np[i]:
                        continue
                    outs.append(
                        EventOutput(
                            event_time_sample=int(ev.t0),
                            event_time_sec=float(ev.t0 / self.fs),
                            center_channel=int(ev.c0),
                            unit_id=-1,
                            confidence=float(np.clip(0.25 * float(ev.q) * float(ev.gate_prob), 0.0, 1.0)),
                            overlap_prob_unavailable=float(ev.overlap_prob),
                            gate_prob=float(ev.gate_prob),
                            encoded_mask=int(bool(ev.encoded_mask)),
                            init_buffered=0,
                            qc_flags=int(ev.qc_flags),
                            embedding=z_np[i].copy(),
                            overlap_prob=float(ev.overlap_prob),
                        )
                    )
            cluster_ms += (time.perf_counter() - t_stage) * 1_000.0
            outputs.extend(outs)
            self.stats.n_births += cl_stats["births"]
            self.stats.n_matches += cl_stats["matches"]
            self.stats.n_drops += cl_stats["drops"]
            self.stats.n_init_flushed += int(cl_stats.get("init_flushed", 0))

        self.stats.last_latency_ms = (time.perf_counter() - t_start) * 1_000.0
        self.stats.last_preprocess_ms = preprocess_ms
        self.stats.last_detect_ms = detect_ms
        self.stats.last_patch_ms = patch_ms
        self.stats.last_model_ms = model_ms
        self.stats.last_cluster_ms = cluster_ms
        self.stats.last_segments = seg_count
        self.stats.last_chunk_events = chunk_events
        self.stats.last_chunk_patches = chunk_patches
        self.stats.last_chunk_encoded = chunk_encoded
        self.stats.last_chunk_cluster_inputs = chunk_cluster_inputs
        self.stats.last_detect_raw_candidates = chunk_detect_raw_candidates
        self.stats.last_detect_merged_events = chunk_detect_merged_events
        self.stats.last_detect_hard_drops = chunk_detect_hard_drops
        self.stats.last_detect_duplicate_drops = chunk_detect_duplicate_drops
        self.stats.last_detect_global_simul_drops = chunk_detect_global_simul_drops
        self.stats.last_detect_local_simul_drops = chunk_detect_local_simul_drops
        self.stats.total_preprocess_ms += preprocess_ms
        self.stats.total_detect_ms += detect_ms
        self.stats.total_patch_ms += patch_ms
        self.stats.total_model_ms += model_ms
        self.stats.total_cluster_ms += cluster_ms
        self.stats.total_detect_raw_candidates += int(chunk_detect_raw_candidates)
        self.stats.total_detect_merged_events += int(chunk_detect_merged_events)
        self.stats.total_detect_hard_drops += int(chunk_detect_hard_drops)
        self.stats.total_detect_duplicate_drops += int(chunk_detect_duplicate_drops)
        self.stats.total_detect_global_simul_drops += int(chunk_detect_global_simul_drops)
        self.stats.total_detect_local_simul_drops += int(chunk_detect_local_simul_drops)
        return outputs, self.monitor.snapshot()


__all__ = [
    "DetectorConfig",
    "EventDetector",
    "FileTail",
    "IntanStreamReader",
    "OnlineSpikePipeline",
    "PatchBuilder",
    "PatchConfig",
    "PipelineConfig",
    "RobustNoiseEstimator",
    "StreamBandpass",
    "StreamCAR",
    "WindowScheduler",
    "event_spatial_stats",
    "loose_detector_defaults",
    "merge_candidates_to_events",
    "pick_negative_peaks",
]
