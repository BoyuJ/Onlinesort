from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
except Exception:  # pragma: no cover
    accuracy_score = None
    adjusted_rand_score = None
    normalized_mutual_info_score = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

try:
    from scipy.stats import norm, spearmanr
except Exception:  # pragma: no cover
    norm = None
    spearmanr = None

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):  # type: ignore[misc]
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _wrap(func):
            return func

        return _wrap

try:
    from tqdm import trange
except Exception:  # pragma: no cover
    def trange(n: int):
        return range(n)


def _cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    true_labels, true_inv = np.unique(y_true, return_inverse=True)
    pred_labels, pred_inv = np.unique(y_pred, return_inverse=True)
    if true_labels.size == 0 or pred_labels.size == 0:
        return float("nan")

    conf = np.zeros((true_labels.size, pred_labels.size), dtype=np.int64)
    np.add.at(conf, (true_inv, pred_inv), 1)

    if linear_sum_assignment is None:
        row_best = conf.max(axis=1)
        return float(np.sum(row_best) / max(1, y_true.size))

    r, c = linear_sum_assignment(conf.max() - conf)
    correct = int(conf[r, c].sum())
    return float(correct / max(1, y_true.size))


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {"ARI": float("nan"), "NMI": float("nan"), "Acc": float("nan"), "RawAcc": float("nan")}
    if y_true.size == 0:
        return out
    if adjusted_rand_score is not None:
        out["ARI"] = float(adjusted_rand_score(y_true, y_pred))
    if normalized_mutual_info_score is not None:
        out["NMI"] = float(normalized_mutual_info_score(y_true, y_pred))
    out["Acc"] = _cluster_accuracy(y_true, y_pred)
    if accuracy_score is not None:
        out["RawAcc"] = float(accuracy_score(y_true, y_pred))
    return out


def count_time_matches_one_to_one(pred_times: np.ndarray, gt_times: np.ndarray, tol_samples: int) -> int:
    if pred_times.size == 0 or gt_times.size == 0:
        return 0
    pred = np.sort(np.asarray(pred_times, dtype=np.int64), kind="mergesort")
    gt = np.sort(np.asarray(gt_times, dtype=np.int64), kind="mergesort")
    i = 0
    j = 0
    matches = 0
    tol = int(tol_samples)
    while i < pred.size and j < gt.size:
        dt = int(pred[i]) - int(gt[j])
        if abs(dt) <= tol:
            matches += 1
            i += 1
            j += 1
        elif dt < -tol:
            i += 1
        else:
            j += 1
    return int(matches)


def detection_metrics(pred_times: np.ndarray, gt_times: np.ndarray, tol_samples: int) -> Dict[str, float | int]:
    pred = np.asarray(pred_times, dtype=np.int64)
    gt = np.asarray(gt_times, dtype=np.int64)
    tp = count_time_matches_one_to_one(pred, gt, tol_samples)
    fp = int(pred.size - tp)
    fn = int(gt.size - tp)
    derived = _metrics_from_counts(float(tp), float(fp), float(fn))
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "Precision": float(derived["precision"]),
        "Recall": float(derived["recall"]),
        "F1": float(derived["f1"]),
        "FP_rate": float(derived["fp_rate"]),
        "FN_rate": float(derived["fn_rate"]),
        "sorted_score": float(derived["sorted_score"]),
    }


@dataclass
class EventSet:
    times_samples: np.ndarray
    labels: np.ndarray
    channels: np.ndarray | None = None
    confidences: np.ndarray | None = None
    unit_positions: Dict[int, np.ndarray] | None = None
    metadata: Dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.times_samples = np.asarray(self.times_samples, dtype=np.int64).reshape(-1)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if self.times_samples.shape != self.labels.shape:
            raise ValueError(
                f"EventSet shape mismatch: times={self.times_samples.shape} labels={self.labels.shape}"
            )
        if self.channels is not None:
            self.channels = np.asarray(self.channels, dtype=np.int32).reshape(-1)
            if self.channels.shape != self.times_samples.shape:
                raise ValueError(
                    f"EventSet channel shape mismatch: channels={self.channels.shape} events={self.times_samples.shape}"
                )
        if self.confidences is not None:
            self.confidences = np.asarray(self.confidences, dtype=np.float32).reshape(-1)
            if self.confidences.shape != self.times_samples.shape:
                raise ValueError(
                    f"EventSet confidence shape mismatch: confidences={self.confidences.shape} events={self.times_samples.shape}"
                )
        if self.unit_positions is not None:
            self.unit_positions = {
                int(unit): np.asarray(pos, dtype=np.float32).reshape(-1)
                for unit, pos in dict(self.unit_positions).items()
            }
        self.metadata = dict(self.metadata or {})

    @property
    def n_events(self) -> int:
        return int(self.times_samples.size)

    def subset(self, indexer: np.ndarray | Sequence[int]) -> "EventSet":
        idx = np.asarray(indexer)
        return EventSet(
            times_samples=self.times_samples[idx],
            labels=self.labels[idx],
            channels=None if self.channels is None else self.channels[idx],
            confidences=None if self.confidences is None else self.confidences[idx],
            unit_positions=self.unit_positions,
            metadata=self.metadata,
        )

    def sorted_copy(self) -> "EventSet":
        if self.n_events <= 1:
            return self
        order = np.argsort(self.times_samples, kind="mergesort")
        return self.subset(order)


@dataclass
class BenchmarkConfig:
    # Kilosort-style benchmark defaults. When tol_samples is set it wins over
    # tol_ms; run_offline currently passes tol_samples=6 for bench.py parity.
    tol_samples: int | None = None
    tol_ms: float | None = 0.3
    min_sort_label: int = 0
    n_check: int = 0
    candidate_match_mode: str = "all"
    score_threshold: float = 0.8
    snr_threshold: float = 8.0
    # Legacy event-level benchmark knobs kept for compatibility.
    event_match_mode: str = "auto"
    channel_radius_ch: int = 3
    window_sec: float = 60.0
    recovery_f1_threshold: float = 0.8
    diagnostics: bool = True


@dataclass(frozen=True)
class EventMatch:
    pred_index: int
    gt_index: int
    dt_samples: int
    channel_delta: int | None = None


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den <= 0:
        return float(default)
    return float(num / den)


def _safe_f1(tp: float, fp: float, fn: float) -> float:
    return _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn, default=0.0)


def _sorted_score_from_rates(fp_rate: float, fn_rate: float) -> float:
    return float(np.clip(1.0 - float(fp_rate) - float(fn_rate), 0.0, 1.0))


def _metrics_from_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = _safe_ratio(float(tp), float(tp + fp), default=0.0)
    recall = _safe_ratio(float(tp), float(tp + fn), default=0.0)
    fp_rate = _safe_ratio(float(fp), float(tp + fp), default=0.0)
    fn_rate = _safe_ratio(float(fn), float(tp + fn), default=0.0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": _safe_f1(float(tp), float(fp), float(fn)),
        "fp_rate": float(fp_rate),
        "fn_rate": float(fn_rate),
        "sorted_score": _sorted_score_from_rates(fp_rate, fn_rate),
    }


def _resolve_event_match_mode(pred: EventSet, gt: EventSet, cfg: BenchmarkConfig) -> str:
    requested = str(cfg.event_match_mode).lower()
    if requested in ("time", "temporal"):
        return "time"
    if requested in ("time+channel", "channel", "spatial", "time_channel"):
        return "time+channel"
    if requested not in ("auto", ""):
        raise ValueError(f"unsupported event_match_mode={cfg.event_match_mode}")
    if pred.channels is not None and gt.channels is not None:
        return "time+channel"
    return "time"


def _solve_cost_assignment(rows: List[int], cols: List[int], cost_mat: np.ndarray) -> List[Tuple[int, int]]:
    if len(rows) == 0 or len(cols) == 0:
        return []
    finite = np.isfinite(cost_mat)
    if not np.any(finite):
        return []
    if linear_sum_assignment is not None:
        dense = np.where(finite, cost_mat, 1e9).astype(np.float64)
        rr, cc = linear_sum_assignment(dense)
        out: List[Tuple[int, int]] = []
        for r_idx, c_idx in zip(rr.tolist(), cc.tolist()):
            if not np.isfinite(cost_mat[r_idx, c_idx]):
                continue
            out.append((rows[r_idx], cols[c_idx]))
        return out
    edges: List[Tuple[float, int, int]] = []
    rr, cc = np.nonzero(finite)
    for r_idx, c_idx in zip(rr.tolist(), cc.tolist()):
        edges.append((float(cost_mat[r_idx, c_idx]), int(rows[r_idx]), int(cols[c_idx])))
    edges.sort(key=lambda item: item[0])
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    out = []
    for _, row, col in edges:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        out.append((int(row), int(col)))
    return out


def _match_events(
    pred: EventSet,
    gt: EventSet,
    *,
    tol_samples: int,
    match_mode: str,
    channel_radius_ch: int,
) -> List[EventMatch]:
    if pred.n_events == 0 or gt.n_events == 0:
        return []

    pred_to_gt: List[List[int]] = [[] for _ in range(pred.n_events)]
    gt_to_pred: List[List[int]] = [[] for _ in range(gt.n_events)]
    gt_times = gt.times_samples
    radius = max(0, int(channel_radius_ch))
    use_channel = match_mode == "time+channel" and pred.channels is not None and gt.channels is not None

    for pi, t_pred in enumerate(pred.times_samples.tolist()):
        lo = int(np.searchsorted(gt_times, int(t_pred) - int(tol_samples), side="left"))
        hi = int(np.searchsorted(gt_times, int(t_pred) + int(tol_samples), side="right"))
        if lo >= hi:
            continue
        for gi in range(lo, hi):
            if use_channel:
                dc = abs(int(pred.channels[pi]) - int(gt.channels[gi]))
                if dc > radius:
                    continue
            pred_to_gt[pi].append(int(gi))
            gt_to_pred[gi].append(int(pi))

    matches: List[EventMatch] = []
    seen_pred = np.zeros((pred.n_events,), dtype=np.bool_)
    seen_gt = np.zeros((gt.n_events,), dtype=np.bool_)

    for start_pi in range(pred.n_events):
        if seen_pred[start_pi] or not pred_to_gt[start_pi]:
            continue
        comp_pred: set[int] = set()
        comp_gt: set[int] = set()
        q_pred: List[int] = [int(start_pi)]
        q_gt: List[int] = []
        seen_pred[start_pi] = True

        while q_pred or q_gt:
            while q_pred:
                pi = q_pred.pop()
                comp_pred.add(int(pi))
                for gi in pred_to_gt[pi]:
                    if seen_gt[gi]:
                        continue
                    seen_gt[gi] = True
                    q_gt.append(int(gi))
            while q_gt:
                gi = q_gt.pop()
                comp_gt.add(int(gi))
                for pi in gt_to_pred[gi]:
                    if seen_pred[pi]:
                        continue
                    seen_pred[pi] = True
                    q_pred.append(int(pi))

        pred_rows = sorted(comp_pred)
        gt_cols = sorted(comp_gt)
        if not pred_rows or not gt_cols:
            continue
        cost = np.full((len(pred_rows), len(gt_cols)), np.inf, dtype=np.float32)
        gt_pos = {int(gi): idx for idx, gi in enumerate(gt_cols)}
        for row_idx, pi in enumerate(pred_rows):
            for gi in pred_to_gt[pi]:
                col_idx = gt_pos.get(int(gi))
                if col_idx is None:
                    continue
                dt = abs(int(pred.times_samples[pi]) - int(gt.times_samples[gi]))
                ch_cost = 0.0
                ch_delta: int | None = None
                if use_channel and pred.channels is not None and gt.channels is not None:
                    ch_delta = abs(int(pred.channels[pi]) - int(gt.channels[gi]))
                    ch_cost = 1e-3 * float(ch_delta)
                cost[row_idx, col_idx] = float(dt) + ch_cost
        for pi, gi in _solve_cost_assignment(pred_rows, gt_cols, cost):
            ch_delta = None
            if use_channel and pred.channels is not None and gt.channels is not None:
                ch_delta = abs(int(pred.channels[pi]) - int(gt.channels[gi]))
            matches.append(
                EventMatch(
                    pred_index=int(pi),
                    gt_index=int(gi),
                    dt_samples=int(pred.times_samples[pi] - gt.times_samples[gi]),
                    channel_delta=ch_delta,
                )
            )

    matches.sort(key=lambda item: int(gt.times_samples[item.gt_index]) if gt.n_events > 0 else int(pred.times_samples[item.pred_index]))
    return matches


def match_events(
    pred: EventSet,
    gt: EventSet,
    *,
    tol_samples: int | None = None,
    tol_ms: float | None = None,
    fs: float | None = None,
    event_match_mode: str = "auto",
    channel_radius_ch: int = 3,
) -> List[EventMatch]:
    if tol_samples is None:
        if tol_ms is None:
            raise ValueError("set tol_samples or tol_ms")
        if fs is None:
            fs_values = [
                float(meta["fs"])
                for meta in (pred.metadata or {}, gt.metadata or {})
                if "fs" in meta
            ]
            if not fs_values:
                raise ValueError("fs is required when tol_ms is used and EventSet metadata has no fs")
            fs = fs_values[0]
        tol_samples = int(round(float(tol_ms) * float(fs) / 1000.0))
    cfg = BenchmarkConfig(
        tol_ms=0.0 if tol_ms is None else float(tol_ms),
        event_match_mode=str(event_match_mode),
        channel_radius_ch=int(channel_radius_ch),
    )
    pred_order = np.argsort(pred.times_samples, kind="mergesort")
    gt_order = np.argsort(gt.times_samples, kind="mergesort")
    pred_sorted = pred.subset(pred_order)
    gt_sorted = gt.subset(gt_order)
    match_mode = _resolve_event_match_mode(pred_sorted, gt_sorted, cfg)
    sorted_matches = _match_events(
        pred_sorted,
        gt_sorted,
        tol_samples=max(0, int(tol_samples)),
        match_mode=match_mode,
        channel_radius_ch=int(channel_radius_ch),
    )
    return [
        EventMatch(
            pred_index=int(pred_order[int(match.pred_index)]),
            gt_index=int(gt_order[int(match.gt_index)]),
            dt_samples=int(match.dt_samples),
            channel_delta=match.channel_delta,
        )
        for match in sorted_matches
    ]


def _build_unit_overlap(
    pred: EventSet,
    gt: EventSet,
    matches: Sequence[EventMatch],
    *,
    min_sort_label: int,
    pred_unit_values: np.ndarray | None = None,
    gt_unit_values: np.ndarray | None = None,
) -> Dict[str, Any]:
    if gt_unit_values is None:
        gt_units = np.unique(gt.labels.astype(np.int64, copy=False))
    else:
        gt_units = np.asarray(gt_unit_values, dtype=np.int64).reshape(-1)
    pred_sort_mask = pred.labels >= int(min_sort_label)
    if pred_unit_values is None:
        pred_units = np.unique(pred.labels[pred_sort_mask].astype(np.int64, copy=False))
    else:
        pred_units = np.asarray(pred_unit_values, dtype=np.int64).reshape(-1)
    gt_unit_to_idx = {int(unit): idx for idx, unit in enumerate(gt_units.tolist())}
    pred_unit_to_idx = {int(unit): idx for idx, unit in enumerate(pred_units.tolist())}

    gt_counts = np.zeros((gt_units.size,), dtype=np.int64)
    pred_counts = np.zeros((pred_units.size,), dtype=np.int64)
    tp = np.zeros((pred_units.size, gt_units.size), dtype=np.int64)

    for idx, unit in enumerate(gt_units.tolist()):
        gt_counts[idx] = int(np.sum(gt.labels == int(unit)))
    for idx, unit in enumerate(pred_units.tolist()):
        pred_counts[idx] = int(np.sum(pred.labels == int(unit)))

    for match in matches:
        p_label = int(pred.labels[match.pred_index])
        if p_label < int(min_sort_label):
            continue
        g_label = int(gt.labels[match.gt_index])
        p_idx = pred_unit_to_idx.get(p_label)
        g_idx = gt_unit_to_idx.get(g_label)
        if p_idx is None or g_idx is None:
            continue
        tp[p_idx, g_idx] += 1

    pair_f1 = np.zeros_like(tp, dtype=np.float32)
    if pred_units.size > 0 and gt_units.size > 0:
        fp = pred_counts[:, None] - tp
        fn = gt_counts[None, :] - tp
        denom = (2 * tp + fp + fn).astype(np.float32)
        valid = denom > 0
        pair_f1[valid] = (2.0 * tp[valid].astype(np.float32)) / denom[valid]

    pred_to_gt: Dict[int, int] = {}
    gt_to_pred: Dict[int, int] = {}
    if pred_units.size > 0 and gt_units.size > 0:
        if linear_sum_assignment is not None:
            max_tp = float(np.max(tp)) if tp.size > 0 else 0.0
            bonus = tp.astype(np.float32) / max(1.0, max_tp + 1.0)
            cost = 1.0 - pair_f1 - (1e-6 * bonus)
            rr, cc = linear_sum_assignment(cost.astype(np.float64))
            for r_idx, c_idx in zip(rr.tolist(), cc.tolist()):
                if int(tp[r_idx, c_idx]) <= 0 or float(pair_f1[r_idx, c_idx]) <= 0.0:
                    continue
                pred_to_gt[int(pred_units[r_idx])] = int(gt_units[c_idx])
                gt_to_pred[int(gt_units[c_idx])] = int(pred_units[r_idx])
        else:
            edges: List[Tuple[float, int, int, int]] = []
            for r_idx in range(pred_units.size):
                for c_idx in range(gt_units.size):
                    if int(tp[r_idx, c_idx]) <= 0:
                        continue
                    edges.append(
                        (
                            float(pair_f1[r_idx, c_idx]),
                            int(tp[r_idx, c_idx]),
                            int(r_idx),
                            int(c_idx),
                        )
                    )
            edges.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
            used_pred: set[int] = set()
            used_gt: set[int] = set()
            for score, _, r_idx, c_idx in edges:
                if score <= 0.0 or r_idx in used_pred or c_idx in used_gt:
                    continue
                used_pred.add(r_idx)
                used_gt.add(c_idx)
                pred_to_gt[int(pred_units[r_idx])] = int(gt_units[c_idx])
                gt_to_pred[int(gt_units[c_idx])] = int(pred_units[r_idx])

    return {
        "pred_units": pred_units.astype(np.int64, copy=False),
        "gt_units": gt_units.astype(np.int64, copy=False),
        "pred_counts": pred_counts.astype(np.int64, copy=False),
        "gt_counts": gt_counts.astype(np.int64, copy=False),
        "tp": tp.astype(np.int64, copy=False),
        "pair_f1": pair_f1.astype(np.float32, copy=False),
        "pred_unit_to_idx": pred_unit_to_idx,
        "gt_unit_to_idx": gt_unit_to_idx,
        "pred_to_gt": pred_to_gt,
        "gt_to_pred": gt_to_pred,
    }


def _mapped_sort_summary(
    overlap: Dict[str, Any],
    *,
    recovery_f1_threshold: float,
    mapping_overlap: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    mapping_overlap = overlap if mapping_overlap is None else mapping_overlap
    pred_units = overlap["pred_units"]
    gt_units = overlap["gt_units"]
    pred_counts = overlap["pred_counts"]
    gt_counts = overlap["gt_counts"]
    tp = overlap["tp"]
    pair_f1 = overlap["pair_f1"]
    pred_to_gt = mapping_overlap["pred_to_gt"]
    gt_to_pred = mapping_overlap["gt_to_pred"]
    pred_unit_to_idx = overlap["pred_unit_to_idx"]
    gt_unit_to_idx = overlap["gt_unit_to_idx"]

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    macro_scores: List[float] = []
    recovered = 0

    for gt_unit in gt_units.tolist():
        g_idx = gt_unit_to_idx[int(gt_unit)]
        mapped_pred = gt_to_pred.get(int(gt_unit))
        if mapped_pred is None:
            unit_tp = 0
            unit_fp = 0
            unit_fn = int(gt_counts[g_idx])
            unit_f1 = 0.0
        else:
            p_idx = pred_unit_to_idx[int(mapped_pred)]
            unit_tp = int(tp[p_idx, g_idx])
            unit_fp = int(pred_counts[p_idx] - unit_tp)
            unit_fn = int(gt_counts[g_idx] - unit_tp)
            unit_f1 = float(pair_f1[p_idx, g_idx])
        micro_tp += int(unit_tp)
        micro_fp += int(unit_fp)
        micro_fn += int(unit_fn)
        macro_scores.append(float(unit_f1))
        if float(unit_f1) >= float(recovery_f1_threshold):
            recovered += 1

    for pred_unit in pred_units.tolist():
        if int(pred_unit) in pred_to_gt:
            continue
        p_idx = pred_unit_to_idx[int(pred_unit)]
        micro_fp += int(pred_counts[p_idx])

    derived = _metrics_from_counts(float(micro_tp), float(micro_fp), float(micro_fn))
    return {
        "sort_tp_micro_mapped": int(micro_tp),
        "sort_fp_micro_mapped": int(micro_fp),
        "sort_fn_micro_mapped": int(micro_fn),
        "sort_precision_micro_mapped": float(derived["precision"]),
        "sort_recall_micro_mapped": float(derived["recall"]),
        "sort_f1_micro_mapped": float(derived["f1"]),
        "sort_fp_rate_micro_mapped": float(derived["fp_rate"]),
        "sort_fn_rate_micro_mapped": float(derived["fn_rate"]),
        "sorted_score": float(derived["sorted_score"]),
        "sort_f1_macro_mapped": float(np.mean(np.asarray(macro_scores, dtype=np.float64))) if macro_scores else 0.0,
        "unit_recovery_rate_at_0.8": _safe_ratio(
            float(recovered),
            float(len(macro_scores)),
            default=0.0,
        ),
    }


def _pair_metrics_for_unit(
    pred_counts: np.ndarray,
    gt_counts: np.ndarray,
    tp: np.ndarray,
    pair_f1: np.ndarray,
    *,
    pred_idx: int,
    gt_idx: int,
) -> Dict[str, float | int]:
    unit_tp = int(tp[pred_idx, gt_idx])
    unit_fp = int(pred_counts[pred_idx] - unit_tp)
    unit_fn = int(gt_counts[gt_idx] - unit_tp)
    derived = _metrics_from_counts(float(unit_tp), float(unit_fp), float(unit_fn))
    return {
        "tp": int(unit_tp),
        "fp": int(unit_fp),
        "fn": int(unit_fn),
        "pair_f1": float(pair_f1[pred_idx, gt_idx]),
        "precision": float(derived["precision"]),
        "recall": float(derived["recall"]),
        "fp_rate": float(derived["fp_rate"]),
        "fn_rate": float(derived["fn_rate"]),
        "sorted_score": float(derived["sorted_score"]),
    }


def _per_unit_report(overlap: Dict[str, Any]) -> List[Dict[str, Any]]:
    pred_units = overlap["pred_units"]
    gt_units = overlap["gt_units"]
    pred_counts = overlap["pred_counts"]
    gt_counts = overlap["gt_counts"]
    tp = overlap["tp"]
    pair_f1 = overlap["pair_f1"]
    pred_unit_to_idx = overlap["pred_unit_to_idx"]
    rows: List[Dict[str, Any]] = []

    for g_idx, gt_unit in enumerate(gt_units.tolist()):
        col_tp = tp[:, g_idx] if tp.size > 0 else np.zeros((0,), dtype=np.int64)
        candidate_top20_size = 0
        selection_method = "top20_pair_f1_then_sorted_score"
        selection_pair_f1 = 0.0
        selection_sorted_score = 0.0

        if col_tp.size > 0 and int(np.max(col_tp)) > 0:
            pair_scores = pair_f1[:, g_idx]
            positive_candidates = np.flatnonzero(col_tp > 0).tolist()
            ranked_candidates = sorted(
                positive_candidates,
                key=lambda idx: (float(pair_scores[idx]), int(col_tp[idx]), -int(pred_counts[idx])),
                reverse=True,
            )
            top_candidates = ranked_candidates[:20]
            candidate_top20_size = int(len(top_candidates))

            best_idx = int(
                max(
                    top_candidates,
                    key=lambda idx: (
                        float(
                            _pair_metrics_for_unit(
                                pred_counts,
                                gt_counts,
                                tp,
                                pair_f1,
                                pred_idx=int(idx),
                                gt_idx=int(g_idx),
                            )["sorted_score"]
                        ),
                        float(pair_scores[idx]),
                        int(col_tp[idx]),
                        -int(pred_counts[idx]),
                    ),
                )
            )
            best_pred = int(pred_units[best_idx])
            pair_metrics = _pair_metrics_for_unit(
                pred_counts,
                gt_counts,
                tp,
                pair_f1,
                pred_idx=best_idx,
                gt_idx=g_idx,
            )
            unit_tp = int(pair_metrics["tp"])
            unit_fp = int(pair_metrics["fp"])
            unit_fn = int(pair_metrics["fn"])
            unit_f1 = float(pair_metrics["pair_f1"])
            selection_pair_f1 = float(pair_metrics["pair_f1"])
            selection_sorted_score = float(pair_metrics["sorted_score"])
            n_pred_fragments = int(np.sum(col_tp > 0))
            row_tp = tp[best_idx, :]
            n_gt_merge_partners = int(np.sum(row_tp > 0))
        else:
            best_pred = -1
            unit_tp = 0
            unit_fp = 0
            unit_fn = int(gt_counts[g_idx])
            unit_f1 = 0.0
            n_pred_fragments = 0
            n_gt_merge_partners = 0
        derived = _metrics_from_counts(float(unit_tp), float(unit_fp), float(unit_fn))
        rows.append(
            {
                "gt_unit": int(gt_unit),
                "tp": int(unit_tp),
                "fp": int(unit_fp),
                "fn": int(unit_fn),
                "precision": float(derived["precision"]),
                "recall": float(derived["recall"]),
                "f1": float(unit_f1),
                "fp_rate": float(derived["fp_rate"]),
                "fn_rate": float(derived["fn_rate"]),
                "sorted_score": float(derived["sorted_score"]),
                "selection_pair_f1": float(selection_pair_f1),
                "selection_sorted_score": float(selection_sorted_score),
                "candidate_top20_size": int(candidate_top20_size),
                "selection_method": str(selection_method),
                "best_pred_unit": int(best_pred),
                "n_pred_fragments": int(n_pred_fragments),
                "n_gt_merge_partners": int(n_gt_merge_partners),
            }
        )
    return rows


def _window_edges(all_times: np.ndarray, window_samples: int) -> List[Tuple[int, int]]:
    if all_times.size == 0 or window_samples <= 0:
        return []
    start = int(np.min(all_times))
    stop = int(np.max(all_times)) + 1
    if stop <= start:
        stop = start + 1
    return [
        (int(t0), int(min(t0 + window_samples, stop)))
        for t0 in range(start, stop, window_samples)
    ]


def _window_overlap(
    pred: EventSet,
    gt: EventSet,
    matches: Sequence[EventMatch],
    *,
    min_sort_label: int,
    start: int,
    stop: int,
    pred_unit_values: np.ndarray,
    gt_unit_values: np.ndarray,
) -> Dict[str, Any]:
    pred_mask = (pred.times_samples >= int(start)) & (pred.times_samples < int(stop))
    gt_mask = (gt.times_samples >= int(start)) & (gt.times_samples < int(stop))
    pred_subset = pred.subset(pred_mask)
    gt_subset = gt.subset(gt_mask)
    pred_index_lookup = {int(idx): new_idx for new_idx, idx in enumerate(np.flatnonzero(pred_mask).tolist())}
    gt_index_lookup = {int(idx): new_idx for new_idx, idx in enumerate(np.flatnonzero(gt_mask).tolist())}
    win_matches: List[EventMatch] = []
    for match in matches:
        if match.pred_index not in pred_index_lookup or match.gt_index not in gt_index_lookup:
            continue
        win_matches.append(
            EventMatch(
                pred_index=int(pred_index_lookup[match.pred_index]),
                gt_index=int(gt_index_lookup[match.gt_index]),
                dt_samples=int(match.dt_samples),
                channel_delta=match.channel_delta,
            )
        )
    return _build_unit_overlap(
        pred_subset,
        gt_subset,
        win_matches,
        min_sort_label=min_sort_label,
        pred_unit_values=np.asarray(pred_unit_values, dtype=np.int64),
        gt_unit_values=np.asarray(gt_unit_values, dtype=np.int64),
    )


def benchmark_with_gt(
    pred: EventSet,
    gt: EventSet,
    *,
    fs: float,
    cfg: BenchmarkConfig | None = None,
    h5_path: str | Path | None = None,
    channel_positions: np.ndarray | None = None,
    quality_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = cfg or BenchmarkConfig()
    pred_sorted = pred.sorted_copy()
    gt_sorted = gt.sorted_copy()
    tol_samples = max(0, int(round(float(cfg.tol_ms) * float(fs) / 1000.0)))
    match_mode = _resolve_event_match_mode(pred_sorted, gt_sorted, cfg)
    matches = _match_events(
        pred_sorted,
        gt_sorted,
        tol_samples=tol_samples,
        match_mode=match_mode,
        channel_radius_ch=int(cfg.channel_radius_ch),
    )

    tp_detect = int(len(matches))
    fp_detect = int(pred_sorted.n_events - tp_detect)
    fn_detect = int(gt_sorted.n_events - tp_detect)
    detect_metrics_summary = _metrics_from_counts(float(tp_detect), float(fp_detect), float(fn_detect))

    overlap = _build_unit_overlap(
        pred_sorted,
        gt_sorted,
        matches,
        min_sort_label=int(cfg.min_sort_label),
    )
    sort_summary = _mapped_sort_summary(
        overlap,
        recovery_f1_threshold=float(cfg.recovery_f1_threshold),
    )
    summary = {
        "detect_tp": int(tp_detect),
        "detect_fp": int(fp_detect),
        "detect_fn": int(fn_detect),
        "detect_precision": float(detect_metrics_summary["precision"]),
        "detect_recall": float(detect_metrics_summary["recall"]),
        "detect_f1": float(detect_metrics_summary["f1"]),
        "detect_fp_rate": float(detect_metrics_summary["fp_rate"]),
        "detect_fn_rate": float(detect_metrics_summary["fn_rate"]),
        "detect_sorted_score": float(detect_metrics_summary["sorted_score"]),
        "sort_tp_micro_mapped": int(sort_summary["sort_tp_micro_mapped"]),
        "sort_fp_micro_mapped": int(sort_summary["sort_fp_micro_mapped"]),
        "sort_fn_micro_mapped": int(sort_summary["sort_fn_micro_mapped"]),
        "sort_precision_micro_mapped": float(sort_summary["sort_precision_micro_mapped"]),
        "sort_recall_micro_mapped": float(sort_summary["sort_recall_micro_mapped"]),
        "sort_f1_micro_mapped": float(sort_summary["sort_f1_micro_mapped"]),
        "sort_fp_rate_micro_mapped": float(sort_summary["sort_fp_rate_micro_mapped"]),
        "sort_fn_rate_micro_mapped": float(sort_summary["sort_fn_rate_micro_mapped"]),
        "sorted_score": float(sort_summary["sorted_score"]),
        "sort_f1_macro_mapped": float(sort_summary["sort_f1_macro_mapped"]),
        "unit_recovery_rate_at_0.8": float(sort_summary["unit_recovery_rate_at_0.8"]),
    }

    all_times = np.concatenate(
        [
            pred_sorted.times_samples.astype(np.int64, copy=False),
            gt_sorted.times_samples.astype(np.int64, copy=False),
        ],
        axis=0,
    ) if (pred_sorted.n_events + gt_sorted.n_events) > 0 else np.zeros((0,), dtype=np.int64)
    window_samples = max(0, int(round(float(cfg.window_sec) * float(fs))))
    windows: List[Dict[str, Any]] = []
    for start, stop in _window_edges(all_times, window_samples):
        pred_mask = (pred_sorted.times_samples >= int(start)) & (pred_sorted.times_samples < int(stop))
        gt_mask = (gt_sorted.times_samples >= int(start)) & (gt_sorted.times_samples < int(stop))
        tp_window = 0
        for match in matches:
            gt_time = int(gt_sorted.times_samples[match.gt_index])
            if int(start) <= gt_time < int(stop):
                tp_window += 1
        fp_window = int(np.sum(pred_mask)) - int(tp_window)
        fn_window = int(np.sum(gt_mask)) - int(tp_window)
        win_overlap = _window_overlap(
            pred_sorted,
            gt_sorted,
            matches,
            min_sort_label=int(cfg.min_sort_label),
            start=int(start),
            stop=int(stop),
            pred_unit_values=overlap["pred_units"],
            gt_unit_values=overlap["gt_units"],
        )
        win_sort = _mapped_sort_summary(
            win_overlap,
            recovery_f1_threshold=float(cfg.recovery_f1_threshold),
            mapping_overlap=overlap,
        )
        win_detect = _metrics_from_counts(float(tp_window), float(max(0, fp_window)), float(max(0, fn_window)))
        windows.append(
            {
                "start_sample": int(start),
                "stop_sample": int(stop),
                "start_sec": float(start / max(float(fs), 1e-12)),
                "stop_sec": float(stop / max(float(fs), 1e-12)),
                "detect_tp": int(tp_window),
                "detect_fp": int(max(0, fp_window)),
                "detect_fn": int(max(0, fn_window)),
                "detect_precision": float(win_detect["precision"]),
                "detect_recall": float(win_detect["recall"]),
                "detect_f1": float(win_detect["f1"]),
                "detect_fp_rate": float(win_detect["fp_rate"]),
                "detect_fn_rate": float(win_detect["fn_rate"]),
                "detect_sorted_score": float(win_detect["sorted_score"]),
                "sort_precision_micro_mapped": float(win_sort["sort_precision_micro_mapped"]),
                "sort_recall_micro_mapped": float(win_sort["sort_recall_micro_mapped"]),
                "sort_f1_micro_mapped": float(win_sort["sort_f1_micro_mapped"]),
                "sort_fp_rate_micro_mapped": float(win_sort["sort_fp_rate_micro_mapped"]),
                "sort_fn_rate_micro_mapped": float(win_sort["sort_fn_rate_micro_mapped"]),
                "sorted_score": float(win_sort["sorted_score"]),
                "sort_f1_macro_mapped": float(win_sort["sort_f1_macro_mapped"]),
                "unit_recovery_rate_at_0.8": float(win_sort["unit_recovery_rate_at_0.8"]),
            }
        )

    no_gt_report: Dict[str, Any] | None = None
    diagnostics: Dict[str, Any] = {}
    if bool(cfg.diagnostics):
        matched_pred_idx = np.array([m.pred_index for m in matches], dtype=np.int64)
        matched_gt_idx = np.array([m.gt_index for m in matches], dtype=np.int64)
        sort_valid = (
            (matched_pred_idx.size > 0)
            and np.any(pred_sorted.labels[matched_pred_idx] >= int(cfg.min_sort_label))
        )
        ari = float("nan")
        if sort_valid and adjusted_rand_score is not None:
            keep = pred_sorted.labels[matched_pred_idx] >= int(cfg.min_sort_label)
            ari = float(
                adjusted_rand_score(
                    gt_sorted.labels[matched_gt_idx][keep],
                    pred_sorted.labels[matched_pred_idx][keep],
                )
            )
        pair_rows = []
        pred_units = overlap["pred_units"]
        gt_units = overlap["gt_units"]
        tp_mat = overlap["tp"]
        pair_f1 = overlap["pair_f1"]
        for pred_unit, gt_unit in overlap["pred_to_gt"].items():
            p_idx = overlap["pred_unit_to_idx"][int(pred_unit)]
            g_idx = overlap["gt_unit_to_idx"][int(gt_unit)]
            pair_rows.append(
                {
                    "pred_unit": int(pred_unit),
                    "gt_unit": int(gt_unit),
                    "tp": int(tp_mat[p_idx, g_idx]),
                    "pair_f1": float(pair_f1[p_idx, g_idx]),
                }
            )
        diagnostics = {
            "tol_samples": int(tol_samples),
            "event_match_mode": str(match_mode),
            "n_pred_events": int(pred_sorted.n_events),
            "n_gt_events": int(gt_sorted.n_events),
            "n_matched_events": int(len(matches)),
            "n_pred_units_sortable": int(pred_units.size),
            "n_gt_units": int(gt_units.size),
            "ari": float(ari),
            "mapped_pairs": pair_rows,
            "unmapped_pred_units": [
                int(unit)
                for unit in pred_units.tolist()
                if int(unit) not in overlap["pred_to_gt"]
            ],
            "unmapped_gt_units": [
                int(unit)
                for unit in gt_units.tolist()
                if int(unit) not in overlap["gt_to_pred"]
            ],
        }

    if h5_path is not None and pred_sorted.channels is not None:
        no_gt_quality_kwargs = dict(quality_kwargs or {})
        if channel_positions is not None and "channel_positions" not in no_gt_quality_kwargs:
            no_gt_quality_kwargs["channel_positions"] = channel_positions
        try:
            no_gt_report = benchmark_no_gt(
                pred_sorted,
                h5_path=h5_path,
                fs=float(fs),
                cfg=cfg,
                quality_kwargs=no_gt_quality_kwargs,
            )
            no_gt_summary = dict(no_gt_report.get("summary", {}))
            summary.update(
                {
                    "no_gt_n_events_total": int(no_gt_summary.get("n_events_total", 0)),
                    "no_gt_n_events_kept": int(no_gt_summary.get("n_events_kept", 0)),
                    "no_gt_n_units_kept": int(no_gt_summary.get("n_units_kept", 0)),
                    "no_gt_n_units_evaluated": int(no_gt_summary.get("n_units_evaluated", 0)),
                    "no_gt_DBI": float(no_gt_summary.get("DBI", float("nan"))),
                    "no_gt_topological_similarity": float(no_gt_summary.get("topological_similarity", float("nan"))),
                    "no_gt_SNR_mean": float(no_gt_summary.get("SNR_mean", float("nan"))),
                    "no_gt_FP_est_mean": float(no_gt_summary.get("FP_est_mean", float("nan"))),
                    "no_gt_missed_spikes_FN_mean": float(no_gt_summary.get("missed_spikes_FN_mean", float("nan"))),
                    "no_gt_sorted_score_mean": float(no_gt_summary.get("sorted_score_mean", float("nan"))),
                    "no_gt_contamination_score_mean": float(no_gt_summary.get("contamination_score_mean", float("nan"))),
                    "no_gt_RPV_rate_mean": float(no_gt_summary.get("RPV_rate_mean", float("nan"))),
                }
            )
        except Exception as exc:
            if bool(cfg.diagnostics):
                diagnostics["no_gt_error"] = str(exc)

    return {
        "summary": summary,
        "windows": windows,
        "per_unit": _per_unit_report(overlap),
        "diagnostics": diagnostics,
        "no_gt": no_gt_report,
    }


def benchmark_no_gt(
    pred: EventSet,
    *,
    h5_path: str | Path,
    fs: float,
    cfg: BenchmarkConfig | None = None,
    quality_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = cfg or BenchmarkConfig()
    if pred.channels is None:
        raise ValueError("benchmark_no_gt requires pred.channels")
    raw = evaluate_no_gt_quality(
        h5_path=h5_path,
        pred_times=pred.times_samples,
        pred_channels=pred.channels,
        pred_labels=pred.labels,
        fs=fs,
        min_label=int(cfg.min_sort_label),
        **dict(quality_kwargs or {}),
    )
    summary = {
        "n_events_total": int(raw.get("n_events_total", 0)),
        "n_events_kept": int(raw.get("n_events_kept", 0)),
        "n_units_kept": int(raw.get("n_units_kept", 0)),
        "n_units_evaluated": int(raw.get("n_units_evaluated", 0)),
        "DBI": float(raw.get("DBI", float("nan"))),
        "topological_similarity": float(raw.get("topological_similarity", float("nan"))),
        "SNR_mean": float(raw.get("SNR", {}).get("mean", float("nan"))),
        "FP_est_mean": float(raw.get("FP_est", {}).get("mean", float("nan"))),
        "missed_spikes_FN_mean": float(raw.get("missed_spikes_FN", {}).get("mean", float("nan"))),
        "sorted_score_mean": float(raw.get("sorted_score", {}).get("mean", float("nan"))),
        "contamination_score_mean": float(raw.get("contamination_score", {}).get("mean", float("nan"))),
        "RPV_rate_mean": float(raw.get("RPV_rate", {}).get("mean", float("nan"))),
    }
    diagnostics = raw if bool(cfg.diagnostics) else {}
    return {
        "summary": summary,
        "unit_metrics": list(raw.get("unit_metrics", [])),
        "diagnostics": diagnostics,
    }


def _agg(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _mad_std(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _infer_channel_axis(signal_shape: Tuple[int, int], n_channels: int) -> int:
    n0, n1 = int(signal_shape[0]), int(signal_shape[1])
    if n1 == n_channels and n0 != n_channels:
        return 1
    if n0 == n_channels and n1 != n_channels:
        return 0
    if n1 >= n_channels and n0 > n1:
        return 1
    if n0 >= n_channels and n1 > n0:
        return 0
    return 1


def _signal_n_samples(signal: h5py.Dataset, channel_axis: int) -> int:
    return int(signal.shape[0]) if channel_axis == 1 else int(signal.shape[1])


def _read_trace_subsample(signal: h5py.Dataset, channel_axis: int, channel: int, step: int) -> np.ndarray:
    if channel_axis == 1:
        return np.asarray(signal[::step, int(channel)], dtype=np.float32)
    return np.asarray(signal[int(channel), ::step], dtype=np.float32)


def _read_snip(signal: h5py.Dataset, channel_axis: int, channel: int, lo: int, hi: int) -> np.ndarray:
    if channel_axis == 1:
        return np.asarray(signal[int(lo) : int(hi), int(channel)], dtype=np.float32)
    return np.asarray(signal[int(channel), int(lo) : int(hi)], dtype=np.float32)


def _estimate_noise_std_subsampled(
    signal: h5py.Dataset,
    channel_axis: int,
    channel: int,
    max_points: int,
) -> float:
    n_t = _signal_n_samples(signal, channel_axis)
    if n_t <= 0:
        return 0.0
    step = max(1, n_t // max(1, int(max_points)))
    trace_sub = _read_trace_subsample(signal, channel_axis, channel=channel, step=step)
    return _mad_std(trace_sub)


def _extract_wave_features(
    signal: h5py.Dataset,
    channel_axis: int,
    times: np.ndarray,
    channels: np.ndarray,
    picked_idx: np.ndarray,
    half_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    signed_amp: List[float] = []
    n_t = _signal_n_samples(signal, channel_axis)
    for idx in picked_idx.tolist():
        t = int(times[idx])
        ch = int(channels[idx])
        lo = max(0, t - int(half_window))
        hi = min(n_t, t + int(half_window) + 1)
        if hi - lo < 3:
            continue
        snip = _read_snip(signal, channel_axis, channel=ch, lo=lo, hi=hi)
        if snip.size < 3:
            continue
        trough_i = int(np.argmin(snip))
        peak_i = trough_i + int(np.argmax(snip[trough_i:]))
        width = float(max(1, peak_i - trough_i))
        amp_neg = float(np.min(snip))
        amp_abs = float(np.max(np.abs(snip)))
        p2p = float(np.max(snip) - np.min(snip))
        energy = float(np.mean(snip * snip))
        feats.append(np.array([amp_neg, amp_abs, p2p, energy, width], dtype=np.float32))
        signed_amp.append(amp_neg)
    if not feats:
        return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(feats, axis=0), np.asarray(signed_amp, dtype=np.float32)


def _compute_dbi(features: np.ndarray, labels: np.ndarray) -> float:
    if features.shape[0] < 2:
        return float("nan")
    units = np.unique(labels)
    if units.size < 2:
        return float("nan")
    centers = {}
    dispersions = {}
    for u in units.tolist():
        x = features[labels == int(u)]
        if x.shape[0] == 0:
            continue
        mu = np.mean(x, axis=0)
        centers[int(u)] = mu
        dispersions[int(u)] = float(np.mean(np.linalg.norm(x - mu, axis=1)))
    valid_units = sorted(centers.keys())
    if len(valid_units) < 2:
        return float("nan")
    r_vals = []
    for i, ui in enumerate(valid_units):
        mu_i = centers[ui]
        si = dispersions[ui]
        max_r = -np.inf
        for j, uj in enumerate(valid_units):
            if i == j:
                continue
            mu_j = centers[uj]
            sj = dispersions[uj]
            dij = float(np.linalg.norm(mu_i - mu_j))
            if dij <= 1e-12:
                continue
            rij = (si + sj) / dij
            if rij > max_r:
                max_r = rij
        if np.isfinite(max_r):
            r_vals.append(float(max_r))
    if not r_vals:
        return float("nan")
    return float(np.mean(np.asarray(r_vals, dtype=np.float64)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    vals = x[order]
    i = 0
    while i < vals.size:
        j = i + 1
        while j < vals.size and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denom = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(xc * yc) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return float("nan")
    if spearmanr is not None:
        r, _ = spearmanr(x, y)
        return float(r)
    return _pearsonr(_rankdata(x), _rankdata(y))


def _normal_cdf(z: float) -> float:
    if norm is not None:
        return float(norm.cdf(float(z)))
    return float(0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0))))


def evaluate_no_gt_quality(
    *,
    h5_path: str | Path,
    pred_times: np.ndarray,
    pred_channels: np.ndarray,
    pred_labels: np.ndarray,
    fs: float,
    channel_positions: np.ndarray | None = None,
    min_label: int = 0,
    refractory_ms: float = 1.0,
    detect_threshold_std: float = 6.5,
    half_window: int = 20,
    max_spikes_per_unit: int = 400,
    min_spikes_per_unit: int = 40,
    max_units: int = 256,
    rng_seed: int = 123,
    noise_std_max_points: int = 80000,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(rng_seed))
    times = np.asarray(pred_times, dtype=np.int64)
    channels = np.asarray(pred_channels, dtype=np.int32)
    labels = np.asarray(pred_labels, dtype=np.int64)
    if not (times.shape == channels.shape == labels.shape):
        raise ValueError(f"shape mismatch: time={times.shape}, channel={channels.shape}, label={labels.shape}")

    keep = labels >= int(min_label)
    times_k = times[keep]
    ch_k = channels[keep]
    lab_k = labels[keep]
    if times_k.size == 0:
        return {
            "n_events_total": int(times.size),
            "n_events_kept": int(times_k.size),
            "n_units_kept": 0,
            "n_units_evaluated": 0,
            "duration_sec": float("nan"),
            "DBI": float("nan"),
            "topological_similarity": float("nan"),
            "RPV_rate": _agg([]),
            "FP_est": _agg([]),
            "missed_spikes_FN": _agg([]),
            "SNR": _agg([]),
            "sorted_score": _agg([]),
            "contamination_score": _agg([]),
            "unit_metrics": [],
        }

    n_channels = int(np.max(ch_k)) + 1
    if channel_positions is not None and int(np.asarray(channel_positions).shape[0]) >= n_channels:
        pos_raw = np.asarray(channel_positions, dtype=np.float32)[:n_channels]
        if pos_raw.ndim != 2 or pos_raw.shape[1] < 2:
            pos2d = np.stack([np.arange(n_channels, dtype=np.float32), np.zeros((n_channels,), dtype=np.float32)], axis=1)
        else:
            pos2d = pos_raw[:, :2].astype(np.float32, copy=False)
    else:
        pos2d = np.stack([np.arange(n_channels, dtype=np.float32), np.zeros((n_channels,), dtype=np.float32)], axis=1)

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        signal_name = "recordings" if "recordings" in f else ("preprocess_data" if "preprocess_data" in f else None)
        if signal_name is None:
            raise KeyError("input h5 missing 'recordings' or 'preprocess_data'")
        signal = f[signal_name]
        if signal.ndim != 2:
            raise ValueError(f"{signal_name} must be 2D, got shape={signal.shape}")
        ch_axis = _infer_channel_axis((int(signal.shape[0]), int(signal.shape[1])), n_channels=n_channels)
        if ch_axis == 1 and int(signal.shape[1]) < n_channels:
            raise ValueError(f"{signal_name} has {signal.shape[1]} channels but predictions use {n_channels}")
        if ch_axis == 0 and int(signal.shape[0]) < n_channels:
            raise ValueError(f"{signal_name} has {signal.shape[0]} channels but predictions use {n_channels}")
        duration_sec = float(_signal_n_samples(signal, ch_axis) / max(float(fs), 1e-12))

        channel_noise_std = np.zeros((n_channels,), dtype=np.float64)
        for ch in np.unique(ch_k).tolist():
            channel_noise_std[int(ch)] = _estimate_noise_std_subsampled(
                signal,
                channel_axis=ch_axis,
                channel=int(ch),
                max_points=int(noise_std_max_points),
            )

        units_all = np.unique(lab_k)
        if units_all.size > int(max_units) > 0:
            counts = np.array([np.sum(lab_k == int(u)) for u in units_all], dtype=np.int64)
            order = np.argsort(-counts, kind="mergesort")
            units = units_all[order[: int(max_units)]]
        else:
            units = units_all

        unit_records: List[Dict[str, object]] = []
        all_features: List[np.ndarray] = []
        all_feat_labels: List[np.ndarray] = []
        feature_centers: List[np.ndarray] = []
        space_centers: List[np.ndarray] = []
        refractory_samples = max(1, int(round(float(fs) * float(refractory_ms) / 1000.0)))

        for u in units.tolist():
            idx = np.flatnonzero(lab_k == int(u))
            n_spikes = int(idx.size)
            if n_spikes < int(min_spikes_per_unit):
                continue
            if n_spikes > int(max_spikes_per_unit):
                picked = rng.choice(idx, size=int(max_spikes_per_unit), replace=False)
            else:
                picked = idx

            feat, amp_signed = _extract_wave_features(
                signal,
                channel_axis=ch_axis,
                times=times_k,
                channels=ch_k,
                picked_idx=np.asarray(picked, dtype=np.int64),
                half_window=int(half_window),
            )
            if feat.shape[0] == 0:
                continue

            all_features.append(feat)
            all_feat_labels.append(np.full((feat.shape[0],), int(u), dtype=np.int64))

            main_ch_vals, main_ch_counts = np.unique(ch_k[idx], return_counts=True)
            main_ch = int(main_ch_vals[int(np.argmax(main_ch_counts))])
            template_peak = float(np.max(np.abs(np.mean(feat[:, :3], axis=0))))
            noise_std = float(channel_noise_std[main_ch])
            snr = float(template_peak / max(1e-12, noise_std))

            t_unit = np.sort(times_k[idx], kind="mergesort")
            if t_unit.size > 1:
                isi = np.diff(t_unit)
                n_v = int(np.sum(isi <= refractory_samples))
                rpv_rate = float(n_v / max(1, t_unit.size - 1))
            else:
                n_v = 0
                rpv_rate = 0.0

            mfr = float(n_spikes / max(1e-12, duration_sec))
            contamination = float(1.0 - (n_v / max(1e-12, n_spikes * mfr * (float(refractory_ms) / 1000.0))))
            contamination = float(np.clip(contamination, 0.0, 1.0))
            fp_est = float(np.clip(1.0 - contamination, 0.0, 1.0))

            mu = float(np.mean(amp_signed))
            sd = float(np.std(amp_signed))
            thr = float(-abs(float(detect_threshold_std)) * noise_std)
            if sd <= 1e-12:
                fn_est = 0.0
            else:
                z = (thr - mu) / sd
                det_prob = _normal_cdf(z)
                fn_est = float(np.clip(1.0 - det_prob, 0.0, 1.0))

            sorted_score = float(np.clip(1.0 - fp_est - fn_est, 0.0, 1.0))

            feat_center = np.mean(feat, axis=0)
            feature_centers.append(feat_center)
            space_centers.append(pos2d[main_ch])

            unit_records.append(
                {
                    "unit": int(u),
                    "n_spikes": int(n_spikes),
                    "main_channel": int(main_ch),
                    "RPV_count": int(n_v),
                    "RPV_rate": float(rpv_rate),
                    "mean_firing_rate_hz": float(mfr),
                    "SNR": float(snr),
                    "missed_spikes_FN": float(fn_est),
                    "FP_est": float(fp_est),
                    "sorted_score": float(sorted_score),
                    "contamination_score": float(contamination),
                }
            )

    if all_features:
        feat_all = np.concatenate(all_features, axis=0)
        lab_all = np.concatenate(all_feat_labels, axis=0)
        dbi = _compute_dbi(feat_all, lab_all)
    else:
        dbi = float("nan")

    if len(feature_centers) >= 2:
        f_cent = np.asarray(feature_centers, dtype=np.float64)
        s_cent = np.asarray(space_centers, dtype=np.float64)
        d_feat = []
        d_space = []
        for i in range(f_cent.shape[0]):
            for j in range(i + 1, f_cent.shape[0]):
                d_feat.append(float(np.linalg.norm(f_cent[i] - f_cent[j])))
                d_space.append(float(np.linalg.norm(s_cent[i] - s_cent[j])))
        topo_sim = _spearman_corr(np.asarray(d_feat, dtype=np.float64), np.asarray(d_space, dtype=np.float64))
    else:
        topo_sim = float("nan")

    return {
        "n_events_total": int(times.size),
        "n_events_kept": int(times_k.size),
        "n_units_kept": int(np.unique(lab_k).size),
        "n_units_evaluated": int(len(unit_records)),
        "duration_sec": float(duration_sec),
        "DBI": float(dbi),
        "topological_similarity": float(topo_sim),
        "RPV_rate": _agg([float(r["RPV_rate"]) for r in unit_records]),
        "FP_est": _agg([float(r["FP_est"]) for r in unit_records]),
        "missed_spikes_FN": _agg([float(r["missed_spikes_FN"]) for r in unit_records]),
        "SNR": _agg([float(r["SNR"]) for r in unit_records]),
        "sorted_score": _agg([float(r["sorted_score"]) for r in unit_records]),
        "contamination_score": _agg([float(r["contamination_score"]) for r in unit_records]),
        "unit_metrics": unit_records,
    }


def dump_debug_events(path: str | Path, patches: np.ndarray, z: np.ndarray, max_items: int = 64) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(max_items, patches.shape[0])
    np.savez_compressed(path, patches=patches[:n], embedding=z[:n])


def plot_embedding_2d(path: str | Path, z2d: np.ndarray, labels: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    if labels is None:
        plt.scatter(z2d[:, 0], z2d[:, 1], s=8)
    else:
        plt.scatter(z2d[:, 0], z2d[:, 1], c=labels, s=8, cmap="tab20")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --- Kilosort-style 1v1 benchmark implementation ---
@njit
def nmatch(ss0: np.ndarray, ss: np.ndarray, dt: int = 6):
    i = 0
    j = 0
    n0 = 0

    ntmax = len(ss0)
    is_matched = np.zeros(len(ss), dtype=np.bool_)
    is_matched0 = np.zeros(len(ss0), dtype=np.bool_)

    while i < len(ss):
        while j + 1 < ntmax and ss0[j + 1] <= ss[i] + dt:
            j += 1

        if np.abs(ss0[j] - ss[i]) <= dt:
            n0 += 1
            is_matched[i] = True
            is_matched0[j] = True

        i += 1
    return n0, is_matched, is_matched0


def _resolve_tol_samples(fs: float, cfg: BenchmarkConfig) -> int:
    if cfg.tol_samples is not None:
        return max(0, int(cfg.tol_samples))
    if cfg.tol_ms is None:
        return 6
    return max(0, int(round(float(cfg.tol_ms) * float(fs) / 1000.0)))


def _classification_acc_true_positive(
    pred_sorted: EventSet,
    gt_sorted: EventSet,
    *,
    tol_samples: int,
    min_sort_label: int,
    pred_to_gt_1to1: Dict[int, int],
    gt_units_keep: set[int] | None = None,
) -> Dict[str, float | int]:
    # Measure TP classification on the same temporal matches using both the
    # 1-to-1 remapped unit ids and the legacy raw-label equality fallback.
    empty_result: Dict[str, float | int] = {
        "classification_acc_true_positive": float("nan"),
        "raw_label_match_acc_true_positive": float("nan"),
        "n_true_positive_events": 0,
        "n_true_positive_events_correct": 0,
        "n_true_positive_events_correct_raw_label_match": 0,
        "n_true_positive_events_unmapped_pred_unit": 0,
    }
    matches = _match_events(
        pred_sorted,
        gt_sorted,
        tol_samples=int(tol_samples),
        match_mode="time",
        channel_radius_ch=0,
    )
    if not matches:
        return dict(empty_result)

    matched_pred_idx = np.asarray([int(match.pred_index) for match in matches], dtype=np.int64)
    matched_gt_idx = np.asarray([int(match.gt_index) for match in matches], dtype=np.int64)
    keep = pred_sorted.labels[matched_pred_idx] >= int(min_sort_label)
    if not np.any(keep):
        return dict(empty_result)

    y_pred = pred_sorted.labels[matched_pred_idx][keep].astype(np.int64, copy=False)
    y_true = gt_sorted.labels[matched_gt_idx][keep].astype(np.int64, copy=False)
    if gt_units_keep is not None:
        unit_keep = np.asarray([int(unit) in gt_units_keep for unit in y_true.tolist()], dtype=np.bool_)
        y_pred = y_pred[unit_keep]
        y_true = y_true[unit_keep]
        if y_true.size == 0:
            return dict(empty_result)
    n_total = int(y_true.size)
    n_correct_raw = int(np.sum(y_true == y_pred))
    mapped_correct = np.zeros((n_total,), dtype=np.bool_)
    n_unmapped = 0
    for idx, pred_unit in enumerate(y_pred.tolist()):
        mapped_gt = pred_to_gt_1to1.get(int(pred_unit))
        if mapped_gt is None:
            n_unmapped += 1
            continue
        mapped_correct[idx] = int(mapped_gt) == int(y_true[idx])
    n_correct = int(np.sum(mapped_correct))
    return {
        "classification_acc_true_positive": float(n_correct / max(1, n_total)),
        "raw_label_match_acc_true_positive": float(n_correct_raw / max(1, n_total)),
        "n_true_positive_events": int(n_total),
        "n_true_positive_events_correct": int(n_correct),
        "n_true_positive_events_correct_raw_label_match": int(n_correct_raw),
        "n_true_positive_events_unmapped_pred_unit": int(n_unmapped),
    }


def _one_to_one_micro_counts(
    *,
    gt_units: np.ndarray,
    pred_units: np.ndarray,
    gt_to_pred: Dict[int, int],
    pred_to_gt: Dict[int, int],
    gt_to_idx: Dict[int, int],
    pred_to_idx: Dict[int, int],
    pair_n_match: np.ndarray,
    pred_counts: np.ndarray,
    n_gt_spikes: np.ndarray,
    gt_units_keep: set[int] | None = None,
) -> Dict[str, float | int]:
    tp = 0
    fp = 0
    fn = 0
    keep_all = gt_units_keep is None
    for gt_unit in gt_units.tolist():
        if (not keep_all) and int(gt_unit) not in gt_units_keep:
            continue
        g_idx = gt_to_idx[int(gt_unit)]
        pred_unit = gt_to_pred.get(int(gt_unit))
        if pred_unit is None:
            fn += int(n_gt_spikes[g_idx])
            continue
        p_idx = pred_to_idx.get(int(pred_unit))
        if p_idx is None:
            fn += int(n_gt_spikes[g_idx])
            continue
        n_tp = int(pair_n_match[g_idx, p_idx])
        tp += int(n_tp)
        fp += int(max(0, int(pred_counts[p_idx]) - n_tp))
        fn += int(max(0, int(n_gt_spikes[g_idx]) - n_tp))
    for pred_unit in pred_units.tolist():
        if int(pred_unit) in pred_to_gt:
            continue
        if not keep_all:
            continue
        p_idx = pred_to_idx[int(pred_unit)]
        fp += int(pred_counts[p_idx])
    derived = _metrics_from_counts(float(tp), float(fp), float(fn))
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(derived["precision"]),
        "recall": float(derived["recall"]),
        "f1": float(derived["f1"]),
        "fp_rate": float(derived["fp_rate"]),
        "fn_rate": float(derived["fn_rate"]),
        "sorted_score": float(derived["sorted_score"]),
    }


def _unit_snr_from_metadata(gt: EventSet) -> Dict[int, float]:
    meta = dict(gt.metadata or {})
    raw = meta.get("unit_snr") or meta.get("unit_snr_peak") or meta.get("snr_by_unit")
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, float] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _unit_snr_from_h5(h5_path: str | Path | None) -> Dict[int, float]:
    if h5_path is None:
        return {}
    path = Path(h5_path)
    if not path.exists():
        return {}
    try:
        with h5py.File(path, "r") as f:
            out: Dict[int, float] = {}
            if "spiketrains" in f and isinstance(f["spiketrains"], h5py.Group):
                for key, child in f["spiketrains"].items():
                    if not isinstance(child, h5py.Group):
                        continue
                    snr_key = "annotations/snr"
                    if snr_key not in child:
                        continue
                    try:
                        unit_id = int(key)
                        snr = float(np.asarray(child[snr_key][()]).reshape(()))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(snr):
                        out[int(unit_id)] = float(snr)
                if out:
                    return out

            if "voltage_peaks" not in f:
                return {}
            peaks = np.asarray(f["voltage_peaks"][:], dtype=np.float32)
            if peaks.ndim != 2 or peaks.shape[0] == 0 or peaks.shape[1] == 0:
                return {}
            ids = np.arange(peaks.shape[0], dtype=np.int64)
            signal_name = "recordings" if "recordings" in f else ("preprocess_data" if "preprocess_data" in f else None)
            if signal_name is None:
                return {}
            signal = f[signal_name]
            if signal.ndim != 2:
                return {}
            ch_axis = _infer_channel_axis((int(signal.shape[0]), int(signal.shape[1])), n_channels=int(peaks.shape[1]))
            noise_cache: Dict[int, float] = {}
            for unit_id, row in zip(ids.tolist(), peaks):
                ch = int(np.argmax(np.abs(row)))
                if ch not in noise_cache:
                    noise_cache[ch] = _estimate_noise_std_subsampled(
                        signal,
                        channel_axis=ch_axis,
                        channel=ch,
                        max_points=80000,
                    )
                amp = float(np.max(np.abs(row)))
                out[int(unit_id)] = float(amp / max(1e-12, float(noise_cache[ch])))
            return out
    except Exception:
        return {}


def _nanmean_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def _nanmedian_or_nan(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else float("nan")


def _unit_time_index(event_set: EventSet, units: np.ndarray) -> Dict[int, np.ndarray]:
    lookup: Dict[int, np.ndarray] = {}
    labels = event_set.labels.astype(np.int64, copy=False)
    times = event_set.times_samples.astype(np.int64, copy=False)
    for unit in units.tolist():
        mask = labels == int(unit)
        lookup[int(unit)] = np.sort(times[mask], kind="mergesort")
    return lookup


def _dominant_channel_index(event_set: EventSet, units: np.ndarray) -> Dict[int, int]:
    if event_set.channels is None:
        return {}
    lookup: Dict[int, int] = {}
    labels = event_set.labels.astype(np.int64, copy=False)
    channels = event_set.channels.astype(np.int64, copy=False)
    for unit in units.tolist():
        ch = channels[labels == int(unit)]
        if ch.size == 0:
            continue
        vals, counts = np.unique(ch, return_counts=True)
        lookup[int(unit)] = int(vals[int(np.argmax(counts))])
    return lookup


def _unit_y_positions(event_set: EventSet, units: np.ndarray) -> Dict[int, float]:
    if event_set.unit_positions is None:
        return {}
    lookup: Dict[int, float] = {}
    for unit in units.tolist():
        pos = event_set.unit_positions.get(int(unit))
        if pos is None:
            continue
        arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            lookup[int(unit)] = float(arr[1])
        elif arr.size == 1:
            lookup[int(unit)] = float(arr[0])
    return lookup


def _rank_candidates(
    gt_unit: int,
    pred_units: np.ndarray,
    *,
    gt_y: Dict[int, float],
    pred_y: Dict[int, float],
    gt_ch: Dict[int, int],
    pred_ch: Dict[int, int],
    cfg: BenchmarkConfig,
) -> np.ndarray:
    pred_units = np.asarray(pred_units, dtype=np.int64).reshape(-1)
    if pred_units.size == 0:
        return pred_units

    mode = str(cfg.candidate_match_mode).lower()
    limit = int(cfg.n_check)

    def _limit(arr: np.ndarray) -> np.ndarray:
        if limit > 0:
            return arr[:limit]
        return arr

    if mode in ("all", "none", ""):
        return pred_units

    if mode in ("auto", "position", "unit_position", "spatial"):
        gt_val = gt_y.get(int(gt_unit))
        if gt_val is not None and pred_y:
            ranked = sorted(
                pred_units.tolist(),
                key=lambda unit: (
                    abs(float(gt_val) - float(pred_y.get(int(unit), np.inf))),
                    int(unit),
                ),
            )
            return _limit(np.asarray(ranked, dtype=np.int64))

    if mode in ("auto", "channel", "dominant_channel"):
        gt_val = gt_ch.get(int(gt_unit))
        if gt_val is not None and pred_ch:
            ranked = sorted(
                pred_units.tolist(),
                key=lambda unit: (
                    abs(int(gt_val) - int(pred_ch.get(int(unit), 10**9))),
                    int(unit),
                ),
            )
            return _limit(np.asarray(ranked, dtype=np.int64))

    # When we do not have a reliable spatial proxy, evaluate all predicted
    # units so the scoring itself stays as close as possible to bench.py.
    return pred_units


def _match_single_gt_unit(
    gt_unit: int,
    gt_times_by_unit: Dict[int, np.ndarray],
    pred_times_by_unit: Dict[int, np.ndarray],
    pred_units: np.ndarray,
    *,
    tol_samples: int,
    gt_y: Dict[int, float],
    pred_y: Dict[int, float],
    gt_ch: Dict[int, int],
    pred_ch: Dict[int, int],
    cfg: BenchmarkConfig,
) -> Dict[str, Any]:
    ss = gt_times_by_unit.get(int(gt_unit))
    if ss is None or len(ss) == 0:
        raise ValueError(f"No GT spikes for unit {gt_unit}")

    candidates = _rank_candidates(
        int(gt_unit),
        pred_units,
        gt_y=gt_y,
        pred_y=pred_y,
        gt_ch=gt_ch,
        pred_ch=pred_ch,
        cfg=cfg,
    )
    if candidates.size == 0:
        raise ValueError(f"No predicted units available for GT unit {gt_unit}")

    fmax = 0.0
    miss = 1.0
    fpos = 0.0
    best_ind = int(candidates[0])
    best_n_match = 0
    best_pred_spikes = int(pred_times_by_unit.get(best_ind, np.zeros((0,), dtype=np.int64)).size)
    matched_all = -1.0 * np.ones((candidates.size,), dtype=np.float64)

    no_match = 0
    for j, pred_unit in enumerate(candidates.tolist()):
        ss0 = pred_times_by_unit.get(int(pred_unit))
        if ss0 is None or len(ss0) == 0:
            no_match += 1
            continue

        n0, _, _ = nmatch(ss0, ss, dt=tol_samples)
        fmax_new = 1 - max(0.0, 1 - n0 / len(ss)) - max(0.0, 1 - n0 / len(ss0))
        matched_all[j] = float(n0)

        if fmax_new > fmax:
            miss = max(0.0, 1 - n0 / len(ss))
            fpos = max(0.0, 1 - n0 / len(ss0))
            fmax = float(fmax_new)
            best_ind = int(pred_unit)
            best_n_match = int(n0)
            best_pred_spikes = int(len(ss0))

    if no_match == candidates.size:
        raise ValueError(f"No matching predicted spikes for GT unit {gt_unit}")

    if best_n_match <= 0:
        best_ind = -1
        best_pred_spikes = 0

    return {
        "fmax": float(fmax),
        "fmiss": float(miss),
        "fpos": float(fpos),
        "best_ind": int(best_ind),
        "matched_all": matched_all,
        "top_inds": candidates.astype(np.int64, copy=False),
        "n_matched": int(best_n_match),
        "n_gt_spikes": int(len(ss)),
        "n_pred_spikes_best": int(best_pred_spikes),
    }


def num_correct(fpos: np.ndarray, fmiss: np.ndarray, score_threshold: float = 0.8) -> int:
    score = 1 - np.asarray(fpos, dtype=np.float64) - np.asarray(fmiss, dtype=np.float64)
    return int(np.sum(score >= float(score_threshold)))


def _solve_assignment_on_indices(
    gt_indices: np.ndarray,
    pred_indices: np.ndarray,
    pair_score: np.ndarray,
    pair_n_match: np.ndarray,
) -> List[tuple[int, int]]:
    gt_indices = np.asarray(gt_indices, dtype=np.int64).reshape(-1)
    pred_indices = np.asarray(pred_indices, dtype=np.int64).reshape(-1)
    if gt_indices.size == 0 or pred_indices.size == 0:
        return []

    local_score = pair_score[np.ix_(gt_indices, pred_indices)]
    local_n_match = pair_n_match[np.ix_(gt_indices, pred_indices)]
    valid = np.isfinite(local_score) & (local_n_match > 0)

    if not np.any(valid):
        return []

    reward = np.where(valid, np.maximum(local_score, 0.0), 0.0)
    if not np.any(reward > 0):
        return []

    assignments: List[tuple[int, int]] = []
    if linear_sum_assignment is not None:
        n_row, n_col = reward.shape
        n = int(max(n_row, n_col))
        reward_pad = np.zeros((n, n), dtype=np.float64)
        reward_pad[:n_row, :n_col] = reward

        max_reward = float(np.max(reward_pad))
        cost = max_reward - reward_pad
        rr, cc = linear_sum_assignment(cost)
        for r, c in zip(rr.tolist(), cc.tolist()):
            if r >= n_row or c >= n_col:
                continue
            if reward[r, c] <= 0:
                continue
            assignments.append((int(gt_indices[r]), int(pred_indices[c])))
        return assignments

    edges: List[tuple[float, int, int]] = []
    for r in range(reward.shape[0]):
        for c in range(reward.shape[1]):
            w = float(reward[r, c])
            if w <= 0:
                continue
            edges.append((w, int(gt_indices[r]), int(pred_indices[c])))

    edges.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for _, g_idx, p_idx in edges:
        if g_idx in used_gt or p_idx in used_pred:
            continue
        used_gt.add(g_idx)
        used_pred.add(p_idx)
        assignments.append((g_idx, p_idx))
    return assignments


def _build_mapping_from_assignments(
    assignments: List[tuple[int, int]],
    gt_units: np.ndarray,
    pred_units: np.ndarray,
) -> Dict[str, Dict[int, int]]:
    gt_to_pred: Dict[int, int] = {}
    pred_to_gt: Dict[int, int] = {}
    for g_idx, p_idx in assignments:
        if g_idx < 0 or g_idx >= gt_units.size:
            continue
        if p_idx < 0 or p_idx >= pred_units.size:
            continue
        g_unit = int(gt_units[g_idx])
        p_unit = int(pred_units[p_idx])
        gt_to_pred[g_unit] = p_unit
        pred_to_gt[p_unit] = g_unit
    return {
        "gt_to_pred": gt_to_pred,
        "pred_to_gt": pred_to_gt,
    }


def _one_to_one_from_independent_with_rematch(
    *,
    gt_units: np.ndarray,
    pred_units: np.ndarray,
    best_ind: np.ndarray,
    pair_score: np.ndarray,
    pair_n_match: np.ndarray,
) -> Dict[str, Any]:
    if gt_units.size == 0 or pred_units.size == 0:
        return {
            "gt_to_pred": {},
            "pred_to_gt": {},
            "assignments": [],
            "conflict_groups": 0,
            "conflicted_gt": 0,
            "unmatched_after_conflict": 0,
            "sum_score": 0.0,
        }

    pred_to_idx = {int(unit): idx for idx, unit in enumerate(pred_units.tolist())}

    kept: Dict[int, int] = {}
    conflict_groups = 0
    conflicted_gt = 0
    matched_gts = []
    for g_idx, pred_unit in enumerate(best_ind.tolist()):
        p_idx = pred_to_idx.get(int(pred_unit))
        if p_idx is None:
            continue
        if pair_n_match[g_idx, p_idx] <= 0:
            continue
        matched_gts.append((g_idx, p_idx))

    pred_to_gt_indices: Dict[int, List[int]] = {}
    for g_idx, p_idx in matched_gts:
        pred_to_gt_indices.setdefault(int(p_idx), []).append(int(g_idx))

    unmatched_gt: set[int] = set()
    for p_idx, g_list in pred_to_gt_indices.items():
        if len(g_list) == 1:
            kept[int(g_list[0])] = int(p_idx)
            continue
        conflict_groups += 1
        conflicted_gt += int(len(g_list))
        best_g = min(
            g_list,
            key=lambda g: (
                -float(pair_score[g, p_idx]),
                int(gt_units[g]),
            ),
        )
        kept[int(best_g)] = int(p_idx)
        for g in g_list:
            if g != best_g:
                unmatched_gt.add(int(g))

    for g_idx in range(gt_units.size):
        if g_idx in kept:
            continue
        if not np.any(pair_n_match[g_idx, :] > 0):
            continue
        unmatched_gt.add(int(g_idx))

    used_pred = set(kept.values())
    available_pred = [idx for idx in range(pred_units.size) if idx not in used_pred]
    rematched = _solve_assignment_on_indices(
        np.asarray(sorted(unmatched_gt), dtype=np.int64),
        np.asarray(available_pred, dtype=np.int64),
        pair_score,
        pair_n_match,
    )

    assignments = [(g, p) for g, p in kept.items()]
    assignments.extend(rematched)
    mapping = _build_mapping_from_assignments(assignments, gt_units, pred_units)

    sum_score = 0.0
    for g_idx, p_idx in assignments:
        sum_score += float(pair_score[g_idx, p_idx])

    return {
        "gt_to_pred": mapping["gt_to_pred"],
        "pred_to_gt": mapping["pred_to_gt"],
        "assignments": assignments,
        "conflict_groups": int(conflict_groups),
        "conflicted_gt": int(conflicted_gt),
        "unmatched_after_conflict": int(len(unmatched_gt)),
        "sum_score": float(sum_score),
    }


def _one_to_one_global_optimal(
    *,
    gt_units: np.ndarray,
    pred_units: np.ndarray,
    pair_score: np.ndarray,
    pair_n_match: np.ndarray,
) -> Dict[str, Any]:
    assignments = _solve_assignment_on_indices(
        np.arange(gt_units.size, dtype=np.int64),
        np.arange(pred_units.size, dtype=np.int64),
        pair_score,
        pair_n_match,
    )
    mapping = _build_mapping_from_assignments(assignments, gt_units, pred_units)
    sum_score = 0.0
    for g_idx, p_idx in assignments:
        sum_score += float(pair_score[g_idx, p_idx])
    return {
        "gt_to_pred": mapping["gt_to_pred"],
        "pred_to_gt": mapping["pred_to_gt"],
        "assignments": assignments,
        "sum_score": float(sum_score),
    }


def compare_recordings(
    pred: EventSet,
    gt: EventSet,
    *,
    fs: float,
    cfg: BenchmarkConfig | None = None,
) -> Dict[str, Any]:
    cfg = cfg or BenchmarkConfig()
    pred_sorted = pred.sorted_copy()
    gt_sorted = gt.sorted_copy()
    tol_samples = _resolve_tol_samples(fs, cfg)

    gt_units = np.unique(gt_sorted.labels.astype(np.int64, copy=False))
    pred_units = np.unique(
        pred_sorted.labels[pred_sorted.labels >= int(cfg.min_sort_label)].astype(np.int64, copy=False)
    )

    gt_times_by_unit = _unit_time_index(gt_sorted, gt_units)
    pred_times_by_unit = _unit_time_index(pred_sorted, pred_units)
    gt_y = _unit_y_positions(gt_sorted, gt_units)
    pred_y = _unit_y_positions(pred_sorted, pred_units)
    gt_ch = _dominant_channel_index(gt_sorted, gt_units)
    pred_ch = _dominant_channel_index(pred_sorted, pred_units)
    pred_unit_to_idx = {int(unit): idx for idx, unit in enumerate(pred_units.tolist())}

    pair_n_match = np.zeros((gt_units.size, pred_units.size), dtype=np.int64)
    pair_fmiss = np.full((gt_units.size, pred_units.size), np.nan, dtype=np.float64)
    pair_fpos = np.full((gt_units.size, pred_units.size), np.nan, dtype=np.float64)
    pair_score = np.full((gt_units.size, pred_units.size), np.nan, dtype=np.float64)

    pred_counts = np.zeros((pred_units.size,), dtype=np.int64)
    for p_idx, pred_unit in enumerate(pred_units.tolist()):
        pred_counts[p_idx] = int(pred_times_by_unit.get(int(pred_unit), np.zeros((0,), dtype=np.int64)).size)

    matched_all: List[np.ndarray] = []
    top_inds: List[np.ndarray] = []
    fmax = np.zeros((gt_units.size,), dtype=np.float64)
    fmiss = np.full((gt_units.size,), np.nan, dtype=np.float64)
    fpos = np.full((gt_units.size,), np.nan, dtype=np.float64)
    best_ind = np.full((gt_units.size,), -1, dtype=np.int64)
    n_matched = np.zeros((gt_units.size,), dtype=np.int64)
    n_gt_spikes = np.zeros((gt_units.size,), dtype=np.int64)
    n_pred_spikes_best = np.zeros((gt_units.size,), dtype=np.int64)

    n_skipped = 0
    for idx in range(gt_units.size):
        gt_unit = int(gt_units[idx])
        try:
            out = _match_single_gt_unit(
                gt_unit,
                gt_times_by_unit,
                pred_times_by_unit,
                pred_units,
                tol_samples=tol_samples,
                gt_y=gt_y,
                pred_y=pred_y,
                gt_ch=gt_ch,
                pred_ch=pred_ch,
                cfg=cfg,
            )
            fmax[idx] = float(out["fmax"])
            fmiss[idx] = float(out["fmiss"])
            fpos[idx] = float(out["fpos"])
            best_ind[idx] = int(out["best_ind"])
            n_matched[idx] = int(out["n_matched"])
            n_gt_spikes[idx] = int(out["n_gt_spikes"])
            n_pred_spikes_best[idx] = int(out["n_pred_spikes_best"])
            matched_all.append(np.asarray(out["matched_all"], dtype=np.float64))
            top_inds.append(np.asarray(out["top_inds"], dtype=np.int64))

            gt_count = max(1, int(out["n_gt_spikes"]))
            cand_units = np.asarray(out["top_inds"], dtype=np.int64)
            cand_matches = np.asarray(out["matched_all"], dtype=np.float64)
            for j, pred_unit in enumerate(cand_units.tolist()):
                p_idx = pred_unit_to_idx.get(int(pred_unit))
                if p_idx is None:
                    continue
                n0 = int(max(0.0, float(cand_matches[j])))
                pred_count = int(pred_counts[p_idx])
                if pred_count <= 0:
                    continue
                miss = max(0.0, 1.0 - float(n0) / float(gt_count))
                fpos_val = max(0.0, 1.0 - float(n0) / float(pred_count))
                pair_n_match[idx, p_idx] = int(n0)
                pair_fmiss[idx, p_idx] = float(miss)
                pair_fpos[idx, p_idx] = float(fpos_val)
                pair_score[idx, p_idx] = float(1.0 - fpos_val - miss)
        except ValueError:
            n_skipped += 1
            n_gt_spikes[idx] = int(gt_times_by_unit.get(int(gt_unit), np.zeros((0,), dtype=np.int64)).size)
            matched_all.append(np.asarray([], dtype=np.float64))
            top_inds.append(np.asarray([], dtype=np.int64))

    one_to_one_rematched = _one_to_one_from_independent_with_rematch(
        gt_units=gt_units,
        pred_units=pred_units,
        best_ind=best_ind,
        pair_score=pair_score,
        pair_n_match=pair_n_match,
    )
    one_to_one_global = _one_to_one_global_optimal(
        gt_units=gt_units,
        pred_units=pred_units,
        pair_score=pair_score,
        pair_n_match=pair_n_match,
    )

    chosen_one_to_one = one_to_one_global
    if float(one_to_one_rematched["sum_score"]) > float(one_to_one_global["sum_score"]):
        chosen_one_to_one = one_to_one_rematched

    return {
        "gt_units": gt_units.astype(np.int64, copy=False),
        "pred_units": pred_units.astype(np.int64, copy=False),
        "fmax": fmax,
        "fmiss": fmiss,
        "fpos": fpos,
        "best_ind": best_ind,
        "matched_all": matched_all,
        "top_inds": top_inds,
        "n_matched": n_matched,
        "n_gt_spikes": n_gt_spikes,
        "n_pred_spikes_best": n_pred_spikes_best,
        "pair_n_match": pair_n_match,
        "pair_fmiss": pair_fmiss,
        "pair_fpos": pair_fpos,
        "pair_score": pair_score,
        "pred_counts": pred_counts,
        "one_to_one_rematched": one_to_one_rematched,
        "one_to_one_global": one_to_one_global,
        "one_to_one": chosen_one_to_one,
        "n_skipped": int(n_skipped),
        "tol_samples": int(tol_samples),
        "pred_sorted": pred_sorted,
        "gt_sorted": gt_sorted,
    }


def benchmark_with_gt(
    pred: EventSet,
    gt: EventSet,
    *,
    fs: float,
    cfg: BenchmarkConfig | None = None,
    h5_path: str | Path | None = None,
    channel_positions: np.ndarray | None = None,
    quality_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = cfg or BenchmarkConfig()
    compared = compare_recordings(pred, gt, fs=fs, cfg=cfg)

    gt_units = compared["gt_units"]
    pred_units = compared["pred_units"]
    fmax = compared["fmax"]
    fmiss = compared["fmiss"]
    fpos = compared["fpos"]
    best_ind = compared["best_ind"]
    n_matched = compared["n_matched"]
    n_gt_spikes = compared["n_gt_spikes"]
    n_pred_spikes_best = compared["n_pred_spikes_best"]
    pair_score = compared["pair_score"]
    pair_n_match = compared["pair_n_match"]
    pair_fmiss = compared["pair_fmiss"]
    pair_fpos = compared["pair_fpos"]
    pred_counts = compared["pred_counts"]

    one_to_one = compared["one_to_one"]
    gt_to_pred_1to1: Dict[int, int] = one_to_one["gt_to_pred"]
    pred_to_gt_1to1: Dict[int, int] = one_to_one["pred_to_gt"]
    tp_classification = _classification_acc_true_positive(
        compared["pred_sorted"],
        compared["gt_sorted"],
        tol_samples=int(compared["tol_samples"]),
        min_sort_label=int(cfg.min_sort_label),
        pred_to_gt_1to1=pred_to_gt_1to1,
    )

    pred_to_idx = {int(unit): idx for idx, unit in enumerate(pred_units.tolist())}
    gt_to_idx = {int(unit): idx for idx, unit in enumerate(gt_units.tolist())}
    micro_1to1 = _one_to_one_micro_counts(
        gt_units=gt_units,
        pred_units=pred_units,
        gt_to_pred=gt_to_pred_1to1,
        pred_to_gt=pred_to_gt_1to1,
        gt_to_idx=gt_to_idx,
        pred_to_idx=pred_to_idx,
        pair_n_match=pair_n_match,
        pred_counts=pred_counts,
        n_gt_spikes=n_gt_spikes,
    )

    unit_snr = _unit_snr_from_h5(h5_path)
    unit_snr.update(_unit_snr_from_metadata(compared["gt_sorted"]))
    snr_threshold = float(getattr(cfg, "snr_threshold", 8.0))
    high_snr_gt_units = {int(unit) for unit, snr in unit_snr.items() if float(snr) > snr_threshold}
    high_snr_gt_units &= {int(unit) for unit in gt_units.tolist()}
    tp_classification_snr_gt8 = _classification_acc_true_positive(
        compared["pred_sorted"],
        compared["gt_sorted"],
        tol_samples=int(compared["tol_samples"]),
        min_sort_label=int(cfg.min_sort_label),
        pred_to_gt_1to1=pred_to_gt_1to1,
        gt_units_keep=high_snr_gt_units,
    )
    micro_snr_gt8 = _one_to_one_micro_counts(
        gt_units=gt_units,
        pred_units=pred_units,
        gt_to_pred=gt_to_pred_1to1,
        pred_to_gt=pred_to_gt_1to1,
        gt_to_idx=gt_to_idx,
        pred_to_idx=pred_to_idx,
        pair_n_match=pair_n_match,
        pred_counts=pred_counts,
        n_gt_spikes=n_gt_spikes,
        gt_units_keep=high_snr_gt_units,
    )
    n_gt_events_snr_gt8 = int(
        sum(int(n_gt_spikes[gt_to_idx[int(unit)]]) for unit in high_snr_gt_units if int(unit) in gt_to_idx)
    )

    best_ind_1to1 = np.full((gt_units.size,), -1, dtype=np.int64)
    n_matched_1to1 = np.zeros((gt_units.size,), dtype=np.int64)
    fmiss_1to1 = np.full((gt_units.size,), np.nan, dtype=np.float64)
    fpos_1to1 = np.full((gt_units.size,), np.nan, dtype=np.float64)
    n_pred_spikes_best_1to1 = np.zeros((gt_units.size,), dtype=np.int64)

    for gt_unit, pred_unit in gt_to_pred_1to1.items():
        g_idx = gt_to_idx.get(int(gt_unit))
        p_idx = pred_to_idx.get(int(pred_unit))
        if g_idx is None or p_idx is None:
            continue
        best_ind_1to1[g_idx] = int(pred_unit)
        n_matched_1to1[g_idx] = int(pair_n_match[g_idx, p_idx])
        fmiss_1to1[g_idx] = float(pair_fmiss[g_idx, p_idx])
        fpos_1to1[g_idx] = float(pair_fpos[g_idx, p_idx])
        n_pred_spikes_best_1to1[g_idx] = int(pred_counts[p_idx])

    score = 1 - fpos - fmiss
    score_1to1 = 1 - fpos_1to1 - fmiss_1to1
    score_1to1_all_gt = np.where(np.isfinite(score_1to1), score_1to1, 0.0)
    valid = np.isfinite(score)
    valid_1to1 = np.isfinite(score_1to1)
    recovered = int(np.sum(score >= float(cfg.score_threshold)))
    recovered_1to1 = int(np.sum(score_1to1 >= float(cfg.score_threshold)))
    n_gt_events = int(compared["gt_sorted"].n_events)
    detect_summary = detection_metrics(
        compared["pred_sorted"].times_samples,
        compared["gt_sorted"].times_samples,
        int(compared["tol_samples"]),
    )
    classification_acc_true_positive = float(tp_classification["classification_acc_true_positive"])
    classification_acc_over_gt_events = (
        float(tp_classification["n_true_positive_events_correct"]) / float(n_gt_events)
        if n_gt_events > 0
        else float("nan")
    )
    sorted_score_mean_1to1_matched_only = _nanmean_or_nan(score_1to1)
    sorted_score_mean_1to1_all_gt = (
        float(np.mean(score_1to1_all_gt.astype(np.float64, copy=False)))
        if gt_units.size > 0
        else float("nan")
    )

    n_gt_matched_independent = int(np.sum((best_ind >= 0) & (n_matched > 0)))
    n_pred_used_independent = int(np.unique(best_ind[(best_ind >= 0) & (n_matched > 0)]).size)
    n_gt_matched_1to1 = int(len(gt_to_pred_1to1))
    n_pred_used_1to1 = int(len(pred_to_gt_1to1))
    macro_f1_1to1: List[float] = []
    for idx in range(gt_units.size):
        tp_u = float(n_matched_1to1[idx])
        fp_u = float(max(0, n_pred_spikes_best_1to1[idx] - n_matched_1to1[idx]))
        fn_u = float(max(0, n_gt_spikes[idx] - n_matched_1to1[idx]))
        macro_f1_1to1.append(float(_metrics_from_counts(tp_u, fp_u, fn_u)["f1"]))

    per_unit: List[Dict[str, Any]] = []
    for idx, gt_unit in enumerate(gt_units.tolist()):
        independent_pred_unit = int(best_ind[idx])
        pred_unit_1to1 = int(best_ind_1to1[idx])
        was_conflict_resolved = bool(independent_pred_unit != pred_unit_1to1 and pred_unit_1to1 >= 0)
        independent_pred_idx = pred_to_idx.get(int(independent_pred_unit))
        per_unit.append(
            {
                "gt_unit": int(gt_unit),
                "gt_snr": float(unit_snr.get(int(gt_unit), float("nan"))),
                "gt_snr_gt_threshold": bool(int(gt_unit) in high_snr_gt_units),
                "best_pred_unit": int(best_ind[idx]),
                "best_pred_unit_independent": int(best_ind[idx]),
                "best_pred_unit_1to1": int(best_ind_1to1[idx]),
                "tp": int(n_matched[idx]),
                "tp_independent": int(n_matched[idx]),
                "tp_1to1": int(n_matched_1to1[idx]),
                "n_gt_spikes": int(n_gt_spikes[idx]),
                "n_pred_spikes_best": int(n_pred_spikes_best[idx]),
                "n_pred_spikes_best_independent": int(n_pred_spikes_best[idx]),
                "n_pred_spikes_best_1to1": int(n_pred_spikes_best_1to1[idx]),
                "fmiss": float(fmiss[idx]),
                "fmiss_independent": float(fmiss[idx]),
                "fmiss_1to1": float(fmiss_1to1[idx]),
                "fpos": float(fpos[idx]),
                "fpos_independent": float(fpos[idx]),
                "fpos_1to1": float(fpos_1to1[idx]),
                "sorted_score": float(score[idx]),
                "score": float(score[idx]),
                "sorted_score_independent": float(score[idx]),
                "sorted_score_1to1": float(score_1to1[idx]),
                "score_1to1": float(score_1to1[idx]),
                "fmax": float(fmax[idx]),
                "candidate_count": int(len(compared["top_inds"][idx])),
                "top_pred_units": compared["top_inds"][idx].astype(np.int64, copy=False).tolist(),
                "matched_candidates": compared["matched_all"][idx].astype(np.float64, copy=False).tolist(),
                "n_pred_fragments": int(np.sum(pair_n_match[idx, :] > 0)),
                "n_gt_merge_partners": (
                    int(np.sum(pair_n_match[:, int(independent_pred_idx)] > 0))
                    if independent_pred_idx is not None
                    else 0
                ),
                "was_conflict_resolved": was_conflict_resolved,
            }
        )

    summary = {
        "n_gt_units": int(gt_units.size),
        "n_pred_units_sortable": int(pred_units.size),
        "n_gt_events": int(n_gt_events),
        "n_pred_events": int(compared["pred_sorted"].n_events),
        "detect_tp": int(detect_summary["TP"]),
        "detect_fp": int(detect_summary["FP"]),
        "detect_fn": int(detect_summary["FN"]),
        "detect_precision": float(detect_summary["Precision"]),
        "detect_recall": float(detect_summary["Recall"]),
        "detect_f1": float(detect_summary["F1"]),
        "detect_sorted_score": float(detect_summary["sorted_score"]),
        "n_true_positive_events": int(tp_classification["n_true_positive_events"]),
        "n_true_positive_events_correct": int(tp_classification["n_true_positive_events_correct"]),
        "n_true_positive_events_correct_raw_label_match": int(
            tp_classification["n_true_positive_events_correct_raw_label_match"]
        ),
        "n_true_positive_events_unmapped_pred_unit": int(
            tp_classification["n_true_positive_events_unmapped_pred_unit"]
        ),
        "classification_acc_true_positive": float(classification_acc_true_positive),
        "classification_acc_true_positive_conditional": float(classification_acc_true_positive),
        "classification_acc_over_gt_events": float(classification_acc_over_gt_events),
        "acc_1to1_tp": float(classification_acc_true_positive),
        "acc_1to1_global": float(classification_acc_over_gt_events),
        "raw_label_match_acc_true_positive": float(tp_classification["raw_label_match_acc_true_positive"]),
        "tp_1to1": int(micro_1to1["tp"]),
        "fp_1to1": int(micro_1to1["fp"]),
        "fn_1to1": int(micro_1to1["fn"]),
        "precision_1to1": float(micro_1to1["precision"]),
        "recall_1to1": float(micro_1to1["recall"]),
        "f1_1to1": float(micro_1to1["f1"]),
        "sorted_score_1to1": float(micro_1to1["sorted_score"]),
        "sort_tp_micro_mapped": int(micro_1to1["tp"]),
        "sort_fp_micro_mapped": int(micro_1to1["fp"]),
        "sort_fn_micro_mapped": int(micro_1to1["fn"]),
        "sort_precision_micro_mapped": float(micro_1to1["precision"]),
        "sort_recall_micro_mapped": float(micro_1to1["recall"]),
        "sort_f1_micro_mapped": float(micro_1to1["f1"]),
        "sort_f1_macro_mapped": float(np.mean(np.asarray(macro_f1_1to1, dtype=np.float64))) if macro_f1_1to1 else 0.0,
        "snr_threshold": float(snr_threshold),
        "snr_gt8_n_gt_units": int(len(high_snr_gt_units)),
        "snr_gt8_n_gt_events": int(n_gt_events_snr_gt8),
        "snr_gt8_n_true_positive_events": int(tp_classification_snr_gt8["n_true_positive_events"]),
        "snr_gt8_n_true_positive_events_correct": int(tp_classification_snr_gt8["n_true_positive_events_correct"]),
        "snr_gt8_acc_1to1_tp": float(tp_classification_snr_gt8["classification_acc_true_positive"]),
        "snr_gt8_acc_1to1_global": (
            float(tp_classification_snr_gt8["n_true_positive_events_correct"]) / float(n_gt_events_snr_gt8)
            if n_gt_events_snr_gt8 > 0
            else float("nan")
        ),
        "snr_gt8_precision_1to1": (
            float(micro_snr_gt8["precision"]) if n_gt_events_snr_gt8 > 0 else float("nan")
        ),
        "snr_gt8_recall_1to1": (
            float(micro_snr_gt8["recall"]) if n_gt_events_snr_gt8 > 0 else float("nan")
        ),
        "snr_gt8_sorted_score_1to1": (
            float(micro_snr_gt8["sorted_score"]) if n_gt_events_snr_gt8 > 0 else float("nan")
        ),
        "units_evaluated": int(np.sum(valid)),
        "units_evaluated_1to1": int(np.sum(valid_1to1)),
        "units_skipped": int(compared["n_skipped"]),
        "num_correct_score_ge_threshold": int(recovered),
        "num_correct_score_ge_threshold_1to1": int(recovered_1to1),
        "unit_recovery_rate_at_0.8": float(recovered / max(1, gt_units.size)),
        "unit_recovery_rate_at_0.8_1to1": float(recovered_1to1 / max(1, gt_units.size)),
        "n_gt_units_matched_independent": int(n_gt_matched_independent),
        "n_gt_units_matched_1to1": int(n_gt_matched_1to1),
        "gt_match_rate_independent": float(n_gt_matched_independent / max(1, gt_units.size)),
        "gt_match_rate_1to1": float(n_gt_matched_1to1 / max(1, gt_units.size)),
        "n_pred_units_used_independent": int(n_pred_used_independent),
        "n_pred_units_used_1to1": int(n_pred_used_1to1),
        "pred_unit_utilization_independent": float(n_pred_used_independent / max(1, pred_units.size)),
        "pred_unit_utilization_1to1": float(n_pred_used_1to1 / max(1, pred_units.size)),
        "sorted_score": _nanmean_or_nan(score),
        "sorted_score_mean": _nanmean_or_nan(score),
        "sorted_score_mean_1to1": float(sorted_score_mean_1to1_matched_only),
        "sorted_score_mean_1to1_matched_only": float(sorted_score_mean_1to1_matched_only),
        "sorted_score_mean_1to1_all_gt": float(sorted_score_mean_1to1_all_gt),
        "sorted_score_median": _nanmedian_or_nan(score),
        "sorted_score_median_1to1": _nanmedian_or_nan(score_1to1),
        "sorted_score_p10": float(np.nanpercentile(score[valid], 10)) if np.any(valid) else float("nan"),
        "sorted_score_p10_1to1": float(np.nanpercentile(score_1to1[valid_1to1], 10)) if np.any(valid_1to1) else float("nan"),
        "sorted_score_p90": float(np.nanpercentile(score[valid], 90)) if np.any(valid) else float("nan"),
        "sorted_score_p90_1to1": float(np.nanpercentile(score_1to1[valid_1to1], 90)) if np.any(valid_1to1) else float("nan"),
        "fmax_mean": _nanmean_or_nan(fmax),
        "fmiss_mean": _nanmean_or_nan(fmiss),
        "fmiss_mean_1to1": _nanmean_or_nan(fmiss_1to1),
        "fpos_mean": _nanmean_or_nan(fpos),
        "fpos_mean_1to1": _nanmean_or_nan(fpos_1to1),
    }

    diagnostics: Dict[str, Any] = {}
    if bool(cfg.diagnostics):
        diagnostics = {
            "tol_samples": int(compared["tol_samples"]),
            "candidate_match_mode": str(cfg.candidate_match_mode),
            "event_match_mode": "time",
            "n_matched_events": int(detect_summary["TP"]),
            "n_check": int(cfg.n_check),
            "score_threshold": float(cfg.score_threshold),
            "gt_units": gt_units.astype(np.int64, copy=False).tolist(),
            "pred_units": pred_units.astype(np.int64, copy=False).tolist(),
            "gt_unit_snr": {str(k): float(v) for k, v in unit_snr.items()},
            "snr_gt8_units": sorted(int(unit) for unit in high_snr_gt_units),
            "matching_mode": "global_one_to_one_max_score",
            "one_to_one_pred_to_gt": {str(k): int(v) for k, v in pred_to_gt_1to1.items()},
            "one_to_one_gt_to_pred": {str(k): int(v) for k, v in gt_to_pred_1to1.items()},
            "sum_score_independent": float(np.nansum(score)),
            "sum_score_1to1": float(one_to_one["sum_score"]),
            "sum_score_1to1_global": float(compared["one_to_one_global"]["sum_score"]),
            "sum_score_1to1_rematched": float(compared["one_to_one_rematched"]["sum_score"]),
            "conflict_groups": int(compared["one_to_one_rematched"]["conflict_groups"]),
            "conflicted_gt_units": int(compared["one_to_one_rematched"]["conflicted_gt"]),
            "unmatched_after_conflict": int(compared["one_to_one_rematched"]["unmatched_after_conflict"]),
        }

    no_gt_report: Dict[str, Any] | None = None
    if h5_path is not None and pred.channels is not None:
        no_gt_quality_kwargs = dict(quality_kwargs or {})
        if channel_positions is not None and "channel_positions" not in no_gt_quality_kwargs:
            no_gt_quality_kwargs["channel_positions"] = channel_positions
        try:
            no_gt_report = benchmark_no_gt(
                compared["pred_sorted"],
                h5_path=h5_path,
                fs=float(fs),
                cfg=cfg,
                quality_kwargs=no_gt_quality_kwargs,
            )
            no_gt_summary = dict(no_gt_report.get("summary", {}))
            summary.update(
                {
                    "no_gt_n_events_total": int(no_gt_summary.get("n_events_total", 0)),
                    "no_gt_n_events_kept": int(no_gt_summary.get("n_events_kept", 0)),
                    "no_gt_n_units_kept": int(no_gt_summary.get("n_units_kept", 0)),
                    "no_gt_n_units_evaluated": int(no_gt_summary.get("n_units_evaluated", 0)),
                    "no_gt_DBI": float(no_gt_summary.get("DBI", float("nan"))),
                    "no_gt_topological_similarity": float(no_gt_summary.get("topological_similarity", float("nan"))),
                    "no_gt_SNR_mean": float(no_gt_summary.get("SNR_mean", float("nan"))),
                    "no_gt_FP_est_mean": float(no_gt_summary.get("FP_est_mean", float("nan"))),
                    "no_gt_missed_spikes_FN_mean": float(no_gt_summary.get("missed_spikes_FN_mean", float("nan"))),
                    "no_gt_sorted_score_mean": float(no_gt_summary.get("sorted_score_mean", float("nan"))),
                    "no_gt_contamination_score_mean": float(no_gt_summary.get("contamination_score_mean", float("nan"))),
                    "no_gt_RPV_rate_mean": float(no_gt_summary.get("RPV_rate_mean", float("nan"))),
                }
            )
        except Exception as exc:  # pragma: no cover
            if bool(cfg.diagnostics):
                diagnostics["no_gt_error"] = str(exc)

    return {
        "summary": summary,
        "windows": [],
        "per_unit": per_unit,
        "diagnostics": diagnostics,
        "no_gt": no_gt_report,
    }



__all__ = [
    "BenchmarkConfig",
    "EventSet",
    "benchmark_no_gt",
    "benchmark_with_gt",
    "clustering_metrics",
    "compare_recordings",
    "count_time_matches_one_to_one",
    "detection_metrics",
    "dump_debug_events",
    "evaluate_no_gt_quality",
    "match_events",
    "nmatch",
    "num_correct",
    "plot_embedding_2d",
]
