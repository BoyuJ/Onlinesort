from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_yaml


@dataclass
class FileUIDInfo:
    path: Path
    fs: float
    spikes_by_uid: Dict[int, np.ndarray]
    peak_ch_by_uid: Dict[int, int]
    mean_wave_by_uid: Dict[int, np.ndarray]


def _resolve_path(path_value: str | Path, root: Path) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return root / p


def _read_spiketrains(path: Path) -> Tuple[float, Dict[int, np.ndarray]]:
    with h5py.File(path, "r") as f:
        if "info" in f and "recordings" in f["info"] and "fs" in f["info"]["recordings"]:
            fs = float(f["info"]["recordings"]["fs"][()])
        else:
            raise KeyError(f"{path}: cannot find info/recordings/fs")

        if "spiketrains" not in f:
            raise KeyError(f"{path}: cannot find spiketrains group")
        grp = f["spiketrains"]
        out: Dict[int, np.ndarray] = {}
        for key in grp.keys():
            try:
                uid = int(key)
            except ValueError:
                uid = int(key.split("_")[-1])
            times = np.asarray(grp[key]["times"][:], dtype=np.float64)
            times = np.sort(times)
            out[uid] = times
    return fs, out


def _read_peak_channels_from_voltage_peaks(path: Path) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with h5py.File(path, "r") as f:
        if "voltage_peaks" not in f:
            return out
        peaks = np.asarray(f["voltage_peaks"][:], dtype=np.float32)
        if peaks.ndim != 2 or peaks.shape[0] == 0:
            return out
        peak_ch = np.argmax(np.abs(peaks), axis=1).astype(np.int32, copy=False)
        if "template_ids" in f:
            template_ids = np.asarray(f["template_ids"][:]).reshape(-1)
        else:
            template_ids = np.arange(peaks.shape[0], dtype=np.int64)
        if template_ids.size != peaks.shape[0]:
            template_ids = np.arange(peaks.shape[0], dtype=np.int64)
        for uid_raw, ch_raw in zip(template_ids.tolist(), peak_ch.tolist()):
            try:
                out[int(uid_raw)] = int(ch_raw)
            except Exception:
                continue
    return out


def _extract_uid_peak_waveforms_from_recordings(
    path: Path,
    *,
    fs: float,
    spikes_by_uid: Dict[int, np.ndarray],
    left: int,
    right: int,
    max_spikes_per_uid: int,
    seed: int,
) -> Tuple[Dict[int, int], Dict[int, np.ndarray]]:
    peak_ch_by_uid: Dict[int, int] = {}
    mean_wave_by_uid: Dict[int, np.ndarray] = {}
    rng = np.random.default_rng(seed)

    with h5py.File(path, "r") as f:
        if "recordings" not in f:
            return peak_ch_by_uid, mean_wave_by_uid
        rec = f["recordings"]
        n_samples = int(rec.shape[0])
        n_channels = int(rec.shape[1])

        for uid, times_sec in spikes_by_uid.items():
            if times_sec.size == 0:
                continue
            spikes = np.round(times_sec * fs).astype(np.int64, copy=False)
            spikes = spikes[(spikes >= 0) & (spikes < n_samples)]
            if spikes.size == 0:
                continue
            if max_spikes_per_uid > 0 and spikes.size > max_spikes_per_uid:
                spikes = np.sort(rng.choice(spikes, size=max_spikes_per_uid, replace=False))

            ch_votes: List[int] = []
            waves: List[np.ndarray] = []
            for s in spikes.tolist():
                seg_start = max(0, int(s) - left)
                seg_end = min(n_samples, int(s) + right)
                seg = np.asarray(rec[seg_start:seg_end, :], dtype=np.float32)
                if seg.size == 0:
                    continue
                flat = int(np.argmax(np.abs(seg)))
                t_rel, c_peak = np.unravel_index(flat, seg.shape)
                t_peak = int(seg_start + int(t_rel))

                wave = np.zeros((left + right,), dtype=np.float32)
                wave_lo = int(t_peak - left)
                wave_hi = int(t_peak + right)
                src_lo = max(0, wave_lo)
                src_hi = min(n_samples, wave_hi)
                dst_lo = max(0, -wave_lo)
                dst_hi = dst_lo + (src_hi - src_lo)
                if src_hi > src_lo:
                    wave[dst_lo:dst_hi] = np.asarray(rec[src_lo:src_hi, int(c_peak)], dtype=np.float32).reshape(-1)

                ch_votes.append(int(c_peak))
                waves.append(wave)

            if not waves:
                continue

            votes = np.asarray(ch_votes, dtype=np.int32)
            dominant_peak_ch = int(np.bincount(votes, minlength=max(1, n_channels)).argmax())
            idx_dom = np.where(votes == dominant_peak_ch)[0]
            if idx_dom.size > 0:
                used_waves = [waves[i] for i in idx_dom.tolist()]
            else:
                used_waves = waves

            mean_wave = np.mean(np.stack(used_waves, axis=0), axis=0).astype(np.float32, copy=False)
            mean_wave = mean_wave - float(np.mean(mean_wave))
            norm = float(np.linalg.norm(mean_wave))
            if norm > 1e-8:
                mean_wave = mean_wave / norm

            peak_ch_by_uid[int(uid)] = dominant_peak_ch
            mean_wave_by_uid[int(uid)] = mean_wave

    return peak_ch_by_uid, mean_wave_by_uid


def load_file_info(
    path: Path,
    *,
    left: int,
    right: int,
    max_spikes_per_uid: int,
    seed: int,
) -> FileUIDInfo:
    fs, spikes = _read_spiketrains(path)
    peak_ch, mean_wave = _extract_uid_peak_waveforms_from_recordings(
        path,
        fs=fs,
        spikes_by_uid=spikes,
        left=left,
        right=right,
        max_spikes_per_uid=max_spikes_per_uid,
        seed=seed,
    )
    if not peak_ch:
        peak_ch = _read_peak_channels_from_voltage_peaks(path)
    return FileUIDInfo(
        path=path,
        fs=fs,
        spikes_by_uid=spikes,
        peak_ch_by_uid=peak_ch,
        mean_wave_by_uid=mean_wave,
    )


def _array_equal_or_close(a: np.ndarray, b: np.ndarray, atol: float) -> Tuple[bool, float]:
    if a.shape != b.shape:
        return False, float("inf")
    if a.size == 0:
        return True, 0.0
    diff = np.abs(a - b)
    max_abs = float(np.max(diff))
    return bool(max_abs <= atol), max_abs


def _sample_overlap_ratio(times_a: np.ndarray, fs_a: float, times_b: np.ndarray, fs_b: float) -> float:
    if times_a.size == 0 and times_b.size == 0:
        return 1.0
    s_a = np.round(times_a * fs_a).astype(np.int64, copy=False)
    s_b = np.round(times_b * fs_b).astype(np.int64, copy=False)
    if s_a.size == 0 or s_b.size == 0:
        return 0.0
    inter = np.intersect1d(s_a, s_b, assume_unique=False).size
    denom = max(s_a.size, s_b.size)
    return float(inter / max(1, denom))


def compare_files(
    base: FileUIDInfo,
    other: FileUIDInfo,
    *,
    atol_sec: float,
    wave_corr_thresh: float,
) -> Dict[str, object]:
    u_base = set(base.spikes_by_uid.keys())
    u_other = set(other.spikes_by_uid.keys())
    shared = sorted(u_base & u_other)
    only_base = sorted(u_base - u_other)
    only_other = sorted(u_other - u_base)

    exact_time_match = 0
    close_time_match = 0
    same_count = 0
    peak_ch_compared = 0
    peak_ch_match = 0
    wave_corr_compared = 0
    wave_corr_high = 0
    wave_corrs: List[float] = []
    overlap_ratios: List[float] = []
    mismatched_uids: List[Dict[str, object]] = []

    for uid in shared:
        t0 = base.spikes_by_uid[uid]
        t1 = other.spikes_by_uid[uid]

        counts_equal = int(t0.size == t1.size)
        same_count += counts_equal
        exact = bool(t0.shape == t1.shape and np.array_equal(t0, t1))
        if exact:
            exact_time_match += 1

        close, max_abs = _array_equal_or_close(t0, t1, atol=atol_sec)
        if close:
            close_time_match += 1

        overlap = _sample_overlap_ratio(t0, base.fs, t1, other.fs)
        overlap_ratios.append(overlap)

        ch0 = base.peak_ch_by_uid.get(uid)
        ch1 = other.peak_ch_by_uid.get(uid)
        if ch0 is not None and ch1 is not None:
            peak_ch_compared += 1
            if int(ch0) == int(ch1):
                peak_ch_match += 1

        corr = None
        w0 = base.mean_wave_by_uid.get(uid)
        w1 = other.mean_wave_by_uid.get(uid)
        if w0 is not None and w1 is not None and w0.shape == w1.shape and w0.size > 0:
            corr = float(np.dot(w0, w1) / max(np.linalg.norm(w0) * np.linalg.norm(w1), 1e-8))
            corr = float(np.clip(corr, -1.0, 1.0))
            wave_corr_compared += 1
            wave_corrs.append(corr)
            if corr >= wave_corr_thresh:
                wave_corr_high += 1

        if (not close) or (overlap < 0.95) or (not counts_equal) or (corr is not None and corr < wave_corr_thresh):
            mismatched_uids.append(
                {
                    "uid": int(uid),
                    "count_base": int(t0.size),
                    "count_other": int(t1.size),
                    "max_abs_time_diff_sec": None if np.isinf(max_abs) else float(max_abs),
                    "sample_overlap_ratio": float(overlap),
                    "peak_ch_base": None if ch0 is None else int(ch0),
                    "peak_ch_other": None if ch1 is None else int(ch1),
                    "wave_corr": None if corr is None else float(corr),
                }
            )

    n_shared = len(shared)
    n_total_ref = len(u_base)
    exact_ratio = float(exact_time_match / n_shared) if n_shared > 0 else 0.0
    close_ratio = float(close_time_match / n_shared) if n_shared > 0 else 0.0
    count_ratio = float(same_count / n_shared) if n_shared > 0 else 0.0
    peak_ratio = float(peak_ch_match / peak_ch_compared) if peak_ch_compared > 0 else None
    wave_corr_high_ratio = float(wave_corr_high / wave_corr_compared) if wave_corr_compared > 0 else None
    wave_corr_mean = float(np.mean(wave_corrs)) if wave_corrs else None
    wave_corr_median = float(np.median(wave_corrs)) if wave_corrs else None
    wave_corr_min = float(np.min(wave_corrs)) if wave_corrs else None
    mean_overlap = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0

    likely_same_uid_semantics = (
        len(only_base) == 0
        and len(only_other) == 0
        and n_shared == n_total_ref
        and close_ratio >= 0.98
        and count_ratio >= 0.98
        and mean_overlap >= 0.98
        and (wave_corr_high_ratio is None or wave_corr_high_ratio >= 0.95)
    )

    return {
        "base_file": str(base.path),
        "other_file": str(other.path),
        "shared_uids": int(n_shared),
        "uids_only_in_base": [int(u) for u in only_base],
        "uids_only_in_other": [int(u) for u in only_other],
        "same_spike_count_ratio": count_ratio,
        "exact_time_match_ratio": exact_ratio,
        "close_time_match_ratio": close_ratio,
        "mean_sample_overlap_ratio": mean_overlap,
        "peak_channel_match_ratio": peak_ratio,
        "peak_channel_compared_uids": int(peak_ch_compared),
        "wave_corr_thresh": float(wave_corr_thresh),
        "wave_corr_compared_uids": int(wave_corr_compared),
        "wave_corr_ge_thresh_ratio": wave_corr_high_ratio,
        "wave_corr_mean": wave_corr_mean,
        "wave_corr_median": wave_corr_median,
        "wave_corr_min": wave_corr_min,
        "likely_same_uid_semantics": bool(likely_same_uid_semantics),
        "mismatched_uid_examples": mismatched_uids[:20],
    }


def _paths_from_config(config_path: Path) -> List[Path]:
    cfg = load_yaml(config_path)
    data_cfg = cfg.get("data", {})
    raw_paths = data_cfg.get("mearec_h5_paths", [])
    if not isinstance(raw_paths, list):
        raise ValueError("config data.mearec_h5_paths must be a list")
    if not raw_paths:
        raise ValueError("config data.mearec_h5_paths is empty")
    return [_resolve_path(str(p), ROOT) for p in raw_paths]


def _print_summary(file_infos: List[FileUIDInfo], pair_reports: List[Dict[str, object]]) -> None:
    print("=== File Overview ===")
    for fi in file_infos:
        print(
            f"- {fi.path}\n"
            f"  fs={fi.fs:.6f} uids={len(fi.spikes_by_uid)}"
            f" spikes_total={sum(v.size for v in fi.spikes_by_uid.values())}"
        )
    print("")
    print("=== Pairwise UID Consistency ===")
    for rep in pair_reports:
        print(
            f"- {Path(rep['base_file']).name} vs {Path(rep['other_file']).name}\n"
            f"  shared_uids={rep['shared_uids']} likely_same_uid_semantics={rep['likely_same_uid_semantics']}\n"
            f"  same_count={rep['same_spike_count_ratio']:.4f} close_time={rep['close_time_match_ratio']:.4f} "
            f"exact_time={rep['exact_time_match_ratio']:.4f} overlap={rep['mean_sample_overlap_ratio']:.4f}"
        )
        if rep["peak_channel_match_ratio"] is not None:
            print(
                f"  peak_channel_match={rep['peak_channel_match_ratio']:.4f} "
                f"(compared_uids={rep['peak_channel_compared_uids']})"
            )
        if rep["wave_corr_ge_thresh_ratio"] is not None:
            print(
                f"  wave_corr_mean={rep['wave_corr_mean']:.4f} "
                f"wave_corr_ge_{rep['wave_corr_thresh']:.2f}={rep['wave_corr_ge_thresh_ratio']:.4f} "
                f"(compared_uids={rep['wave_corr_compared_uids']})"
            )
        if rep["uids_only_in_base"] or rep["uids_only_in_other"]:
            print(
                f"  uids_only_in_base={len(rep['uids_only_in_base'])} "
                f"uids_only_in_other={len(rep['uids_only_in_other'])}"
            )
        if rep["mismatched_uid_examples"]:
            first = rep["mismatched_uid_examples"][0]
            print(
                "  example_mismatch="
                f"uid={first['uid']} count_base={first['count_base']} count_other={first['count_other']} "
                f"overlap={first['sample_overlap_ratio']:.4f} wave_corr={first['wave_corr']}"
            )
    print("")
    all_ok = all(bool(r["likely_same_uid_semantics"]) for r in pair_reports)
    print(f"FINAL_VERDICT likely_same_uid_semantics_across_all_files={all_ok}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check whether same uid across MEArec h5 files likely refers to the same neuron."
    )
    ap.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "pretrain_nnx32_peak1.yaml"),
        help="pretrain config path; used when --h5 is not provided",
    )
    ap.add_argument(
        "--h5",
        type=str,
        nargs="+",
        default=None,
        help="explicit h5 paths; overrides config data.mearec_h5_paths",
    )
    ap.add_argument(
        "--time-atol-sec",
        type=float,
        default=1e-7,
        help="absolute tolerance (sec) for per-uid spike-time equality check",
    )
    ap.add_argument("--left", type=int, default=20, help="left samples for peak-centered waveform extraction")
    ap.add_argument("--right", type=int, default=44, help="right samples for peak-centered waveform extraction")
    ap.add_argument(
        "--max-spikes-per-uid",
        type=int,
        default=200,
        help="max spikes per uid when extracting peak-channel waveforms from recordings (0=all)",
    )
    ap.add_argument(
        "--wave-corr-thresh",
        type=float,
        default=0.8,
        help="uid-level waveform correlation threshold",
    )
    ap.add_argument("--seed", type=int, default=42, help="random seed for spike subsampling")
    ap.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="optional path to save full JSON report",
    )
    args = ap.parse_args()

    if args.h5:
        paths = [_resolve_path(p, ROOT) for p in args.h5]
    else:
        paths = _paths_from_config(_resolve_path(args.config, ROOT))

    if len(paths) < 2:
        raise ValueError("need at least 2 h5 files to compare uid consistency")

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"h5 file not found: {p}")

    file_infos = [
        load_file_info(
            p,
            left=max(1, int(args.left)),
            right=max(1, int(args.right)),
            max_spikes_per_uid=max(0, int(args.max_spikes_per_uid)),
            seed=int(args.seed) + i * 9973,
        )
        for i, p in enumerate(paths)
    ]
    base = file_infos[0]
    pair_reports = [
        compare_files(
            base,
            other,
            atol_sec=float(args.time_atol_sec),
            wave_corr_thresh=float(args.wave_corr_thresh),
        )
        for other in file_infos[1:]
    ]

    report = {
        "files": [str(p) for p in paths],
        "time_atol_sec": float(args.time_atol_sec),
        "left": int(args.left),
        "right": int(args.right),
        "max_spikes_per_uid": int(args.max_spikes_per_uid),
        "wave_corr_thresh": float(args.wave_corr_thresh),
        "pair_reports": pair_reports,
    }
    if args.json_out:
        out_path = _resolve_path(args.json_out, ROOT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(file_infos, pair_reports)


if __name__ == "__main__":
    main()
