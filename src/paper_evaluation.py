"""Paper scoring: clip both streams, keep missed GT, then average recording scores."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from .eval import EventSet, benchmark_with_gt, match_events


def score_recording(prediction, truth, *, fs, config, start_s=5.0, stop_s=100.0):
    if not (np.isfinite(start_s) and np.isfinite(stop_s) and 0 <= start_s < stop_s):
        raise ValueError('scoring interval must satisfy 0 <= start < stop')
    lower, upper = start_s * fs, stop_s * fs
    pred = prediction.subset((prediction.times_samples >= lower) &
                             (prediction.times_samples < upper) & (prediction.labels >= 0))
    gt = truth.subset((truth.times_samples >= lower) & (truth.times_samples < upper))
    summary = benchmark_with_gt(pred, gt, fs=float(fs), cfg=config)['summary']
    result = {key: float(summary[key + '_1to1']) for key in ['precision', 'recall', 'f1']}
    result.update({key: int(summary[key + '_1to1']) for key in ['tp', 'fp', 'fn']})
    matches = match_events(pred, gt, tol_samples=max(1, int(round(fs * .0005))),
                           event_match_mode='time+channel', channel_radius_ch=config.channel_radius_ch)
    yt = np.array([gt.labels[m.gt_index] for m in matches], dtype=np.int64)
    yp = np.array([pred.labels[m.pred_index] for m in matches], dtype=np.int64)
    result.update(nmi=float(normalized_mutual_info_score(yt, yp)) if len(yt) else 0.,
                  ari=float(adjusted_rand_score(yt, yp)) if len(yt) else 0., matched_events=len(yt))
    return result


def aggregate_records(rows):
    summaries = []
    keys = sorted({(r['method'], r['start_s'], r['stop_s']) for r in rows})
    for method, start, stop in keys:
        subset = [r for r in rows if (r['method'], r['start_s'], r['stop_s']) == (method, start, stop)]
        for condition in ['static', 'drift', 'overall']:
            group = subset if condition == 'overall' else [r for r in subset if r['condition'] == condition]
            if group:
                summaries.append(dict(method=method, condition=condition, start_s=start, stop_s=stop,
                    recordings=len(group), **{key: float(np.mean([r[key] for r in group]))
                    for key in ['precision', 'recall', 'f1', 'nmi', 'ari']}))
    return summaries


def paired_comparisons(rows):
    comparisons = []
    for method in sorted({r['method'] for r in rows} - {'OnlineSort'}):
        windows = sorted({(r['start_s'], r['stop_s']) for r in rows})
        for start, stop in windows:
            groups = [{r['name']: r for r in rows if r['method'] == m and
                       (r['start_s'], r['stop_s']) == (start, stop)} for m in ['OnlineSort', method]]
            if not groups[0] or set(groups[0]) != set(groups[1]):
                continue
            names = sorted(groups[0])
            if any(groups[0][n]['condition'] != groups[1][n]['condition'] for n in names):
                raise ValueError('paired methods disagree on recording conditions')
            online = float(np.mean([groups[0][n]['precision'] for n in names]))
            baseline = float(np.mean([groups[1][n]['precision'] for n in names]))
            comparisons.append(dict(method=method, start_s=start, stop_s=stop, recordings=len(names),
                precision_difference=online-baseline,
                precision_relative_improvement_percent=100*(online/baseline-1) if baseline > 0 else None))
    return comparisons
