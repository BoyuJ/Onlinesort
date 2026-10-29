#!/usr/bin/env python3
"""Score saved predictions independently of inference and without retraining."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_offline as ro
from scripts.run_online import online_output_path
from src.benchmark_io import load_gt_events, load_mysort_npz
from src.dataset import MEArecH5Reader
from src.eval import EventSet
from src.paper_evaluation import score_recording, aggregate_records, paired_comparisons
from src.reproducibility import load_recording_manifest, resolve_data_records, sha256, environment, write_json


def path(value, root=ROOT):
    p = Path(value).expanduser()
    return p if p.is_absolute() else root / p


def load_truth(file, config, fs):
    file = Path(file)
    if file.suffix.lower() == '.npz':
        with np.load(file, allow_pickle=False) as arrays:
            return EventSet(arrays['times'], arrays['labels'], arrays['channels'] if 'channels' in arrays else None)
    reader = MEArecH5Reader(file)
    if abs(reader.get_fs() - fs) > 1e-6:
        raise ValueError("GT sampling rate differs from evaluation configuration")
    n_channels = reader.get_recordings_shape()[1]
    positions, neighbors, _ = ro.resolve_neighbor_graph(channel_positions=reader.read_channel_positions(),
        n_channels=n_channels, electrode_config=config['offline']['electrode_config'],
        neighbor_source='config', fallback_k=config['patch']['k'], root=ROOT, logger=__import__('logging').getLogger(__name__))
    truth = load_gt_events(file, format='mearec')
    truth, _ = ro.align_gt_events_for_benchmark(truth, input_path=file, fs=fs,
        n_channels=n_channels, cfg=config, bench_cfg_raw=config['benchmark'], neighbors_lookup=neighbors)
    return truth


def write_csv(file, rows):
    with Path(file).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['method'])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/infer.yaml')
    parser.add_argument('--manifest', default='configs/manifests/benchmark.json')
    parser.add_argument('--data-root', default='data/recordings')
    parser.add_argument('--predictions-dir', default='data/cache/onlinesort_infer')
    parser.add_argument('--prediction-manifest', help='JSON records with name, prediction, gt, condition, optional method/start_s/stop_s; paths relative to manifest')
    parser.add_argument('--prediction', help='one prediction NPZ; requires --gt')
    parser.add_argument('--gt', help='MEArec H5, or peak-aligned NPZ with times, labels, channels')
    parser.add_argument('--condition', choices=['static', 'drift'], default='static')
    parser.add_argument('--start-s', type=float, default=5.)
    parser.add_argument('--stop-s', type=float, default=100.)
    parser.add_argument('--initialization-s', type=float, default=5.)
    parser.add_argument('--output-dir', default='data/reports/paper')
    parser.add_argument('--check-reference', action='store_true', help='verify complete paper cohort and four-decimal reference means')
    args = parser.parse_args(argv)
    cfg = ro.load_yaml(path(args.config));fs = float(cfg.get('fs', 32000))
    raw = cfg['benchmark']
    bc = ro.build_benchmark_config(ro.resolve_benchmark_backend(raw), bench_cfg_raw=raw,
                                   label_cfg=cfg['labeling'], det_cfg=cfg['detection'])
    if args.prediction:
        if not args.gt:
            parser.error('--prediction requires --gt')
        tasks = [dict(name=Path(args.prediction).stem, prediction=path(args.prediction), gt=path(args.gt), condition=args.condition)]
    elif args.prediction_manifest:
        manifest_file = path(args.prediction_manifest)
        tasks = json.loads(manifest_file.read_text())['records']
        tasks = [dict(r, prediction=path(r['prediction'],manifest_file.parent), gt=path(r['gt'],manifest_file.parent)) for r in tasks]
    else:
        manifest = load_recording_manifest(path(args.manifest))
        inputs = resolve_data_records(manifest, args.data_root, ROOT)
        tasks = [dict(r, prediction=online_output_path(path(args.predictions_dir) / (r['name']+'.npz'), args.initialization_s),gt=gt)
                 for r,gt in zip(manifest['records'],inputs)]
    if not tasks:
        raise ValueError('No predictions to score')
    rows=[];provenance=[];seen=set()
    for task in tasks:
        name=task['name'];method=task.get('method','OnlineSort')
        if task['condition'] not in ('static', 'drift'):
            raise ValueError('recording condition must be static or drift')
        start=float(task.get('start_s',args.start_s));stop=float(task.get('stop_s',args.stop_s))
        key=(name,method,start,stop)
        if key in seen:
            raise ValueError(f'Duplicate prediction entry: {key}')
        seen.add(key)
        prediction_file=Path(task['prediction']);gt_file=Path(task['gt'])
        # Seal identity predictions before accessing any GT.
        digest=sha256(prediction_file)
        pred=load_mysort_npz(prediction_file,fs=fs)
        gt=load_truth(gt_file,cfg,fs)
        scores=score_recording(pred,gt,fs=fs,config=bc,start_s=start,stop_s=stop)
        if sha256(prediction_file)!=digest:
            raise RuntimeError('Predictions changed during scoring')
        rows.append(dict(name=name,method=method,condition=task['condition'],start_s=start,stop_s=stop,**scores))
        provenance.append(dict(name=name,method=method,predictions_sha256=digest,gt_name=gt_file.name))
    summary=aggregate_records(rows)
    out=path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'per_record.csv',rows);write_csv(out/'summary.csv',summary)
    comparisons=paired_comparisons(rows)
    write_csv(out/'comparisons.csv',comparisons)
    reference_check=None
    if args.check_reference:
        expected=json.loads((ROOT/'configs/paper_reference.json').read_text())['metrics']
        paper_manifest=load_recording_manifest(ROOT/'configs/manifests/benchmark.json')
        expected_names={r['name'] for r in paper_manifest['records']}
        selected=[r for r in rows if r['method']=='OnlineSort' and (r['start_s'],r['stop_s'])==(5.,100.)]
        if {r['name'] for r in selected}!=expected_names or len(selected)!=384:
            raise ValueError('--check-reference requires exactly the complete 384-record [5,100) cohort')
        checks={}
        for condition,targets in expected.items():
            result=next(r for r in summary if r['method']=='OnlineSort' and r['condition']==condition and r['start_s']==5. and r['stop_s']==100.)
            checks[condition]={k:abs(result[k]-v)<=.0001 for k,v in targets.items()}
        reference_check=dict(passed=all(v for c in checks.values() for v in c.values()),checks=checks)
    write_json(out/'run.json',dict(environment=environment(),config=cfg,scoring='one-to-one, peak-aligned, 0.5 ms; recording-weighted',
        provenance=provenance,comparisons=comparisons,reference_check=reference_check))
    for row in summary:print(row)
    if reference_check and not reference_check['passed']:
        raise SystemExit('Computed metrics differ from paper reference; see run.json')
    return 0


if __name__ == '__main__':
    main()
