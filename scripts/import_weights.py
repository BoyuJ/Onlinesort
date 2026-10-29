#!/usr/bin/env python3
"""Import a separately obtained paper checkpoint; no downloads or bundled weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reproducibility import sha256, write_json


def import_checkpoint(source, role, destination=None):
    from src.model_io import build_model_config, load_torch_file
    from src.nn.model import SpikeEmbeddingModel
    from src.utils import load_yaml

    reference = json.loads((ROOT / 'configs/paper_reference.json').read_text())
    expected = reference['checkpoints'][role]
    source = Path(source).expanduser().resolve()
    if sha256(source) != expected['sha256']:
        raise ValueError(f"Checkpoint SHA256 does not match paper {role}")
    payload = load_torch_file(source)
    state = payload.get('model_state_dict', payload)
    model = SpikeEmbeddingModel(build_model_config(load_yaml(ROOT / 'configs/train.yaml')['model']))
    # The stage-one file includes an unused historical overlap head.
    state = {k: v for k, v in state.items() if not k.startswith('gate_net.overlap_head.')}
    model.load_state_dict(state, strict=True)
    if payload.get('metadata', {}).get('stage') != expected['stage']:
        raise ValueError('Checkpoint stage does not match requested role')
    target = Path(destination) if destination else ROOT / expected['destination']
    if not target.is_absolute():
        target = ROOT / target
    if target.exists():
        if sha256(target) != expected['sha256']:
            raise FileExistsError(f'Refusing to overwrite different weights: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if sha256(target) != expected['sha256']:
        raise IOError('Copied checkpoint failed SHA256 verification')
    write_json(target.with_suffix('.import.json'), {'role': role, 'sha256': expected['sha256'],
               'stage': expected['stage'], 'source_name': source.name})
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='checkpoint obtained separately')
    parser.add_argument('--role', choices=['encoder', 'stage1'], required=True)
    parser.add_argument('--destination', help='optional project-relative or absolute destination')
    args = parser.parse_args()
    print(import_checkpoint(args.source, args.role, args.destination))


if __name__ == '__main__':
    main()
