"""Portable recording manifests and run provenance for the public entry points."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def load_recording_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    records = manifest.get('records', [])
    if not records:
        raise ValueError('recording manifest is empty')
    names = [record['name'] for record in records]
    if len(names) != len(set(names)):
        raise ValueError('manifest recording names must be unique')
    for record in records:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('manifest recording paths must be relative to --data-root')
    return manifest


def resolve_data_records(manifest: dict, data_root: str | Path, project_root: Path) -> list[Path]:
    root = Path(data_root).expanduser()
    if not root.is_absolute():
        root = project_root / root
    paths = [root / record['path'] for record in manifest['records']]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing recording(s); prepare the data or set --data-root: ' + ', '.join(missing[:3]))
    return paths


def environment() -> dict:
    packages = {}
    for name in ['torch', 'numpy', 'scipy', 'scikit-learn', 'hdbscan', 'h5py', 'PyYAML']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
