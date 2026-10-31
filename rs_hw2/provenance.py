"""Small, explicit experiment manifests; never inspect credential values."""
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from .data import sha256


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('No CSV rows')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def start_run(root, label, config):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = f'{stamp}_{label}_job{os.getenv("SLURM_JOB_ID", "local")}'
    run_dir = root / 'results' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = root / 'checkpoints' / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(run_dir / 'run.log'), logging.StreamHandler(sys.stdout)], force=True)
    source_root = Path(__file__).resolve().parents[1]
    files = [p for directory in ('rs_hw2', 'scripts', 'tests', 'configs', 'slurm') for p in (source_root / directory).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.sbatch', '.sh')]
    manifest = {
        'run_id': run_id, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': os.getenv('RS_SOURCE_COMMIT', 'uncommitted'),
        'source_root': str(source_root), 'source_sha256': {str(p.relative_to(source_root)): sha256(p) for p in files},
        'command': sys.argv, 'hostname': socket.gethostname(), 'platform': platform.platform(),
        'slurm': {key: os.getenv(key) for key in ('SLURM_JOB_ID', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_PARTITION', 'SLURM_JOB_QOS')},
        'python': sys.version, 'python_prefix': sys.prefix, 'base_prefix': sys.base_prefix,
        'packages': {name: version(name) for name in ('numpy', 'scipy', 'torch', 'matplotlib')},
        'config': config,
    }
    write_json(run_dir / 'manifest.json', manifest)
    write_json(run_dir / 'config.json', config)
    logging.info('Run directory: %s', run_dir)
    return run_dir, checkpoint_dir, manifest
