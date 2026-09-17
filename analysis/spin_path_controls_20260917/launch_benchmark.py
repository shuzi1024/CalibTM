"""Bounded GEANT cost measurement after all six matched trajectories complete."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

root = Path('/mnt/suzikun/CalibTM')
here = root / 'analysis/spin_path_controls_20260917'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu', type=int, choices=(0, 1), default=0)
args = parser.parse_args()
auth = json.loads((here / 'AUTHORIZATION.json').read_text())
assert time.time() < auth['deadline_unix'] - 300
for variant in ('spin_direct', 'context_only', 'direct_only'):
    for seed in (41001, 41002):
        job = root / 'outputs/spin-path-controls-20260917-v1/geant' / variant / f'seed{seed}'
        result = json.loads((job / 'result.json').read_text())
        assert result['state'] == 'complete' and result['epochs_completed'] == 160
        assert result['protocol'] == 'spin-path-controls-development-20260917-v1'
script = here / 'benchmark_inference.py'
assert hashlib.sha256(script.read_bytes()).hexdigest() == '3d981e2a719034ff0c1795712f3d1262c5d5ad422f1a232010b45c54d5f92031'
busy = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
    '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
assert not busy, 'Selected GPU is occupied'
output, log = here / 'INFERENCE_BENCHMARK.json', here / 'INFERENCE_BENCHMARK.log'
assert not output.exists() and not log.exists(), 'Preserve existing benchmark files'
remaining = int(auth['deadline_unix'] - time.time() - 10)
command = ['timeout', '--signal=TERM', '--kill-after=10s', str(remaining) + 's',
    str(root / '.venv/bin/python'), '-u', str(script), '--repo-root', str(root),
    '--run-root', 'outputs/spin-path-controls-20260917-v1', '--device', 'cuda:0',
    '--deadline-unix', str(auth['deadline_unix']), '--output', str(output)]
environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
    CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
with log.open('xb') as stream:
    proc = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
record = {'hostname': socket.gethostname(), 'gpu': args.gpu, 'timeout_pid': proc.pid,
    'started_unix': time.time(), 'deadline_unix': auth['deadline_unix'],
    'command': command, 'log': str(log), 'output': str(output)}
(here / 'INFERENCE_BENCHMARK_LAUNCH.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
