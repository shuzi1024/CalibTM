"""Bounded Abilene cost measurement after all six matched trajectories complete."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    parser.add_argument('--repo-root', type=Path, default=Path('/mnt/suzikun/CalibTM'))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    here = root / 'analysis/spin_path_controls_20260917'
    auth = json.loads((here / 'AUTHORIZATION.json').read_text())
    if time.time() >= auth['deadline_unix'] - 300:
        raise RuntimeError('At least five minutes of the authorized allocation must remain')
    run_root = root / 'outputs/spin-path-controls-20260917-v1'
    for variant in ('spin_direct', 'context_only', 'direct_only'):
        for seed in (41001, 41002):
            job = run_root / 'abilene' / variant / f'seed{seed}'
            result = json.loads((job / 'result.json').read_text())
            identity = {'state': 'complete', 'epochs_completed': 160,
                        'protocol': 'spin-path-controls-development-20260917-v1',
                        'dataset': 'abilene', 'variant': variant, 'seed': seed,
                        'config_sha256': sha(job / 'config.json'),
                        'best_checkpoint_sha256': sha(job / 'best.pt')}
            if any(result.get(key) != value for key, value in identity.items()):
                raise RuntimeError(f'Completed Abilene result/checkpoint identity differs: {variant}/{seed}')
    script = here / 'benchmark_inference_abilene.py'
    script_sha = 'eed2c73816dfd141a6ff53af23ebdec529d471dce945a51a4ef285138aa725cf'
    if sha(script) != script_sha:
        raise RuntimeError('Abilene benchmark source differs from the reviewed version')
    busy = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if busy:
        raise RuntimeError('Selected GPU is occupied')
    output = here / 'INFERENCE_BENCHMARK_ABILENE.json'
    log = here / 'INFERENCE_BENCHMARK_ABILENE.log'
    launch_path = here / 'INFERENCE_BENCHMARK_ABILENE_LAUNCH.json'
    if any(path.exists() for path in (output, log, launch_path)):
        raise RuntimeError('Preserve existing Abilene benchmark artifacts')
    remaining = int(auth['deadline_unix'] - time.time() - 10)
    if remaining < 290:
        raise RuntimeError('Insufficient time remaining after identity checks')
    command = ['timeout', '--signal=TERM', '--kill-after=10s', str(remaining) + 's',
        str(root / '.venv/bin/python'), '-u', str(script), '--repo-root', str(root),
        '--run-root', str(run_root), '--device', 'cuda:0',
        '--deadline-unix', str(auth['deadline_unix']), '--output', str(output)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with log.open('xb') as stream:
        proc = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'hostname': socket.gethostname(), 'gpu': args.gpu, 'timeout_pid': proc.pid,
        'dataset': 'abilene', 'started_unix': time.time(), 'deadline_unix': auth['deadline_unix'],
        'benchmark_sha256': script_sha, 'command': command, 'log': str(log), 'output': str(output)}
    with launch_path.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
