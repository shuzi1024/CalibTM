"""One bounded sequential queue on one idle GPU; only new experiment outputs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/mnt/suzikun/CalibTM')
HERE = ROOT / 'analysis/spin_path_controls_20260917'
OUTPUT = ROOT / 'outputs/spin-path-controls-20260917-v1'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def worker(args, auth):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT / f'queue-{socket.gethostname()}-gpu{args.gpu}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for variant in args.variants.split(','):
            job = OUTPUT / args.dataset / variant / f'seed{args.seed}'
            command = [str(ROOT / '.venv/bin/python'), '-u', '-m',
                       'experiments.spin_path_controls_v1.runner',
                       '--dataset', args.dataset, '--variant', variant,
                       '--seed', str(args.seed), '--device', 'cuda:0',
                       '--deadline-unix', str(auth['deadline_unix'])]
            if time.time() >= auth['deadline_unix'] - 120:
                raise RuntimeError('No time for another task within allocation')
            # A complete matching result must be verified by the runner itself.
            phases = ['train'] if (job / 'result.json').exists() else ['smoke', 'train']
            for phase in phases:
                if phase == 'smoke' and (job / 'smoke/SMOKE.json').exists():
                    report = json.loads((job / 'smoke/SMOKE.json').read_text())
                    if report.get('passed'):
                        continue
                cmd = command + (['--smoke'] if phase == 'smoke' else
                                 (['--resume'] if (job / 'last.pt').exists() else []))
                stamp = f'{args.dataset}-{variant}-seed{args.seed}-{phase}-{time.time_ns()}'
                logfile = OUTPUT / 'logs' / (stamp + '.log')
                logfile.parent.mkdir(parents=True, exist_ok=True)
                started = time.time()
                print(json.dumps({'event': 'start', 'variant': variant, 'phase': phase,
                                  'seed': args.seed, 'log': str(logfile)}), flush=True)
                with logfile.open('xb') as stream:
                    run = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT)
                record = {'command': cmd, 'hostname': socket.gethostname(),
                          'gpu': args.gpu, 'started_unix': started,
                          'finished_unix': time.time(), 'returncode': run.returncode,
                          'log': str(logfile)}
                write(logfile.with_suffix('.exit.json'), record)
                print(json.dumps({'event': 'exit', **record}), flush=True)
                if run.returncode:
                    raise RuntimeError(f'{variant}/{phase} returned {run.returncode}')
                expected = job / ('smoke/SMOKE.json' if phase == 'smoke' else 'result.json')
                result = json.loads(expected.read_text())
                if phase == 'smoke':
                    assert result['passed'] and result['exact_checkpoint_continuation']
                else:
                    assert result['state'] == 'complete' and result['epochs_completed'] == 160
        print(json.dumps({'event': 'queue_complete', 'finished_unix': time.time()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--dataset', choices=('geant', 'abilene'), default='geant')
    parser.add_argument('--seed', type=int, choices=(41001, 41002), required=True)
    parser.add_argument('--variants', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    assert set(args.variants.split(',')) <= {'spin_direct', 'context_only', 'direct_only'}
    auth = json.loads((HERE / 'AUTHORIZATION.json').read_text())
    if args.worker:
        return worker(args, auth)
    assert time.time() < auth['deadline_unix'] - 300
    report = json.loads((HERE / 'CPU_CHECKS.json').read_text())
    assert report['passed'], 'Synthetic checks must pass before GPU work'
    frozen = json.loads((HERE / 'SOURCE_FREEZE.json').read_text())
    for name, expected in frozen['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    busy = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not busy, f'GPU {args.gpu} already occupied'
    remaining = int(auth['deadline_unix'] - time.time() - 10)
    command = ['timeout', '--signal=TERM', '--kill-after=10s', str(remaining) + 's',
               str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()),
               '--worker', '--gpu', str(args.gpu), '--dataset', args.dataset,
               '--seed', str(args.seed), '--variants', args.variants]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    label = f'{args.dataset}-{socket.gethostname()}-gpu{args.gpu}-{time.time_ns()}'
    log = OUTPUT / 'logs' / (label + '.queue.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('xb') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'hostname': socket.gethostname(), 'gpu': args.gpu, 'seed': args.seed,
              'dataset': args.dataset, 'variants': args.variants.split(','),
              'timeout_pid': proc.pid, 'command': command, 'log': str(log),
              'started_unix': time.time(), 'deadline_unix': auth['deadline_unix']}
    write(log.with_suffix('.launch.json'), record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
