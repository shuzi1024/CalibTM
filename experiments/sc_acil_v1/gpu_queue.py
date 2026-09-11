"""Operational queue; concurrency changes no scientific identity."""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.acil_innovation_v1.config import CUBLAS_WORKSPACE_CONFIG

from .execution import require_all_acil_dependencies
from .freeze import load_and_verify_freeze
from .jobs import planned_jobs
from .protocol import canonical_json_bytes


def gpu_slots(gpus: tuple[str, ...], workers_per_gpu: int) -> tuple[str, ...]:
    if not gpus or any(not value.isdecimal() for value in gpus):
        raise ValueError("GPU ids must be canonical nonnegative integers")
    if len(set(gpus)) != len(gpus) or workers_per_gpu < 1:
        raise ValueError("GPU ids must be unique and worker count positive")
    return tuple(gpu for gpu in gpus for _ in range(workers_per_gpu))


def _worker_command(stage, job, output_root):
    return (
        sys.executable,
        "-m",
        "experiments.sc_acil_v1.worker",
        "--stage",
        stage,
        "--job-id",
        job.job_id,
        "--output-root",
        str(output_root),
    )


def run_queue(
    stage: str,
    output_root: Path,
    gpus: tuple[str, ...],
    *,
    workers_per_gpu: int,
) -> Path:
    if stage not in {"fit_acil", "formal_gate"}:
        raise ValueError("stage is outside the frozen queue")
    freeze = load_and_verify_freeze()
    output_root = Path(output_root)
    if stage == "formal_gate":
        require_all_acil_dependencies(
            output_root, manifest_sha256=str(freeze["manifest_sha256"])
        )
    slots = gpu_slots(gpus, workers_per_gpu)
    jobs = planned_jobs(stage)
    parent = output_root / "_gpu_queue"
    parent.mkdir(parents=True, exist_ok=True)
    queue_dir = parent / f"{stage}-{time.time_ns()}-{os.getpid()}"
    queue_dir.mkdir()
    logs = queue_dir / "logs"
    logs.mkdir()
    pending = deque(enumerate(jobs))
    free = deque(enumerate(slots))
    active = {}
    records = [
        {
            "job": job.to_json(),
            "state": "not_launched",
            "gpu": None,
            "command": None,
            "exit_code": None,
            "log": None,
        }
        for job in jobs
    ]
    stop_signal = None

    def stop(signum, _frame):
        nonlocal stop_signal
        stop_signal = int(signum)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, stop)
    try:
        while (pending or active) and stop_signal is None:
            while pending and free and stop_signal is None:
                job_index, job = pending.popleft()
                slot_index, gpu = free.popleft()
                log_path = logs / f"{job.job_id}.log"
                handle = log_path.open("xb")
                command = _worker_command(stage, job, output_root)
                env = os.environ.copy()
                env.update(
                    {
                        "CUDA_VISIBLE_DEVICES": gpu,
                        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                    }
                )
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    close_fds=True,
                    start_new_session=True,
                )
                records[job_index].update(
                    {
                        "state": "running",
                        "gpu": gpu,
                        "command": list(command),
                        "log": str(log_path),
                    }
                )
                active[slot_index] = (job_index, gpu, process, handle)
            completed = False
            for slot_index, (job_index, gpu, process, handle) in tuple(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                records[job_index]["exit_code"] = int(code)
                records[job_index]["state"] = "succeeded" if code == 0 else "failed"
                del active[slot_index]
                free.append((slot_index, gpu))
                completed = True
            if active and not completed:
                time.sleep(0.1)
    finally:
        if active:
            for slot_index, (job_index, _gpu, process, handle) in tuple(active.items()):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                handle.close()
                records[job_index]["exit_code"] = int(process.returncode)
                records[job_index]["state"] = "terminated"
                del active[slot_index]
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    failed = any(row["state"] != "succeeded" for row in records)
    summary = {
        "schema": "sc-acil-v1:gpu-queue:v1",
        "stage": stage,
        "manifest_sha256": freeze["manifest_sha256"],
        "gpus": list(gpus),
        "workers_per_gpu": workers_per_gpu,
        "planned_job_count": len(jobs),
        "signal": stop_signal,
        "status": "interrupted" if stop_signal is not None else (
            "complete_with_failures" if failed else "complete"
        ),
        "jobs": records,
    }
    path = queue_dir / "queue_summary.json"
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(summary) + b"\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen SC-ACIL stage")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--workers-per-gpu", required=True, type=int)
    args = parser.parse_args()
    gpus = tuple(args.gpus.split(","))
    path = run_queue(
        args.stage,
        args.output_root,
        gpus,
        workers_per_gpu=args.workers_per_gpu,
    )
    payload = json_load(path)
    print(canonical_json_bytes({"summary": str(path), "status": payload["status"]}).decode("ascii"))
    return 0 if payload["status"] == "complete" else 1


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="ascii"))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["gpu_slots", "main", "run_queue"]

