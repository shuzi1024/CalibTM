"""Operational multi-GPU queue for complete frozen job grids.

This module schedules only identities returned by :func:`planned_jobs`.  It has
no scientific override, retry policy, data access, or occupancy-script role.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import IO, Sequence

from .adjudication import load_stage_adjudication
from .config import CUBLAS_WORKSPACE_CONFIG, canonical_json_bytes
from .jobs import PlannedJob, planned_jobs
from .manifest import active_manifest_stages


_WORKER_MODULE = "experiments.acil_innovation_v1.worker"
_POLL_INTERVAL_SECONDS = 0.05
_TERMINATE_TIMEOUT_SECONDS = 10.0
_DIRECT_UPSTREAM = {
    "stage_h": "stage0_acil_tune",
    "stage_i": "stage_h",
    "full_tune": "stage_i",
}


@dataclass(frozen=True, slots=True)
class QueueRunResult:
    exit_code: int
    summary_path: Path


@dataclass(slots=True)
class _ActiveWorker:
    job_index: int
    gpu: str
    process: object
    log_handle: IO[bytes]


@dataclass(slots=True)
class _SignalState:
    signum: int | None = None


def _parse_gpus(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise TypeError("gpus must be a comma-separated string")
    parts = value.split(",")
    if not parts or any(not part for part in parts):
        raise ValueError("gpus must contain at least one GPU index")
    if any(not part.isdecimal() or str(int(part)) != part for part in parts):
        raise ValueError("GPU indices must be canonical nonnegative integers")
    if len(set(parts)) != len(parts):
        raise ValueError("GPU indices must be unique")
    return tuple(parts)


def _normalize_gpus(gpus: Sequence[str]) -> tuple[str, ...]:
    if isinstance(gpus, (str, bytes)):
        raise TypeError("gpus must be a sequence of canonical GPU index strings")
    try:
        values = tuple(gpus)
    except TypeError as exc:
        raise TypeError("gpus must be a sequence") from exc
    if not values or any(not isinstance(value, str) for value in values):
        raise TypeError("gpus must be a nonempty sequence of strings")
    parsed = _parse_gpus(",".join(values))
    if parsed != values:
        raise ValueError("GPU sequence is not canonical")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a complete frozen ACIL-Innovation stage on isolated GPUs"
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpus", required=True, type=_parse_gpus)
    return parser


def _worker_command(stage: str, job: PlannedJob, output_root: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        _WORKER_MODULE,
        "--stage",
        stage,
        "--job-id",
        job.job_id,
        "--output-root",
        str(output_root),
    )


def _write_summary_exclusive(path: Path, payload: dict[str, object]) -> None:
    content = canonical_json_bytes(payload) + b"\n"
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _terminate_worker(worker: _ActiveWorker) -> int:
    process = worker.process
    returncode = process.poll()
    if returncode is None:
        process.terminate()
        try:
            returncode = process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=_TERMINATE_TIMEOUT_SECONDS)
    return int(returncode)


def _initial_records(jobs: tuple[PlannedJob, ...]) -> list[dict[str, object]]:
    return [
        {
            "attempt_count": 0,
            "command": None,
            "dataset": job.dataset,
            "exit_code": None,
            "gpu": None,
            "job_id": job.job_id,
            "launch_error": None,
            "log_path": None,
            "method": job.method,
            "seed_bundle": job.seed_bundle,
            "stage": job.stage,
            "state": "not_launched",
        }
        for job in jobs
    ]


def run_gpu_queue(
    stage: str, output_root: str | Path, gpus: Sequence[str]
) -> QueueRunResult:
    """Run every fixed job once with at most one worker on each listed GPU."""

    if not isinstance(stage, str) or not stage:
        raise TypeError("stage must be a nonempty string")
    if stage not in active_manifest_stages():
        raise ValueError(f"stage {stage!r} is outside the active manifest")
    gpu_ids = _normalize_gpus(gpus)
    upstream = _DIRECT_UPSTREAM.get(stage)
    if upstream is not None:
        decision = load_stage_adjudication(upstream, Path(output_root))
        if decision.get("stage") != upstream or decision.get("verdict") != "proceed":
            raise RuntimeError(
                f"stage {stage!r} requires an exact upstream proceed adjudication"
            )
    jobs = planned_jobs(stage)
    root = Path(output_root)
    queue_parent = root / "_gpu_queue"
    queue_parent.mkdir(parents=True, exist_ok=True)
    queue_directory = queue_parent / stage
    queue_directory.mkdir(mode=0o755, exist_ok=False)
    log_directory = queue_directory / "logs"
    log_directory.mkdir(mode=0o755)
    summary_path = queue_directory / "queue_summary.json"

    records = _initial_records(jobs)
    pending = deque(range(len(jobs)))
    free_gpus = deque(gpu_ids)
    active: dict[str, _ActiveWorker] = {}
    state = _SignalState()
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def request_stop(signum, _frame) -> None:
        if state.signum is None:
            state.signum = int(signum)

    for signum in previous_handlers:
        signal.signal(signum, request_stop)

    fatal_error: BaseException | None = None
    try:
        while pending or active:
            if state.signum is not None:
                break

            while pending and free_gpus and state.signum is None:
                job_index = pending.popleft()
                job = jobs[job_index]
                gpu = free_gpus.popleft()
                record = records[job_index]
                log_path = log_directory / f"{job.job_id}.log"
                log_handle = log_path.open("xb")
                record.update(
                    {
                        "attempt_count": 1,
                        "gpu": gpu,
                        "log_path": str(log_path),
                        "state": "launching",
                    }
                )
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
                command = _worker_command(stage, job, root)
                record["command"] = list(command)
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        close_fds=True,
                        start_new_session=True,
                    )
                except BaseException as exc:
                    log_handle.close()
                    record.update(
                        {
                            "launch_error": f"{type(exc).__name__}: {exc}",
                            "state": "launch_failed",
                        }
                    )
                    free_gpus.append(gpu)
                    if not isinstance(exc, Exception):
                        raise
                    continue
                record["state"] = "running"
                active[gpu] = _ActiveWorker(
                    job_index=job_index,
                    gpu=gpu,
                    process=process,
                    log_handle=log_handle,
                )

            completed_any = False
            for gpu, worker in tuple(active.items()):
                returncode = worker.process.poll()
                if returncode is None:
                    continue
                worker.log_handle.close()
                record = records[worker.job_index]
                record["exit_code"] = int(returncode)
                record["state"] = "succeeded" if returncode == 0 else "failed"
                del active[gpu]
                free_gpus.append(gpu)
                completed_any = True
            if active and not completed_any and state.signum is None:
                time.sleep(_POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        state.signum = signal.SIGINT
    except BaseException as exc:
        fatal_error = exc
    finally:
        if active:
            terminated_state = "terminated" if state.signum is not None else "aborted"
            for gpu, worker in tuple(active.items()):
                try:
                    returncode = _terminate_worker(worker)
                    records[worker.job_index]["exit_code"] = returncode
                    records[worker.job_index]["state"] = terminated_state
                finally:
                    worker.log_handle.close()
                    del active[gpu]
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    launched_count = sum(int(record["attempt_count"] == 1) for record in records)
    completed_states = {"succeeded", "failed", "terminated", "aborted"}
    completed_count = sum(record["state"] in completed_states for record in records)
    if state.signum is not None:
        status = "interrupted"
        exit_code = 128 + state.signum
    elif fatal_error is not None:
        status = "aborted"
        exit_code = 1
    elif any(record["state"] in {"failed", "launch_failed"} for record in records):
        status = "complete_with_failures"
        exit_code = 1
    else:
        status = "complete"
        exit_code = 0
    summary: dict[str, object] = {
        "completed_job_count": completed_count,
        "gpus": list(gpu_ids),
        "jobs": records,
        "launched_job_count": launched_count,
        "planned_job_count": len(jobs),
        "queue_directory": str(queue_directory),
        "schema_version": 1,
        "signal": state.signum,
        "stage": stage,
        "status": status,
        "worker_module": _WORKER_MODULE,
    }
    _write_summary_exclusive(summary_path, summary)
    if fatal_error is not None:
        raise fatal_error
    return QueueRunResult(exit_code=exit_code, summary_path=summary_path)


def main() -> int:
    arguments = _build_parser().parse_args()
    result = run_gpu_queue(
        arguments.stage,
        arguments.output_root,
        arguments.gpus,
    )
    print(
        canonical_json_bytes(
            {
                "exit_code": result.exit_code,
                "summary_path": str(result.summary_path),
            }
        ).decode("ascii"),
        flush=True,
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["QueueRunResult", "main", "run_gpu_queue"]
