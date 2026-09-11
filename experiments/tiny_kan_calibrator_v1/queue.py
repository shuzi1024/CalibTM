"""Resumable single-GPU queue for the closed 18-job tiny-KAN grid."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.acil_innovation_v1.config import CUBLAS_WORKSPACE_CONFIG

from .jobs import Job, PROTOCOL_ID, expected_jobs
from .result_io import verified_success, write_exclusive
from .source_identity import source_record


PACKAGE = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE.parents[1]


def _command(job: Job, output: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "experiments.tiny_kan_calibrator_v1.run_job",
        "--dataset",
        job.dataset,
        "--seed-bundle",
        str(job.seed_bundle),
        "--method",
        job.method,
        "--output",
        str(output),
    ]


def _successful_attempt(job_root: Path, job: Job) -> Path | None:
    for attempt in sorted(job_root.glob("attempt-*")):
        output = attempt / "result.json"
        payload = verified_success(output)
        if payload is not None and payload.get("job") == job.to_json():
            return output
    return None


def _next_attempt(job_root: Path) -> Path:
    indices: list[int] = []
    for path in job_root.glob("attempt-*"):
        try:
            indices.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return job_root / f"attempt-{max(indices, default=0) + 1:03d}"


def _matching_live_worker(attempt: Path, job: Job) -> bool:
    """Identify a surviving worker without mistaking a reused PID for it."""

    try:
        pid = int((attempt / "pid").read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError):
        return False
    if pid < 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    command_path = Path("/proc") / str(pid) / "cmdline"
    try:
        command = command_path.read_bytes().split(b"\x00")
    except OSError:
        # Without procfs, a live PID is conservatively treated as the worker;
        # silently launching a duplicate is the more damaging failure mode.
        return True
    expected_output = str((attempt / "result.json").resolve()).encode()
    required = {
        b"experiments.tiny_kan_calibrator_v1.run_job",
        job.dataset.encode(),
        str(job.seed_bundle).encode(),
        job.method.encode(),
        expected_output,
    }
    return required.issubset(set(command))


def _live_attempts(job_root: Path, job: Job) -> tuple[Path, ...]:
    return tuple(
        attempt
        for attempt in sorted(job_root.glob("attempt-*"))
        if _matching_live_worker(attempt, job)
    )


def queue_plan(*, max_workers: int) -> dict[str, object]:
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    return {
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "1",
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            "PYTHONHASHSEED": "0",
        },
        "jobs": [asdict(job) for job in expected_jobs()],
        "protocol": PROTOCOL_ID,
        "schema": f"{PROTOCOL_ID}:launch-plan:v1",
        "source": source_record(),
        "test_access": False,
    }


def run_queue(
    output_root: str | Path,
    *,
    max_workers: int,
    poll_seconds: float = 0.25,
) -> dict[str, object]:
    plan = queue_plan(max_workers=max_workers)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "launch_plan.json"
    if not plan_path.exists():
        write_exclusive(plan_path, plan)
    else:
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise RuntimeError("existing launch plan differs from current source/grid")

    jobs_root = root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    skipped: list[dict[str, object]] = []
    pending: list[Job] = []
    for job in expected_jobs():
        job_root = jobs_root / job.job_id
        success = _successful_attempt(job_root, job)
        if success is None:
            live = _live_attempts(job_root, job)
            if live:
                names = ", ".join(path.name for path in live)
                raise RuntimeError(
                    f"job {job.job_id} still has a live detached worker in {names}; "
                    "wait for it or terminate it before resuming"
                )
            pending.append(job)
        else:
            skipped.append({"job": job.to_json(), "result": str(success)})

    environment = os.environ.copy()
    environment.update(plan["environment"])
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    running: dict[int, tuple[subprocess.Popen[bytes], Job, object, float, Path]] = {}
    completed: list[dict[str, object]] = []
    started = time.monotonic()
    while pending or running:
        while pending and len(running) < max_workers:
            job = pending.pop(0)
            job_root = jobs_root / job.job_id
            attempt = _next_attempt(job_root)
            attempt.mkdir(parents=True, exist_ok=False)
            log = (attempt / "run.log").open("xb")
            process = subprocess.Popen(
                _command(job, attempt / "result.json"),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (attempt / "pid").write_text(f"{process.pid}\n", encoding="ascii")
            running[process.pid] = (process, job, log, time.monotonic(), attempt)
        for pid, (process, job, log, job_started, attempt) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            output = attempt / "result.json"
            valid = _successful_attempt(jobs_root / job.job_id, job) == output
            completed.append(
                {
                    "elapsed_seconds": time.monotonic() - job_started,
                    "exit_code": int(code),
                    "job": job.to_json(),
                    "result": str(output),
                    "verified_success": bool(valid),
                }
            )
            del running[pid]
        if pending or running:
            time.sleep(float(poll_seconds))

    all_jobs = expected_jobs()
    successful_total = sum(
        _successful_attempt(jobs_root / job.job_id, job) is not None
        for job in all_jobs
    )
    failed = [
        row
        for row in completed
        if row["exit_code"] != 0 or not row["verified_success"]
    ]
    report = {
        "completed_this_run": completed,
        "elapsed_seconds": time.monotonic() - started,
        "failed_this_run": len(failed),
        "max_workers": max_workers,
        "planned_jobs": len(all_jobs),
        "protocol": PROTOCOL_ID,
        "resumed_successes": skipped,
        "schema": f"{PROTOCOL_ID}:queue-run:v1",
        "status": (
            "succeeded"
            if successful_total == len(all_jobs) and not failed
            else "failed"
        ),
        "successful_jobs_total": successful_total,
        "test_access": False,
    }
    queue_runs = root / "queue_runs"
    queue_runs.mkdir(parents=True, exist_ok=True)
    write_exclusive(queue_runs / f"run-{time.time_ns()}.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--launch", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.launch:
        print(json.dumps(queue_plan(max_workers=args.max_workers), indent=2, sort_keys=True))
        return 0
    report = run_queue(args.output_root, max_workers=args.max_workers)
    print(
        json.dumps(
            {
                "failed_this_run": report["failed_this_run"],
                "status": report["status"],
                "successful_jobs_total": report["successful_jobs_total"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "queue_plan", "run_queue"]
