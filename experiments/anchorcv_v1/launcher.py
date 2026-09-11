"""Audited multi-GPU AnchorCV queue with occupancy restoration on every exit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import IO, Mapping, Sequence

from .freeze import verify_freeze_record
from .job import JobSpec
from .job_runtime import verify_completed_job
from .occupancy import (
    DEFAULT_IDLE_SECONDS,
    DEFAULT_OPERATION_LOG,
    restore as restore_occupancy,
    stop as stop_occupancy,
)
from .protocol import fingerprint


_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = Path(os.environ.get("CALIBTM_PYTHON", sys.executable))


class GridFailure(RuntimeError):
    pass


@dataclass(slots=True)
class _Active:
    spec: JobSpec
    gpu: int
    process: subprocess.Popen
    log_handle: IO[bytes]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_stage_specs(
    stage: str,
    output_root: Path,
    freeze: Mapping[str, object],
) -> tuple[JobSpec, ...]:
    if stage not in {"prototype", "extension"}:
        raise ValueError("stage must be exactly prototype or extension")
    if freeze.get("git_available") is not False:
        raise ValueError("freeze git_available must be false")
    if freeze.get("git_commit") is not None:
        raise ValueError("freeze git_commit must be null")
    source_sha = freeze.get("source_tree_sha256")
    config_sha = freeze.get("config_sha256")
    if config_sha != fingerprint():
        raise ValueError("freeze config_sha256 differs from the active protocol")
    grid = (
        (("abilene", 1), ("geant", 1))
        if stage == "prototype"
        else (
            ("abilene", 2),
            ("geant", 2),
            ("abilene", 3),
            ("geant", 3),
        )
    )
    return tuple(
        JobSpec(
            dataset=dataset,
            seed_bundle=bundle,
            stage=stage,
            output_root=Path(output_root),
            device="cuda",
            source_tree_sha256=str(source_sha),
            config_sha256=str(config_sha),
        )
        for dataset, bundle in grid
    )


def job_command(spec: JobSpec, *, freeze_record: Path) -> list[str]:
    return [
        str(_PYTHON),
        "-m",
        "experiments.anchorcv_v1.run_job",
        "--dataset",
        spec.dataset,
        "--seed-bundle",
        str(spec.seed_bundle),
        "--stage",
        spec.stage,
        "--output-root",
        str(spec.output_root),
        "--freeze-record",
        str(Path(freeze_record).resolve()),
    ]


def choose_gpu(gpus: Sequence[int], active_gpus: set[int]) -> int | None:
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU indices must be nonempty and unique")
    candidates = [int(gpu) for gpu in gpus if int(gpu) not in active_gpus]
    return min(candidates) if candidates else None


def _append_status(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
        )


def _terminate(active: Sequence[_Active]) -> None:
    for item in active:
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and any(
        item.process.poll() is None for item in active
    ):
        time.sleep(0.1)
    for item in active:
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_grid(
    specs: Sequence[JobSpec],
    *,
    freeze_record: Path,
    gpus: Sequence[int],
) -> dict[str, object]:
    if not specs:
        raise ValueError("grid must contain at least one job")
    choose_gpu(gpus, set())
    roots = {spec.output_root for spec in specs}
    if len(roots) != 1:
        raise ValueError("all jobs must share one output root")
    output_root = next(iter(roots))
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)
    status_path = output_root / "queue_status.jsonl"
    pending = [spec for spec in specs if not verify_completed_job(spec)]
    completed_before = len(specs) - len(pending)
    active: list[_Active] = []
    failures: list[dict[str, object]] = []
    launched = 0
    interrupted = False

    def handle_signal(signum, _frame):
        nonlocal interrupted
        interrupted = True
        _append_status(
            status_path,
            {"event": "signal", "signal": signum, "time": _utc_now()},
        )

    previous = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    for sig in previous:
        signal.signal(sig, handle_signal)
    try:
        while pending or active:
            if interrupted:
                _terminate(active)
                raise KeyboardInterrupt("AnchorCV queue interrupted")
            while pending:
                gpu = choose_gpu(gpus, {item.gpu for item in active})
                if gpu is None:
                    break
                spec = pending.pop(0)
                log_path = logs / f"{spec.job_id}.log"
                handle = log_path.open("ab", buffering=0)
                environment = os.environ.copy()
                environment.update(
                    {
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTHONPATH": str(_ROOT),
                    }
                )
                process = subprocess.Popen(
                    job_command(spec, freeze_record=freeze_record),
                    cwd=_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active.append(_Active(spec, gpu, process, handle))
                launched += 1
                _append_status(
                    status_path,
                    {
                        "event": "launched",
                        "gpu": gpu,
                        "job_id": spec.job_id,
                        "pid": process.pid,
                        "time": _utc_now(),
                    },
                )
            finished = [
                item for item in active if item.process.poll() is not None
            ]
            for item in finished:
                returncode = int(item.process.returncode)
                item.log_handle.close()
                active.remove(item)
                event = {
                    "event": "finished",
                    "gpu": item.gpu,
                    "job_id": item.spec.job_id,
                    "returncode": returncode,
                    "time": _utc_now(),
                }
                _append_status(status_path, event)
                verification_error = None
                try:
                    completed = verify_completed_job(item.spec)
                except (OSError, ValueError, TypeError) as exc:
                    completed = False
                    verification_error = f"{type(exc).__name__}: {exc}"
                if verification_error is not None:
                    event["verification_error"] = verification_error
                    _append_status(
                        status_path,
                        {
                            "event": "verification_failed",
                            "job_id": item.spec.job_id,
                            "message": verification_error,
                            "time": _utc_now(),
                        },
                    )
                if returncode != 0 or not completed:
                    failures.append(event)
            if pending or active:
                time.sleep(1.0)
    finally:
        if active:
            _terminate(active)
            for item in active:
                item.log_handle.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    summary = {
        "status": "failed" if failures else "succeeded",
        "job_count": len(specs),
        "completed_before_launch": completed_before,
        "launched": launched,
        "failed": failures,
        "finished_at": _utc_now(),
    }
    summary_path = output_root / "queue_summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise GridFailure(
            f"{len(failures)} AnchorCV jobs failed; retained in {output_root}"
        )
    return summary


def _start_detached_watchdog() -> int:
    process = subprocess.Popen(
        [
            str(_PYTHON),
            "-m",
            "experiments.anchorcv_v1.occupancy",
            "--log-path",
            str(DEFAULT_OPERATION_LOG),
            "watchdog",
            "--idle-seconds",
            str(DEFAULT_IDLE_SECONDS),
        ],
        cwd=_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return int(process.pid)


def authorize_extension(
    *,
    prototype_output_root: Path,
    freeze_record: Path,
    freeze: Mapping[str, object],
) -> dict[str, object]:
    """Recompute the prototype audit from exact artifacts before extension."""

    # Delayed import avoids a module cycle: the independent reviewer reuses
    # build_stage_specs, while this authority path must never trust a report
    # supplied by the caller.
    from .review import review_prototype

    report = review_prototype(
        output_root=Path(prototype_output_root),
        freeze_record=Path(freeze_record),
    )
    if report.get("verdict") != "proceed":
        raise ValueError("extension is forbidden without audited prototype proceed")
    integrity = report.get("integrity")
    if not isinstance(integrity, Mapping) or integrity.get("valid") is not True:
        raise ValueError("extension is forbidden without valid prototype integrity")
    common = integrity.get("common_identity")
    if not isinstance(common, Mapping) or (
        common.get("source_tree_sha256") != freeze.get("source_tree_sha256")
        or common.get("config_sha256") != freeze.get("config_sha256")
    ):
        raise ValueError("prototype audit freeze identity mismatch")
    return report


def launch_stage(
    stage: str,
    *,
    output_root: Path,
    freeze_record: Path,
    gpus: Sequence[int] = (0, 1, 2, 3),
    prototype_output_root: Path | None = None,
) -> dict[str, object]:
    freeze = verify_freeze_record(freeze_record)
    if stage == "extension":
        if prototype_output_root is None:
            raise ValueError("extension requires the exact prototype output root")
        authorize_extension(
            prototype_output_root=prototype_output_root,
            freeze_record=freeze_record,
            freeze=freeze,
        )
    specs = build_stage_specs(stage, output_root, freeze)
    pending = [spec for spec in specs if not verify_completed_job(spec)]
    if not pending:
        return run_grid(specs, freeze_record=freeze_record, gpus=gpus)
    _start_detached_watchdog()
    stop_occupancy(log_path=DEFAULT_OPERATION_LOG)
    try:
        return run_grid(specs, freeze_record=freeze_record, gpus=gpus)
    finally:
        restore_occupancy(log_path=DEFAULT_OPERATION_LOG)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prototype", "extension"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--prototype-output-root", type=Path)
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value.strip())
    summary = launch_stage(
        args.stage,
        output_root=args.output_root,
        freeze_record=args.freeze_record,
        gpus=gpus,
        prototype_output_root=args.prototype_output_root,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GridFailure",
    "authorize_extension",
    "build_stage_specs",
    "choose_gpu",
    "job_command",
    "launch_stage",
    "run_grid",
]
