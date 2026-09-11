"""Audited four-GPU launcher for the fixed six-job AnchorCV final gate."""

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

from .gate_data import gate_window_schedule_sha256
from .gate_identity import (
    GateJobSpec,
    VerifiedGateAuthority,
    verify_gate_authority,
)
from .occupancy import (
    DEFAULT_IDLE_SECONDS,
    DEFAULT_OPERATION_LOG,
    restore as restore_occupancy,
    stop as stop_occupancy,
)


_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = Path(os.environ.get("CALIBTM_PYTHON", sys.executable))
_GRID = tuple(
    (bundle, dataset)
    for bundle in (1, 2, 3)
    for dataset in ("abilene", "geant")
)


class GateGridFailure(RuntimeError):
    """One or more retained gate jobs failed execution or verification."""


@dataclass(slots=True)
class _Active:
    spec: GateJobSpec
    gpu: int
    process: subprocess.Popen
    log_handle: IO[bytes]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_gate_specs(
    output_root: Path,
    authority: VerifiedGateAuthority,
) -> tuple[GateJobSpec, ...]:
    """Build the immutable bundle-major 3×2 final-gate grid."""

    return tuple(
        GateJobSpec(
            dataset=dataset,
            seed_bundle=bundle,
            output_root=Path(output_root),
            authority=authority,
            window_schedule_sha256=gate_window_schedule_sha256(dataset),
        )
        for bundle, dataset in _GRID
    )


def gate_job_command(
    spec: GateJobSpec,
    *,
    authority_record: Path,
    extension_report: Path,
    freeze_record: Path,
    prototype_output_root: Path,
    extension_output_root: Path,
) -> list[str]:
    """Build a worker command containing no caller-selectable scientific axis."""

    return [
        str(_PYTHON),
        "-m",
        "experiments.anchorcv_v1.run_gate_job",
        "--job-id",
        spec.job_id,
        "--output-root",
        str(spec.output_root.resolve()),
        "--authority-record",
        str(Path(authority_record).resolve()),
        "--extension-report",
        str(Path(extension_report).resolve()),
        "--freeze-record",
        str(Path(freeze_record).resolve()),
        "--prototype-output-root",
        str(Path(prototype_output_root).resolve()),
        "--extension-output-root",
        str(Path(extension_output_root).resolve()),
    ]


def _choose_gpu(
    gpus: Sequence[int],
    active_gpus: set[int],
) -> int | None:
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU indices must be nonempty and unique")
    if any(isinstance(gpu, bool) or not isinstance(gpu, int) for gpu in gpus):
        raise TypeError("GPU indices must be integers")
    available = [gpu for gpu in gpus if gpu not in active_gpus]
    return min(available) if available else None


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


def verify_completed_gate_job(spec: GateJobSpec) -> bool:
    # Delayed import keeps schedule/grid tests independent from gate runtime.
    from .gate_runtime import verify_completed_gate_job as verify

    return verify(spec)


def _classify_grid(
    specs: Sequence[GateJobSpec],
) -> tuple[list[GateJobSpec], list[dict[str, object]]]:
    pending: list[GateJobSpec] = []
    failures: list[dict[str, object]] = []
    for spec in specs:
        try:
            complete = verify_completed_gate_job(spec)
        except (OSError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "event": "preflight_verification_failed",
                    "job_id": spec.job_id,
                    "verification_error": f"{type(exc).__name__}: {exc}",
                    "time": _utc_now(),
                }
            )
        else:
            if not complete:
                pending.append(spec)
    return pending, failures


def run_gate_grid(
    specs: Sequence[GateJobSpec],
    *,
    authority_record: Path,
    extension_report: Path,
    freeze_record: Path,
    prototype_output_root: Path,
    extension_output_root: Path,
    gpus: Sequence[int],
) -> dict[str, object]:
    """Run all pending jobs without reading or comparing scientific metrics."""

    if len(specs) != 6:
        raise ValueError("gate grid must contain exactly six jobs")
    expected = build_gate_specs(specs[0].output_root, specs[0].authority)
    if tuple(spec.job_id for spec in specs) != tuple(
        spec.job_id for spec in expected
    ):
        raise ValueError(
            "gate grid must be the exact authority-bound bundle-major grid"
        )
    _choose_gpu(gpus, set())
    roots = {spec.output_root for spec in specs}
    if len(roots) != 1:
        raise ValueError("all gate jobs must share one output root")
    output_root = next(iter(roots))
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)
    status_path = output_root / "queue_status.jsonl"

    pending, failures = _classify_grid(specs)
    completed_before = len(specs) - len(pending) - len(failures)
    for failure in failures:
        _append_status(status_path, failure)
    active: list[_Active] = []
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
        caught: signal.getsignal(caught)
        for caught in (signal.SIGINT, signal.SIGTERM)
    }
    for caught in previous:
        signal.signal(caught, handle_signal)
    try:
        while pending or active:
            if interrupted:
                _terminate(active)
                raise KeyboardInterrupt("AnchorCV final-gate queue interrupted")
            while pending:
                gpu = _choose_gpu(gpus, {item.gpu for item in active})
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
                try:
                    process = subprocess.Popen(
                        gate_job_command(
                            spec,
                            authority_record=authority_record,
                            extension_report=extension_report,
                            freeze_record=freeze_record,
                            prototype_output_root=prototype_output_root,
                            extension_output_root=extension_output_root,
                        ),
                        cwd=_ROOT,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    handle.close()
                    event = {
                        "event": "launch_failed",
                        "gpu": gpu,
                        "job_id": spec.job_id,
                        "message": f"{type(exc).__name__}: {exc}",
                        "time": _utc_now(),
                    }
                    failures.append(event)
                    _append_status(status_path, event)
                    continue
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
                event: dict[str, object] = {
                    "event": "finished",
                    "gpu": item.gpu,
                    "job_id": item.spec.job_id,
                    "returncode": returncode,
                    "time": _utc_now(),
                }
                try:
                    complete = verify_completed_gate_job(item.spec)
                except (OSError, TypeError, ValueError) as exc:
                    complete = False
                    event["verification_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                if returncode != 0 or not complete:
                    failures.append(dict(event))
                _append_status(status_path, event)
            if pending or active:
                time.sleep(1.0)
    finally:
        if active:
            _terminate(active)
            for item in active:
                item.log_handle.close()
        for caught, handler in previous.items():
            signal.signal(caught, handler)

    summary = {
        "status": "failed" if failures else "succeeded",
        "job_count": len(specs),
        "completed_before_launch": completed_before,
        "launched": launched,
        "failed": failures,
        "finished_at": _utc_now(),
    }
    (output_root / "queue_summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise GateGridFailure(
            f"{len(failures)} final-gate jobs failed; artifacts retained in "
            f"{output_root}"
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


def launch_gate(
    *,
    output_root: Path,
    authority_record: Path,
    extension_report: Path,
    freeze_record: Path,
    prototype_output_root: Path,
    extension_output_root: Path,
    gpus: Sequence[int] = (0, 1, 2, 3),
) -> dict[str, object]:
    """Verify exact authority, then run the fixed grid with occupancy safety."""

    output_root = Path(output_root).resolve()
    authority_record = Path(authority_record).resolve()
    extension_report = Path(extension_report).resolve()
    freeze_record = Path(freeze_record).resolve()
    prototype_output_root = Path(prototype_output_root).resolve()
    extension_output_root = Path(extension_output_root).resolve()
    authority = verify_gate_authority(
        authority_record=authority_record,
        extension_report=extension_report,
        freeze_record=freeze_record,
        prototype_output_root=prototype_output_root,
        extension_output_root=extension_output_root,
    )
    specs = build_gate_specs(output_root, authority)
    pending, _preflight_failures = _classify_grid(specs)
    arguments = {
        "authority_record": authority_record,
        "extension_report": extension_report,
        "freeze_record": freeze_record,
        "prototype_output_root": prototype_output_root,
        "extension_output_root": extension_output_root,
        "gpus": gpus,
    }
    if not pending:
        return run_gate_grid(specs, **arguments)

    _start_detached_watchdog()
    interrupted_signal: int | None = None

    def defer_signal(signum, _frame):
        nonlocal interrupted_signal
        interrupted_signal = int(signum)

    previous = {
        caught: signal.getsignal(caught)
        for caught in (signal.SIGINT, signal.SIGTERM)
    }
    for caught in previous:
        signal.signal(caught, defer_signal)
    try:
        stop_occupancy(log_path=DEFAULT_OPERATION_LOG)
        if interrupted_signal is not None:
            raise KeyboardInterrupt(
                f"AnchorCV final-gate launch interrupted by signal "
                f"{interrupted_signal}"
            )
        return run_gate_grid(specs, **arguments)
    finally:
        try:
            restore_occupancy(log_path=DEFAULT_OPERATION_LOG)
            if interrupted_signal is not None:
                raise KeyboardInterrupt(
                    f"AnchorCV final-gate launch interrupted by signal "
                    f"{interrupted_signal}"
                )
        finally:
            for caught, handler in previous.items():
                signal.signal(caught, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--authority-record", required=True, type=Path)
    parser.add_argument("--extension-report", required=True, type=Path)
    parser.add_argument("--freeze-record", required=True, type=Path)
    parser.add_argument("--prototype-output-root", required=True, type=Path)
    parser.add_argument("--extension-output-root", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        gpus = tuple(
            int(value.strip())
            for value in args.gpus.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--gpus must be a comma-separated integer list") from exc
    summary = launch_gate(
        output_root=args.output_root,
        authority_record=args.authority_record,
        extension_report=args.extension_report,
        freeze_record=args.freeze_record,
        prototype_output_root=args.prototype_output_root,
        extension_output_root=args.extension_output_root,
        gpus=gpus,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GateGridFailure",
    "build_gate_specs",
    "build_parser",
    "gate_job_command",
    "launch_gate",
    "run_gate_grid",
]
