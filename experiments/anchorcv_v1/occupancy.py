"""Exact, fail-closed lifecycle management for the external GPU occupier.

This module is deliberately operational.  Its log lives outside the AnchorCV
result tree and none of its state is admissible as research evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence


OCCUPANCY_CWD = Path(os.environ.get("CALIBTM_OCCUPANCY_DIR", Path.cwd()))
OCCUPANCY_ARGV = tuple(
    os.environ.get("CALIBTM_OCCUPANCY_ARGV", "python -u neural.py").split()
)
DEFAULT_OPERATION_LOG = OCCUPANCY_CWD / "anchorcv_occupancy_operational.log"
DEFAULT_LOCK_PATH = OCCUPANCY_CWD / ".anchorcv_occupancy.lock"
DEFAULT_WATCHDOG_LOCK_PATH = (
    OCCUPANCY_CWD / ".anchorcv_occupancy_watchdog.lock"
)
DEFAULT_IDLE_SECONDS = 9_900.0
DEFAULT_POLL_SECONDS = 60.0
DEFAULT_MAX_SAMPLE_GAP_SECONDS = 300.0
NVIDIA_SMI_TIMEOUT_SECONDS = 15.0
NVIDIA_SMI_ARGV = (
    "/usr/bin/nvidia-smi",
    "--query-gpu=utilization.gpu",
    "--format=csv,noheader,nounits",
)
OCCUPANCY_STOP_SIGNAL = signal.SIGINT


class OccupancyError(RuntimeError):
    """Base class for fail-closed occupancy errors."""


class OccupancyAmbiguityError(OccupancyError):
    """More than one independent process has the exact occupancy identity."""


class OccupancyIdentityChangedError(OccupancyError):
    """The process identity changed between discovery and signalling."""


class OccupancyStopTimeoutError(OccupancyError):
    """The exact process did not leave before the stop timeout."""


class GpuProbeError(OccupancyError):
    """GPU utilization could not be established exactly."""


class WatchdogAlreadyRunningError(OccupancyError):
    """Another long-lived watchdog owns the singleton lock."""


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    ppid: int
    start_ticks: int
    state: str
    cwd: Path
    argv: tuple[str, ...]


@dataclass(frozen=True)
class OccupancyStatus:
    running: bool
    identity: ProcessIdentity | None


@dataclass(frozen=True)
class OperationResult:
    action: str
    pid: int | None


@dataclass(frozen=True)
class WatchdogDecision:
    state: str
    pid: int | None = None
    idle_elapsed_seconds: float | None = None
    utilizations: tuple[int, ...] | None = None
    error: str | None = None


def _parse_cmdline(raw: bytes) -> tuple[str, ...] | None:
    if not raw or not raw.endswith(b"\0"):
        return None
    fields = raw[:-1].split(b"\0")
    if not fields or any(not field for field in fields):
        return None
    return tuple(field.decode("utf-8", "surrogateescape") for field in fields)


def _read_identity(proc_root: Path, pid: int) -> ProcessIdentity | None:
    """Read one stable-enough identity snapshot, returning ``None`` on a race."""

    base = proc_root / str(pid)
    try:
        raw_stat = (base / "stat").read_bytes()
        end = raw_stat.rfind(b") ")
        if end < 0:
            return None
        fields = raw_stat[end + 2 :].split()
        if len(fields) < 20:
            return None
        state = fields[0].decode("ascii")
        ppid = int(fields[1])
        start_ticks = int(fields[19])
        argv = _parse_cmdline((base / "cmdline").read_bytes())
        if argv is None:
            return None
        cwd = Path(os.readlink(base / "cwd"))
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        ProcessLookupError,
        UnicodeError,
        ValueError,
    ):
        return None
    return ProcessIdentity(
        pid=pid,
        ppid=ppid,
        start_ticks=start_ticks,
        state=state,
        cwd=cwd,
        argv=argv,
    )


def _is_exact_occupancy(identity: ProcessIdentity) -> bool:
    return (
        identity.state != "Z"
        and identity.cwd == OCCUPANCY_CWD
        and identity.argv == OCCUPANCY_ARGV
    )


def _is_same_live_process(
    expected: ProcessIdentity, observed: ProcessIdentity | None
) -> bool:
    """Compare stable proc identity fields while allowing S/R/D transitions."""

    return (
        observed is not None
        and observed.state != "Z"
        and observed.pid == expected.pid
        and observed.ppid == expected.ppid
        and observed.start_ticks == expected.start_ticks
        and observed.cwd == expected.cwd
        and observed.argv == expected.argv
    )


def find_occupancy_parent(
    *, proc_root: Path = Path("/proc")
) -> ProcessIdentity | None:
    """Find the unique top-level process with the complete fixed identity.

    Worker children may inherit the same argv and cwd.  They are excluded by
    choosing the single exact match whose PPID is not another exact match.
    Multiple independent roots are ambiguous and therefore never signalled.
    """

    matches: list[ProcessIdentity] = []
    try:
        entries = tuple(proc_root.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        raise OccupancyError(f"cannot scan process root {proc_root}") from exc
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        identity = _read_identity(proc_root, int(entry.name))
        if identity is not None and _is_exact_occupancy(identity):
            matches.append(identity)
    if not matches:
        return None
    matching_pids = {identity.pid for identity in matches}
    parents = [
        identity for identity in matches if identity.ppid not in matching_pids
    ]
    if len(parents) != 1:
        roots = sorted(identity.pid for identity in parents)
        raise OccupancyAmbiguityError(
            "expected one exact occupancy parent; "
            f"found {len(parents)} independent roots {roots}"
        )
    return parents[0]


def status(*, proc_root: Path = Path("/proc")) -> OccupancyStatus:
    identity = find_occupancy_parent(proc_root=proc_root)
    return OccupancyStatus(running=identity is not None, identity=identity)


def _append_operational_event(
    log_path: Path, event: str, **fields: object
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        **fields,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )


def stop(
    *,
    proc_root: Path = Path("/proc"),
    killer: Callable[[int, int], None] = os.kill,
    timeout_seconds: float = 20.0,
    poll_seconds: float = 0.1,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log_path: Path | None = None,
) -> OperationResult:
    """Interrupt only the rediscovered exact parent, never a fuzzy name match.

    ``neural.py`` handles ``KeyboardInterrupt`` by terminating its four worker
    children.  A raw SIGTERM would bypass that cleanup path and could orphan
    the GPU workers.
    """

    if timeout_seconds < 0 or poll_seconds <= 0:
        raise ValueError("stop timing values must be non-negative/positive")
    identity = find_occupancy_parent(proc_root=proc_root)
    if identity is None:
        result = OperationResult(action="already_stopped", pid=None)
        if log_path is not None:
            _append_operational_event(log_path, "occupancy_already_stopped")
        return result

    # Close the PID-reuse window as far as procfs permits: start ticks, PPID,
    # cwd, and complete argv all have to match the discovery snapshot.  Process
    # state is intentionally excluded because S/R/D can change every tick; Z
    # remains a terminal rejection.
    current = _read_identity(proc_root, identity.pid)
    if not _is_same_live_process(identity, current):
        raise OccupancyIdentityChangedError(
            f"occupancy PID {identity.pid} changed before SIGINT"
        )
    try:
        killer(identity.pid, OCCUPANCY_STOP_SIGNAL)
    except ProcessLookupError:
        result = OperationResult(action="stopped", pid=identity.pid)
        if log_path is not None:
            _append_operational_event(
                log_path, "occupancy_stopped", pid=identity.pid, raced=True
            )
        return result

    deadline = clock() + timeout_seconds
    while True:
        observed = _read_identity(proc_root, identity.pid)
        if not _is_same_live_process(identity, observed):
            result = OperationResult(action="stopped", pid=identity.pid)
            if log_path is not None:
                _append_operational_event(
                    log_path, "occupancy_stopped", pid=identity.pid, raced=False
                )
            return result
        if clock() >= deadline:
            raise OccupancyStopTimeoutError(
                f"occupancy PID {identity.pid} survived SIGINT for "
                f"{timeout_seconds:g} seconds"
            )
        sleep(poll_seconds)


@contextmanager
def _exclusive_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def watchdog_instance_lock(
    lock_path: Path = DEFAULT_WATCHDOG_LOCK_PATH,
) -> Iterator[None]:
    """Own the singleton watchdog lock for the complete daemon lifetime."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(
                handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise WatchdogAlreadyRunningError(
                f"watchdog lock is already held: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def restore(
    *,
    proc_root: Path = Path("/proc"),
    log_path: Path = DEFAULT_OPERATION_LOG,
    lock_path: Path = DEFAULT_LOCK_PATH,
    popen: Callable[..., object] = subprocess.Popen,
) -> OperationResult:
    """Start the exact occupier in a new session unless it already exists."""

    with _exclusive_lock(lock_path):
        existing = find_occupancy_parent(proc_root=proc_root)
        if existing is not None:
            _append_operational_event(
                log_path, "occupancy_restore_skipped", pid=existing.pid
            )
            return OperationResult(action="already_running", pid=existing.pid)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("ab", buffering=0) as output:
                process = popen(
                    list(OCCUPANCY_ARGV),
                    cwd=str(OCCUPANCY_CWD),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            pid = int(getattr(process, "pid"))
        except Exception as exc:
            _append_operational_event(
                log_path,
                "occupancy_restore_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _append_operational_event(
            log_path,
            "occupancy_restore_started",
            pid=pid,
            argv=list(OCCUPANCY_ARGV),
            cwd=str(OCCUPANCY_CWD),
        )
        return OperationResult(action="restored", pid=pid)


def gpu_utilizations(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[int, ...]:
    """Return utilization percentages for every visible NVIDIA GPU."""

    try:
        completed = run(
            list(NVIDIA_SMI_ARGV),
            capture_output=True,
            text=True,
            check=False,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuProbeError(f"nvidia-smi execution failed: {exc}") from exc
    if completed.returncode != 0:
        raise GpuProbeError(
            f"nvidia-smi returned {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines()]
    if not lines or any(not line for line in lines):
        raise GpuProbeError("nvidia-smi returned an empty utilization vector")
    try:
        values = tuple(int(line) for line in lines)
    except ValueError as exc:
        raise GpuProbeError("nvidia-smi returned a non-integer utilization") from exc
    if any(value < 0 or value > 100 for value in values):
        raise GpuProbeError("nvidia-smi returned utilization outside [0, 100]")
    return values


class IdleWatchdog:
    """State machine for evidence-free operational continuity.

    An idle interval requires repeated successful all-zero probes.  A busy
    sample, a failed probe, time moving backwards, or too large a sample gap
    resets the interval.
    """

    def __init__(
        self,
        *,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        max_sample_gap_seconds: float = DEFAULT_MAX_SAMPLE_GAP_SECONDS,
        finder: Callable[[], ProcessIdentity | None] | None = None,
        probe: Callable[[], Sequence[int]] | None = None,
        restorer: Callable[[], object] | None = None,
    ) -> None:
        if idle_seconds <= 0 or max_sample_gap_seconds <= 0:
            raise ValueError("watchdog timing values must be positive")
        self.idle_seconds = float(idle_seconds)
        self.max_sample_gap_seconds = float(max_sample_gap_seconds)
        self.finder = finder or (lambda: find_occupancy_parent())
        self.probe = probe or gpu_utilizations
        self.restorer = restorer or restore
        self._idle_started_at: float | None = None
        self._last_sample_at: float | None = None

    def _reset(self, *, last_sample_at: float | None) -> None:
        self._idle_started_at = None
        self._last_sample_at = last_sample_at

    def step(self, *, now: float | None = None) -> WatchdogDecision:
        sampled_at = time.monotonic() if now is None else float(now)
        try:
            identity = self.finder()
        except OccupancyError as exc:
            self._reset(last_sample_at=sampled_at)
            return WatchdogDecision(
                state="identity_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        if identity is not None:
            self._reset(last_sample_at=None)
            return WatchdogDecision(
                state="occupancy_running", pid=identity.pid
            )

        try:
            utilizations = tuple(int(value) for value in self.probe())
            if not utilizations or any(
                value < 0 or value > 100 for value in utilizations
            ):
                raise GpuProbeError("invalid GPU utilization vector")
        except (GpuProbeError, OSError, subprocess.SubprocessError) as exc:
            self._reset(last_sample_at=sampled_at)
            return WatchdogDecision(
                state="probe_failed",
                error=f"{type(exc).__name__}: {exc}",
            )

        gap_invalid = (
            self._last_sample_at is None
            or sampled_at < self._last_sample_at
            or sampled_at - self._last_sample_at
            > self.max_sample_gap_seconds
        )
        self._last_sample_at = sampled_at
        if any(value > 0 for value in utilizations):
            self._idle_started_at = None
            return WatchdogDecision(
                state="gpu_busy", utilizations=utilizations
            )
        if gap_invalid or self._idle_started_at is None:
            self._idle_started_at = sampled_at
        elapsed = max(0.0, sampled_at - self._idle_started_at)
        if elapsed < self.idle_seconds:
            return WatchdogDecision(
                state="idle_tracking",
                idle_elapsed_seconds=elapsed,
                utilizations=utilizations,
            )

        try:
            result = self.restorer()
        except Exception as exc:
            self._idle_started_at = sampled_at
            return WatchdogDecision(
                state="restore_failed",
                idle_elapsed_seconds=elapsed,
                utilizations=utilizations,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._reset(last_sample_at=None)
        pid = result.pid if isinstance(result, OperationResult) else None
        state = (
            "occupancy_running"
            if isinstance(result, OperationResult)
            and result.action == "already_running"
            else "restored"
        )
        return WatchdogDecision(
            state=state,
            pid=pid,
            idle_elapsed_seconds=elapsed,
            utilizations=utilizations,
        )


def _run_watchdog_loop(
    *,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_sample_gap_seconds: float = DEFAULT_MAX_SAMPLE_GAP_SECONDS,
    proc_root: Path = Path("/proc"),
    log_path: Path = DEFAULT_OPERATION_LOG,
    lock_path: Path = DEFAULT_LOCK_PATH,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    watchdog = IdleWatchdog(
        idle_seconds=idle_seconds,
        max_sample_gap_seconds=max_sample_gap_seconds,
        finder=lambda: find_occupancy_parent(proc_root=proc_root),
        probe=gpu_utilizations,
        restorer=lambda: restore(
            proc_root=proc_root,
            log_path=log_path,
            lock_path=lock_path,
        ),
    )
    _append_operational_event(
        log_path,
        "watchdog_started",
        idle_seconds=idle_seconds,
        poll_seconds=poll_seconds,
        max_sample_gap_seconds=max_sample_gap_seconds,
    )
    previous_state: str | None = None
    try:
        while True:
            decision = watchdog.step(now=clock())
            if (
                decision.state != previous_state
                or decision.state
                in {"restored", "restore_failed", "probe_failed", "identity_error"}
            ):
                _append_operational_event(
                    log_path, "watchdog_decision", **asdict(decision)
                )
            previous_state = decision.state
            sleep(poll_seconds)
    except KeyboardInterrupt:
        _append_operational_event(log_path, "watchdog_stopped")


def run_watchdog(
    *,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    max_sample_gap_seconds: float = DEFAULT_MAX_SAMPLE_GAP_SECONDS,
    proc_root: Path = Path("/proc"),
    log_path: Path = DEFAULT_OPERATION_LOG,
    lock_path: Path = DEFAULT_LOCK_PATH,
    watchdog_lock_path: Path = DEFAULT_WATCHDOG_LOCK_PATH,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    with watchdog_instance_lock(watchdog_lock_path):
        _run_watchdog_loop(
            idle_seconds=idle_seconds,
            poll_seconds=poll_seconds,
            max_sample_gap_seconds=max_sample_gap_seconds,
            proc_root=proc_root,
            log_path=log_path,
            lock_path=lock_path,
            sleep=sleep,
            clock=clock,
        )


def _identity_json(identity: ProcessIdentity | None) -> dict[str, object] | None:
    if identity is None:
        return None
    return {
        "pid": identity.pid,
        "ppid": identity.ppid,
        "start_ticks": identity.start_ticks,
        "state": identity.state,
        "cwd": str(identity.cwd),
        "argv": list(identity.argv),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-path", type=Path, default=DEFAULT_OPERATION_LOG
    )
    parser.add_argument(
        "--lock-path", type=Path, default=DEFAULT_LOCK_PATH
    )
    parser.add_argument(
        "--watchdog-lock-path",
        type=Path,
        default=DEFAULT_WATCHDOG_LOCK_PATH,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("find")
    subparsers.add_parser("status")
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--timeout-seconds", type=float, default=20.0)
    subparsers.add_parser("restore")
    watchdog_parser = subparsers.add_parser("watchdog")
    watchdog_parser.add_argument(
        "--idle-seconds", type=float, default=DEFAULT_IDLE_SECONDS
    )
    watchdog_parser.add_argument(
        "--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS
    )
    watchdog_parser.add_argument(
        "--max-sample-gap-seconds",
        type=float,
        default=DEFAULT_MAX_SAMPLE_GAP_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command in {"find", "status"}:
            current = status()
            print(
                json.dumps(
                    {
                        "running": current.running,
                        "identity": _identity_json(current.identity),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "stop":
            result = stop(
                timeout_seconds=args.timeout_seconds,
                log_path=args.log_path,
            )
            print(json.dumps(asdict(result), sort_keys=True))
            return 0
        if args.command == "restore":
            result = restore(
                log_path=args.log_path, lock_path=args.lock_path
            )
            print(json.dumps(asdict(result), sort_keys=True))
            return 0
        run_watchdog(
            idle_seconds=args.idle_seconds,
            poll_seconds=args.poll_seconds,
            max_sample_gap_seconds=args.max_sample_gap_seconds,
            log_path=args.log_path,
            lock_path=args.lock_path,
            watchdog_lock_path=args.watchdog_lock_path,
        )
        return 0
    except OccupancyError as exc:
        print(
            json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "ok": False,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
