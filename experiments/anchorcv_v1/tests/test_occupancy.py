from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from contextlib import ExitStack
from pathlib import Path

import pytest

from experiments.anchorcv_v1 import occupancy


def _write_process(
    proc_root: Path,
    *,
    pid: int,
    ppid: int,
    argv: tuple[str, ...] = occupancy.OCCUPANCY_ARGV,
    cwd: str = str(occupancy.OCCUPANCY_CWD),
    start_ticks: int = 1234,
    state: str = "S",
) -> None:
    base = proc_root / str(pid)
    base.mkdir(parents=True)
    (base / "cmdline").write_bytes(
        b"\0".join(item.encode("utf-8") for item in argv) + b"\0"
    )
    os.symlink(cwd, base / "cwd")
    # Fields after ") " begin at process state (field 3).  starttime is field
    # 22, hence index 19 in this suffix.
    suffix = [
        state,
        str(ppid),
        str(pid),
        str(pid),
        "0",
        "-1",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "20",
        "0",
        "1",
        "0",
        str(start_ticks),
    ]
    (base / "stat").write_text(
        f"{pid} (python worker) {' '.join(suffix)}\n", encoding="ascii"
    )


def test_find_selects_only_exact_top_level_parent(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_process(proc_root, pid=101, ppid=1)
    _write_process(proc_root, pid=102, ppid=101)
    _write_process(
        proc_root,
        pid=103,
        ppid=1,
        argv=("python3.10", "-u", "neural.py"),
    )
    _write_process(
        proc_root,
        pid=104,
        ppid=1,
        argv=occupancy.OCCUPANCY_ARGV + ("--extra",),
    )
    _write_process(proc_root, pid=105, ppid=1, cwd="/tmp/not-occupancy")
    _write_process(proc_root, pid=106, ppid=1, state="Z")

    found = occupancy.find_occupancy_parent(proc_root=proc_root)

    assert found is not None
    assert found.pid == 101
    assert found.ppid == 1
    assert found.argv == occupancy.OCCUPANCY_ARGV
    assert found.cwd == occupancy.OCCUPANCY_CWD
    assert found.start_ticks == 1234


def test_find_fails_closed_on_two_independent_exact_parents(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_process(proc_root, pid=201, ppid=1)
    _write_process(proc_root, pid=301, ppid=1)

    with pytest.raises(occupancy.OccupancyAmbiguityError):
        occupancy.find_occupancy_parent(proc_root=proc_root)


def test_status_reports_absence_without_signalling(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_process(
        proc_root,
        pid=401,
        ppid=1,
        argv=("/usr/bin/python3.10", "-u", "neural.py.backup"),
    )

    result = occupancy.status(proc_root=proc_root)

    assert result.running is False
    assert result.identity is None


def test_stop_revalidates_identity_then_terms_only_exact_parent(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_process(proc_root, pid=501, ppid=1)
    _write_process(proc_root, pid=502, ppid=501)
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        shutil.rmtree(proc_root / str(pid))

    result = occupancy.stop(
        proc_root=proc_root,
        killer=fake_kill,
        timeout_seconds=0.1,
        poll_seconds=0.001,
        sleep=lambda _: None,
    )

    assert result.action == "stopped"
    assert result.pid == 501
    assert signals == [(501, signal.SIGINT)]
    # The child is never directly signalled by this module.
    assert (proc_root / "502").exists()


def test_stop_fails_closed_if_pid_identity_changes_before_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    original = occupancy.ProcessIdentity(
        pid=601,
        ppid=1,
        start_ticks=111,
        state="S",
        cwd=occupancy.OCCUPANCY_CWD,
        argv=occupancy.OCCUPANCY_ARGV,
    )
    reused = occupancy.ProcessIdentity(
        pid=601,
        ppid=1,
        start_ticks=222,
        state="S",
        cwd=occupancy.OCCUPANCY_CWD,
        argv=occupancy.OCCUPANCY_ARGV,
    )
    monkeypatch.setattr(occupancy, "find_occupancy_parent", lambda **_: original)
    monkeypatch.setattr(occupancy, "_read_identity", lambda *_: reused)
    signals: list[tuple[int, int]] = []

    with pytest.raises(occupancy.OccupancyIdentityChangedError):
        occupancy.stop(
            proc_root=proc_root,
            killer=lambda pid, sig: signals.append((pid, sig)),
        )

    assert signals == []


def test_stop_accepts_normal_sleeping_to_running_state_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    discovered = occupancy.ProcessIdentity(
        pid=651,
        ppid=1,
        start_ticks=111,
        state="S",
        cwd=occupancy.OCCUPANCY_CWD,
        argv=occupancy.OCCUPANCY_ARGV,
    )
    running = occupancy.ProcessIdentity(
        pid=651,
        ppid=1,
        start_ticks=111,
        state="R",
        cwd=occupancy.OCCUPANCY_CWD,
        argv=occupancy.OCCUPANCY_ARGV,
    )
    monkeypatch.setattr(occupancy, "find_occupancy_parent", lambda **_: discovered)
    monkeypatch.setattr(occupancy, "_read_identity", lambda *_: running)
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        signals.append((pid, sig))
        raise ProcessLookupError

    result = occupancy.stop(proc_root=proc_root, killer=fake_kill)

    assert result.action == "stopped"
    assert signals == [(651, occupancy.OCCUPANCY_STOP_SIGNAL)]


def test_stop_rejects_zombie_during_pre_signal_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    discovered = occupancy.ProcessIdentity(
        pid=671,
        ppid=1,
        start_ticks=111,
        state="S",
        cwd=occupancy.OCCUPANCY_CWD,
        argv=occupancy.OCCUPANCY_ARGV,
    )
    zombie = occupancy.ProcessIdentity(
        pid=671,
        ppid=1,
        start_ticks=111,
        state="Z",
        cwd=occupancy.OCCUPANCY_CWD,
        argv=occupancy.OCCUPANCY_ARGV,
    )
    monkeypatch.setattr(occupancy, "find_occupancy_parent", lambda **_: discovered)
    monkeypatch.setattr(occupancy, "_read_identity", lambda *_: zombie)
    signals: list[tuple[int, int]] = []

    with pytest.raises(occupancy.OccupancyIdentityChangedError):
        occupancy.stop(
            proc_root=proc_root,
            killer=lambda pid, sig: signals.append((pid, sig)),
        )

    assert signals == []


def test_restore_uses_fixed_argv_cwd_session_and_append_log(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    log_path = tmp_path / "ops" / "occupancy.log"
    lock_path = tmp_path / "ops" / "occupancy.lock"
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeProcess:
        pid = 701

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        calls.append((args, kwargs))
        return FakeProcess()

    result = occupancy.restore(
        proc_root=proc_root,
        log_path=log_path,
        lock_path=lock_path,
        popen=fake_popen,
    )

    assert result.action == "restored"
    assert result.pid == 701
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (list(occupancy.OCCUPANCY_ARGV),)
    assert kwargs["cwd"] == str(occupancy.OCCUPANCY_CWD)
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert getattr(kwargs["stdout"], "mode") == "ab"
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("{")
    ]
    assert records[-1]["event"] == "occupancy_restore_started"
    assert records[-1]["pid"] == 701


def test_restore_does_not_duplicate_existing_parent(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_process(proc_root, pid=801, ppid=1)
    calls: list[object] = []

    result = occupancy.restore(
        proc_root=proc_root,
        log_path=tmp_path / "occupancy.log",
        lock_path=tmp_path / "occupancy.lock",
        popen=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert result.action == "already_running"
    assert result.pid == 801
    assert calls == []


def test_gpu_utilizations_uses_fixed_query_and_rejects_bad_output() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, stdout="0\n17\n0\n0\n", stderr="")

    values = occupancy.gpu_utilizations(run=fake_run)

    assert values == (0, 17, 0, 0)
    assert calls[0][0] == (list(occupancy.NVIDIA_SMI_ARGV),)
    assert calls[0][1]["timeout"] == occupancy.NVIDIA_SMI_TIMEOUT_SECONDS

    with pytest.raises(occupancy.GpuProbeError):
        occupancy.gpu_utilizations(
            run=lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 0, stdout="0\nnot-a-number\n", stderr=""
            )
        )


def test_watchdog_restores_once_after_9900_seconds_of_observed_zero() -> None:
    restored: list[str] = []
    watchdog = occupancy.IdleWatchdog(
        idle_seconds=9900,
        max_sample_gap_seconds=20_000,
        finder=lambda: None,
        probe=lambda: (0, 0, 0, 0),
        restorer=lambda: restored.append("restore")
        or occupancy.OperationResult(action="restored", pid=901),
    )

    assert watchdog.step(now=0).state == "idle_tracking"
    assert watchdog.step(now=9899).state == "idle_tracking"
    decision = watchdog.step(now=9900)

    assert decision.state == "restored"
    assert decision.idle_elapsed_seconds == 9900
    assert restored == ["restore"]


def test_watchdog_busy_probe_failure_and_long_gap_reset_idle_clock() -> None:
    readings: list[object] = [(0, 0), (0, 9), occupancy.GpuProbeError("failed"), (0, 0)]
    restored: list[str] = []

    def probe() -> tuple[int, ...]:
        value = readings.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    watchdog = occupancy.IdleWatchdog(
        idle_seconds=100,
        max_sample_gap_seconds=20,
        finder=lambda: None,
        probe=probe,
        restorer=lambda: restored.append("restore"),
    )

    assert watchdog.step(now=0).state == "idle_tracking"
    assert watchdog.step(now=10).state == "gpu_busy"
    assert watchdog.step(now=15).state == "probe_failed"
    # Gap from the last successful/probe attempt is too large, so this sample
    # starts a fresh continuous-idle interval.
    decision = watchdog.step(now=100)

    assert decision.state == "idle_tracking"
    assert decision.idle_elapsed_seconds == 0
    assert restored == []


def test_watchdog_existing_occupancy_skips_gpu_probe() -> None:
    identity = occupancy.ProcessIdentity(
        pid=1001,
        ppid=1,
        start_ticks=3,
        state="S",
        cwd=occupancy.OCCUPANCY_CWD,
        argv=occupancy.OCCUPANCY_ARGV,
    )
    probed: list[str] = []
    watchdog = occupancy.IdleWatchdog(
        finder=lambda: identity,
        probe=lambda: probed.append("probe"),
        restorer=lambda: pytest.fail("must not restore"),
    )

    decision = watchdog.step(now=0)

    assert decision.state == "occupancy_running"
    assert decision.pid == 1001
    assert probed == []


def test_watchdog_instance_lock_is_nonblocking_and_singleton(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "ops" / "watchdog.lock"

    with ExitStack() as stack:
        stack.enter_context(occupancy.watchdog_instance_lock(lock_path))
        with pytest.raises(occupancy.WatchdogAlreadyRunningError):
            with occupancy.watchdog_instance_lock(lock_path):
                pytest.fail("second watchdog lock must not be acquired")

    # Releasing the first instance makes the lock reusable.
    with occupancy.watchdog_instance_lock(lock_path):
        pass
