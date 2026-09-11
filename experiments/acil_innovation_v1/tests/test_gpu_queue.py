from __future__ import annotations

import inspect
import json
from pathlib import Path
import signal
import sys

import pytest


class _ImmediateProcess:
    calls = []
    active_by_gpu = set()
    max_active = 0
    exit_codes = []

    def __init__(self, command, **kwargs):
        self.command = tuple(command)
        self.kwargs = kwargs
        self.gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        assert self.gpu not in self.active_by_gpu
        self.active_by_gpu.add(self.gpu)
        self.__class__.max_active = max(
            self.__class__.max_active, len(self.active_by_gpu)
        )
        self.returncode = self.exit_codes[len(self.calls)]
        self._poll_count = 0
        self.terminated = False
        self.killed = False
        self.__class__.calls.append(self)

    def poll(self):
        self._poll_count += 1
        if self._poll_count == 1:
            return None
        self.active_by_gpu.discard(self.gpu)
        return self.returncode

    def wait(self, timeout=None):
        self.active_by_gpu.discard(self.gpu)
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.killed = True
        self.returncode = -signal.SIGKILL


def _reset_immediate(exit_codes):
    _ImmediateProcess.calls = []
    _ImmediateProcess.active_by_gpu = set()
    _ImmediateProcess.max_active = 0
    _ImmediateProcess.exit_codes = list(exit_codes)


def test_queue_cli_and_api_expose_only_stage_output_root_and_gpus() -> None:
    from experiments.acil_innovation_v1 import gpu_queue

    assert tuple(inspect.signature(gpu_queue.run_gpu_queue).parameters) == (
        "stage",
        "output_root",
        "gpus",
    )
    destinations = {
        action.dest
        for action in gpu_queue._build_parser()._actions
        if action.dest != "help"
    }
    assert destinations == {"stage", "output_root", "gpus"}
    forbidden = {"data", "split", "method", "seed", "job_id", "retry"}
    assert forbidden.isdisjoint(destinations)


def test_queue_runs_every_planned_job_once_isolates_gpus_and_collects_failures(
    tmp_path: Path, monkeypatch
) -> None:
    from experiments.acil_innovation_v1 import gpu_queue
    from experiments.acil_innovation_v1.jobs import planned_jobs

    jobs = planned_jobs("stage_h")
    _reset_immediate([0, 7, 0, 3, 0, 0])
    monkeypatch.setattr(gpu_queue.subprocess, "Popen", _ImmediateProcess)
    monkeypatch.setattr(gpu_queue.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        gpu_queue,
        "load_stage_adjudication",
        lambda stage, output_root: {"stage": stage, "verdict": "proceed"},
    )

    result = gpu_queue.run_gpu_queue("stage_h", tmp_path, ("0", "2"))

    assert result.exit_code == 1
    assert len(_ImmediateProcess.calls) == len(jobs)
    assert _ImmediateProcess.max_active == 2
    assert [
        call.command[call.command.index("--job-id") + 1]
        for call in _ImmediateProcess.calls
    ] == [job.job_id for job in jobs]
    assert {call.gpu for call in _ImmediateProcess.calls} == {"0", "2"}
    assert all(
        call.command[:3]
        == (
            sys.executable,
            "-m",
            "experiments.acil_innovation_v1.worker",
        )
        for call in _ImmediateProcess.calls
    )
    assert all("--method" not in call.command for call in _ImmediateProcess.calls)
    assert all("--seed" not in call.command for call in _ImmediateProcess.calls)
    assert all(
        call.kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        for call in _ImmediateProcess.calls
    )
    assert len({call.kwargs["stdout"].name for call in _ImmediateProcess.calls}) == len(
        jobs
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "complete_with_failures"
    assert summary["planned_job_count"] == len(jobs)
    assert summary["launched_job_count"] == len(jobs)
    assert summary["completed_job_count"] == len(jobs)
    assert [row["job_id"] for row in summary["jobs"]] == [job.job_id for job in jobs]
    assert [row["exit_code"] for row in summary["jobs"]] == [0, 7, 0, 3, 0, 0]
    assert all(row["attempt_count"] == 1 for row in summary["jobs"])
    assert all(Path(row["log_path"]).is_file() for row in summary["jobs"])
    assert all(
        row["command"] == list(call.command)
        for row, call in zip(summary["jobs"], _ImmediateProcess.calls)
    )

    with pytest.raises(FileExistsError):
        gpu_queue.run_gpu_queue("stage_h", tmp_path, ("0", "2"))
    assert len(_ImmediateProcess.calls) == len(jobs)


def test_sigterm_terminates_active_workers_and_records_unlaunched_jobs(
    tmp_path: Path, monkeypatch
) -> None:
    from experiments.acil_innovation_v1 import gpu_queue
    from experiments.acil_innovation_v1.jobs import planned_jobs

    jobs = planned_jobs("stage_h")
    installed = {}
    previous = {
        signal.SIGINT: object(),
        signal.SIGTERM: object(),
    }

    class HangingProcess(_ImmediateProcess):
        calls = []
        active_by_gpu = set()
        max_active = 0
        exit_codes = [0] * len(jobs)

        def poll(self):
            return self.returncode if self.terminated or self.killed else None

    def fake_getsignal(signum):
        return previous[signum]

    def fake_signal(signum, handler):
        installed[signum] = handler

    fired = False

    def trigger_sigterm(_):
        nonlocal fired
        if not fired:
            fired = True
            installed[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(gpu_queue.subprocess, "Popen", HangingProcess)
    monkeypatch.setattr(gpu_queue.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(gpu_queue.signal, "signal", fake_signal)
    monkeypatch.setattr(gpu_queue.time, "sleep", trigger_sigterm)
    monkeypatch.setattr(
        gpu_queue,
        "load_stage_adjudication",
        lambda stage, output_root: {"stage": stage, "verdict": "proceed"},
    )

    result = gpu_queue.run_gpu_queue("stage_h", tmp_path, ("0", "1"))

    assert result.exit_code == 128 + signal.SIGTERM
    assert len(HangingProcess.calls) == 2
    assert all(process.terminated for process in HangingProcess.calls)
    assert not any(process.killed for process in HangingProcess.calls)
    assert installed[signal.SIGINT] is previous[signal.SIGINT]
    assert installed[signal.SIGTERM] is previous[signal.SIGTERM]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "interrupted"
    assert summary["signal"] == signal.SIGTERM
    assert summary["launched_job_count"] == 2
    assert summary["completed_job_count"] == 2
    assert [row["state"] for row in summary["jobs"][:2]] == [
        "terminated",
        "terminated",
    ]
    assert all(row["state"] == "not_launched" for row in summary["jobs"][2:])


@pytest.mark.parametrize("value", ("", "0,0", "-1", "0,", "gpu0"))
def test_gpu_list_rejects_empty_duplicate_or_noncanonical_values(value: str) -> None:
    from experiments.acil_innovation_v1.gpu_queue import _parse_gpus

    with pytest.raises((TypeError, ValueError)):
        _parse_gpus(value)


def test_queue_rejects_future_formal_stage_before_creating_output(tmp_path: Path) -> None:
    from experiments.acil_innovation_v1.gpu_queue import run_gpu_queue

    with pytest.raises(ValueError, match="active manifest"):
        run_gpu_queue("formal_gate", tmp_path, ("0",))
    assert list(tmp_path.iterdir()) == []


def test_conditional_stage_requires_exact_upstream_proceed_before_any_launch(
    tmp_path: Path, monkeypatch
) -> None:
    from experiments.acil_innovation_v1 import gpu_queue

    calls = []
    monkeypatch.setattr(
        gpu_queue,
        "load_stage_adjudication",
        lambda stage, output_root: {
            "stage": stage,
            "verdict": "kill",
        },
    )
    monkeypatch.setattr(
        gpu_queue.subprocess,
        "Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="proceed"):
        gpu_queue.run_gpu_queue("stage_h", tmp_path, ("0",))
    assert calls == []
    assert list(tmp_path.iterdir()) == []
