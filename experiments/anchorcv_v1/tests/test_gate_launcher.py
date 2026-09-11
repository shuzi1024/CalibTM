from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.anchorcv_v1 import gate_identity, gate_launcher
from experiments.anchorcv_v1.gate_identity import VerifiedGateAuthority


def _authority() -> VerifiedGateAuthority:
    bindings = []
    for bundle in (1, 2, 3):
        for dataset in ("abilene", "geant"):
            marker = f"{bundle}{dataset[0]}" * 32
            bindings.append(
                {
                    "dataset": dataset,
                    "seed_bundle": bundle,
                    "source_training_stage": (
                        "prototype" if bundle == 1 else "extension"
                    ),
                    "source_training_job_id": marker,
                    "source_result_sha256": "1" * 64,
                    "source_manifest_sha256": "2" * 64,
                    "neural_checkpoint_file_sha256": "3" * 64,
                    "neural_checkpoint_tensor_sha256": "4" * 64,
                    "acil_checkpoint_file_sha256": "5" * 64,
                    "best_epoch": 3,
                    "best_source_dev_nmae": 0.1,
                }
            )
    return VerifiedGateAuthority(
        {
            "source_tree_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "checkpoint_bindings": bindings,
        },
        "c" * 64,
        _seal=gate_identity._AUTHORITY_SEAL,
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "output_root": tmp_path / "gate",
        "authority_record": tmp_path / "authority.json",
        "extension_report": tmp_path / "extension-report.json",
        "freeze_record": tmp_path / "freeze.json",
        "prototype_output_root": tmp_path / "prototype",
        "extension_output_root": tmp_path / "extension",
    }


def test_gate_grid_is_exact_bundle_major_six_and_output_root_independent(
    tmp_path: Path,
) -> None:
    authority = _authority()
    first = gate_launcher.build_gate_specs(tmp_path / "a", authority)
    second = gate_launcher.build_gate_specs(tmp_path / "b", authority)

    expected = [
        (1, "abilene"),
        (1, "geant"),
        (2, "abilene"),
        (2, "geant"),
        (3, "abilene"),
        (3, "geant"),
    ]
    assert [(item.seed_bundle, item.dataset) for item in first] == expected
    assert [item.job_id for item in first] == [item.job_id for item in second]
    assert len({item.job_id for item in first}) == 6
    assert all(len(item.window_schedule_sha256) == 64 for item in first)
    assert (
        first[0].window_schedule_sha256
        == second[0].window_schedule_sha256
    )
    with pytest.raises(ValueError, match="exactly six"):
        gate_launcher.run_gate_grid(
            first[:1],
            gpus=(0,),
            **{
                key: value
                for key, value in _paths(tmp_path).items()
                if key != "output_root"
            },
        )


def test_gate_runner_command_passes_no_free_scientific_axis(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    spec = gate_launcher.build_gate_specs(
        paths["output_root"], _authority()
    )[0]

    command = gate_launcher.gate_job_command(spec, **{
        key: value
        for key, value in paths.items()
        if key != "output_root"
    })
    destinations = {
        token[2:].replace("-", "_")
        for token in command
        if token.startswith("--")
    }
    assert destinations == {
        "job_id",
        "output_root",
        "authority_record",
        "extension_report",
        "freeze_record",
        "prototype_output_root",
        "extension_output_root",
    }
    encoded = " ".join(command).lower()
    for forbidden in (
        "--dataset",
        "--seed-bundle",
        "--bundle",
        "--cohort",
        "--split",
        "--test",
        "--data",
        "--device",
    ):
        assert forbidden not in encoded


def test_gate_runner_command_resolves_authority_roots_before_worker_cwd_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    spec = gate_launcher.build_gate_specs(
        Path("gate-output"), _authority()
    )[0]
    command = gate_launcher.gate_job_command(
        spec,
        authority_record=Path("authority.json"),
        extension_report=Path("extension-report.json"),
        freeze_record=Path("freeze.json"),
        prototype_output_root=Path("prototype"),
        extension_output_root=Path("extension"),
    )
    for flag in (
        "--output-root",
        "--authority-record",
        "--extension-report",
        "--freeze-record",
        "--prototype-output-root",
        "--extension-output-root",
    ):
        value = Path(command[command.index(flag) + 1])
        assert value.is_absolute()


def test_launcher_parser_has_only_authority_roots_output_and_gpus() -> None:
    destinations = {
        action.dest for action in gate_launcher.build_parser()._actions
    }
    assert destinations == {
        "help",
        "output_root",
        "authority_record",
        "extension_report",
        "freeze_record",
        "prototype_output_root",
        "extension_output_root",
        "gpus",
    }


class _ImmediateProcess:
    next_pid = 5000

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.pid = _ImmediateProcess.next_pid
        _ImmediateProcess.next_pid += 1
        self.returncode = 0

    def poll(self) -> int:
        return self.returncode


def test_queue_launches_four_then_remaining_two_without_reading_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    specs = gate_launcher.build_gate_specs(paths["output_root"], _authority())
    verification_count = {spec.job_id: 0 for spec in specs}
    processes: list[_ImmediateProcess] = []

    def verify(spec: object) -> bool:
        job_id = spec.job_id
        verification_count[job_id] += 1
        return verification_count[job_id] > 1

    def popen(command: list[str], **kwargs: Any) -> _ImmediateProcess:
        process = _ImmediateProcess(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(
        gate_launcher, "verify_completed_gate_job", verify
    )
    monkeypatch.setattr(gate_launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(gate_launcher.time, "sleep", lambda seconds: None)

    summary = gate_launcher.run_gate_grid(
        specs,
        gpus=(0, 1, 2, 3),
        **{
            key: value
            for key, value in paths.items()
            if key not in {"output_root"}
        },
    )

    assert summary["status"] == "succeeded"
    assert summary["launched"] == 6
    assert len(processes) == 6
    assert [
        process.kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        for process in processes
    ] == ["0", "1", "2", "3", "0", "1"]
    assert all(count == 2 for count in verification_count.values())
    assert not any(
        "result.json" in " ".join(process.command)
        for process in processes
    )


def test_verification_failure_is_retained_and_fails_whole_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    specs = gate_launcher.build_gate_specs(
        paths["output_root"], _authority()
    )
    spec = specs[0]
    spec.job_directory.mkdir(parents=True)
    sentinel = spec.job_directory / "partial.keep"
    sentinel.write_text("retain", encoding="utf-8")
    calls = 0

    def verify(_spec: object) -> bool:
        nonlocal calls
        calls += 1
        if calls <= len(specs):
            return False
        if _spec.job_id == spec.job_id:
            raise ValueError("registered result hash mismatch")
        return True

    monkeypatch.setattr(
        gate_launcher, "verify_completed_gate_job", verify
    )
    monkeypatch.setattr(
        gate_launcher.subprocess,
        "Popen",
        lambda command, **kwargs: _ImmediateProcess(command, **kwargs),
    )
    monkeypatch.setattr(gate_launcher.time, "sleep", lambda seconds: None)

    with pytest.raises(gate_launcher.GateGridFailure, match="retained"):
        gate_launcher.run_gate_grid(
            specs,
            gpus=(0, 1, 2, 3),
            **{
                key: value
                for key, value in paths.items()
                if key != "output_root"
            },
        )

    assert sentinel.read_text(encoding="utf-8") == "retain"
    summary = json.loads(
        (paths["output_root"] / "queue_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "failed"
    assert summary["failed"][0]["verification_error"].startswith(
        "ValueError:"
    )


def test_process_launch_failure_is_recorded_and_remaining_grid_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    specs = gate_launcher.build_gate_specs(paths["output_root"], _authority())
    verification_count = {spec.job_id: 0 for spec in specs}
    popen_calls = 0

    def verify(spec: object) -> bool:
        verification_count[spec.job_id] += 1
        return verification_count[spec.job_id] > 1

    def popen(command: list[str], **kwargs: Any) -> _ImmediateProcess:
        nonlocal popen_calls
        popen_calls += 1
        if popen_calls == 1:
            raise OSError("synthetic spawn failure")
        return _ImmediateProcess(command, **kwargs)

    monkeypatch.setattr(
        gate_launcher, "verify_completed_gate_job", verify
    )
    monkeypatch.setattr(gate_launcher.subprocess, "Popen", popen)
    monkeypatch.setattr(gate_launcher.time, "sleep", lambda seconds: None)

    with pytest.raises(gate_launcher.GateGridFailure, match="retained"):
        gate_launcher.run_gate_grid(
            specs,
            gpus=(0, 1, 2, 3),
            **{
                key: value
                for key, value in paths.items()
                if key != "output_root"
            },
        )

    summary = json.loads(
        (paths["output_root"] / "queue_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["launched"] == 5
    assert len(summary["failed"]) == 1
    assert summary["failed"][0]["event"] == "launch_failed"
    assert "synthetic spawn failure" in summary["failed"][0]["message"]


@pytest.mark.parametrize("failure", [RuntimeError("failed"), KeyboardInterrupt()])
def test_pending_queue_restores_occupancy_on_failure_or_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    paths = _paths(tmp_path)
    authority = _authority()
    events: list[str] = []
    monkeypatch.setattr(
        gate_launcher,
        "verify_gate_authority",
        lambda **kwargs: authority,
    )
    monkeypatch.setattr(
        gate_launcher,
        "verify_completed_gate_job",
        lambda spec: False,
    )
    monkeypatch.setattr(
        gate_launcher,
        "_start_detached_watchdog",
        lambda: events.append("watchdog") or 999,
    )
    monkeypatch.setattr(
        gate_launcher,
        "stop_occupancy",
        lambda **kwargs: events.append("stop"),
    )
    monkeypatch.setattr(
        gate_launcher,
        "restore_occupancy",
        lambda **kwargs: events.append("restore"),
    )
    monkeypatch.setattr(
        gate_launcher,
        "run_gate_grid",
        lambda *args, **kwargs: events.append("queue") or (_ for _ in ()).throw(
            failure
        ),
    )

    with pytest.raises(type(failure)):
        gate_launcher.launch_gate(gpus=(0, 1, 2, 3), **paths)

    assert events == ["watchdog", "stop", "queue", "restore"]


def test_no_pending_jobs_do_not_stop_occupancy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    authority = _authority()
    monkeypatch.setattr(
        gate_launcher,
        "verify_gate_authority",
        lambda **kwargs: authority,
    )
    monkeypatch.setattr(
        gate_launcher,
        "verify_completed_gate_job",
        lambda spec: True,
    )
    monkeypatch.setattr(
        gate_launcher,
        "_start_detached_watchdog",
        lambda: pytest.fail("no watchdog is needed for a completed grid"),
    )
    monkeypatch.setattr(
        gate_launcher,
        "stop_occupancy",
        lambda **kwargs: pytest.fail("completed grid must not stop occupancy"),
    )
    monkeypatch.setattr(
        gate_launcher,
        "restore_occupancy",
        lambda **kwargs: pytest.fail(
            "completed grid must not perturb occupancy"
        ),
    )
    monkeypatch.setattr(
        gate_launcher,
        "run_gate_grid",
        lambda *args, **kwargs: {
            "status": "succeeded",
            "launched": 0,
        },
    )

    summary = gate_launcher.launch_gate(gpus=(0, 1, 2, 3), **paths)
    assert summary == {"status": "succeeded", "launched": 0}


def test_launcher_resolves_the_same_exact_roots_for_authority_and_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    relative = {
        "output_root": Path("gate"),
        "authority_record": Path("authority.json"),
        "extension_report": Path("extension-report.json"),
        "freeze_record": Path("freeze.json"),
        "prototype_output_root": Path("prototype"),
        "extension_output_root": Path("extension"),
    }
    authority_calls: list[dict[str, object]] = []
    queue_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        gate_launcher,
        "verify_gate_authority",
        lambda **kwargs: authority_calls.append(kwargs) or _authority(),
    )
    monkeypatch.setattr(
        gate_launcher,
        "verify_completed_gate_job",
        lambda spec: True,
    )
    monkeypatch.setattr(
        gate_launcher,
        "run_gate_grid",
        lambda specs, **kwargs: queue_calls.append(
            {"specs": specs, **kwargs}
        )
        or {"status": "succeeded"},
    )

    gate_launcher.launch_gate(gpus=(0, 1, 2, 3), **relative)

    assert len(authority_calls) == len(queue_calls) == 1
    for field in (
        "authority_record",
        "extension_report",
        "freeze_record",
        "prototype_output_root",
        "extension_output_root",
    ):
        expected = (tmp_path / relative[field]).resolve()
        assert authority_calls[0][field] == expected
        assert queue_calls[0][field] == expected
    assert all(
        spec.output_root == (tmp_path / "gate").resolve()
        for spec in queue_calls[0]["specs"]
    )


def test_stop_failure_still_runs_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        gate_launcher,
        "verify_gate_authority",
        lambda **kwargs: _authority(),
    )
    monkeypatch.setattr(
        gate_launcher,
        "verify_completed_gate_job",
        lambda spec: False,
    )
    monkeypatch.setattr(
        gate_launcher,
        "_start_detached_watchdog",
        lambda: events.append("watchdog") or 999,
    )

    def fail_stop(**kwargs: object) -> None:
        events.append("stop")
        raise RuntimeError("occupancy stop failed")

    monkeypatch.setattr(gate_launcher, "stop_occupancy", fail_stop)
    monkeypatch.setattr(
        gate_launcher,
        "restore_occupancy",
        lambda **kwargs: events.append("restore"),
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        gate_launcher.launch_gate(gpus=(0, 1, 2, 3), **paths)

    assert events == ["watchdog", "stop", "restore"]


def test_sigterm_between_stop_and_queue_is_deferred_until_after_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    authority = _authority()
    events: list[str] = []
    installed: dict[int, object] = {}
    previous = {
        gate_launcher.signal.SIGINT: object(),
        gate_launcher.signal.SIGTERM: object(),
    }
    monkeypatch.setattr(
        gate_launcher,
        "verify_gate_authority",
        lambda **kwargs: authority,
    )
    monkeypatch.setattr(
        gate_launcher,
        "verify_completed_gate_job",
        lambda spec: False,
    )
    monkeypatch.setattr(
        gate_launcher,
        "_start_detached_watchdog",
        lambda: events.append("watchdog") or 999,
    )
    monkeypatch.setattr(
        gate_launcher.signal,
        "getsignal",
        lambda caught: previous[caught],
    )

    def record_signal(caught: int, handler: object) -> None:
        installed[caught] = handler

    monkeypatch.setattr(gate_launcher.signal, "signal", record_signal)

    def signal_during_stop(**kwargs: object) -> None:
        events.append("stop")
        handler = installed[gate_launcher.signal.SIGTERM]
        assert callable(handler)
        handler(gate_launcher.signal.SIGTERM, None)

    monkeypatch.setattr(
        gate_launcher, "stop_occupancy", signal_during_stop
    )
    monkeypatch.setattr(
        gate_launcher,
        "restore_occupancy",
        lambda **kwargs: events.append("restore"),
    )
    monkeypatch.setattr(
        gate_launcher,
        "run_gate_grid",
        lambda *args, **kwargs: pytest.fail(
            "a deferred termination must not start the gate queue"
        ),
    )

    with pytest.raises(KeyboardInterrupt, match="signal 15"):
        gate_launcher.launch_gate(gpus=(0, 1, 2, 3), **paths)

    assert events == ["watchdog", "stop", "restore"]
    assert installed == previous
