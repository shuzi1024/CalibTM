from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path
import signal

import pytest


_MANIFEST_SHA256 = "b" * 64
_PROVENANCE_SHA256 = "c" * 64
_FREEZE_SHA256 = "d" * 64


def _target_job(stage: str = "stage0_acil_tune"):
    from experiments.acil_innovation_v1.jobs import planned_jobs

    return planned_jobs(stage)[0]


def _make_orphan(
    output_root: Path,
    *,
    stage: str = "stage0_acil_tune",
    manifest_sha256: str = _MANIFEST_SHA256,
    provenance_sha256: str = _PROVENANCE_SHA256,
):
    from experiments.acil_innovation_v1.result_io import begin_job

    job = _target_job(stage)
    writer = begin_job(
        stage=stage,
        job_id=job.job_id,
        output_root=output_root,
        manifest_sha256=manifest_sha256,
        provenance_sha256=provenance_sha256,
    )
    (writer.staging_directory / "partial-metric.bin").write_bytes(
        b"must-be-archived-without-being-read"
    )
    return job, writer.staging_directory


def _queue_payload(
    output_root: Path,
    *,
    stage: str,
    target_job_id: str,
    state: str = "failed",
    exit_code: int | None = 75,
    command_override: list[str] | None = None,
) -> dict[str, object]:
    from experiments.acil_innovation_v1.gpu_queue import _worker_command
    from experiments.acil_innovation_v1.jobs import planned_jobs

    jobs = planned_jobs(stage)
    rows = []
    for job in jobs:
        selected = job.job_id == target_job_id
        command = list(_worker_command(stage, job, output_root)) if selected else None
        if selected and command_override is not None:
            command = list(command_override)
        rows.append(
            {
                "attempt_count": 1 if selected else 0,
                "command": command,
                "dataset": job.dataset,
                "exit_code": exit_code if selected else None,
                "gpu": "0" if selected else None,
                "job_id": job.job_id,
                "launch_error": None,
                "log_path": str(
                    output_root
                    / "_gpu_queue"
                    / stage
                    / "logs"
                    / f"{job.job_id}.log"
                )
                if selected
                else None,
                "method": job.method,
                "seed_bundle": job.seed_bundle,
                "stage": job.stage,
                "state": state if selected else "not_launched",
            }
        )
    completed = {"succeeded", "failed", "terminated", "aborted"}
    signal_value = signal.SIGTERM if state == "terminated" else None
    status = {
        "failed": "complete_with_failures",
        "terminated": "interrupted",
        "aborted": "aborted",
        "succeeded": "complete",
    }.get(state, "aborted")
    return {
        "completed_job_count": sum(row["state"] in completed for row in rows),
        "gpus": ["0"],
        "jobs": rows,
        "launched_job_count": 1,
        "planned_job_count": len(jobs),
        "queue_directory": str(output_root / "_gpu_queue" / stage),
        "schema_version": 1,
        "signal": signal_value,
        "stage": stage,
        "status": status,
        "worker_module": "experiments.acil_innovation_v1.worker",
    }


def _write_queue_summary(
    output_root: Path,
    *,
    stage: str,
    job_id: str,
    state: str = "failed",
    exit_code: int | None = 75,
    command_override: list[str] | None = None,
) -> Path:
    from experiments.acil_innovation_v1.result_io import (
        write_canonical_json_exclusive,
    )

    queue_root = output_root / "_gpu_queue" / stage
    (queue_root / "logs").mkdir(parents=True)
    payload = _queue_payload(
        output_root,
        stage=stage,
        target_job_id=job_id,
        state=state,
        exit_code=exit_code,
        command_override=command_override,
    )
    return write_canonical_json_exclusive(
        queue_root / "queue_summary.json", payload
    ).path


def _patch_active_freeze(monkeypatch, recovery) -> None:
    monkeypatch.setattr(
        recovery,
        "_active_freeze_hashes",
        lambda: (_MANIFEST_SHA256, _PROVENANCE_SHA256, _FREEZE_SHA256),
    )


def test_public_recovery_api_has_only_fixed_job_identity_and_output_root() -> None:
    from experiments.acil_innovation_v1.recovery import (
        archive_infrastructure_attempt,
    )

    assert tuple(inspect.signature(archive_infrastructure_attempt).parameters) == (
        "stage",
        "job_id",
        "output_root",
    )


def test_archive_moves_orphan_atomically_without_reading_partial_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.recovery as recovery
    from experiments.acil_innovation_v1.config import protocol_config_sha256

    _patch_active_freeze(monkeypatch, recovery)
    job, staging = _make_orphan(tmp_path)
    summary_path = _write_queue_summary(
        tmp_path, stage=job.stage, job_id=job.job_id
    )
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name == "partial-metric.bin":
            raise AssertionError("recovery read a partial metric payload")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    archived = recovery.archive_infrastructure_attempt(
        job.stage, job.job_id, tmp_path
    )

    assert archived == (
        tmp_path
        / "_infrastructure_attempts"
        / job.stage
        / job.job_id
        / "attempt-0001"
    )
    assert archived.is_dir()
    assert not staging.exists()
    assert not (tmp_path / job.stage / job.job_id).exists()
    assert not (tmp_path / job.stage / f".{job.job_id}.inprogress").exists()
    assert original_read_bytes(archived / "partial-metric.bin") == (
        b"must-be-archived-without-being-read"
    )
    incident = json.loads(original_read_bytes(archived / "incident.json"))
    assert incident == {
        "attempt": "attempt-0001",
        "command": _queue_payload(
            tmp_path, stage=job.stage, target_job_id=job.job_id
        )["jobs"][0]["command"],
        "config_sha256": protocol_config_sha256(),
        "exit_code": 75,
        "failure_kind": "ex_tempfail",
        "freeze_sha256": _FREEZE_SHA256,
        "job": job.to_json(),
        "manifest_sha256": _MANIFEST_SHA256,
        "protocol": "acil-innovation-v1",
        "provenance_sha256": _PROVENANCE_SHA256,
        "queue_summary": str(summary_path.relative_to(tmp_path)),
        "queue_summary_sha256": hashlib.sha256(
            original_read_bytes(summary_path)
        ).hexdigest(),
        "schema": "acil-innovation-v1:infrastructure-attempt:v1",
        "signal": None,
        "started_sha256": hashlib.sha256(
            original_read_bytes(archived / "started.json")
        ).hexdigest(),
        "state": "failed",
    }
    with pytest.raises((FileNotFoundError, ValueError)):
        recovery.archive_infrastructure_attempt(job.stage, job.job_id, tmp_path)
    assert not (tmp_path / job.stage / f".{job.job_id}.inprogress").exists()


def test_recovery_archive_uses_the_unsupported_renameat2_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    import ctypes
    import errno

    import experiments.acil_innovation_v1.recovery as recovery
    import experiments.acil_innovation_v1.result_io as result_io

    class UnsupportedRenameAt2:
        argtypes = None
        restype = None

        def __call__(self, *_args) -> int:
            ctypes.set_errno(errno.EINVAL)
            return -1

    fake_libc = type(
        "FakeLibc", (), {"renameat2": UnsupportedRenameAt2()}
    )()
    _patch_active_freeze(monkeypatch, recovery)
    job, staging = _make_orphan(tmp_path)
    _write_queue_summary(tmp_path, stage=job.stage, job_id=job.job_id)
    monkeypatch.setattr(
        result_io.ctypes, "CDLL", lambda *_args, **_kwargs: fake_libc
    )

    archived = recovery.archive_infrastructure_attempt(
        job.stage, job.job_id, tmp_path
    )

    assert archived.is_dir()
    assert not staging.exists()
    assert (archived / "partial-metric.bin").read_bytes() == (
        b"must-be-archived-without-being-read"
    )


def test_negative_worker_returncode_derives_worker_signal_not_queue_signal(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.recovery as recovery

    _patch_active_freeze(monkeypatch, recovery)
    job, _ = _make_orphan(tmp_path)
    _write_queue_summary(
        tmp_path,
        stage=job.stage,
        job_id=job.job_id,
        state="terminated",
        exit_code=-signal.SIGKILL,
    )
    archived = recovery.archive_infrastructure_attempt(
        job.stage, job.job_id, tmp_path
    )
    incident = json.loads((archived / "incident.json").read_text(encoding="ascii"))
    assert incident["exit_code"] == -signal.SIGKILL
    assert incident["signal"] == signal.SIGKILL
    assert incident["failure_kind"] == "signal"


def test_recovery_resumes_only_an_exact_prepared_incident_after_rename_failure(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.recovery as recovery

    _patch_active_freeze(monkeypatch, recovery)
    job, staging = _make_orphan(tmp_path)
    _write_queue_summary(tmp_path, stage=job.stage, job_id=job.job_id)
    original_rename = recovery._rename_directory_noreplace

    def fail_rename(source, destination):
        raise OSError("synthetic rename interruption")

    monkeypatch.setattr(recovery, "_rename_directory_noreplace", fail_rename)
    with pytest.raises(OSError, match="rename interruption"):
        recovery.archive_infrastructure_attempt(job.stage, job.job_id, tmp_path)
    assert (staging / "incident.json").is_file()
    monkeypatch.setattr(recovery, "_rename_directory_noreplace", original_rename)

    archived = recovery.archive_infrastructure_attempt(
        job.stage, job.job_id, tmp_path
    )
    assert archived.name == "attempt-0001"
    assert not staging.exists()


@pytest.mark.parametrize(
    "case",
    (
        "live",
        "zero_exit",
        "ordinary_exit",
        "algorithmic_result",
        "final_exists",
        "command_drift",
        "freeze_hash_drift",
    ),
)
def test_recovery_rejects_non_infrastructure_or_unbound_attempts(
    tmp_path: Path, monkeypatch, case: str
) -> None:
    import experiments.acil_innovation_v1.recovery as recovery

    _patch_active_freeze(monkeypatch, recovery)
    manifest = "9" * 64 if case == "freeze_hash_drift" else _MANIFEST_SHA256
    job, staging = _make_orphan(tmp_path, manifest_sha256=manifest)
    state = "running" if case == "live" else "failed"
    exit_code = (
        0
        if case == "zero_exit"
        else 1
        if case == "ordinary_exit"
        else None
        if case == "live"
        else 75
    )
    command_override = None
    if case == "command_drift":
        command_override = ["python", "unregistered-worker.py"]
    _write_queue_summary(
        tmp_path,
        stage=job.stage,
        job_id=job.job_id,
        state=state,
        exit_code=exit_code,
        command_override=command_override,
    )
    if case == "algorithmic_result":
        from experiments.acil_innovation_v1.result_io import (
            write_canonical_json_exclusive,
        )

        write_canonical_json_exclusive(
            staging / "result.json", {"status": "algorithmic_failure"}
        )
    if case == "final_exists":
        (tmp_path / job.stage / job.job_id).mkdir()

    with pytest.raises((FileExistsError, RuntimeError, ValueError)):
        recovery.archive_infrastructure_attempt(job.stage, job.job_id, tmp_path)
    assert staging.exists()
    assert not (tmp_path / "_infrastructure_attempts").exists()


def test_recovery_rejects_future_stage_and_unknown_job_without_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.recovery as recovery

    _patch_active_freeze(monkeypatch, recovery)
    with pytest.raises(ValueError, match="active manifest"):
        recovery.archive_infrastructure_attempt("formal_gate", "0" * 64, tmp_path)
    with pytest.raises(ValueError, match="registered job"):
        recovery.archive_infrastructure_attempt(
            "stage0_acil_tune", "0" * 64, tmp_path
        )
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_recovery_rejects_symlink_inprogress_directory(
    tmp_path: Path, monkeypatch
) -> None:
    import experiments.acil_innovation_v1.recovery as recovery

    _patch_active_freeze(monkeypatch, recovery)
    job = _target_job()
    stage_root = tmp_path / job.stage
    stage_root.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (stage_root / f".{job.job_id}.inprogress").symlink_to(
        elsewhere, target_is_directory=True
    )
    _write_queue_summary(tmp_path, stage=job.stage, job_id=job.job_id)

    with pytest.raises(ValueError, match="plain|symlink|in-progress"):
        recovery.archive_infrastructure_attempt(job.stage, job.job_id, tmp_path)
    assert not (tmp_path / "_infrastructure_attempts").exists()
